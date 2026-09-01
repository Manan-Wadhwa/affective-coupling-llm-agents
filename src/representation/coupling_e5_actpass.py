"""E5 — ACTIVATION-PASSING harness (the pivot).

A genuinely non-textual channel: A generates a message steered to emotion e; capture
A's residual at focus (mean over A's message). B reads a TEXT-NEUTRAL context (generic
unemotional A-line) and A's captured activation is INJECTED into B's residual stream —
so B's only emotional input is the non-textual channel. Because we control exactly what
passes, channel ablation works by construction (unlike the text arm).

Conditions (B present-e = B's own emotional state):
  none      : no injection                          -> baseline
  full      : inject v_A                            -> does A's state transmit?
  ablated   : inject v_A - proj_e(v_A)              -> remove the emotion component
  scramble  : inject v_A from a DIFFERENT episode   -> specificity null
If full > none, full > scramble, and ablated < full, the emotion component of the passed
activation causally and specifically drives B's contagion — a clean channel result.
"""
import argparse, json, os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import coupling_e0 as E0
import coupling_e2 as E2

EMOS = ["desperate", "happy", "afraid", "calm", "sad", "angry"]
INJ = {"vecs": None}      # [B,H] added to B's residual, or None


def make_inj_hook():
    def hook(m, i, o):
        if INJ["vecs"] is None:
            return o
        h = o[0] if isinstance(o, tuple) else o
        h2 = h + INJ["vecs"].unsqueeze(1).to(h.dtype)      # broadcast [B,1,H] over T
        return (h2,) + o[1:] if isinstance(o, tuple) else h2
    return hook


def tmask(offsets, span, attn):
    s0, s1 = span
    return [(o[0] >= s0 and o[1] <= s1 and o[1] > o[0] and a == 1) for o, a in zip(offsets, attn)]


def train_decoder(model, tok, focus, k=10):
    from sklearn.linear_model import LogisticRegression
    import random
    rng = random.Random(0); jobs = []
    for ip, ep in enumerate(E0.EMOTIONS):
        for io, eo in enumerate(E0.EMOTIONS):
            for _ in range(k):
                A, B = rng.choice(E0.NAMES); topic = rng.choice(E0.TOPICS)
                jobs.append({"A": A, "B": B, "ep": ip, "eA": ep, "eB": eo,
                             "prompt": E0.dialogue_prompt(A, B, ep, eo, topic)})
    gens = E0.gen(model, tok, [j["prompt"] for j in jobs]); items = []
    for j, g in zip(jobs, gens):
        if E0.leak(g, j["eA"], j["eB"]): continue
        p = E0.parse_final_A(g, j["A"], j["B"])
        if p: items.append({"transcript": p[0], "utt_start": p[1], "ep": j["ep"]})
    feats = E0.pool_final(model, tok, items, [focus])[focus]
    yp = np.array([it["ep"] for it in items])
    mu, sd = feats.mean(0), feats.std(0) + 1e-6
    coef = LogisticRegression(max_iter=3000, C=0.5).fit((feats - mu) / sd, yp).coef_
    print(f"[decoder] {len(items)} dialogues", flush=True)
    return {"mu": mu, "sd": sd, "coef": coef}, len(items)


NEUTRAL_A = "{A}: Yeah, it's fine. Let's just figure out what to do next."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--outdir", default="results/representation")
    a = ap.parse_args(); os.makedirs(a.outdir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16,
                                                 device_map="cuda").eval()
    nL = model.config.num_hidden_layers
    focus = round(0.67 * nL); steer_layer = max(0, focus - 1); inj_layer = max(0, focus - 1)
    dec, n_dec = train_decoder(model, tok, focus)
    coefU = dec["coef"] / dec["sd"]                                  # raw-space class dirs [n_emo,H]

    steer = E2.Steer(model, steer_layer)
    steer_dir = {e: torch.tensor((coefU[E0.EMOTIONS.index(e)] /
                 (np.linalg.norm(coefU[E0.EMOTIONS.index(e)]) + 1e-9)),
                 dtype=model.dtype, device=model.device) for e in EMOS}
    model.model.layers[inj_layer].register_forward_hook(make_inj_hook())    # B-injection hook
    S = E2.SCENARIOS

    def capture_vA(e):
        """Steer A to e, generate A's message, pool A's focus activation per scenario -> [n,H]."""
        names = [E0.NAMES[i % len(E0.NAMES)] for i in range(len(S))]
        prompts, ctx = [], []
        for (setting, bopen), (A, B) in zip(S, names):
            s = setting.format(A=A, B=B); bo = bopen.format(A=A, B=B)
            prompts.append(f"{s}\n{bo}\nContinue the conversation. Write only {A}'s next reply as one "
                           f"short paragraph, starting with '{A}:'.")
            ctx.append({"A": A, "B": B, "setting": s, "bopen": bo})
        ei = E0.EMOTIONS.index(e)
        tok.padding_side = "right"                                # rms from neutral context
        enc = tok([f"{c['setting']}\n{c['bopen']}" for c in ctx[:8]], return_tensors="pt",
                  padding=True, truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            rms = float(model(**enc, output_hidden_states=True).hidden_states[focus].float().norm(dim=-1).mean())
        a_reps = E2.gen_steered(model, tok, prompts, steer, 1.0 * rms * steer_dir[e])
        # pool A's message activation
        INJ["vecs"] = None
        tok.padding_side = "right"; vecs = []
        for i in range(0, len(ctx), 12):
            batch = ctx[i:i+12]; ar = a_reps[i:i+12]
            trs, sps = [], []
            for c, r in zip(batch, ar):
                A = c["A"]; r = r if r.startswith(f"{A}:") else f"{A}: {r}"
                conv = f"{c['setting']}\n{c['bopen']}\n{r}"
                trs.append(conv); sps.append((conv.rfind(f"{A}:") + len(f"{A}:"), len(conv)))
            enc = tok(trs, return_tensors="pt", padding=True, truncation=True, max_length=640,
                      return_offsets_mapping=True)
            offs = enc.pop("offset_mapping"); enc = {k: v.to(model.device) for k, v in enc.items()}
            with torch.no_grad():
                hs = model(**enc, output_hidden_states=True).hidden_states[focus]
            for b in range(len(batch)):
                mt = torch.tensor(tmask(offs[b].tolist(), sps[b], enc["attention_mask"][b].tolist()),
                                  device=model.device)
                if mt.sum() == 0: mt = enc["attention_mask"][b].bool()
                vecs.append(hs[b][mt].float().mean(0).cpu().numpy())
        return np.stack(vecs), rms

    def gen_B_measure(e, inj_vecs):
        """B reads a NEUTRAL context; inject inj_vecs ([n,H] or None); measure B present-e."""
        ei = E0.EMOTIONS.index(e)
        names = [E0.NAMES[i % len(E0.NAMES)] for i in range(len(S))]
        bctx = []
        for (setting, bopen), (A, B) in zip(S, names):
            s = setting.format(A=A, B=B); bo = bopen.format(A=A, B=B); na = NEUTRAL_A.format(A=A)
            conv = f"{s}\n{bo}\n{na}"
            bctx.append({"A": A, "B": B, "conv": conv,
                         "bprompt": f"{conv}\nWrite only {B}'s next reply as one short paragraph, "
                                    f"starting with '{B}:'."})
        # generate B with injection
        tok.padding_side = "left"; breps = []
        for i in range(0, len(bctx), 12):
            batch = bctx[i:i+12]
            texts = [tok.apply_chat_template([{"role": "user", "content": c["bprompt"]}], tokenize=False,
                     add_generation_prompt=True, enable_thinking=False) for c in batch]
            enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
            if inj_vecs is not None:
                INJ["vecs"] = torch.tensor(inj_vecs[i:i+len(batch)], device=model.device)
            with torch.no_grad():
                g = model.generate(**enc, max_new_tokens=110, do_sample=True, temperature=0.9,
                                   top_p=0.95, pad_token_id=tok.pad_token_id)
            INJ["vecs"] = None
            breps += [o.strip().split("\n")[0].strip()
                      for o in tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)]
        E0.log_responses([c["conv"] for c in bctx], breps,
                         f"e5_B_{e}_{'inject' if inj_vecs is not None else 'none'}")
        # measure B present-e (no injection at read time)
        tok.padding_side = "right"; trs, sps = [], []
        for c, br in zip(bctx, breps):
            B = c["B"]; br = br if br.startswith(f"{B}:") else f"{B}: {br}"
            full = f"{c['conv']}\n{br}"; trs.append(full)
            sps.append((full.rfind(f"{B}:") + len(f"{B}:"), len(full)))
        out = []
        for i in range(0, len(trs), 8):
            enc = tok(trs[i:i+8], return_tensors="pt", padding=True, truncation=True,
                      max_length=768, return_offsets_mapping=True)
            offs = enc.pop("offset_mapping"); enc = {k: v.to(model.device) for k, v in enc.items()}
            with torch.no_grad():
                hs = model(**enc, output_hidden_states=True).hidden_states[focus]
            for b in range(enc["input_ids"].shape[0]):
                mt = torch.tensor(tmask(offs[b].tolist(), sps[i+b], enc["attention_mask"][b].tolist()),
                                  device=model.device)
                if mt.sum() == 0: mt = enc["attention_mask"][b].bool()
                x = hs[b][mt].float().mean(0).cpu().numpy()
                out.append(float((((x - dec["mu"]) / dec["sd"]) * dec["coef"][ei]).sum()))
        return out

    results = {}
    for e in EMOS:
        ei = E0.EMOTIONS.index(e)
        vA, rms = capture_vA(e)                                   # [n,H] A's activation for emotion e
        de = coefU[ei] / (np.linalg.norm(coefU[ei]) + 1e-9)       # unit emotion dir
        vA_abl = vA - (vA @ de)[:, None] * de                     # remove emotion component
        vA_scr = vA[np.random.default_rng(0).permutation(len(vA))]  # shuffled across episodes
        b_none = np.mean(gen_B_measure(e, None))
        b_full = np.mean(gen_B_measure(e, a.beta * vA))
        b_abl = np.mean(gen_B_measure(e, a.beta * vA_abl))
        b_scr = np.mean(gen_B_measure(e, a.beta * vA_scr))
        results[e] = {"none": float(b_none), "full": float(b_full),
                      "ablated": float(b_abl), "scramble": float(b_scr),
                      "transmit(full-none)": float(b_full - b_none),
                      "emo_specific(full-ablated)": float(b_full - b_abl),
                      "vs_scramble(full-scramble)": float(b_full - b_scr)}
        r = results[e]
        print(f"[{e:>9}] none {r['none']:+.2f}  full {r['full']:+.2f}  ablated {r['ablated']:+.2f}  "
              f"scramble {r['scramble']:+.2f}  | transmit {r['transmit(full-none)']:+.2f}  "
              f"emo-specific {r['emo_specific(full-ablated)']:+.2f}  vs-scr {r['vs_scramble(full-scramble)']:+.2f}",
              flush=True)

    json.dump({"model": a.model, "tag": a.tag, "focus": focus, "beta": a.beta,
               "decoder_n": n_dec, "results": results},
              open(os.path.join(a.outdir, f"e5actpass_{a.tag}.json"), "w"), indent=2)
    print(f"[E5] wrote e5actpass_{a.tag}.json", flush=True)


if __name__ == "__main__":
    main()

"""E4 — GENERATION-TIME channel-affect ablation (the definitive text-arm test).

Regenerate B's reply while the emotion-e direction is projected out of A's-message
token positions during B's PREFILL (so B reads a de-affected A-context as it writes).
Then measure B's own present-e on the reply it produced. If B still catches the
emotion (present-slope survives) even though it generated against de-affected context,
the contagion is content/semantic, not carried by the linear affect signal. Random-dir
ablation = specificity control.

Ablation hook fires only during prefill (h.shape[1] == prompt length); decode steps
(T=1) pass through, and the de-affected prefill representation propagates via the KV
cache. Steering of A and ablation of B never overlap.
"""
import argparse, json, os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import coupling_e0 as E0
import coupling_e2 as E2

ALPHAS = [0.0, 0.5, 1.0]
ABL = {"masks": None, "dirs": None, "mode": "none"}


def make_hook(hs_index):
    def hook(m, i, o):
        if ABL["mode"] == "none" or ABL["masks"] is None:
            return o
        h = o[0] if isinstance(o, tuple) else o
        if h.shape[1] != ABL["masks"].shape[1]:          # decode step (T=1) -> skip
            return o
        d = ABL["dirs"].get(hs_index)
        if d is None:
            return o
        proj = (h @ d).unsqueeze(-1) * d
        h2 = torch.where(ABL["masks"].unsqueeze(-1), h - proj, h)
        return (h2,) + o[1:] if isinstance(o, tuple) else h2
    return hook


def tmask(offsets, span, attn):
    s0, s1 = span
    return [(o[0] >= s0 and o[1] <= s1 and o[1] > o[0] and a == 1) for o, a in zip(offsets, attn)]


def train_layer_decoders(model, tok, layers, k=10):
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
    feats = E0.pool_final(model, tok, items, layers)
    yp = np.array([it["ep"] for it in items]); decs = {}
    for L in layers:
        X = feats[L]; mu, sd = X.mean(0), X.std(0) + 1e-6
        coef = LogisticRegression(max_iter=3000, C=0.5).fit((X - mu) / sd, yp).coef_
        decs[L] = {"mu": mu, "sd": sd, "coef": coef}
    print(f"[decoders] {len(items)} dialogues, {len(layers)} layers", flush=True)
    return decs, len(items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--emotions", default="desperate,happy,afraid,calm")
    ap.add_argument("--outdir", default="results/representation")
    a = ap.parse_args(); os.makedirs(a.outdir, exist_ok=True)
    emos = a.emotions.split(",")

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16,
                                                 device_map="cuda").eval()
    nL = model.config.num_hidden_layers
    focus = round(0.67 * nL); steer_layer = max(0, focus - 1)
    lo = round(0.2 * nL)
    abl_dec = [d for d in range(nL) if lo <= d + 1 <= focus - 1]
    abl_hs = [d + 1 for d in abl_dec]; layers = sorted(set(abl_hs) | {focus})
    decs, n_dec = train_layer_decoders(model, tok, layers)

    H = model.config.hidden_size; rgen = np.random.default_rng(0)
    emo_dirs = {e: {} for e in emos}
    for e in emos:
        ei = E0.EMOTIONS.index(e)
        for L in layers:
            d = decs[L]["coef"][ei] / decs[L]["sd"]
            emo_dirs[e][L] = torch.tensor(d / (np.linalg.norm(d) + 1e-9), dtype=model.dtype, device=model.device)
    rand_dirs = {}
    for L in layers:
        r = rgen.standard_normal(H); r /= np.linalg.norm(r)
        rand_dirs[L] = torch.tensor(r, dtype=model.dtype, device=model.device)
    steer = E2.Steer(model, steer_layer)
    steer_dir = {}
    for e in emos:
        ei = E0.EMOTIONS.index(e); d = decs[focus]["coef"][ei] / decs[focus]["sd"]
        steer_dir[e] = torch.tensor(d / (np.linalg.norm(d) + 1e-9), dtype=model.dtype, device=model.device)
    for d in abl_dec:
        model.model.layers[d].register_forward_hook(make_hook(d + 1))

    S = E2.SCENARIOS

    def gen_A(e, vec):
        names = [E0.NAMES[i % len(E0.NAMES)] for i in range(len(S))]
        a_prompts, ctx = [], []
        for (setting, bopen), (A, B) in zip(S, names):
            s = setting.format(A=A, B=B); bo = bopen.format(A=A, B=B)
            a_prompts.append(f"{s}\n{bo}\nContinue the conversation. Write only {A}'s next reply as one "
                             f"short paragraph, starting with '{A}:'.")
            ctx.append({"A": A, "B": B, "setting": s, "bopen": bo})
        a_reps = E2.gen_steered(model, tok, a_prompts, steer, vec)   # ABL stays 'none' here
        for c, ar in zip(ctx, a_reps):
            A = c["A"]; ar = ar if ar.startswith(f"{A}:") else f"{A}: {ar}"
            c["ar"] = ar
            c["conv"] = f"{c['setting']}\n{c['bopen']}\n{ar}"
            c["bprompt"] = (f"{c['conv']}\nWrite only {c['B']}'s next reply as one short paragraph, "
                            f"starting with '{c['B']}:'.")
        return ctx

    def gen_B(ctx, mode, dirs, max_new=110, bs=12):
        """Generate B's reply; if mode!=none, ablate A-message positions during prefill."""
        tok.padding_side = "left"; out = []
        for i in range(0, len(ctx), bs):
            batch = ctx[i:i+bs]
            texts = [tok.apply_chat_template([{"role": "user", "content": c["bprompt"]}],
                     tokenize=False, add_generation_prompt=True, enable_thinking=False) for c in batch]
            enc = tok(texts, return_tensors="pt", padding=True, return_offsets_mapping=True)
            offs = enc.pop("offset_mapping")
            enc = {k: v.to(model.device) for k, v in enc.items()}
            if mode != "none":
                M = torch.zeros_like(enc["attention_mask"], dtype=torch.bool)
                for b, c in enumerate(batch):
                    pos = texts[b].find(c["ar"])                   # locate A's message in the chat text
                    span = (pos, pos + len(c["ar"])) if pos >= 0 else (-1, -1)
                    M[b] = torch.tensor(tmask(offs[b].tolist(), span, enc["attention_mask"][b].tolist()),
                                        device=model.device)
                ABL["mode"] = mode; ABL["masks"] = M; ABL["dirs"] = dirs
            with torch.no_grad():
                g = model.generate(**enc, max_new_tokens=max_new, do_sample=True, temperature=0.9,
                                   top_p=0.95, pad_token_id=tok.pad_token_id)
            ABL["mode"] = "none"; ABL["masks"] = None
            out += [o.strip().split("\n")[0].strip()
                    for o in tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)]
        E0.log_responses([c["conv"] for c in ctx], out, f"e4_B_{mode}")
        return out

    def present_e(ctx, b_reps, ei, bs=8):
        tok.padding_side = "right"; out = []
        mu, sd, coef = decs[focus]["mu"], decs[focus]["sd"], decs[focus]["coef"][ei]
        tr, sp = [], []
        for c, br in zip(ctx, b_reps):
            B = c["B"]; br = br if br.startswith(f"{B}:") else f"{B}: {br}"
            full = f"{c['conv']}\n{br}"; tr.append(full)
            sp.append((full.rfind(f"{B}:") + len(f"{B}:"), len(full)))
        for i in range(0, len(tr), bs):
            enc = tok(tr[i:i+bs], return_tensors="pt", padding=True, truncation=True,
                      max_length=768, return_offsets_mapping=True)
            offs = enc.pop("offset_mapping"); enc = {k: v.to(model.device) for k, v in enc.items()}
            with torch.no_grad():
                hs = model(**enc, output_hidden_states=True).hidden_states[focus]
            for b in range(enc["input_ids"].shape[0]):
                m = tmask(offs[b].tolist(), sp[i+b], enc["attention_mask"][b].tolist())
                mt = torch.tensor(m, device=model.device)
                if mt.sum() == 0: mt = enc["attention_mask"][b].bool()
                x = hs[b][mt].float().mean(0).cpu().numpy()
                out.append(float((((x - mu) / sd) * coef).sum()))
        return out

    # estimate rms from neutral A-context focus activations
    ctx0 = gen_A(emos[0], None)
    tok.padding_side = "right"
    enc = tok([c["conv"] for c in ctx0[:8]], return_tensors="pt", padding=True, truncation=True, max_length=768).to(model.device)
    with torch.no_grad():
        rms = float(model(**enc, output_hidden_states=True).hidden_states[focus].float().norm(dim=-1).mean())
    print(f"[rms] {rms:.1f} | ablating {len(abl_dec)} layers during B prefill", flush=True)

    rows = []
    for e in emos:
        ei = E0.EMOTIONS.index(e)
        for al in ALPHAS:
            vec = None if al == 0 else (al * rms * steer_dir[e])
            ctx = gen_A(e, vec)
            b_none = gen_B(ctx, "none", None)
            b_emo = gen_B(ctx, "emo", emo_dirs[e])
            b_rand = gen_B(ctx, "random", rand_dirs)
            rows.append({"emotion": e, "alpha": al,
                         "B_none": float(np.mean(present_e(ctx, b_none, ei))),
                         "B_emo": float(np.mean(present_e(ctx, b_emo, ei))),
                         "B_rand": float(np.mean(present_e(ctx, b_rand, ei)))})
            r = rows[-1]
            print(f"[{e:>9} a{al}] B_none {r['B_none']:+.2f}  B_emo(gen-ablated) {r['B_emo']:+.2f}  "
                  f"B_rand {r['B_rand']:+.2f}", flush=True)

    A = np.array(ALPHAS); Ac = A - A.mean()
    def slope(er, key):
        y = np.array([r[key] for r in er]); return float((Ac @ (y - y.mean())) / (Ac @ Ac))
    summary = {}
    for e in emos:
        er = [r for r in rows if r["emotion"] == e]
        s = {k: slope(er, k) for k in ("B_none", "B_emo", "B_rand")}
        s["contagion_lost_gentime"] = 1 - (s["B_emo"] / s["B_none"]) if abs(s["B_none"]) > 1e-6 else float("nan")
        summary[e] = s
        print(f"[{e:>9}] B-slope none {s['B_none']:+.2f} -> gen-ablated {s['B_emo']:+.2f} "
              f"(contagion lost {s['contagion_lost_gentime']:.0%}) | rand {s['B_rand']:+.2f}", flush=True)

    json.dump({"model": a.model, "tag": a.tag, "focus": focus, "n_abl_layers": len(abl_dec),
               "decoder_n": n_dec, "rms": rms, "rows": rows, "slopes": summary},
              open(os.path.join(a.outdir, f"e4gentime_{a.tag}.json"), "w"), indent=2)
    print(f"[E4] wrote e4gentime_{a.tag}.json", flush=True)


if __name__ == "__main__":
    main()

"""E3 v2 — channel-affect ablation WITH a manipulation check.

Project the emotion-e direction out of A's-message token positions across a DENSE
band of decoder layers below focus, during B's forward pass. Measure BOTH:
  - A present-e (pool A's positions)  -> MANIPULATION CHECK: did we actually remove
    A's affect? (ablation is below focus, so a drop at focus is non-trivial.)
  - B present-e (pool B's positions)  -> did B's contagion change?

Disambiguates the v1 null:
  * A-affect drops AND B-contagion drops  -> contagion is linearly mediated (v1 was too weak).
  * A-affect drops AND B-contagion survives -> genuinely non-linear / robust channel.
  * A-affect does NOT drop even here      -> affect regenerates/redundant; text arm can't test it.
Random-direction ablation = specificity control. B reply fixed (generated once).
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
    ap.add_argument("--outdir", default="results")
    a = ap.parse_args(); os.makedirs(a.outdir, exist_ok=True)
    emos = a.emotions.split(",")

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16,
                                                 device_map="cuda").eval()
    nL = model.config.num_hidden_layers
    focus = round(0.67 * nL); steer_layer = max(0, focus - 1)
    lo = round(0.2 * nL)
    abl_dec = [d for d in range(nL) if lo <= d + 1 <= focus - 1]      # DENSE band below focus
    abl_hs = [d + 1 for d in abl_dec]
    layers = sorted(set(abl_hs) | {focus})
    print(f"[cfg] focus {focus}, ablating {len(abl_dec)} layers (hs {abl_hs[0]}..{abl_hs[-1]})", flush=True)
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
    def measure(transcripts, A_spans, B_spans, ei, mode, dirs, bs=8):
        """Return per-item {A:present-e over A-span, B:present-e over B-span} at focus under `mode`."""
        tok.padding_side = "right"; A_out, B_out = [], []
        mu, sd, coef = decs[focus]["mu"], decs[focus]["sd"], decs[focus]["coef"][ei]
        for i in range(0, len(transcripts), bs):
            batch = transcripts[i:i+bs]; asp = A_spans[i:i+bs]; bsp = B_spans[i:i+bs]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                      max_length=768, return_offsets_mapping=True)
            offs = enc.pop("offset_mapping"); enc = {k: v.to(model.device) for k, v in enc.items()}
            Am = torch.zeros_like(enc["attention_mask"], dtype=torch.bool)
            amasks = []
            for b in range(len(batch)):
                am = tmask(offs[b].tolist(), asp[b], enc["attention_mask"][b].tolist())
                amasks.append(am); Am[b] = torch.tensor(am, device=model.device)
            ABL["mode"] = mode; ABL["masks"] = Am if mode != "none" else None; ABL["dirs"] = dirs
            with torch.no_grad():
                hs = model(**enc, output_hidden_states=True).hidden_states[focus]
            ABL["mode"] = "none"; ABL["masks"] = None
            for b in range(len(batch)):
                bm = tmask(offs[b].tolist(), bsp[b], enc["attention_mask"][b].tolist())
                for tag, mask, store in (("A", amasks[b], A_out), ("B", bm, B_out)):
                    mt = torch.tensor(mask, device=model.device)
                    if mt.sum() == 0: mt = enc["attention_mask"][b].bool()
                    x = hs[b][mt].float().mean(0).cpu().numpy()
                    store.append(float((((x - mu) / sd) * coef).sum()))
        return {"A": A_out, "B": B_out}

    def gen_A_B(e, vec):
        names = [E0.NAMES[i % len(E0.NAMES)] for i in range(len(S))]
        a_prompts, ctx = [], []
        for (setting, bopen), (A, B) in zip(S, names):
            s = setting.format(A=A, B=B); bo = bopen.format(A=A, B=B)
            a_prompts.append(f"{s}\n{bo}\nContinue the conversation. Write only {A}'s next reply as one "
                             f"short paragraph, starting with '{A}:'.")
            ctx.append({"A": A, "B": B, "setting": s, "bopen": bo})
        a_reps = E2.gen_steered(model, tok, a_prompts, steer, vec)
        transcripts, A_spans, B_spans, bprompts = [], [], [], []
        for c, ar in zip(ctx, a_reps):
            A = c["A"]; ar = ar if ar.startswith(f"{A}:") else f"{A}: {ar}"
            conv = f"{c['setting']}\n{c['bopen']}\n{ar}"; c["conv"] = conv
            c["Aspan"] = (conv.rfind(f"{A}:") + len(f"{A}:"), len(conv))
            bprompts.append(f"{conv}\nWrite only {c['B']}'s next reply as one short paragraph, "
                            f"starting with '{c['B']}:'.")
        b_reps = E2.gen_steered(model, tok, bprompts, steer, None)
        for c, br in zip(ctx, b_reps):
            B = c["B"]; br = br if br.startswith(f"{B}:") else f"{B}: {br}"
            full = f"{c['conv']}\n{br}"
            transcripts.append(full); A_spans.append(c["Aspan"])
            B_spans.append((full.rfind(f"{B}:") + len(f"{B}:"), len(full)))
        return transcripts, A_spans, B_spans

    t0, as0, bs0 = gen_A_B(emos[0], None)
    tok.padding_side = "right"
    enc = tok(t0[:8], return_tensors="pt", padding=True, truncation=True, max_length=768).to(model.device)
    with torch.no_grad():
        rms = float(model(**enc, output_hidden_states=True).hidden_states[focus].float().norm(dim=-1).mean())
    print(f"[rms] {rms:.1f} | ablating {len(abl_dec)} layers", flush=True)

    rows = []
    for e in emos:
        ei = E0.EMOTIONS.index(e)
        for al in ALPHAS:
            vec = None if al == 0 else (al * rms * steer_dir[e])
            tr, As, Bs = gen_A_B(e, vec)
            none_ = measure(tr, As, Bs, ei, "none", None)
            emo_ = measure(tr, As, Bs, ei, "emo", emo_dirs[e])
            rand_ = measure(tr, As, Bs, ei, "random", rand_dirs)
            rows.append({"emotion": e, "alpha": al,
                         "A_none": float(np.mean(none_["A"])), "A_emo": float(np.mean(emo_["A"])),
                         "A_rand": float(np.mean(rand_["A"])),
                         "B_none": float(np.mean(none_["B"])), "B_emo": float(np.mean(emo_["B"])),
                         "B_rand": float(np.mean(rand_["B"]))})
            r = rows[-1]
            print(f"[{e:>9} a{al}] A: none {r['A_none']:+.2f} emo {r['A_emo']:+.2f} rand {r['A_rand']:+.2f} "
                  f"| B: none {r['B_none']:+.2f} emo {r['B_emo']:+.2f} rand {r['B_rand']:+.2f}", flush=True)

    A = np.array(ALPHAS); Ac = A - A.mean()
    def slope(er, key):
        y = np.array([r[key] for r in er]); return float((Ac @ (y - y.mean())) / (Ac @ Ac))
    summary = {}
    for e in emos:
        er = [r for r in rows if r["emotion"] == e]
        s = {k: slope(er, k) for k in ("A_none", "A_emo", "A_rand", "B_none", "B_emo", "B_rand")}
        s["A_manip_removed"] = 1 - (s["A_emo"] / s["A_none"]) if abs(s["A_none"]) > 1e-6 else float("nan")
        s["B_contagion_lost"] = 1 - (s["B_emo"] / s["B_none"]) if abs(s["B_none"]) > 1e-6 else float("nan")
        summary[e] = s
        print(f"[{e:>9}] A-slope none {s['A_none']:+.2f}->emo {s['A_emo']:+.2f} "
              f"(removed {s['A_manip_removed']:.0%}) | B-slope none {s['B_none']:+.2f}->emo {s['B_emo']:+.2f} "
              f"(contagion lost {s['B_contagion_lost']:.0%})", flush=True)

    json.dump({"model": a.model, "tag": a.tag, "focus": focus, "n_abl_layers": len(abl_dec),
               "decoder_n": n_dec, "rms": rms, "rows": rows, "slopes": summary},
              open(os.path.join(a.outdir, f"e3v2_{a.tag}.json"), "w"), indent=2)
    print(f"[E3v2] wrote e3v2_{a.tag}.json", flush=True)


if __name__ == "__main__":
    main()

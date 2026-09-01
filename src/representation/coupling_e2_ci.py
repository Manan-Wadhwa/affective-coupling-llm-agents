"""E2 tightening: bootstrap CIs on the contagion slopes + a paraphraser-affect
control. Reuses the helpers already defined in coupling_e2 / coupling_e0.

- Multi-sample: each (emotion, alpha) run over R repeats x scenarios -> per-sample
  present/other/present_para scores; slope CIs by bootstrap over samples.
- Paraphraser control: measure A's OWN present-emotion projection on the original
  steered message vs the neutralized paraphrase. If paraphrase strips A's affect,
  B's surviving contagion is meaningful; if not, "survives paraphrase" is a
  paraphraser leak (the confound flagged in review).
- alpha capped at 1.0 (drop the model-breaking alpha=2 regime).
"""
import argparse, json, os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import coupling_e0 as E0
import coupling_e2 as E2

ALPHAS = [0.0, 0.5, 1.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--outdir", default="results/representation")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                 device_map="cuda").eval()
    nL = model.config.num_hidden_layers
    focus = round(0.67 * nL); steer_layer = max(0, focus - 1)
    steer = E2.Steer(model, steer_layer)
    decs = E2.train_decoders(model, tok, focus)
    raw = {}
    for e in E2.STEER_EMOS:
        d = decs["Cp"][E0.EMOTIONS.index(e)] / decs["sd"]
        raw[e] = torch.tensor(d / (np.linalg.norm(d) + 1e-9), dtype=model.dtype, device=model.device)

    scores = {e: {a: {"present": [], "other": [], "present_para": []} for a in ALPHAS}
              for e in E2.STEER_EMOS}
    aff = {e: {"orig": [], "para": []} for e in E2.STEER_EMOS}
    S = E2.SCENARIOS

    for e in E2.STEER_EMOS:
        ei = E0.EMOTIONS.index(e)
        for al in ALPHAS:
            vec = None if al == 0 else (al * decs["rms"] * raw[e])
            for rep in range(args.repeats):
                names = [E0.NAMES[(i + rep) % len(E0.NAMES)] for i in range(len(S))]
                a_prompts, ctx = [], []
                for (setting, bopen), (A, B) in zip(S, names):
                    s = setting.format(A=A, B=B); bo = bopen.format(A=A, B=B)
                    a_prompts.append(f"{s}\n{bo}\nContinue the conversation. Write only {A}'s next "
                                     f"reply as one short paragraph, starting with '{A}:'.")
                    ctx.append({"A": A, "B": B, "setting": s, "bopen": bo})
                a_reps = E2.gen_steered(model, tok, a_prompts, steer, vec)
                # A affect: original vs paraphrase (present-e on A's own utterance)
                a_tr, a_sp, para_prompts = [], [], []
                for c, ar in zip(ctx, a_reps):
                    A = c["A"]; ar = ar if ar.startswith(f"{A}:") else f"{A}: {ar}"
                    conv = f"{c['setting']}\n{c['bopen']}\n{ar}"
                    c["ar"] = ar; c["conv"] = conv
                    a_tr.append(conv); a_sp.append((conv.rfind(f"{A}:") + len(f"{A}:"), len(conv)))
                    para_prompts.append(f"Rewrite this line in plain, emotionally neutral wording, "
                                        f"keeping only the literal information: \"{ar}\". "
                                        f"Start with '{A}:'. Output only the rewritten line.")
                a_sc = E2.utterance_score(model, tok, a_tr, a_sp, focus, decs)
                aff[e]["orig"] += a_sc["present"][:, ei].tolist()
                para_reps = E2.gen_steered(model, tok, para_prompts, steer, None)
                pa_tr, pa_sp = [], []
                for c, pr in zip(ctx, para_reps):
                    A = c["A"]; pr = pr if pr.startswith(f"{A}:") else f"{A}: {pr}"
                    pconv = f"{c['setting']}\n{c['bopen']}\n{pr}"; c["pconv"] = pconv
                    pa_tr.append(pconv); pa_sp.append((pconv.rfind(f"{A}:") + len(f"{A}:"), len(pconv)))
                pa_sc = E2.utterance_score(model, tok, pa_tr, pa_sp, focus, decs)
                aff[e]["para"] += pa_sc["present"][:, ei].tolist()
                # B replies (raw + paraphrased context), measure present/other on B
                bp = [f"{c['conv']}\nWrite only {c['B']}'s next reply as one short paragraph, "
                      f"starting with '{c['B']}:'." for c in ctx]
                b_reps = E2.gen_steered(model, tok, bp, steer, None)
                b_tr, b_sp = [], []
                for c, br in zip(ctx, b_reps):
                    B = c["B"]; br = br if br.startswith(f"{B}:") else f"{B}: {br}"
                    full = f"{c['conv']}\n{br}"; b_tr.append(full)
                    b_sp.append((full.rfind(f"{B}:") + len(f"{B}:"), len(full)))
                b_sc = E2.utterance_score(model, tok, b_tr, b_sp, focus, decs)
                scores[e][al]["present"] += b_sc["present"][:, ei].tolist()
                scores[e][al]["other"] += b_sc["other"][:, ei].tolist()
                pbp = [f"{c['pconv']}\nWrite only {c['B']}'s next reply as one short paragraph, "
                       f"starting with '{c['B']}:'." for c in ctx]
                pb_reps = E2.gen_steered(model, tok, pbp, steer, None)
                pb_tr, pb_sp = [], []
                for c, br in zip(ctx, pb_reps):
                    B = c["B"]; br = br if br.startswith(f"{B}:") else f"{B}: {br}"
                    full = f"{c['pconv']}\n{br}"; pb_tr.append(full)
                    pb_sp.append((full.rfind(f"{B}:") + len(f"{B}:"), len(full)))
                pb_sc = E2.utterance_score(model, tok, pb_tr, pb_sp, focus, decs)
                scores[e][al]["present_para"] += pb_sc["present"][:, ei].tolist()
            print(f"[{e:>9} a{al}] n={len(scores[e][al]['present'])} "
                  f"present_mean {np.mean(scores[e][al]['present']):+.2f}", flush=True)

    # bootstrap slope CIs over samples (per alpha), alpha in ALPHAS
    rng = np.random.default_rng(0)
    A = np.array(ALPHAS); Ac = A - A.mean()
    def slope_ci(key, e, nboot=2000):
        cols = [np.array(scores[e][a][key]) for a in ALPHAS]
        point = float((Ac @ (np.array([c.mean() for c in cols]) - np.mean([c.mean() for c in cols]))) / (Ac @ Ac))
        bs = []
        for _ in range(nboot):
            means = np.array([rng.choice(c, len(c)).mean() for c in cols])
            bs.append((Ac @ (means - means.mean())) / (Ac @ Ac))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        return {"slope": point, "ci_lo": float(lo), "ci_hi": float(hi),
                "sig": bool(lo > 0 or hi < 0)}
    summary = {}
    for e in E2.STEER_EMOS:
        summary[e] = {"present": slope_ci("present", e), "other": slope_ci("other", e),
                      "present_para": slope_ci("present_para", e),
                      "A_affect_orig": float(np.mean(aff[e]["orig"])),
                      "A_affect_para": float(np.mean(aff[e]["para"])),
                      "A_affect_stripped": float(np.mean(aff[e]["orig"]) - np.mean(aff[e]["para"]))}
        s = summary[e]
        print(f"[CI {e:>9}] present {s['present']['slope']:+.2f} "
              f"[{s['present']['ci_lo']:+.2f},{s['present']['ci_hi']:+.2f}]"
              f"{'*' if s['present']['sig'] else ' '}  other {s['other']['slope']:+.2f}"
              f"[{s['other']['ci_lo']:+.2f},{s['other']['ci_hi']:+.2f}]  "
              f"para {s['present_para']['slope']:+.2f}  "
              f"A_strip {s['A_affect_stripped']:+.2f} ({s['A_affect_orig']:+.2f}->{s['A_affect_para']:+.2f})",
              flush=True)

    out = {"model": args.model, "tag": args.tag, "focus": focus, "alphas": ALPHAS,
           "repeats": args.repeats, "n_per_cell": len(scores[E2.STEER_EMOS[0]][0.0]["present"]),
           "decoder_n": decs["n"], "summary": summary, "raw_scores": scores}
    json.dump(out, open(os.path.join(args.outdir, f"e2ci_{args.tag}.json"), "w"), indent=2)
    print(f"[E2-CI] wrote e2ci_{args.tag}.json", flush=True)


if __name__ == "__main__":
    main()

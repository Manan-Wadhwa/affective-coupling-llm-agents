"""Does the probe-fitting choice change the CONCLUSIONS, not just the direction?

The stability diagnostic showed two independent multinomial-logistic fits of the same
emotion direction agree at cos 0.41 (27B) / 0.22 (8B), while difference-of-means on
identical data reaches ~0.89. That is a statement about the estimator. This asks the
question that actually matters for the paper: run the core battery BOTH ways on the
SAME pooled activations and report which conclusions change.

Critically, both estimators see identical features, identical labels, identical folds
and identical seeds -- only the fitting line differs. Otherwise a reviewer attributes
any difference to generation noise and the comparison dissolves.

Reports, per estimator:
  E0   present/other decode accuracy (2-fold, held-out), cross-decode leakage,
       present<->other cosine (the near-orthogonality claim the whole program rests
       on), within-present cross-emotion cosine
  STAB split-half cos(v1, v2) for the desperate direction
  E2   contagion dose-response: slope of B's present- and other-emotion readout
       against the strength A was steered at, per emotion

Writes results/dualest_<tag>.json.
"""
import argparse, json, os, random
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from transformers import AutoModelForCausalLM, AutoTokenizer
import coupling_e0 as E0
import coupling_e2 as E2

ALPHAS = [0.0, 0.5, 1.0, 2.0]
STEER_EMOS = ["desperate", "afraid", "happy", "calm", "sad", "angry"]


# ---------- pool (generated once, shared by both estimators) ----------
def build_pool(model, tok, focus, k, seed):
    rng = random.Random(seed)
    jobs = []
    for ip, ep in enumerate(E0.EMOTIONS):
        for io, eo in enumerate(E0.EMOTIONS):
            for _ in range(k):
                A, B = rng.choice(E0.NAMES)
                jobs.append({"A": A, "B": B, "ep": ip, "eo": io, "eA": ep, "eB": eo,
                             "prompt": E0.dialogue_prompt(A, B, ep, eo, rng.choice(E0.TOPICS))})
    print(f"[pool] generating {len(jobs)} dialogues", flush=True)
    gens = E0.gen(model, tok, [j["prompt"] for j in jobs])
    items, yp, yo = [], [], []
    for j, g in zip(jobs, gens):
        if E0.leak(g, j["eA"], j["eB"]):
            continue
        p = E0.parse_final_A(g, j["A"], j["B"])
        if p is None:
            continue
        tr, us = p
        items.append({"transcript": tr, "utt_start": us, "ep": j["ep"], "eo": j["eo"]})
        yp.append(j["ep"]); yo.append(j["eo"])
    print(f"[pool] kept {len(items)}/{len(jobs)}", flush=True)
    feats = E0.pool_final(model, tok, items, [focus])[focus]
    return feats, np.array(yp), np.array(yo)


# ---------- the ONLY line that differs between arms ----------
def fit(X, y, method, n_cls):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = (X - mu) / sd
    if method == "logreg":
        C = LogisticRegression(max_iter=3000, C=0.5).fit(Xs, y).coef_
        if C.shape[0] != n_cls:            # binary edge case
            C = np.vstack([-C[0], C[0]])
        return mu, sd, C
    R = np.zeros((n_cls, X.shape[1]))
    for i in range(n_cls):
        pos, neg = Xs[y == i], Xs[y != i]
        if len(pos) >= 2 and len(neg) >= 2:
            R[i] = pos.mean(0) - neg.mean(0)
    return mu, sd, R


def unit(v):
    return v / (np.linalg.norm(v) + 1e-9)


def e0_metrics(X, yp, yo, method, n_cls, seed=0):
    """2-fold held-out decode + probe geometry. Same folds for every estimator."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    f1, f2 = idx[: len(idx) // 2], idx[len(idx) // 2:]
    acc_p, acc_o, leak = [], [], []
    for tr, te in ((f1, f2), (f2, f1)):
        mu, sd, Cp = fit(X[tr], yp[tr], method, n_cls)
        _, _, Co = fit(X[tr], yo[tr], method, n_cls)
        Z = (X[te] - mu) / sd
        acc_p.append(float((np.argmax(Z @ Cp.T, 1) == yp[te]).mean()))
        acc_o.append(float((np.argmax(Z @ Co.T, 1) == yo[te]).mean()))
        # leakage: can the PRESENT probe predict the OTHER speaker's emotion?
        leak.append(float((np.argmax(Z @ Cp.T, 1) == yo[te]).mean()))
    mu, sd, Cp = fit(X, yp, method, n_cls)
    _, _, Co = fit(X, yo, method, n_cls)
    cross = float(np.mean([abs(unit(Cp[i]) @ unit(Co[i])) for i in range(n_cls)]))
    within = float(np.mean([abs(unit(Cp[i]) @ unit(Cp[j]))
                            for i in range(n_cls) for j in range(n_cls) if i < j]))
    # split-half stability of the desperate direction
    di = E0.EMOTIONS.index("desperate")
    _, s1, C1 = fit(X[f1], yp[f1], method, n_cls)
    _, s2, C2 = fit(X[f2], yp[f2], method, n_cls)
    stab = float(unit(C1[di] / s1) @ unit(C2[di] / s2))
    return {"present_acc": float(np.mean(acc_p)), "other_acc": float(np.mean(acc_o)),
            "cross_decode_leak": float(np.mean(leak)),
            "present_other_cos": cross, "within_present_cos": within,
            "stability_cos": stab, "chance": 1.0 / n_cls}


def contagion(model, tok, focus, steer, decs, method_name, bs=16):
    """E2 dose-response: steer A, read B's present/other readout. decs = (mu,sd,Cp,Co,rms)."""
    mu, sd, Cp, Co, rms = decs
    dirs = {e: torch.tensor(unit(Cp[E0.EMOTIONS.index(e)] / sd), dtype=model.dtype,
                            device=model.device) for e in STEER_EMOS}
    S = E2.SCENARIOS
    out = {}
    for e in STEER_EMOS:
        ei = E0.EMOTIONS.index(e)
        pres, oth = [], []
        for al in ALPHAS:
            vec = None if al == 0 else (al * rms * dirs[e])
            names = [E0.NAMES[i % len(E0.NAMES)] for i in range(len(S))]
            aps, ctx = [], []
            for (setting, bopen), (A, B) in zip(S, names):
                s = setting.format(A=A, B=B); bo = bopen.format(A=A, B=B)
                aps.append(f"{s}\n{bo}\nContinue the conversation. Write only {A}'s next "
                           f"reply as one short paragraph, starting with '{A}:'.")
                ctx.append({"A": A, "B": B, "setting": s, "bopen": bo})
            areps = E2.gen_steered(model, tok, aps, steer, vec, bs=bs)
            bps = []
            for c, ar in zip(ctx, areps):
                A, B = c["A"], c["B"]
                ar = ar if ar.startswith(f"{A}:") else f"{A}: {ar}"
                c["convo"] = f"{c['setting']}\n{c['bopen']}\n{ar}"
                bps.append(f"{c['convo']}\nWrite only {B}'s next reply as one short "
                           f"paragraph, starting with '{B}:'.")
            breps = E2.gen_steered(model, tok, bps, steer, None, bs=bs)
            trs, spans = [], []
            for c, br in zip(ctx, breps):
                B = c["B"]; br = br if br.startswith(f"{B}:") else f"{B}: {br}"
                full = f"{c['convo']}\n{br}"
                trs.append(full); spans.append((full.rfind(f"{B}:") + len(f"{B}:"), len(full)))
            sc = E2.utterance_score(model, tok, trs, spans, focus,
                                    {"mu": mu, "sd": sd, "Cp": Cp, "Co": Co})
            pres.append(float(sc["present"][:, ei].mean()))
            oth.append(float(sc["other"][:, ei].mean()))
        a = np.array(ALPHAS); b = a - a.mean()
        slope = lambda y: float((b @ (np.array(y) - np.mean(y))) / (b @ b))
        out[e] = {"present_by_alpha": pres, "other_by_alpha": oth,
                  "present_slope": slope(pres), "other_slope": slope(oth)}
        print(f"[{method_name}|{e:>9}] present_slope {out[e]['present_slope']:+.3f} "
              f"other_slope {out[e]['other_slope']:+.3f}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--k", type=int, default=80)
    ap.add_argument("--outdir", default="results/estimator")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-contagion", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    hf = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16,
                                              device_map="cuda").eval()
    focus = round(0.67 * hf.config.num_hidden_layers)
    steer = E2.Steer(hf, max(0, focus - 1))
    print(f"[load] {a.model} focus={focus}", flush=True)

    X, yp, yo = build_pool(hf, tok, focus, a.k, a.seed)
    n_cls = len(E0.EMOTIONS)
    rms = float(np.linalg.norm(X, axis=1).mean())

    out = {"model": a.model, "focus": focus, "k": a.k, "seed": a.seed,
           "n_pool": int(len(X)), "rms": rms, "arms": {}}

    for method in ("logreg", "dom"):
        print(f"\n===== estimator: {method} =====", flush=True)
        m = e0_metrics(X, yp, yo, method, n_cls, seed=a.seed)
        print(f"[{method}] E0 present {m['present_acc']:.3f} other {m['other_acc']:.3f} "
              f"leak {m['cross_decode_leak']:.3f} present<->other cos {m['present_other_cos']:.3f} "
              f"stability {m['stability_cos']:.3f}", flush=True)
        arm = {"e0": m}
        if not a.skip_contagion:
            mu, sd, Cp = fit(X, yp, method, n_cls)
            _, _, Co = fit(X, yo, method, n_cls)
            arm["contagion"] = contagion(hf, tok, focus, steer, (mu, sd, Cp, Co, rms), method)
        out["arms"][method] = arm
        json.dump(out, open(os.path.join(a.outdir, f"dualest_{a.tag}.json"), "w"), indent=2)

    print(f"\n[done] wrote {a.outdir}/dualest_{a.tag}.json", flush=True)


if __name__ == "__main__":
    main()

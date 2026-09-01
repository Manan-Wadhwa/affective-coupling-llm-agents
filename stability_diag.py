"""Is the emotion direction unstable because of the CONCEPT or because of the ESTIMATOR?

The P1 stability gate failed on both models (cos(v1,v2)=0.427 on Qwen3.6-27B,
0.195 on Llama-3-8B-abliterated). The decoder in coupling_e2.train_decoders is a
multinomial logistic regression fit on ~170-400 pooled examples in 5120 dims --
wildly underdetermined, so an unstable fit is the expected outcome rather than
evidence about emotions.

This separates the two explanations in ONE generation pass:
  * generate a large dialogue pool once, split it into two DISJOINT halves,
  * for each target n and each estimator, fit the desperate direction on each
    half independently, and report cos(v_half1, v_half2).
If cos rises toward 1 with n, or difference-of-means beats logistic regression,
the instability is estimator variance and is fixable. If cos stays low for every
n and every estimator, the direction itself does not reproduce -- which is a
finding about the probe literature, not a bug.

Writes results/stabdiag_<tag>.json.
"""
import argparse, json, os, random
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from transformers import AutoModelForCausalLM, AutoTokenizer
import coupling_e0 as E0

TARGET_N = [150, 300, 600, 1200, 2400]


def build_pool(model, tok, focus, k, seed):
    rng = random.Random(seed)
    jobs = []
    for ip, ep in enumerate(E0.EMOTIONS):
        for io, eo in enumerate(E0.EMOTIONS):
            for _ in range(k):
                A, B = rng.choice(E0.NAMES)
                topic = rng.choice(E0.TOPICS)
                jobs.append({"A": A, "B": B, "ep": ip, "eo": io, "eA": ep, "eB": eo,
                             "prompt": E0.dialogue_prompt(A, B, ep, eo, topic)})
    print(f"[pool] generating {len(jobs)} dialogues", flush=True)
    gens = E0.gen(model, tok, [j["prompt"] for j in jobs])
    items = []
    for j, g in zip(jobs, gens):
        if E0.leak(g, j["eA"], j["eB"]):
            continue
        p = E0.parse_final_A(g, j["A"], j["B"])
        if p is None:
            continue
        tr, us = p
        items.append({"transcript": tr, "utt_start": us, "ep": j["ep"], "eo": j["eo"]})
    print(f"[pool] kept {len(items)}/{len(jobs)} after leak+parse filtering", flush=True)
    feats = E0.pool_final(model, tok, items, [focus])[focus]
    y = np.array([it["ep"] for it in items])
    return feats, y


def unit(v):
    return v / (np.linalg.norm(v) + 1e-9)


def fit_logreg(X, y, ei, C=0.5):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = (X - mu) / sd
    coef = LogisticRegression(max_iter=3000, C=C).fit(Xs, y).coef_
    return unit(coef[ei] / sd)


def fit_dom(X, y, ei):
    """Difference of means: v = mean(class ei) - mean(rest), in standardized space."""
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = (X - mu) / sd
    pos, neg = Xs[y == ei], Xs[y != ei]
    if len(pos) < 2 or len(neg) < 2:
        return None
    return unit((pos.mean(0) - neg.mean(0)) / sd)


def subsample(idx, n, rng):
    return idx if len(idx) <= n else rng.choice(idx, n, replace=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--k", type=int, default=70, help="dialogues per (present,other) cell")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    hf = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16,
                                              device_map="cuda").eval()
    focus = round(0.67 * hf.config.num_hidden_layers)
    print(f"[load] {a.model} layers={hf.config.num_hidden_layers} focus={focus}", flush=True)

    X, y = build_pool(hf, tok, focus, a.k, a.seed)
    ei = E0.EMOTIONS.index("desperate")

    rng = np.random.default_rng(a.seed)
    order = rng.permutation(len(X))
    h1, h2 = order[: len(order) // 2], order[len(order) // 2:]

    out = {"model": a.model, "focus": focus, "k": a.k, "seed": a.seed,
           "n_pool": int(len(X)), "n_half1": int(len(h1)), "n_half2": int(len(h2)),
           "hidden": int(X.shape[1]), "results": []}

    for n in TARGET_N:
        if n > min(len(h1), len(h2)):
            continue
        row = {"n_per_half": n}
        for name, fn in (("logreg", fit_logreg), ("diff_of_means", fit_dom)):
            cs = []
            for rep in range(3):                       # 3 independent subsamples
                r = np.random.default_rng(a.seed + 100 * rep)
                i1, i2 = subsample(h1, n, r), subsample(h2, n, r)
                v1, v2 = fn(X[i1], y[i1], ei), fn(X[i2], y[i2], ei)
                if v1 is None or v2 is None:
                    continue
                cs.append(float(v1 @ v2))
            if cs:
                row[name] = {"cos_mean": float(np.mean(cs)), "cos_all": cs}
        out["results"].append(row)
        msg = "  ".join(f"{m}={row[m]['cos_mean']:+.3f}" for m in ("logreg", "diff_of_means") if m in row)
        print(f"[n={n:5d}/half] {msg}", flush=True)

    p = os.path.join(a.outdir, f"stabdiag_{a.tag}.json")
    json.dump(out, open(p, "w"), indent=2)
    print(f"[done] wrote {p}", flush=True)


if __name__ == "__main__":
    main()

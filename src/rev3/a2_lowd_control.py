"""a2_lowd_control.py -- Reviewer 2's control: is the logistic tie-break just n < d?

Project the layer's activations to d' << n (PCA fitted on the whole pool, or a fixed random
projection) and repeat the split-half battery: split-half cosine of logistic (C = 0.5) and
difference-of-means directions at n per half in the grid, training accuracy, HELD-OUT accuracy
on the other half, and the correlation between the two estimators' projections on held-out
items. Same splits and seeds as a2_followup.split_half_local. Output: a2_lowd_{tag}.json.
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import acl_core as C

TARGETS = (75, 150, 300, 600, 1200, 2000); SEEDS = tuple(range(10))


def fit_logreg(X, y, seed):
    from sklearn.linear_model import LogisticRegression
    mu, sd = X.mean(0), X.std(0) + 1e-6; Xs = (X - mu) / sd
    m = LogisticRegression(max_iter=3000, C=0.5, random_state=seed).fit(Xs, y)
    W = np.asarray(m.coef_); n_cls = int(y.max()) + 1; assert W.shape[0] == n_cls
    return {"C": W, "mu": mu, "sd": sd, "b": np.asarray(m.intercept_), "train_acc": float((m.predict(Xs) == y).mean()), "model": m}


def fit_dom(X, y, seed):
    d = C.fit_direction(X, y, "dom", seed=seed); return d


def raw_dirs(dec, n_cls):
    return np.stack([C.raw_direction(dec, i) for i in range(n_cls)])


def heldout_acc(dec, X, y):
    return C.decode_acc(dec, X, y)


def run(X, y, targets, seeds, label):
    N, n_cls = len(y), int(y.max()) + 1; out = {}
    for n in targets:
        if 2 * n > N: continue
        cells = []
        for s in seeds:
            idx = np.random.default_rng(s).permutation(N); h1, h2 = idx[:n], idx[n:2 * n]
            if len(set(y[h1])) < n_cls or len(set(y[h2])) < n_cls: continue
            L1, L2 = fit_logreg(X[h1], y[h1], s), fit_logreg(X[h2], y[h2], s); D1, D2 = fit_dom(X[h1], y[h1], s), fit_dom(X[h2], y[h2], s)
            dl1, dl2 = raw_dirs(L1, n_cls), raw_dirs(L2, n_cls); dd1, dd2 = raw_dirs(D1, n_cls), raw_dirs(D2, n_cls)
            # projections of held-out half (h2) on the half-1 directions, per class, correlation between estimators
            Xo = X[h2]; corr = float(np.mean([np.corrcoef(Xo @ dl1[i], Xo @ dd1[i])[0, 1] for i in range(n_cls)]))
            def unit(M): return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
            sl1, sl2 = unit(L1["C"]), unit(L2["C"]); sd1, sd2 = unit(D1["C"]), unit(D2["C"])   # standardised-coordinate rows
            cells.append({"sh_logreg": float(np.mean(np.sum(dl1 * dl2, 1))), "sh_dom": float(np.mean(np.sum(dd1 * dd2, 1))),
                          "sh_logreg_std": float(np.mean(np.sum(sl1 * sl2, 1))), "sh_dom_std": float(np.mean(np.sum(sd1 * sd2, 1))),
                          "cos_logreg_dom_std": float(np.mean(np.sum(sl1 * sd1, 1))),
                          "train_acc_logreg": L1["train_acc"], "heldout_acc_logreg": heldout_acc({"C": L1["C"], "mu": L1["mu"], "sd": L1["sd"], "b": L1["b"]}, X[h2], y[h2]),
                          "heldout_acc_dom": heldout_acc(D1, X[h2], y[h2]), "proj_corr_logreg_dom_heldout": corr,
                          "cos_logreg_dom_same_half": float(np.mean(np.sum(dl1 * dd1, 1)))})
        agg = {k: {"mean": float(np.mean([c[k] for c in cells])), "ci": [float(x) for x in np.percentile([c[k] for c in cells], [2.5, 97.5])]} for k in cells[0]}
        agg["n_seeds"] = len(cells); out[str(n)] = agg
        print(f"[lowd {label}] n={n:5d} sh(std) logreg {agg['sh_logreg_std']['mean']:.3f} dom {agg['sh_dom_std']['mean']:.3f} | sh(raw) logreg {agg['sh_logreg']['mean']:.3f} dom {agg['sh_dom']['mean']:.3f} | train acc {agg['train_acc_logreg']['mean']:.3f} heldout logreg {agg['heldout_acc_logreg']['mean']:.3f} dom {agg['heldout_acc_dom']['mean']:.3f} | proj corr {agg['proj_corr_logreg_dom_heldout']['mean']:.3f} cos same-half {agg['cos_logreg_dom_same_half']['mean']:.3f}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--feats"); ap.add_argument("--layer", type=int, default=43); ap.add_argument("--tag", default="qwen36-27b-l43")
    ap.add_argument("--dims", default="full,1000,300,100"); ap.add_argument("--outdir", default="/marimo/results"); ap.add_argument("--seeds", type=int, default=10)
    a = ap.parse_args(); z = np.load(a.feats); y = np.asarray(z["yp"]).astype(int); X = np.asarray(z[f"L{a.layer}"], dtype=np.float64)
    Xc = X - X.mean(0); U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    prov = C.Provenance(script="src/rev3/a2_lowd_control.py", config={"feats": a.feats, "layer": a.layer, "dims": a.dims, "targets": list(TARGETS), "seeds": a.seeds, "n_pool": int(len(y)), "d": int(X.shape[1]), "projection": "PCA fitted on the whole pool (centred), top-d' components; 'full' = no projection"},
                        model_id="", model_revision="", seeds={"splits": list(range(a.seeds))}, code_sha=C.code_hash(__file__, C.__file__))
    res = {"tag": a.tag, "layer": a.layer, "n_pool": int(len(y)), "d": int(X.shape[1]), "by_dim": {}}
    for dim in a.dims.split(","):
        if dim == "full": Xp = X
        else: k = int(dim); Xp = Xc @ Vt[:k].T
        res["by_dim"][dim] = {"d_eff": int(Xp.shape[1]), "cells": run(Xp, y, TARGETS, tuple(range(a.seeds)), f"d'={dim}")}
    C.write_result(os.path.join(a.outdir, f"a2_lowd_{a.tag}.json"), res, prov); print("LOWD_DONE", flush=True)


if __name__ == "__main__":
    main()

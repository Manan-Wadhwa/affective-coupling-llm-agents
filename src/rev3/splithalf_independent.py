"""Independent second implementation of the split-half stability statistic (§7).

RESEARCH_PLAN rev 3 §7: "Dual implementation -- scoped to three metrics. The transmission
budget beta (B7), the split-half stability statistic (A1/A2), and the J-space transfer
measure (C1). Written without reference to each other; disagreement halts the day."

This is the second implementation of the second metric. It deliberately shares NO code with
`acl_core.py` -- not the estimators, not the splitting, not the cosine -- and it is written
from the definition rather than from the other implementation:

    Fit the same estimator twice, on two subsamples of the data that have no example in
    common. Take the cosine between the two resulting directions, in the space the
    direction is actually used in. Repeat over independent random splits and report the
    distribution.

Deliberate differences from acl_core, so that agreement is evidence rather than a shared
bug:
  * splitting: acl_core permutes and takes two disjoint blocks; this one assigns each
    STRATUM independently, so both halves are class-balanced by construction
  * logistic fit: scipy L-BFGS on the multinomial cross-entropy with explicit L2, rather
    than sklearn's LogisticRegression
  * difference of means: computed in raw space and then rescaled, rather than computed in
    standardized space and divided back
  * cosine: computed on float64 raw-space vectors normalised here

Usage:
    python splithalf_independent.py --npz feats_qwen36-27b.npz --layer 43
    python splithalf_independent.py --selftest
"""
import argparse, itertools, json, sys
import numpy as np


# --------------------------------------------------------------------------------------
# estimators, written from scratch
# --------------------------------------------------------------------------------------

def _softmax_rows(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def fit_logreg_lbfgs(Xz, y, n_cls, l2=2.0, maxiter=400):
    """Multinomial logistic regression by L-BFGS on the penalised cross-entropy.

    sklearn's C=0.5 corresponds to an L2 penalty coefficient of 1/C = 2 on the summed
    loss, which is what `l2` is here.
    """
    from scipy.optimize import minimize
    n, d = Xz.shape
    Y = np.zeros((n, n_cls))
    Y[np.arange(n), y] = 1.0

    def obj(w):
        W = w.reshape(n_cls, d)
        Z = Xz @ W.T
        P = _softmax_rows(Z)
        nll = -np.sum(Y * np.log(P + 1e-12))
        grad = (P - Y).T @ Xz
        return nll + 0.5 * l2 * np.sum(W * W), (grad + l2 * W).ravel()

    r = minimize(obj, np.zeros(n_cls * d), jac=True, method="L-BFGS-B",
                 options={"maxiter": maxiter})
    return r.x.reshape(n_cls, d)


def fit_diff_of_means_raw(X, y, n_cls):
    """One-vs-rest difference of means, computed in RAW space, then divided by the
    feature scale -- the reverse order from acl_core, which works in standardized space
    and divides back. Algebraically the same target, different rounding path."""
    sd = X.std(axis=0) + 1e-6
    out = np.zeros((n_cls, X.shape[1]))
    for k in range(n_cls):
        m_pos = X[y == k].mean(axis=0)
        m_neg = X[y != k].mean(axis=0)
        out[k] = (m_pos - m_neg) / (sd * sd)      # (mu+ - mu-)/sd in std space, /sd to raw
    return out


def stratified_disjoint_halves(y, n_per_half, rng, n_cls):
    """Assign each class's indices independently, so both halves are class-balanced by
    construction. acl_core instead permutes globally and cuts two blocks."""
    per_cls = n_per_half // n_cls
    h1, h2 = [], []
    for k in range(n_cls):
        idx = np.where(y == k)[0]
        if len(idx) < 2 * per_cls:
            return None, None
        pick = rng.permutation(idx)
        h1.append(pick[:per_cls])
        h2.append(pick[per_cls:2 * per_cls])
    return np.concatenate(h1), np.concatenate(h2)


def split_half_independent(X, y, estimator="dom", n_per_half=600, n_splits=10, seed0=0):
    """Return the per-split, per-class cosines between two disjoint fits."""
    X = np.asarray(X, dtype=np.float64)
    n_cls = int(y.max()) + 1
    rows = []
    for s in range(n_splits):
        rng = np.random.default_rng(10_000 + seed0 + s)
        i1, i2 = stratified_disjoint_halves(y, n_per_half, rng, n_cls)
        if i1 is None:
            continue
        dirs = []
        for idx in (i1, i2):
            Xi, yi = X[idx], y[idx]
            if estimator == "dom":
                D = fit_diff_of_means_raw(Xi, yi, n_cls)
            elif estimator == "logreg":
                mu, sd = Xi.mean(0), Xi.std(0) + 1e-6
                W = fit_logreg_lbfgs((Xi - mu) / sd, yi, n_cls)
                D = W / sd                        # back to raw space
            else:
                raise ValueError(estimator)
            dirs.append(D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-30))
        rows.append([float(dirs[0][k] @ dirs[1][k]) for k in range(n_cls)])
    A = np.array(rows)
    return {"estimator": estimator, "n_per_half": int(n_per_half),
            "n_splits": int(len(A)),
            "mean": float(A.mean()) if len(A) else float("nan"),
            "per_class": A.mean(axis=0).tolist() if len(A) else [],
            "per_split": A.tolist()}


# --------------------------------------------------------------------------------------
# the cross-check
# --------------------------------------------------------------------------------------

def compare(X, y, layer_label="", tol=0.05, n_list=(150, 300, 600), n_splits=8):
    """Run BOTH implementations and report the gap. §7: disagreement halts the day."""
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    # sandbox layout puts everything in one directory; the repo keeps the core in src/lib
    for cand in (here, os.path.join(here, "..", "lib")):
        if os.path.exists(os.path.join(cand, "acl_core.py")):
            sys.path.insert(0, cand)
            break
    import acl_core as C
    verdict, table = "AGREE", []
    for est in ("dom", "logreg"):
        for n in n_list:
            if 2 * n > len(y):
                continue
            a = C.split_half(X, y, est, n, seeds=range(n_splits))["mean"]
            b = split_half_independent(X, y, est, n, n_splits=n_splits)["mean"]
            gap = abs(a - b)
            if gap > tol:
                verdict = "DISAGREE"
            table.append({"estimator": est, "n_per_half": n, "acl_core": round(a, 4),
                          "independent": round(b, 4), "gap": round(gap, 4),
                          "ok": bool(gap <= tol)})
            print(f"  {layer_label} {est:>7} n={n:<5} acl_core {a:+.4f}   "
                  f"independent {b:+.4f}   gap {gap:.4f}  {'ok' if gap<=tol else 'DISAGREE'}",
                  flush=True)
    return {"verdict": verdict, "tolerance": tol, "rows": table}


def selftest():
    """Synthetic data with known structure, so both implementations have a right answer."""
    rng = np.random.default_rng(0)
    n_cls, d, n = 6, 400, 3000
    V = rng.standard_normal((n_cls, d)); V /= np.linalg.norm(V, axis=1, keepdims=True)
    Q = rng.standard_normal((d, 15)); Q /= np.linalg.norm(Q, axis=0, keepdims=True)
    y = rng.integers(0, n_cls, n)
    X = (rng.standard_normal((n, 15)) @ Q.T * 5.0 + rng.standard_normal((n, d)) + 2.5 * V[y])
    print("[selftest] synthetic, 6 classes, d=400, anisotropic nuisance")
    res = compare(X.astype(np.float32), y, layer_label="[synth]", n_list=(150, 300, 600))
    print(f"\n[selftest] {res['verdict']}")
    return 0 if res["verdict"] == "AGREE" else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", help="pool feature cache written by acl_core.pool_features")
    ap.add_argument("--layer", type=int)
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest or not a.npz:
        sys.exit(selftest())
    z = np.load(a.npz)
    X, y = z[f"L{a.layer}"], z["yp"]
    print(f"[dual] {a.npz} L{a.layer}: X{X.shape}")
    res = compare(X, y, layer_label=f"L{a.layer}")
    res["source"] = {"npz": a.npz, "layer": a.layer, "n": int(len(y))}
    print(f"\n[dual] {res['verdict']}")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=2)
        print(f"[dual] wrote {a.out}")
    sys.exit(0 if res["verdict"] == "AGREE" else 1)


if __name__ == "__main__":
    main()

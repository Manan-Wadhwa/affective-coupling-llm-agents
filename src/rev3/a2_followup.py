#!/usr/bin/env python3
"""A2-followup — the five things the A2 estimator sweep got wrong, measured (CPU-only).

An independent review of `results/rev3/a2_estimator_<tag>.json` found five defects. One
line each, and what this driver does about them:

  (1) REGULARISATION SCHEDULE, NOT A LEARNING CURVE. A2's `logreg` holds C = 0.5 fixed
      across the n grid, so sklearn's per-sample penalty weight 1/(2*C*n) weakens 27x from
      n=75 to n=2000, and the apparent plateau of the logistic split-half curve after
      n=600 reverses under C=0.05 -- the curve's shape is the schedule, not the estimator.
      -> adds `logreg_lam`, which holds the per-sample penalty FIXED across n (below).
  (2) ONLY RAW-SPACE COSINES WERE COMPUTED. `acl_core.split_half` takes `space=`, and A2
      only ever called it with the default 'raw' (C/sd unit vectors); the standardised
      coefficient row ('std') was never measured, so we cannot tell how much of the
      logistic instability is the 1/sd rescaling.  -> records BOTH spaces for every cell.
  (3) THE "dom" RAW DIRECTION IS NOT CAA's MEAN DIFFERENCE. `fit_direction` builds dom in
      standardised space, C = (mu+ - mu-)/sd, and `raw_direction` divides by sd again, so
      the steering vector is (mu+ - mu-)/sd^2. The logreg-vs-dom cosine A2 reports (0.59
      at layer 43) therefore does NOT measure the quantity arXiv 2604.08169 reports as
      0.98-0.99, which is CAA's RAW mean difference against a raw logistic weight.
      -> computes the raw mean difference explicitly and reports every pairing, so the
      reader can see which pair the other paper's number corresponds to.
  (4) THE LEDOIT-WOLF SHRINKAGE IS DISCARDED. `lw_shrinkage_and_apply` returns lambda and
      `fit_direction` throws it away (`sol, _lam = ...`), so the covariance regime that
      mass_mean_cov actually operated in is unrecorded.  -> records lambda per (layer, n).
  (5) THE CROSS-ESTIMATOR SUBSAMPLE WAS ONE FIXED DRAW. A2 used the same 2000 rows
      (rng seed a.seed+1) at every depth and reported a point value with no CI.
      -> 10 independently seeded draws, with a bootstrap CI over draws.

WHAT THIS DRIVER MEASURES. From the cached activations alone (no model, no GPU):
    layer x estimator x n x space  ->  split-half cosine, mean / CI / per-seed / per-class
    layer x pair                   ->  cross-estimator cosine at n=2000 over 10 draws + CI
    layer x n                      ->  the Ledoit-Wolf lambda mass_mean_cov discards
Estimators in the sweep: `logreg` (C=0.5, exactly as A2), `logreg_cv` (acl_core's tuned
fit, which A2 excluded from the sweep on cost grounds), `logreg_lam` (new, defined here),
and `dom`.

THE C_n FORMULA. sklearn minimises  0.5*||w||^2 + C * sum_{i=1..n} loss_i, so the penalty
weight PER SAMPLE is 1/(2*C*n). To hold that constant at its n=600 value across the whole
grid:

        C_n = C_REF * N_REF / n_fit  =  0.5 * 600 / n_fit

which gives C_600 = 0.5 exactly -- so at n=600 `logreg_lam` and `logreg` are the same fit,
and the selftest asserts that to 1e-9. Everywhere else `logreg_lam` isolates the estimator
from the schedule: any remaining n-dependence in its curve is sample size, not penalty.

`logreg_lam` is implemented in THIS file, not in acl_core: acl_core stays byte-identical
to the copy that produced the committed results, so old and new numbers stay comparable.
For the same reason the split-half loop is re-implemented here as `split_half_local` --
it mirrors `acl_core.split_half`'s permutation, disjoint halves, all-classes-present check
and cosine EXACTLY, but takes a fit callable so a locally defined estimator can go through
it. `check_dom_equivalence` asserts the two agree to 1e-9 on synthetic data (per-seed, both
spaces) and runs on every invocation, not only under --selftest.

Usage:  python a2_followup.py --feats /marimo/work/feats_qwen36-27b.npz --tag qwen36-27b \
                              [--layers 16,43,54] [--outdir /marimo/out] [--selftest]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import warnings

import numpy as np

_WARN_COUNTS: dict = {}


class _dedupe_warnings:
    """Print each distinct warning once, and COUNT the rest into the result file.

    `logreg_cv` reaches sklearn's LogisticRegressionCV, which on sklearn >= 1.9 emits three
    FutureWarnings per fit; the sweep makes 2 * len(seeds) * len(n) * len(layers) of them
    and the tables would be unreadable. A plain `filterwarnings('once')` does not survive
    the `catch_warnings` block sklearn uses internally, hence recording them here. Nothing
    is silenced: the first instance is printed and the totals are reported, so a
    ConvergenceWarning storm is still visible as a count.
    """

    def __enter__(self):
        self._cm = warnings.catch_warnings(record=True)
        self._w = self._cm.__enter__()
        warnings.simplefilter("always")
        return self

    def __exit__(self, *exc):
        caught = list(self._w)
        self._cm.__exit__(*exc)
        for m in caught:
            key = f"{m.category.__name__}: {m.message}"
            _WARN_COUNTS[key] = _WARN_COUNTS.get(key, 0) + 1
            if _WARN_COUNTS[key] == 1:
                print(f"  [warn] {key}", flush=True)
        return False

# sandbox layout puts everything in one directory; the repo keeps the core in src/lib
_HERE = os.path.dirname(os.path.abspath(__file__))
ACL_CORE_PATH = ""
for _c in (_HERE, os.path.join(_HERE, "..", "lib")):
    if os.path.exists(os.path.join(_c, "acl_core.py")):
        sys.path.insert(0, _c)
        ACL_CORE_PATH = os.path.abspath(os.path.join(_c, "acl_core.py"))
        break
import acl_core as C  # noqa: E402


# --- declared BEFORE the run (§6 hyperparameter grid declaration) ----------------------
TARGET_N = (75, 150, 300, 600, 1200, 2000)          # identical to A2
SPLIT_SEEDS = tuple(range(10))                      # identical to A2
SPACES = ("raw", "std")
SH_ESTIMATORS = ("logreg", "logreg_cv", "logreg_lam", "dom")
CROSS_N = 2000                                      # A2's cross-estimator n
CROSS_SEEDS = tuple(range(10))                      # A2 used ONE draw; finding (5)
C_REF, N_REF = 0.5, 600
C_N_FORMULA = ("C_n = C_REF * N_REF / n_fit = 0.5 * 600 / n_fit; sklearn's objective is "
               "0.5*||w||^2 + C*sum_i loss_i, so the per-sample penalty 1/(2*C*n) is held "
               "at its C=0.5, n=600 value for every n. C_600 == 0.5 exactly.")

# The five vectors compared at n=2000, named <estimator>_<space>. All 10 ordered pairs are
# recorded; PAIRS_OF_INTEREST says which ones the review asked for and what each means.
# caa_raw is the only one that never touches standardisation.
CROSS_VECS = ("caa_raw", "logreg_raw", "dom_raw", "logreg_std", "dom_std")
PAIRS_OF_INTEREST = {
    "caa_raw|logreg_raw": "review item (i)x(ii): THE 2604.08169 quantity -- CAA's raw mean "
                          "difference vs the raw-space logistic weight (their 0.98/0.99)",
    "logreg_raw|dom_raw": "review item (ii)x(iii): what A2 actually reported as "
                          "'logreg vs dom' (0.5922 at layer 43)",
    "logreg_std|dom_std": "review item (iv): the same comparison in standardised space",
    "caa_raw|dom_raw": "review item (v): CAA's raw mean difference vs raw_direction(dom) "
                       "= (mu+ - mu-)/sd^2 -- the cost of the double sd division",
    "caa_raw|dom_std": "diagnostic: one sd division only, (mu+ - mu-)/sd",
}


# ---------------------------------------------------------------------------
# logreg_lam — the fixed-per-sample-penalty logistic fit (new; finding 1)
# ---------------------------------------------------------------------------

def fit_logreg_lam(X: np.ndarray, y: np.ndarray, n_cls: int | None = None,
                   seed: int = 0) -> dict:
    """Multinomial logistic on standardised X with C_n = 0.5 * 600 / n_fit.

    Returns the same dict shape as `acl_core.fit_direction` ({'mu','sd','C','b','method'})
    so `raw_direction`, `decode_acc` and the split-half space logic apply unchanged. The
    body deliberately mirrors fit_direction's logreg branch line for line -- same
    standardisation (sd + 1e-6), same solver arguments (max_iter=3000, random_state=seed),
    same per-class midpoint intercept -- so that at n_fit == 600, where C_n == 0.5, this
    reproduces method='logreg' bit for bit.
    """
    from sklearn.linear_model import LogisticRegression
    n_cls = n_cls or int(y.max()) + 1
    n_fit = int(X.shape[0])
    c_n = C_REF * N_REF / float(n_fit)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = ((X - mu) / sd).astype(np.float64)
    W = np.asarray(LogisticRegression(max_iter=3000, C=c_n, random_state=seed)
                   .fit(Xs, y).coef_, dtype=np.float64)
    # sklearn returns one coefficient row per class it SAW, so a subsample missing a class
    # would silently shift every later class's row up by one. Fail instead.
    assert W.shape[0] == n_cls, (
        f"logreg_lam: fit on {X.shape[0]} rows saw {W.shape[0]} classes, expected {n_cls} "
        f"(present: {sorted(set(np.asarray(y).tolist()))}) -- coefficient rows would "
        f"misalign with the class index")
    Cm = np.zeros((n_cls, X.shape[1]), dtype=np.float64)
    Cm[:] = W
    b = np.zeros(n_cls)
    for i in range(n_cls):
        pos, neg = Xs[y == i], Xs[y != i]
        if len(pos) < 1 or len(neg) < 1:
            continue
        b[i] = -0.5 * (pos @ Cm[i]).mean() - 0.5 * (neg @ Cm[i]).mean()
    return {"mu": mu, "sd": sd, "C": Cm, "b": b, "method": "logreg_lam", "C_n": c_n}


def make_fit(est: str):
    """(X, y, n_cls, seed) -> decoder dict, for either an acl_core method or a local one."""
    if est == "logreg_lam":
        return lambda X, y, n_cls, seed: fit_logreg_lam(X, y, n_cls=n_cls, seed=seed)
    return lambda X, y, n_cls, seed: C.fit_direction(X, y, est, n_cls=n_cls, seed=seed)


# ---------------------------------------------------------------------------
# split_half_local — acl_core.split_half with a pluggable fit (findings 1, 2)
# ---------------------------------------------------------------------------

def split_half_local(X: np.ndarray, y: np.ndarray, fit_fn, n_per_half: int,
                     seeds=SPLIT_SEEDS, spaces=SPACES, method: str = "") -> dict:
    """Faithful copy of `acl_core.split_half`, differing ONLY in that the estimator is a
    callable and that both spaces come out of one pass over the same fits.

    Everything that could change a number is reproduced exactly: n_cls from y.max()+1 over
    the FULL label vector, np.random.default_rng(s).permutation(len(y)), the two disjoint
    blocks idx[:n] and idx[n:2n], the skip when either half is missing a class, the fit
    seed = split seed, and the two cosine definitions (raw: C/sd unit vectors; std:
    unit-normalised coefficient rows, both with acl_core's +1e-9 norm guard).

    Computing both spaces from one pair of fits is not an approximation: every estimator
    used here is deterministic given (X, y, n_cls, seed), so the second acl_core call would
    refit the identical halves. `check_dom_equivalence` verifies this against acl_core.

    Returns {space: <the dict acl_core.split_half returns for that space>}.
    """
    n_cls = int(y.max()) + 1
    per_seed = {sp: [] for sp in spaces}
    used = []
    for s in seeds:
        rng = np.random.default_rng(s)
        idx = rng.permutation(len(y))
        need = 2 * n_per_half
        if len(idx) < need:
            continue
        h1, h2 = idx[:n_per_half], idx[n_per_half:need]
        if len(set(y[h1])) < n_cls or len(set(y[h2])) < n_cls:
            continue
        d1 = fit_fn(X[h1], y[h1], n_cls, s)
        d2 = fit_fn(X[h2], y[h2], n_cls, s)
        for sp in spaces:
            cos = []
            for i in range(n_cls):
                if sp == "raw":
                    a, b = C.raw_direction(d1, i), C.raw_direction(d2, i)
                else:
                    a = d1["C"][i] / (np.linalg.norm(d1["C"][i]) + 1e-9)
                    b = d2["C"][i] / (np.linalg.norm(d2["C"][i]) + 1e-9)
                cos.append(float(a @ b))
            per_seed[sp].append(cos)
        used.append(int(s))
    out = {}
    for sp in spaces:
        P = np.array(per_seed[sp], dtype=float)
        if P.size == 0:
            out[sp] = {"n_per_half": int(n_per_half), "method": method, "space": sp,
                       "n_seeds": 0, "mean": float("nan"),
                       "ci": [float("nan"), float("nan")],
                       "per_class": [], "per_seed": [], "seeds_used": []}
            continue
        m = P.mean(1)
        out[sp] = {"n_per_half": int(n_per_half), "method": method, "space": sp,
                   "n_seeds": int(len(P)), "mean": float(m.mean()), "ci": C.ci_of(m),
                   "per_class": P.mean(0).tolist(),
                   "per_class_labels": C.EMOTIONS[:P.shape[1]],
                   "per_seed": P.tolist(), "seeds_used": used}
    return out


def check_dom_equivalence(X: np.ndarray, y: np.ndarray, n_list=(75, 150),
                          seeds=SPLIT_SEEDS, tol: float = 1e-9) -> dict:
    """`split_half_local` must be `acl_core.split_half` for a method both can run.

    dom is the cheap one, so it is the witness: per-seed x per-class cosines must match to
    `tol` in BOTH spaces. Asserts, and returns the gaps so they land in the result file.
    """
    fit = make_fit("dom")
    gaps = {}
    for n in n_list:
        if 2 * n > len(y):
            continue
        loc = split_half_local(X, y, fit, n, seeds=seeds, method="dom")
        for sp in SPACES:
            ref = C.split_half(X, y, "dom", n, seeds=seeds, space=sp)
            A = np.array(ref["per_seed"], dtype=float)
            B = np.array(loc[sp]["per_seed"], dtype=float)
            assert A.shape == B.shape, f"shape {A.shape} != {B.shape} at n={n}/{sp}"
            g = float(np.max(np.abs(A - B))) if A.size else 0.0
            gaps[f"n{n}/{sp}"] = g
            assert g <= tol, f"split_half_local != acl_core.split_half at n={n}/{sp}: {g}"
    return gaps


# ---------------------------------------------------------------------------
# cross-estimator block (findings 3, 5)
# ---------------------------------------------------------------------------

def _unit_rows(M: np.ndarray) -> np.ndarray:
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)


def caa_raw_dirs(X: np.ndarray, y: np.ndarray, n_cls: int) -> np.ndarray:
    """CAA's steering vector: mu_pos - mu_neg on RAW activations, one-vs-rest, no
    standardisation anywhere (Rimsky et al.; the quantity 2604.08169 correlates against a
    logistic weight). This is NOT what acl_core's 'dom' produces -- see finding (3)."""
    Xf = np.asarray(X, dtype=np.float64)
    D = np.zeros((n_cls, Xf.shape[1]))
    for i in range(n_cls):
        pos, neg = Xf[y == i], Xf[y != i]
        if len(pos) < 2 or len(neg) < 2:
            continue
        D[i] = pos.mean(0) - neg.mean(0)
    return _unit_rows(D)


def cross_vectors(X: np.ndarray, y: np.ndarray, n_cls: int, seed: int) -> dict:
    """The five [n_cls, d] unit-row matrices compared at n=CROSS_N."""
    dl = C.fit_direction(X, y, "logreg", n_cls=n_cls, seed=seed)
    dd = C.fit_direction(X, y, "dom", n_cls=n_cls, seed=seed)
    return {
        "caa_raw": caa_raw_dirs(X, y, n_cls),
        "logreg_raw": np.stack([C.raw_direction(dl, i) for i in range(n_cls)]),
        "dom_raw": np.stack([C.raw_direction(dd, i) for i in range(n_cls)]),
        "logreg_std": _unit_rows(np.asarray(dl["C"], dtype=np.float64)),
        "dom_std": _unit_rows(np.asarray(dd["C"], dtype=np.float64)),
    }


def cross_pairs(vecs: dict) -> dict:
    """All ordered pairs (i<j) of CROSS_VECS -> per-class cosines."""
    out = {}
    for i, a in enumerate(CROSS_VECS):
        for b in CROSS_VECS[i + 1:]:
            out[f"{a}|{b}"] = [float(vecs[a][k] @ vecs[b][k]) for k in range(len(vecs[a]))]
    return out


# ---------------------------------------------------------------------------
# Ledoit-Wolf lambda (finding 4)
# ---------------------------------------------------------------------------

def lw_lambda_cell(X: np.ndarray, y: np.ndarray, n: int, n_cls: int,
                   seed: int = SPLIT_SEEDS[0]) -> dict | None:
    """The lambda `fit_direction` computes and discards (`sol, _lam = ...`), on the FIRST
    split seed's first half -- i.e. on exactly the subsample a mass_mean_cov split-half fit
    at this n would have used.

    The inputs mirror fit_direction's mass_mean_cov branch: Xs is standardised, Xc is Xs
    centred, and V is the one-vs-rest mean-difference matrix in standardised space, so the
    returned lambda is the number that fit is operating under.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    if len(idx) < 2 * n:
        return None
    h1 = idx[:n]
    Xh, yh = X[h1], y[h1]
    mu, sd = Xh.mean(0), Xh.std(0) + 1e-6
    Xs = ((Xh - mu) / sd).astype(np.float64)
    Xc = Xs - Xs.mean(0)
    dm, keep = np.zeros((n_cls, Xs.shape[1])), []
    for i in range(n_cls):
        pos, neg = Xs[yh == i], Xs[yh != i]
        if len(pos) < 2 or len(neg) < 2:
            continue
        dm[i] = pos.mean(0) - neg.mean(0)
        keep.append(i)
    if not keep:
        return None
    # The solve is not skippable: lw_shrinkage_and_apply has no lambda-only path, and
    # recomputing lambda locally would mean a SECOND implementation of the quantity this
    # cell exists to record. Measured cost is 0.6 s at n=2000, d=5120, so it is not worth
    # the divergence risk. `_sol` is deliberately discarded -- only lambda is the finding.
    _sol, lam = C.lw_shrinkage_and_apply(Xc, dm[keep])
    return {"lambda": float(lam), "n": int(n), "d": int(Xs.shape[1]),
            "n_classes_kept": len(keep), "split_seed": int(seed)}


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------

def _merge_counts(*ds) -> dict:
    out: dict = {}
    for d in ds:
        for k, v in (d or {}).items():
            out[k] = out.get(k, 0) + int(v)
    return out


def _ckpt_warnings(ck, layers) -> dict:
    """Warning counts read back from the checkpoint, so a fully RESUMED run reports the
    counts of the run that actually did the fitting instead of an empty dict."""
    return _merge_counts(*[ck.get(f"warnings/{L}") for L in layers
                           if ck.has(f"warnings/{L}")])


def _agg_pairs(per_seed_pairs: list, degenerate: bool = False) -> dict:
    """[{pair: [per-class cos]}] over draws -> {pair: {mean, ci, per_class, per_seed}}.

    `degenerate` marks the case where the draws are not independent (pool <= cross_n), so
    the CI cannot be quoted as a sampling CI. It is stamped on EVERY pair rather than only
    next to the block, so a number cannot be lifted out of this file without its caveat.
    """
    out = {}
    for pair in per_seed_pairs[0]:
        P = np.array([d[pair] for d in per_seed_pairs], dtype=float)   # [seeds, n_cls]
        m = P.mean(1)
        out[pair] = {"mean": float(m.mean()), "ci": C.ci_of(m),
                     "n_seeds": int(len(P)), "per_class": P.mean(0).tolist(),
                     "per_class_labels": C.EMOTIONS[:P.shape[1]],
                     "per_seed": P.tolist()}
        if degenerate:
            out[pair]["degenerate"] = True
            out[pair]["ci_note"] = ("n_pool <= cross_n, so every draw is a permutation of "
                                    "the WHOLE pool: the draws are identical, the CI has "
                                    "no sampling content and collapses to zero width")
        if pair in PAIRS_OF_INTEREST:
            out[pair]["note"] = PAIRS_OF_INTEREST[pair]
    return out


def run(feats: dict, y: np.ndarray, layers, tag: str, outdir: str, workdir: str,
        seed: int, target_n=TARGET_N, split_seeds=SPLIT_SEEDS, cross_n: int = CROSS_N,
        cross_seeds=CROSS_SEEDS, source: dict | None = None,
        equivalence: dict | None = None) -> dict:
    N = len(y)
    n_cls = int(y.max()) + 1
    # the config key also carries the feature file, so a resumed run cannot silently merge
    # cells computed from a different npz that happens to have the same n
    ck = C.Checkpoint(os.path.join(workdir, f"a2f_cells_{tag}.json"),
                      {"n": N, "layers": list(layers), "targets": list(target_n),
                       "ests": list(SH_ESTIMATORS), "spaces": list(SPACES),
                       "seeds": list(split_seeds), "cross_n": cross_n,
                       "cross_seeds": list(cross_seeds), "c_n": C_N_FORMULA,
                       "feats": (source or {}).get("feats", ""), "base_seed": int(seed)})

    prov = C.Provenance(
        script="src/rev3/a2_followup.py",
        config={"layers": list(layers), "target_n": list(target_n),
                "split_seeds": list(split_seeds), "estimators": list(SH_ESTIMATORS),
                "spaces": list(SPACES), "cross_n": cross_n,
                "cross_seeds": list(cross_seeds), "cross_vectors": list(CROSS_VECS),
                "c_n_formula": C_N_FORMULA, "c_ref": C_REF, "n_ref": N_REF,
                "feats": (source or {}).get("feats", ""), "n_pool": N},
        model_id=(source or {}).get("model", ""), model_revision="",
        seeds={"base": seed, "splits": list(split_seeds), "cross": list(cross_seeds)},
        control_pointers={
            "split_half_statistic": "a2_followup.py::split_half_local -- line-for-line "
                                    "copy of acl_core.py::split_half (same permutation, "
                                    "same disjoint halves, same all-classes-present skip, "
                                    "same cosines) with a pluggable fit; verified equal to "
                                    "acl_core.split_half for method='dom' to 1e-9",
            "logreg_lam": "a2_followup.py::fit_logreg_lam -- " + C_N_FORMULA,
            "caa_raw": "a2_followup.py::caa_raw_dirs -- mu_pos - mu_neg on RAW X, no "
                       "standardisation; acl_core 'dom' + raw_direction is instead "
                       "(mu_pos - mu_neg)/sd^2, which is why A2's logreg-vs-dom cosine is "
                       "not the arXiv 2604.08169 quantity",
            "lw_lambda": "acl_core.py::lw_shrinkage_and_apply returns (sol, lambda); "
                         "fit_direction discards lambda -- recorded here on the first "
                         "split seed's first half, standardised then centred, exactly as "
                         "the mass_mean_cov branch prepares it",
            "cross_estimator_ci": "10 independently seeded n=2000 draws (A2 used one fixed "
                                  "draw, rng seed base+1, at every depth, with no CI)",
        })
    prov.code_sha = C.code_hash(os.path.abspath(__file__), ACL_CORE_PATH)

    out = {"tag": tag, "n_pool": int(N), "n_classes": n_cls, "emotions": C.EMOTIONS,
           "layers": list(layers), "target_n": list(target_n),
           "split_seeds": list(split_seeds), "estimators": list(SH_ESTIMATORS),
           "spaces": list(SPACES), "cross_n": cross_n, "cross_seeds": list(cross_seeds),
           "c_n_formula": C_N_FORMULA, "source": source or {},
           "equivalence_check": equivalence or {},
           "review_findings": {
               "1_fixed_C": "A2's logreg holds C=0.5 while n varies 27x; logreg_lam fixes "
                            "the per-sample penalty instead",
               "2_space": "A2 only ever used space='raw'; both spaces are recorded here",
               "3_caa": "acl_core 'dom' raw direction is (mu+ - mu-)/sd^2, not CAA's raw "
                        "mean difference; both are reported",
               "4_lw_lambda": "lw_shrinkage_and_apply's lambda was discarded; recorded",
               "5_cross_ci": "the cross-estimator subsample was one fixed draw; 10 seeded "
                             "draws with a CI here"},
           "by_layer": {}, "lw_lambda": {}}
    outpath = os.path.join(outdir, f"a2_followup_{tag}.json")

    for L in layers:
        X = feats[L]
        entry = {"n": int(X.shape[0]), "d": int(X.shape[1]), "stability": {},
                 "cross_estimator": {}}
        print(f"\n===== layer {L}   X{tuple(X.shape)} =====", flush=True)
        w0 = dict(_WARN_COUNTS)          # for this layer's share of the warning counts

        # ---- split-half stability: estimator x n x space -----------------------------
        for est in SH_ESTIMATORS:
            fit = make_fit(est)
            row = {}
            for n in target_n:
                if 2 * n > N:
                    continue
                cids = {sp: f"sh/{L}/{est}/{n}/{sp}" for sp in SPACES}
                if all(ck.has(c) for c in cids.values()):
                    row[str(n)] = {sp: ck.get(cids[sp]) for sp in SPACES}
                    continue
                t0 = time.time()
                with _dedupe_warnings():
                    r = split_half_local(X, y, fit, n, seeds=split_seeds, method=est)
                for sp in SPACES:
                    ck.put(cids[sp], r[sp])
                row[str(n)] = {sp: r[sp] for sp in SPACES}
                print(f"    [{est} n={n}] raw {r['raw']['mean']:+.4f}  "
                      f"std {r['std']['mean']:+.4f}  ({time.time() - t0:.1f}s)", flush=True)
            entry["stability"][est] = row
        for sp in SPACES:
            print(f"  stability[{sp}]", flush=True)
            for est in SH_ESTIMATORS:
                row = entry["stability"][est]
                print(f"    {est:>11} " + "  ".join(
                    f"n{n}:{row[str(n)][sp]['mean']:+.3f}" for n in target_n
                    if str(n) in row), flush=True)

        # ---- cross-estimator at n=cross_n, 10 seeded draws ---------------------------
        nb = min(cross_n, N)
        # With a pool this small every draw is a permutation of the WHOLE pool, so the ten
        # "independent" draws are the same rows and the CI collapses to zero width while
        # still reporting n_seeds=10. The block still runs (a small pool is worth a point
        # estimate) but is flagged everywhere the number appears.
        degenerate = N <= cross_n
        if degenerate:
            print(f"  [cross] n_pool={N} <= cross_n={cross_n}: every draw is the whole "
                  f"pool, the draws are NOT independent and the CI is not a sampling CI "
                  f"-- flagged degenerate=True", flush=True)
        per_seed_pairs = []
        for j, cs in enumerate(cross_seeds):
            cid = f"cross/{L}/{cs}"
            if ck.has(cid):
                per_seed_pairs.append(ck.get(cid))
                continue
            dseed = int(seed) + 1000 + int(cs)
            sub = np.random.default_rng(dseed).permutation(N)[:nb]
            assert len(set(y[sub].tolist())) == n_cls, (
                f"cross draw {cs} at layer {L} is missing a class "
                f"({sorted(set(y[sub].tolist()))}) -- not stratified enough")
            with _dedupe_warnings():
                v = cross_vectors(X[sub], y[sub], n_cls, dseed)
            p = cross_pairs(v)
            ck.put(cid, p)
            per_seed_pairs.append(p)
        entry["cross_estimator"] = _agg_pairs(per_seed_pairs, degenerate=degenerate)
        entry["cross_estimator_n"] = int(nb)
        entry["cross_estimator_degenerate"] = bool(degenerate)
        for pair in PAIRS_OF_INTEREST:
            r = entry["cross_estimator"][pair]
            print(f"  cross {pair:>24}  {r['mean']:+.4f} "
                  f"[{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}]  n_draws={r['n_seeds']}",
                  flush=True)

        # ---- Ledoit-Wolf lambda ------------------------------------------------------
        lw = {}
        for n in target_n:
            cid = f"lw_lambda/{L}/{n}"
            if ck.has(cid):
                lw[str(n)] = ck.get(cid)
                continue
            r = lw_lambda_cell(X, y, n, n_cls, seed=split_seeds[0])
            if r is None:
                continue
            ck.put(cid, r)
            lw[str(n)] = r
        out["lw_lambda"][str(L)] = lw
        print("  lw_lambda   " + "  ".join(f"n{n}:{lw[str(n)]['lambda']:.4f}"
                                           for n in target_n if str(n) in lw), flush=True)

        out["by_layer"][str(L)] = entry
        # this layer's share of the warnings goes INTO the checkpoint, merged with whatever
        # an earlier run recorded for it -- otherwise a fully resumed run (which fits
        # nothing and so warns about nothing) would overwrite the counts with {}
        wcid = f"warnings/{L}"
        wdelta = {k: v - w0.get(k, 0) for k, v in _WARN_COUNTS.items()
                  if v - w0.get(k, 0) > 0}
        if wdelta:
            ck.put(wcid, _merge_counts(ck.get(wcid) if ck.has(wcid) else {}, wdelta))
        out["warnings"] = _ckpt_warnings(ck, layers)
        C.write_result(outpath, out, prov)

    # ---- headline ---------------------------------------------------------------------
    hl_layer = 43 if 43 in layers else layers[len(layers) // 2]
    e = out["by_layer"][str(hl_layer)]
    nbig = max([n for n in target_n if 2 * n <= N] or [target_n[0]])

    def _sh(est, sp):
        c = e["stability"].get(est, {}).get(str(nbig))
        return None if c is None else round(c[sp]["mean"], 4)

    out["headline"] = {
        "layer": hl_layer, "n_per_half": nbig,
        "logreg_raw": _sh("logreg", "raw"), "logreg_std": _sh("logreg", "std"),
        "logreg_lam_raw": _sh("logreg_lam", "raw"), "logreg_lam_std": _sh("logreg_lam", "std"),
        "logreg_cv_raw": _sh("logreg_cv", "raw"), "logreg_cv_std": _sh("logreg_cv", "std"),
        "dom_raw": _sh("dom", "raw"), "dom_std": _sh("dom", "std"),
        "caa_vs_logreg_raw": round(e["cross_estimator"]["caa_raw|logreg_raw"]["mean"], 4),
        "logreg_vs_dom_raw": round(e["cross_estimator"]["logreg_raw|dom_raw"]["mean"], 4),
        "logreg_vs_dom_std": round(e["cross_estimator"]["logreg_std|dom_std"]["mean"], 4),
        "caa_vs_dom_raw": round(e["cross_estimator"]["caa_raw|dom_raw"]["mean"], 4),
        "lw_lambda_at_n": {n: out["lw_lambda"][str(hl_layer)][str(n)]["lambda"]
                           for n in map(str, target_n)
                           if n in out["lw_lambda"][str(hl_layer)]},
    }
    print("\n[A2F headline]", json.dumps(out["headline"], indent=2, default=str), flush=True)
    out["warnings"] = _ckpt_warnings(ck, layers)
    if out["warnings"]:
        print("[A2F warnings] " + json.dumps(out["warnings"], indent=2)[:800], flush=True)
    C.write_result(outpath, out, prov)
    return out


# ---------------------------------------------------------------------------
# selftest
# ---------------------------------------------------------------------------

def synth(n: int = 1500, d: int = 200, n_cls: int = 6, seed: int = 0):
    """Planted class means over an anisotropic nuisance background (same construction as
    splithalf_independent.py::selftest, scaled down)."""
    rng = np.random.default_rng(seed)
    V = rng.standard_normal((n_cls, d))
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    Q = rng.standard_normal((d, 10))
    Q /= np.linalg.norm(Q, axis=0, keepdims=True)
    y = np.concatenate([np.full(n // n_cls, i) for i in range(n_cls)])
    y = rng.permutation(np.concatenate([y, rng.integers(0, n_cls, n - len(y))]))
    X = (rng.standard_normal((n, 10)) @ Q.T * 4.0 + rng.standard_normal((n, d))
         + 2.0 * V[y])
    # a per-column scale, so raw and std space are genuinely different
    X *= (1.0 + 3.0 * rng.random(d))
    return X.astype(np.float32), y


def selftest(outdir: str, workdir: str, seed: int = 0) -> int:
    t0 = time.time()
    X, y = synth()
    n_cls = int(y.max()) + 1
    print(f"[selftest] synthetic X{X.shape}, {n_cls} classes, planted means", flush=True)

    print("[selftest] split_half_local vs acl_core.split_half (method='dom')", flush=True)
    gaps = check_dom_equivalence(X, y, n_list=(75, 150, 600), seeds=SPLIT_SEEDS)
    for k, v in gaps.items():
        print(f"    {k:>12}  max|gap| {v:.3e}", flush=True)

    print("[selftest] logreg_lam(n=600) must equal logreg(C=0.5)", flush=True)
    idx = np.random.default_rng(0).permutation(len(y))[:N_REF]
    a = fit_logreg_lam(X[idx], y[idx], n_cls=n_cls, seed=0)
    b = C.fit_direction(X[idx], y[idx], "logreg", n_cls=n_cls, seed=0)
    assert abs(a["C_n"] - C_REF) < 1e-12, f"C_n at n=600 is {a['C_n']}, not {C_REF}"
    gC = float(np.max(np.abs(a["C"] - b["C"])))
    gb = float(np.max(np.abs(a["b"] - b["b"])))
    gv = max(float(np.max(np.abs(C.raw_direction(a, i) - C.raw_direction(b, i))))
             for i in range(n_cls))
    cos = [float(C.raw_direction(a, i) @ C.raw_direction(b, i)) for i in range(n_cls)]
    print(f"    C_n {a['C_n']}  max|dC| {gC:.3e}  max|db| {gb:.3e}  "
          f"max|d raw_dir| {gv:.3e}  min raw cos {min(cos):.12f}", flush=True)
    assert gC <= 1e-9 and gb <= 1e-9, f"logreg_lam != logreg at n=600: {gC}, {gb}"
    # The raw directions are compared ELEMENTWISE, not by cosine: raw_direction divides by
    # (norm + 1e-9), so a vector's cosine with itself is 1 - 2e-9/|d| < 1 by construction.
    # That epsilon artefact is ~1.5e-9 here and would fail a 1e-9 cosine test on two
    # bit-identical fits.
    assert gv <= 1e-9, f"logreg_lam raw direction != logreg at n=600: {gv}"

    print("[selftest] full pipeline, one layer, reduced n grid", flush=True)
    res = run({0: X}, y, [0], "selftest", outdir, workdir, seed,
              target_n=(75, 150, 600), split_seeds=SPLIT_SEEDS, cross_n=600,
              cross_seeds=CROSS_SEEDS, source={"synthetic": True},
              equivalence={"dom_gaps": gaps, "logreg_lam_vs_logreg_at_600":
                           {"max_abs_dC": gC, "max_abs_db": gb,
                            "max_abs_d_raw_dir": gv, "min_raw_cos": min(cos)}})
    sh = res["by_layer"]["0"]["stability"]
    assert abs(sh["logreg_lam"]["600"]["raw"]["mean"]
               - sh["logreg"]["600"]["raw"]["mean"]) <= 1e-9, \
        "logreg_lam and logreg split-half disagree at n=600"
    assert abs(sh["logreg_lam"]["600"]["std"]["mean"]
               - sh["logreg"]["600"]["std"]["mean"]) <= 1e-9
    print(f"\n[selftest] {time.time() - t0:.1f}s", flush=True)
    print("SELFTEST_OK", flush=True)
    return 0


def _ensure_dir(path: str) -> str:
    """Use `path` if we can create it, otherwise a temp dir (so --selftest runs anywhere)."""
    try:
        os.makedirs(path, exist_ok=True)
        return path
    except Exception as ex:
        alt = tempfile.mkdtemp(prefix="a2f_")
        print(f"[a2f] {path} unusable ({ex}); using {alt}", flush=True)
        return alt


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--feats", help="npz written by acl_core.pool_features")
    ap.add_argument("--tag", default="selftest")
    ap.add_argument("--layers", default="16,43,54")
    ap.add_argument("--outdir", default="/marimo/out")
    ap.add_argument("--workdir", default="/marimo/work")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--n-ref", type=int, default=600,
                    help="anchor n for the fixed per-sample penalty: C_n = 0.5 * n_ref / n_fit (default 600)")
    a = ap.parse_args()
    global N_REF, C_N_FORMULA
    N_REF = a.n_ref
    C_N_FORMULA = C_N_FORMULA.replace("0.5 * 600 / n_fit", f"0.5 * {N_REF} / n_fit")
    outdir, workdir = _ensure_dir(a.outdir), _ensure_dir(a.workdir)

    if a.selftest:
        sys.exit(selftest(outdir, workdir, a.seed))
    if not a.feats:
        ap.error("--feats is required unless --selftest")

    z = np.load(a.feats)
    y = np.asarray(z["yp"])
    # pool_features writes n alongside yp/yo; if they disagree the npz was assembled from
    # two different pools and every label is suspect
    if "n" in z.files:
        assert int(z["n"]) == len(y), (
            f"{a.feats} is inconsistent: header n={int(z['n'])} but yp has {len(y)} rows")
    else:
        print(f"[a2f] {a.feats} has no 'n' header; taking n from yp", flush=True)
    layers = [int(s) for s in a.layers.split(",") if s.strip()]
    missing = [L for L in layers if f"L{L}" not in z.files]
    if missing:
        ap.error(f"{a.feats} has no {missing} (keys: {sorted(z.files)})")
    feats = {L: np.asarray(z[f"L{L}"]) for L in layers}
    print(f"[a2f] {a.feats}: n={len(y)} layers={layers} "
          f"d={feats[layers[0]].shape[1]} classes={int(y.max()) + 1}", flush=True)

    # the equivalence witness runs on every invocation, not only under --selftest
    Xs_, ys_ = synth(n=900, d=120)
    gaps = check_dom_equivalence(Xs_, ys_, n_list=(75, 150), seeds=SPLIT_SEEDS)
    print(f"[a2f] split_half_local == acl_core.split_half (dom): "
          f"max gap {max(gaps.values()):.3e}", flush=True)

    run(feats, y, layers, a.tag, outdir, workdir, a.seed,
        source={"feats": os.path.abspath(a.feats), "n_from_npz": int(z["n"])
                if "n" in z.files else len(y)},
        equivalence={"dom_gaps": gaps, "note": "on synthetic data, d=120"})
    print("A2F_DONE", flush=True)


if __name__ == "__main__":
    main()

"""a2_margin_diag.py -- what the fixed-penalty logistic fit is doing as n grows (rev-3 A2 add-on).

Reports 24/25's critic read the DECLINE of the fixed-per-sample-penalty logistic direction's
split-half cosine with n as "the penalty carried the small-n fit; as n grows the likelihood
takes over and the solution moves onto the unstable near-MLE direction". The follow-up files
cannot test that. This script refits the SAME multinomial logistic (same standardisation,
same solver arguments, same splits and seeds as a2_followup.py::split_half_local /
fit_logreg_lam) under three penalty schedules on each half and records, per (layer, n, seed):

  * training accuracy, training log-loss, min multinomial margin, ||W||_F, iterations,
    convergence -- separability and how hard the likelihood is pushing;
  * split-half cosine (raw space, mean over classes) for each schedule -- 'lam' must
    reproduce a2_followup's logreg_lam numbers for the shared seeds (the selftest checks it);
  * within-half cosines between schedules and against difference-of-means: cos(lam, weak),
    cos(lam, fixedC), cos(lam, dom), cos(weak, dom).

If the critic is right: the weak-penalty ('weak', C = 100x the fixed C) split-half cosine is
LOW at every n, cos(lam, weak) RISES with n, and cos(lam, dom) FALLS with n. If instead
cos(lam, weak) is flat, the decline is not the fit drifting toward the MLE.

Schedules: lam  -> C_n = 0.5 * N_REF / n  (per-sample penalty fixed at its N_REF value)
           fixedC -> C = 0.5
           weak -> C = 50 (a proxy for the MLE direction; the true MLE does not exist on
                   separable halves, and lbfgs at C -> inf would just run to max_iter)
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import acl_core as C  # noqa: E402

C_REF, C_WEAK = 0.5, 50.0
N_REF = 600
TARGETS = (75, 150, 300, 600, 1200, 2000)
SEEDS = tuple(range(5))
MAX_ITER = 3000


def fit_multinomial(X, y, c, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = ((X - mu) / sd).astype(np.float64)
    m = LogisticRegression(max_iter=MAX_ITER, C=c, random_state=seed).fit(Xs, y)
    W = np.asarray(m.coef_, dtype=np.float64)
    S = m.decision_function(Xs)
    true = S[np.arange(len(y)), y]
    S2 = S.copy(); S2[np.arange(len(y)), y] = -np.inf
    margin = true - S2.max(1)
    n_iter = int(np.max(m.n_iter_))
    return {"C": W, "sd": sd, "mu": mu, "train_acc": float((m.predict(Xs) == y).mean()),
            "logloss": float(log_loss(y, m.predict_proba(Xs), labels=list(range(W.shape[0])))),
            "min_margin": float(margin.min()), "median_margin": float(np.median(margin)),
            "w_frob": float(np.linalg.norm(W)), "n_iter": n_iter,
            "converged": bool(n_iter < MAX_ITER), "c": float(c)}


def raw_dirs(dec, n_cls):
    return np.stack([C.raw_direction(dec, i) for i in range(n_cls)])


def mean_cos(A, B):
    return float(np.mean(np.sum(A * B, axis=1)))


def run_layer(X, y, n_ref, targets=TARGETS, seeds=SEEDS):
    N, n_cls = len(y), int(y.max()) + 1
    out = {}
    for n in targets:
        if 2 * n > N:
            continue
        cells = []
        for s in seeds:
            idx = np.random.default_rng(s).permutation(N)
            h1, h2 = idx[:n], idx[n:2 * n]
            if len(set(y[h1])) < n_cls or len(set(y[h2])) < n_cls:
                continue
            sched = {"lam": C_REF * n_ref / float(n), "fixedC": C_REF, "weak": C_WEAK}
            fits = {k: [fit_multinomial(X[h], y[h], c, s) for h in (h1, h2)]
                    for k, c in sched.items()}
            dom = [C.fit_direction(X[h], y[h], "dom", seed=s) for h in (h1, h2)]
            dirs = {k: [raw_dirs(f, n_cls) for f in fits[k]] for k in fits}
            ddirs = [raw_dirs(d, n_cls) for d in dom]
            cell = {"seed": int(s), "n": int(n)}
            for k in fits:
                cell[f"sh_{k}"] = mean_cos(dirs[k][0], dirs[k][1])
                for stat in ("train_acc", "logloss", "min_margin", "median_margin", "w_frob",
                             "n_iter", "converged"):
                    cell[f"{k}_{stat}"] = [fits[k][i][stat] for i in (0, 1)]
            cell["sh_dom"] = mean_cos(ddirs[0], ddirs[1])
            for a, b in (("lam", "weak"), ("lam", "fixedC"), ("weak", "fixedC")):
                cell[f"cos_{a}_{b}"] = float(np.mean([mean_cos(dirs[a][i], dirs[b][i])
                                                       for i in (0, 1)]))
            for a in ("lam", "weak", "fixedC"):
                cell[f"cos_{a}_dom"] = float(np.mean([mean_cos(dirs[a][i], ddirs[i])
                                                       for i in (0, 1)]))
            cells.append(cell)
        agg = {"n": int(n), "n_seeds": len(cells), "C_lam": C_REF * n_ref / float(n)}
        for key in cells[0]:
            if key in ("seed", "n"):
                continue
            vals = np.array([np.mean(c[key]) if isinstance(c[key], list) else c[key]
                             for c in cells], dtype=float)
            agg[key] = {"mean": float(vals.mean()), "per_seed": vals.tolist()}
        out[str(n)] = agg
        print(f"[margin] n={n:5d} C_lam={agg['C_lam']:.3f} sh lam/fixedC/weak/dom = "
              f"{agg['sh_lam']['mean']:.3f}/{agg['sh_fixedC']['mean']:.3f}/"
              f"{agg['sh_weak']['mean']:.3f}/{agg['sh_dom']['mean']:.3f} | "
              f"cos(lam,weak) {agg['cos_lam_weak']['mean']:.3f} cos(lam,dom) "
              f"{agg['cos_lam_dom']['mean']:.3f} | lam acc {agg['lam_train_acc']['mean']:.3f} "
              f"margin {agg['lam_min_margin']['mean']:.2f} |W| {agg['lam_w_frob']['mean']:.2f} "
              f"conv {agg['lam_converged']['mean']:.1f}; weak acc "
              f"{agg['weak_train_acc']['mean']:.3f} conv {agg['weak_converged']['mean']:.1f}",
              flush=True)
    return out


def synth(n=1500, d=200, n_cls=6, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, n_cls, n)
    M = rng.standard_normal((n_cls, d)) * 0.8
    X = M[y] + rng.standard_normal((n, d)) * (1.0 + 2.0 * (np.arange(d) < 20))
    return X.astype(np.float32), y


def selftest():
    import a2_followup as F
    X, y = synth()
    F.N_REF = N_REF
    mine = run_layer(X, y, N_REF, targets=(75, 150, 300), seeds=(0, 1, 2))
    for n in (75, 150, 300):
        ref = F.split_half_local(X, y, F.fit_logreg_lam, n, seeds=(0, 1, 2), spaces=("raw",))
        got = np.array(mine[str(n)]["sh_lam"]["per_seed"])
        want = np.array(ref["raw"]["per_seed"]).mean(1) if "per_seed" in ref["raw"] else None
        assert want is not None and np.allclose(got, want, atol=1e-9), (n, got, want)
        assert abs(mine[str(n)]["sh_dom"]["mean"] -
                   C.split_half(X, y, "dom", n, seeds=range(3))["mean"]) < 1e-9
        assert 0.0 <= mine[str(n)]["lam_train_acc"]["mean"] <= 1.0
        assert abs(mine[str(n)]["cos_lam_fixedC"]["mean"] - 1.0) < 1e-9 if n == 600 else True
    print("[selftest] sh_lam reproduces a2_followup.split_half_local(fit_logreg_lam) per seed "
          "to 1e-9 at n = 75, 150, 300; sh_dom reproduces acl_core.split_half('dom')")
    print("SELFTEST_OK", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats"); ap.add_argument("--tag", default="selftest")
    ap.add_argument("--layers", default="43"); ap.add_argument("--n-ref", type=int, default=600)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--targets", default=",".join(map(str, TARGETS)))
    ap.add_argument("--outdir", default="/marimo/results")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    global N_REF
    N_REF = a.n_ref
    z = np.load(a.feats)
    y = np.asarray(z["yp"]).astype(int)
    layers = [int(x) for x in a.layers.split(",")]
    targets = tuple(int(x) for x in a.targets.split(","))
    prov = C.Provenance(script="src/rev3/a2_margin_diag.py",
                        config={"feats": a.feats, "layers": layers, "targets": list(targets),
                                "seeds": list(range(a.seeds)), "n_ref": N_REF, "c_ref": C_REF,
                                "c_weak": C_WEAK, "max_iter": MAX_ITER, "n_pool": int(len(y))},
                        model_id="", model_revision="", seeds={"splits": list(range(a.seeds))},
                        control_pointers={"fit": "a2_margin_diag.py::fit_multinomial mirrors "
                                          "a2_followup.py::fit_logreg_lam (standardise by sd+1e-6, "
                                          "LogisticRegression(max_iter=3000, C, random_state=seed)); "
                                          "selftest asserts equality of sh_lam with "
                                          "a2_followup.split_half_local to 1e-9",
                                          "splits": "np.random.default_rng(s).permutation(N); "
                                                    "idx[:n], idx[n:2n]; skip if a class is missing "
                                                    "(acl_core.split_half's rule)"},
                        code_sha=C.code_hash(__file__, C.__file__))
    out = {"tag": a.tag, "n_pool": int(len(y)), "n_ref": N_REF, "by_layer": {}}
    for L in layers:
        print(f"[margin] layer {L}", flush=True)
        out["by_layer"][str(L)] = run_layer(np.asarray(z[f"L{L}"]), y, N_REF, targets,
                                            tuple(range(a.seeds)))
    C.write_result(os.path.join(a.outdir, f"a2_margin_{a.tag}.json"), out, prov)
    print("MARGIN_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

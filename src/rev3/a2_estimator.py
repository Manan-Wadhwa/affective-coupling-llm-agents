"""A2 — characterise the estimator regime (RESEARCH_PLAN rev 3, §2.3).

Paper A's claim is that the coefficient row of a multinomial logistic regression fit at
n << d does not converge to a stable direction with increasing data, while
difference-of-means does. Rev 2 evidence for that is two numbers per model. This produces
the full grid the plan asks for, from ONE generation pass:

    layer x estimator x n  ->  split-half cosine (10 independent splits, with a CI)
                              held-out 6-way decode accuracy
                              cross-estimator cosine at matched n

plus, per layer, the regime diagnostics that the n/d story alone does not explain
(§2.2 anomaly: the 8B has the better n/d ratio and the WORSE stability):

    n/d, Fisher ratio, covariance participation ratio, entropy effective rank,
    top-1 / top-10 variance fraction, per-class mean-difference SNR, activation norm

It also re-derives the post-fix stability gate that AUDIT finding 5 classifies as
category (b) -- observed in a run log, artifact lost with the lease -- so that number
finally has a committed file behind it.

Both the PRESENT-speaker and OTHER-speaker label sets are fitted, because the
speaker-orthogonality result is the one thing that survived both estimators and A3's
exhibit is the present-vs-other dissociation.

Engages 2604.08169 directly: it reports CAA mean-difference and logistic directions as
near-identical (0.99/0.98 compassion, 0.80 Qwen honesty). `cross_estimator` measures that
same quantity here, so the disagreement is reported in the other paper's own units.

Usage:  python a2_estimator.py --model Qwen/Qwen3.6-27B --tag qwen36-27b --k 160
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import acl_core as C

# Declared BEFORE the run (§6 hyperparameter grid declaration). All of it is reported.
DEPTH_FRACS = (0.25, 0.35, 0.45, 0.55, 0.67, 0.75, 0.85)
TARGET_N = (75, 150, 300, 600, 1200, 2000)
SPLIT_SEEDS = tuple(range(10))
ESTIMATORS = ("logreg", "logreg_c005", "logreg_cv", "ridge", "dom", "dom_norm",
              "pca_diff", "mass_mean_cov", "lda_shrunk")
# logreg_cv is compared on DECODE accuracy but left out of the split-half sweep: an inner
# 3-fold over 4 C values costs 12 fits per call, ~8 s at d=5120, and the sweep would spend
# more time on it than on the other eight estimators together. Its stability is bounded by
# logreg and logreg_c005, which bracket the C grid it searches.
SH_ESTIMATORS = tuple(m for m in ESTIMATORS if m != "logreg_cv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--k", type=int, default=160, help="dialogues per 6x6 cell")
    ap.add_argument("--outdir", default="/marimo/out")
    ap.add_argument("--workdir", default="/marimo/work")
    ap.add_argument("--gen-bs", type=int, default=64)
    ap.add_argument("--pool-bs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True); os.makedirs(a.workdir, exist_ok=True)

    prov = C.Provenance(
        script="src/rev3/a2_estimator.py",
        config={"k": a.k, "depth_fracs": DEPTH_FRACS, "target_n": TARGET_N,
                "split_seeds": list(SPLIT_SEEDS), "estimators": list(ESTIMATORS),
                "gen_bs": a.gen_bs, "max_new": 240, "temp": 0.9, "top_p": 0.95},
        model_id=a.model, model_revision="", seeds={"pool": a.seed, "splits": list(SPLIT_SEEDS)},
        control_pointers={
            "split_half_statistic": "acl_core.py::split_half (fits on two DISJOINT "
                                    "subsamples; space='raw' compares the direction "
                                    "downstream code actually steers with)",
            "pca_diff_pairing": "acl_core.py::fit_direction, method='pca_diff' -- pairs "
                                "positives/negatives on the other-speaker emotion so the "
                                "nuisance cancels inside each difference",
            "decode_intercept": "acl_core.py::fit_direction returns per-class intercept b; "
                                "decode_acc uses it so accuracy compares directions rather "
                                "than one-vs-rest score offsets",
        })

    h = C.load(a.model)
    prov.model_revision = h.revision
    hs_idx = h.hs_indices(DEPTH_FRACS)
    print(f"[A2] {h.n_layers} layers, sweeping hidden_states {hs_idx}, focus {h.focus()}",
          flush=True)

    items = C.generate_pool(h, a.k, os.path.join(a.workdir, f"pool_{a.tag}.jsonl"),
                            seed=a.seed, bs=a.gen_bs)
    feats, yp, yo = C.pool_features(h, items, hs_idx,
                                    cache=os.path.join(a.workdir, f"feats_{a.tag}.npz"),
                                    bs=a.pool_bs)
    N = len(items)
    print(f"[A2] pool n={N}; freeing the model before the CPU-bound half", flush=True)
    n_layers, focus = h.n_layers, h.focus()      # keep what the analysis still needs
    layer_types, revision = h.layer_types, h.revision
    h.model = None; h.layers = None
    try:
        import gc, torch
        gc.collect(); torch.cuda.empty_cache()
        print(f"[A2] cuda allocated now {torch.cuda.memory_allocated()/2**30:.1f} GiB", flush=True)
    except Exception:
        pass

    ck = C.Checkpoint(os.path.join(a.workdir, f"a2_cells_{a.tag}.json"),
                      {"k": a.k, "n": N, "fracs": DEPTH_FRACS, "targets": TARGET_N,
                       "ests": ESTIMATORS, "seeds": SPLIT_SEEDS})

    out = {"model": a.model, "tag": a.tag, "n_pool": N, "n_layers": n_layers,
           "focus": focus, "hs_indices": hs_idx, "emotions": C.EMOTIONS,
           "layer_types": layer_types,
           "by_layer": {}, "notes": {}}

    for L in hs_idx:
        X = feats[L]
        entry = {"depth_frac": round(L / n_layers, 3),
                 "diagnostics_present": C.diagnostics(X, yp),
                 "stability": {}, "decode": {}, "cross_estimator": {}}
        print(f"\n===== hidden_states[{L}]  (depth {entry['depth_frac']}) =====", flush=True)
        d = entry["diagnostics_present"]
        print(f"  n/d {d['n_over_d']:.3f}  fisher {d['fisher_ratio']:.4f}  "
              f"PR {d['participation_ratio']:.1f}  eff_rank {d['effective_rank']:.1f}  "
              f"snr {d['class_snr_mean']:.3f}  |x| {d['act_norm_mean']:.1f}", flush=True)

        # held-out decode accuracy, one 70/30 split, all estimators, present + other
        cut = int(0.7 * N)
        rs = np.random.default_rng(a.seed).permutation(N)
        tr, te = rs[:cut], rs[cut:]
        for lbl, yy in (("present", yp), ("other", yo)):
            for m in ESTIMATORS:
                cid = f"dec/{L}/{lbl}/{m}"
                if ck.has(cid):
                    entry["decode"].setdefault(lbl, {})[m] = ck.get(cid); continue
                # pair on the OTHER label set: yo when fitting present, yp when fitting other.
                # (Pairing on the label being fitted empties every stratum and returns a zero
                # direction -- found by results/reports/17, 2026-09-04.)
                po = (yo if lbl == "present" else yp) if m == "pca_diff" else None
                dec = C.fit_direction(X[tr], yy[tr], m, seed=a.seed,
                                      pair_on=(po[tr] if po is not None else None))
                v = round(C.decode_acc(dec, X[te], yy[te]), 4)
                ck.put(cid, v)
                entry["decode"].setdefault(lbl, {})[m] = v
            print(f"  decode[{lbl:>7}] " +
                  "  ".join(f"{m}:{entry['decode'][lbl][m]:.3f}" for m in ESTIMATORS), flush=True)

        # split-half stability vs n
        for m in SH_ESTIMATORS:
            row = {}
            for n in TARGET_N:
                if 2 * n > N:
                    continue
                cid = f"sh/{L}/{m}/{n}"
                if ck.has(cid):
                    row[str(n)] = ck.get(cid); continue
                r = C.split_half(X, yp, m, n, seeds=SPLIT_SEEDS,
                                 pair_on=(yo if m == "pca_diff" else None))
                r = {k: r[k] for k in ("mean", "ci", "n_seeds", "per_class", "per_seed")}
                ck.put(cid, r); row[str(n)] = r
            entry["stability"][m] = row
            print(f"  stab[{m:>13}] " +
                  "  ".join(f"n{n}:{row[str(n)]['mean']:+.3f}" for n in TARGET_N
                            if str(n) in row), flush=True)

        # cross-estimator agreement at the largest available n (the 2604.08169 quantity)
        nbig = max([n for n in TARGET_N if n <= N] or [N])
        # A SEEDED RANDOM subsample, not the pool head: the pool is stored in (emotion-pair)
        # job order, so the first nbig rows hold only ~3 of the 6 present emotions;
        # fit_direction then allocates y.max()+1 rows and indexing class 3+ raised IndexError
        # (crashed the 2026-09-04 box-2 run after layer 16). n_cls is passed explicitly too.
        sub = np.random.default_rng(a.seed + 1).permutation(N)[:nbig]
        Xb, ypb, yob = X[sub], yp[sub], yo[sub]
        NC = len(C.EMOTIONS)
        base = {}
        for m in ESTIMATORS:
            base[m] = C.fit_direction(Xb, ypb, m, n_cls=NC, seed=a.seed,
                                      pair_on=(yob if m == "pca_diff" else None))
        for m1 in ESTIMATORS:
            entry["cross_estimator"][m1] = {}
            for m2 in ESTIMATORS:
                cs = [float(C.raw_direction(base[m1], i) @ C.raw_direction(base[m2], i))
                      for i in range(len(C.EMOTIONS))]
                entry["cross_estimator"][m1][m2] = round(float(np.mean(cs)), 4)
        entry["cross_estimator_n"] = nbig
        print("  cross-est logreg vs dom: "
              f"{entry['cross_estimator']['logreg']['dom']:+.3f}   "
              f"logreg vs mass_mean_cov: {entry['cross_estimator']['logreg']['mass_mean_cov']:+.3f}",
              flush=True)

        # present/other subspace geometry under each estimator (the survivor result)
        entry["speaker_geometry"] = {}
        for m in ("logreg", "dom", "mass_mean_cov"):
            dp = C.fit_direction(Xb, ypb, m, n_cls=NC, seed=a.seed,
                                 pair_on=(yob if m == "pca_diff" else None))
            do = C.fit_direction(Xb, yob, m, n_cls=NC, seed=a.seed,
                                 pair_on=(ypb if m == "pca_diff" else None))
            cross = [abs(float(C.raw_direction(dp, i) @ C.raw_direction(do, i)))
                     for i in range(len(C.EMOTIONS))]
            within = [abs(float(C.raw_direction(dp, i) @ C.raw_direction(dp, j)))
                      for i in range(len(C.EMOTIONS)) for j in range(len(C.EMOTIONS)) if i != j]
            entry["speaker_geometry"][m] = {
                "present_other_cos": round(float(np.mean(cross)), 4),
                "within_present_cos": round(float(np.mean(within)), 4)}
        out["by_layer"][str(L)] = entry
        C.write_result(os.path.join(a.outdir, f"a2_estimator_{a.tag}.json"), out, prov)

    # focus-layer headline, in the shape the README table needs
    f = out["by_layer"][str(focus)]
    nbig = max([n for n in TARGET_N if 2 * n <= N] or [TARGET_N[0]])
    out["headline"] = {
        "focus_layer": focus, "n_per_half": nbig,
        "logreg_split_half": f["stability"]["logreg"].get(str(nbig), {}).get("mean"),
        "dom_split_half": f["stability"]["dom"].get(str(nbig), {}).get("mean"),
        "logreg_decode_present": f["decode"]["present"]["logreg"],
        "dom_decode_present": f["decode"]["present"]["dom"],
        "logreg_vs_dom_cos": f["cross_estimator"]["logreg"]["dom"],
    }
    print("\n[A2 headline]", json.dumps(out["headline"], indent=2), flush=True)
    C.write_result(os.path.join(a.outdir, f"a2_estimator_{a.tag}.json"), out, prov)
    print("A2_DONE", flush=True)


if __name__ == "__main__":
    main()

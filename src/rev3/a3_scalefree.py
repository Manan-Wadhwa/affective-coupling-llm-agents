#!/usr/bin/env python3
"""A3, re-read in scale-free units (RESEARCH_PLAN rev 3 §2.3 A3, 2026-09-04 amendment).

`a3_dissociation.py` compares B's *present*-e and *other*-e readouts as raw projections onto
each readout estimator's coefficient rows. The rows' norms are not stored, but the present
channel's score SD at alpha=0 is 1.65-2.4x the other channel's under difference-of-means
(median ~1.9; results/reports/18), so "present shift exceeds other shift" on the raw scale
partly measures scale, not direction (results/reports/15).

This script re-reads the per-sample values in `a3_dissociation_<tag>.json` (`rows`) and puts
each channel on its own scale before comparing:

  z0   : divide each channel by its own SD at alpha = 0 (the unsteered cell) -- "how many
         baseline SDs did the readout move per unit dose"
  zall : divide by the channel's pooled SD over all doses. KEPT FOR THE RECORD ONLY: the
         pooled SD is inflated by the dose response itself, so zall penalises whichever
         channel actually moved (results/reports/18). Prefer z0; the critique's Cohen's-d,
         within-scenario-SD and rank re-reads all agree with z0 (dom->dom: 1/6, 0 sig).

Everything else is identical to the driver: scenario-blocked slope CIs (`acl_core.slope_ci`,
block = scenario) and the paired contrast (`acl_core.paired_slope_contrast`), seed 0.
It prints the exhibit under raw / z0 / zall and writes one JSON. It computes counts and
contrasts only; it does not decide what they mean.

Usage: python a3_scalefree.py results/rev3/a3_dissociation_qwen36-27b.json [--json out.json]
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import numpy as np
import acl_core as C


def gather(rows, est_steer, e, key):
    per = {}
    for r in rows:
        if r["steer_est"] == est_steer and r["emotion"] == e:
            per.setdefault(r["alpha"], []).extend(r[key])
    return {a: np.asarray(v, float) for a, v in per.items()}


def blocks(rows, est_steer, e, alpha):
    return np.concatenate([np.asarray(r["scenario"]) for r in rows
                           if r["steer_est"] == est_steer and r["emotion"] == e and r["alpha"] == alpha])


def scale(P, mode):
    if mode == "raw":
        return P
    sd = P[0.0].std() if mode == "z0" else np.concatenate(list(P.values())).std()
    sd = sd if sd > 0 else 1.0
    return {a: v / sd for a, v in P.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result")
    ap.add_argument("--json", dest="out")
    a = ap.parse_args()
    d = json.load(open(a.result))
    rows, doses, configs = d["rows"], d["doses"], [tuple(c) for c in d["configs"]]
    emos = list(dict.fromkeys(r["emotion"] for r in rows))
    out = {"source": a.result, "modes": ["raw", "z0", "zall"], "results": {}}
    print(f"[a3-scalefree] {len(rows)} rows · configs {configs} · doses {doses}")
    for est_steer, est_read in configs:
        key = f"{est_steer}->{est_read}"
        out["results"][key] = {}
        line = []
        for mode in out["modes"]:
            n_gt, n_sig_gt, n_sig_lt, per = 0, 0, 0, {}
            for e in emos:
                P = scale(gather(rows, est_steer, e, f"present_{est_read}"), mode)
                O = scale(gather(rows, est_steer, e, f"other_{est_read}"), mode)
                if any(len(P.get(x, [])) == 0 or len(O.get(x, [])) == 0 for x in doses):
                    continue
                blk = blocks(rows, est_steer, e, doses[0])
                c = C.paired_slope_contrast(doses, P, O)
                per[e] = {"present": C.slope_ci(doses, P, block=blk),
                          "other": C.slope_ci(doses, O, block=blk),
                          "present_minus_other": c}
                n_gt += int(c["diff"] > 0)
                n_sig_gt += int(c["sig"] and c["diff"] > 0)
                n_sig_lt += int(c["sig"] and c["diff"] < 0)
            out["results"][key][mode] = {"present_gt_other": n_gt, "sig_present_gt": n_sig_gt,
                                         "sig_other_gt": n_sig_lt, "of": len(per), "by_emotion": per}
            line.append(f"{mode:>4}: present>other {n_gt}/{len(per)} (sig {n_sig_gt}), other>present sig {n_sig_lt}")
        print(f"\n[{key}]\n   " + "\n   ".join(line))
        for e in emos:
            cells = [out["results"][key][m]["by_emotion"].get(e) for m in out["modes"]]
            if all(cells):
                print(f"   {e:9} " + "  ".join(f"{m}: {c['present_minus_other']['diff']:+7.2f} [{c['present_minus_other']['ci'][0]:+.2f},{c['present_minus_other']['ci'][1]:+.2f}]{'*' if c['present_minus_other']['sig'] else ' '}" for m, c in zip(out["modes"], cells)))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Re-derive B1's summary from its per-cell checkpoint.

Two jobs, both required by RESEARCH_PLAN rev 3 §8 ("no phase gate sits inside an
uncheckpointed run -- gates land at boundaries where partial output is already
interpretable"):

  1. RECOVERY. If the sweep dies with the lease before `b1_e4rerun.py` reaches its
     analysis block, `b1_cells_<tag>.json` still holds every per-sample value that was
     computed. This turns that file into the same summary, so a lost lease costs GPU time
     and not a result.
  2. PARTIAL READ. Run it against an in-progress checkpoint to see the verdict forming,
     with the cells that exist, rather than waiting blind.

It reports how complete the grid is and refuses to state a verdict on a grid that is not
full, because "manipulation check passed for k of 6" means nothing if only two emotions
have been run.

Usage: python b1_analyze.py results/rev3/b1_cells_qwen36-27b.json [--json out.json]
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _c in (os.path.dirname(os.path.abspath(__file__)),
           os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib")):
    if os.path.exists(os.path.join(_c, "acl_core.py")):
        sys.path.insert(0, _c); break
import numpy as np
import acl_core as C

ARMS = ("none", "emo", "rand", "orth", "cross")


def analyze(rows, doses, emos):
    def sel(e, arm, d, key):
        v = [r for r in rows if r["emotion"] == e and r["arm"] == arm and r["alpha"] == d]
        return np.concatenate([np.array(x[key]) for x in v]) if v else np.array([])

    def blocks(e, arm, d):
        v = [r for r in rows if r["emotion"] == e and r["arm"] == arm and r["alpha"] == d]
        return np.concatenate([np.array(x["scenario"]) for x in v]) if v else None

    summary = {}
    for e in emos:
        s = {}
        have_all = True
        for arm in ARMS:
            ps = {d: sel(e, arm, d, "B_present_e") for d in doses}
            if any(len(v) == 0 for v in ps.values()):
                have_all = False
                continue
            os_ = {d: sel(e, arm, d, "B_other_e") for d in doses}
            s[arm] = {"present_slope": C.slope_ci(doses, ps, block=blocks(e, arm, doses[0])),
                      "other_slope": C.slope_ci(doses, os_, block=blocks(e, arm, doses[0])),
                      "mean_by_dose": {str(d): float(ps[d].mean()) for d in doses},
                      "degenerate_by_dose": {
                          str(d): float(np.mean([r["degenerate_frac"] for r in rows
                                                 if r["emotion"] == e and r["arm"] == arm
                                                 and r["alpha"] == d] or [np.nan]))
                          for d in doses}}
        for a, b in (("emo", "none"), ("emo", "rand"), ("emo", "orth"), ("emo", "cross")):
            if a in s and b in s:
                s[f"{a}_vs_{b}"] = C.paired_slope_contrast(
                    doses, {d: sel(e, a, d, "B_present_e") for d in doses},
                    {d: sel(e, b, d, "B_present_e") for d in doses})
        if "none" in s:
            s["mc_steer"] = C.slope_ci(doses, {d: sel(e, "none", d, "A_readout_e")
                                               for d in doses})
        top = doses[-1]
        mc = {}
        for arm in ARMS:
            v = sel(e, arm, top, "ctx_readout_e")
            mc[arm] = {"mean": float(v.mean()) if len(v) else None,
                       "ci": C.ci_of(v) if len(v) else None, "n": int(len(v))}
        if mc["none"]["mean"] is not None and mc["emo"]["mean"] is not None \
                and mc["rand"]["mean"] is not None:
            mc["drop_emo"] = mc["none"]["mean"] - mc["emo"]["mean"]
            mc["drop_rand"] = mc["none"]["mean"] - mc["rand"]["mean"]
            mc["specific_drop"] = mc["drop_emo"] - mc["drop_rand"]
            # the control CIs must not overlap, or "specific" is not established
            mc["separated"] = bool(mc["emo"]["ci"][1] < mc["rand"]["ci"][0])
            mc["passes"] = bool(mc["drop_emo"] > 0 and mc["drop_emo"] > mc["drop_rand"])
        else:
            mc["passes"] = None
        s["mc_ablate"] = mc
        s["_complete"] = have_all
        summary[e] = s
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--json", dest="out")
    a = ap.parse_args()
    blob = json.load(open(a.checkpoint))
    cfg = blob.get("config", {})
    rows = [r for v in blob["cells"].values() for r in v]
    doses = [float(d) for d in cfg.get("doses", sorted({r["alpha"] for r in rows}))]
    all_emos = list(cfg.get("emos", sorted({r["emotion"] for r in rows})))
    have = sorted({r["emotion"] for r in rows})
    reps = int(cfg.get("reps", 0))
    expect = len(all_emos) * len(doses) * max(reps, 1)
    got = len(blob["cells"])
    print(f"[b1-analyze] {got}/{expect} cells ({100*got/max(expect,1):.0f}%) · "
          f"{len(rows)} arm-rows · doses {doses} · emotions present {have}")

    summary = analyze(rows, doses, all_emos)
    print(f"\n{'emotion':>10} {'MC-abl':>7} {'sep':>4} {'drop_emo':>9} {'drop_rand':>10} "
          f"{'emo-vs-rand slope diff':>24} {'sig':>4}")
    for e in all_emos:
        s = summary.get(e, {})
        mc = s.get("mc_ablate", {})
        evr = s.get("emo_vs_rand")
        if mc.get("passes") is None:
            print(f"{e:>10} {'--':>7}"); continue
        d = f"{evr['diff']:+.2f} CI[{evr['ci'][0]:+.2f},{evr['ci'][1]:+.2f}]" if evr else "--"
        print(f"{e:>10} {str(mc['passes']):>7} {str(mc.get('separated')):>4} "
              f"{mc['drop_emo']:>+9.3f} {mc['drop_rand']:>+10.3f} {d:>24} "
              f"{str(evr['sig']) if evr else '--':>4}")

    done = [e for e in all_emos if summary.get(e, {}).get("mc_ablate", {}).get("passes")
            is not None]
    n_pass = sum(1 for e in done if summary[e]["mc_ablate"]["passes"])
    n_block = sum(1 for e in done
                  if summary[e].get("emo_vs_rand", {}).get("sig")
                  and summary[e]["emo_vs_rand"]["diff"] < 0)
    print(f"\nmanipulation check passes {n_pass}/{len(done)} emotions measured; "
          f"emo blocks vs rand in {n_block}/{len(done)}")
    if len(done) < len(all_emos) or got < expect:
        print("PARTIAL GRID -- no verdict. A count out of six means nothing when fewer "
              "than six have been run.")
    else:
        verdict = ("manipulation_check_failed" if n_pass < len(all_emos) / 2 else
                   "blocking" if n_block >= 4 else "not_blocking")
        print(f"VERDICT: {verdict}")
    if a.out:
        json.dump({"summary": summary, "n_cells": got, "n_expected": expect,
                   "doses": doses, "emotions": all_emos}, open(a.out, "w"),
                  indent=2, default=C._jsonable)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

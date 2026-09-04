#!/usr/bin/env python3
"""Re-read the rev-3 arm contrasts with the scenario-blocked, dose-paired bootstrap.

Three blind critiques (results/reports/16, 18, 20) found that the paired contrasts in B1, the
B1 follow-up and A3 were paired across arms but neither scenario-blocked nor dose-paired, and
that some significance calls flip under the correct resampling. `acl_core.paired_slope_contrast`
gained `block=` and `pair_doses=` on 2026-09-05 (commit 26e535d). This script recomputes every
headline contrast both ways from the committed per-sample rows and writes one JSON so the
blocked numbers have a committed source. It computes nothing new about the experiments; the
unit of independence is the scenario (29 per cell), as the drivers' own docstrings state.

Usage: python reblock_contrasts.py [--json results/rev3/reblocked_contrasts.json]
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import numpy as np
import acl_core as C

EMOS = ["desperate", "afraid", "happy", "calm", "sad", "angry"]


def gather(rows, key, sel):
    """{dose: values}, {dose: scenario ids} for rows matching sel(row)."""
    P, B = {}, {}
    for r in rows:
        if sel(r):
            P.setdefault(r["alpha"], []).extend(r[key]); B.setdefault(r["alpha"], []).extend(r["scenario"])
    return {d: np.asarray(v, float) for d, v in P.items()}, {d: np.asarray(v) for d, v in B.items()}


def both(doses, P, O, blk):
    old = C.paired_slope_contrast(doses, P, O)
    new = C.paired_slope_contrast(doses, P, O, block=blk, pair_doses=True)
    return {"unblocked": {k: old[k] for k in ("diff", "ci", "sig")},
            "blocked_paired": {k: new[k] for k in ("diff", "ci", "sig", "n_blocks")},
            "sig_changed": bool(old["sig"] != new["sig"]),
            "detectable_effect_blocked": float((new["ci"][1] - new["ci"][0]) / 2)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--json", default="results/rev3/reblocked_contrasts.json")
    a = ap.parse_args()
    out = {"source_commit_of_core_change": "26e535d", "experiments": {}}
    # ---- B1 (completed run) and its follow-up: B_present_e per arm
    for name, path, pairs in (
        ("b1_e4rerun", "results/rev3/b1_cells_qwen36-27b.json", [("emo", "rand"), ("emo", "none"), ("emo", "orth"), ("emo", "cross")]),
        ("b1_followup", "results/rev3/b1f_cells_qwen36-27b.json", [("emo", "rand"), ("emo", "ceiling"), ("ceiling", "none"), ("text", "none"), ("emo", "none")]),
    ):
        rows = [r for v in json.load(open(path))["cells"].values() for r in v]
        doses = sorted({r["alpha"] for r in rows})
        res = {}
        for e in EMOS:
            res[e] = {}
            for a_, b_ in pairs:
                P, blk = gather(rows, "B_present_e", lambda r: r["emotion"] == e and r["arm"] == a_)
                O, _ = gather(rows, "B_present_e", lambda r: r["emotion"] == e and r["arm"] == b_)
                if not P or not O or any(d not in P or d not in O for d in doses):
                    continue
                res[e][f"{a_}_vs_{b_}"] = both(doses, P, O, blk)
        out["experiments"][name] = {"doses": doses, "by_emotion": res}
    # ---- A3: present minus other per configuration
    d = json.load(open("results/rev3/a3_dissociation_qwen36-27b.json")); rows = d["rows"]; doses = d["doses"]
    res = {}
    for est_steer, est_read in d["configs"]:
        key = f"{est_steer}->{est_read}"; res[key] = {}
        for e in EMOS:
            P, blk = gather(rows, f"present_{est_read}", lambda r: r["emotion"] == e and r["steer_est"] == est_steer)
            O, _ = gather(rows, f"other_{est_read}", lambda r: r["emotion"] == e and r["steer_est"] == est_steer)
            if not P or not O:
                continue
            res[key][e] = both(doses, P, O, blk)
    out["experiments"]["a3_dissociation"] = {"doses": doses, "by_config": res}
    # ---- print
    for name, ex in out["experiments"].items():
        print(f"\n== {name} ==")
        groups = ex.get("by_emotion") or ex.get("by_config")
        for g, per in groups.items():
            for k, v in per.items():
                o, n = v["unblocked"], v["blocked_paired"]
                flag = "  <-- sig changed" if v["sig_changed"] else ""
                print(f"  {g:14} {k:18} unblocked {o['diff']:+7.2f} [{o['ci'][0]:+.1f},{o['ci'][1]:+.1f}]{'*' if o['sig'] else ' '}   blocked+paired {n['diff']:+7.2f} [{n['ci'][0]:+.1f},{n['ci'][1]:+.1f}]{'*' if n['sig'] else ' '}  n_blocks={n['n_blocks']}{flag}")
    # counts summary
    summ = {}
    for name, ex in out["experiments"].items():
        groups = ex.get("by_emotion") or ex.get("by_config")
        for g, per in groups.items():
            for k, v in per.items():
                key = f"{name}/{k}" if "by_emotion" in ex else f"{name}/{g}"
                s = summ.setdefault(key, {"sig_neg_unblocked": 0, "sig_neg_blocked": 0, "sig_pos_unblocked": 0, "sig_pos_blocked": 0, "n": 0})
                s["n"] += 1
                s["sig_neg_unblocked"] += int(v["unblocked"]["sig"] and v["unblocked"]["diff"] < 0); s["sig_neg_blocked"] += int(v["blocked_paired"]["sig"] and v["blocked_paired"]["diff"] < 0)
                s["sig_pos_unblocked"] += int(v["unblocked"]["sig"] and v["unblocked"]["diff"] > 0); s["sig_pos_blocked"] += int(v["blocked_paired"]["sig"] and v["blocked_paired"]["diff"] > 0)
    out["counts"] = summ
    print("\n== significant counts (negative / positive), unblocked -> blocked+paired ==")
    for k, s in summ.items():
        print(f"  {k:36} neg {s['sig_neg_unblocked']}->{s['sig_neg_blocked']}   pos {s['sig_pos_unblocked']}->{s['sig_pos_blocked']}   of {s['n']}")
    json.dump(out, open(a.json, "w"), indent=1); print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()

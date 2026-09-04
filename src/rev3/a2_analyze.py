#!/usr/bin/env python3
"""Read A2's per-cell checkpoint and print what it contains, per layer.

Mirror of `b1_analyze.py` for the estimator battery (RESEARCH_PLAN rev 3 §8: gates land at
boundaries where partial output is already interpretable). `a2_estimator.py` writes every
decode-accuracy and split-half cell to `a2_cells_<tag>.json` the moment it is computed, but
its per-layer diagnostics and cross-estimator cosines live only in the final
`a2_estimator_<tag>.json`. If the lease dies first, this turns the checkpoint into tables.

It prints only values that are literally in the file (means, CIs, seed counts) and says
which of the declared depths are present. It computes nothing new and states no verdict.

Usage: python a2_analyze.py results/rev3/a2_cells_qwen36-27b.json [--json out.json]
"""
import argparse, json, sys


def analyze(blob):
    cfg, cells = blob["config"], blob["cells"]
    ests, targets = cfg["ests"], cfg["targets"]
    layers = sorted({int(k.split("/")[1]) for k in cells})
    out = {"config": cfg, "layers_present": layers, "by_layer": {}}
    for L in layers:
        dec = {lbl: {m: cells[f"dec/{L}/{lbl}/{m}"] for m in ests
                     if f"dec/{L}/{lbl}/{m}" in cells} for lbl in ("present", "other")}
        sh = {}
        for m in ests:
            row = {str(n): cells[f"sh/{L}/{m}/{n}"] for n in targets
                   if f"sh/{L}/{m}/{n}" in cells}
            if row:
                sh[m] = row
        n_dec = sum(len(v) for v in dec.values())
        n_sh = sum(len(v) for v in sh.values())
        out["by_layer"][str(L)] = {"decode": dec, "split_half": sh,
                                   "n_cells": n_dec + n_sh}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--json", dest="out")
    a = ap.parse_args()
    blob = json.load(open(a.checkpoint))
    cfg = blob["config"]
    res = analyze(blob)
    n_layers_planned = len(cfg["fracs"])
    print(f"[a2-analyze] {len(res['layers_present'])}/{n_layers_planned} depths present "
          f"(hidden_states {res['layers_present']}); k={cfg['k']} pool n={cfg['n']} "
          f"seeds={len(cfg['seeds'])}")
    for L, e in res["by_layer"].items():
        print(f"\n== hidden_states[{L}]  ({e['n_cells']} cells)")
        for lbl, row in e["decode"].items():
            if row:
                print(f"  decode[{lbl:>7}] " + "  ".join(f"{m}:{v:.3f}" for m, v in row.items()))
        if e["split_half"]:
            print("  split-half (mean of seeds)" + "".join(f"{n:>8}" for n in cfg["targets"]))
            for m, row in e["split_half"].items():
                print(f"  {m:>13} " + "".join(
                    f"{row[str(n)]['mean']:8.3f}" if str(n) in row else f"{'-':>8}"
                    for n in cfg["targets"]))
    if len(res["layers_present"]) < n_layers_planned:
        print(f"\nPARTIAL: {n_layers_planned - len(res['layers_present'])} of "
              f"{n_layers_planned} declared depths absent. Per-layer diagnostics and "
              f"cross-estimator cosines are not in the checkpoint at all.")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

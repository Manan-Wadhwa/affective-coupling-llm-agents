"""b1e_draws_aggregate.py -- PREREG_B1e addendum 3: the direction-resampling spread.

Reads the ten B1e draws (probe pool re-split with split seeds 1-10; arms none, emo_all,
permdir_all; emotions afraid, sad) plus the original run (split seed 0) as the reference. Per draw
and emotion: none slope, `emo_all - permdir_all` (decision contrast) and `emo_all - none` with
their scenario-blocked, dose-paired CIs and blocked fractions, `permdir_all - none`, the permdir
dose-0 readout shift, the MC-steer gate, the footprint ratio. Across draws: median, min, max of
each blocked fraction and the number of draws whose contrast is significantly negative. Writes a
JSON summary and a LaTeX table. Nothing is recomputed from rows: every number is the driver's own
summary, so a draw's numbers are the ones its result file reports. CPU only.
"""
import argparse, glob, hashlib, json, os
import numpy as np

EMOS = ("afraid", "sad")
CONTRASTS = ("emo_all_vs_permdir_all", "emo_all_vs_none", "permdir_all_vs_none")


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def one(path, S_emos=EMOS):
    d = json.load(open(path)); S = d["summary"]
    ent = {"file": path, "sha256_16": sha(path), "split_seed": d.get("split_seed", d.get("_provenance", {}).get("config", {}).get("split_seed")),
           "verdict": d.get("verdict"), "arms": d.get("arms"), "emotions": d.get("emotions"),
           "footprint_ratio_permdir_over_emo": (d.get("footprint") or {}).get("ratio_permdir_over_emo"),
           "direction_stability": d.get("direction_stability"), "per_emotion": {}}
    for e in S_emos:
        s = S.get(e)
        if not s or s.get("none") is None: ent["per_emotion"][e] = None; continue
        ns = s["none"]["present_slope"]; none_slope = float(ns["slope"]); pe = {"none_slope": none_slope, "none_slope_ci": ns["ci"], "testable": bool(ns["sig"]), "contrasts": {}}
        for cn in CONTRASTS:
            c = s.get(cn)
            if not c: pe["contrasts"][cn] = None; continue
            bf = (s.get("blocked_fraction") or {}).get(cn)
            frac = bf["point"] if bf and bf.get("defined") else (float(-c["diff"] / none_slope) if abs(none_slope) > 1e-9 else None)
            rng_ = bf["range"] if bf and bf.get("defined") else (sorted(-x / none_slope for x in c["ci"]) if abs(none_slope) > 1e-9 else None)
            pe["contrasts"][cn] = {"diff": c["diff"], "ci": c["ci"], "sig": bool(c["sig"]), "sig_negative": bool(c["sig"] and c["diff"] < 0),
                                   "blocked_fraction": frac, "blocked_range": rng_, "n_blocks": c.get("n_blocks")}
        pe["permdir_dose0_shift"] = float(s["permdir_all"]["ctx_readout_by_dose"]["0.0"] - s["none"]["ctx_readout_by_dose"]["0.0"]) if s.get("permdir_all") else None
        pe["mc_steer_sig"] = (s.get("mc_steer") or {}).get("sig"); pe["quality_excluded"] = s.get("_quality_excluded_cells")
        ent["per_emotion"][e] = pe
    return ent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="*", default=None, help="draw result files (default: glob results/rev3/b1e_footprint_qwen36-27b-draw*.json)")
    ap.add_argument("--ref", default="results/rev3/b1e_footprint_qwen36-27b.json")
    ap.add_argument("--out", default="results/rev3/b1e_draws_summary.json")
    ap.add_argument("--tex", default="docs/writeups/table_b1e_draws.tex")
    a = ap.parse_args()
    files = a.files or sorted(glob.glob("results/rev3/b1e_footprint_qwen36-27b-draw*.json"), key=lambda p: int(p.rsplit("draw", 1)[1].split(".")[0]))
    draws = [one(p) for p in files]; ref = one(a.ref) if a.ref and os.path.exists(a.ref) else None
    out = {"n_draws": len(draws), "draws": draws, "reference_split0": ref, "across_draws": {}, "note": "spread across probe-pool splits (PREREG_B1e addendum 3); reference run (split 0) is NOT included in the medians"}
    for e in EMOS:
        out["across_draws"][e] = {}
        for cn in CONTRASTS:
            vals = [dr["per_emotion"][e]["contrasts"][cn] for dr in draws if dr["per_emotion"].get(e) and dr["per_emotion"][e]["contrasts"].get(cn)]
            fr = [v["blocked_fraction"] for v in vals if v["blocked_fraction"] is not None]
            out["across_draws"][e][cn] = {"n": len(vals), "median_blocked_fraction": float(np.median(fr)) if fr else None,
                                          "min": float(min(fr)) if fr else None, "max": float(max(fr)) if fr else None,
                                          "n_sig_negative": int(sum(v["sig_negative"] for v in vals)), "n_sig_positive": int(sum(v["sig"] and v["diff"] > 0 for v in vals)),
                                          "reference_blocked_fraction": (ref["per_emotion"][e]["contrasts"][cn]["blocked_fraction"] if ref and ref["per_emotion"].get(e) and ref["per_emotion"][e]["contrasts"].get(cn) else None)}
        out["across_draws"][e]["n_testable"] = int(sum(1 for dr in draws if dr["per_emotion"].get(e) and dr["per_emotion"][e]["testable"]))
        out["across_draws"][e]["n_mc_steer_sig"] = int(sum(1 for dr in draws if dr["per_emotion"].get(e) and dr["per_emotion"][e]["mc_steer_sig"]))
        out["across_draws"][e]["max_abs_permdir_dose0_shift"] = float(max((abs(dr["per_emotion"][e]["permdir_dose0_shift"]) for dr in draws if dr["per_emotion"].get(e) and dr["per_emotion"][e]["permdir_dose0_shift"] is not None), default=float("nan")))
    out["verdicts"] = {str(dr["split_seed"]): dr["verdict"] for dr in draws}
    fp = [dr["footprint_ratio_permdir_over_emo"] for dr in draws if dr["footprint_ratio_permdir_over_emo"] is not None]
    out["footprint_ratio_permdir_over_emo"] = {"median": float(np.median(fp)) if fp else None, "min": float(min(fp)) if fp else None, "max": float(max(fp)) if fp else None}
    os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(out, open(a.out, "w"), indent=1)
    # console
    print(f"{len(draws)} draws: seeds {[dr['split_seed'] for dr in draws]} verdicts {out['verdicts']}")
    for e in EMOS:
        print(f" {e}")
        for cn in CONTRASTS:
            x = out["across_draws"][e][cn]
            per = [dr["per_emotion"][e]["contrasts"][cn]["blocked_fraction"] for dr in draws if dr["per_emotion"].get(e) and dr["per_emotion"][e]["contrasts"].get(cn)]
            print(f"   {cn:24s} n={x['n']} median {x['median_blocked_fraction']} range [{x['min']}, {x['max']}] sig-neg {x['n_sig_negative']}/{x['n']} (ref {x['reference_blocked_fraction']}) per-draw {[None if v is None else round(v,3) for v in per]}")
    # tex
    def f(v):
        if v is None: return "--"
        s = f"{v:.2f}"; return "0.00" if s == "-0.00" else s
    lab = {"emo_all_vs_permdir_all": r"\texttt{emo\_all} $-$ \texttt{permdir\_all}", "emo_all_vs_none": r"\texttt{emo\_all} $-$ \texttt{none}", "permdir_all_vs_none": r"\texttt{permdir\_all} $-$ \texttt{none}"}
    L = [r"\begin{tabular}{llrrrr}", r"\toprule", r"Emotion & Contrast & Draw 0 & Median & Range & Sig.\ neg. \\", r"\midrule"]
    for e in EMOS:
        for i, cn in enumerate(CONTRASTS):
            x = out["across_draws"][e][cn]
            L.append(f"{e if i == 0 else ''} & {lab[cn]} & {f(x['reference_blocked_fraction'])} & {f(x['median_blocked_fraction'])} & [{f(x['min'])}, {f(x['max'])}] & {x['n_sig_negative']}/{x['n']} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    os.makedirs(os.path.dirname(a.tex), exist_ok=True); open(a.tex, "w").write("\n".join(L) + "\n")
    print("wrote", a.out, a.tex)


if __name__ == "__main__":
    main()

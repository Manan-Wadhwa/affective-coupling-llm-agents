"""paper_appendix_tables.py -- two appendix tables asked for under review, from committed files only.

(1) Reply coherence per arm (AC condition 9): mean words, degenerate and refusal rates, distinct-2 and
    perplexity of B's reply over every arm-row of an arm (six emotions x four doses x three reps), with
    the arm-minus-none difference and a bootstrap CI over the 72 paired (emotion, dose, rep) cells.
(2) Arm-minus-none contrasts in slope units with their scenario-blocked, dose-paired CIs (AC condition 4),
    the numbers behind Table 2's blocked fractions (results/rev3/paper_revision_stats.json; the rank-23
    frame from results/rev3/b1d_randk23_summary.json).
Also prints BH-FDR q-values for B1c's decision contrast under the six-emotion family (AC condition 6)
if per-emotion p-values are present. CPU only. Outputs docs/writeups/table_coherence.tex,
docs/writeups/table_arm_vs_none_slopes.tex, results/rev3/paper_appendix_tables.json.
"""
import json, numpy as np
FILES = {"B1c": "results/rev3/b1c_alllayer_qwen36-27b.json", "B1d": "results/rev3/b1d_subspace_qwen36-27b.json",
         "B1e": "results/rev3/b1e_footprint_qwen36-27b.json", "B1d-23": "results/rev3/b1d_subspace_qwen36-27b-randk23.json"}
NAMES = {("B1c","emo_all"): "emotion direction (hs 13--63)", ("B1c","emo13_42"): "emotion direction (hs 13--42)", ("B1c","rand_all"): "random direction",
         ("B1d","sub_all"): "affect subspace (rank 5)", ("B1d","perm_all"): "permuted-label subspace (rank 5)", ("B1d","randsub_all"): "random 5-frame",
         ("B1d-23","randsub_all"): "random 23-frame, footprint-matched", ("B1e","permdir_all"): "permuted-label direction (rank 1)", ("B1e","pc1_all"): "top principal direction"}
METRICS = [("n_words","words"), ("degenerate_frac","degenerate"), ("refusal_frac","refusal"), ("distinct2","distinct-2"), ("perplexity","perplexity")]
EMOS = ["desperate","afraid","happy","calm","sad","angry"]
rng = np.random.default_rng(0); out = {"coherence": {}, "slopes": {}}
def cells(rows, arm):
    return {(r["emotion"], float(r["alpha"]), r["rep"]): r for r in rows if r["arm"] == arm}
L = [r"\begin{tabular}{llrrrrr}", r"\toprule", r"run & arm & words & degenerate & refusal & distinct-2 & perplexity \\", r"\midrule"]
for run, path in FILES.items():
    d = json.load(open(path)); rows = d["rows"]; none = cells(rows, "none")
    arms = ["none"] + [a for (rn, a) in NAMES if rn == run and any(r["arm"] == a for r in rows)]
    for arm in arms:
        c = cells(rows, arm); keys = sorted(set(c) & set(none)); ent = {}; tex = []
        for m, lab in METRICS:
            v = np.array([c[k][m] for k in keys], float); v0 = np.array([none[k][m] for k in keys], float)
            ok = np.isfinite(v) & np.isfinite(v0); v, v0 = v[ok], v0[ok]; diff = v - v0
            bs = diff[rng.integers(0, len(diff), size=(3000, len(diff)))].mean(1)
            ent[m] = {"mean": float(v.mean()), "diff_vs_none": float(diff.mean()), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "n_cells": int(len(diff))}
            if m in ("degenerate_frac", "refusal_frac"): s = f"{100*v.mean():.1f}\\%"
            elif m == "n_words": s = f"{v.mean():.0f}"
            else: s = f"{v.mean():.2f}"
            if arm != "none" and not (ent[m]["ci"][0] <= 0 <= ent[m]["ci"][1]): s += "$^{\\cdot}$"
            tex.append(s)
        out["coherence"][f"{run}.{arm}"] = ent
        L.append(f"{run} & {'none (baseline)' if arm == 'none' else NAMES[(run, arm)]} & " + " & ".join(tex) + r" \\")
    L.append(r"\midrule")
L = L[:-1] + [r"\bottomrule", r"\end{tabular}"]
open("docs/writeups/table_coherence.tex", "w").write("\n".join(L) + "\n")
# (2) slope-unit contrasts
S = json.load(open("results/rev3/paper_revision_stats.json")); K = json.load(open("results/rev3/b1d_randk23_summary.json"))
ROWS = [("B1c","b1c_27b","emo_all","emotion direction (rank 1, hs 13--63)"), ("B1c","b1c_27b","emo13_42","emotion direction (rank 1, hs 13--42)"), ("B1c","b1c_27b","rand_all","random direction (rank 1)"),
        ("B1d","b1d_27b","sub_all","affect subspace (rank 5)"), ("B1d","b1d_27b","perm_all","permuted-label subspace (rank 5)"), ("B1d","b1d_27b","randsub_all","random 5-frame (rank 5)"),
        ("B1d","randk23",None,"random 23-frame, footprint-matched (rank 23)"), ("B1e","b1e_27b","permdir_all","permuted-label direction (rank 1)"), ("B1e","b1e_27b","pc1_all","top principal direction (rank 1)")]
T = [r"\begin{tabular}{llcccccc}", r"\toprule", r"run & arm $-$ none (slope units) & desperate & afraid & happy & calm$^{u}$ & sad & angry \\", r"\midrule",
     "none-arm slope & & " + " & ".join(f"{S['b1c_27b']['per_emotion'][e]['none_slope']:+.0f} [{S['b1c_27b']['per_emotion'][e]['none_slope_ci'][0]:+.0f}, {S['b1c_27b']['per_emotion'][e]['none_slope_ci'][1]:+.0f}]" for e in EMOS) + r" \\", r"\midrule"]
for run, key, arm, name in ROWS:
    cellv = []
    for e in EMOS:
        if key == "randk23": c = K["per_emotion"][e]["randsub_vs_none"]
        else: c = S[key]["per_emotion"][e]["arm_vs_none"][arm]
        cellv.append(f"{c['diff']:+.0f} [{c['ci'][0]:+.0f}, {c['ci'][1]:+.0f}]"); out["slopes"][f"{run}.{arm or 'randsub23'}.{e}"] = {"diff": c["diff"], "ci": c["ci"]}
    T.append(f"{run} & {name} & " + " & ".join(cellv) + r" \\")
T += [r"\bottomrule", r"\end{tabular}"]
open("docs/writeups/table_arm_vs_none_slopes.tex", "w").write("\n".join(T) + "\n")
# (3) six-emotion family q for B1c decision contrast
b = json.load(open(FILES["B1c"])); pv = (b.get("counts") or {}).get("p_values") or {}
print("B1c counts.p_values:", pv, "| bh_fdr_q:", (b.get("counts") or {}).get("bh_fdr_q"))
json.dump(out, open("results/rev3/paper_appendix_tables.json", "w"), indent=1); print("wrote tables")

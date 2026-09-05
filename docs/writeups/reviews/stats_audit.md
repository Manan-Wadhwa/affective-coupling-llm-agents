# Statistical consistency audit — `review_bundle/submission`

## A. Hard inconsistencies

1. **Pool too small for the split-half grid.** §2 "1,615-item probe pool" (App. A: registered 2,160) vs §3/Repro. "n ∈ {75,…,2000} **per half**", "reachable range (n ≤ 2000)": two *disjoint* halves of 2000 need 4,000 items. Fix: cap n ≤ ⌊pool/2⌋.
2. **Three incompatible ranges for the 27B logistic split-half cosine.** Abstract "0.57"; §2 "0.41–0.57"; §3 "plateaus at 0.55–0.60". §3 also gives tuned-C **0.61** and weak-penalty **0.52**, outside both; 0.41 matches no 27B n=2000 value (Fig 1: 0.566/0.554).
3. **The "inert" random control is neither norm-matched nor inert.** §6 "Norm-matched random and orthogonal directions … inert in every run" vs §5 B1c "unit-norm … **removing twice the norm** … moves happy by **+28%**".
4. **Its footprint has two values:** §5 "twice the norm" vs App. A "B1c's inert random direction (**1.22**)".
5. **A label-free rank-1 direction that blocks.** Abstract/§6/Fig 4 "the fitted rank-1 permuted direction blocks nothing" vs §5 B1e: PC1, label-free and rank 1, "blocks angry by **61%**".
6. **BH-FDR inconsistent.** Fig 4 "the rank-5 permuted subspace … **blocks** afraid, sad and angry" and Table 1 "significant contrasts in bold" vs §5/§7/App. A "**no** per-emotion contrast clears the registered BH-FDR rule (q = 0.051)".
7. **Significance without a q.** §5 B1c "afraid (21%) and sad (33%) are the only **significant** blocks" — no q, though B1d/B1e give them. App. A's FDR "ran over five emotions rather than six" also contradicts Repro.'s "BH over the testable emotions" (=five).
8. **Denominator switch.** §5 "removes a median **9%** of the dose-response" / Abstract "leaves 91%" read as emotion-vs-*none*, but 0.09 is the median of Table 1's **emotion − random** row — a different ratio, since random itself moves happy +28%.
9. **Each block reported twice.** "the emotion direction blocks" afraid = **21%** (§5 B1c, Intro) and **25%** (Abstract, §5 B1e); sad = **33%** and **29%**. The differing baselines are never stated.
10. **"at most a third" (Abstract)** vs §5 B1e sad "29% [7%, **56%**]" and §5 8B "sad blocked by **36%**".
11. **"across a 27-fold increase in n"** (§3) covers both models; on the 8B, n runs 75→1200 = **16-fold**.
12. **Penalty factor.** §3 "**a hundredfold** … (0.52 against 0.58)" vs Fig 2 "a **1000×** weaker penalty", legend "C = 500" (0.5→500).
13. **"a significant dip at 600"** (§3): Figs 1–2 show 0.597→0.577 with fully overlapping bands; no test given.
14. **8B upper window, both ways.** §5 "the upper ablation window changes the readout by **nothing measurable** there" vs App. A "fails … when the upper window changes nothing, **which it does not on this model**".
15. **B1d mixes scales, flips sign.** "afraid (−49.5), sad (−26.7), angry (−44.3)" are unlabelled raw slope units (49.5/212 = 0.23 = Table 1); the next sentence, "less blocking for afraid (**+29.3**), more for angry (**−60.6**)", reverses the sign convention and matches no table cell (0.14, −0.31). "as much as the emotion direction did" fails for angry (0.23 vs 0.09).
16. **B1e footprint.** §5 "0.85× its footprint" vs App. A "0.84× afraid, **0.77×** sad", so the "0.85×–5.8× band" excludes the control at its lower end.
17. **Median CI cannot summarise its data.** §5 "median 9%, CI **[4%, 25%]** over five testable emotions" — sad's estimate is 33%. Fix: label it a scenario bootstrap of the median.

## B. Under-specified

1. "r ≈ 0.85 over five points; blocked-per-removed ≈ 0.27" — n=5, no CI or p (B1e/App. A argue the opposite).
2. Table 1 — no CIs, q-values or n per cell.
3. 8B "the same two emotions to significance (15%, 31%)" — no CI, no q.
4. "extra blocking bounded under 0.08 for four emotions" — four unnamed, no interval.
5. "moves happy by +28% with overlapping CIs" — CIs not shown, contrast not computed.
6. "q = 0.051" — one q for a whole family; per-contrast q's and family size absent.
7. B1d "16–21% of variance", "cosine 0.71–0.81", "a third of its span in common" — no n or CI; overlap undefined.
8. B1e "cosine 0.004 … not stamped in the file"; "footprint (6.5 against 0.5 per position)" — unit undefined.
9. §4 "the manipulation check on A passes for all six emotions" — no statistic or threshold.
10. Repro. "432 or 288 arm-rows at 19–21 s each" vs 6×4×3×29 = 2,088 generations per arm — aggregation unstated.

**Count: 17 hard inconsistencies, 10 under-specifications.**

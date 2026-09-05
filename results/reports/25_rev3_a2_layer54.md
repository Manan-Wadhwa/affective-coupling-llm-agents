# 25 · Rev-3 A2 follow-up — 27B layer 54 at both fixed-penalty anchors (the depth lost twice to leases)

**Files:** `results/rev3/a2_followup_qwen36-27b-l54.json` (anchor 600; box 6, 3108 s, stamped 07:03:48Z) and `a2_followup_qwen36-27b-l54-nref2000.json` (anchor 2000; 2667 s, 07:48:17Z), both code_sha 2a45c175; checkpoints `a2f_cells_qwen36-27b-l54*.json`. Features regenerated from the saved pool on box 6 (kept 4253, as before). · **Companions:** reports 21 and 23 (layer 43 at the same two anchors), 24 (the 8B).

## What the files show
Layer 54, raw space, mean over six classes (std space ≈ +0.02 throughout):

| n per half | logreg | logreg_cv | lam, anchor 600 | lam, anchor 2000 | gap (2000 − 600), of 60 negative | dom |
|---|---|---|---|---|---|---|
| 75 | 0.410 | 0.438 | 0.536 | 0.533 | −0.003, 48 | 0.617 |
| 150 | 0.483 | 0.528 | 0.605 | 0.596 | −0.009, 56 | 0.772 |
| 300 | 0.523 | 0.567 | **0.608** | 0.588 | −0.019, 60 | 0.871 |
| 600 | 0.573 | 0.582 | 0.573 | 0.561 | −0.012, 41 | 0.931 |
| 1200 | 0.564 | 0.584 | 0.566 | 0.550 | −0.016, 38 | 0.965 |
| 2000 | 0.554 | 0.585 | 0.563 | 0.554 | −0.009, 32 | 0.979 |

Fixed-penalty 150 → 2000, paired over the ten shared splits: −0.042 at both anchors (0 of 10 and 1 of 10 positive); 300 → 600: −0.028 (0 of 10) at anchor 2000. `logreg`, `logreg_cv` and `dom` are bit-identical between the two files (anchor-independent, as they should be). `logreg_lam` at n = 2000 equals plain `logreg` at n = 2000 in the anchor-2000 file (difference 0.0). Cross-estimator (ten subsamples of n = 2000, 47% of the pool each): CAA-raw ↔ logreg-raw 0.439 (0.427 at layer 43), CAA ↔ dom 0.768, logreg ↔ dom 0.585.

## What can be inferred
- At layer 54 the fixed-penalty logistic direction peaks at n = 150–300 and declines to n = 2000 at both anchors: the 8B's shape (report 24), milder (−0.04 vs −0.09). Layer 43 (reports 21/23) is the exception with its dip-and-recovery; the "no net rise" statement holds at every depth and both anchors on the 27B. [supported by files]
- The anchor moves the level by ≤ 0.02 and the stronger penalty is again the higher curve, so the stability-maximising penalty remains unbracketed (anchors below 600 untested). [supported by files]
- Plain logistic at fixed C repeats the layer-43 ordering violation here (peak 0.573 at n = 600, 0.554 at 2000). [supported by file]
- The critic's reading of the decline (the penalty carrying the small-n fit under separability) applies here too and is still untested; the per-n margin/training-loss diagnostic is the next step and is queued for box 6. [needs: that diagnostic]

## Defects
As in reports 23/24: empty `model`/`model_revision`; `c_n_formula` prose stale in the anchor-2000 file; `cross_n` = 2000 on a 4253 pool (47% overlap per draw, not flagged); the box's `acl_core` copy predates the rank-k patch (immaterial here).

## Status
Layer 54 complete at both anchors. Closes the "layer 54 lost" items in RESEARCH_PLAN §12/§13.4.

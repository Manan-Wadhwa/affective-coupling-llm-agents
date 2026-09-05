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
- At layer 54 the fixed-penalty logistic direction peaks at n = 150–300, drops once between 300 and 600 (−0.035, 0 of 10 positive) and is flat after that (600 → 1200 −0.007, 1200 → 2000 −0.003, neither significant), at both anchors. That is not the 8B's shape (significant at every step from 300); layer 43 dips at the same place and differs only in its tail (+0.006 vs −0.010). Layer 54 does sit below layer 43 over 300 → 2000 (−0.029, 10 of 10). And "no net rise" is not universal: layer 16 in the anchor-600 file rises +0.024 over 300 → 2000 (10 of 10). [supported by files; corrected after the critique]
- The anchor moves the level by ≤ 0.02 and the stronger penalty is again the higher curve, so the stability-maximising penalty remains unbracketed (anchors below 600 untested). [supported by files]
- Plain logistic at fixed C repeats the layer-43 ordering violation here (peak 0.573 at n = 600, 0.554 at 2000). [supported by file]
- The critic's reading of the decline (the penalty carrying the small-n fit under separability) applies here too and is still untested; the per-n margin/training-loss diagnostic is the next step and is queued for box 6. [needs: that diagnostic]

## Defects
The layer-43 and layer-54 files come from three driver builds (ae3621bb, ccd27d64, 2a45c175); the shared cells (plain logreg, tuned C, dom) agree bit for bit across them, which is the only cross-build check. As in reports 23/24: empty `model`/`model_revision`; `c_n_formula` prose stale in the anchor-2000 file; `cross_n` = 2000 on a 4253 pool (47% overlap per draw, not flagged); the box's `acl_core` copy predates the rank-k patch (immaterial here).

## Status
Layer 54 complete at both anchors. Closes the "layer 54 lost" items in RESEARCH_PLAN §12/§13.4.


### Delta critique — 27B layer-54, both anchors (blind: the four follow-up JSONs and the driver)

**C1 ✓** 0.5358/0.6050/0.6077/0.5730/0.5664/0.5634; 150→2000 −0.0416, 0/10; plain logreg 0.5730→0.5541; logreg_cv 0.582–0.585; dom 0.7722→0.9789.

**C2 ✓** 0.5332/0.5957/0.5883/0.5606/0.5500/0.5541; 150→2000 −0.0416, 1/10; gaps −0.0026…−0.0194 with 48/56/60/41/38/32 of 60 negative; `logreg_lam@2000 == logreg@2000`, and `logreg`/`logreg_cv`/`dom` across anchors, all to max |diff| = **0.0**.

**C3 — not fair.** Step by step at L54 (anchor 600) only **300→600 is significant** (−0.0347, t −5.77, 0/10); 600→1200 is −0.0066 (t −0.93, 2/10) and 1200→2000 −0.0030 (t −1.04, 5/10). That is a peak, one drop, then flat — not the 8B, where every step from 300 was significant (t −3.8 to −7.8). L43 dips at the *same* place (300→600, −0.0212); the only difference is the tail (+0.0058 vs −0.0096). L54 is genuinely below L43 (slope difference 300→2000 −0.0290, t −13.54, 10/10), so the layers do differ — but "L54 is the 8B's shape, L43 the odd one out" mis-describes both. And **"no net rise at every depth" is false**: layer 16 in the anchor-600 file rises +0.0242 over 300→2000 (t +3.80, 10/10).

**C4 ✓** 0.4391 [0.4312, 0.4472] vs 0.4268 at L43; CAA↔dom 0.7684 vs 0.7219; logreg↔dom 0.5847.

**(3) New at L54.** Ledoit–Wolf λ is uniformly below L43's (0.566→0.042 vs 0.637→0.057): shrinkage weakens with depth as well as n. All 6 per-class cosines fall 300→2000 (−0.023 to −0.074) against 5/6 at L43, so no single emotion drives it. std tracks raw at +0.02–0.03 throughout, so this is not the `/sd` conversion. CAA↔dom rises with depth (0.722→0.768) — the sd² distortion is milder deeper.

**(4) Defects.** `model`/`model_revision` empty again; `code_sha` is `2a45c175…` for both L54 files — same build as the 8B run, different from both L43 files (`ae3621bb`, `ccd27d64`), so L43-vs-L54 spans three driver versions. `cross_n` = 2000 on a 4253 pool (47 % pairwise overlap), no degenerate flag; the N_REF=2000 file's prose still asserts "C_600 == 0.5 exactly"; `equivalence_check` still synthetic at d=120.

### Reconciliation (author)

- **C3 rejected, accepted in full:** the "8B's shape, milder" sentence and "no net rise at every depth" are withdrawn everywhere (registry `a2f54.*`, README, RESULTS_LOG, plan). The corrected statement is one significant drop at 300 → 600 then flat, layer 43 dipping at the same place, layer 54 below layer 43, and layer 16 rising.
- **C1, C2, C4 verified** to the digit, including the bit-identity of the anchor-independent cells.
- **New at layer 54** (all six per-class cosines fall, Ledoit-Wolf λ lower with depth, CAA ↔ dom higher with depth) added to the report body as the critic's findings.
- The three-build provenance note is added under Defects.

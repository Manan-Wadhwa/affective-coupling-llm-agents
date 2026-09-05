# 23 · Rev-3 A2 follow-up — second fixed-penalty anchor (N_REF = 2000), layer 43; layer 54 partial

**Files:** `results/rev3/a2_followup_qwen36-27b-nref2000.json` (layer 43 complete, provenance-stamped 20:53Z; code_sha ccd27d64, acl_core 23c758a9), `a2f_cells_qwen36-27b-nref2000.json` (checkpoint: 22 of 48 layer-54 cells) · **Script:** `src/rev3/a2_followup.py --n-ref 2000 --layers 43,54` on box 4 (`sb-ea69c0d19be26d97`), CPU · **Companion:** report 21 (anchor 600, `a2_followup_qwen36-27b.json`), which shares the ten split seeds.

**Lost.** Layer 54 stopped being pulled at 22:24Z when the operator's laptop suspended; the box's lease ended before it woke (05:51Z). The two queued sweeps (layer 54 at anchor 600; the 8B fixed-penalty arm on layers 14, 21, 27) were never pulled and are lost. Whether they ran on the box is unknown.

## What was run
Same battery as report 21 with the per-sample penalty held at its n = 2000 value: `C_n = 0.5 · 2000 / n`, so the per-sample weight is 1/2000 at every n (the 600 anchor holds 1/600). `logreg_lam` at n = 2000 is therefore the same fit as plain `logreg` at n = 2000 (0.5664 in both).

## What the file shows
Layer 43, split-half cosine of the direction (10 disjoint splits; mean over six classes):

| n per half | logreg | logreg_cv | **logreg_lam, anchor 2000** | logreg_lam, anchor 600 (report 21) | paired Δ (2000 − 600), of 60 negative | dom |
|---|---|---|---|---|---|---|
| 150 | 0.471 | 0.506 | **0.574** [0.562, 0.586] | 0.581 | −0.007, 55/60 | 0.732 |
| 300 | 0.525 | 0.559 | **0.578** [0.565, 0.589] | 0.596 | −0.019, 60/60 | 0.842 |
| 600 | 0.575 | 0.578 | **0.554** [0.544, 0.565] | 0.575 | −0.021, 59/60 | 0.915 |
| 1200 | 0.568 | 0.601 | **0.556** [0.552, 0.561] | 0.578 | −0.022, 51/60 | 0.955 |
| 2000 | 0.566 | 0.608 | **0.566** [0.561, 0.572] | 0.581 | −0.015, 36/60 | 0.974 |

Standardised space: anchor 2000 = 0.599, 0.601, 0.576, 0.577, 0.588; anchor 600 = 0.606, 0.619, 0.597, 0.599, 0.603. Cross-estimator CAA-raw ↔ logreg-raw at n = 2000: 0.427 [0.421, 0.431], identical to report 21 (same cells).

Layer 54, partial (checkpoint, no provenance stamp): plain `logreg` raw 0.410, 0.484, 0.523, 0.573, 0.564, 0.554 at n = 75…2000 (peak at 600, then declining, the same ordering violation as layer 43); `logreg_cv` 0.438, 0.528, 0.567, 0.582, 0.584 up to n = 1200. No `logreg_lam` or `dom` cells reached layer 54.

## What can be inferred
- With the per-sample penalty fixed at a second, weaker value the logistic direction again does not become more reproducible with n: range 0.024 over n = 150…2000, with the same dip at n = 600 the report-21 critic flagged. [supported by file]
- The plateau level is mildly anchor-dependent: the weaker penalty sits 0.007–0.022 lower at every n (paired over shared splits), so "≈ 0.58" in report 21 is an anchor-specific number; "0.55–0.60, never approaching difference-of-means" is the anchor-robust statement. [supported by both files]
- Layer 54 plain-logreg replicates the peak-then-decline shape from a second depth, on 22 cells and without provenance. [partial; descriptive only]
- The 8B fixed-penalty arm remains untested. [needs: a rerun]

## Defects to fix before quoting
`model` and `model_revision` are empty strings in the file (the driver reads them from the features' `source`, which carries none); the recorded `c_n_formula` sentence has the right formula (`0.5 * 2000 / n_fit`) but stale prose ("held at its n=600 value … C_600 == 0.5 exactly"). Both are in the registry notes.

## Status
Layer 43 complete; layer 54 partial; 8B not run. Feeds RESEARCH_PLAN §12 (to-do "second fixed-λ anchor" done; "8B fixed-λ arm" and "layer 54" reopened).

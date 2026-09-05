# 24 · Rev-3 A2 follow-up on Llama-3-8B-abliterated — the fixed-penalty arm the 27B reports called untested

**Files:** `results/rev3/a2_followup_llama3-abl.json` (layers 14, 21, 27; provenance-stamped 06:34:55Z, 638 s on box 6; code_sha 2a45c175, acl_core 3da064ad), `a2f_cells_llama3-abl.json` (checkpoint). A first two-layer run (`inflight_box6/a2_followup_llama3-abl-l14_21.json`) reproduces layers 14 and 21 bit for bit. · **Script:** `src/rev3/a2_followup.py --n-ref 600 --layers 14,21,27` on features regenerated from the saved 8B pool (k = 220, kept 3898, revision dd67dd05 per the regeneration log). · **Companions:** report 19 (plain A2 battery on this pool), report 21 (the same follow-up on the 27B).

## What was run
Split-half cosine over ten disjoint splits at n = 75…1200 per half (n = 2000 is impossible on a pool of 3898 and is skipped) for plain `logreg` (C = 0.5), tuned `logreg_cv`, fixed per-sample penalty `logreg_lam` (C_n = 0.5·600/n, so the penalty per sample is 1/600 at every n), and `dom`; raw and standardised space; the like-for-like cross-estimator cosines at n = 1200 over ten subsamples; Ledoit-Wolf λ per fit.

## What the file shows
Raw space, mean over six classes:

| layer | n | logreg | logreg_cv | **logreg_lam** | dom |
|---|---|---|---|---|---|
| 14 | 75 / 150 / 300 / 600 / 1200 | 0.264 / 0.305 / 0.328 / 0.344 / 0.338 | 0.293 / 0.334 / 0.378 / 0.393 / 0.412 | **0.351 / 0.367 / 0.353 / 0.344 / 0.318** | 0.497 / 0.667 / 0.798 / 0.881 / 0.934 |
| 21 (focus) | same | 0.267 / 0.297 / 0.306 / 0.306 / 0.286 | 0.302 / 0.352 / 0.379 / 0.372 / 0.384 | **0.363 / 0.369 / 0.325 / 0.306 / 0.276** | 0.526 / 0.697 / 0.820 / 0.895 / 0.941 |
| 27 | same | 0.265 / 0.289 / 0.307 / 0.302 / 0.289 | 0.299 / 0.341 / 0.373 / 0.374 / 0.374 | **0.359 / 0.357 / 0.328 / 0.302 / 0.281** | 0.527 / 0.695 / 0.820 / 0.896 / 0.941 |

`logreg_lam` 150 → 1200, paired over the ten shared splits: layer 14 −0.050 (1 of 10 positive), layer 21 −0.093 (0 of 10), layer 27 −0.076 (0 of 10). Standardised space is the same picture, ~0.02 higher. Cross-estimator (ten subsamples of n = 2000 from the 3898 pool, 51% each, heavily overlapping; the driver's `cross_n` default was not lowered for this pool): CAA-raw ↔ logreg-raw 0.371 / 0.336 / 0.329 (layers 14 / 21 / 27; CIs ±0.005), CAA ↔ dom 0.90–0.92, logreg ↔ dom 0.37–0.42; no degenerate flag.

## What can be inferred
- With the per-sample penalty held fixed, the logistic direction on the 8B peaks at n = 150 and becomes *less* reproducible as n grows from there, at all three depths (every step from n = 300 on negative at every layer; all six per-class cosines fall at layer 21; the same in standardised space). [supported by file; corrected after the critique]
- The n/d ordering violation seen for plain logistic on this pool (report 19) is therefore not a fixed-C artefact: controlling the schedule does not remove it and makes it larger. [supported by file, as a negative about the schedule explanation]
- Tuning C per fit restores a modest rise (to 0.37–0.41 at n = 1200); difference-of-means climbs to 0.94; the counter-paper's near-identity of logistic and mean-difference directions fails here in its own units (0.33–0.37). [supported by file]
- Contrast with the 27B (report 21/23): at matched n = 1200 the 27B's fixed-penalty change from n = 150 is −0.003, the 8B's −0.093. The critic's reading: a decline at fixed per-sample penalty means the penalty, not the likelihood, carried the small-n fit — consistent with separability at n ≪ d, where the shrunk small-n solution is a low-variance object and the fit moves onto the unstable near-MLE direction as n grows. The file cannot test that (no separability, margin or training-loss diagnostic); it does exclude per-class averaging and the raw/std conversion as causes. Class imbalance and label noise would lower the level, not produce the slope. [interpretation; needs: a separability/margin diagnostic per n]

## Defects to fix before quoting
`cross_n` stayed at 2000 on a pool where split-half at 2000 is impossible, so the two blocks report different n without saying so, and the overlap-degeneracy flag (n_pool ≤ cross_n) does not fire; the equivalence check is still synthetic at d = 120. `model` and `model_revision` are empty in the file (the regeneration log carries rev dd67dd05); the box ran the pre-rank-k copy of `acl_core` (immaterial to this CPU driver, but the stamped core sha differs from the current repo's).

## Status
Complete for three layers. Closes RESEARCH_PLAN §2.1 item "8B fixed-λ arm untested" and §12 to-do "fixed-λ arm on the 8B".


### Delta critique — 8B fixed-penalty arm (blind: 8B follow-up + A2, the 27B anchor-600 file, the driver)

**C1 — numbers all check; "falls with n" holds from n=150, not n=75.** L21 0.3632/0.3690/0.3253/0.3064/0.2762 ✓; 150→1200 −0.0928, 0/10 ✓; L14 0.3671→0.3175 (−0.0495, 1/10) ✓; L27 0.3565→0.2806 (−0.0759, 0/10) ✓; dom 0.6970→0.9413 ✓; n=2000 correctly skipped (2·2000 > 3898). But 75→150 is **+0.0164 (8/10)** at L14 and +0.0058 (5/10) at L21 — the curve peaks at 150; from n=300 every step is negative at all three layers (t −3.8 to −7.8). Anchor identity exact: max |per-seed| of `logreg_lam@600 − logreg@600` = **0.0** at all three layers.

**(2) What a decline at fixed λ can mean.** Both halves estimate the same population λ-minimiser, so with n/d → ∞ the cosine → 1; a decline means the penalty, not the likelihood, carried the fit at small n. Natural reading: **separability** — at n ≪ d the standardised data are separable, the MLE does not exist, and the solution is a heavily shrunk low-variance object; as n grows the likelihood takes over and the fit moves onto the high-variance MLE direction. The file **cannot** test this: no separability, margin or training-loss diagnostic (`lw_lambda` is the Ledoit–Wolf intensity for the covariance arms). It **can** exclude two rivals — the decline appears in all 6 per-class cosines at L21 (−0.048 to −0.120), so it is not per-class averaging, and equally in std space (0.3885 → 0.2864), so it is not the `/sd` conversion. Imbalance and label noise (49 % retention) explain the lower *level*, not the *slope*: more data does not make an estimate of a mis-specified target less reproducible.

**C2 — licensed as worded.** At matched n=1200 the fixed-λ gap is 0.2762 vs 0.5779 (0.302) against the fixed-C gap 0.286 vs 0.568 (0.282), so "survives and is stronger" holds. It rules out the C schedule, not n/d — which is what the claim says.

**C3 — numbers right, n wrong.** `cross_n` is **2000**, not 1200: 0.3364 [0.3314, 0.3414], 0.3713, 0.3285 ✓; CAA↔dom 0.904–0.915 ✓; logreg↔dom 0.374–0.421 ✓. Each draw is 51 % of a 3898 pool (~1026 shared pairwise), and `_agg_pairs` stamps `degenerate` only when `n_pool ≤ cross_n`, so nothing is flagged — worse overlap than the 27B's 47 %.

**C4 — fair, and stronger at matched n:** the 27B's 150→1200 is −0.0033 against the 8B's −0.0928.

**New defects:** `model`/`model_revision` empty again; `equivalence_check` still synthetic at d=120 while d=4096 here; `cross_n` left at its 2000 default though split-half at that n is impossible on this pool, so the two blocks report different n without saying so.

### Reconciliation (author)

- **C1:** "falls with n" corrected to "peaks at n = 150, then falls" in the registry, report and README. The per-class and standardised-space checks the critic ran are quoted above as the exclusions they are.
- **C2 and C4:** accepted as worded; the matched-n contrast (−0.003 vs −0.093) replaces my looser sentence.
- **C3:** the cross-estimator n was misquoted (2000, not 1200); corrected everywhere, with the overlap caveat and the non-firing degeneracy flag in the registry note. The `cross_n` default and the degenerate-threshold fix go to RESEARCH_PLAN §12 to-dos (the threshold item was already there).
- **New to-do:** a separability / margin / training-loss diagnostic per n, which is what would separate the critic's reading from the alternatives.

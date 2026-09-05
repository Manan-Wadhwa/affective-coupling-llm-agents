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

`logreg_lam` 150 → 1200, paired over the ten shared splits: layer 14 −0.050 (1 of 10 positive), layer 21 −0.093 (0 of 10), layer 27 −0.076 (0 of 10). Standardised space is the same picture, ~0.02 higher. Cross-estimator at n = 1200: CAA-raw ↔ logreg-raw 0.371 / 0.336 / 0.329 (layers 14 / 21 / 27; CIs ±0.005), CAA ↔ dom 0.90–0.92, logreg ↔ dom 0.37–0.42; no degenerate flag.

## What can be inferred
- With the per-sample penalty held fixed, the logistic direction on the 8B becomes *less* reproducible as n grows, at all three depths, monotonically from n = 150. [supported by file]
- The n/d ordering violation seen for plain logistic on this pool (report 19) is therefore not a fixed-C artefact: controlling the schedule does not remove it and makes it larger. [supported by file, as a negative about the schedule explanation]
- Tuning C per fit restores a modest rise (to 0.37–0.41 at n = 1200); difference-of-means climbs to 0.94; the counter-paper's near-identity of logistic and mean-difference directions fails here in its own units (0.33–0.37). [supported by file]
- Contrast with the 27B (report 21/23): there the fixed-penalty curve neither rises nor falls net; here it falls. What differs between the pools (n/d, class balance, label noise from the abliterated model's outputs) is not separated by this design. [needs: the critic's reading; a per-class breakdown]

## Defects to fix before quoting
`model` and `model_revision` are empty in the file (the regeneration log carries rev dd67dd05); the box ran the pre-rank-k copy of `acl_core` (immaterial to this CPU driver, but the stamped core sha differs from the current repo's).

## Status
Complete for three layers. Closes RESEARCH_PLAN §2.1 item "8B fixed-λ arm untested" and §12 to-do "fixed-λ arm on the 8B".

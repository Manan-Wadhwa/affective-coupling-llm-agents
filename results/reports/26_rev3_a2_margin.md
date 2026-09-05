# 26 · Rev-3 A2 margin diagnostic — what the logistic fit is doing while its direction fails to converge

**Files:** `results/rev3/a2_margin_qwen36-27b-l43.json` (58 s), `a2_margin_qwen36-27b-l54.json` (515 s), `a2_margin_llama3-abl-l21.json` — box 6, features regenerated from the saved pools (27B kept 4253; 8B kept 3898), code_sha in each file. · **Script:** `src/rev3/a2_margin_diag.py` (selftest: its fixed-penalty split-half cosine reproduces `a2_followup.split_half_local(fit_logreg_lam)` per seed to 1e-9; its dom split-half reproduces `acl_core.split_half`). · **Why:** the critics of reports 24 and 25 read the fixed-penalty decline as "the penalty carried the small-n fit; the likelihood takes over as n grows and moves the solution onto the unstable near-MLE direction", and said the follow-up files could not test it. This can.

## What was run
For each layer, n ∈ {75, 150, 300, 600, 1200, 2000} per half and the first five of the follow-up's ten split seeds: the same multinomial logistic (standardise by sd + 1e-6, lbfgs, max_iter 3000, random_state = seed) on each half under three penalties — `lam` (C_n = 0.5·600/n, the follow-up's fixed per-sample schedule), `fixedC` (C = 0.5) and `weak` (C = 50, a hundred times weaker) — plus difference-of-means on the same halves. Recorded per fit: training accuracy, training log-loss, minimum and median multinomial margin, ‖W‖_F, iterations, convergence; per pair of halves: split-half cosine (raw space, mean over classes) for each estimator; within a half: cosines between the three logistic solutions and against difference-of-means.

## What the files show (27B)

| layer | n | train acc (lam / weak) | log-loss (lam) | min margin | ‖W‖ | iters (lam / weak) | split-half: lam / fixedC / weak / dom | cos(lam, weak) | cos(lam, dom) |
|---|---|---|---|---|---|---|---|---|---|
| 43 | 75 | 1.000 / 1.000 | 0.000 | 6.73 | 1.07 | 8 / 9 | 0.515 / 0.407 / 0.509 / 0.572 | 1.000 | 0.974 |
| 43 | 150 | 1.000 / 1.000 | 0.000 | 6.37 | 1.18 | 9 / 9 | 0.585 / 0.475 / 0.576 / 0.732 | 1.000 | 0.929 |
| 43 | 300 | 1.000 / 1.000 | 0.001 | 5.76 | 1.34 | 10 / 10 | 0.605 / 0.534 / 0.574 / 0.842 | 0.998 | 0.854 |
| 43 | 600 | 1.000 / 1.000 | 0.001 | 4.94 | 1.54 | 12 / 12 | 0.576 / 0.576 / 0.525 / 0.917 | 0.996 | 0.753 |
| 43 | 1200 | 1.000 / 1.000 | 0.001 | 4.00 | 1.75 | 18 / 13 | 0.573 / 0.566 / 0.531 / 0.954 | 0.987 | 0.677 |
| 43 | 2000 | 1.000 / 1.000 | 0.002 | 3.42 | 1.94 | 21 / 16 | 0.581 / 0.565 / 0.521 / 0.974 | 0.981 | 0.634 |
| 54 | 75 → 2000 | 1.000 throughout | ≤ 0.002 | 6.79 → 3.42 | 1.08 → 1.94 | ≤ 25 | 0.547 → 0.563 / 0.421 → 0.553 / 0.541 → 0.505 / 0.622 → 0.979 | 1.000 → 0.975 | 0.967 → 0.612 |

The `lam` split-half column reproduces the follow-up's `logreg_lam` numbers on the shared seeds (0.581 at layer 43, n = 2000, in both).

## What the 8B shows
Layer 21 (648 s; n = 2000 impossible on this pool). Same columns as above:

| layer | n | train acc (lam / weak) | log-loss (lam) | min margin | ‖W‖ | iters (lam / weak) | split-half: lam / fixedC / weak / dom | cos(lam, weak) | cos(lam, dom) |
|---|---|---|---|---|---|---|---|---|---|
| 21 (8B) | 75 | 1.000 / 1.000 | 0.000 | 6.82 | 1.21 | 11 / 10 | 0.365 / 0.278 / 0.367 / 0.545 | 0.998 | 0.913 |
| 21 (8B) | 150 | 1.000 / 1.000 | 0.001 | 6.21 | 1.46 | 11 / 11 | 0.373 / 0.300 / 0.341 / 0.698 | 0.998 | 0.821 |
| 21 (8B) | 300 | 1.000 / 1.000 | 0.001 | 5.28 | 1.79 | 18 / 14 | 0.324 / 0.302 / 0.293 / 0.817 | 0.990 | 0.668 |
| 21 (8B) | 600 | 1.000 / 1.000 | 0.003 | 4.41 | 2.23 | 32 / 16 | 0.313 / 0.313 / 0.288 / 0.893 | 0.982 | 0.551 |
| 21 (8B) | 1200 | 1.000 / 1.000 | 0.005 | 3.37 | 2.85 | 78 / 20 | 0.282 / 0.293 / 0.288 / 0.941 | 0.964 | 0.434 |

The `lam` split-half column reproduces report 24's `logreg_lam` on the shared seeds (0.282 vs 0.276 over ten seeds at n = 1200; the five-seed subset). The 8B's decline is visible here as the separator leaving the mean-difference direction (cos 0.91 → 0.43) for a support-vector-pinned direction whose own reproducibility is ~0.29 (the `weak` column), with the training loss ≤ 0.005 throughout.

## What can be inferred
- Every logistic fit at every n from 75 to 2000 per half, under all three penalties, separates its training half perfectly: accuracy 1.000, log-loss ≤ 0.002, minimum margin ≥ 3.4, converged in ≤ 25 iterations. n per half never exceeds 0.39·d, so this is the separable regime throughout. [supported by files]
- Hence the critic's mechanism is not what happens: there is no point at which the likelihood "takes over", because it is saturated from n = 75 on. The fitted direction is the penalty's tie-break among separating hyperplanes — the minimum-norm, max-margin direction — at every n the follow-up used. [supported by files]
- Weakening the penalty a hundred-fold changes that direction by almost nothing (cosine ≥ 0.975 on the same half) and does not make it more reproducible (split-half 0.52 vs 0.58 at n = 2000). The plateau at ~0.55–0.60 is therefore the reproducibility of the max-margin separator itself in the n ≪ d regime, not a penalty artefact and not a schedule artefact. [supported by files]
- The separator starts near the difference-of-means direction and departs from it as n grows (cosine 0.97 → 0.63 at layer 43, 0.97 → 0.61 at layer 54), while difference-of-means becomes reproducible (0.57 → 0.97). With few points the max-margin hyperplane is close to the mean difference; with more, support points pin it elsewhere. This is the mechanism behind Paper A's non-convergence result and is the reason the counter-paper's near-identity of logistic and mean-difference directions does not hold in this regime. [supported by files; the mechanism sentence is ours]
- On the 8B the same regime holds, and its fixed-penalty decline (report 24) is the separator moving from a mean-difference-like direction at small n to a support-vector-pinned direction at larger n, not the likelihood taking over. [supported by file; mechanism sentence ours]
- What this does not show: behaviour at n > d (never reached on these pools), or that the max-margin direction is *wrong* for steering — only that it is not reproducible at these n. [scope]

## Status
Complete for 27B layers 43 and 54 and 8B layer 21. Feeds RESEARCH_PLAN §2.1 (Paper A contribution 2) with the mechanism, and closes the §12 to-do "separability / margin / training-loss diagnostic".

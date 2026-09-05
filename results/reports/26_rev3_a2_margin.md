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
- Every logistic fit at every n from 75 to 2000 per half, under all three penalties, separates its training half perfectly: accuracy exactly 1.000 in every cell, arm and seed; worst single-fit log-loss 0.0055 (8B), minimum margin ≥ 3.3, at most 80 iterations. n per half never exceeds 0.39·d, so this is the separable regime throughout. [supported by files; bounds corrected after the critique from seed means to extremes]
- Hence the critic's mechanism is not what happens: there is no point at which the likelihood "takes over", because it is saturated from n = 75 on. The fitted direction is the penalty's tie-break among separating hyperplanes at every n the follow-up used. Whether it sits at the max-margin (C → ∞) limit is not shown: C = 50 is one point, ‖W‖ moves only 1.9 → 3.0, and the direction becomes *more* penalty-sensitive with n (cos(lam, weak) 0.9995 → 0.981 on the 27B, 0.998 → 0.964 on the 8B). [supported by files; the max-margin identification withdrawn after the critique]
- Weakening the penalty a hundred-fold changes that direction by little (cosine ≥ 0.96 on the same half) and does not make it more reproducible (split-half 0.52 vs 0.58 at n = 2000). Since the split-half cosine falls as C rises, the 0.55–0.60 plateau is an upper bound on the C → ∞ value, not a penalty artefact that a different C would remove. [supported by files; corrected after the critique]
- The separator starts near the difference-of-means direction and departs from it as n grows (cosine 0.97 → 0.63 at layer 43, 0.97 → 0.61 at layer 54), while difference-of-means becomes reproducible (0.57 → 0.97). A support-vector reading (few points: the separator is near the mean difference; more points pin it elsewhere) fits, but so does shrinkage of every small-n estimator toward one near-minimum-norm object that each then leaves for its own target; the file records no support-vector count, so the mechanism is a story and only the divergence is a finding. Either way it is why the counter-paper's near-identity of logistic and mean-difference directions does not hold in this regime. [numbers supported by files; mechanism not established]
- On the 8B the same regime holds, and its fixed-penalty decline (report 24) is the separator leaving a mean-difference-like direction at small n for a less reproducible one at larger n, with the training loss saturated throughout — not the likelihood taking over. [supported by file]
- What this does not show: behaviour at n > d (never reached on these pools), or that the max-margin direction is *wrong* for steering — only that it is not reproducible at these n. [scope]

## Status
Complete for 27B layers 43 and 54 and 8B layer 21. Feeds RESEARCH_PLAN §2.1 (Paper A contribution 2) with the mechanism, and closes the §12 to-do "separability / margin / training-loss diagnostic".


### Delta critique — margin/separability diagnostic (blind: a2_margin_diag.py, the three margin JSONs, the follow-up files)

**C1 — the finding holds; all three bounds are misquoted.** `train_acc` is exactly 1.0000 in every cell, arm and seed ✓. But the quoted limits are seed *means*, not extremes: worst per-seed log-loss **0.00545** (8B n=1200) not ≤0.005; smallest min-margin **3.275** (L54 n=2000) not ≥3.4; max iterations **80** not 78 — cosmetic, but stated as bounds.

**C5 ✓ and stronger than claimed.** `sh_lam` reproduces the follow-up's `logreg_lam` per seed on the **real** data to 3.3e-15 (L43), 6.8e-15 (L54), 5.0e-11 (8B) — not just 1e-9 on synthetic. `--selftest` passes. Caveat: the synthetic regime (d=200, n≤300) has sh rising to 0.94 and cos(lam,dom)=0.98 — it witnesses the code, not the phenomenon.

**C3/C4 numbers ✓** (cos(lam,weak) ≥0.9755 / ≥0.9645; 0.5209 vs 0.5805; 0.2878 vs 0.2817; cos(lam,dom) 0.974→0.634, 0.967→0.612, 0.913→0.434).

**(2) C2 is half-established.** Separation is beyond doubt, so "the likelihood takes over" is dead. But "≈ max-margin" is not shown: C=50 is one point, and the L2 path reaches max-margin only as ‖W‖→∞, logarithmically — ‖W‖ moves just 1.9→3.0 (L43 n=2000). Decisively, **cos(lam,weak) degrades with n** (0.9995→0.9814 at L43; 0.9983→0.9645 at 8B): the direction becomes *more* penalty-sensitive exactly where the claim is made, and a 0.019 direction change costs 0.06 of split-half. Since sh falls monotonically in C on the 27B, **0.55–0.60 is an upper bound at C≈0.15–0.5**, with the C→∞ value looking ≤0.52. C3's "reproducibility of the max-margin separator itself" overreaches.

**(3) C4's mechanism is a story.** An equally good account: at n≪d with C_lam=4 every estimator is shrunk toward the same near-minimum-norm object, and as n grows each resolves its own distinct population target, so cos(lam,dom) falls to cos(w*,Δμ*) — no support points needed. The file records no support-vector count, active set, or margin-span fraction. Two weak hints favour the story: median/min margin rises 1.4→2.5, and the closer-to-max-margin arm is further from dom (0.547 vs 0.634).

**(4) It refutes my report-24 hypothesis.** There is no separability crossover — the data are separable at n=2000 too. Plateau (21/23/25) and decline (24) are both properties of an *interpolating* classifier at d≫n whose direction is set by extreme points, not averages. Suggested item 2: *"On separable activations — every logistic fit from n=75 to 2000 attains training accuracy 1.0 — the multinomial direction is the penalty's tie-break among separating hyperplanes, not a likelihood estimate. Its split-half reproducibility plateaus at 0.55–0.60 (27B) and falls to 0.28 (8B) across a 27-fold n increase; difference-of-means, an average rather than an extremum, rises to 0.94–0.98 on the same splits, and the two diverge (cos 0.97→0.43–0.63)."*

**(5) Defects.** 5 seeds against the follow-up's 10 and **no CI anywhere**; `model`/`model_revision` empty again; a single `C_WEAK=50`, so "penalty-independent" rests on two points; `fit_multinomial` omits the `n_cls` misalignment assert `fit_logreg_lam` carries (a2_followup.py:165), relying on the outer guard at line 77.

### Reconciliation (author)

- **C1:** bounds corrected from seed means to per-seed extremes everywhere (registry `a2m.separable_every_n`, `a2m.8b_same_regime`).
- **C2/C3:** the max-margin identification is withdrawn; the surviving statement is "the penalty's tie-break among separating hyperplanes, increasingly penalty-sensitive with n; the plateau is an upper bound on the C → ∞ value". Registry note rewritten.
- **C4:** the support-vector sentence is now labelled a story, with the critic's shrinkage alternative next to it; the divergence itself stays quotable.
- **C5:** the reproduction is stronger than I claimed (1e-15 on real data); recorded.
- **(4):** the critic's Paper A item-2 sentence is adopted in RESEARCH_PLAN §2.1 verbatim.
- **Defects** (5 seeds, no CI, one weak C, missing `n_cls` assert): recorded as §12 to-dos; the script is not edited post hoc for these files.


## Addendum — v2 with ten seeds, CIs and a second weak penalty (2026-09-05 09:04Z; `a2_margin_*-v2.json`)
Answering the critique's defects. Layer 43, n = 2000: `lam` 0.581 [0.570, 0.589] vs C = 500 0.522 [0.499, 0.535]; cos(lam, dom) 0.633 [0.618, 0.641]; accuracy 1.000 everywhere, worst log-loss 0.0022. 8B layer 21, n = 1200: `lam` 0.276 [0.255, 0.290] vs C = 500 0.278 [0.253, 0.293]; cos(lam, dom) 0.435 [0.423, 0.450]; worst log-loss 0.0056. The five-seed numbers sit inside these intervals. **New caveat:** C = 500 reproduces C = 50 exactly (cos(weak, weak2) = 1.000 and the same ‖W‖ at every n), so lbfgs stops in the same flat region of the separable objective at both and the weak arms say nothing about the C → ∞ limit — which strengthens the critic's point that the max-margin identification is unproven. The lam-vs-weak gap on the 27B (0.58 vs 0.52, CIs disjoint) stands: raising C lowers reproducibility. Layer 54 v2 (n = 2000): `lam` 0.563 [0.557, 0.572] vs C = 500 0.506 [0.497, 0.516]; cos(lam, dom) 0.613; cos(weak, weak2) 1.000; accuracy 1.000, worst log-loss 0.0023 — same picture as layer 43.

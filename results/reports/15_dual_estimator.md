# 15 · Dual-estimator battery

**File:** `results/estimator/dualest_qwen36-27b.json` · **Script:** `src/estimator/dual_estimator_battery.py` (+ `src/core/coupling_e2.py`) · **Model:** Qwen3.6-27B · **Registry:** `dualest.stability_dissociation` QUOTABLE · **AUDIT:** 4

## What was run
One pooled corpus (k 80, n 2170, focus 43, seed 0); the E0 battery and a contagion dose-response are computed twice on identical activations, labels and folds — once with logistic-regression directions, once with difference-of-means. The contagion arm sweeps α ∈ {0, 0.5, 1, 2}.

## What the file shows
| metric | logreg | diff-of-means |
|---|---|---|
| present decode | 0.926 | 0.850 |
| other decode | 0.799 | 0.566 |
| cross-decode leak | 0.174 | 0.156 |
| present↔other cos | 0.039 | 0.095 |
| within-present cos | 0.189 | 0.328 |
| direction stability | 0.394 | 0.940 |

Contagion, per-emotion `present_by_alpha` over four α values; the stored 4-point slopes are negative for desperate/happy/calm (logreg) and happy/calm (dom), driven by the α = 2 cell (e.g. logreg happy 2.37 → 3.67 → 3.55 → 0.44). Recomputed here by OLS over α ∈ {0, 0.5, 1} only (not stored in the file): present slopes are positive for 6/6 under both estimators (logreg +0.51 to +3.24; dom +4.3 to +219); present > other for 2/6 under logreg (afraid, angry) and 4/6 under dom (desperate, afraid, happy, angry). No CI on any contagion number.

## What can be inferred
- On identical data, logistic regression classifies better and reproduces its direction far worse. [supported by file]
- The present/other split is near-orthogonal under both estimators; within-present similarity roughly doubles under the stable one. [supported by file]
- On the α ≤ 1 grid all six present slopes are positive under both estimators; the negative published slopes come from including α = 2. [supported by file, derived here]
- The RESEARCH_PLAN's "5 of 6 under difference-of-means" for present > other is **not** what this recomputation gives (4 of 6); the plan does not say which file or grid that count came from. [needs: A3]
- No contagion number here carries uncertainty. [needs: A3]

## Status
Registry: QUOTABLE for the dissociation. The matched-grid slopes quoted in the plan (+0.615 / +95.4) are derived, not stored.

### Independent critique (blind: saw only the scripts and the JSON)

## 1. What the code actually computes

**Pool and folds (shared).** `build_pool` (`src/estimator/dual_estimator_battery.py:36-59`) generates the 6×6 crossed dialogue pool once, at `main:179`, giving `X, yp, yo`; `n_pool` = 2170 (`/n_pool`), `rms` = 81.155 (`/rms`, computed at `:181`). Both arms then call `e0_metrics(X, yp, yo, method, n_cls, seed=a.seed)` (`:186-188`) with the same `seed=0`, so the permutation and the two folds at `:86-87` are byte-identical across arms. The only differing line is the branch in `fit` (`:63-76`). **For E0 the "identical activations, labels, folds" framing is true.**

**stability_cos** (`:102-106`): *one* split — the same single `f1`/`f2` from `:87`; fit present decoder on each half; take row `di` = index of `"desperate"` only; divide by that half's per-feature `sd` to return to raw space; unit-normalise; signed dot product. So: one split, one emotion, present decoder only, raw space, no resampling, no absolute value.

**present_other_cos** (`:97-99`): full-pool fits (not folds), `mean_i |cos(unit(Cp[i]), unit(Co[i])))|` over the 6 matched class rows, in **standardised** space (no `/sd`). Note the space is inconsistent with `stability_cos`, which is raw space.

**Contagion arm** (`contagion`, `:113-156`). Steering direction (`:116`) is `unit(Cp[e]/sd)` — *the arm's own* `Cp`. Readout (`:146-147`) passes the *same arm's* `mu, sd, Cp, Co` into `E2.utterance_score` (`src/core/coupling_e2.py:129-150`), which returns `Xs @ Cp.T` and `Xs @ Co.T`. **Within an arm, the steering direction and the `present` readout direction are the same vector.** The `other` readout uses `Co`, which was never injected.

Magnitude: `vec = al * rms * dirs[e]` (`:124`), added at layer `focus-1 = 42` to *every* token position (`coupling_e2.py:106-108`), i.e. at α=2 a perturbation of norm 162 against a mean activation norm of 81.

n per cell = `len(E2.SCENARIOS)` = **29** (`coupling_e2.py:29-88`), **one generation per scenario per α**, no repetitions. Sampling is stochastic and unseeded (`coupling_e2.py:120`, `do_sample=True, temperature=0.9`). B's reply is generated unsteered (`:140`) and measurement is unsteered (`:146`, no `steer` argument).

**Consequence:** the two arms do *not* share generations. Different `Cp` ⇒ different steering vectors ⇒ different A replies ⇒ different B replies, plus unseeded sampling noise. The docstring's "identical features, identical labels, identical folds… only the fitting line differs" (`:9-11`) applies to E0 only, **not** to the contagion arm.

**CIs:** none anywhere. `contagion` stores only `.mean()` over the 29 samples (`:148-149`); no SEM, no per-scenario values, no bootstrap. The JSON has no variance field of any kind.

---

## 2. Recomputation from the JSON

3-point OLS on α ∈ {0, 0.5, 1.0}. Note an identity: with `b = [-0.5, 0, 0.5]`, `b·b = 0.5`, so the 3-point OLS slope **equals `y(α=1) − y(α=0)` exactly** — the α=0.5 point contributes zero weight. The "matched dose grid" slope is a two-point difference.

| emotion | logreg present | logreg other | dom present | dom other |
|---|---|---|---|---|
| desperate | **+0.615** | +3.310 | **+95.391** | +58.226 |
| afraid | +2.821 | +1.988 | +187.706 | +101.733 |
| happy | +1.180 | +4.371 | +96.777 | +75.145 |
| calm | +0.509 | +1.415 | +4.268 | +8.795 |
| sad | +3.242 | +4.386 | +114.936 | +113.047 |
| angry | +3.186 | +2.321 | +219.211 | +106.490 |

- **Present slopes > 0: 6/6 in both arms.** Supports that half of C3.
- desperate: +0.615 (logreg), +95.391 (dom) — C3's quoted figures reproduce exactly.
- **present > other: dom 5/6 (calm the exception); logreg 2/6 (afraid, angry).** C4 says 1/6 for logreg. I cannot reproduce 1/6 under any definition I tried: 4-point slopes give 3/6 (afraid, sad, angry); α=1−α=0 differences give 2/6 (identical to the 3-point slopes by the identity above); requiring present > 0 as well gives 2/6.
- α=2 collapse: `/arms/logreg/contagion/happy/present_by_alpha` = `[2.3679, 3.6658, **3.5474**, **0.4409**]` — C3's "3.55 → 0.44" is exact. But the collapse is **not uniform**: `/arms/logreg/contagion/angry/present_by_alpha[2:]` = `[3.4881, 3.3216]` (flat), and `/arms/dom/contagion/afraid/present_by_alpha[2:]` = `[44.963, **66.075**]` (rises).

E0 values, all confirmed: `/arms/logreg/e0/present_acc` 0.92627, `/arms/dom/e0/present_acc` 0.84977; `/arms/logreg/e0/stability_cos` 0.39354, `/arms/dom/e0/stability_cos` 0.94030; `/arms/logreg/e0/present_other_cos` 0.03892, `/arms/dom/e0/present_other_cos` 0.09500. Also present but uncited: `other_acc` 0.79862 vs 0.56636; `within_present_cos` 0.18888 vs 0.32838; `cross_decode_leak` 0.17373 vs 0.15622 against `chance` 0.16667.

---

## 3. Objections

### C1 — "better classifier, worse direction, identical everything"

The *data-identity* part is verified (`:179`, `:186-188`). The numbers are in the JSON. Three problems.

**(a) `stability_cos` rests on a single split.** One permutation, `seed=0` (`:86`), one emotion (`:103`), one decoder. There is no sampling distribution for 0.394 vs 0.940 and no code path that would produce one. With 6 emotions available and a cheap re-permutation loop, the omission is not forced.

**(b) Split-half self-consistency is not a quality criterion for a direction.** It is maximised by any estimator that discards data-specific signal. DOM is an unregularised, unwhitened mean difference over ~5120 correlated features, so it is dominated by the top shared components of the activation covariance — which is exactly why its six *present* directions are more mutually parallel: `within_present_cos` 0.3284 (dom) vs 0.1889 (logreg). Higher stability and higher mutual redundancy are the same fact. Calling DOM "the better direction" on this evidence begs the question; the only external check in the repo (the contagion arm) shows logreg also gives 6/6 positive slopes.

**(c) Different estimands, unmatched regularisation.** `LogisticRegression(..., C=0.5)` (`:67`) fits a *multinomial softmax*; `coef_` row *i* is a softmax weight vector, not a one-vs-rest contrast. DOM row *i* (`:71-76`) is literally one-vs-rest. Comparing `cos` of "row *i*" across estimators compares different objects. No `C` sweep, no norm matching, no accuracy-matched comparison.

The decode half is the solid part: 2170 test predictions, paired on identical folds; a 0.076 accuracy gap is far outside binomial noise (SE ≈ 0.006). But per-fold accuracies are averaged away at `:107` and not stored, so even that is unauditable from the JSON.

### C2 — speaker orthogonality survives both estimators

Both cosines are small, and 0.039 vs 0.095 differ by 2.4×. There is **no null distribution and no CI** — nothing in `e0_metrics` permutes labels or bootstraps. Under isotropy in the ~5120 dims the code comment cites (`coupling_e2.py:177`), E|cos| ≈ √(2/πH) ≈ 0.011, so 0.039 and 0.095 are 3.5× and 8.6× the isotropic null; but activations are strongly anisotropic, so that null is the wrong one and the right one (row-permutation of `yo`) was not run.

DOM's `other_acc` is 0.566 vs logreg's 0.799 — its `Co` is a much weaker estimate, which changes what a cosine against it means. So the two cosines are not two independent measurements of one quantity.

"Property of the model, not of the objective" is over-claimed from n=2 estimators that are both linear fits, on the same standardised features (`:64`), from the same pool, at the same layer, with the same pooling. That varies the fitting line, not the objective in any broad sense.

The stronger evidence for the claim is in the JSON but not used: `cross_decode_leak` is 0.1737 (logreg) and 0.1562 (dom) against `chance` 0.1667 — the present probe predicts the *other* speaker's emotion at chance in both arms. And `present_other_cos` ≪ `within_present_cos` in both arms (0.039 vs 0.189; 0.095 vs 0.328). Those are the defensible statements.

### C3 — matched-grid slopes, α=2 exclusion

**(a) "Slope" overstates what was fit.** As shown above, the 3-point OLS slope is algebraically `y(1) − y(0)`. There is no dose-response evidence in it — a single two-point difference cannot distinguish a dose-response from a step.

**(b) The two numbers quoted side by side are not comparable.** +0.615 and +95.4 are projections onto unnormalised vectors: `Xs @ Cp.T` (`coupling_e2.py:150`). Logreg's `Cp` is L2-shrunk at C=0.5; DOM's is an unshrunk sum of ~5120 standardised mean differences, so its norm is larger by roughly √H × effect size. The units are arbitrary and estimator-specific. Reporting them in one sentence implies DOM shows a ~150× stronger effect; it shows a differently scaled one. Pooled cell SDs bear this out: 1.95 (logreg present) vs 159.9 (dom present).

**(c) No CI, n=29, one draw.** Each cell is a mean over 29 unseeded temperature-0.9 generations (`coupling_e2.py:120`), reps=1. Nothing in either file computes a standard error. 6/6 positive under a sign test is p = 1/64 = 0.016 *if* the six emotions were independent — they are not: they share the same 29 scenarios, the same name assignment (`:125`), the same B-model, and six directions that are mutually correlated at `within_present_cos` 0.189/0.328.

**(d) `present` is a matched filter to the intervention.** The injected vector *is* `unit(Cp[e]/sd)` (`:116`) and the present readout *is* `Xs @ Cp[e]` (`:147`). Positive slopes on that channel are close to guaranteed by construction whenever any trace of the steering survives into B's text.

**(e) The α=2 exclusion is not in this script.** `dual_estimator_battery.py:31` hardcodes `ALPHAS = [0.0, 0.5, 1.0, 2.0]`; `:150-153` fits and stores 4-point slopes; the JSON contains only 4-point slopes. Nothing here excludes α=2 — the exclusion is applied afterwards, in prose, to numbers this script did not compute. A declaration living in `coupling_e2_ci.py` does not govern `dual_estimator_battery.py`. I cannot read that file. To accept "pre-data" I would need: (i) the declaration's literal text and its position in the file; (ii) a commit timestamp for the line that predates the mtime of `dualest_qwen36-27b.json` / the run log; (iii) evidence the declaration is stated as a general α-cap rule, not a rule that names α=2 after α=2 was seen to misbehave.

Independently: the exclusion is *substantively* defensible (α·rms = 162 vs a mean activation norm of 81.155, `/rms` — a 2× off-manifold push at every token position), but it is also outcome-correlated. It flips logreg desperate from −0.058 to +0.615, happy from −1.146 to +1.180, calm from −0.441 to +0.509, and dom happy from −23.4 to +96.8 and calm from −52.3 to +4.3. **The "all six positive under both estimators" result exists only because α=2 is dropped**: on the full grid it is 3/6 (logreg) and 4/6 (dom). That is precisely the configuration where a pre-registration claim needs hard evidence.

### C4 — present shift exceeds other shift, 5/6 vs 1/6

**(a) The logreg count is wrong.** I get **2/6** (afraid +2.821 > +1.988; angry +3.186 > +2.321), not 1/6, under the same 3-point rule that reproduces C3's desperate figures exactly.

**(b) `present` and `other` are on different scales *within* an arm, and this drives the whole result.** From the α=0 cells, |present| exceeds |other| by 2.1× to 25× in every dom emotion (mean |present| 136.7 vs mean |other| 57.9 across all dom cells; ratio 2.36). In the logreg arm the ratio runs the other way (1.84 vs 2.99; ratio 0.62). So `‖Cp[i]‖ > ‖Co[i]‖` in dom and `<` in logreg, and a raw slope comparison inherits that.

Dividing each channel's slope by that channel's own pooled cell SD (logreg: 1.951 present / 3.319 other; dom: 159.88 / 59.50) reverses the result completely:

| | raw | scale-normalised |
|---|---|---|
| dom present > other | **5/6** | **0/6** |
| logreg present > other | 2/6 | 3/6 |

The claimed dissociation is a norm artifact of the DOM estimator, not a property of the model. Under any scale-free comparison, DOM shows present > other in **zero** of six emotions.

**(c) Even at matched norms the comparison is not fair**, per C3(d): `present` reads the injected direction, `other` does not.

**(d) 5/6 vs 1/6 with no CI on any of the twelve slopes establishes nothing.** Twelve numbers, each a difference of two 29-sample means from single unseeded draws, no SEM, no bootstrap, no permutation. And the two arms' contagion cells come from *different generations*, so any logreg-vs-dom difference here confounds estimator with sampling noise — the exact confound the script's docstring (`:9-11`) says it was built to eliminate.

---

## 4. What would be needed

**C1.** Bootstrap or repeat the split ≥200 times over all 6 emotions and both decoders; report a distribution for `stability_cos`, not one number. Sweep `C` for logreg (and report stability at matched decode accuracy). Add a same-estimator norm-matched control. Most importantly, replace self-consistency with an *external* criterion for "better direction" — e.g. steering efficacy at matched ‖vec‖, or held-out causal effect — since a constant vector scores 1.0 on the current one. Nearest defensible claim as stated: *"logreg decodes better held-out (0.926 vs 0.850) while its fitted coefficient vector is far less reproducible across a data split (single split-half cos 0.394 vs 0.940); which is preferable depends on whether the vector is used as a probe or as an intervention, which this experiment does not test."*

**C2.** A label-permutation null for `present_other_cos` (shuffle `yo`, refit `Co`, ≥1000 draws) and a bootstrap CI; report the two arms' cosines against their own nulls rather than against each other. Compute the cosine in one space consistently (`:99` uses standardised, `:106` uses raw). Nearest defensible claim: *"the present probe predicts the other speaker's emotion at chance under both estimators (leak 0.174 / 0.156 vs chance 0.167), and matched present/other directions are far less aligned than same-decoder cross-emotion directions (0.039 vs 0.189 logreg; 0.095 vs 0.328 dom)."*

**C3.** (i) Add ≥1 intermediate α and multiple reps per cell with a fixed generation seed; store per-scenario scores so a cluster-bootstrapped CI (cluster = scenario) can be attached to each slope. (ii) Fit slopes over ≥3 informative points — the current 3-point grid is a 2-point difference. (iii) Normalise both readouts (e.g. `Cp[i] ← unit(Cp[i])`, or report Cohen's d against the α=0 cell) so the two arms' numbers share units. (iv) Add a control direction (random unit vector at matched α·rms) — without it, positive present slopes are not distinguishable from a generic "steering perturbs B's text" effect. (v) For the α=2 exclusion: publish the declaration text, its commit timestamp, and the run timestamp; state it as a general on-manifold criterion (e.g. "α·rms ≤ ‖h‖") rather than naming α=2.

**C4.** Report the comparison in scale-free units (per-channel z, or unit-normalised `Cp`/`Co`), with a paired CI per emotion and a paired test across emotions clustered by scenario. Add the fair-comparison control: steer along `Cp[e]` and read *both* channels, then steer along `Co[e]` and read both — the dissociation only means something if the readout advantage follows the injected direction. And correct the logreg count to 2/6.

---

## 5. Verdicts

- **C1 — SUPPORTED-WITH-CAVEATS.** The data-identity, the decode gap (`/arms/*/e0/present_acc`) and the two stability numbers are all real, but `stability_cos` is a single split of a single emotion and self-consistency does not license "better direction."
- **C2 — SUPPORTED-WITH-CAVEATS.** Both cosines are small and reproduce from the JSON, but there is no null, no CI, and n=2 near-identical estimators cannot separate "property of the model" from "property of linear fits on these features."
- **C3 — SUPPORTED-WITH-CAVEATS on the slope arithmetic; UNDETERMINABLE on the pre-data exclusion.** 6/6 positive and the two quoted slopes reproduce exactly, but the "slope" is a two-point difference, the two numbers are on incommensurable scales, `present` is a matched filter to the injected vector, there are no CIs, and the 6/6 exists only because α=2 is dropped (full grid: 3/6 and 4/6) — so the pre-data status is load-bearing and I cannot see `coupling_e2_ci.py`; in any case nothing in `dual_estimator_battery.py` excludes it.
- **C4 — NOT SUPPORTED.** The logreg count is 2/6, not 1/6, and the dom 5/6 is an artifact of `‖Cp‖ ≈ 2.4 × ‖Co‖` in that arm: after per-channel scale normalisation dom gives 0/6 and logreg 3/6, reversing the claimed asymmetry.

## Reconciliation
AUDIT 4 covers only the α = 2 collapse. The critique's findings that are new and load-bearing: (i) `stability_cos` is one split of one emotion; (ii) the contagion arms do **not** share generations (different steering vectors, unseeded sampling), so the docstring's "only the fitting line differs" holds for E0 only; (iii) present and other readouts are on different scales *within* an arm (dom ‖Cp‖ ≈ 2.4 × ‖Co‖), and after per-channel scale normalisation the dom "present > other" count goes **5/6 → 0/6** while logreg goes 2/6 → 3/6 — the dissociation the plan's A3 treats as Paper A's exhibit is, on this file, a norm artifact of the estimator; (iv) the logreg matched-grid count is 2/6, not the 1/6 the plan quotes (the 1/6 is from `e2ci`, a different run); (v) the α = 2 exclusion is applied in prose to numbers this script computed on a four-point grid. On (v), `git log -S` shows the "drop the model-breaking alpha=2 regime" docstring in `coupling_e2_ci.py` has existed since commit c976a8e (2026-07-25); this file was produced 2026-08-31 and committed 2026-09-01 (88c6be4). The declaration genuinely predates the data, but it governs `coupling_e2_ci.py`, and applying it to this script's four-point grid is a re-analysis choice that should be stated as such. Verdicts: C1, C2 caveated; C3 caveated / undeterminable on pre-registration; C4 NOT SUPPORTED. A3 must compare present and other in scale-free units or the exhibit does not exist.

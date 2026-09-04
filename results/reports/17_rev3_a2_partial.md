# 17 · Rev-3 A2 — estimator battery (partial checkpoint layer 16 → completed run, 7 depths)

**File:** `results/rev3/a2_cells_qwen36-27b.json` · **Scripts:** `src/rev3/a2_estimator.py`, `src/rev3/a2_analyze.py`, `src/lib/acl_core.py` · **Model:** Qwen/Qwen3.6-27B · **Registry:** `a2.l16_dom_2000`, `a2.l16_logreg_2000` PENDING

## What was run
k = 160 dialogues per 6 × 6 cell, pool n = 4261; for each of seven depths, nine estimators are fit; held-out decode accuracy (one 70/30 split) and disjoint split-half cosine at six n values over ten seeds. Only hidden_states[16] (depth 0.25) is in the checkpoint; the focus layer (43) is absent.

## What the file shows
Layer 16 split-half at n = 75 → 2000: logreg 0.31 → 0.50; ridge 0.25 → 0.31; dom 0.50 → 0.96; pca_diff 0.23 → 0.93; mass_mean_cov 0.21 → 0.16; lda_shrunk 0.25 → 0.20. Decode (present): logreg family 0.91, dom 0.80. Full tables in `results/rev3/README.md`.

## What can be inferred
- At this layer, difference-of-means directions converge toward reproducibility with n; logistic rows improve and flatten below 0.5; the two covariance-corrected estimators get *less* reproducible with n. [supported by file]
- "Better classifier, worse direction" holds in pattern at layer 16. [supported by file]
- Nothing transfers to the focus layer until it is run. [needs: the running A2]
- Whether ridge (the RAPTOR-style remedy) closes the gap at the focus layer is the open A0 question. [needs: A2 layer 43]

## Status
Superseded by the from-scratch rerun launched 2026-09-04 (all seven depths).

### Independent critique (blind: saw only the scripts and the JSON)

---

## 1. What the code actually computes

**Pool and labels.** `pool_jobs` (acl_core.py:369–380) builds the fully-crossed 6×6 design and stores `"ep": ip` — the *index of the emotion named in the prompt*. `dialogue_prompt` (acl_core.py:133–138) instructs "…{A} is feeling deeply {eA}… never state or name any emotion". `generate_pool` drops any transcript containing a leak word (acl_core.py:151–154, 441–443) and any that fails `parse_final_A`. `pool_features` sets `yp = [it["ep"]]` (acl_core.py:457). **So the labels are the instructed emotion, never a measured or verified one**, and the only filter is lexical. The JSON records `config.k = 160`, `config.n = 4261` — i.e. 4261 of 6·6·160 = 5760 jobs survived (74%). Because the leak filter is keyed on emotion-specific word lists, retention is emotion-dependent by construction; the resulting class imbalance is not recorded anywhere in the file and cannot be checked from it.

**Feature.** Mean-pooled hidden state over the char span of A's *final* utterance, with a silent fallback to the whole sequence when the span tokenizes to nothing (acl_core.py:354–361).

**How `split_half` forms halves** (acl_core.py:668–712). Per seed `s`: **one** permutation `idx = rng.permutation(len(y))` (line 682), then `h1 = idx[:n]`, `h2 = idx[n:2n]` (line 687). Three consequences the comment on line 686 gets wrong or hides:

- Disjoint: **yes**. Independent permutations for the two halves: **no**, and it doesn't matter — one permutation cut in two is a valid disjoint split.
- "stratified: take n_per_half from each half of a class-balanced shuffle" (line 686) is **false**. It is a plain global permutation. The only guard is line 688, which merely checks all 6 classes are *present*. Halves are not class-balanced.
- The permutation depends only on `s`, **not on `n`**. So at a given seed the halves are strictly nested across the n-grid: `idx[0:75] ⊂ idx[0:150] ⊂ … ⊂ idx[0:2000]` (verified: `default_rng(0).permutation(4261)` is the same array at every n). **The six points on every "vs n" curve are not independent measurements.** At n=2000 the two halves cover 4000/4261 = 93.9 % of the pool.

Both halves are fitted with the *same* `seed=s` (lines 692–693), so `pca_diff`'s internal `default_rng(seed)` (line 592) runs an identical random stream in the two supposedly independent fits.

**Which space the cosine is in.** `space="raw"` is the default (line 670) and `a2_estimator.py:151` never overrides it. `raw_direction` returns `C[cls]/sd`, normalised (acl_core.py:653–656). That *is* the object used for intervention: `Steer._hk` adds the vector to a decoder block's output (acl_core.py:774–777) and `Ablate._mk` projects it out of the raw hidden state (acl_core.py:809–811). It is **not** the space used for measurement: `decode_acc` scores with `Xs @ C.T` in standardized space (acl_core.py:662–665). So C2 puts an accuracy computed in standardized space next to a reproducibility computed in raw space.

Two knock-on effects worth naming:

- For `dom`, `C[i] = mean(pos) − mean(neg)` in *standardized* space (line 584), so `raw_direction` = `(μ₊ − μ₋)/sd²`. **The thing called "difference-of-means" is a diagonally-whitened mean difference (diagonal LDA / naive Bayes), not CAA's raw mean difference.** Comparison to 2604.08169's CAA number is therefore not like-for-like.
- The cosine is taken after multiplying both vectors by a common diagonal `1/sd`, which is heavily heterogeneous in a residual stream. That concentrates the inner product on a few coordinates and mechanically changes the number. `space="std"` exists (acl_core.py:698–700) and was never run. **This is the cheapest missing control in the study.**

**Are the estimators' "directions" the same kind of object? No.** `logreg` rows are multinomial-softmax coefficients — a discriminative contrast against the softmax-weighted rest (acl_core.py:564–572); `ridge` is OLS on ±1 one-vs-rest targets (573–577); `dom` is a diagonally-whitened mean difference (578–584); `mass_mean_cov` is `Σ⁻¹Δμ` with total scatter (625); `lda_shrunk` is the same with *pooled within-class* scatter (622–623) — one-vs-rest, so the docstring's "multiclass LDA" (line 551) is a mislabel; `pca_diff` is a top singular vector of a paired difference cloud (585–618). These estimate **different population quantities**. Split-half is well defined *within* an estimator (fidelity to its own limit); it is not comparable *across* them as "direction quality".

**Hyperparameters.** `C=0.5` / `C=0.05` (line 571), `alpha=1000.0` (line 577), Ledoit–Wolf λ data-driven (line 521). **None is tuned per n**, and none is even n-normalised: sklearn minimises `0.5‖w‖² + C·Σᵢloss`, so the per-sample penalty is `1/(2Cn)` — effective regularisation weakens by 27× across the n-grid. `logreg_cv`, the estimator whose declared purpose is "is it fixable by tuning?" (acl_core.py:545), is **excluded from the split-half sweep** for cost (a2_estimator.py:43–47). The one question the tuned arm existed to answer is unanswered.

**`decode_acc`.** One 70/30 split, `rng = default_rng(a.seed)` with `a.seed=0` (a2_estimator.py:125–127) — n_train=2982, n_test=1279 — no repeats, no CI. This is *bit-identical* to `split_half`'s seed-0 permutation (verified), so the decode training set contains every split-half pair for n ≤ 1200: the two headline numbers in C2 are not independent evidence. `decode_acc` also **discards each estimator's native decision rule**: sklearn's intercept is dropped and replaced by a midpoint intercept `b` (acl_core.py:644–649), then rows are norm-normalised (663–664). The reported "logistic regression accuracy" is not logistic regression's accuracy.

**`lw_shrinkage_and_apply` (acl_core.py:488–532).** The Gram identity and Woodbury path are **exact**. I checked against `sklearn.covariance.LedoitWolf` at (n,d) = (200,80), (60,300), (40,1000): λ agrees to all printed digits and `Σ⁻¹V` agrees to max relative error 1.0e-15. The `a ≤ 1e-30` branch (line 525) silently forms a d×d `pinv` — at d≈5120 that would be minutes per fit, and it returns a pseudo-inverse, not an inverse. **λ is returned and then thrown away** (`sol, _lam = …`, acl_core.py:635). The single number that would diagnose the covariance result is computed on every fit and never recorded.

**Could the covariance estimators lose reproducibility with n mechanically? Yes — I reproduced it.** λ falls from 0.855 (n=75) to 0.208 (n=2000) on synthetic n≪d data, so `Σ = λμI + (1−λ)S` moves from near-isotropic (≈ `dom`) toward a rank-deficient sample covariance, and the whitening increasingly amplifies the worst-estimated eigendirections. Running `acl_core.split_half` itself on **synthetic data with a fixed, known true direction, d=2000, 6 classes, 5 seeds**:

| n/half | 75 | 300 | 600 | 1200 | 2000 |
|---|---|---|---|---|---|
| dom | 0.391 | 0.735 | 0.847 | 0.915 | 0.947 |
| mass_mean_cov | 0.404 | 0.701 | **0.740** | 0.691 | **0.640** |
| lda_shrunk | 0.402 | 0.700 | **0.744** | 0.699 | **0.651** |

The covariance-corrected estimators peak and then decline while `dom` rises monotonically — on data containing no representations at all.

---

## 2/3/4. Claim by claim

### C1 — split-half vs n at layer 16

**Numbers.** All check out against `cells["sh/16/<est>/<n>"].mean`: dom 0.4974 → 0.9624; logreg 0.3096 → 0.4951; ridge 0.2493 → 0.3128; mass_mean_cov 0.2098 → 0.1610; lda_shrunk 0.2523 → 0.1993. Every cell has `n_seeds = 10`.

**Per-seed spread** (sd of the 10 seed-means): dom 0.031 (n=75) → 0.0020 (n=2000); logreg 0.036 → 0.0085; mass_mean_cov 0.027 → 0.0068. CI widths: 0.006–0.037. Those CIs come from `ci_of` (acl_core.py:943–950), a percentile bootstrap over **10** values — anticonservative, and it captures **only split-choice noise inside one fixed pool**. There is one generation pass, one model, one seed. A ±0.003 interval on a number with a single data replicate overstates precision by an unknown factor.

**Monotonicity.** dom, dom_norm, pca_diff: strictly increasing. logreg: strictly increasing. ridge: **not** monotone (0.2857 at n=300 → 0.2836 at n=600). mass_mean_cov: **not** monotone (0.2051 at 150 → 0.2064 at 300); lda_shrunk non-increasing with a tie at 75/150. Paired across the shared seeds, the declines are real where claimed: mass_mean_cov 300→600 = −0.026 (1/10 seeds positive), 1200→2000 = −0.014 (1/10); lda_shrunk 300→600 = −0.028 (0/10).

**"Flattens between 1200 and 2000."** Paired over the same 10 seeds: Δ = +0.0069, sd 0.0114, t = 1.91, **6/10 seeds positive** — not distinguishable from zero, but with 10 seeds there is essentially no power to detect a step this size, so this is absence of evidence. The *deceleration* is well-supported: increments are +0.054, +0.056, +0.043, +0.026, +0.007. Note also that at n=2000 the halves consume 93.9 % of the pool, so "flattens" and "ran out of corpus" are confounded — the curve terminates where it does because `2n > N` (a2_estimator.py:146), not because it converged.

**Objections.**

1. The comparison is between estimators of **different estimands** (§1). A split-half cosine measures fidelity to *each estimator's own* population direction. Under a signal-plus-independent-noise model, split-half ≈ signal fraction, so cos(estimate, own limit) ≈ √(split-half): logreg 0.70, dom 0.98, mass_mean_cov 0.40. Those are three different targets. "dom is the better direction" does not follow.
2. **dom's stability is close to a law-of-large-numbers identity.** It is a difference of two sample means; its sampling error falls as 1/√n along every coordinate, with no matrix inversion, no optimisation, no regularisation constant. Reaching 0.96 at n=2000 is the expected behaviour of a mean and says nothing about whether the axis is emotion rather than an artifact of the prompt template, the name/topic lists, or the leak filter's emotion-dependent attrition. **A direction can be perfectly reproducible and completely wrong.**
3. The covariance decline is reproduced on synthetic data with fixed truth (§1 table). As worded — a bare empirical description — C1 is fine. Any representational reading of it is refuted by that table.
4. The n-curve points are **nested** (§1), so the six points are one dependent trajectory per seed, not six measurements.
5. `ridge` (alpha=1000, un-tuned) is not "the logistic family", and its ranking is a statement about alpha=1000.
6. **This is the 0.25-depth layer, not the focus layer.** `config.fracs[0] = 0.25` with the only layer present being 16 implies n_layers = 64; `splithalf_independent.py:27` documents `--layer 43`, and round(0.67·64) = 43 is the `focus()` convention (acl_core.py:174–176). The pre-declared headline layer (a2_estimator.py:195–204) is absent, and `diagnostics` (n/d, Fisher ratio, participation ratio) is not written to the checkpoint at all — `a2_analyze.py:62–65` says so explicitly. **The reader cannot check the regime the whole argument is about.**

**To make it defensible:** rerun each n on a fresh permutation; record λ per (n, estimator); report `space="std"` beside `space="raw"`; hold effective per-sample regularisation fixed across n (or tune per n) so estimator identity is not confounded with penalty strength; run the same sweep on synthetic fixed-truth data as a published null; drop the cross-estimator ranking or restate it as fidelity-to-own-limit; report the other six depths before the sentence says "at layer 16" in a context that reads as general.

---

### C2 — "better classifier, worse direction"

**Numbers.** `cells["dec/16/present/logreg"] = 0.9085`, `cells["dec/16/present/dom"] = 0.7991`; `cells["sh/16/logreg/600"].mean = 0.4624`, `cells["sh/16/dom/600"].mean = 0.8869`. All four match. The accuracies are internally consistent with a single 1279-item test set (1162/1279 and 1022/1279).

**Objections.**

1. **One split, no CI, no repeats** (a2_estimator.py:125–127). The binomial SE at n=1279 is ~0.008/0.011, so a 0.109 gap survives *test-set* noise comfortably; what is untested is train-split variability and the fact that the split reuses `split_half`'s seed-0 permutation exactly, so the two halves of C2 share data.
2. **Neither number is the estimator's own classifier.** `decode_acc` replaces sklearn's fitted intercept with a midpoint intercept and normalises rows (acl_core.py:644–649, 663–664). Whatever "logistic regression is the better classifier" means, this is not a measurement of it.
3. **The sentence's rhetorical force is self-defeating in the same table.** `dec/16/present/mass_mean_cov = 0.8905` and `dec/16/present/lda_shrunk = 0.9023` — near-logreg accuracy — while those two have the **lowest** split-half cosines in the whole sweep (0.161, 0.199). If high accuracy with low cosine indicts logistic regression, it indicts the covariance estimators three times harder, and the natural reading is that the cosine is measuring reproducibility of high-dimensional nuisance, not direction quality.
4. `dec/16/present/logreg`, `logreg_c005` and `ridge` are **all exactly 0.9085** (1162/1279 each). Three quite different estimators making the identical number of test errors is possible but is an unflagged coincidence that no check in the pipeline would catch.
5. **A real bug in the neighbouring column.** For `lbl="other"`, `a2_estimator.py:133` passes `pair_on = yo` while the label is also `yo`. In `fit_direction`'s pca_diff branch (acl_core.py:597–609), every stratum then has an empty positive or empty negative set, so `left` stays empty, `continue` fires for every class, and `C` remains the zeros allocated at line 562. `decode_acc` then argmaxes an all-zero score matrix and returns the class-0 base rate. `cells["dec/16/other/pca_diff"] = 0.1931` = 247/1279 — **not an accuracy**. It is reported in the same table with no guard.

**To make it defensible:** repeated stratified splits (or k-fold) with a CI and a paired McNemar test; each estimator scored with its *own* decision rule as well as the normalised one; the direction claim restated as "logistic coefficient rows converge to their own limit more slowly than the diagonally-whitened mean difference does to its", with a zero-direction guard in `fit_direction` so degenerate fits cannot be reported as accuracies.

---

### C3 — the independent second implementation

**Runnable check.** I ran `src/rev3/splithalf_independent.py --selftest`. Output: dom gaps 0.0199 (n=150), 0.0011 (300), 0.0005 (600); logreg 0.0067 (150), 0.0050 (300), 0.0058 (600). **The "≤0.006 at n ≥ 300" figure reproduces exactly** (max 0.0058). "Shares no code with acl_core" is true for the statistic (splithalf_independent.py:38–124 imports only numpy/scipy); acl_core is imported only inside `compare` (line 140) to run the other implementation.

**Same estimand?** In space, **yes**: `fit_diff_of_means_raw` returns `(μ₊−μ₋)/sd²` (line 77), matching `raw_direction` of `dom`; the logreg path returns `W/sd` (line 114), matching `C/sd`. Penalty is matched correctly (sklearn's `C=0.5` ⇒ `l2=2.0`, lines 44–49, 61). In *statistic*, **no**: the splits are stratified per class (lines 81–93) where acl_core's are not, and the seeds are disjoint (`10_000 + seed0 + s`, line 102). Agreement is therefore agreement of the **expectations** of two slightly different statistics, not a bit-for-bit cross-check. The independent logreg also omits the intercept (line 63, `n_cls*d` parameters) while sklearn fits one, and caps L-BFGS at 400 iterations (line 44).

**Objections.**

1. **The selftest never enters the regime the paper is about.** `selftest` uses d=400, n up to 600 (lines 163, 169) — n ≥ d at the top of the grid. The study runs d≈5120 with n ≤ 2000, i.e. n ≪ d **always**. And agreement degrades exactly as you approach that regime: the n=150 dom gap is 0.0199, ~4× the n≥300 gaps. The claim's own "n ≥ 300" qualifier is where the evidence stops being reassuring.
2. **The gap is the same size as the Monte-Carlo noise of the comparison.** Eight splits each, no error bar on the gap. From the real data, dom's seed-to-seed sd at n=300 is 0.0139 (`cells["sh/16/dom/300"].per_seed`), so an 8-split mean carries SE ≈ 0.005. A 0.005 agreement between two 8-split means is a null result at that resolution: it cannot exclude biases below ~0.01–0.02, which is the size of most of the effects C1 relies on.
3. **Only 2 of 8 estimators are dual-implemented** (line 142: `("dom", "logreg")`). `ridge`, `pca_diff`, `mass_mean_cov` and `lda_shrunk` have no second implementation — and the covariance pair is where the surprising, counterintuitive result lives. The docstring's claim that the Woodbury path "is verified against sklearn.covariance.LedoitWolf in the unit test" (acl_core.py:508) refers to a file I was not permitted to open; I verified the identity myself and it holds to 1e-15, but that is my check, not the repo's.
4. The declared tolerance in code is `tol = 0.05` (line 131). "≤0.006" is an observed value quoted post hoc, ~8× tighter than the criterion the gate actually enforces. A pre-registered gate of 0.05 would pass an implementation that was wrong by ten times the entire logreg-vs-dom flattening effect.
5. Nothing in the released JSON records that the dual check was ever run on the *real* features (`--npz`, line 183). The result file contains no dual-implementation row.

**To make it defensible:** run `compare` at d≈5120 on the actual `feats_qwen36-27b.npz` at layer 16 and at the focus layer, and commit the output; report a Monte-Carlo error bar on the gap (more splits, or a paired bootstrap); extend the dual implementation to `mass_mean_cov`/`lda_shrunk`; state the gate as the pre-declared 0.05, not the observed 0.006; and align the split rule (or run both rules through both implementations) so the comparison isolates the arithmetic rather than the sampling scheme.

---

## 5. Verdicts

| Claim | Verdict | One line |
|---|---|---|
| **C1** | **SUPPORTED-WITH-CAVEATS** | Every number matches `cells["sh/16/…"].mean`, but the n-points are nested rather than independent, the CIs are 10-seed split noise inside one generation pass, "flattens" rests on a step the design has no power to resolve (6/10 seeds positive), ridge is not logistic, and the covariance decline reproduces on synthetic fixed-truth data — so it is an estimator artifact, not a finding about the model. |
| **C2** | **NOT SUPPORTED as worded** | The four numbers are in the file, but "better classifier" rests on one un-repeated 70/30 split with no CI, scored with a substituted decision rule, on a permutation identical to the split-half seed-0 permutation; and the same table (`mass_mean_cov` 0.8905 decode / 0.1610 cosine) shows high accuracy with low cosine is not evidence about direction quality. The adjacent `dec/16/other/pca_diff = 0.1931` is a class-0 base rate produced by a zero direction, not an accuracy. |
| **C3** | **SUPPORTED-WITH-CAVEATS** | I reproduced the selftest and the ≤0.006 figure (max 0.0058 at n ≥ 300) and confirmed both implementations take the cosine in the same raw space, but the check runs at d=400 with n ≥ d — never the n ≪ d regime being studied, where the one available near-regime point already degrades to 0.0199 — covers only 2 of 8 estimators (not the covariance pair), has no error bar on a gap the size of its own Monte-Carlo noise, and was never run on the real features. |

## Reconciliation
No prior audit covers rev 3. What the critique adds, in order of consequence:
- **A bug in `a2_estimator.py`** (line 133): for the "other" label the paired-PCA estimator was told to pair on the same label it was fitting, emptying every stratum and returning a zero direction; `dec/16/other/pca_diff = 0.193` is a class-0 base rate, not an accuracy. Patched 2026-09-04 (pair on the *other* label set), verified on synthetic data (row norms 0 → 1, decode 0.13 → 0.67), deployed to both sandboxes; box 2's run was restarted from its pool cache before reaching the decode loop. The layer-16 checkpoint's `dec/16/other/pca_diff` cell is invalid.
- **The covariance-corrected estimators' decline with n is estimator behaviour.** Reproduced on synthetic fixed-truth data; the Ledoit–Wolf λ falls with n (0.86 → 0.21) and the whitening amplifies badly-estimated eigendirections. λ is computed on every fit and discarded (`acl_core.py:635`); it should be recorded.
- **`split_half`'s "stratified" comment is false** (plain permutation), and the permutation depends only on the seed, so the n-grid points within a seed are nested prefixes — one trajectory per seed, not six measurements.
- **"dom" in raw space is a diagonally-whitened mean difference** (`(μ₊−μ₋)/sd²`), not CAA's raw mean difference; the comparison to 2604.08169 is not like-for-like until `space='std'` is also reported (never run).
- **Regularisation is not n-normalised** (per-sample penalty 1/(2Cn)); `logreg_cv` — the estimator meant to answer "is it fixable by tuning" — is excluded from the split-half sweep.
- **`decode_acc` uses a substituted decision rule** and one split whose permutation is identical to the split-half seed-0 permutation; three estimators tie at exactly 0.9085; `mass_mean_cov` (0.891 decode / 0.161 cosine) shows high accuracy with low cosine is not evidence about "direction quality".
- **"Flattens" is not resolvable** (Δ +0.007, 6/10 seeds positive); "decelerates" is supported. Registry and README wording adjusted.
- **The dual implementation** reproduces (max gap 0.0058 at n ≥ 300) but only at d = 400 with n ≥ d, only for dom/logreg, with a coded tolerance of 0.05, and was never run on the real features. RESEARCH_PLAN §7's "satisfied" is downgraded to partial.
Verdicts: C1 SWC, C2 NOT SUPPORTED as worded, C3 SWC.


## Update 2026-09-04 · the full seven-depth run completed

Files: `results/rev3/a2_estimator_qwen36-27b.json`, full cells. Tables in `results/rev3/README.md`.
Against the critique above: the `pair_on` bug and a second head-slice bug in the cross-estimator
block were patched before the decode loop and the cross-estimator block ran on this pool; the
layer-16 numbers here are from a *different* pool (n = 4253 vs 4261) and moved slightly (dom
0.962 → 0.966, logreg 0.495 → 0.523 at n = 2000). Focus layer: dom 0.974, logreg
0.566, logreg↔dom 0.592, decode 0.951 vs 0.883. The
critique's structural points (raw space only, nested n-points, C not n-normalised, decode
rule substituted, dual implementation never run in-regime) all still apply to this file. A
delta critique follows.

### Delta critique — completed A2 run (blind: saw only the scripts and the JSONs)

**Both patches verified.** `a2_estimator.py:136` now pairs on the opposite label set; `dec/*/other/pca_diff` moves from the old class-0 base rate to 0.312–0.462 (`by_layer.*.decode.other.pca_diff`). `a2_estimator.py:169–175` takes a seeded subsample and passes `n_cls=NC`. Soundness check: `cross_estimator[m][m] == 1.0000` for all 9 estimators at all 7 depths (min over the file = 1.0), which is only possible if no class direction is the zero row — so all 6 classes were present. The matrix is exactly symmetric. Checkpoint and final file agree on all 462 cells.

**1. Headline.** All eight numbers reproduce: `headline.dom_split_half` 0.9736, ci `[0.9722,0.9748]`; `logreg_split_half` 0.5664, ci `[0.5608,0.5718]`; decode 0.9506/0.8832 (= 1213/1276 and 1127/1276, one 70/30 split, still no CI, a2_estimator.py:125–127); `logreg_vs_dom_cos` 0.5922; dom n=2000 spans 0.9656–0.9789 across all 7 depths; logreg↔dom 0.5433–0.5993.

**"Stops improving after n=600" is unresolvable as stated, and the count is threshold-dependent.** Peak at n=600 occurs on 4 of 7 depths (22, 43, 48, 54); "5 of 7" requires counting hs35 (+0.0032) as stopped. Paired across the 10 shared seeds:

| | 600→1200 | 1200→2000 | 600→2000 |
|---|---|---|---|
| hs43 | −0.0074 (t −1.69) | −0.0014 (t −0.32) | −0.0088 (t −1.50, **5/10 positive**) |
| hs48 | −0.0015 (t −0.23) | −0.0096 (t −3.96) | −0.0112 (t −1.42) |
| hs54 | −0.0091 (t −1.57) | −0.0098 (t −1.64) | −0.0189 (t −1.95) |

Pooling the four declining depths: −0.0112, t = −1.78, 8/10 seeds negative. **Suggestive of a decline, not a plateau, and not resolvable at 10 seeds** — while hs16 (+0.0469, t +7.68, 10/10) and hs29 (+0.0165, t +3.30) are *significantly still improving*. Contrast dom at hs43: +0.0401 then +0.0184, 10/10 both.

**The decisive new fact: it flips with C.** `logreg_c005` 600→2000 is positive at 6 of 7 depths (+0.038, +0.006, +0.028, +0.025, **+0.014, +0.013**, −0.005) where `logreg` is negative at 4. At hs43 `logreg_c005` ends *above* `logreg` (0.5749 vs 0.5664). A 10× change in an untuned constant reverses the headline's qualitative claim, exactly as the fixed-C/1-over-Cn confound predicts. `logreg_cv` is still excluded from the sweep (a2_estimator.py:47).

**2. Cross-estimator.** The fix is sound, but `sub` is recomputed inside the layer loop from `default_rng(a.seed+1)` (line 169) — the *identical* 2000 rows at every depth. So "0.54–0.60 on all depths" is one subsample read seven times, not replication, and no cross_estimator entry has a CI. Structurally the matrix shows three clusters: logreg/logreg_c005/logreg_cv ≥0.98; dom/dom_norm/pca_diff ≥0.967; mass_mean_cov nearer logreg (0.490) than dom (0.168).

**0.592 is not the published quantity.** `raw_direction` returns `C/sd` (acl_core.py:653–656), so dom's compared vector is `(μ₊−μ₋)/sd²` and logreg's is `w_std/sd`. 2604.08169's 0.98–0.99 is CAA's raw `μ₊−μ₋` against a raw logistic weight. Setting 0.592 against it is a category error in the paper's favour. It is **not recoverable from either JSON** (no `C`, no `sd` stored); it needs a rerun that also emits `cos(C_dom⊙sd, w_raw)` and the `space="std"` cosine.

**3. speaker_geometry.** `within_present_cos` is 0.1896–0.1960 for logreg at *every* depth and 0.1917–0.1965 for mass_mean_cov — that is `1/(K−1) = 0.2`, the value forced algebraically when six one-vs-rest contrasts sum to zero. It is not a measurement. dom sits at 0.3204 (hs43), i.e. **0.12 above the floor: dom's six directions share a common non-emotion component.** Normalised by each estimator's own within-set cosine, present/other is 0.17 (logreg), 0.30 (dom), 0.46 (mmc) — ~1.8×, not 2–4×. So the asymmetry is not a speaker-geometry finding; it is dom's shared nuisance showing up in both label sets, which further undercuts reading dom's 0.974 split-half as evidence of a *correct* direction.

**4. Diagnostics.** `n_over_d = 0.831` at all seven depths because d=5120 and n=4253 are the same at every layer — **it is constant by construction and can carry no across-depth information**; the file provides no test of the n≪d premise. What varies goes the wrong way: hs54 has the best Fisher (0.189) and lowest effective rank (230) — the most favourable regime — yet the worst logreg trend (−0.019) and the best dom (0.9789). **This file undercuts the premise.**

**5. `_provenance`.** `code_sha` is `""` (default at acl_core.py:1072, never set at a2_estimator.py:63–79). `acl_core_sha` matches on disk, but the one file whose version this whole rerun is about — the patched driver — has no hash. `seeds.pool = 0` is identical to the previous run, which produced n=4261, not 4253: at temp 0.9 the seed does not pin generation, so provenance cannot distinguish the two pools. Model revision, split seeds, timings and library versions are present and useful.

**6. Verdicts**

| Claim | Was | Now | One line |
|---|---|---|---|
| C1 | SUPPORTED-WITH-CAVEATS | **SUPPORTED-WITH-CAVEATS (weakened)** | All 7 depths now present and every number checks, but the "plateau" is 5/10-positive noise at hs43, reverses sign under `logreg_c005` at 6/7 depths, and the covariance decline still has the synthetic fixed-truth explanation. |
| C2 | NOT SUPPORTED as worded | **NOT SUPPORTED as worded** | Unchanged: 0.951 vs 0.883 is still one un-repeated split with a substituted decision rule, and `mass_mean_cov` (0.9373 decode / 0.1759 cosine) still shows accuracy and cosine are not measuring the same thing. |
| C3 | SUPPORTED-WITH-CAVEATS | **UNDETERMINABLE for this run** | Neither new JSON contains a dual-implementation row; the check was never run at d=5120 on these features, and `code_sha` is empty. |

## Reconciliation (delta)
Accepted in full. Corrections made to the record:
- **The "plateau after n = 600" is not established and flips with C.** Peak at 600 on 4 of 7 depths, pooled decline t ≈ −1.8 at 10 seeds; `logreg_c005` rises from 600 → 2000 on 6 of 7 depths and ends above `logreg` at the focus layer (0.575 vs 0.566). The registry entry `a2.focus_logreg_split_half` and the README/plan wording now say the shape of the logistic curve depends on the untuned constant, which is what report 14 predicted; the tuned-C sweep is the open question, not a footnote.
- **0.592 is not 2604.08169's quantity.** The compared vectors are `w/sd` and `(μ₊−μ₋)/sd²`, not raw logistic weight vs raw mean difference; the comparison is dropped from the registry text and kept only as "needs the raw-space CAA cosine emitted by a rerun".
- **`within_present_cos` ≈ 0.19 is the 1/(K−1) floor** forced when six one-vs-rest contrasts sum to zero; dom's 0.32 is 0.12 above it, i.e. dom's six directions share a non-emotion component. The present/other asymmetry normalised by each estimator's own floor is ~1.8×, not 2–4×; README corrected.
- **n/d is constant by construction** (same n, same d at every layer) and the depth with the most favourable Fisher/rank has the worst logistic trend; the file undercuts the n ≪ d premise rather than supporting it. Plan §10 row corrected.
- The cross-estimator subsample is the same 2000 rows at every depth (no replication, no CI); `code_sha` is empty; the pool seed does not pin generation (4261 vs 4253 from seed 0). Added to the to-dos.
Updated verdicts: C1 SWC (weakened) · C2 NOT SUPPORTED as worded · C3 UNDETERMINABLE.

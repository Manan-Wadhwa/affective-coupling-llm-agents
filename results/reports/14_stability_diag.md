# 14 · Split-half stability diagnostic

**Files:** `results/estimator/stabdiag_qwen36-27b.json`, `stabdiag_llama3-abl-big.json`, `stabdiag_llama3-abl.json` · **Script:** `src/estimator/stability_diag.py` (+ `src/core/coupling_e2.py`) · **Registry:** `stabdiag.logreg_27b_600`, `stabdiag.dom_27b_600`, `stabdiag.dom_8b_1200` QUOTABLE; the small 8B file UNREGISTERED

## What was run
A pooled dialogue corpus is split into two disjoint halves; the same estimator (multinomial logistic regression, or difference of means) is fit on each half at several n per half and the cosine between the two resulting directions is recorded, over 3 random splits.

## What the files show
| file | pool | n/half → logreg | → diff-of-means |
|---|---|---|---|
| Qwen3.6-27B (k 70, focus 43, d 5120) | 1853 | 150: 0.300 · 300: 0.348 · 600: **0.409** | 0.687 · 0.808 · **0.893** |
| Llama-3-8B-abl big (k 220, focus 21, d 4096) | 3927 | 150: 0.126 · 300: 0.183 · 600: 0.190 · 1200: **0.219** | 0.425 · 0.694 · 0.822 · **0.898** |
| Llama-3-8B-abl small (k 70) | 1180 | 150: 0.181 · 300: 0.158 | 0.471 · 0.622 |
3 splits per cell; sd across splits 0.001–0.043.

## What can be inferred
- On both models, difference-of-means directions reproduce across disjoint halves far better than logistic-regression rows at every n tested. [supported by file]
- Logistic agreement rises with n on both models (0.30 → 0.41; 0.13 → 0.22) but remains well below difference-of-means at the largest n. "Does not improve with data" is not what the files show; "improves slowly and stays low" is. [supported by file]
- Three splits per cell bound how precisely any of these means is known. [supported by file]
- Whether a tuned regularisation would close the gap is not tested here. [needs: A2, which sweeps ridge and C]

## Status
Registry: three QUOTABLE numbers resolve. A2 (running) is the full grid with 10 splits and CIs.

### Independent critique (blind: saw only the scripts and the JSONs)

## 1. What the code actually computes

**Pool and labels.** `stability_diag.py:33-39` builds a fully crossed 6×6 grid of (present, other) emotions with `k` dialogues per cell, one generation pass (`:41`). Items surviving the leak filter and the parser (`:44-49`) are kept; features are the pooled final-utterance activations at `focus = round(0.67*num_layers)` (`:52`, `:97`). Labels are `y = it["ep"]` (`:53`) — the *prompted* present emotion, never a verified one. So the fitted direction is partly a prompt-template direction.

**Halves.** One permutation, sliced once (`:104-105`): `h1`/`h2` are disjoint, not stratified by class or by (ep,eo) cell. **There is only one split-half partition in the entire run.** The `for rep in range(3)` loop (`:117`) resamples *subsets within those same fixed halves* (`:118-119`), so `cos_all` (3 entries in every row of every file) is 3 nested subsamples of one split, not 3 splits. At `n=600` of `n_half=926` and `n=1200` of `n_half=1963`, consecutive reps share ~65% / ~61% of their data in expectation, which is why the reported spread collapses (qwen `results[2].diff_of_means.cos_all` = `[0.89353, 0.89199, 0.89244]`, sd 0.0008). That sd measures nothing but subsample overlap.

**Space.** `fit_logreg` (`:62-65`) standardizes, fits, then returns `unit(coef[ei]/sd)` — the raw-space weight vector. `fit_dom` (`:70-75`) standardizes, takes the standardized-space one-vs-rest mean difference, then divides by `sd` *again*, giving `raw_diff/sd²` — i.e. diagonal-LDA in raw space. Despite the docstring at `:69` ("in standardized space"), both estimators are compared **in raw space**, consistently and matching the steering consumer `coupling_e2.py:233` (`decs["Cp"][ei] / decs["sd"]`). But the *other* consumer, measurement, uses the direction in standardized space (`coupling_e2.py:149-150`, `Xs @ decs["Cp"].T`). Cosine is not invariant to the diagonal `1/sd` rescaling, so the reported agreement is the agreement of the steering vector only; agreement of the probe as used is never measured. I cannot compute the standardized-space cosines — the features are not in the repo.

**"Agreement."** Not a per-class average and not an absolute value: a single signed dot product of two unit vectors for the single class `ei = EMOTIONS.index("desperate")` = 5 (`:101`, `:123`). For logreg that is **one row of a 6-class multinomial fit**; for dom it is a **one-vs-rest contrast** (`:72`). Different estimands.

**Regularization.** `C=0.5` is a default at `:61`, never passed at the call site (`:120`), never tuned; the same fixed `C=0.5` appears in production at `coupling_e2.py:200-201`. `max_iter=3000`; no convergence check is recorded.

---

## 2–5. Per claim

### C1 — "logistic 0.409 at n=600/half on the 27B; difference-of-means 0.893"

**Numbers.** Exact. `stabdiag_qwen36-27b.json` `results[2].n_per_half` = 600; `results[2].logreg.cos_mean` = 0.40873337785402936; `results[2].diff_of_means.cos_mean` = 0.8926525712013245. `hidden` = 5120, `n_pool` = 1853, `n_half1/2` = 926/927, `k` = 70, `seed` = 0. Three reps each; sd across reps 0.0243 (logreg) and 0.0008 (dom).

**Objections.**
1. *"Split-half" is one split.* Both numbers come from a single permutation (`:104`). The dominant variance component — which dialogues land in which half — is never resampled. The quoted sds are subsample-overlap artifacts, not split-to-split error bars. There is no interval on either number.
2. *The comparison is a hyperparameter comparison, not an estimator comparison.* sklearn minimizes `½wᵀw + C·Σᵢ loss_ᵢ`, so the effective per-sample ridge is `1/(2Cn)`. Difference-of-means is (up to scale) exactly the `C→0` limit of this same family: the gradient of the one-vs-rest logistic loss at `w=0` on standardized data is proportional to `μ_pos − μ_neg`. So the experiment compares an *untuned member of a family* against *its own strongly-regularized limit*, at `n/d = 0.117`, and reports that the limit wins. That is the expected result, and it is a statement about `C=0.5`, not about logistic regression.
3. *Different estimands.* logreg's `coef[5]` must trade off against 5 competing softmax rows; dom's is a single OvR contrast. Also `coef[ei]` indexes positionally without consulting `classes_` (`:65`) — silently wrong if a class were absent from a subsample; unstratified subsampling (`:79`) makes that a latent, unchecked risk (not triggered at these n).
4. *n=600 is the ceiling for this pool*, not a chosen operating point (`n_half=926`, so 1200 is skipped by `:112-113`). The 27B is never observed above `n/d = 0.12`.
5. *Deployment gap.* `coupling_e2.train_decoders` defaults to `k=10` (`:153`), i.e. ~360 jobs → a few hundred kept items total, no split. The regime where the deployed decoder actually sits is nearer the `n=150` row, where dom agreement is 0.687 (27B) and 0.425 (8B) — not 0.89.

**To make the claim as worded:** ≥20 independent permutations of the pool with mean ± sd (or a bootstrap CI) per estimator, and a statement of the comparison space. **Nearest defensible claim as it stands:** "On one random half-split of a 1853-dialogue Qwen3.6-27B pool, an untuned `C=0.5` multinomial fit reproduced its desperate row at cos 0.41 across halves at n=600/half, while the class-mean contrast on the identical subsamples reached 0.89 (raw-space cosine, 3 overlapping subsamples, no split-level error bar)."

**Verdict: SUPPORTED-WITH-CAVEATS** — the two numbers are exactly what the JSON contains, but they rest on one split with no error bar, and the logreg side is an untuned `C`.

---

### C2 — "difference-of-means 0.898 at n=1200/half on the 8B"

**Numbers.** Exact, in `stabdiag_llama3-abl-big.json`: `results[3].n_per_half` = 1200, `results[3].diff_of_means.cos_mean` = 0.8982250889142355, `cos_all` = `[0.90307, 0.89540, 0.89621]`. `model` = `failspy/Llama-3-8B-Instruct-abliterated`, `hidden` = 4096, `n_pool` = 3927, `n_half1/2` = 1963/1964, `k` = 220. Note the *other* 8B file, `stabdiag_llama3-abl.json`, has `n_pool` = 1180 and only reaches n=300 — "the 8B" must mean the k=220 run, and the claim should say so.

**Objections.**
1. Same single-split problem as C1, aggravated: at n=1200 of 1963 the three reps overlap ~61%, so `sd = 0.0042` is close to meaningless as a precision estimate. The honest scale of variability is visible *between* the two 8B files at matched n: at n=150, dom is 0.4246 (`llama3-abl-big` `results[0]`) vs 0.4714 (`llama3-abl` `results[0]`) — same model, same `focus`=21, same `seed`, different pool, a 0.047 gap that dwarfs every within-row sd.
2. The number is n-specific and the reader must not generalize it: the same file gives dom 0.4246 / 0.6936 / 0.8224 / 0.8982 at n = 150 / 300 / 600 / 1200 (`results[*].diff_of_means.cos_mean`). Agreement is still climbing steeply (+0.155 per doubling of n, fitted over these four points); 0.898 is where this pool ran out, not a plateau.
3. Raw-space cosine only; the probe use at `coupling_e2.py:150` is in standardized space and is not measured.
4. The label is the prompted emotion (`:53`), so this is reproducibility of an estimator on a fixed synthetic DGP, not reproducibility across generation runs.

**To make the claim as worded:** multiple permutations with a CI, plus one point past the apparent knee to show it is a plateau rather than a still-rising curve. **Nearest defensible claim:** "On a 3927-dialogue 8B pool, the class-mean contrast reproduces across a disjoint half-split at cos 0.898 at n=1200/half (n/d = 0.29), up monotonically from 0.425 at n=150; single split, 3 overlapping subsamples."

**Verdict: SUPPORTED-WITH-CAVEATS** — the value is exactly as reported; the caveats are the single split and that it is the last measured point on a still-rising curve, not a converged value.

---

### C3 — "logistic does not improve with data: 0.126 → 0.219 across 8× n; the coefficient row … does not converge to a stable direction with increasing data, while difference-of-means does"

**Numbers.** The endpoints are exact: `stabdiag_llama3-abl-big.json` `results[0].logreg.cos_mean` = 0.12605366359154382 (n=150) and `results[3].logreg.cos_mean` = 0.21904703478018442 (n=1200); 150→1200 is 8×.

**This is the weakest claim, and the wording is contradicted by the file it cites.**
1. *It does improve, monotonically.* The full row is 0.1261 → 0.1826 → 0.1900 → 0.2190 (`results[0..3].logreg.cos_mean`). Every step is upward. "Does not improve with data" is false as written; +74% over 8× is a positive slope of ~0.029 per doubling of n. The defensible statement is "improves ~5× too slowly to be usable" — at that rate reaching 0.9 needs on the order of twenty further doublings — not "does not improve."
2. *Four points from one split cannot support "does not converge."* No asymptotic claim is testable here: the maximum `n/d` reached is 0.293 (8B) and 0.117 (27B). The experiment never leaves the `n ≪ d` regime it is making a claim about, and there is no fit, no extrapolation, and no error bar at the split level (three overlapping subsamples of one partition, `:104-105`, `:117-119`).
3. *The other 8B file goes the other way.* `stabdiag_llama3-abl.json` gives logreg 0.1808 (n=150) → 0.1581 (n=300) — a *decrease* — where `llama3-abl-big` gives 0.1261 → 0.1826 at the same n on the same model and layer. Two runs disagree in sign of the trend over the only n-range they share. Whichever direction one prefers, the run-to-run spread (~0.055 at n=150) is comparable to the entire n=150→600 movement in the big file (0.064). The trend being asserted is not larger than the noise between pools.
4. *`C` is fixed and this actively manufactures the result.* With sklearn's parameterization the effective per-sample ridge is `1/(2Cn)`, so the 8× increase in n also weakens regularization 8-fold. The experiment therefore confounds "more data" (helps) with "less effective shrinkage in a `p ≫ n` problem" (hurts), and reports the sum. A `C` tuned per n by cross-validation would shrink harder at small n; in the strong-shrinkage limit the logistic direction *becomes* the class-mean contrast, so a tuned logreg should track dom to within noise. **Yes, tuning `C` would very plausibly change the answer**, and nothing in `stability_diag.py` rules that out (`:61`, `:120`; no CV anywhere in either file).
5. *The logreg/dom comparison is not fair as framed.* dom "has no hyperparameter" precisely because it is the `C→0` endpoint of the logistic family. Comparing an arbitrary interior point against the family's own endpoint and concluding the family fails is a category error. Add: multinomial-row vs OvR estimands (`:65` vs `:72`), and `max_iter=3000` with no recorded convergence status — a non-converged lbfgs at n=1200, C=0.5, d=4096 would itself produce path-dependent directions indistinguishable from statistical instability.

**To make the claim as worded:** (a) ≥20 independent splits with CIs; (b) `C` chosen by CV at each n (and ideally a `C ∝ 1/n` arm holding effective λ fixed) so the estimator is not handicapped as n grows; (c) n extended to and past `n ≈ d` so "converge" is about an observed regime; (d) agreement of *all six* rows, or an explicit statement that only the desperate row is tested; (e) both models, both pools, agreeing on the sign of the trend. **Nearest defensible claim from the data in hand:** "On one 8B pool, an untuned `C=0.5` multinomial fit's desperate row improved only from 0.126 to 0.219 across an 8× increase in n (~0.03 per doubling), while the class-mean contrast on the identical subsamples went 0.425 → 0.898 (~0.15 per doubling) — roughly a 5× difference in rate; a second 8B pool shows no logreg trend at all over its shorter n-range."

**Verdict: NOT SUPPORTED** — the endpoints are real, but "does not improve" contradicts the monotone increase in the cited file, "does not converge" is an asymptotic claim from four points inside `n/d ≤ 0.29` on a single split, the two 8B files disagree on the trend's sign, and the fixed `C=0.5` confounds the n-sweep in exactly the direction that produces the conclusion.

## Reconciliation
Not in AUDIT.md at all — these numbers were treated as uncontested. The critique finds: (i) one permutation defines the halves; the three "splits" are nested subsamples inside it, so the reported sd is subsample overlap, not split-to-split error; (ii) both estimators are compared in raw space while the *probe* is used in standardised space, and cosine is not invariant to the `1/sd` rescaling; (iii) `C = 0.5` is fixed, so the effective ridge weakens 8× across the n sweep, confounding "more data" with "less shrinkage" — difference-of-means is the `C → 0` limit of the same family; (iv) "does not improve with data" contradicts the monotone rise in the cited file; the small 8B file shows the opposite trend over its range. Verdicts: C1, C2 SUPPORTED-WITH-CAVEATS; C3 NOT SUPPORTED. Rev-3's `acl_core.split_half` draws ten independent permutations and offers both spaces, and A2 sweeps `C`, so (i) and (iii) are addressed in the running A2 if both spaces are reported; the registry's `stabdiag.*` entries remain quotable as numbers but the sentence in RESEARCH_PLAN §2.2 ("does not improve with data") does not.

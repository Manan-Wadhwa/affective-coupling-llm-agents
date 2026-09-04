# 18 · Rev-3 A3 — present-vs-other dissociation, with scale-free re-read

**Files:** `results/rev3/a3_dissociation_qwen36-27b.json`, `results/rev3/a3_scalefree_qwen36-27b.json` · **Scripts:** `src/rev3/a3_dissociation.py`, `src/rev3/a3_scalefree.py`, `src/lib/acl_core.py` · **Model:** Qwen/Qwen3.6-27B rev 6a9e13bd · **Registry:** `a3.exhibit_dom_raw`, `a3.exhibit_dom_scalefree`, `a3.published_config_reversal`, `a3.direction_stability` — all quotable

## What was run
A2's probe pool (n = 4253) split into disjoint DIR/READ halves; steering directions from DIR, readout probes from READ. A is steered at α ∈ {0, 0.5, 1} along the logistic or difference-of-means direction; B replies through the seeded generator; B's reply is scored for present-e and other-e under both readout estimators from the same activations. Three configurations: logreg→logreg (published), dom→logreg, dom→dom. 2 reps × 29 scenarios. Per-arm slope CIs scenario-blocked; the present−other contrast paired. The scale-free script re-reads the per-sample rows with each channel divided by its own SD (α = 0, or pooled).

## What the files show
See `results/rev3/README.md` A3 for the full tables. Counts of present > other: logreg→logreg 0/6 on every scale (other > present significant 4/6 raw, 3/6 z0); dom→logreg 1/6; dom→dom raw 4/6 (2 sig) → z0 2/6 (0 sig, 3 sig reversed) → zall 1/6. Stability on this pool: dom 0.915, logreg 0.576 at n = 600/half.

## What can be inferred
- Under the published configuration with a non-circular probe, B's other-speaker readout rises at least as fast as its present readout for every emotion. [supported by file]
- The raw dom→dom count (4/6) is a norm artifact: on either per-channel scale the count drops to 1–2 of 6 with no significant present-dominant emotion and three significant reversals. [supported by file, derived by script]
- The plan's Paper A exhibit ("false under logreg, true under the stable estimator") is not present in this data. [supported by file]
- Why the other-speaker readout moves more is not identifiable here. [needs: cross-decode leak on this pool, content-matched pairs]

## Status
Complete. Exhibit withdrawn in RESEARCH_PLAN §2.3 and §10.

### Independent critique (blind: saw only the scripts and the JSONs)

## 1. What the code actually computes

**Pool and splits.** `probe_k=160` → 6×6×160 jobs, kept `$.probe_n = 4253` after leak/parse filtering (`acl_core.py:369-380`, `:383-452`). `a3_dissociation.py:94-95` takes one permutation and cuts it in half: `DIR` = 2126, `READ` = 2127 (`$.dir_half_n`, `$.read_half_n`). Steering directions fit on `X[DIR]` (`:99-100`), both readout probes on `X[READ]` (`:101-104`).

**Is the READ probe independent of the steering direction?** Only in the finite-sample sense. Disjoint samples remove shared *estimation noise*; both still target the same population axis, which is the point — otherwise the readout could not detect the steering at all. So "non-circular" (`a3_dissociation.py:28-30, 81-82`) is a claim about noise-sharing, not about independence of the estimand. Note `read_present` passes `pair_on=yo[READ]` and `read_other` passes none (`:101-104`); `pair_on` is consumed only by `pca_diff` (`acl_core.py:553, 597`), so the asymmetry is inert for `logreg`/`dom`.

**Scoring.** `a3_dissociation.py:158-161`. Both channels are scored as `((f - mu)/sd) @ C[ei]`. Because both probes are fit on the *same* `X[READ]`, `mu` and `sd` are byte-identical (`acl_core.py:560`); the only difference between channels is the coefficient row, and its **norm is not divided out**. The library's own decoder does divide by the row norm (`acl_core.py:663-664`, "so this compares directions, not score scale") — the A3 readout omits exactly that step. The intercept `b` is also dropped, which is harmless for a slope.

**Norm ratio — recoverable?** No. From `rows` you get the *score* SD ratio, which equals (‖C_p‖/‖C_o‖)·(σ along û_p / σ along û_o). The standardized activation covariance is nowhere near isotropic, so the second factor is not boundable from the file. What is observable (my computation, `$.rows`, α=0):

| readout | present/other SD ratio at α=0 |
|---|---|
| `dom` | 2.08, 1.80, 1.65, 2.22, 2.40, 1.79 (desperate…angry) |
| `logreg` | 1.08, 0.93, 0.97, 0.85, 1.07, 0.77 |

`a3_scalefree.py:6-7` asserts "~2.4x"; 2.40 is the *maximum* over emotions, not typical (median ≈1.9). Under `logreg` readout the ratio is ≈1, which is why raw and z0 barely differ there.

**The slope.** `ols_slope` (`acl_core.py:953-956`) with x = (0, 0.5, 1) gives centred x = (−0.5, 0, 0.5), so slope = y(1) − y(0) **exactly**; the α=0.5 cell contributes nothing to any number in either JSON. Two thirds of the dose grid is decorative.

**The test.** "present > other" is `contrast["diff"] > 0` (`a3_dissociation.py:212`), i.e. sign of [y_p(1)−y_p(0)] − [y_o(1)−y_o(0)]. Significance is a 95% percentile bootstrap CI excluding 0 (`acl_core.py:1015`).

**The resampling — the main code finding.** `slope_ci` is scenario-blocked (`acl_core.py:972-983`), and the driver passes `block=blk` for the marginal slopes (`a3_dissociation.py:207-208`). But `paired_slope_contrast` (`acl_core.py:994-1016`) has **no `block` argument at all**, and draws a *fresh* `ix` inside the per-dose loop (`:1008-1011`). So it is paired across channels, **not blocked by scenario, and not paired across doses**. Every significance star on `present_minus_other` — the only quantity C1 and C2 rest on — comes from the unblocked bootstrap that the driver's own docstring (`a3_dissociation.py:14-17`) condemns in the published work. The provenance stamp `$._provenance.control_pointers.blocked_bootstrap` is true of `slope_ci` and false of the statistic actually being reported.

**n and reps.** `$._provenance.config.reps = 2`, `n_scenarios = 29`. n per dose per cell = 58, but the two reps differ only in a cyclic name-pair shift (`a3_dissociation.py:123`) and sampling seed; the 29 scenario texts are identical. Effective independent units = 29.

**A shared baseline arm.** At α=0, `vec is None` (`:131`) and the generation seeds (`:133`, `:146`) depend on `rep`, not on emotion — so the α=0 cell is the *same 58 B-utterances for all six emotions* within a `steer_est`. Evidence from `$.rows`: cross-emotion correlations of `present_dom` at α=0 are large and semantically structured (desperate↔afraid +0.70, desperate↔happy −0.55, desperate↔calm −0.59), impossible if the rows were different texts. Consequence: the six per-emotion contrasts share their entire subtrahend y(0). They are not six independent votes.

**Stability.** `split_half(X, yp, m, n_per_half=min(N//2, 600), seeds=range(10))` (`:105-106`), `space` defaults to `"raw"` (`acl_core.py:670`), cosine of `raw_direction` (`:696-697`).

## 2. Recomputation from `$.rows`

I re-derived every count independently. **All three claims' arithmetic reproduces exactly**, and `a3_scalefree_qwen36-27b.json` matches my numbers cell for cell:

| config | raw | z0 | zall |
|---|---|---|---|
| logreg→logreg | 0/6, 0 sig, **4 rev** | 0/6, 0, 3 rev | 1/6, 0, 2 rev |
| dom→logreg | 1/6, 1 sig, 5 rev | 1/6, 1, 4 rev | 1/6, 1, 5 rev |
| dom→dom | **4/6, 2 sig**, 0 rev | **2/6, 0, 3 rev** | **1/6**, 0, 3 rev |

The scale-free script is arithmetically correct. Two nits: `a3_scalefree.py:69` guards only `P`, never `O`; and `scale()` (`:42-47`) recomputes an SD from data that is then re-bootstrapped as if the divisor were fixed — the CI ignores the estimation error in the denominator (minor at n=58).

**Is z0 or zall the right normaliser? Neither, and zall is the worse of the two.** `zall` divides by a pooled SD that is *inflated by the dose response itself*. Measured (`$.rows`, dom→dom): pooled/baseline SD inflation is present 1.17–1.88 vs other 1.14–2.06, so `zall` differentially penalises whichever channel actually moved — by 0.67× for `sad` and 1.26× for `calm`. A normaliser that shrinks a channel *because* it responded is disqualified on its own terms. `z0` (Glass's Δ) is defensible: the divisor is estimated from the unsteered cell only. But it is still a *noise* normaliser, and the substantive question is a *signal* one.

**Alternatives I computed** (dom→dom, my own scenario-clustered test):

- Cohen's *d* on the pooled within-cell SD: present>other **1/6**, 0 sig, **3 sig reversed** (desperate p=1.0e-3, happy p=6.9e-5, sad p=2.3e-4).
- Within-scenario residual SD at α=0: **1/6**, 0 sig, 4 reversed.
- Rank / van-der-Waerden normal scores (scale- *and* monotone-free — the strongest scale-free test available here): **1/6**, 0 sig, **3 reversed**.

So the raw 4/6 collapses to 1/6 under every scale-free reading I could construct. C2's *substantive* conclusion is more robust than the two normalisers it happens to use.

**Corrected inference.** slope = y(1)−y(0), so the contrast is exactly the mean of 58 paired differences D_i = [p_i(1)−p_i(0)] − [o_i(1)−o_i(0)] — a one-sample mean, clusterable on 29 scenarios. Redone that way (dose-paired + scenario-clustered, 20k boot, plus a t/Wilcoxon on the 29 scenario means), the *counts* are unchanged (point estimates are identical by construction) and significance moves in only one place: **dom→dom raw goes from 0 to 1 significantly reversed** (`happy`). The unblocked bootstrap was, here, roughly self-cancelling — the missing scenario clustering (anti-conservative) offset the missing dose pairing (conservative). Luck, not design.

**Multiplicity.** BH-FDR over the 6 emotions in a family: C2's three reversals survive comfortably (q = 8.5e-3, 2.5e-3, 1.8e-5), and survive even a global BH over all 90 contrasts I ran (q = 0.011, 0.0028, 5.5e-5). C1's four do not: `desperate` falls to q = 0.071, leaving **3 of 6**.

**Power.** MDE (80%, α=.05) on the 29-scenario paired contrast, z0 units: 0.62–0.91 baseline-SD per unit dose. The two dom→dom cells where present still exceeds other under z0 are `afraid` +0.24 (MDE 0.91) and `angry` +0.29 (MDE 0.81) — both far inside the noise floor. "2 of 6 with 0 significant" is two uninformative nulls, not two weak positives.

## 3. Strongest objections

**(a) The design cannot distinguish "B models A" from "the residual stream still contains A."** `a3_dissociation.py:154` runs `pool_spans` over `full` = scenario + B's opener + A's *steered* reply + B's reply, masking to B's reply span only (`:153`). Masking the span does not mask attention: every token of B's reply attends to A's steered text. The probes were trained the same way (`acl_core.py:141-148` — the pool span is A's final utterance *inside its transcript*), so the "other" probe is, by construction, partly a detector of what the interlocutor's visible text conveys. "Other rises more than present" is the *expected* result of directly installing emotion e in A's text, with no representational-coupling content whatsoever. Nothing in either JSON separates this. There is no random-direction control, no orthogonal-direction control, no readout on A's own span, no shuffled-context control, and no leak filter on the A3 generations (`leak()` at `acl_core.py:151-154` is applied to the pool only, never to the steered A replies).

**(b) Probe overlap is real and unreported.** No cross-decode figure exists in the file. From `$.rows` I can measure the overlap on the test distribution: corr(present_e, other_e) at α=0 under `dom` readout = +0.58, +0.13, +0.42, −0.21, −0.09, +0.49; under `logreg` = +0.03…+0.44 (pooled over doses, up to +0.60). Cross-*emotion* overlap is worse: corr(P_desperate, P_afraid) = +0.70. Directions this correlated are not six independent channels.

**(c) "Noisier/lower-norm other probe" is the one alternative that *is* excluded.** Dividing by each channel's own SD is precisely the fix for that, and the reversals get *stronger*, not weaker, under z0/d/rank. Credit where due: (c) is ruled out; (a) and (b) are not touched.

**(d) The logreg result (C1) has a mechanical explanation.** `$.direction_stability.logreg.mean = 0.576` means the DIR-half steering axis and the READ-half present-probe are ~55° apart *in expectation*. The present readout is therefore attenuated toward the axis it is meant to detect, while the other channel reads a contextual echo that no such misalignment attenuates. 0/6 under logreg is what an unstable probe predicts regardless of any coupling story.

**(e) Is 2 reps × 29 scenarios enough?** For C2's three reversals, yes — they survive scenario-level clustering and global FDR. For C1's four, no — one is FDR-fragile. But two deeper problems remain for both: the six contrasts share an identical α=0 arm (§1), so "k of 6" is not k independent findings; and rep-to-rep reliability of the scenario-level present shift is poor (r = +0.36, +0.58, +0.01, +0.38, +0.03, +0.47 for desperate…angry), meaning the two "reps" add almost no independent information for `happy` and `sad`. Everything is one model, one pool, one seed (`$._provenance.seeds = {pool: 0, run: 0}`); no across-pool or across-model variance is anywhere in the interval.

## 4. What would be needed

- **Report the row norms.** ‖C_present[ei]‖ and ‖C_other[ei]‖ are two numbers `fit_direction` already has; not writing them to the JSON is what forces the whole scale-free re-read to be inferential. C2 cannot say "norm artifact" without them.
- **Normalise by signal, not noise.** The defensible unit is the channel's own discriminability in the probe pool: express each dose response as a fraction of the pool's e-vs-not-e separation along that channel (or as an AUC/accuracy shift). That makes present and other commensurable in the only sense the claim needs. z0 is a serviceable fallback; `zall` should be dropped.
- **Give `paired_slope_contrast` a `block=` argument** and pair the draw across doses. The correct statistic is a one-sample cluster bootstrap on D_i.
- **Controls that separate (a)/(b):** a matched random and an orthogonal steering direction; the cross-decode matrix (present-probe accuracy on other-labels and vice versa) on the READ half; the readout applied to A's span; a context-ablated readout (B's reply re-encoded without A's steered turn); leak-word filtering of the steered A replies.
- **Break the shared baseline** (make α=0 generation depend on emotion), pre-register the family, report FDR.
- **For C3:** report stability at the n actually used (2126/half), not 600; report cross-*class* cosine alongside within-class, since a dom estimator that returns a near-common axis for all six emotions scores ~0.9 while carrying little class-specific signal; and report a decoding-accuracy curve next to the cosine, since stability is not validity.
- **Drop α=0.5 or add a fourth dose.** As built it costs a third of the compute and enters no result.

## 5. Verdicts

**C1 — SUPPORTED-WITH-CAVEATS.** The counts reproduce exactly from `$.rows` (0/6, 4 significant reversals; `$.results["logreg->logreg"].by_emotion`), but "significant" comes from an unblocked, dose-unpaired bootstrap (`acl_core.py:994-1016`), and under a scenario-clustered test with BH-FDR it is 3 of 6, not 4.

**C2 — SUPPORTED-WITH-CAVEATS.** Every number is exactly reproducible and the collapse survives Cohen's *d*, within-scenario SD and rank normalisation as well — but "norm artifact" is not established (row norms are absent; only a 1.65–2.40× *score-SD* ratio is observable), `zall` is a biased normaliser that penalises the responding channel, and the "2 of 6" that remain are both far below the design's MDE.

**C3 — SUPPORTED as stated, but not relevant as used.** `$.direction_stability` gives dom 0.9151 [0.9112, 0.9188] vs logreg 0.5763, 10 seeds, `space="raw"`, `n_per_half=600` — exactly as worded. I cannot verify it (the activations are not in the file), it is not the stability of the n=2126 probes A3 actually used, and a high within-class split-half cosine is evidence of estimator stability only, not of class-specific validity.

## Reconciliation
No prior audit covers A3. The critique reproduced every count from the per-sample rows and then went further; the record was corrected as follows.
- **The conclusion holds; the wording did not.** Under Cohen's d, within-scenario SD, and rank normalisation the dom→dom count is 1 of 6 with 0 significant and 3 reversed — stronger than my z0 reading. But "norm artifact" is not established: the row norms are not in the file, only a 1.65–2.4× score-SD ratio (I had quoted the 2.4 maximum as typical). Registry and script docstring reworded.
- **`zall` is a biased normaliser** — the pooled SD is inflated by the dose response, so it penalises the channel that moved. Dropped from the claim text; kept in the script output for the record, labelled.
- **The paired contrast is neither scenario-blocked nor dose-paired** (`acl_core.paired_slope_contrast`), while the provenance stamp claims a blocked bootstrap for it. Redone correctly the counts are unchanged; dom→dom raw gains one significant reversal (happy), and the published-configuration reversals are 3 of 6 after clustering and FDR, not 4. Registry text corrected.
- **The six emotions share one α = 0 baseline** (unsteered generation seeded by rep, not emotion), so "k of 6" is not k independent findings. The same holds for B1. Added to the plan's to-dos.
- **The design cannot separate "B models A" from "B's residual stream attends to A's steered text"**: no random/orthogonal steering control, no cross-decode figure, no readout on A's span, no context-ablated readout, no leak filter on the steered replies. These are all in the plan's §6 controls register and were not in this driver. Added as a required follow-up before any positive reading of "other > present".
- Stability is reported at n = 600/half, not at the 2126/half the probes actually used.
- The two "present > other" survivors under z0 (afraid +0.24, angry +0.29) are far below the design's MDE (0.6–0.9 baseline-SD per unit dose) — uninformative nulls, not weak positives.
Verdicts: C1 SWC · C2 SWC · C3 SUPPORTED as stated, not relevant as used.

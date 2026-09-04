# 06 · CMI — A→B coupling in nats on the real channel

**File:** `results/information/cmi_passed_qwen36-27b.json` · **Script:** `src/information/cmi_passed.py` (imports `src/core/coupling_e2.py`) · **Model:** Qwen/Qwen3.6-27B · **Status in registry:** RETRACTED-CLAIM (`cmi.coupling_nats`), AUDIT 2

## What was run
A is steered along each of six emotion directions at β = 0.3 across 29 scenarios (6 × 29 = 174 episodes). A's and B's replies are projected onto a 6-d emotion basis at the focus layer. A Gaussian information estimate is computed between A's 6-d projection and B's, and compared to a null in which A's rows are permuted across the whole pooled set (20 permutations).

## What the file shows
`cmi_inject_nats` 0.264 · `cmi_scramble_null_nats` 0.067 · `null_sd` 0.017 · `coupling_nats` 0.197 · `z_score` 11.5 · `n` 174 · `beta` 0.3. No per-emotion breakdown, no permutation count, no seed, no CI, no β = 0 arm.

## What can be inferred
- A's and B's emotion projections are linearly dependent across the pooled 174 episodes: the plug-in estimate exceeds the full-pool permutation floor by 0.197. [supported by file]
- The pooled set is six emotion blocks; any per-block mean shared by A and B contributes to the statistic, and the permutation destroys block alignment, so the file cannot separate "B's episode depends on A's episode" from "both carry the emotion label". [needs: within-block permutation or per-emotion estimates]
- "z = 11.5" is a ratio to the sd of 20 permutation draws, not a tail probability. [supported by file, given the script's permutation count]
- Whether shared scenario context, rather than A's state, drives the dependence is not testable here. [needs: β = 0 arm or conditioning on scenario]

## Status
Registry: RETRACTED-CLAIM. AUDIT 2: pools six emotions into one matrix and permutes across emotion blocks, so the statistic mostly recovers which emotion this was. RESEARCH_PLAN B8: adopt or cut.

### Independent critique (blind: saw only the script and the JSON)

**1. What the code computes.**
`gaussian_cmi` (`src/information/cmi_passed.py:40-47`) advertises `I(X;Y|Z)`, but the only call site is line 154, `gaussian_cmi(eA, eB)` — `Z` defaults to `None`, so execution takes the branch at lines 44-45: **unconditional** Gaussian MI, `0.5(logdet ΣX + logdet ΣY − logdet ΣXY)`. Nothing is conditioned on. "CMI" in the filename, docstring, and JSON keys is a misnomer.

X = `eA`, the 6-d projection (`proj6`, :71-73) of A's residual at layer `focus`, pooled over A's own reply span, while A is steered along `Cp[e]/sd` (:63-65, :85). Y = `eB`, the same projection of B's reply span. The measured B state is a *re-encode of B's generated text with no injection active* (`INJ` cleared at :124; readout loop :127-140), so the channel is A-activation → B-generation → B-text, not an algebraic leak.

Pooling (:145-150): 6 emotions × 29 scenarios (`coupling_e2.py:29-88`) = 174 rows, stacked with `np.vstack`. Emotion is a *block* variable; scenario is paired by index across blocks.

Null (:155): `eA[rng.permutation(len(eA))]` — a single permutation over the **entire pooled 174**, repeated 20 times. It destroys, simultaneously, (a) episode pairing, (b) emotion-block alignment, (c) scenario alignment. `cmi_null`/`null_sd` = mean/`np.std` (ddof=0) of those 20 (:156); `coupling = inject − null` (:157); `z = coupling/(null_sd+1e-9)` (:161).

**2. JSON.** `.n = 174` (= 6×29, consistent). `.beta = 0.3`. `.cmi_inject_nats = 0.26384469334378435`, `.cmi_scramble_null_nats = 0.06660250356001882`, `.null_sd = 0.017139567946755557`, `.coupling_nats = 0.19724218978376554`, `.z_score = 11.508001770435575`. The arithmetic is self-consistent. The JSON records **no** permutation count (20 is only knowable from :155), no seed, no CI, no per-emotion or per-scenario breakdown, and no no-injection control.

**3. Objections.**

- *The null is the wrong null.* B's prompt contains the same scenario text A saw (`:110-113`: `setting` + `bopen` + `NEUTRAL_A`). Scenario is therefore a common cause of `eA` and `eB` with **zero** information flow. A full-pool permutation destroys that alignment, so scenario-driven covariance is counted as "coupling." There is no β=0 arm; `scramble` is not a no-channel control.
- *Block means suffice.* The estimator uses second moments only. If `eA` were constant-per-emotion and `eB` depended only on the emotion label, the statistic would be identical. So yes — 0.197 can arise with exactly zero episode-level coupling. The permutation tests "is emotion (and scenario) recoverable in both," not "does A's episode predict B's episode."
- *Nats are not interpretable.* `Cp` rows are difference-of-means directions (`coupling_e2.py:186-196, 202-203`), mutually near-dependent, so the 6×6 covariances are near-singular and `RIDGE = 1e-3` (:24, :43) materially sets the logdets — the value scales with an arbitrary regularizer. A difference of two plug-in estimates is not itself an information quantity. Reading it back: 0.264 nats spread over 6 canonical correlations implies ρ≈0.29 — modest linear dependence, 11% of the log 6 = 1.79 nats a perfect emotion channel would carry.
- *z is not a z.* 20 permutations bound the attainable p at ~1/21; the null of a log-det statistic is skewed, not Gaussian; `ddof=0` inflates z. "z = 11.5" is a ratio, not a tail probability.

**4. To make the claim.** Condition (`Z`) on scenario one-hot or on an injection-free `eB` baseline; permute *within* emotion×scenario blocks to test episode-level coupling; run a β=0 arm as the true floor; ≥1000 permutations reporting a rank-based p; bootstrap CI over scenarios; a ridge/dimension sensitivity sweep; a nonparametric (KSG) cross-check. Nearest defensible claim now: *steering A shifts B's emotion projection in a direction that is linearly predictable from A's, with pooled Gaussian dependence of ~0.20 nats, magnitude not separated from shared-scenario context.*

**5. Verdict: NOT SUPPORTED** — the statistic is unconditional Gaussian MI on 6-cluster pooled data with a null that destroys a known shared-context confound and no β=0 control, so 0.197 nats is not attributable to an A→B channel and "z = 11.5" from 20 permutations is not a significance level.

## Reconciliation
Agrees with AUDIT 2 and sharpens it in three ways not in the audit: (1) the conditioning argument is never passed (`Z=None` at the only call site), so the quantity is unconditional Gaussian MI and the name "CMI" is a misnomer; (2) B's prompt contains the same scenario text A saw, so scenario is a common cause that the full-pool permutation destroys — there is no β = 0 arm to floor it; (3) the "z" rests on 20 permutations and a ridge-sensitive log-det. Verdict NOT SUPPORTED is consistent with the registry's retraction.

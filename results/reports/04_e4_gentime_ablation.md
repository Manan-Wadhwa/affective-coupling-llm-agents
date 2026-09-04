# 04 · E4 — generation-time affect ablation (original run)

**File:** `results/representation/e4gentime_qwen36-27b.json` · **Scripts:** `src/representation/coupling_e4_gentime.py`, `src/core/coupling_e2.py` · **Model:** Qwen/Qwen3.6-27B · **Registry:** UNREGISTERED · superseded by rev-3 B1 · **AUDIT:** 9, 18

## What was run
As E3, but the ablation hooks are live on decoder layers 12–41 at A's token positions *during B's prefill*, so B is regenerated under the intervention. Directions from per-layer logistic fits (decoder_n = 266). One random unit direction per layer, one seed. 29 scenarios, one rep. No manipulation check on A's side; no per-item values.

## What the file shows
`slopes.<e>.B_none / B_emo / B_rand`: desperate 0.94 / 0.70 / 1.17; afraid 1.87 / 1.45 / 1.64; sad 1.29 / 0.92 / 1.34; angry 1.02 / 0.61 / 0.84; happy −0.23 / −0.41 / −0.51; calm 0.06 / 0.35 / 0.27. `contagion_lost_gentime` for calm is −5.26 (division by a near-zero baseline).

## What can be inferred
- For the four emotions with a positive baseline slope, the ablated arm's slope is below both the none and random arms (mean retention 0.71 vs 0.99). The file contains no test, so this is a point-estimate pattern only. [supported by file]
- The published reading "indistinguishable from random" was never computed and the point estimates trend the other way. [supported by file]
- Whether the ablation removed anything from A's representation is unknown: no manipulation check, no direction-stability check. [needs: B1]
- A's tokens and 34 of 64 layers are untouched, so B can recompute affect from A's words; a non-blocking outcome is expected by construction and does not by itself locate the channel. [supported by code]

## Status
Not in the registry. RESEARCH_PLAN §1.3 judged it uninterpretable (direction split-half 0.394). B1 reruns it with a stable direction, a generation-side check, and recorded removed norm.

### Independent critique (blind: saw only the scripts and the JSON)

Scope note: I did not read `coupling_e0.py`, so `EMOTIONS`, `NAMES`, `dialogue_prompt`, `gen`, `leak`, `parse_final_A`, `pool_final` and `log_responses` are unverified black boxes. All line numbers are `src/representation/coupling_e4_gentime.py` unless prefixed `e2:` (= `src/core/coupling_e2.py`). All JSON paths are in `results/representation/e4gentime_qwen36-27b.json`.

---

## C1 — "gen-time ablation does not block contagion; indistinguishable from a norm-matched random control"

**1. What the code computes.**

*Direction estimator.* `train_layer_decoders` (46-68) fits `LogisticRegression(max_iter=3000, C=0.5)` per layer (65) on pooled final-A-utterance features (61-62). `$.decoder_n` = **266** dialogues, hidden size ~5120, ~44 items/class. E4 does **not** call `E2.train_decoders`; it imports only `E2.Steer` (101), `E2.gen_steered` (119), `E2.SCENARIOS` (109). This matters: e2:177-185 is the authors' own written finding that this exact logreg fit "does not converge — two independent fits of the SAME direction agree at cos 0.41 (27B, n=600/half)… an unstable fit makes every downstream number non-reproducible," and e2:202-203 makes difference-of-means the default *there*. E4 uses the repudiated estimator, at n=266 — less than half the n at which cos was 0.41. **No reproducibility check exists in E4.** The same unstable fit supplies the ablation direction (95-96), the steering vector (104-105), and the measurement probe (157).

*When ablation is live.* Hooks are registered permanently on decoder layers 12-41 (86-87, 106-107; `$.focus`=43, `$.n_abl_layers`=30 ⇒ nL=64, lo=13), gated by `ABL["mode"]` (27), applied only when `h.shape[1] == masks.shape[1]` i.e. B's prefill (30), on A's-message token positions only (36, 141-143), reset to none after each `generate` (149). Layers **0-11 and 42-63 are never ablated**, and A's token identities are untouched (`gen_A` runs with ABL "none", 119) — so A's *text* is identical in all three arms.

*Manipulation check.* **None.** `present_e` (155-175) is called only on B's replies (195-197), span = B's utterance (162), in an un-ablated forward pass (149 precedes 168). Nothing reads A's span under ablation; nothing logs `‖proj‖` (35-36); a failed span match silently disables ablation for that item ((-1,-1) at 142) with no counter.

*Random control.* One unit-norm Gaussian per layer, single draw, `rgen` seeded at 90, drawn at 97-100 outside the emotion loop — shared across all 6 emotions and all alphas. "Norm-matched" means *the direction is unit norm* (96 vs 100), not that equal energy is removed. In 5120 dims a random unit direction removes ≈1.4% of the residual norm by construction, so this control is built to do nothing.

*"Indistinguishable."* Never computed. There is no test, CI, SD, or emo-vs-rand comparison anywhere; only `contagion_lost_gentime = 1 - B_emo/B_none` (209).

**2. Does the JSON support it?** Slopes (`$.slopes.<e>.B_none/B_emo/B_rand`): desperate 0.942 / 0.702 / 1.170; afraid 1.871 / 1.452 / 1.635; sad 1.292 / 0.919 / 1.344; angry 1.021 / 0.609 / 0.841; happy −0.229 / −0.409 / −0.508; calm 0.056 / 0.349 / 0.268. In **4/4 emotions with a real baseline contagion slope, emo < rand**; mean retained fraction emo/none = **0.71** vs rand/none = **0.99**. That is a ~29-point gap in the *opposite* direction to the claim. n = 29 scenarios/cell (e2:29-88), one sample per cell, temperature 0.9 with **no torch seeding anywhere**; per-item scores are discarded at 195-197; `$.rows` records no n, SD, or reps.

**3. Objections.** (a) "Indistinguishable" is asserted, not tested, and the point estimates trend against it (4/4 sign, p≈0.06 one-sided). (b) The file contains its own noise floor: at α=0 the emo−none deltas are +0.014/+0.078/+0.094/+0.023/+0.120/+0.015 (sd 0.046), while at α=1.0 they are −0.226/−0.102/−0.324/+0.316/−0.253/−0.398 — mostly 4-8× the α=0 spread, i.e. the ablation *is* doing something dose-dependently. (c) happy (`B_none` = −0.229, wrong sign) and calm (`B_none` = 0.056, giving the absurd `contagion_lost_gentime` = **−5.264** through the 1e-6 guard at 209) have no contagion to block; they pad the "no difference" impression. (d) With A's text and 34 of 64 layers untouched, the model can re-derive affect from token identities downstream of layer 41 — non-blocking is close to guaranteed by construction.

**4. What would be needed.** Per-scenario paired scores retained; ≥3-5 reseeded reps per cell; a bootstrap/paired test of (none−emo) vs (none−rand) with a stated equivalence margin; ≥20 resampled random directions rather than one; energy-matched rather than norm-matched controls; the dom estimator plus a split-half cosine for every ablated layer.

**5. Verdict: NOT SUPPORTED.** "Indistinguishable" is never tested and the recorded slopes show emo below rand in every emotion that has a baseline effect (mean retention 0.71 vs 0.99).

---

## C2 — "the transmitted affect is not carried by the filterable emotion direction"

**1./3. What the code can and cannot license.** The experiment has no positive control, so "the direction was removed and affect still flowed" cannot be separated from "the direction was never removed."

- *Estimator noise.* The ablated vectors come from the estimator e2:177-185 documents as reproducing at cos ≈0.41 at n=600; here `$.decoder_n` = 266 (65, 88). A rank-1 projection along a vector that shares ~40% cosine with the "true" direction removes ~16% of the relevant variance. Nothing in E4 measures this — no split-half, no per-layer cosine.
- *Geometry.* Projecting along the readout functional `coef/sd` (95-96, 35-36) zeroes *that layer's probe variance*, nothing more; it is not shown to remove the feature, and no probe is re-read after ablation.
- *Coverage.* Only decoder layers 12-41 (86-87). Layers 0-11 feed un-ablated A-content into KV; layers 42-63 are free to rebuild it; decode steps are explicitly skipped (30). The input tokens themselves — A's emotional words — are never altered (119).
- *No manipulation check on either side.* No A-side readout under ablation (195-197 measure B only), no check that A's text or A's readable affect changed, no `‖proj‖`/`h·d` statistic, no count of items where the span match failed (141-142).

Taken together, the single most likely explanation of `$.slopes.*.B_rand ≈ B_none` is that the random projection removed ~1.4% of the norm and did nothing; the same file gives no way to exclude that `B_emo` differs from `B_none` by only ~29% for the same reason (a partially-wrong direction removing a partial component), rather than because affect travels by some other route. The one internal signal available — the α-dependence of emo−none against the α=0 floor (see C1.3b) — points toward the direction carrying *some* of it.

**2. JSON.** Nothing in the file speaks to the mechanism. The only mechanism-relevant fields are `$.focus` = 43, `$.n_abl_layers` = 30, `$.decoder_n` = 266, `$.rms` = 131.89. There is no recorded quantity describing what the ablation did to any representation.

**4. What would be needed.** (i) A positive control: recompute the present-e probe over A's span *during the ablated prefill* and show it is driven to chance; (ii) ablation across all layers up to the readout, or an ablated-then-rewritten text arm, so the model cannot recompute from tokens; (iii) split-half cosine ≥ ~0.8 for each ablated direction (dom, per e2:202-203) before any null is interpretable; (iv) energy-matched control directions and a stated equivalence bound. Nearest defensible claim from what exists: *"A rank-1 projection of a logreg-estimated emotion direction over layers 13-42 of B's prefill of A's message reduces the contagion dose-response by roughly 30% on average across the four emotions with a positive baseline slope, versus ~0% for a single random unit direction; with n=1 sample per cell and no reps, neither the reduction nor its difference from the control is statistically established."*

**5. Verdict: UNDETERMINABLE.** No manipulation check, no direction-stability check, and 34/64 layers plus all input tokens untouched — the design cannot distinguish "affect isn't in the direction" from "the direction was never effectively removed."

## Reconciliation
Agrees with RESEARCH_PLAN §1.3 that the result is uninterpretable rather than null. Two additions matter for reading B1: (i) the original point estimates show *partial blocking* (emo below rand in 4/4 emotions with baseline contagion), not indistinguishability — the plan's premise that the published run showed "random" is itself unverified; (ii) because A's tokens and layers above the ablation window are untouched, B can rebuild affect from token identities, so B1's "does not block" outcome needs a token-level arm before it says anything about *where* affect travels. Verdicts: C1 NOT SUPPORTED, C2 UNDETERMINABLE.

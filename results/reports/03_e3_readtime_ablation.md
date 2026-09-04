# 03 · E3 — read-time affect ablation

**File:** `results/representation/e3v2_qwen36-27b.json` · **Scripts:** `src/representation/coupling_e3_ablate.py`, `src/core/coupling_e2.py` · **Model:** Qwen/Qwen3.6-27B · **Registry:** `e3.affect_removed`, `e3.text_channel_intact` RETRACTED-CLAIM · **AUDIT:** 9, 18

## What was run
A's line is generated under steering; B's reply is generated from the unmodified text. Then, in a single scoring pass over the finished transcript, the emotion direction (per-layer logistic fits, decoder_n = 261) is projected out of A's token positions at hidden states 13–42, and the present-emotion probe at layer 43 is read on A's span and on B's span. Arms: none / emo / random (one Gaussian unit direction, seed 0). 29 scenarios, one rep, α ∈ {0, 0.5, 1.0}.

## What the file shows
`slopes.<e>.A_manip_removed` ranges 0.81 (desperate) to 0.95 (calm); `B_contagion_lost` ranges 0.05 (calm) to 0.14 (happy). Random arm: `A_rand/A_none` within 0.98–1.01 and `B_rand/B_none` within 0.98–1.02 for all six. 18 rows of means; no per-item values, no CI, no seed.

## What can be inferred
- Projecting the direction out of A's tokens during a scoring pass lowers the probe reading on A's span by 81–95%, relative to the α-difference. [supported by file]
- The probe reading on B's already-written tokens changes by 5–14% under the same edit. [supported by file]
- B never generates under the ablation, so nothing here measures whether B's *reply* would change; "the channel cannot be severed" is not a statement the design can make. [supported by code: RESEARCH_PLAN §1.1]
- The random arm shows an arbitrary direction does nothing; it does not show the effect is specific to emotion versus any data-derived direction. [supported by file]
- Uncertainty on every ratio is unknown. [needs: per-item scores, reps]

## Status
Registry: both claims RETRACTED (numbers resolve; sentence does not follow). Superseded by B1.

### Independent critique (blind: saw only the scripts and the JSON)

**Shared trace (applies to all three claims).** Per (emotion, alpha) cell, `gen_A_B` (`coupling_e3_ablate.py:138-160`) generates A's message at :146 (`E2.gen_steered`, steering vector on) and B's reply at :154 (`E2.gen_steered(..., None)`). At both moments `ABL["mode"]=="none"` (initialized :23, reset :128), so `make_hook` returns `o` unmodified (:28). **No text is ever generated under ablation.** Ablation happens only inside `measure`, on the finished transcript: masks/dirs are set at :125, one forward pass at :127, mode reset at :128. The hook (:26-37) projects the direction out only at A-span positions (`torch.where(ABL["masks"]...)`, :35; `Am` built from `A_spans` at :120-124). Hooks are on decoder layers `abl_dec` (:107-108), i.e. hidden states 13-42 (`focus=43`, `n_abl_layers=30`, json `focus`, `n_abl_layers`). Readout is `hidden_states[focus]` = 43 (:127) — **exactly one transformer block downstream of the last ablated layer**. A- and B-scores come from the same pass (:129-135). Directions and probe are per-layer `LogisticRegression` fits on 261 pooled examples (:45-67, :64; json `decoder_n: 261`), n=29 scenarios/cell (`E2.SCENARIOS`, 29 tuples). Verified numerically: with `ALPHAS=[0,0.5,1.0]`, `Ac=[-0.5,0,0.5]`, so `slope` (:188-189) reduces exactly to `y(α=1.0) − y(α=0.0)`; every one of the 36 reported slopes matches that difference to 1e-9. **The α=0.5 row carries zero weight.** No seeding of sampling anywhere (`do_sample=True`, `coupling_e2.py:120`); no per-item scores, SDs, CIs, or reps in the JSON (only means, :179-182).

---

### C1 — "81–95% of A's affect is removed by the read-time ablation"

**1. What is computed.** `A_manip_removed = 1 − A_emo_slope/A_none_slope` (:194). Numerator/denominator are α=1.0 minus α=0.0 differences of `mean(measure(...)["A"])` under `mode="emo"` (:176) and `mode="none"` (:175) — different forward passes over *identical* transcripts, which is fine. The pooled A-readout mask at :131 is `amasks[b]` — **the same token mask used to apply the ablation** at :123-124. Layers 13-42 ablated; readout at 43.

**2. JSON.** Range confirmed: `slopes.desperate.A_manip_removed = 0.8094`, `slopes.calm.A_manip_removed = 0.9476`. n=29/cell, 1 rep, no CI.

**3. Objections.**
(a) *Near-tautology.* The probe direction is `decs[43]["coef"][ei]/sd` (:114); the last ablated direction is `decs[42]["coef"][ei]/sd` (:96-97) — adjacent-layer fits on the *same* 261 pooled features, read out at the *same* positions one block later. The script never reports `cos(emo_dirs[e][42], probe_dir)`. The docstring's defence ("ablation is below focus, so a drop at focus is non-trivial", :6) collapses when "below" means one layer.
(b) *It is not "affect."* It is a linear readout. At α=0, ablation moves the score in *inconsistent directions* — desperate `rows[0]`: `A_none −1.2103 → A_emo +0.3027`; sad `rows[12]`: `−0.2789 → −2.0687`. A −1.79 shift on an unsteered baseline is not "removing affect"; it is an uncontrolled intercept shift, and the slope ratio hides it.
(c) *Estimator disowned by the authors.* `coupling_e2.py:176-185` documents that this exact `LogisticRegression` fit "does not converge", two fits of the same direction agreeing at cos 0.41 at n=600/half, and switches E2's default to difference-of-means. `coupling_e3_ablate.py:64` still uses `LogisticRegression(max_iter=3000, C=0.5)` at n=261 — below the n the authors called unstable. The fix never reaches E3.
(d) *Two data points.* The slope is `y(1.0) − y(0.0)`, each a mean of 29 single samples; no SD is stored, so the ratio's uncertainty is unverifiable from the JSON.
(e) *Silent condition-mixing.* At `truncation=True, max_length=768` (:118), an over-long transcript yields an empty span mask and :133 falls back to `enc["attention_mask"][b].bool()` — pooling the whole sequence. Frequency unknowable; transcripts are not saved.

**4. What would be needed.** Report cosine between ablated directions and the focus probe; ablate strictly at layers ≤ ~0.5·nL and re-check; use the difference-of-means estimator with a held-out probe (fit direction on split 1, probe on split 2) so removal is not measured with a co-fitted vector; report per-item paired scores with CIs; validate "affect removed" behaviourally (does a continuation generated under ablation read as less emotional to an independent rater?).

**5. Verdict: SUPPORTED-WITH-CAVEATS as an arithmetic statement about the probe; NOT SUPPORTED as a claim about "A's affect."** The number is reproducible from the JSON, but a co-fitted, one-layer-downstream probe read at the very positions that were projected makes the drop close to forced, and the α=0 intercept shifts contradict a clean-removal reading.

---

### C2 — "the text channel cannot be severed: contagion stays 86–95% intact"

**1. What is computed.** `B_contagion_lost = 1 − B_emo_slope/B_none_slope` (:195), from `mean(measure(...)["B"])` at :181. The B-mask is built at :130 from `B_spans`; **B's positions are never in the ablation mask** (:120-124 uses A-spans only). B's reply text is byte-identical across `none`/`emo`/`random` — it was generated once, unablated, at :154, and `measure` is called three times on the same `tr` (:175-177).

**2. JSON.** `slopes.happy.B_contagion_lost = 0.13767` → 86.2% intact; `slopes.calm.B_contagion_lost = 0.04692` → 95.3%. Range as worded checks out. n=29, 1 rep, no CI.

**3. Objections.**
(a) **"Cannot be severed" does not follow from what was measured.** Severing a channel means B *behaves* differently — B must generate under ablation. B never does (:154 runs with mode "none"). What was measured is whether a probe reading of already-written B tokens shifts when A's tokens are edited at read time. A fully severed channel and an unseverable one are indistinguishable in this design, because the outcome variable (B's text) is frozen by construction. The docstring even states the design choice: "B reply fixed (generated once)" (:13). The claim is generative; the measurement is post-hoc and correlational.
(b) **The A-drops/B-survives contrast is predicted by the masking alone.** A's pooled readout dropped because A's own residuals were projected; B's did not because B's were not touched. The mean over B's positions is dominated by B's own residual content; cross-position leakage into B via attention at layers ≤42 is second-order. The script computes no ceiling — it never ablates at B's positions or at all positions — so "86-95% intact" has no denominator establishing that this readout is movable at all. Without that, 95% intact is uninformative.
(c) **Unstable ratio with an unpaired denominator.** Numerator `B_none − B_emo` is paired (same transcripts, same B text) and small: happy `rows[3..5]` gives `B_none` 2.1883→2.6853 (slope 0.4970) and `B_emo` 2.1885→2.6171 (slope 0.4286). The *denominator* is a two-point α-difference across independently sampled transcripts, so it carries full generation noise. A 0.05 shift in either endpoint moves "intact" by ~10 points; no SD is stored to bound it.
(d) **Non-monotone dose-response, hidden.** `B_none` is non-monotone in α for happy (2.188, 3.277, 2.685), calm (1.687, 2.391, 2.388) and sad (−1.370, 0.498, 0.311); `A_none` for happy (1.639, 5.463, 4.266) and afraid (−0.919, 2.691, 2.092). Because the slope estimator discards α=0.5, the summary cannot show this. Four of six emotions have a non-monotone arm.
(e) **"Contagion" is not established in E3.** E2 makes the present-vs-other dissociation load-bearing (`coupling_e2.py:6-9`, `Cp`/`Co` at :200-203) precisely because a present-e score at B's positions could be B *modelling A* as e. `train_layer_decoders` fits present only (`yp` at :61); there is no `other` decoder anywhere in E3. So the quantity called "contagion" here has not been separated from social modelling.

**4. What would be needed.** Regenerate B's reply *with the hook live* (set `ABL` before `gen_steered`, handling the KV cache and the growing A-mask) and measure B's affect and behaviour in the ablated-generation condition — that is the only design in which "severed" is meaningful. Add the ceiling condition (ablate B's positions, and all positions) to calibrate what a severed readout looks like. Add E2's `other` decoder. Report paired per-item deltas with bootstrap CIs over the 29 scenarios and ≥3 reps. Nearest defensible claim as-is: *"editing the emotion direction out of A's token activations at read time leaves the emotion readout on B's already-generated tokens essentially unchanged (4-14% reduction)"* — a statement about locality of the probe, not about a channel.

**5. Verdict: NOT SUPPORTED as worded.** The severing manipulation was never applied to B's generation; the observed asymmetry is what the A-only position mask predicts a priori, and no ceiling condition exists to make "86-95% intact" interpretable.

---

### C3 — "a random-direction control shows the effect is specific to the emotion direction"

**1. What is computed.** `rand_dirs` (:98-101): one i.i.d. Gaussian unit vector per layer from `rgen = np.random.default_rng(0)` (:91), shared across all emotions, applied by the same hook path (`mode="random"`, :177) at the same A-only positions and the same layers 13-42.

**2. JSON.** The random arm is indistinguishable from no ablation. `slopes.desperate.A_none = 4.4218` vs `A_rand = 4.4070` (ratio 0.997); across all six emotions `A_rand/A_none` ∈ [0.980, 1.010] and `B_rand/B_none` ∈ [0.979, 1.015]. No CI, no second seed, n=29.

**3. Objections.**
(a) **A single unstructured Gaussian in ~5k dimensions is the weakest possible control** — it is near-orthogonal to every structured direction by construction, so it removes essentially no variance from any probe. It licenses only "randomly-chosen directions do nothing," which was never in doubt. It cannot distinguish the emotion direction from any other *data-derived* direction: another emotion's direction, a speaker/formatting direction, the top residual PC, or a direction fit to shuffled labels.
(b) **Not magnitude-matched.** The emo arm removes `(h·d)d` where `d` is aligned with a large component of the activation; the random arm removes a projection of roughly `‖h‖/√H`. The two conditions perturb the residual by wildly different amounts, so any difference is confounded with perturbation size. The script never records `‖proj‖` in either arm (nothing logged around :34).
(c) **Shared-noise circularity.** Given the estimator instability documented at `coupling_e2.py:176-185`, a plausible alternative account is that ablation direction (layer 42) and probe (layer 43) share fitting noise from the same 261 examples, and the "specificity" is co-adaptation of two fits on one dataset rather than an emotion-specific subspace. A random vector cannot detect this; a shuffled-label direction fit on the same data would.
(d) **One seed.** `default_rng(0)` gives a single random draw for the whole experiment; no variability estimate for the control arm.
(e) The control is applied only to the read-time manipulation, so whatever it establishes inherits C2's limitation — it says nothing about direction-specificity of a generative effect.

**4. What would be needed.** Add (i) shuffled-label directions fit by the identical pipeline on the same data — the matched-noise control; (ii) other-emotion directions (does ablating "calm" reduce the "angry" readout?); (iii) a norm-matched random direction, or report removed-projection magnitude per arm; (iv) ≥10 random seeds with a distribution rather than one draw. Nearest defensible claim now: *"an arbitrary unit direction, ablated identically, changes nothing (≤2%), so the observed A-readout drop is not an artifact of perturbing the residual stream per se."*

**5. Verdict: SUPPORTED-WITH-CAVEATS, but far weaker than "specific to the emotion direction."** The random arm rules out only a generic-perturbation artifact; with no shuffled-label, other-emotion, or norm-matched control and one seed, "specific to the emotion direction" is not what was tested.

---

### Cross-cutting, not verifiable from the permitted files

- `coupling_e0` (`EMOTIONS`, `NAMES`, `TOPICS`, `gen`, `leak`, `parse_final_A`, `pool_final`) was not read per instructions. Two consequences I could not check: whether `E0.EMOTIONS` order matches sklearn's sorted `classes_` (if any class is absent after the `leak`/`parse` filter at :57-59, `coef_[ei]` at :96/:105/:114 indexes the wrong row), and whether the ~30-60% of jobs discarded by that filter biases class balance in the 261 training dialogues.
- No transcripts, per-item scores, spans, or truncation counts are persisted (`json.dump` at :201-203 stores means only), so none of the above can be re-derived from the result file; re-running cannot reproduce it either, since generation is sampled with no seed.

## Reconciliation
Agrees with AUDIT 9 and RESEARCH_PLAN §1.1 (B frozen; readout one block above the last ablated layer). New: (i) at α = 0 the ablation shifts the A-readout in inconsistent directions (an intercept shift the ratio hides); (ii) no ceiling condition exists (ablating B's or all positions), so "86–95% intact" has no denominator; (iii) the α = 0.5 arm is non-monotone for four emotions and invisible to the endpoint slope; (iv) a 768-token truncation fallback silently pools the whole sequence with no counter. Verdicts: C1 caveated-as-arithmetic / not-supported-as-affect, C2 NOT SUPPORTED, C3 SUPPORTED-WITH-CAVEATS.

# 05 · E5 — activation-passing channel

**Files:** `results/representation/e5actpass_qwen36-b02.json` (β 0.2), `e5actpass_qwen36-b03.json` (β 0.3), `e5actpass_qwen36-27b.json` (β 0.5) · **Scripts:** `src/representation/coupling_e5_actpass.py`, `src/core/coupling_e2.py` · **Model:** Qwen/Qwen3.6-27B · **Registry:** `e5.beta03_all6`, `e5.emotion_causally_responsible` RETRACTED-CLAIM (β 0.3 file); the other two UNREGISTERED · **AUDIT:** 1, 12, 15, 18

## What was run
A is steered to emotion e; A's reply is re-encoded unsteered and its pooled layer-43 activation `vA` is injected into B's layer 42 at every position during B's generation, scaled by β, with B given a neutral A line. Conditions: none / full / ablated (probe direction removed from `vA`) / scramble (rows of `vA` permuted across the 29 scenarios *within the same emotion*). Readout: B's reply projected on the same probe. 29 scenarios, one draw per condition, decoder refit per run (decoder_n 269 / 266 / 257).

## What the file shows
`results.<e>.transmit(full-none)` positive for 3/6 at β 0.2, **6/6 at β 0.3**, 5/6 at β 0.5. `emo_specific(full-ablated)` positive 3/6, 4/6, 3/6 (means +0.00, +0.10, −0.03). `vs_scramble(full-scramble)` positive 5/6, 3/6, 4/6 (means +0.10, −0.04, +0.07). The `none` baseline, which does not depend on β, moves by up to 1.98 across the three runs (calm). No CI anywhere.

## What can be inferred
- At β = 0.3 all six `transmit` values are positive; at the two other β values they are not, and transmit is non-monotone in β for four emotions. [supported by file]
- The scramble permutes episodes within one emotion, so it preserves the emotion component and cannot test emotion-specificity. [supported by code]
- The one metric that removes the probe direction (`emo_specific`) is positive for 4/6 at β 0.3 with mean +0.10, smaller than the run-to-run drift of the `none` baseline. [supported by file]
- "Causally responsible" is not testable from these files. [needs: off-emotion injection arm, CIs, frozen decoder across β]

## Status
Registry: both β 0.3 claims RETRACTED (claim `e5.emotion_causally_responsible` said 6, file gives 3). RESEARCH_PLAN B4 rebuilds the scramble.

### Independent critique (blind: saw only the scripts and the JSONs)

## Setup the two claims share

**What is passed.** Per emotion `e`, `capture_vA` (`coupling_e5_actpass.py:92-128`) steers A at layer `focus-1=42` (`:81`, `:107`, magnitude `1.0*rms*unit_dir`, α hard-coded 1.0), generates A's reply, then re-encodes the *plain-text* transcript **unsteered** and mean-pools A's reply tokens at layer `focus=43` (`:122`, `:127`), giving `vA` [29,H]. n=29 = `len(E2.SCENARIOS)` (`coupling_e2.py:29-88`). So the "non-textual channel" content is the activation A's *words* induce, not the steering vector.

**How it is injected.** `make_inj_hook` (`:28-35`) adds `INJ["vecs"].unsqueeze(1)` to layer-42 output at **every** position (`:33`), registered at `:89`, live only during B's generation (`:148-153`); the measurement forward pass (`:165-176`) runs with `INJ["vecs"]=None` (`:153`) — so the probe is not directly contaminated by the injected vector. B's context is text-neutral (`NEUTRAL_A`, `:66`, `:136`).

**Conditions** (`:186-189`): `none` = no vector; `full` = `beta*vA`; `ablated` = `beta*(vA - (vA·de)de)` with `de = unit(coefU[ei])` (`:183-184`); `scramble` = `beta*vA[perm]`, `perm = default_rng(0).permutation(29)` (`:185`).

**Readout** (`:176`): `((x-mu)/sd · coef[ei]).sum()` — the same single direction that is the steering vector (`:86-88`), the ablation target (`:183`), and the probe. Metrics are means over 29 (`:190-194`); no per-item values, no n, no CI in the JSON (keys: `model, tag, focus, beta, decoder_n, results`).

**Decoder.** `train_decoder` (`:43-63`) uses `LogisticRegression(C=0.5)` at `:61` — the estimator `coupling_e2.py:177-185` explicitly deprecates as non-convergent ("two independent fits of the SAME direction agree at cos 0.41 (27B)"), replaced there by difference-of-means. E5 never got the fix. It is refit per run on unseeded generations (`decoder_n` = 269 / 266 / 257 at `$.decoder_n`), so **the three files are on three different measurement axes**. Nothing seeds torch; all four conditions are single stochastic draws (`:151`, `temperature=0.9`).

---

## C1 — "the activation-passing channel works at beta=0.3: all 6 emotions transmit"

**Numbers** (`$.results.<emo>."transmit(full-none)"`), beta=0.3 (`e5actpass_qwen36-b03.json`): desperate +0.406, happy +0.407, afraid +0.329, calm +0.075, sad +0.088, angry +0.716 → **6/6 positive**, mean +0.337. beta=0.2 (`-b02.json`): −0.025, +0.249, +0.077, −0.020, +0.464, −0.283 → **3/6**. beta=0.5 (`-27b.json`): −0.218, +0.390, +0.863, +0.565, +0.533, +0.003 → **5/6**. The arithmetic in the claim is correct.

**Objection 1 — the noise floor is bigger than the effect.** `none` does not depend on beta, yet at `$.results.<emo>.none` it moves across the three runs: calm +1.911 / −0.065 / −0.055 (range 1.98), sad −1.098 / −0.193 / −0.010 (1.09), desperate (0.90), afraid (0.81); median between-run range 0.86, median sd 0.46. The claimed mean transmit at beta=0.3 is +0.337, and four of the six per-emotion values (+0.406, +0.407, +0.329, and both small ones) sit inside that band. Two sources feed this drift — decoder refit (`:82`) and unseeded sampling — and both are internal to the pipeline and unreported. With n=29, one draw per condition, and no CI, no emotion is shown to be distinguishable from zero.

**Objection 2 — beta looks chosen after the fact.** I cannot verify run order (I did not read history; all three files share one mtime). But the tag naming is suggestive: beta=0.5 carries the plain default tag `qwen36-27b` (`$.tag`) while the others carry beta-suffixed tags `qwen36-b02`/`b03`, consistent with 0.5 being the original run and 0.2/0.3 a later sweep. Whatever the order, the sweep gives 3/6, 6/6, 5/6 positives at 0.2/0.3/0.5 and transmit is **non-monotonic in beta for 4/6 emotions** (desperate −0.025→+0.406→−0.218; angry −0.283→+0.716→+0.003; sad +0.464→+0.088→+0.533; happy flat). A real injected-dose effect should grow with beta; only afraid and calm do. "6/6 at the middle dose, fewer either side" is the signature of picking the best of three noisy draws. And because the decoder is refit per run, the three files are not even on a common scale, so the sweep cannot be read as a dose-response *by the authors' own construction* — which removes the only cheap defence.

**Objection 3 — no emotion-neutral injection control.** `full − none` compares "a vector of norm ≈0.3·‖h‖ added at every position, every step" against "nothing". Any large perturbation of B's residual stream changes B's text and can move a 1-D probe. The script contains no norm-matched random or off-emotion vector; `scramble` is same-emotion (below). So "transmit" is not attributable to emotion by this contrast.

**To support it as worded:** per-scenario values retained; paired bootstrap/t CIs over the 29 items; ≥3 seeds per cell with a shared, frozen decoder across all beta; a norm-matched random-direction injection arm; and pre-registration of beta (or reporting all betas as the primary result). Nearest defensible claim: *"injecting A's pooled activation shifts B's probe score positively for most emotions at 0.2–0.5, with per-emotion effects of the same order as run-to-run drift."*

**Verdict: NOT SUPPORTED (as worded).** 6/6 holds only in the one file whose "identical-by-construction" `none` baseline drifts by more than the effect, on a per-run decoder, n=29, no CI, non-monotonic in beta.

---

## C2 — "the emotion component is causally responsible; a scrambled activation transmits far less"

**What the scramble actually is.** `:185` permutes **rows of `vA`**, i.e. scenario/episode index, *within a single emotion* — `vA` is built inside the `for e in EMOS` loop (`:180-182`) and holds only emotion `e`. The permutation is over 29 items with 0 fixed points (verified: `default_rng(0).permutation(29)`), and the docstring says so plainly: "inject v_A from a DIFFERENT **episode**" (`:13`). Consequences: (a) the *multiset of injected vectors is identical* between `full` and `scramble` — only the pairing with B's scenario differs; (b) the emotion component is **fully preserved**, and its mean over items is exactly equal in the two conditions; (c) since the score is a mean of a linear projection over items, any injection effect that is additive and context-independent cancels out of `full − scramble` by construction. So `vs_scramble` measures **episode/context alignment**, not emotion specificity. Labelling it "specificity null" (`:13`) is a misdescription of what the line computes.

**The real specificity metric is `emo_specific`,** and it is exact: `de` (`:183`) is the unit version of `coefU[ei]`, and the readout (`:176`) projects onto `coef[ei]/sd = coefU[ei]`, so `ablated` removes precisely the readout direction and nothing else.

**Numbers.** `$.results.<emo>."emo_specific(full-ablated)"` at beta=0.3: desperate **−0.213**, happy +0.246, afraid +0.132, calm **−0.110**, sad +0.079, angry +0.439 → **4/6 positive**, mean +0.096, sd 0.236. At beta=0.2: −0.262, −0.044, −0.193, +0.154, +0.237, +0.121 → 3/6, mean +0.002. At beta=0.5: −0.465, −0.211, +0.259, +0.086, +0.208, −0.059 → 3/6, mean −0.030. Across all three betas the mean specificity effect is ≈0.

`$.results.<emo>."vs_scramble(full-scramble)"` at beta=0.3: −0.064, +0.012, +0.193, **−0.472**, **−0.259**, +0.363 → **3/6 positive, mean −0.038**. beta=0.2: 5/6, mean +0.098. beta=0.5: 4/6, mean +0.070.

**Objection 1 — the data contradict the second half of the claim.** At the claimed operating point, scramble does not transmit "far less": it transmits *more* than full for 3 of 6 emotions, and the mean of `vs_scramble` is negative (−0.038). The stronger scramble results are at beta=0.2 (5/6, +0.098) — a different file, a different decoder, and the beta the authors did not adopt. There is no beta at which both `emo_specific` and `vs_scramble` are jointly positive for most emotions: the docstring's own joint criterion (`:14-15`, transmit>0 ∧ full>scramble ∧ ablated<full) is met by **1/6** at beta=0.2 (sad), **3/6** at beta=0.3 (happy, afraid, angry), **2/6** at beta=0.5 (afraid, calm).

**Objection 2 — even if it were positive, the scramble tests nothing about emotion.** Per (a)–(c) above, a scrambled activation carries the *same emotion*. A result there would license "the injected vector's episode-specific content matters", never "the emotion component is causally responsible". The needed condition — inject `vA` captured under emotion `e'≠e` — does not exist in the script.

**Objection 3 — circularity.** The direction steered into A (`:86-88`), the direction ablated (`:183`), and the measuring probe (`:176`) are one and the same vector from one unconverged logreg fit (`:61`), which `coupling_e2.py:177-185` reports reproduces at cos 0.41 on this model. "Causally responsible" is being asserted about a direction with no independent validation (no held-out probe, no behavioural readout, and — unlike E2, `coupling_e2.py:150`, `:304` — no *other*-speaker decoder, so E5 cannot separate "B feels e" from "B depicts A as e").

**To support it as worded:** an off-emotion injection arm (`vA` from `e'≠e`, norm-matched); ablation and readout on *independent* fits of the emotion direction, with the ablation validated to be ≥k-dimensional rather than rank-1; CIs on `full−ablated`; and results replicated at more than one beta. Nearest defensible claim from these files: *"removing the probe direction from the injected vector reduces the probe score for 4/6 emotions at beta=0.3 (mean +0.10, no CI); episode-scrambling shows no consistent effect."*

**Verdict: NOT SUPPORTED.** The scramble permutes episodes within one emotion, so it cannot test emotion-specificity at all, and at beta=0.3 it goes the *wrong way* (3/6 positive, mean −0.038); the one metric that does test the emotion component is positive for only 4/6, mean +0.096, uncircumscribed by any CI and smaller than the run-to-run drift measured on `none`.

## Reconciliation
Agrees with AUDIT 1 (β chosen on outcome; scramble within one emotion) and 12. New: (i) the decoder is refit per run, so the three β files are on different measurement axes and the sweep is not a dose-response by construction; (ii) the β-independent `none` baseline drifts by more than the claimed effect, giving an internal noise floor the files did not report; (iii) the script's own joint criterion (transmit>0 ∧ full>scramble ∧ ablated<full) is met by at most 3/6 at any β. Verdicts: both NOT SUPPORTED — consistent with the registry.

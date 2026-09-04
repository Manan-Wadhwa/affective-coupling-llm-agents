# 11 · Multi-turn accumulation

**File:** `results/behavioral/behavioral_multiturn_qwen36-27b.json` · **Scripts:** `src/behavioral/behavioral_multiturn.py`, `src/core/coupling_e2.py` · **Model:** Qwen3.6-27B · **Registry:** `behavioral.multiturn_accumulation` RETRACTED-CLAIM · **AUDIT:** 8 · **Raw transcripts:** `results/generations/responses_multiturn_qwen36_sample200.jsonl`

## What was run
8 scenarios × 4 name-rotations, 8 turns. A is steered desperate or calm at 1.0 × rms; the neutral arm is unsteered. Each turn, B's newest line is projected on the desperate direction, and a separate single-turn prompt yields a cheat probability. OLS slope over the 8 turn means; bootstrap resamples each turn's 32 samples independently.

## What the file shows
| condition | B-desperate slope [CI] | sig | mean cheat | cheat slope [CI] |
|---|---|---|---|---|
| desperate | +0.404 [0.19, 0.61] | true | 0.085 | −0.0002 [−0.007, +0.007] |
| calm | +0.432 [0.24, 0.62] | true | 0.234 | −0.008 [−0.023, +0.008] |
| neutral | +0.306 [0.16, 0.46] | true | 0.108 | −0.005 [−0.017, +0.006] |
Accumulation contrast (desp − calm, on cheat): +0.008 [−0.011, +0.024], not significant. No desp − calm contrast on the B-desperate readout exists in the file.

## What can be inferred
- The desperate readout rises with turn index in all three arms, including the unsteered one, and calm's slope is the largest. [supported by file]
- The rise is therefore not attributable to A's steered state. [supported by file]
- B's cheat probability is flat across turns in every arm and is lower under desperate A (0.085) than under calm A (0.234) at every turn — opposite to the hypothesis. [supported by file]
- Whether the rise reflects transcript length, escalating scenarios, or the truncation fallback is not separable here. [needs: length-matched and random-direction arms]

## Status
Registry: RETRACTED. RESEARCH_PLAN B6 preregisters the reversal for the rebuilt instrument.

### Independent critique (blind: saw only the scripts and the JSON)

## 1. What the code actually computes

**Conditions** (`behavioral_multiturn.py:136,141-142`). `conds = ("desperate","calm","neutral")`; `vec = None if cond == "neutral" else (1.0 * rms * sdir[cond])`. So **neutral is not a steered control — it is the *unsteered* run** (`E2.gen_steered` calls `steer.off()` when `vec is None`, `coupling_e2.py:118`). Desperate and calm both receive an additive perturbation of norm `1.0*rms` at layer `focus-1` (`:81,142`), where `rms` is the mean L2 norm of pooled activations (`coupling_e2.py:206`) — i.e. a perturbation as large as the signal. Neutral receives zero perturbation. The design therefore confounds *which* emotion with *whether any* intervention occurred.

**Loop** (`:143-171`). 4 reps × 8 SCEN scenarios = `n_per_turn: 32` (`/n_per_turn`). Reps differ **only in name assignment** (`:144`), so there are 8 distinct scenarios, not 32 independent items. Per turn: A generates steered (`:153`), B generates unsteered (`:160`), both at temperature 0.9 (`coupling_e2.py:120`) with **no generation seed** (the only `default_rng(0)` is for the bootstrap, `:178`) — the run is not reproducible.

**Readout** (`:108-124`). Hidden states at layer `focus = round(0.67*L)` (`:75`), mean-pooled over the char span of **B's newest line** (`:165`), projected on `desp` (`:80`). Two probe problems:
- `desp = Cp[di]/sd` (`:80`) is applied to already-standardized `z = (h-mu)/sd` (`:122-123`), giving `Xs @ (Cp/sd)` — an **extra elementwise 1/sd** versus the canonical present-score `Xs @ Cp.T` (`coupling_e2.py:150`). This is a different functional from E2's probe, arbitrarily up-weighting low-variance dims. Units are uninterpretable.
- `Co` — the other-speaker decoder that `coupling_e2.py:7-8` calls the load-bearing dissociation ("B models A as e" vs "B feels e") — is **never used** in this script (grep: no `Co`). There is no control for social modeling.
- `bmask` (`:104-106`) omits the `o[1] <= s1` upper bound present in `coupling_e2.py:144` (harmless here, span ends the string), but `truncation=True, max_length=1024` (`:112`) with default right-truncation means long late-turn transcripts lose B's line entirely, and `if m.sum()==0: m = attention_mask` (`:120-121`) silently **falls back to pooling the whole transcript, including all of steered-A's lines**. Turn-0 transcripts are ~320-420 chars (jsonl lines 1-2); at ~70-110 tokens/turn the 8-turn tail can exceed 1024. I cannot quantify the affected fraction from 2 lines — flagging as an unverified but mechanically real turn-dependent artifact.

**Cheat** (`:126-134,90-102`). Not behavior: a fresh single-turn prompt appends the transcript plus an A/B menu, and P(cheat) is the softmax over the first-token letter logits, averaged over both option orderings. Steering is off.

**Slopes/CIs** (`:181-202`). OLS of the 8 turn-*means* on turn index. `boot_slope` (`:186-195`) resamples **each turn's 32 samples independently**, and `boot_delta` (`:197-202`) resamples turn-1 and turn-8 independently. But turns 1-8 come from the *same* 32 conversations: **the data are clustered and paired, and the bootstrap treats them as independent**. The correct unit is the conversation (arguably the scenario, n=8), resampled with all its turns attached. `sig` (`:212-213`) is just CI-excludes-zero, uncorrected across 12+ intervals. The docstring promises "a scramble-paired null on the slope" (`:18`) — **not implemented**.

---

## 2. Does the JSON support the claims?

**Per-turn `bdesp` means** (`/per_turn_mean/*/bdesp`):

| turn | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| desperate | −1.423 | −0.059 | 1.438 | 1.985 | 1.984 | 1.846 | 1.369 | 2.231 |
| calm | −1.795 | −0.966 | −0.650 | −0.406 | −0.089 | 0.580 | 1.274 | 1.222 |
| neutral | −3.002 | −1.854 | −1.326 | −1.206 | −1.195 | −0.818 | −0.687 | −0.381 |

**`bdesp` slopes**: desperate `+0.4041`, CI `[0.1878, 0.6139]`, `sig true` (`/desperate/bdesp_slope*`); calm `+0.4324`, CI `[0.2393, 0.6246]`, `sig true` (`/calm/bdesp_slope*`); neutral `+0.3062`, CI `[0.1611, 0.4558]`, `sig true` (`/neutral/bdesp_slope*`). Deltas turn8−turn1: 3.654 `[1.659, 5.641]`, 3.017 `[1.239, 4.853]`, 2.621 `[1.303, 3.979]`.

**Cheat slopes** (all n.s.): desperate `−0.000208` CI `[−0.00741, 0.00687]` `sig false`; calm `−0.00768` CI `[−0.02250, 0.00775]` `sig false`; neutral `−0.00512` CI `[−0.01689, 0.00590]` `sig false`. Deltas: `−0.0094 [−0.0773, 0.0551]`, `−0.0537 [−0.1867, 0.0845]`, `−0.0728 [−0.1879, 0.0382]`.

**Accumulation contrast** (`/accum_contrast_desp_minus_calm_slope`): `mean 0.00757`, `ci [−0.01052, 0.02421]`, **`sig false`**. Note this contrast is computed on **cheat only** (`:216-224`); **the JSON contains no desperate-minus-calm contrast on `bdesp` at all**.

**Mean cheat rate across turns** (from `/per_turn_mean/*/cheat`): desperate **0.0854**, calm **0.2341**, neutral **0.1077**. Desperate is *below* calm at every one of the 8 turns.

---

## 3. Strongest objections

**(a) Neutral rises too, and calm rises fastest.** All three `bdesp_slope_sig` are `true`, and the largest slope is **calm (0.4324)**, not desperate (0.4041); their CIs overlap almost completely. A drift that occurs identically with a desperate vector, a calm vector, and *no vector at all* is not contagion from A's steered state. The most parsimonious reading is that the pooled representation drifts along the "desperate" direction as a function of turn index for reasons common to every condition: all 8 SCEN items are escalating deadline/fraud dilemmas (`:30-47`) so any conversation gets more stressed; the transcript grows monotonically, so token count, position, and (past ~1024 tokens) the truncation fallback at `:120-121` all covary perfectly with turn. The probe is measuring "how far into a high-pressure transcript are we," not "how much of A's state crossed into B."

**(b) The one thing steering does move is the *level*, and it moves the wrong control.** Neutral sits ~1.2-2.0 below calm at every turn (turn 1: −3.002 vs −1.795). Adding a **calm** vector to A raises B's *desperate* readout by more than a full unit relative to no steering. Whatever `1.0*rms` steering does, a large part of it is emotion-nonspecific — it pushes A's text off-distribution (see the desperate transcripts: "a metaphor for my own damnation", "I will move to a third world country"), and B's readout responds to that. Desperate exceeds calm by only 0.372 at turn 1 and 0.095 at turn 7.

**(c) The desperate trajectory is a step, not an accumulation.** Diffs are `[+1.36, +1.50, +0.55, −0.00, −0.14, −0.48, +0.86]`: **78% of the total rise happens by turn 3**, then it plateaus and dips. Calm is the more nearly linear series (38% by turn 3). "Accumulates over 8 turns" describes calm better than desperate.

**(d) Cheat goes the wrong way, by a lot.** Desperate 0.0854 vs calm 0.2341 — B is **~2.7× less** likely to pick the shortcut under a desperate A, at every turn, and the desperate condition is also below the unsteered neutral (0.1077). The sample transcripts suggest why: extreme steered-A lines trigger assistant-refusal register in B ("I am neither a sentient being nor a metaphor for your damnation… I am a large language model trained by Google… I cannot help you alter, delete, or falsify audit logs"). That is a jailbreak-resistance artifact, not affective coupling, and it simultaneously suppresses cheat and could inflate the desperate readout.

**(e) Inference unit.** With 8 scenarios re-run under 4 name permutations, resampled per-turn-independently, the CIs on the slopes are not credible as stated. A clustered bootstrap over conversations would also absorb the shared intercept; a paired bootstrap is required for the turn8−turn1 delta.

---

## 4. What would be needed

**C1.** A rise in B's readout is not evidence of contagion unless it is (i) measured with the probe the decoder actually defines (fix `:80/:123` to `Xs @ Cp`), (ii) separated from the other-speaker score `Co` so "B is desperate" is distinguished from "B represents A as desperate," and (iii) shown to exceed the drift present with no steering at all. Add a turn-matched control that grows the transcript without an emotional A (e.g. A steered on an orthogonal/random direction of the same norm, and a length-matched filler arm), and remove the truncation fallback (raise `max_length`, or drop items whose span is truncated rather than silently pooling the whole transcript). Nearest defensible claim today: *B's desperate readout rises monotonically with turn index in all conditions; the desperate condition sits at a higher level (+0.37 to +2.39 over calm, +1.58 to +3.19 over neutral) but has no steeper slope.*

**C2.** Requires a slope contrast on `bdesp` with a cluster bootstrap over conversations, plus the neutral arm as a third leg. The JSON does not contain this statistic. Given calm's slope (0.4324) *exceeds* desperate's (0.4041), the contrast would almost certainly straddle zero or point the wrong way. Nearest defensible claim: *the accumulation is not condition-specific; the level offset is.*

**C3.** Cheat would need a real behavioral outcome (a generated action, or at minimum a scored free-form choice) rather than a first-token letter probability, a within-conversation paired analysis, and much more power — the desperate cheat CI `[−0.0074, +0.0069]` is tight enough to bound a per-turn drift below ~0.7 percentage points, which is a publishable *null*, not a change. Nearest defensible claim: *B's cheat probability is flat across 8 turns under desperate A (slope −0.0002, CI [−0.0074, 0.0069]), and is substantially lower under desperate than calm A (0.085 vs 0.234) — a level effect in the direction opposite to the hypothesis.*

---

## 5. Verdicts

- **C1 "representation contagion accumulates over 8 turns"** — **NOT SUPPORTED** as *contagion*: the readout rises, but it rises significantly in the unsteered neutral arm too (`/neutral/bdesp_slope 0.3062`, `sig true`), so the rise is not attributable to A's steered state.
- **C2 "the accumulation is specific to the desperate condition (vs calm)"** — **NOT SUPPORTED**: calm's `bdesp` slope (0.4324) is *larger* than desperate's (0.4041), no `bdesp` contrast exists in the JSON, and the only computed contrast (`/accum_contrast_desp_minus_calm_slope`) is `sig false`.
- **C3 "B's cheating changes across turns under desperate A"** — **NOT SUPPORTED**: `/desperate/cheat_slope −0.000208`, CI `[−0.00741, 0.00687]`, `sig false`; delta turn8−turn1 `−0.0094 [−0.0773, 0.0551]`. Flat, and the cross-condition difference runs desperate < calm (0.085 vs 0.234), opposite to the framing.

## Reconciliation
Agrees with AUDIT 8 on every number. New: (i) the neutral arm receives no perturbation at all, so "which emotion" is confounded with "whether any steering happened" — a calm vector raises B's *desperate* readout by more than a unit over neutral; (ii) transcripts past 1024 tokens fall back to pooling the whole sequence including A's steered lines, an artifact that grows with turn; (iii) the desperate series is a step by turn 3 and a plateau after, not an accumulation; (iv) sample transcripts show B dropping into assistant-refusal register under extreme steered lines, which suppresses cheat and may inflate the readout. Verdicts: all three NOT SUPPORTED.

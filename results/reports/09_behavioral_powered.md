# 09 · Behavioral contagion, "powered" run

**Files:** `results/behavioral/behavioral_powered_{qwen36-27b,llama3-abl,llama3-inst}.json` · **Scripts:** `src/behavioral/behavioral_powered.py`, `src/core/coupling_e2.py` · **Models:** Qwen3.6-27B (primary), Llama-3-8B-Instruct-abliterated, Llama-3-8B-Instruct · **Registry:** `behavioral.honesty_floor_ruled_out` RETRACTED-CLAIM (Qwen); Llama files UNREGISTERED · **AUDIT:** 6, 8

## What was run
8 scenarios × 4 reps = 32 per condition (desperate / calm / neutral). A is steered at 1.0 × rms; B replies unsteered; "cheat" is the softmax over the two option-letter logits at B's first token, averaged over both orderings — a probability, not a sampled decision. Bootstrap 4000 draws, conditions resampled independently. Manipulation check: B's reply projected on the desperate direction.

## What the files show
| model | cheat desp / calm / neutral | desp − calm [CI] | manip contrast [CI] | manip_sig |
|---|---|---|---|---|
| Qwen3.6-27B | 0.252 / 0.194 / 0.219 | +0.058 [−0.008, +0.123] | +1.48 [−0.85, +3.85] | false |
| Llama-3-8B-abl | 0.341 / 0.318 / 0.301 | +0.023 [−0.080, +0.131] | +13.8 [+3.5, +23.9] | true |
| Llama-3-8B-inst | 0.223 / 0.187 / 0.276 | +0.035 [−0.071, +0.137] | −1.12 [−9.70, +7.26] | false |
No MDE, power, or floor computation in any file.

## What can be inferred
- No behavioral contrast excludes zero; all three point estimates are positive and the primary model's CI misses zero by 0.008. [supported by file]
- The CIs exclude only increases larger than roughly 12 absolute points; "not detectable" is accurate, "ruled out" is not computed. [supported by file]
- The manipulation check is significant on one of three models and not on the primary one. [supported by file]
- Whether the readout can move at all on these scenarios is untested. [needs: positive control — RESEARCH_PLAN B6]

## Status
Registry: RETRACTED. B6 rebuilds the instrument from the screened scenario set.

### Independent critique (blind: saw only the scripts and the JSONs)

## 1. What the code actually computes

**"Cheat" is not a behavior.** No judge, no keyword, no sampled choice. `cheat_probs` (`src/behavioral/behavioral_powered.py:61-73`) runs one forward pass, takes the final-position logits (`:68`), `logsumexp` over the token sets `{"A"," A"}` and `{"B"," B"}` (`:57-58`, `:69`), softmaxes those two scalars (`:70`), and returns a probability. It is a two-token letter preference, never an argmax decision — `argmax` appears nowhere in either file (grep: no hits).

**Design.** 8 scenarios (`:20-37`) × `REPS = 4` (`:19`) = 32 values per condition (`:127`, `n_per_cond`). Three conditions: `desperate`, `calm`, `neutral` (`:100`), at a **single** steering magnitude `1.0 * rms` (`:101`) — no dose-response. A's turn is steered (`:109`); B's generation (`:116`) and the measurement pass (`:90`) are unsteered. Both option orderings are run and **averaged per scenario** (`:122-127`), so 64 forwards collapse to 32 points. Generation is sampled at T=0.9/top-p 0.95 with no seed (`src/core/coupling_e2.py:120`); decoders are refit per run (`:48`, `coupling_e2.py:153-209`).

**CIs.** Percentile bootstrap, 4000 draws, `rng` seed 0, resampling the two conditions **independently** (`:131-135`). The design is blocked on scenario (the same 8 recur in every condition) but the analysis is unpaired, discarding the blocking that would shrink the SE. The bootstrap also treats 32 points as iid when they are 8 clusters × 4 reps.

**Power.** None. `grep -niE "power|mde|minimum detect|floor|ceiling"` hits only the docstring (`:1,3,9`), the banner print (`:144`), and the filename (`:149`). "Powered" is a title, not a computation. No MDE, no floor/ceiling analysis, no per-item data saved (`:138-143` writes means only).

**Manipulation check.** `reply_desperate` (`:79-96`) pools hidden states at `focus` over B's generated reply and projects onto `desp` (`:51`). Two problems, both citable:
- `desp = Cp[desperate] / sd` (`:51`) is then dotted with already-standardized activations `z = (h-mu)/sd` (`:94-95`). The module's own convention is `Xs @ Cp.T` (`coupling_e2.py:149-150`); `Cp/sd` is the *raw-space steering* direction (`coupling_e2.py:233`). The probe divides by `sd` twice — it is neither the decoder score nor the steering direction, and its units are arbitrary and run-specific.
- It uses **`Cp` (present)** only; `Co` is never referenced in this file (grep: no hit). Per `coupling_e2.py:7-8`, `present` = "how much B itself feels e", `other` = "how much B models A as e". The check measures B's own desperation, not B's model of A.
- If the reply span mask is empty after truncation at 768 tokens (`:86`), it silently falls back to the **entire sequence including A's steered line** (`:93`). No count of fallbacks is recorded.

The paraphrase control that exists in `coupling_e2.py:274-276, 287-301` (does the effect survive neutralizing A's wording?) is absent here, so lexical echo is unexcluded.

## 2. Do the JSONs support the claims?

All three, `n_per_cond` = 32. SE derived as CI width / 3.92.

| file | `means.desperate.cheat` | `means.calm.cheat` | `means.neutral.cheat` | `behavioral_cheat_desp_minus_calm` | `behavioral_ci` | implied z | `behavioral_sig` |
|---|---|---|---|---|---|---|---|
| `behavioral_powered_qwen36-27b.json` | 0.2522 | 0.1941 | 0.2188 | **+0.0582** | [-0.0078, +0.1234] | **1.74** | false |
| `behavioral_powered_llama3-abl.json` | 0.3407 | 0.3175 | 0.3013 | +0.0231 | [-0.0802, +0.1307] | 0.43 | false |
| `behavioral_powered_llama3-inst.json` | 0.2225 | 0.1871 | 0.2759 | +0.0354 | [-0.0707, +0.1373] | 0.67 | false |

| file | `means.*.bdesp` (desp / calm / neutral) | `manip_check_bdesp_desp_minus_calm` | `manip_check_ci` | z | `manip_sig` |
|---|---|---|---|---|---|
| qwen36-27b (**primary**, `:18`) | -0.802 / -2.276 / **-0.308** | +1.475 | [-0.851, +3.849] | 1.23 | **false** |
| llama3-abl | -8.810 / -22.628 / -23.081 | +13.818 | [+3.451, +23.947] | 2.64 | true |
| llama3-inst | -10.387 / -9.267 / **-13.296** | **-1.121** | [-9.697, +7.259] | -0.26 | false |

C1: all three `behavioral_sig` are false, so "not detectable" is literally what the files say. Nothing in any file addresses a floor. C2: the manipulation check is significant on **one of three** models, and **not** on the primary one.

## 3. Strongest objections

**(a) The primary model's behavioral result is marginal-positive, not null.** `behavioral_ci` [-0.0078, +0.1234] on Qwen misses zero by 0.008 with a point estimate of +0.058 — implied z = 1.74, p ≈ 0.08. That is not Monte-Carlo fragility (the 2.5% percentile sits ~5 bootstrap-MC-SEs below 0 at n=4000), it is a genuinely borderline effect being reported as an absence. And a paired/scenario-blocked analysis, which the design supports and `:131-135` declines to use, would have a smaller SE.

**(b) What the CIs exclude is trivially little.** Against the calm base rates, the upper bounds permit **+64% relative** on Qwen (0.1234 vs 0.1941), **+41%** on llama3-abl, **+73%** on llama3-inst. Only effects larger than roughly a 12-point absolute increase in mean cheat probability are excluded. With 32 points from 8 clustered scenarios, this experiment cannot see anything but an enormous effect; calling that "not detectable" is accurate and calling it evidence of absence is not.

**(c) All three point estimates are positive.** Fixed-effect pooling of the three gives **+0.046, SE 0.025, 95% CI [-0.004, +0.095], z = 1.82** — 3/3 same sign, pooled p ≈ 0.07. The consistent direction is the opposite of what "no reach" predicts, and nothing in the repo's own outputs combines them.

**(d) "The honesty floor is ruled out" is asserted, not computed.** No file contains a floor field, per-item distribution, argmax rate, or positive control. The only supporting fact is that calm/neutral means are 0.19-0.32, away from 0 — but that is a mean over a *two-token softmax*, further shrunk toward 0.5 by the ordering average (`:127`). A set of items each deterministically argmax-honest at p(cheat)≈0.2 produces exactly this mean, and a +0.058 probability shift could flip zero decisions. Ruling out a floor requires showing the measure *can* move — a positive control (direct instruction to cut corners, a higher-stakes scenario, or a larger alpha). The script runs one alpha (`:101`) and no such arm; only 8 hand-written scenarios (`:20-37`), never screened for headroom.

**(e) The manipulation check does not measure what C2 says, and is incoherent on two of three models.** It projects onto `Cp` (present-desperate = *contagion*, `coupling_e2.py:7`), not `Co` (B modeling A, `:8`). Worse, the `neutral` (unsteered) arm falsifies it where it is not significant: on Qwen, `means.neutral.bdesp` = -0.308 is **higher** (more desperate) than `means.desperate.bdesp` = -0.802 — the unsteered baseline out-scores the desperate steering; on llama3-inst the ordering is non-monotone (neutral -13.30 < desperate -10.39 < calm -9.27) and the contrast is **negative**. Only llama3-abl orders sensibly. A probe that ranks unsteered above steered on the primary model is not validated. Units are also arbitrary and run-specific (decoders refit at `:48`, double `1/sd` at `:51`), so the +1.5 vs +13.8 vs -1.1 magnitudes are not comparable and no standardized effect size exists in any file.

**(f) Circularity and unexcluded lexical confound.** Steering direction (`:53-55`) and measurement probe (`:51`) derive from the same `Cp` row of the same per-run fit. Steering is off during B's generation and measurement, so this is not direct read-back, but B echoing A's desperate wording is indistinguishable from B representing anything — and the paraphrase arm that would test this (`coupling_e2.py:287-301`) is not run here.

## 4. What each claim would need

**C1 as worded** needs: (i) a pre-specified MDE and a power calculation at the cluster level (8 scenarios, not 32 rows) — neither exists; (ii) an equivalence test against a stated smallest effect of interest, not a CI that straddles 0; (iii) a positive control proving the cheat readout moves under *some* manipulation, which is the only thing that can "rule out" a floor; (iv) a scenario-blocked (paired) analysis; (v) an actual decision measure (argmax rate, or sampled choices), since "cheating rate" is claimed and a letter-token probability is measured; (vi) reconciliation of the three uniformly positive point estimates.

**Nearest defensible C1:** "In 8 scenarios × 4 reps per condition, steering A to desperate shifted B's cheat-option probability by +0.02 to +0.06 across three models, with 95% CIs that all include zero and upper bounds permitting a 40-70% relative increase; the design cannot resolve effects of that size. Base rates of 0.19-0.34 show the probability readout is not pinned at zero, but no positive control establishes that the measure is sensitive, and no floor test was run."

**C2 as worded** needs: (i) the `Co` (other-speaker) probe, not `Cp`, since the claim is about B's representation *of A*; (ii) significance on the primary model — it is absent (`manip_check_ci` [-0.851, +3.849]); (iii) the `neutral` arm to sit below `desperate` on the probe, which it does not on Qwen or llama3-inst; (iv) a paraphrase arm to separate representation from lexical echo; (v) a standardized effect size so the three models can be compared.

**Nearest defensible C2:** "On one of three models (llama3-abl), B's reply projects more onto the present-desperate direction after A is steered desperate (+13.8 [3.5, 23.9], arbitrary run-specific units). On the primary Qwen model and on llama3-inst the contrast is not significant, and the unsteered neutral arm scores as desperate or more so, so the manipulation check is not established on the primary model."

## 5. Verdict

- **C1 — NOT SUPPORTED as worded.** The null half is SUPPORTED-WITH-CAVEATS (all `behavioral_sig` false, but the primary model's z = 1.74 and CIs exclude only implausibly large effects); the "honesty floor is ruled out" half has no computation, no positive control, and no floor field anywhere in the code or the JSONs.
- **C2 — NOT SUPPORTED.** The check is `manip_sig: false` on the primary model, significant on 1 of 3, measures `Cp` (B's own desperation) rather than the `Co` representation-of-A the claim names, and is contradicted by the unsteered `neutral` arm on two of three models.

## Reconciliation
Agrees with AUDIT 6 (no MDE, CI rules out nothing) and adds four points the audit does not have: (i) the outcome is a two-token letter probability, never an argmax or sampled decision, so "cheating rate" is a misnomer; (ii) the probe multiplies an already-standardised activation by `Cp/sd`, a double standardisation that differs from the project's own present-score convention (`Xs @ Cp`) — this recurs in the induction and multi-turn scripts; (iii) it uses the present row, not the other-speaker row the claim names; (iv) pooling the three models fixed-effect gives +0.046 [−0.004, +0.095], a consistent positive direction the null wording hides. Verdicts: C1 NOT SUPPORTED as worded, C2 NOT SUPPORTED.

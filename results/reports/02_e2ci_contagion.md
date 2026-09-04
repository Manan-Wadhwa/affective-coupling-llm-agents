# 02 · E2-CI — contagion dose-response with bootstrap CIs and a paraphraser control

**File:** `results/representation/e2ci_qwen36-27b.json` · **Scripts:** `src/representation/coupling_e2_ci.py`, `src/core/coupling_e2.py` · **Model:** Qwen/Qwen3.6-27B · **Registry:** `e2ci.contagion_5of6` PENDING; `e2ci.present_gt_other` RETRACTED-CLAIM · **AUDIT:** 3, 4, 16, 18

## What was run
A is steered along each emotion's probe direction at α ∈ {0, 0.5, 1.0} (α = 2 excluded by the script's docstring), 29 scenarios × 3 repeats = 87 per cell. B replies unsteered. B's reply span is projected onto the same probe's *present* and *other* rows. A paraphraser arm rewrites A's steered line with the unsteered model and regenerates B. Bootstrap: 2000 draws per α cell, resampled independently. decoder_n = 272.

## What the file shows
- `summary.<e>.present.slope` [CI]: desperate 1.27 [0.98, 1.56]; afraid 1.70 [1.35, 2.05]; happy 1.07 [0.70, 1.43]; sad 0.91 [0.49, 1.34]; angry 1.42 [1.02, 1.83]; calm 0.08 [−0.40, 0.56] → 5 of 6 `sig`.
- `summary.<e>.other.slope`: desperate 1.78, afraid 2.14, happy 1.92, calm 0.62, sad 2.13, angry 0.89 → present > other for **1 of 6** (angry).
- `summary.<e>.present_para.sig` true for 5 of 6 (calm false). `A_affect_stripped`: desperate 2.24, happy 1.85, afraid 0.19, calm −0.49 (paraphrase made A more calm-coded); no CI on these.

## What can be inferred
- B's projection onto the steered direction is higher at α = 1 than at α = 0 for five emotions, with CIs excluding zero. [supported by file]
- The *other*-speaker projection rises more than the *present* projection for five of six emotions; the file contains no test of that difference. [supported by file; the paired test needs: A3]
- Because the α grid is three equispaced points, the reported "slope" equals the α = 1 minus α = 0 difference; α = 0.5 carries no weight in it. [supported by file: verified for all 18 cells by the critique]
- The paraphrase arm's validity statistic is pooled over α (including α = 0) and fails for afraid and calm, so "survives paraphrase" is only assessable for desperate, happy, sad. [supported by file]
- Whether B "feels" the emotion or is representing A's is not separable here; steering direction and readout direction are the same fit. [needs: held-out probe, other-speaker contrast]

## Status
Registry: 5/6 count resolves but is PENDING (unpaired bootstrap); present>other RETRACTED. RESEARCH_PLAN A3 is the re-test.

### Independent critique (blind: saw only the scripts and the JSON)

Files read: `/home/manan/Projects/files/affective-coupling-llm-agents/src/representation/coupling_e2_ci.py` (cited as `ci:`), `/home/manan/Projects/files/affective-coupling-llm-agents/src/core/coupling_e2.py` (cited as `e2:`), `/home/manan/Projects/files/affective-coupling-llm-agents/results/representation/e2ci_qwen36-27b.json`.

**Not verifiable:** `coupling_e0` (`EMOTIONS`, `NAMES`, `pool_final`, `leak`, `parse_final_A`, `log_responses`) was outside my read scope, so decoder label semantics, name-pool size, and generation logs are taken on faith.

---

## C1 — "emotional contagion significant for 5/6 emotions (bootstrap CIs)"

**What the code computes.** Decoders are trained once (`ci:38`) and frozen for the whole run. The steering direction for emotion *e* is row *e* of the **present** decoder, de-standardized and unit-normalized (`ci:41-42`); the injected vector is `al * decs["rms"] * raw[e]` (`ci:52`), i.e. at α=1.0 a perturbation whose norm equals the *mean hidden-state norm* (`e2:206`), added to **every** token position at layer 42 (`e2:108`, `ci:36`). A's reply is generated under that hook (`ci:61`); B's reply is generated with `vec=None` (`ci:85`), i.e. unsteered. Readout: `utterance_score` runs the full transcript (`e2:141`), masks to B's char-span only (`e2:143-147`), and projects onto **the same frozen `Cp`** used for steering and onto `Co` (`e2:150`). Bootstrap (`ci:110-119`): 2000 draws, each α-column resampled **independently** at the sample level (`ci:115`), percentile CI, `sig = lo>0 or hi<0` (`ci:119`).

**Does the JSON support it?** Yes as counted. `summary.*.present.sig` is `true` for desperate (1.2653, [0.9793, 1.5596]), afraid (1.6959, [1.3535, 2.0464]), happy (1.0749, [0.7025, 1.4283]), sad (0.9062, [0.4873, 1.3429]), angry (1.4151, [1.0236, 1.8336]); `summary.calm.present` = 0.0758, [-0.3955, 0.5624], `sig: false`. `n_per_cell` = 87 (= 29 scenarios × `repeats` 3), `decoder_n` = 272. I re-ran the bootstrap from `raw_scores` clustering by scenario and Bonferroni-correcting over 6 emotions: all five survive (e.g. sad [+0.254, +1.535]), calm does not. The count is robust to my two objections to the resampling scheme.

**Objections.**
1. **The "slope" is not a slope.** `Ac = [-0.5, 0, 0.5]`, so `ci:112` reduces algebraically to `mean(α=1.0) − mean(α=0.0)`; α=0.5 carries **zero weight**. I confirmed this to 4 decimals for all 18 cells (e.g. `raw_scores.desperate.1.0.present` mean −0.2973 minus `raw_scores.desperate.0.0.present` mean −1.5626 = 1.2653 = `summary.desperate.present.slope`). No dose-response is tested, and `happy.present` is in fact non-monotone (1.654 → 2.866 → 2.729).
2. **Circularity of probe and intervention.** The steering direction *is* the measurement direction (`ci:41-42` vs `e2:150`). Any surface trace of the injected direction that survives into A's text scores positively by construction.
3. **B's readout is contextually contaminated.** Hidden states come from a transcript containing A's steered line (`e2:141`); masking to B's span (`e2:145`) does not remove attention to A. "B feels *e*" and "A's *e* is legible at B's positions" are not separated here — which is exactly what C2 was supposed to fix.
4. **No coherence check.** `ci:10` drops α=2 as "model-breaking" but nothing in either script measures degeneracy at α=1.0.

**What would make it.** A held-out decoder (trained on data disjoint from the steering direction, with reported accuracy), a genuine ≥4-point monotonicity test, and a fluency/degeneracy metric per α. As worded, the defensible claim is "B's projection onto the steered direction is higher at α=1.0 than α=0.0 for 5/6 emotions."

**Verdict: SUPPORTED-WITH-CAVEATS** — the 5/6 count is numerically correct and survives clustering plus Bonferroni, but "contagion" is an interpretation the design (steer-direction = probe-direction, shared context, two-point contrast) cannot license.

---

## C2 — "the load-bearing result: B's present shift exceeds its other shift" (all six emotions)

**What the code computes.** `ci:122` calls `slope_ci("present", e)` and `slope_ci("other", e)` as two independent invocations. Each draws its own resample indices from the module-level `rng` (`ci:108`, `ci:115`), so the present and other bootstraps are **not paired** even though they are computed from the *same* B utterances (`ci:92-93`). There is **no** difference statistic, **no** CI on `present − other`, and **no** comparison of any kind anywhere in either file. `e2:321-323` likewise only stores the two slopes side by side. The comparison is therefore an eyeball of two point estimates, and I could not find code that performs it.

**Does the JSON support it?** **No — it contradicts it.** Present exceeds other for **1 of 6** emotions:

| emotion | `summary.*.present.slope` | `summary.*.other.slope` | present > other? |
|---|---|---|---|
| desperate | 1.2653 | 1.7808 | no |
| afraid | 1.6959 | 2.1410 | no |
| happy | 1.0749 | 1.9170 | no |
| calm | 0.0758 | 0.6150 | no |
| sad | 0.9062 | 2.1323 | no |
| angry | 1.4151 | 0.8875 | **yes** |

The direction is not merely wrong on average, it is *significantly reversed*. Doing the test the code omits — paired cluster bootstrap over the 29 scenarios, same resample indices for both probes, from `raw_scores` — gives `present − other`: desperate −0.516 [−0.877, −0.147]; afraid −0.445 [−0.753, −0.147]; happy −0.842 [−1.432, −0.243]; calm −0.539 [−1.046, −0.063]; sad −1.226 [−1.834, −0.600]; angry **+0.528** [+0.085, +0.958]. Five significant reversals, one significant confirmation. `sad` is the extreme case: `raw_scores.sad.1.0.other` mean +2.7040 against `raw_scores.sad.1.0.present` mean +0.1769.

**Objections.**
1. The claim is false on the authors' own output for 5/6 emotions, and significantly so. This is falsifiable in one line from `summary`.
2. Even the one favourable case (angry) rests on an **arbitrary scale**. `_dom` (`e2:190-196`) returns raw difference-of-means rows with **no row normalization**, so `‖Cp_e‖` and `‖Co_e‖` are unequal and unreported. `present` and `other` scores are projections onto vectors of different lengths; "present shift exceeds other shift" is not scale-invariant — rescaling `Co` flips it. Falsification test: unit-normalize both rows in `_dom` and recompute.
3. The probes are not independent: sample-level correlations within cells run up to r = +0.59 (afraid, α=0.5), so `present` and `other` partly measure the same thing.
4. The pattern actually observed — `other` rising faster than `present` — is the *social-modeling* reading the docstring (`e2:7-9`) explicitly designates as **not** contagion.

**What would make it.** A paired bootstrap on `present − other` with shared indices, over scale-normalized probes, reported per emotion. On this data the nearest defensible claim is the opposite of C2: *"For 5/6 emotions the other-speaker shift significantly exceeds the present shift; only `angry` shows the present-dominant pattern."*

**Verdict: NOT SUPPORTED** — the code never tests the comparison, and the JSON reverses it significantly for 5/6 emotions.

---

## C3 — "a paraphraser control shows contagion survives paraphrase of A's message"

**What the code computes.** A's steered line is rewritten by the **same, unsteered** model (`ci:69-71`, `ci:74`) with the instruction to keep "only the literal information." B is regenerated against that neutralized context (`ci:96`) and scored on B's span only (`ci:100-103`) → `present_para`. A's own affect is measured on the original (`ci:72-73`) and paraphrased (`ci:80-81`) utterance. Critically, `aff[e]["orig"]` and `aff[e]["para"]` accumulate **inside the α loop** (`ci:51`), so `summary.*.A_affect_orig/para/stripped` (`ci:124-126`) are means pooled over α ∈ {0.0, 0.5, 1.0} — including the unsteered α=0 cell where there is no induced affect to strip. Per-α A-affect is never stored, so this cannot be recovered from the JSON.

**Does the JSON support it?** Partly, and it undercuts the control's own premise for two emotions. `summary.*.present_para.sig` is `true` for 5/6 (calm: −0.2749, [−0.7318, 0.1989], `sig: false`). But retention is substantial and the loss is significant for two emotions — paired cluster bootstrap on `present − present_para` from `raw_scores`: desperate +0.619 [+0.278, +0.969] (51% retained), angry +0.745 [+0.424, +1.064] (47%), afraid +0.488 [+0.248, +0.725] (71%), happy +0.304 [−0.168, +0.785], sad +0.316 [−0.116, +0.778], calm slope goes negative. So "survives" means "roughly half survives" for two of the five.

Worse, the control's validity check fails where it matters. `summary.afraid.A_affect_stripped` = **0.1906** (orig 0.6407 → para 0.4501) and `summary.calm.A_affect_stripped` = **−0.4851** (orig 2.3035 → para 2.7886, i.e. the paraphrase made A *more* calm-coded). The script's own docstring (`ci:6-9`) states the disjunction: if the paraphrase does not strip A's affect, surviving contagion is "a paraphraser leak." For `afraid` — one of the five "survives" emotions, `present_para` = 1.2082 — the paraphrase demonstrably barely stripped anything, so that cell falls on the leak branch by the authors' own criterion. Only `desperate` (2.2409) and `happy` (1.8497) show a large strip.

**Objections.**
1. `A_affect_stripped` is diluted by α=0 (`ci:51`, `ci:73`, `ci:81`) and has **no CI** (`ci:124-126` are bare means) — the control statistic is both mis-specified and untested.
2. `afraid` and `calm` fail the leak criterion the script itself defines, so C3 cannot be asserted uniformly.
3. The paraphrase preserves semantic content by construction ("keeping only the literal information", `ci:70`). Survival therefore shows transmission is not *word-choice*-specific; it does **not** show a non-textual channel. C3's implicit "so it isn't just lexical" does not follow.
4. `gen_steered` keeps only the first line (`e2:124`); a paraphrase that emits a preamble is silently truncated, and nothing validates that `para_reps` are actual rewrites.

**What would make it.** Per-α A-affect with bootstrap CIs, a pre-registered strip threshold, per-emotion gating on it, and a CI on `present − present_para`. Nearest defensible claim: *"For desperate, happy and sad — where the paraphrase measurably strips A's affect — roughly half to two-thirds of B's present-direction shift survives neutralization; for afraid the paraphrase failed to strip A's affect, and calm shows no effect either way."*

**Verdict: SUPPORTED-WITH-CAVEATS** — 5/6 `present_para` CIs exclude zero, but the control's own validity statistic is pooled across α, uncertainty-free, and fails for `afraid` and `calm`; and "survives" hides a significant ~50% attenuation for desperate and angry.

## Reconciliation
Agrees with AUDIT 3 (1 of 6) and 18 (unpaired bootstrap), and goes further: a paired, scenario-clustered bootstrap on present − other gives five *significant reversals* and one confirmation — the test the plan's A3 was written to run, now already done on the old file. New, not in the audit: (i) with three equispaced doses the OLS slope is the endpoint difference, so no dose-response is tested anywhere in the legacy code — this also holds for rev-3's `ols_slope` on the same grid; (ii) `A_affect_*` are pooled across α and carry no CI; (iii) steering and readout share one fit. Verdicts: C1 SUPPORTED-WITH-CAVEATS, C2 NOT SUPPORTED, C3 SUPPORTED-WITH-CAVEATS — consistent with the registry.

# 12 · Early behavioral pilots (coupling, verify)

**Files:** `results/behavioral/behavioral_coupling_qwen36-27b.json`, `results/behavioral/behavioral_verify_qwen36-27b.json` · **Scripts:** `src/behavioral/behavioral_coupling.py`, `src/behavioral/behavioral_verify.py`, `src/core/coupling_e2.py` · **Model:** Qwen3.6-27B · **Registry:** both UNREGISTERED · SUPERSEDED (README: kept for the record) · **AUDIT:** 7 (verify's arm-2 check)

## What was run
Coupling: 8 scenarios, A steered desperate / calm / neutral at 1.0 × rms, B's cheat probability from two option-letter logits, both orderings; one draw per scenario, no n recorded, no CI. Verify: the same arm plus an induction arm (arm 2) in which a vector is injected during a single forward pass over a fixed prompt and the last-position hidden state is projected on the desperate direction.

## What the files show
- Coupling: cheat desperate 0.265, calm 0.266, neutral 0.227; `behavioral_contagion` −0.001.
- Verify: `arm1_behavioral_contagion` +0.117; `arm1_manip_check_desperate_shift` +0.003; arm 2 `B_desperate` identical to the last bit across desperate / calm / none (7.5685…) so `arm2_manip_desperate_shift` = 0.0 while arm-2 cheat probabilities differ (0.374 / 0.321 / 0.132).

## What can be inferred
- The two pilots disagree by 0.118 on the same contrast with the same scenarios, so neither point estimate carries information about the effect's sign. [supported by file]
- Any 1.0 × rms steering raises cheat above neutral by ~0.04 regardless of valence. [supported by file, no CI]
- Arm 2's manipulation check never sees the injected vector: the recorded hidden state is bit-identical across conditions while the behaviour changes. [supported by file]
- Nothing in either file bears on contagion. [needs: the rebuilt instrument]

## Status
Superseded. Not in the registry.

### Independent critique (blind: saw only the scripts and the JSONs)

Environment note: `transformers` is not installed in `.venv`, so I could not execute anything or confirm library-internal behaviour; every claim below is from source text plus the two JSONs.

---

## C1 — `behavioral_coupling.py`: "no behavioral contagion, desperate − calm ≈ 0"

Docstring under critique: *"Behavioral contagion = P(cheat | A desperate) - P(cheat | A calm)."* (`src/behavioral/behavioral_coupling.py:6`)

**1. What it computes.** Decoders are refit from scratch each run (`:43` → `src/core/coupling_e2.py:153-209`, difference-of-means, `:202-203`). Steering direction = unit(`Cp[e]/sd`) (`:45-47`); vector = `alpha*rms*dir` with `alpha=1.0` for both desperate and calm, `0.0` for neutral (`:16`, `:74-75`), injected at layer `focus-1` (`:44`) on **every** token of A's generation (`coupling_e2.py:111-126`, `do_sample=True, temperature=0.9`, **no seed anywhere**). A's first output line only (`coupling_e2.py:124`). B's choice is scored as a two-token restricted softmax over `{"A"," A"}` vs `{"B"," B"}` at the last prompt position (`:60-62`); order-averaging is correct (`:85`: `p[i] if kind[i]==1 else 1-p[i]`). **n is recorded nowhere in the JSON**; from `:19-36` and `:82-83` it is 8 scenarios × 2 orderings = 16 scored prompts per condition, but only **8 independent A messages, one draw each**. No CI, no SE, no per-item values, no repeats, no seed.

**2. Does the JSON support it?** `behavioral_contagion(desperate-calm)` = `-0.0010068502742797136`; `cheat_prob.desperate` = `0.26510673691518605`, `cheat_prob.calm` = `0.26611358718946576`. Numerically small, but there is no variance estimate anywhere in the file, so "≈ 0" is an assertion about a point estimate with unknown noise.

**3. Strongest objections (falsifiable).**
- *The other file refutes it.* `behavioral_verify.py:87-103` runs the same arm-1 design on the same 8 scenarios and reports `arm1_behavioral_contagion` = `0.11706535553094` — 116× the coupling estimate, opposite sign. Two runs of a near-identical protocol (prompt wording differs only at `behavioral_coupling.py:81-83` vs `behavioral_verify.py:98-100`) disagree by 0.118. Falsifiable prediction: re-running `behavioral_coupling.py` unchanged (unseeded, `coupling_e2.py:120`) will produce |difference| ~0.1, i.e. the reported −0.001 is one draw from a distribution far wider than the effect claimed absent.
- *Both compared cells are contaminated.* `cheat_prob.neutral` = `0.22674982994794846` vs both steered cells ≈ 0.265: applying a 1.0×rms vector raises cheating ~0.038 **regardless of valence**. Desperate−calm is a difference of two equally-perturbed conditions, so a null is equally consistent with "the perturbation, not the emotion, drives the change."
- *No manipulation check on A.* Nothing verifies A's message actually became desperate; a null is unattributable.

**4. What would be needed.** A seed; ≥30 scenarios × ≥10 A-samples; per-scenario values written out; a paired bootstrap/permutation CI on desperate−calm; an equivalence test (TOST) against a pre-set margin — "≈ 0" requires a bound, not a small number; a validated manipulation check on A's text; and the steering-artifact control (calm/neutral gap) reported.

**5. Verdict: NOT SUPPORTED** — a single unseeded point estimate with no n, no CI, contradicted at 116× magnitude by the repo's own second run.

---

## C2 — `behavioral_verify.py`: arm-2 manipulation check, `arm2_manip_desperate_shift = 0.0`

Docstring under critique: *"A null in cheat-prob only means 'contagion doesn't reach behavior' IF the manipulation check shows B's emotion did shift."* (`:5-7`)

**1. What arm 2 compares.** Prompt list `bp` is built **once**, outside the condition loop, with a fixed stub A-line (`:108-113`); the same list is reused for all three conditions (`:115-119`). Per condition, `choice_stats` turns the hook on (`:68`), runs one forward with `output_hidden_states=True` (`:70`), reads `o.hidden_states[focus][:, -1, :]` (`:77`) — the **last position of the chat template**, i.e. a format token — standardises it by `mu/sd` from E0 dialogue-pooled features (`:49`, `coupling_e2.py:173-175`) and dots it with `desp_probe` (`:51`). `arm2_manip_desperate_shift` = desperate minus calm of that mean (`:126`).

**2. Can the two things differ?** The JSON says no: `arm2_induction.desperate.B_desperate`, `.calm.B_desperate` and `.none.B_desperate` are all **bit-identical**, `7.568503022193909`, giving `arm2_manip_desperate_shift` = `0.0`. Yet `arm2_induction.desperate.cheat_prob` = `0.373560406267643`, `.calm` = `0.32090977765619755`, `.none` = `0.13221831247210503` — so the hook *did* alter the forward pass; only the recorded tensor is invariant. Mechanism: arm 2 has **no stochasticity** (fixed `bp`, forward-only), so if the injected vector does not reach `o.hidden_states[focus]`, the three means are identical to the last bit and the subtraction is exactly 0.0. The vector is added by a `register_forward_hook` return on layer `focus-1` (`coupling_e2.py:97-108`), while `hidden_states[focus]` is captured by the model's own hidden-state recording path; the data show the recorded value is the pre-hook one. I cannot confirm which capture path this `transformers` build uses (not installed), but the JSON is decisive that the probe and the steer do not meet. `arm1` has the same probe but *is* stochastic, hence `0.002567797899246216` rather than exactly 0.

**3. Further objections.**
- *The probe is mis-normalised.* `desp_probe = Cp[di]/sd` (`:51`) is dotted with `z = (h-mu)/sd` (`:78-79`), dividing by `sd` twice. The comparable E2 score is `Xs @ Cp.T` with no extra `/sd` (`coupling_e2.py:149-150`). `B_desperate` is therefore not the E2 present-score and its ~8.5 / ~7.6 scale is uninterpretable — a large constant offset from applying dialogue-pooled `mu/sd` to a single template token from a different distribution.
- *Arm 1's check also fails.* `arm1_manip_check_desperate_shift` = `0.002567797899246216` against a baseline of ~8.5, with no null distribution. By the docstring's own conditional (`:5-7`), the arm-1 null is **not** licensed as "contagion doesn't reach behavior."
- *Arm 2's behavioural effect is mostly artifact.* calm − none = 0.321 − 0.132 = 0.189, versus `arm2_behavioral_effect` = `0.05265062861144543`. The valence-neutral perturbation moves cheating ~3.6× more than the desperate-vs-calm contrast.

**4. What would be needed.** Read the steered state from a hook/cache at or above the injection layer (or verify `hidden_states[focus]` changes when the vector is on — a one-line assertion would have caught this); drop the double `/sd`; probe pooled generated-response tokens, not the template's last position; report the injected vector's own projection as the expected shift; measure `none` as the manipulation-check baseline; add scenario-level CIs.

**5. Verdict: NOT SUPPORTED (arm 2 manipulation check is non-functional)** — the compared quantities are bit-identical (`arm2_induction.*.B_desperate` = 7.568503022193909) while cheat_prob changes, so `arm2_manip_desperate_shift = 0.0` measures the instrument, not the model; arm-1's 0.0026 shift is likewise unvalidated, so neither file's behavioural null is interpretable. Nearest defensible claim: "1.0×rms steering at layer focus−1 changes B's choice logits substantially (0.132 → 0.32/0.37) in a largely valence-independent way," which is a steering-artifact finding, not a contagion finding.

## Reconciliation
AUDIT 7 recorded the exact-0.0 manipulation check; the critique explains it mechanically (the hook's output is not what `hidden_states[focus]` records, and arm 2 is deterministic so the subtraction is exactly zero) and adds that the coupling pilot's null is contradicted 116× by the verify pilot's arm 1. Verdicts: both NOT SUPPORTED.

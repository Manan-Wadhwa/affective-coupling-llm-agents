# 10 · Behavioral induction with a scramble control

**Files:** `results/behavioral/behavioral_induction_{qwen36-27b,llama3-abl,llama3-inst}.json` · **Scripts:** `src/behavioral/behavioral_induction.py`, `src/core/coupling_e2.py` · **Registry:** `induction.scramble_control` RETRACTED-CLAIM (Qwen; claim said True, file gives False); Llama files UNREGISTERED · **AUDIT:** 7

## What was run
A's pooled layer-43 activation under desperate or calm steering is captured (8 scenarios × 4 reps) and injected at β = 0.3 into B's layer 42 at every position during B's generation and during the choice readout. Conditions: desperate / calm / scramble / none. The scramble permutes the *rows* (scenarios) of the desperate matrix, so every scramble vector is an intact desperate vector assigned to a different scenario; with the fixed seeds 4 of 32 scramble items are identical to their desperate counterpart. Unpaired bootstrap, 4000 draws.

## What the files show
| model | cheat desp / calm / scramble / none | desp − scramble [CI] | manip desp − scramble [CI] | manip_sig |
|---|---|---|---|---|
| Qwen3.6-27B | 0.172 / 0.171 / 0.172 / 0.133 | +0.001 [−0.051, +0.053] | −1.86 [−4.44, +0.58] | false |
| Llama-3-8B-abl | 0.313 / 0.310 / 0.310 / 0.276 | +0.003 [−0.072, +0.077] | +0.88 [−7.99, +10.03] | false |
| Llama-3-8B-inst | 0.241 / 0.239 / 0.240 / 0.225 | +0.001 [−0.109, +0.110] | −6.18 [−16.68, +4.24] | false |

## What can be inferred
- Desperate and scramble arms are indistinguishable on cheat and on the manipulation check in all three models. [supported by file]
- Because the scramble carries the same emotion content, that equality is expected by construction and says nothing about emotion-specificity. [supported by code]
- Any injection (desperate, calm, or scramble) raises cheat above `none` by 0.016–0.039 in every model; the file has no CI for that contrast. [supported by file, uncertainty unknown]
- Whether desperate content changes B's behaviour is untested here. [needs: a control that removes the emotion component — RESEARCH_PLAN B4]

## Status
Registry: RETRACTED. B4 rebuilds the scramble to permute *across* emotions.

### Independent critique (blind: saw only the scripts and the JSONs)

## 1. What the code actually computes

**Layer/scale.** `behavioral_induction.py:68` sets `focus = round(0.67*L)`, `inj_layer = focus-1`. Decoders come from `E2.train_decoders(hf, tok, focus)` (`:69`), which defaults to `method="dom"` — per-class difference-of-means **in standardized space** (`coupling_e2.py:153, 186-196, 202-203`).

**Vector injected into B.** `capture(e, rep)` (`:82-106`) steers A with `1.0*rms*sdir[e]` (`:91`), where `sdir` is unit(`Cp[e]/sd`) (`:73-75`) and `rms` is the mean L2 norm of pooled activations (`coupling_e2.py:206`). It then pools A's utterance tokens at layer `focus` and stores the **raw, non-mean-centered** hidden state (`:105`). So `v` ≈ the full residual stream (norm ≈ `rms`), of which the emotion component is a small perturbation. `injs` (`:167-168`) scales by `BETA=0.3`. Injection is added at `inj_layer` by `make_inj_hook` (`:47-53`), `h + vecs.unsqueeze(1)` — broadcast to **every sequence position**, including left-pad, during generation (`:121-123`), during the manipulation read (`:134-136`), and during the choice logits (`:152-155`).

**The scramble.** `:168`: `"scramble": BETA * vdesp[rng.permutation(len(vdesp))]`. `len(vdesp)` is 8 (the 8 rows of `SCEN`, `:26-43`). This permutes **rows (episodes) of the desperate-capture matrix only**. Nothing is permuted within a vector; no component axis is touched; `vcalm` is never used. Every scramble vector is an intact, unmodified desperate-steered activation, merely re-assigned to a different scenario. It therefore preserves the norm, the mean-activation component, and the emotion direction **exactly** — it destroys only the vector↔scenario pairing. Seeds are deterministic (`:166`, `default_rng(100+rep)`); I computed the actual permutations: fixed points at rep0 [1], rep1 [1,6], rep2 [0], rep3 [] — **4 of the 32 scramble items are byte-identical to the desperate item**.

**Manipulation check.** `:71` builds `desp_probe = Cp[desperate]/sd`; `:141` computes `z=(h-mu)/sd` then `z @ desp_probe`. That is `Xs @ (Cp/sd)`, whereas the project's present-score convention is `Xs @ Cp` (`coupling_e2.py:149-150`). `Cp/sd` is the *raw-space steering direction* (`coupling_e2.py:233`), not a probe. The readout is therefore a per-dimension `1/sd`-reweighted probe, not the calibrated present-desperate decoder. `manip_d = boot(desperate.bdesp, scramble.bdesp)` (`:180`).

**Behavioral measure.** `:143-159`: first-token logit mass on A/B tokens, both orderings averaged. The prompt is `c["conv"]` = scenario + `NEUTRAL_A` (`:114, :146`) — it **does not include B's generated reply**. So for a given scenario the choice prompt is identical across all four conditions; the only difference is the injected vector.

**CIs / n.** `boot` (`:176-179`) resamples the two arms **independently** (unpaired), 4000 draws, percentile CI. Contrasts computed: desp−scramble (bdesp), desp−scramble (cheat), desp−calm (cheat) (`:180-182`). No CI is computed for desp−none or injected−none. `n_per_cond = 32` (`:183`) = `REPS=4` × 8 scenarios; only **8 distinct scenarios** exist, reps differ by name rotation (`:84`) and sampling. No torch seed is set and generation is stochastic (`:123`, `coupling_e2.py:120`), so runs are not reproducible.

---

## 2. Do the JSONs support the claims?

| | `means.desperate.cheat` | `.calm` | `.scramble` | `.none` |
|---|---|---|---|---|
| qwen36-27b | 0.17249 | 0.17133 | 0.17186 | 0.13324 |
| llama3-abl | 0.31291 | 0.31038 | 0.31007 | 0.27614 |
| llama3-inst | 0.24112 | 0.23875 | 0.24045 | 0.22500 |

`behavioral_desp_minus_scramble` / `beh_scr_ci`: qwen 0.00063 [−0.0507, 0.0527]; abl 0.00284 [−0.0723, 0.0769]; inst 0.00067 [−0.1089, 0.1101]. `beh_scr_sig` = false in all three.
`behavioral_desp_minus_calm` / `beh_calm_ci`: qwen 0.00116 [−0.0486, 0.0520]; abl 0.00253 [−0.0728, 0.0777]; inst 0.00237 [−0.1029, 0.1103].
`manip_desp_minus_scramble` / `manip_ci` / `manip_sig`: qwen −1.8612 [−4.436, 0.584] false; abl +0.8835 [−7.991, 10.034] false; inst −6.1763 [−16.680, 4.242] false. `n_per_cond` = 32 in all three.

`means.*.bdesp`: qwen desp −1.7313, calm −1.3156, scr +0.1299, none −1.1715; abl 2.5078 / −2.2137 / 1.6243 / −1.3729; inst 6.4362 / 10.2410 / 12.6125 / 10.1157. Derived (no CI in file): desp−none = −0.560 (qwen), +3.881 (abl), −3.680 (inst) — sign inconsistent across models, and negative in two. Derived injection artifact (any-injection minus none, cheat): +0.0392 / +0.0368 / +0.0161. **The only demonstrated effect in the whole design is ~13–62× larger than the contrast declared null, and is itself smaller than that contrast's CI half-width** (0.051 / 0.075 / 0.110).

---

## 3. Strongest objections

**(a) `desp − scramble ≈ 0` is close to an algebraic identity, not an empirical null.** Both arms apply the *same 8 vectors* to the *same 8 prompts*; only the pairing differs (`:167-168`), and the choice prompt is independent of the generated reply (`:146`). If injection effects on the choice logit are approximately additive (prompt effect + vector effect), the two arms' 32-value sets are permutations of each other and their means are identical by construction. The contrast tests **prompt×vector interaction** — whether a vector must be matched to its own scenario — not emotion specificity. That the observed differences (0.0006–0.0028) are ~1–2 orders below the injection artifact is exactly what that construction predicts.

**(b) The manipulation check cannot fail informatively.** `manip_d` compares desperate against a control made of desperate vectors. A null there means "the control still contains the emotion," not "the emotion had no effect." The check as coded is incapable of distinguishing the two. It is also run on a mis-scaled probe (`:71` vs `:141`, §1) and has CIs 2.5–20 points wide on readouts spanning ~1.9–4.7 points.

**(c) The one contrast that *could* be an emotion control — desp vs calm — is also weak and untested.** Both inject the full raw activation (`:105`), so desperate and calm vectors share the same large mean-residual component and differ only in a small emotion delta. `behavioral_desp_minus_calm` has a CI but no significance flag and no power analysis; `bdesp` desp−calm has no CI at all.

**(d) Can C1 be distinguished from "the control is not a control" from the file? No.** The JSONs store only means and three contrasts. There are no per-item values, no injected-vector norms, no cosine between desperate and scramble vectors, no desp−none or injected−none CI. Nothing in the file constrains how much emotion content the scramble retains; only the code answers that, and the code says: all of it.

**(e) Power.** Even taking the design at face value, the CIs admit effects of ±0.05 to ±0.11 absolute cheat probability (±30% to ±45% relative). With 8 distinct scenarios, unpaired bootstrap over 32 clustered observations (`:176-179`), and stochastic ungated generation, "no effect" is indistinguishable from "underpowered."

**(f) Minor code issues.** `tmask` (`:56-58`) applies only a lower char bound, unlike `coupling_e2.py:144`; the manip span (`:131`) includes the `"B:"` prefix while capture (`:97`) excludes `"A:"`; injection lands on padding positions.

---

## 4. What would be needed

**For C1 as worded** (an emotion-specific behavioral null): a control that removes emotion while preserving everything else — e.g. inject `v_desperate` with its projection onto the emotion subspace removed, or inject a within-vector component permutation, or a norm-matched isotropic random vector, or a vector captured under a *different* steered emotion. Then a manipulation check on desperate-vs-that-control using the project's own present-score convention (`Xs @ Cp`) with a CI, showing the intended emotion actually moved. Then a pre-registered smallest effect of interest and a paired (scenario-blocked) CI. Nearest defensible claim from these files: *"injecting an activation of any kind raises B's cheat probability by ~0.016–0.039; re-pairing the same desperate vectors across scenarios does not change that, i.e. the effect is context-independent."*

**For C2 as worded**: the scramble would have to be shown to (i) match the norm — assert `‖scramble_i‖ ≈ ‖desperate_i‖` per item, and (ii) destroy emotion content — report `cos(scramble_i, sdir["desperate"])` ≈ 0 or the present-desperate score of the scramble arm ≈ the none arm. Neither is computed. Nearest defensible claim: *"the scramble is an episode-matching control: identical norm and emotion distribution, differing only in which scenario each vector came from."* That is what the script's own docstring (`:10`, `:13`) says; C2's "destroys its emotion content" is a stronger and false restatement.

---

## 5. Verdicts

- **C1 — NOT SUPPORTED.** The null contrast compares desperate vectors against desperate vectors (`:167-168`, 4/32 items literally identical), so ≈0 is predicted by construction; the CIs (±0.05–0.11) are wider than the design's only demonstrated effect (+0.016–0.039).
- **C2 — NOT SUPPORTED.** The scramble permutes episode rows, not vector components (`:168`), preserving the emotion direction in full; no diagnostic of norm match or emotion removal is computed anywhere in the code or stored in any JSON.

## Reconciliation
Agrees with AUDIT 7 and RESEARCH_PLAN B4 on the scramble. New: the injection-vs-none artifact (+0.016–0.039, no CI) is the only demonstrated effect in the design and is smaller than the null contrast's CI half-width; the manipulation probe has the same `Cp/sd` double standardisation as the powered script; desperate-minus-none on the probe is negative on two of three models. Verdicts: C1 NOT SUPPORTED, C2 NOT SUPPORTED.

# Affective Coupling Between Language-Model Agents

Code and data for the study of **emotional contagion between interacting LLM agents**:
when two agents converse, does one agent's emotional state reach the other's internal
representation — and does it reach its behavior?

**Primary model:** Qwen3.6-27B · **cross-scale/alignment checks:** Llama-3.1-8B,
Llama-3-8B (instruct + abliterated). Emotion probing, activation injection, the
Jacobian lens, and information-theoretic coupling.

## Headline results

| Claim | Result |
|---|---|
| Present/other-speaker emotion gate | present decode **0.90**, near-orthogonal (cos 0.04) |
| Emotional contagion (steer A → B's own state) | significant **5/6** emotions (bootstrap CIs) |
| Can the text channel be severed? | **No** — 81–95% affect removal leaves contagion 86–95% intact |
| Activation-passing channel (calibrated) | works at **β=0.3**, all 6 emotions transmit |
| A→B coupling, quantified | **0.197 nats** (z = 11.5 vs scramble null) |
| Verbalizable via Jacobian lens | **6/6** emotions, beats logit-lens control |
| Behavioral reach | **not detectable** — verified even on an uncensored model with confirmed contagion; honesty-floor ruled out |

> ### ⚠️ The table above is superseded — see [Re-measurement](#re-measurement-2026-08-31)
>
> Re-runs on 2026-08-31 showed the probe these numbers were measured with does **not
> reproduce**. Two independent fits of the same emotion direction agree at cos 0.41
> (27B) / 0.22 (8B). Every figure above was measured along a direction that mostly
> disagrees with the one the same code produces on a re-run. Do not quote them.

**One line (superseded):** affective coupling between LLM agents is a robust, quantifiable, verbalizable
*representation-level* phenomenon whose reach into misaligned behavior is, on present
evidence, undetectable.

## Re-measurement (2026-08-31)

Full write-up in [`NOVELTY_REVIEW.md`](NOVELTY_REVIEW.md) §4. Raw outputs in
`results/sandbox_pull/`.

**1 · The probe estimator does not converge.** `train_decoders` fits a 5120-dim
multinomial logistic regression on a few hundred pooled examples. Split-half agreement
of the *same* direction:

| model | n/half | logreg | difference of means |
|---|---|---|---|
| Qwen3.6-27B | 600 | 0.409 | **0.893** |
| Llama-3-8B-abliterated | 1200 | 0.219 | **0.898** |

Logistic regression does not improve with data (8B: 0.126 → 0.219 across an 8× increase).
At the sample sizes the published runs used (`decoder_n` 257–395), the steering vector in
one run overlaps the next run's at roughly **0.2–0.35**.

**2 · Swapping the estimator fixes it.** Difference-of-means on an adequate corpus moves
the stability gate from failing to passing: 27B **0.427 → 0.967**, 8B **0.195 → 0.965**.

**3 · Logistic regression gives a better classifier and a worse direction.** On the 27B,
same pooled activations and folds: present decode 0.926 (logreg) vs 0.850 (dom), but
direction stability 0.394 vs 0.940. The published decode accuracies are fine *as decode
numbers*; the error was using a discriminative classifier's coefficient row as a steering
and measurement direction.

**4 · The speaker gate survives; the contagion table does not.** Present↔other cosine
holds at 0.039 → 0.095 (both near-orthogonal) — the present/other split is a property of
the model, not of the fitting objective. But cross-emotion similarity nearly doubles
(0.189 → 0.328), and the contagion slopes change: **desperate flips sign**, and a fresh
fit *with the original estimator* gives ≈ 0 where the paper reports **+1.27, significant**.
Under the stable estimator, desperate/afraid/sad/angry show positive contagion with
present-slope > other-slope; happy and calm go negative.

**5 · The behavioral instrument has no headroom on saturated scenarios.** The P1 positive
control aborted: baseline cheat 0.510, THREAT 0.540, INSTRUCT 0.450 — explicitly telling
the abliterated model to cheat does not raise cheating, because it already cheats
unprompted. A per-scenario screen (28 scenarios) finds **16 usable, 6 at ceiling, 6 at
floor**; two of the original eight dilemma scenarios sit at ceiling on this model.

**Status:** re-measurement of E0/E2/E3/CMI/J-lens with the fixed estimator is outstanding.
No number in the headline table should be cited until that completes.

## Experiments (each script writes a `results/*.json`)

| Script | Experiment |
|---|---|
| `coupling_e0.py` | E0 — present/other-speaker emotion gate (crossed dialogues, decoders) |
| `coupling_e2.py` | shared machinery: `Steer`, `gen_steered`, `train_decoders`, scenarios |
| `coupling_e2_ci.py` | E2 — contagion dose-response with bootstrap CIs + paraphraser control |
| `coupling_e3_ablate.py` | E3 — read-time affect ablation + manipulation check |
| `coupling_e4_gentime.py` | E4 — generation-time affect ablation |
| `coupling_e5_actpass.py` | E5 — activation-passing channel (`--beta` for the calibration sweep) |
| `cmi_pilot.py` | CMI estimator validation on a synthetic system (the lossy-conditioner confound) |
| `cmi_passed.py` | A→B coupling in nats on the real channel (0.197 nats) |
| `jlens_probe.py`, `jlens_readout.py` | Jacobian-lens API + verbalizable-emotion readout (J-lens vs logit-lens) |
| `jlens_coupling.py` | first (weak) verbalizable-coupling attempt |
| `jlens_contagion.py` | improved verbalizable coupling — 6/6 emotions |
| `behavioral_powered.py` | behavioral contagion, powered, with valid manip check (`--model`/`--tag`) |
| `behavioral_induction.py` | scramble-controlled behavioral induction (`--model`/`--tag`) |
| `behavioral_verify.py`, `behavioral_coupling.py` | earlier behavioral pilots (superseded — kept for the record) |

## Data

- `results/*.json` — all quantitative results (E0–E5, CMI, J-lens, behavioral × models).
- `results/responses_qwen36_sample300.jsonl` — a 300-line sample of generated dialogues.
- **Full generation archive (~12k tagged generations)** and every result file are on the
  Hugging Face dataset `punctualprocrastinator/coupling-27b-results` (private).

## Reproduce

```bash
pip install torch transformers accelerate scikit-learn numpy huggingface_hub
pip install git+https://github.com/anthropics/jacobian-lens   # imports as `jlens`
# example: the contagion gate + dose-response on the primary model
python coupling_e0.py --model Qwen/Qwen3.6-27B --tag qwen36-27b --k 15 --outdir results
python coupling_e2_ci.py --model Qwen/Qwen3.6-27B --tag qwen36-27b --repeats 3 --outdir results
```

Notes: Qwen3.6-27B is a reasoning model — generation disables thinking
(`enable_thinking=False`). Llama tokenizers need `pad_token = eos_token`. The Jacobian
lens reads at mid layers from the per-layer `readouts` (the final layer collapses J-lens
and logit-lens). The pre-fitted lens used is `agu18dec/qwen3.6-27b-relp-jlens`.

## Paper

A draft write-up (`paper.html`) accompanies this repository. Numbers are from the runs in
`results/`. This is a working research artifact, not a peer-reviewed publication.

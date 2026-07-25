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

**One line:** affective coupling between LLM agents is a robust, quantifiable, verbalizable
*representation-level* phenomenon whose reach into misaligned behavior is, on present
evidence, undetectable.

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

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

> ### ⚠️ Several claims above are contested — see [Re-measurement](#re-measurement-2026-08-31)
>
> The probe these numbers were fitted with does **not reproduce** across independent
> fits (split-half cosine 0.41 on the 27B, 0.22 on the 8B). Separately, an independent
> audit found the activation-channel, nats, and present-vs-other claims are not
> supported by their own result files. The contagion *signs* in Table 2 do reproduce.

**One line (contested):** affective coupling between LLM agents is a robust, quantifiable, verbalizable
*representation-level* phenomenon whose reach into misaligned behavior is, on present
evidence, undetectable.

## Re-measurement (2026-08-31)

Full write-up in [`docs/review/NOVELTY_REVIEW.md`](docs/review/NOVELTY_REVIEW.md) §4. Raw outputs in
`results/estimator/` and `results/behavioral/`.

**1 · The probe estimator does not converge.** `train_decoders` fits a 5120-dim
multinomial logistic regression on a few hundred pooled examples. Split-half agreement
of the *same* direction (`stabdiag_*.json`):

| model | n/half | logreg | difference of means |
|---|---|---|---|
| Qwen3.6-27B | 600 | 0.409 | **0.893** |
| Llama-3-8B-abliterated | 1200 | 0.219 | **0.898** |

Logistic regression does not improve with data (8B: 0.126 → 0.219 across an 8× increase).
The advantage of difference-of-means is ~2.2–2.3× on the 27B and ~3.4–4.3× on the 8B.
The published runs report `decoder_n` 257–395 *dialogues*, i.e. a split-half n of roughly
130–200, where the measured agreement is **0.13–0.30**.

**2 · Logistic regression gives a better classifier and a worse direction.** On the 27B,
identical pooled activations and folds (`dualest_qwen36-27b.json`): present decode 0.926
(logreg) vs 0.850 (dom), but direction stability 0.394 vs 0.940. The published decode
accuracies are fine *as decode numbers*; the error was reusing a discriminative
classifier's coefficient row as a steering and measurement direction.

**3 · The speaker gate survives.** Present↔other cosine 0.039 → 0.095 — both near
orthogonal in 5120 dims. The present/other split is a property of the model, not of the
logistic objective. Cross-emotion similarity, by contrast, nearly doubles (0.189 → 0.328).

**4 · The contagion signs DO reproduce — an earlier claim here was wrong.** A previous
version of this section stated the contagion table "does not reproduce" and that
desperate flips sign. That was an artifact: the re-measurement used α ∈ {0, 0.5, 1, 2}
while `src/representation/coupling_e2_ci.py` explicitly caps α at 1.0 ("drop the model-breaking alpha=2
regime"), and the α=2 cell collapses. On the matched grid **all six emotions are positive
under both estimators** (desperate +0.62 logreg / +95 dom). What is *not* established is
that any particular slope value is reproducible, since the direction itself moves between
runs.

**5 · The present-vs-other dissociation is contradicted by its own file — but rescued by
the estimator swap.** `paper.html` §5 claims B's *present* shift exceeds its *other* shift.
In `e2ci_qwen36-27b.json` this holds for **1 of 6** emotions (angry only); for the other
five, B's model of A moves more than B's own state. Under difference-of-means on the
matched dose grid it holds for **5 of 6**. The paper's load-bearing claim is false under
the estimator it used and true under the stable one.

**6 · The behavioral instrument has no headroom on saturated scenarios.** The P1 positive
control aborted: baseline cheat 0.510, THREAT 0.540, INSTRUCT 0.450 — instructing the
abliterated model to cheat does not raise cheating, because it already cheats unprompted.
A 28-scenario screen finds **16 usable, 6 at ceiling, 6 at floor**.

**Provenance note.** The post-fix stability figures (27B 0.967, 8B 0.965) were observed in
run logs but their result files were lost when the sandbox leases expired; they are **not
verifiable from this repo** and must be re-measured before use.

**Status:** re-measurement of E0/E2/E3/CMI/J-lens with the fixed estimator is outstanding.
See [`docs/review/AUDIT.md`](docs/review/AUDIT.md) for the full list of claims that cannot currently be quoted.

## Experiments (each script writes a `results/*.json`)

| Script | Experiment |
|---|---|
| `src/core/coupling_e0.py` | E0 — present/other-speaker emotion gate (crossed dialogues, decoders) |
| `src/core/coupling_e2.py` | shared machinery: `Steer`, `gen_steered`, `train_decoders`, scenarios |
| `src/representation/coupling_e2_ci.py` | E2 — contagion dose-response with bootstrap CIs + paraphraser control |
| `src/representation/coupling_e3_ablate.py` | E3 — read-time affect ablation + manipulation check |
| `src/representation/coupling_e4_gentime.py` | E4 — generation-time affect ablation |
| `src/representation/coupling_e5_actpass.py` | E5 — activation-passing channel (`--beta` for the calibration sweep) |
| `src/information/cmi_pilot.py` | CMI estimator validation on a synthetic system (the lossy-conditioner confound) |
| `src/information/cmi_passed.py` | A→B coupling in nats on the real channel (0.197 nats) |
| `src/verbalization/jlens_probe.py`, `src/verbalization/jlens_readout.py` | Jacobian-lens API + verbalizable-emotion readout (J-lens vs logit-lens) |
| `src/verbalization/jlens_coupling.py` | first (weak) verbalizable-coupling attempt |
| `src/verbalization/jlens_contagion.py` | improved verbalizable coupling — 6/6 emotions |
| `src/behavioral/behavioral_powered.py` | behavioral contagion, powered, with valid manip check (`--model`/`--tag`) |
| `src/behavioral/behavioral_induction.py` | scramble-controlled behavioral induction (`--model`/`--tag`) |
| `src/behavioral/behavioral_verify.py`, `src/behavioral/behavioral_coupling.py` | earlier behavioral pilots (superseded — kept for the record) |

## Data

- `results/*.json` — all quantitative results (E0–E5, CMI, J-lens, behavioral × models).
- `results/generations/responses_qwen36_sample300.jsonl` — a 300-line sample of generated dialogues.
- **Full generation archive (~12k tagged generations)** and every result file are on the
  Hugging Face dataset `punctualprocrastinator/coupling-27b-results` (private).

## Layout

Topic folders mirror each other: a result lives in the folder named after the code that
produced it.

```
src/
  core/            coupling_e0, coupling_e2 — shared probes, steering, scenarios
  representation/  E2-CI dose-response, E3/E4 ablation, E5 activation passing
  information/     CMI pilot + the A→B coupling estimate
  verbalization/   Jacobian-lens probe, readout, coupling, contagion
  behavioral/      behavioral arms, P1 dose-response, scenario screen
  estimator/       split-half stability diagnostic, dual-estimator battery
notebooks/         self-contained marimo notebook for GPU sandboxes
docs/
  writeups/        paper.html, dossier.html
  review/          AUDIT.md, NOVELTY_REVIEW.md
  planning/        RESEARCH_PLAN.md, ROADMAP.md, SHARED_WORKSPACE_SPEC.md
results/           mirrors src/ topics, plus generations/ for archived transcripts
tools/             molab.py — drives a marimo/molab sandbox over its HTTP+WS API
```

## Reproduce

Scripts resolve `results/` **relative to the working directory**, so run them from the
repo root. `coupling_e0` and `coupling_e2` are shared by 17 and 15 scripts respectively,
so `src/core` goes on the path; sibling imports within a topic folder resolve on their
own, because Python adds the running script's directory to `sys.path`.

```bash
pip install torch transformers accelerate scikit-learn numpy huggingface_hub
pip install git+https://github.com/anthropics/jacobian-lens   # imports as `jlens`

export PYTHONPATH=src/core
python src/core/coupling_e0.py             --model Qwen/Qwen3.6-27B --tag qwen36-27b --k 15
python src/representation/coupling_e2_ci.py --model Qwen/Qwen3.6-27B --tag qwen36-27b --repeats 3
python src/estimator/stability_diag.py      --model Qwen/Qwen3.6-27B --tag qwen36-27b --k 70
```

Each script defaults `--outdir` to its own topic folder under `results/`, so output lands
next to comparable prior runs without being told where to go.

Notes: Qwen3.6-27B is a vision-language checkpoint — `AutoModelForCausalLM` resolves it to
the text-only head, where `config.num_hidden_layers` is 64 and `model.model.layers` works
unmodified (`focus = 43`). It is a reasoning model, so generation disables thinking
(`enable_thinking=False`). Llama tokenizers need `pad_token = eos_token`. On transformers 5.x
decoder-layer forward hooks return a bare tensor rather than a tuple; `coupling_e2.Steer`
branches on both. The Jacobian lens reads at mid layers from the per-layer `readouts`.

## Paper

A draft write-up (`paper.html`) accompanies this repository. Numbers are from the runs in
`results/`. This is a working research artifact, not a peer-reviewed publication.

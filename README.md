# Affective Coupling Between Language-Model Agents

Code and data for the study of **emotional contagion between interacting LLM agents**:
when two agents converse, does one agent's emotional state reach the other's internal
representation — and does it reach its behavior?

**Primary model:** Qwen3.6-27B · **cross-scale/alignment checks:** Llama-3-8B
(instruct + abliterated). *No Llama-3.1 artifact exists in this repo; earlier text citing
Llama-3.1-8B was wrong.* Emotion probing, activation injection, the
Jacobian lens, and information-theoretic coupling.

## Where we stand — rev 3, 2026-09-05

The forward-looking summary is [`docs/planning/RESEARCH_PROPOSAL.md`](docs/planning/RESEARCH_PROPOSAL.md);
the operative plan with its status board is [`docs/planning/RESEARCH_PLAN.md`](docs/planning/RESEARCH_PLAN.md);
every result has a report and a blind critique in [`results/reports/`](results/reports/README.md), and every
number a pointer in [`docs/review/claims.json`](docs/review/claims.json) (76 entries, 0 quotable failing).
What the blind critiques changed is in [`docs/review/CRITIQUE_LEDGER.md`](docs/review/CRITIQUE_LEDGER.md).

| finding (rev 3) | number | registry · report |
|---|---|---|
| Difference-of-means directions reproduce across independent fits; logistic directions do not | 27B focus layer, n = 2000/half: dom 0.974 vs logreg 0.566; 8B: 0.941 vs 0.286 | `a2.focus_*`, `a2_8b.*` · 16, 19 |
| A fixed per-sample penalty does not make the logistic direction converge with n | 27B layer 43: no net rise 150 → 2000 at two anchors; layer 54: one drop then flat; 8B: 0.37 → 0.28 | `a2f.*`, `a2f2.*`, `a2f54.*`, `a2f8.*` · 21, 23–25 |
| Why: every logistic fit at every n used is a perfect separator; the direction is the penalty's tie-break and drifts away from the mean difference as n grows | accuracy 1.000 everywhere; cos(logistic, dom) 0.97 → 0.63 (27B), 0.91 → 0.43 (8B) | `a2m.*` · 26 |
| Affect transfer between agents exists and is dose-dependent (representation level) | 5 of 6 emotions, reproduced three times with identical directions | `b1f.*`, `b1c.*`, `b1d.*` · 20, 22, 27 |
| The rank-1 emotion direction is not the channel (pre-registered H1) | median 9% of the transfer removed at all 51 layers [4%, 25%]; afraid 21%, sad 33% | `b1c.*` · 22 |
| A rank-5 affect subspace is not a clean instrument, and a fitted label-free control blocks as much as the emotion direction (pre-registered instrument_failed) | permuted control: afraid −49.5, sad −26.7, angry −44.3 vs emotion direction −51.9, −27.7 | `b1d.*` · 27 |
| A fitted label-free rank-1 control is inert and the emotion direction blocks afraid and sad against it (pre-registered H3); the control's footprint is below the emotion direction's, so the footprint caveat stands | afraid 25% [17%, 33%], sad 29% [7%, 56%]; pc1 (5.8× footprint) blocks angry 61%, not afraid/sad | `b1e.*` · 28 |
| B1e's design on the 8B: the control is footprint-matched (0.96×) and inert; sad is blocked 36% against it (afraid 14%, marginal) | sad −36.8 [−57.1, −16.7] | `b1e8.*` · 30 |
| The transfer pattern and the same two blocked emotions replicate on the 8B (B1c's design; the strict manipulation-check rule fails under a null upper window) | afraid 15%, sad 31%; the rest too wide to tell | `b1c8.*` · 29 |

Not quotable: anything about the text channel (three rewrite designs failed their own checks); the
behavioural reach; the earlier headline results listed under "What is retracted" below.

## Status — Phase 0, repairing the record

Rev 3 of the research plan opens with a phase that produces no new science and holds the
artifact until the record is repaired. Most of what used to be in this space was a table of
seven headline results. An independent claims audit ([`docs/review/AUDIT.md`](docs/review/AUDIT.md),
eighteen findings) found that four of the seven do not survive contact with their own
result files, so the table has been replaced by the registry below rather than annotated
in place.

**Every number in this repository now carries a provenance pointer** — script, config,
output file, and, for any claim that depends on what a control does, the *lines* that
implement the control. The pointers are machine-checkable:

```bash
python tools/check_provenance.py            # all claims
python tools/check_provenance.py --status quotable
```

The registry is [`docs/review/claims.json`](docs/review/claims.json). It separates two
questions that were previously conflated: **does the number resolve** (does its pointer
lead to a committed file that contains it) and **is the sentence built on it supported**.
E3's arithmetic, for example, resolves exactly — and the claim built on it is still
retracted, because what the code measured is not what the claim says it measured.

### What is currently quotable

| Claim | Result | Pointer |
|---|---|---|
| Present/other-speaker emotion gate | present decode **0.899**, cross-cos **0.044** | `results/core/e0_qwen36-27b.json` |
| Logistic direction does not reproduce across independent fits | split-half **0.409** (27B, n=600/half) | `results/estimator/stabdiag_qwen36-27b.json` |
| Difference-of-means does | split-half **0.893** (27B), **0.898** (8B, n=1200/half) | `results/estimator/stabdiag_*.json` |
| Better classifier, worse direction | decode 0.926 vs 0.850; stability **0.394 vs 0.940** | `results/estimator/dualest_qwen36-27b.json` |

### What is retracted, and why

| Former headline | Why it does not stand |
|---|---|
| "Can the text channel be severed? **No**" | E3 never regenerates B. B's reply is written from unmodified text (`coupling_e3_ablate.py:154`) and the ablation is applied afterwards, inside a single scoring pass. The measurement is a re-encoding effect over frozen tokens, not a channel measurement. The 81–95 / 5–14 pair was never a budget. |
| "Activation-passing channel works at **β=0.3**, all 6 transmit" | β was selected on its own outcome (three run, only 0.3 gives 6/6), and the specificity controls fail at the selected β. The scramble is broken by construction — it permutes within one emotion, so it preserves the emotion component and cannot test specificity at all. |
| "A→B coupling, **0.197 nats** (z = 11.5)" | The estimator pools six emotions into one matrix and permutes across emotion blocks, so both variables carry a block mean by construction; the statistic mostly recovers *which of six emotions this was*. |
| "Verbalizable via Jacobian lens, **6/6**, beats logit-lens" | Scores compared across unnormalised scales, n = 16, no CIs, and the layer was selected on the J-lens's own output. |
| "Behavioral reach not detectable; **honesty floor ruled out**" | CI [−0.08, +0.13] at n = 32 on a 0.31 base rate rules out nothing, and no MDE is computed anywhere. The model it rests on failed its own positive control. |
| "Present shift exceeds other shift" (paper §5, the *load-bearing* result) | Holds for **1 of 6** emotions in its own file, and no test of the difference is run. |
| "Contagion accumulates over 8 turns" | It accumulates identically **without** steering; calm's slope exceeds desperate's and the contrast is n.s. |

### Corrections of fact

- **Model identity.** Earlier text cited *Llama-3.1-8B*. No Llama-3.1 artifact exists in
  this repository; every Llama result is Llama-3 (instruct or abliterated). The claim is
  corrected regardless of whether the numbers are re-derived.
- **Dataset.** Earlier text said "all results and the ≈13k archived generations are
  released". The repository holds 300 E0 dialogues and 200 multi-turn transcripts; the
  full archive is not public. Publishing it is a Phase 0 deliverable (§0.6), because it is
  the only thing that lets anyone outside the team check the ablation.
- **Two figures have no committed source** and are pending re-measurement, not quotable:
  the post-fix stability gates (27B 0.967, 8B 0.965). They were observed in run logs whose
  sandbox leases expired before the files were pulled. Only the *pre*-fix 8B value (0.195)
  is committed.
- ***Subliminal Learning*** is cited as motivation only. It is train-time transfer between
  models sharing an initialization, and the mechanism depends on that parameter-space
  proximity; inference-time inter-agent transfer has no analogue.

## Rebuild in progress (rev 3)

`src/lib/acl_core.py` is the single vendored core for the rebuilt experiments — one file,
no repo-relative imports, so it drops into a bare GPU sandbox and still produces results
byte-comparable with a local run. It carries the estimator battery, the controls register
(norm-matched random and orthogonal directions, degeneracy scoring, generation-side
manipulation checks, the frozen-token audit), scenario-blocked bootstrap statistics, and
provenance stamping.

| Driver | Phase | What it settles |
|---|---|---|
| `src/rev3/a2_estimator.py` | A2 | split-half stability across layer × estimator × n, with CIs, plus the regime diagnostics that the n/d story alone does not explain |
| `src/rev3/b1_e4rerun.py` | B1 | the E4 rerun: difference-of-means directions, a **generation-side** manipulation check, norm-matched random *and* orthogonal controls, a declared dose grid, degeneracy at every dose |
| `src/rev3/b1_followup.py`, `b1c_alllayer.py`, `b1d_subspace.py`, `b1e_footprint.py` | B1 family | ceiling/text arms and seeded B (follow-up); pre-registered all-layer rank-1 ablation (B1c); pre-registered rank-5 subspace ablation with permuted-label and random-frame controls (B1d); pre-registered fitted label-free rank-1 control (B1e) |
| `src/rev3/a2_followup.py`, `a2_margin_diag.py` | A2 add-ons | tuned-C, fixed per-sample penalty at two anchors, like-for-like cross-estimator; separability / margin / penalty-sensitivity diagnostic |
| `src/rev3/reblock_contrasts.py`, `b1d_frame_geometry.py` | re-reads | every rev-3 contrast under the scenario-blocked, dose-paired bootstrap; geometry of the B1d frames |

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

- `results/**/*.json` — every committed result file, indexed with its status (quotable / retracted-claim / unregistered / partial) in [`results/README.md`](results/README.md). Not every file backs a quotable claim.
- `results/generations/responses_qwen36_sample300.jsonl` — a 300-line sample of generated dialogues.
- The full generation archive and every result file are on the Hugging Face dataset
  `punctualprocrastinator/coupling-27b-results`, which is **private**. Publishing it is a
  Phase 0 deliverable (§0.6). Counts quoted previously (~13k / 12,333 / ~12k) disagreed
  with each other and with the repo; none should be quoted until the dataset is public.

## Layout

Topic folders mirror each other: a result lives in the folder named after the code that
produced it.

```
src/
  lib/             acl_core.py — single vendored core for the rev-3 rebuild
  rev3/            a2_estimator, a3_dissociation, b1_e4rerun; b1_analyze / a2_analyze read checkpoints; splithalf_independent
  core/            coupling_e0, coupling_e2 — shared probes, steering, scenarios
  representation/  E2-CI dose-response, E3/E4 ablation, E5 activation passing
  information/     CMI pilot + the A→B coupling estimate
  verbalization/   Jacobian-lens probe, readout, coupling, contagion
  behavioral/      behavioral arms, P1 dose-response, scenario screen
  estimator/       split-half stability diagnostic, dual-estimator battery
notebooks/         self-contained marimo notebook for GPU sandboxes
docs/
  writeups/        paper.html, dossier.html
  review/          AUDIT.md, NOVELTY_REVIEW.md, RESULTS_LOG.md, A0_PRIOR_ART.md, claims.json
  planning/        RESEARCH_PLAN.md, ROADMAP.md, SHARED_WORKSPACE_SPEC.md
results/           mirrors src/ topics, plus generations/ and rev3/ — see results/README.md
                   for the per-file status index
tools/             check_provenance.py (claims registry checker), pull_results.py (sandbox
                   checkpoint puller), molab.py (drives a marimo/molab sandbox)
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

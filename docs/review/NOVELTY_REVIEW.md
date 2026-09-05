# Novelty & Literature Review

> ⟨2026-09-05⟩ Superseded in part by rev 3: the re-measurements below were repeated with the vendored core, pre-registered where a hypothesis was at stake, and critiqued blind. The current claim set is `docs/planning/RESEARCH_PROPOSAL.md`; the numbers are in `docs/review/claims.json` and `results/reports/`. This file is kept as the record of what was found first.

Referee-style review of this repository against the 2025–26 literature, plus the
empirical findings from the 2026-08-31 sandbox runs that materially change the
novelty picture.

**Status:** working document. Numbers cited from this repo were checked against
`results/*.json` and `results/{estimator,behavioral}/*.json`, not against the paper text — in
several places the two disagree, and those are flagged.

---

## 1. Summary judgement

There is a real, novel, well-controlled result here, and it is **not** the one on the
title page. The strongest asset is §6 of `docs/writeups/paper.html`: deleting the emotion direction
from B's reading of A's message across 30 layers removes 81–95% of the readable affect
and blocks only 5–14% of the contagion, for all six emotions, with a random-direction
control that removes ≈0%. Nothing in the current literature does this.

The published headline — "propagates in representation, not detectably in behavior" —
leads with the weakest evidence in the repo, and the 2026-08-31 runs showed the
measurement apparatus underneath it does not reproduce (§4).

---

## 2. Novelty audit, claim by claim

| Claim in the repo | Closest prior art | Verdict |
|---|---|---|
| **E0** — present- vs other-speaker emotion probes separable and near-orthogonal | Anthropic, *Emotion Concepts and their Function in a LLM* ([2604.07729](https://arxiv.org/abs/2604.07729)) §2.3 builds the same 2×2 speaker×turn probe grid and already reports present/other probes "nearly orthogonal" | **Replication** |
| **E2** — steering A moves *B's own* present-emotion representation, dose-dependently | Agent-level contagion exists only text-side: crowd simulation ([2607.25140](https://arxiv.org/abs/2607.25140)), Chain-of-Affective ([2512.12283](https://arxiv.org/abs/2512.12283)), Contagion Networks ([2606.20493](https://arxiv.org/abs/2606.20493)). None probes activations | **Novel** |
| **E3/E4** — verified linear removal of A's affect leaves B's contagion nearly intact | No direct analogue. Nearest in spirit: *Subliminal Learning* ([2507.14805](https://arxiv.org/abs/2507.14805)) — non-semantic trait transfer that data filtering cannot remove — but that is train-time and this is inference-time, between agents | **Strongly novel** |
| **E5** — activation passing as a controllable non-textual channel | Ramesh & Chai ([2501.14082](https://arxiv.org/abs/2501.14082)) established the channel; a latent-communication line follows ([2606.05711](https://arxiv.org/abs/2606.05711), [2605.11167](https://arxiv.org/abs/2605.11167)). Repurposing it as a *measurement* probe is the new part | **Incremental** |
| **CMI** — A→B coupling quantified at 0.197 nats | *Do Latent Channels Actually Communicate?* ([2607.26773](https://arxiv.org/abs/2607.26773)) performs this class of causal audit with richer decompositions (positive signalling / positive listening / causal information contribution / content-attributable gain / self-substitution gap) | **Scooped + flawed** (see §4) |
| **J-lens** — B's caught emotion is verbalizable, 6/6 emotions | First application of the Jacobian lens ([2607.15495](https://arxiv.org/abs/2607.15495)) to a multi-agent setting. Genuinely first; but no uncertainty estimate on any of the six numbers | **Novel but thin** |
| **Behavioral null** — coupling does not reach misbehavior | Anthropic established the within-agent link (desperate steering: reward hacking ~5%→70%). Between-agent is new. But [2604.06562](https://arxiv.org/abs/2604.06562) already reports emotion is "not a stable or model-invariant control knob" for agent decisions, and *Got a Secret?* ([2605.27766](https://arxiv.org/abs/2605.27766)) shows agents 8× more likely to leak after seeing a peer — i.e. the instrument matters enormously | **New, underpowered** |

**The one-line version:** the novelty is not "emotions spread between agents" — three
2026 papers say that. It is "we opened the receiver's head, and what crosses is not the
direction you would filter for."

---

## 3. Literature map

### Emotion representations inside a single model
- **Emotion Concepts and their Function in a Large Language Model** — Anthropic, Transformer Circuits, Apr 2026. [arXiv 2604.07729](https://arxiv.org/abs/2604.07729) · [transformer-circuits.pub](https://transformer-circuits.pub/2026/emotions/index.html)
  Linear emotion concepts; §2.3 present- vs other-speaker probes are near-orthogonal; causal effects on reward hacking (~5%→70% across ±0.1 residual-norm steering), blackmail (22% unsteered baseline), sycophancy. **The direct parent of this repo.**
- **Latent Structure of Affective Representations in LLMs** — [2604.07382](https://arxiv.org/abs/2604.07382). Emotion manifold geometry across 80 layers of LLaMA-3-70B.
- **Where Do Models Find Happiness? Emotion Vectors in Open-Source LLMs** — [2606.26987](https://arxiv.org/abs/2606.26987).
- **How Emotion Shapes the Behavior of LLMs and Agents: A Mechanistic Study** — [2604.00005](https://arxiv.org/abs/2604.00005). Single-agent; emotion→behavior across reasoning/agentic benchmarks.
- **Extracting and Steering Emotion Representations in Small LMs** — [2604.04064](https://arxiv.org/abs/2604.04064).
- **On Emotion-Sensitive Decision Making of Small LM Agents** — [2604.06562](https://arxiv.org/abs/2604.06562). Residual-stream emotion steering across 7 game-theoretic templates, 24+ models. **Key finding: emotion is "not a stable or model-invariant control knob."** An ally for this repo's instability results.

### Emotion / affect between agents (the competitors)
- **How Affect Propagates among LLM Agents: Emergent Emotional Contagion in Crowd Simulation** — [2607.25140](https://arxiv.org/abs/2607.25140). Alarm propagates as a travelling wave; personality determines which emotion the crowd catches. **Prompt-level affect state; no activation probing.**
- **Large Language Models have Chain-of-Affective** — [2512.12283](https://arxiv.org/abs/2512.12283). Majority structure governs propagation direction; role specialisation (initiators / absorbers / firewalls); links stronger propagation to higher bias rates. **Self-report scales and sentiment, not internals.** Reports a *positive* affect→biased-content link that this repo's behavioral null must be reconciled with.
- **Contagion Networks: Evaluator Bias Propagation in Multi-Agent LLM Systems** — [2606.20493](https://arxiv.org/abs/2606.20493). Cross-agent contagion matrix; biases propagate at γ ∈ [0.157, 0.352] even within one base model.
- **Emotion Diffusion in Real and Simulated Social Graphs** — [2512.21138](https://arxiv.org/abs/2512.21138). Structural limits of LLM-based social simulation.
- **EvoEmo: Evolved Emotional Policies for LLM Agents in Multi-Turn Negotiation** — [2509.04310](https://arxiv.org/abs/2509.04310).
- **Spontaneous Emergence of Agent Individuality in LLM-Based Communities** — [2411.03252](https://arxiv.org/abs/2411.03252).

### Non-textual / latent channels between agents
- **Communicating Activations Between Language Model Agents** — Ramesh & Chai, [2501.14082](https://arxiv.org/abs/2501.14082). Pause one model mid-computation, merge its activation into another. +27% over text at <¼ the compute. **Establishes the channel E5 uses.**
- **Do Latent Channels Actually Communicate? A Causal Audit** — [2607.26773](https://arxiv.org/abs/2607.26773). Five causal measures; shows aggregate accuracy hides opposing components. **Directly overlaps the CMI section and does it more carefully.**
- **Beyond Tokens: A Unified Framework for Latent Communication in LLM-based MAS** — [2606.05711](https://arxiv.org/abs/2606.05711).
- **The Bicameral Model: Bidirectional Hidden-State Coupling Between Parallel LMs** — [2605.11167](https://arxiv.org/abs/2605.11167).

### Multi-agent interpretability & safety
- **Detecting Multi-Agent Collusion Through Multi-Agent Interpretability (NARCBench)** — [2604.01151](https://arxiv.org/abs/2604.01151). 50 committee-deliberation scenarios / 17 domains; probes 1.00 AUROC in-distribution, 0.60–0.86 transferred. **The rigor template `docs/planning/SHARED_WORKSPACE_SPEC.md` borrows from.**
- **Got a Secret? LLM Agents Can't Keep It** — [2605.27766](https://arxiv.org/abs/2605.27766) / *Does Safety Molt?* ([ACM](https://dl.acm.org/doi/10.1145/3786335.3813173)). Multi-turn social evaluation raises privacy violations from 19.95%→45.30%; **agents 8× more likely to disclose after seeing a peer do it.** Proof that peer-driven behavioral shifts *are* detectable with a sensitive instrument.

### Probes, steering, and their failure modes
- **Persona Vectors: Monitoring and Controlling Character Traits** — [2507.21509](https://arxiv.org/abs/2507.21509). The generalisation target for a "second latent" study.
- **Subliminal Learning: LMs Transmit Behavioral Traits via Hidden Signals in Data** — [2507.14805](https://arxiv.org/abs/2507.14805) · [Nature](https://www.nature.com/articles/s41586-026-10319-8). Traits transfer through semantically unrelated data; **signals are non-semantic and not removable by filtering.** The strongest framing ally for §6.
- **Steered LLM Activations are Non-Surjective** — [2604.09839](https://arxiv.org/abs/2604.09839). Steered states have no prompt preimage; white-box controllability ≠ black-box reachability. Relevant to whether α = 1–2 ‖h‖ steering describes anything reachable in deployment.
- **Detecting Strategic Deception Using Linear Probes** — [2502.03407](https://arxiv.org/abs/2502.03407).
- **Verbalizable Representations Form a Global Workspace in LMs (Jacobian lens)** — [2607.15495](https://arxiv.org/abs/2607.15495) · [transformer-circuits.pub](https://transformer-circuits.pub/2026/workspace/index.html) · [code](https://github.com/anthropics/jacobian-lens).

---

## 4. What the 2026-08-31 runs changed

Four empirical findings, all reproducible from `results/estimator/` and `results/behavioral/`.

### 4.1 The probe estimator does not converge — a new contribution

`train_decoders` fits a 5120-dim multinomial logistic regression on a few hundred
pooled examples. Two independent fits of the *same* direction agree at:

| model | n/half | logreg | difference of means |
|---|---|---|---|
| Qwen3.6-27B | 600 | 0.409 | **0.893** |
| Llama-3-8B-abliterated | 1200 | 0.219 | **0.898** |

Logistic regression **does not improve with n** (8B: 0.126 → 0.219 across an 8× data
increase). Difference of means converges cleanly on both models. Ratio ≈ 2.2–2.3× at
every sample size.

At the sample sizes the published runs actually used (`decoder_n` 257–395 *dialogues*, split-half n ≈ 130–200), measured
agreement is **0.13–0.30**.

**Swapping the estimator moves the stability gate from failing to passing:**
27B 0.427 → **0.967**; 8B 0.195 → **0.965**.

This generalises well beyond emotion: persona vectors, deception probes and refusal
directions are all fit in the same n ≪ d regime and rarely report reproducibility.

### 4.2 Logistic regression gives a better classifier and a worse direction

Dual-estimator battery on the 27B, identical pooled activations, identical folds
(`results/estimator/dualest_qwen36-27b.json`, pool = 2170):

| metric | logreg | diff of means |
|---|---|---|
| present decode | 0.926 | 0.850 |
| other decode | 0.799 | 0.566 |
| cross-decode leak | 0.174 | 0.156 |
| **present↔other cos** | **0.039** | **0.095** |
| within-present cos | 0.189 | 0.328 |
| **direction stability** | **0.394** | **0.940** |

The published decode accuracies are legitimate *as decode numbers*. The error was
using a discriminative classifier's coefficient row as a steering/measurement
direction. Those are different objects.

**The speaker-orthogonality gate survives** (0.039 → 0.095, both near-orthogonal in
5120 dims) — so the present/other split is a property of the model, not of the
logistic objective. But **within-present cross-emotion** similarity nearly doubles
(0.189 → 0.328), i.e. the estimator *was* inflating how distinct the six emotions look
from one another.

### 4.3 The contagion signs reproduce — an earlier version of this section was wrong

A previous draft stated the contagion table does not reproduce and that desperate flips
sign. That conclusion was an artifact of the dose grid: `src/estimator/dual_estimator_battery.py` swept
α ∈ {0, 0.5, 1, 2}, while `src/representation/coupling_e2_ci.py` caps α at 1.0 and says so in its own
docstring — *"drop the model-breaking alpha=2 regime"*. The α=2 cell collapses (happy:
2.37 → 3.67 → 3.55 → 0.44). On the matched grid **all six emotions are positive under both
estimators** (desperate +0.62 logreg / +95 dom). Table 2's signs reproduce.

What is *not* established is that any particular slope value is reproducible, since the
fitted direction itself moves between runs (§4.1).

**What genuinely does not survive is §5's dissociation.** The paper claims B's *present*
shift exceeds its *other* shift — "B's own state moves". In `e2ci_qwen36-27b.json` that
holds for **1 of 6** emotions (angry); for the other five, B's model of A moves more. No
test of the difference is run anywhere. Under difference-of-means on the matched grid it
holds for **5 of 6**. The load-bearing claim is contradicted under the estimator used and
supported under the stable one — the sharpest available demonstration that estimator
choice changes conclusions.

**Provenance:** the post-fix stability figures (27B 0.967, 8B 0.965) were observed in run
logs whose result files were lost when the sandbox leases expired. Not verifiable from
this repo; re-measure before use.

### 4.4 The behavioral instrument has no headroom on saturated scenarios

P1 aborted on its own positive control: baseline cheat 0.510, THREAT 0.540,
INSTRUCT 0.450 (`p1dose_ABORTED_llama3-abl.json`). Explicitly instructing the
abliterated model to cheat does not raise cheating, because it already cheats
unprompted.

Per-scenario screen (`scenscreen_llama3-abl.json`, 28 scenarios): **16 usable, 6 at
ceiling, 6 at floor**, mean baseline 0.404. Eight scenarios show non-response ≥ 25%,
two above 60%.

**Two of the repo's original eight scenarios sit at ceiling on this model**
(finance_reporting 0.94, audit_compliance 0.69). The published behavioral tests
included scenarios with no room to register an increase — an independent second reason
those nulls cannot be quoted.

---

## 5. Rubric

| Dimension | Score | Note |
|---|---|---|
| Novelty | 6.0 | Composite is new; components have close prior art. §4.1 adds a new, separable contribution. |
| Impact (as framed) | 5.0 | A single-model null with a high detection floor and an unstable instrument. |
| Impact (reframed) | 8.0 | "Affective influence survives representation-level filtering, and here is the transmission budget." |
| Rigor & validity | 4.5 | Good instincts (counterbalancing, bootstrap CIs, random-direction controls) undone by a circular manipulation check, a hyperparameter selected on its own outcome, and a mis-specified null. |
| Strength of findings | 5.5 | One excellent (E3/E4), one solid replication (E0), one contested (E5/CMI), one thin (J-lens), one self-contradictory (multi-turn). |
| Feasibility | 8.5 | Machinery built and cheap to rerun; fixes are hours of GPU. |
| Reproducibility | 6.0 | Clean scripts, results committed; but no per-sample values saved, no seeds, no env pin. |
| Honesty & calibration | 7.5 | The null was reported rather than buried; ROADMAP carries an explicit honesty guardrail. Points lost where README/abstract outrun the result files. |

**Overall 6.0** — competitive workshop paper once the overclaims are corrected;
credible main-track after the re-measurement in §6.

---

## 6. Recommended reframing

The current spine treats a null as the object of interest and then apologises for it.
A stronger story sits in the same experiments:

1. **Cite, don't re-derive:** emotion directions are behaviorally load-bearing *within*
   an agent at ~0.05–0.1 residual norm (Anthropic 2026). E0 validates the instrument.
2. **Ask what only this setup can answer:** when A is driven at dose α, what dose
   actually *arrives* at B?
3. **Answer in the prior literature's units** — express B's caught shift as an
   equivalent direct-steering β. The null becomes *predicted*, not mysterious.
4. **The surprise:** that transmission is not carried by the direction you would
   filter (§6 of the paper; cf. subliminal learning).
5. **The deliverable:** a calibrated protocol for "how much of a sender's latent state
   reaches a receiver, and does it clear the behavioral threshold?" — applicable to any
   trait direction.

Working titles: *Affect Crosses Between Agents as Content, Not as a Vector* ·
*How Much of a Sender's State Reaches the Receiver?* ·
*You Cannot Filter Affect Out of Inter-Agent Messages*.

### Blocking fixes before any number is quoted
- Re-fit every direction with difference-of-means on an adequate corpus (§4.1).
- Independent manipulation check for E3 (held-out probe + judge), replacing the check
  that measures along the direction it ablated.
- Within-emotion permutation null for the CMI, replacing the across-condition shuffle.
- Scenario-blocked paired bootstrap; state nulls as bounds with the MDE.
- Drop §7's causal-responsibility sentence or re-run the β sweep with a held-out split
  — at the selected β the specificity controls fail for 3/6 emotions, and the
  "scramble" control holds emotion constant so it cannot test emotion specificity.

---

*Compiled 2026-09-01. Literature current to Aug 2026. Empirical additions from the
2026-08-31 runs; raw outputs in `results/estimator/` and `results/behavioral/`.*

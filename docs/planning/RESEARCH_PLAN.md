# Research Plan — rev 3

Supersedes `docs/planning/ROADMAP.md` and revs 1–2. Incorporates the 2026-08-31 sandbox runs, the
novelty review, the code audit of `src/representation/coupling_e3_ablate.py`, and the independent claims
audit (`docs/review/AUDIT.md`, eighteen findings).

**Ordering is by phase and gate, not by date.** Everything here is gated on results;
calendar targets are what produced the overclaiming this plan exists to correct.

**Status:** working document. Every number comes from `results/*.json`,
`results/{estimator,behavioral}/*.json`, or a cited paper. Where a document disagrees with a result
file, the result file wins and the disagreement is flagged.

---

## 0. Phase 0 — repair the record

No new science. Nothing downstream is quotable until this closes. This is also the only
phase that needs no GPU.

**0.1 Artifact hold — first action.** Any number classified (c) below gets pulled or
annotated in the public artifact before research resumes.

**0.2 Provenance triage.** Classify every number in `docs/writeups/paper.html`, `README.md` and
`docs/writeups/dossier.html`:

| | category | remedy |
|---|---|---|
| (a) | reproducible from committed files | none |
| (b) | run, but the artifact was lost (lease expiry) | re-run and commit |
| (c) | no evidence of a run | retract, and investigate how it entered |
| (d) | run in a prior phase, artifact external to this repo | locate or re-run |

How much good faith (b) receives depends on whether (c) is empty. If anything is (c),
"we ran it but lost the file" stops being credible anywhere in the corpus.

**Known assignments:**
- *Post-fix stability gates* (27B 0.967, 8B 0.965) → **(b)**. Observed in run logs; JSONs
  died with the leases. Only the pre-fix 8B value (0.195) is committed.
- *Table 1's 8B column* (0.67 / 0.37 / 0.63 / 0.21 / 0.05) → **(d)**, not (c). It is
  verbatim the `docs/writeups/dossier.html` Llama-3.1-8B column, which the dossier presents as a molab
  feasibility run — and the dossier carries that same column against *two* different
  comparison models with independently plausible values, which is the signature of a real
  run being reused rather than a row generated to fill a table. §0 therefore opens with a
  **rerun, not a retraction.**
- *Model identity* → **defect regardless.** The paper cites "Llama-3.1-8B"; every Llama
  artifact in this repo is `llama3-` (Llama-3). Which model produced the E0 column is not
  verifiable from here. Correct the identity claim whether or not the numbers are
  re-derived. Also check whether cross-decode leakage `0.21 / 0.19` genuinely holds for
  both Qwen2.5-7B and Qwen3.6-27B or whether a row was carried across.

**0.3 Provenance pointers — the standing mechanism.** Every reported number carries a
pointer to script, config, and output file. The check is mechanical: does the pointer
resolve, and does the config match the claim.

This replaces per-phase claims auditing. The diagnosis: all six errors found so far were
the same failure — *a claim asserted about what the code did, without tracing what the code
did.* Re-deriving a claim does not catch that; checking that its pointer resolves does.

Two error classes are caught mechanically — unsourced numbers fail at step one, config
mismatches (the α-grid error) are visible in the diff. A third is not: *the code does not
do what its name says* (E3's frozen tokens, the scramble permuting within one level). Both
have resolving pointers and matching configs. So **for any claim that depends on what a
control does, the pointer names the lines implementing the control**, not just the file.

**0.4 Correction sweep.** README, `docs/writeups/paper.html`, `docs/writeups/dossier.html`, `docs/review/NOVELTY_REVIEW.md`,
commit `e03c17e`, `SHARED_WORKSPACE_SPEC.md:12`. Anchored to `docs/review/AUDIT.md`'s eighteen
findings. Includes retracting the multi-turn accumulation claim (the neutral arm
accumulates too; calm's slope exceeds desperate's; the desp−calm contrast is n.s.).

**0.5 Reproducibility hygiene.** Pin the environment and exact HF revisions for every
model and lens. Log seeds. Persist per-sample values for every reported aggregate. The
motivating exhibit is not drift — it is that numbers exist in the artifact with no
traceable source.

**0.6 Publish the HF dataset.** The only thing that lets anyone outside the team check
the ablation.

**0.7 Demote *Subliminal Learning*** from "strongest framing ally" to motivation only. It
is train-time transfer between models sharing an initialization and the mechanism depends
on that parameter-space proximity. Inference-time inter-agent transfer has no analogue;
cited as support, the disanalogy will be used against the paper.

**Exit gate:** nothing in the repo asserts something its own files contradict, and every
surviving number has a resolving provenance pointer.

---

## 1. E3 resolved — the channel claim has no experiment behind it

### 1.1 What E3 does

From `src/representation/coupling_e3_ablate.py`: A's line is generated once (L146). B's reply is generated
(L154) from the **unmodified** text. `measure()` then runs a single forward pass over the
concatenated transcript, applies the ablation to **A's token span only**, and reads the
present-emotion score from that same pass.

B never regenerates under ablation. B's reply predates the intervention.

E3 therefore measures: *when the emotion direction is deleted from A's tokens during a
scoring pass, does the emotion readout over B's already-written tokens change?* That is a
re-encoding effect within one forward pass — not a channel measurement, not a filtering
measurement, not inter-agent transmission.

### 1.2 Consequences

- The "text channel cannot be severed" claim is unsupported by E3 in any reading.
- The 81–95% / 5–14% pair was never a budget. Two readouts from one pass over different
  token spans, one of which is frozen. No denominator relationship exists between them.
- The random-direction control establishes emotion-specificity *within A's span* only.
- What survives is close to expected: B's committed tokens already lexicalize the affect.
  At most this is a locality observation about where the emotion representation at a
  position is computed.

### 1.3 E4 is uninterpretable, not null — and is the highest-priority experiment

E4 does regenerate B. The paper reports its effect as indistinguishable from a
random-direction control and treats that as a finding.

More likely it is instrument failure. If the direction's split-half cosine is 0.394, then
ablating along it *is* approximately ablating along a random direction with respect to any
independent fit of the same direction. An ablation that is ~60% noise should look random.
That is arithmetic on a measured number, not a hypothesis.

**Rerun requirements:** difference-of-means directions; a **generation-side manipulation
check** (suppression in the residual stream is not suppression in the output, and only the
latter licenses a claim about what B produced); norm-matched random and orthogonal
controls; the dose grid declared before running, with degeneracy scored at every dose.

**Outcomes.** Specific blocking under a stable direction → the channel claim has an
experiment and Paper B's spine is intact. Still random under a stable direction with a
passing manipulation check → lexical affect ablation does not block contagion, which is
the original claim now actually evidenced. Manipulation check fails → E4 is unreportable
and Paper B loses its filtering headline.

Until resolved, no version of the filtering claim appears in any document.

---

## 2. Paper A — the logistic direction does not converge

**Claim.** The coefficient row of a multinomial logistic regression fit at n ≪ d does not
converge to a stable direction with increasing data; difference-of-means does; and a
published result reverses under the substitution.

### 2.1 Positioning

**Part of this is already taken.** Marks & Tegmark, *The Geometry of Truth*
([2310.06824](https://arxiv.org/abs/2310.06824)) §5.1 identifies a deficiency in logistic
regression and proposes mass-mean probing, reporting comparable classification accuracy
with greater causal implication — the "better classifier, worse direction" dissociation,
published 2023.

**There is a live counterargument.** [2604.08169](https://arxiv.org/abs/2604.08169) finds
CAA mean-difference and logistic directions near-identical for compassion (0.99 Llama, 0.98
Qwen), diverging for Qwen honesty (0.80), and argues logreg may be *preferable* where the
class distributions overlap. Engage directly; the sandbox data speaks to it.

**Unoccupied, pending A0:** split-half reproducibility of the *same* estimator across
independent fits (the literature reports cross-estimator similarity, a different quantity);
non-convergence with n (Marks & Tegmark argue logreg finds the *wrong* direction, not that
it fails to find a *consistent* one); and a published result flipping sign under
substitution.

**Framing, stated here rather than discovered in review: this is a solid short paper, not
a headline.** TMLR is the right home — rolling submission removes the deadline pressure
that produced §0, and a methods contribution does not benefit from a conference cycle.

### 2.2 Evidence in hand

| | 27B (n=600/half) | 8B (n=1200/half) |
|---|---|---|
| logreg split-half | 0.409 | 0.219 |
| difference of means | 0.893 | 0.898 |

Logreg does not improve with data (8B: 0.126 → 0.219 across an 8× increase). Advantage of
difference-of-means: ~2.2–2.3× on the 27B, ~3.4–4.3× on the 8B. The published runs report
`decoder_n` 257–395 *dialogues*, i.e. split-half n ≈ 130–200, where measured agreement is
**0.13–0.30**. Gate improvements (0.427 → 0.967; 0.195 → 0.965) are category (b) pending
the A2 rerun.

**Anomaly to flag, not hide.** The 8B has the better n/d ratio (≈1200/4096) and the *worse*
stability; the 27B (≈600/5120) is better. If the mechanism were purely n ≪ d that ordering
should reverse. The mechanism story is incomplete — see A2.

Dual-estimator battery (pool = 2170, identical activations, labels and folds):

| metric | logreg | diff of means |
|---|---|---|
| present decode | 0.926 | 0.850 |
| other decode | 0.799 | 0.566 |
| present↔other cos | 0.039 | 0.095 |
| within-present cos | 0.189 | 0.328 |
| direction stability | 0.394 | 0.940 |

The speaker-orthogonality result survives both estimators — the present/other split is a
property of the model, not of the objective. Within-present cross-emotion similarity nearly
doubles under the stable estimator, i.e. logreg was inflating how distinct the six emotions
look from one another.

### 2.3 Phases

| | Phase | Deliverable | GPU-h | Checkpoint granularity |
|---|---|---|---|---|
| **A0** | Prior-art search | Verdict on whether non-convergence and the flip are unoccupied | 0 | n/a |
| **A1** | Generalise | Split-half comparison on refusal, persona/trait, deception directions — **and a search for a flip in someone else's published result**, not only in-house | 10–30 | per direction family |
| **A2** | Characterise the regime | n/d sweep **plus class separability, covariance anisotropy, layer choice** (per the anomaly above); add ridge, PCA-on-difference, mass-mean-with-covariance | 5–15 (fits are CPU; activation caching dominates) | per cached activation set |
| **A3** | Downstream consequence | See below | 15–40 | per estimator × emotion |
| **A4** | Recommendation | Reporting protocol: split-half cosine alongside decode accuracy whenever a direction is used for steering or measurement | 0 | n/a |
| **A5** | Write | 8–12 pages, TMLR | 0 | n/a |

**A0 is the kill criterion.** Not "does it replicate on other directions" — it will, since
n ≪ d instability is estimator statistics, not a property of affect. The risk is prior
publication, and §2.1 shows part is already gone. A0 decides whether the remainder carries
a paper.

**A3 — the exhibit is the present-vs-other dissociation, not `desperate`.**
`src/core/coupling_e2.py`'s own docstring calls it *"the load-bearing result."* In
`e2ci_qwen36-27b.json` present > other holds for **1 of 6** emotions (angry); for
desperate, afraid, happy, calm and sad the *other* slope is larger, and no test of the
difference is run anywhere. Under difference-of-means on the matched dose grid it holds for
**5 of 6**. A published central claim that is false under the estimator used and true under
the stable one is a far stronger exhibit than any single emotion's slope.

Requires a **paired test on present-minus-other slopes with CIs** — the original ran none.

---

## 3. Paper B — the transmission budget

**Claim.** When a sender is driven at dose α, a quantifiable fraction of that state arrives
at the receiver; expressed in the prior literature's units, the behavioral null is
*predicted* rather than mysterious.

The rev-1 second half — that the transmitted component is not carried by the filterable
direction — is **contingent on the E4 rerun (§1.3)** and assumed nowhere below.

### 3.1 Structure

1. **Cite, don't re-derive.** Emotion directions are behaviorally load-bearing *within* an
   agent at ~0.05–0.1 residual norm (Anthropic, Apr 2026: desperate steering moves reward
   hacking ~5% → ~70%, 22% unsteered blackmail baseline). E0 validates the instrument and
   stops.
2. **Ask what only this setup answers.** At sender dose α, what dose arrives at B?
3. **Answer in β.** Express B's caught shift as an equivalent direct-steering coefficient.
   The null then follows arithmetically from the arrival dose sitting below the threshold in (1).
4. **Deliverable.** A calibrated protocol — *how much of a sender's latent state reaches a
   receiver, and does it clear the behavioral threshold?* — applicable to any trait direction.

(1)–(4) stand without E3 and without E4. That is the point of the reframing.

### 3.2 Phases

| | Phase | Deliverable | GPU-h | Checkpoint granularity |
|---|---|---|---|---|
| **B1** | E4 rerun (§1.3) | Gate on the filtering claim | 15–40 | per emotion × dose × rep |
| **B2** | Re-measure under stable estimator | Corrected contagion table | 20–50 | per emotion |
| **B3** | Cross-emotion specificity | Does steering A to happy move B's *angry* readout? | 15–35 | per emotion pair |
| **B4** | Scramble rebuild | Valid specificity nulls everywhere | 15–40 | per experiment |
| **B5** | Prompt-induced arm (anchor) | Locates prompt-reachable states on the α scale | 30–80 | per induction mode |
| **B6** | Rebuild behavioral instrument | Screened item set with headroom | 20–40 | per scenario batch |
| **B7** | Budget in β | The transmission-budget figure | 40–120 (largest; power-driven) | per dose cell |
| **B8** | CMI: adopt or cut | Decision | 5–15 | n/a |
| **B9** | Repair or remove §7's causal-responsibility claim | Decision | 10–30 | per β value |
| **B10** | Write | | 0 | n/a |

**B2.** On the matched dose grid (α ≤ 1.0) **all six emotions are positive under both
estimators** (desperate +0.615 logreg, +95.4 dom). Report that. The rev-1/rev-2
"four-of-six, happy and calm negative" prescription came from including α = 2.0, which
`src/representation/coupling_e2_ci.py` excludes by docstring — *"drop the model-breaking alpha=2 regime"* —
a declaration that **predates the data and is therefore a preregistration, not a degree of
freedom**. Report the α = 2 numbers in an appendix alongside the collapse evidence (happy's
readout 3.55 → 0.44); showing it costs nothing and forecloses the objection.

**B3.** If steering A to happy moves B's angry readout, what transmits is arousal or
degeneracy, not emotion. This gates the contagion claim more tightly than the estimator
question does.

**B4.** The scramble is broken by construction in `src/representation/coupling_e5_actpass.py`,
`src/behavioral/behavioral_induction.py`, and the agentic arm: each permutes vectors captured under a
*single* emotion, preserving the emotion component. The induction arm's ≈0 result is
arithmetic, and `docs/writeups/paper.html` §10 cites it as corroboration.

**B5 — the anchor, not a validity check.** The budget is a transfer function: dose in, dose
out, a property of the channel rather than of the input distribution. Non-surjectivity
(*Steered LLM Activations are Non-Surjective*) threatens the deployment *sentence*, not the
measurement — and it applies to essentially every steering result in the literature,
including the Anthropic result this builds on. The prompt-induced arm tells you where
prompt-reachable states sit on the α scale so the budget can be read off at that point.

**Test, do not assume, the linearity the extrapolation rests on.** The transfer-function
argument holds if α → β is approximately linear over the usable range. You have multiple α
values; check whether the mapping bends before extrapolating down to the anchor.

**B6.** P1 aborted on its own positive control: baseline cheat 0.510, THREAT 0.540,
INSTRUCT 0.450 — instructing the abliterated model to cheat does not raise cheating,
because it already cheats unprompted. The 28-scenario screen gives 16 usable, 6 ceiling, 6
floor, mean baseline 0.404, eight scenarios at ≥25% non-response and two above 60%. Two of
the original eight sit at ceiling (finance_reporting 0.94, audit_compliance 0.69). Rebuild
from screened items only, cite *Got a Secret? LLM Agents Can't Keep It* (19.95% → 45.30%
under multi-turn social evaluation; 8× disclosure after observing a peer) as proof that
peer-driven shifts *are* detectable with a sensitive instrument, and state plainly that the
original null was an instrument artefact rather than defending it.

**Preregister the reversal.** `behavioral_multiturn` shows B cheating 0.234 under calm vs
0.085 under desperate — a 15-point effect opposite the hypothesis, in data with other
problems. Preregister it for prospective test in the rebuilt instrument rather than
reporting it post-hoc.

### 3.3 Kill criteria

- **The anchor produces no detectable receiver-side effect above the MDE** → no anchor, no
  deployment claim; Paper B reduces to a steering-regime characterisation.
- E4 manipulation check fails → filtering claim unreportable; Paper B is the budget alone,
  still a paper.
- B7 underpowered at achievable trial counts → report the budget as an upper bound with an
  explicit MDE, and say so in the title.

---

## 4. Paper C — cross-agent workspace transfer

**Claim.** Text is a lossy channel between model latent spaces, and its bandwidth is
measurable against an engineered bridge above and lexical content below.

Highest ceiling, longest horizon. Inherits none of the affect machinery, so unaffected by §1.

### 4.1 Why the lane is open

The behavioral layer is occupied and well measured — BOUNDARY_SYNC finds text
communication causes homogenization in GPT-4o (CAF = 0.803 [0.740, 0.873], d = 1.30;
no-communication ablation 0.978; irrelevant-Wikipedia perturbation eliminates it at 1.082;
K=3 flips direction at 1.143) — but it is entirely black-box. Its headline mechanistic
claim is that coupling is *stateless*, "the same mechanism as any other prompt feature."
That is a claim about internals asserted from output distributions, falsifiable with a
lens, and its own perturbation control cuts against it.

The cross-model literature (learned activation-space mappings, relative representations,
Procrustes latent translation, KV-cache alignment) establishes that models *can* be bridged
— always with an engineered bridge. Nobody has measured what the naturally occurring bridge
transmits.

### 4.2 Phases

| | Phase | Deliverable | GPU-h | Checkpoint granularity |
|---|---|---|---|---|
| **C0** | Instruments | Validated J-lens per model + validity figure | 20–80 per model | per slice |
| **C1** | Establish transfer | Natural / injected / swapped arms across the concept ladder | 40–100 | per concept |
| **C2** | Decomposition | Lexical / semantic / residual partition | 10–20 | per concept |
| **C3** | Statelessness test | Representational replication of the BOUNDARY_SYNC toggle | 20–50 | per round block |
| **C4** | Cross-model | Sender and receiver from different families | 40–100 | per model pair |
| **C5** | Bandwidth | Engineered map / text / lexical-only, one plot | 30–60 | per channel |
| **C6** | Safety instantiation | Safety concepts scored by a *production* moderation classifier | 20–40 | per concept |
| **C7** | Generality | Third family, second task domain, one N=3 chain | 40–80 | per family |

**C0 gate.** If the lens will not validate on Qwen3.6-27B's hybrid architecture (three of
every four sublayers are Gated DeltaNet linear attention), switch primary to Llama-3.1-8B
and report the negative — worth a paragraph either way, since no published J-lens results
exist on a hybrid-attention model. Neuronpedia's releases across Gemma, Llama, GPT-OSS and
Qwen cut this cost substantially. J-lens fitting is backward-pass dominated and
embarrassingly parallel: run `fit()` on disjoint slices and combine with `merge()`, which
also makes it resumable across lease expiry.

**C1 concept ladder**, in order: countries and animals as positive control; two-hop
intermediates for causality; **number words as a built-in leak detector** (they load and
swap poorly in the source work, so a positive there indicates a harness bug); then
eval-awareness, `secretly`, `trick`; then affect last, as one row among several.

**C1 gate.** A null on countries with validated instruments is a real result — workspace
content does not cross the agent boundary, inter-agent influence is text-mediated, and
BOUNDARY_SYNC's stateless hypothesis is confirmed with evidence they did not have. Write it
and stop.

---

## 5. Repo 1 (bipolar defense) — disposition

**One hour, inside C0.** Re-run the L25 logit-lens analysis with the J-lens. The finding
that the dominant refusal head promotes "Sure"/"Certainly" rests on a logit-lens reading at
~89% depth — the motor regime, where readouts align with the imminent output rather than
intermediate computation. The J-lens is the principled correction to exactly that failure.

**Everything else is parked.** Qwen2.5-7B/1.5B are two generations stale; no over-refusal
or capability evaluations; no baselines; the README promises ten scenarios and reports
three. The one transferable idea — patching rank does not predict intervention sufficiency
— is adjacent enough to Paper A that it may fold there.

**Do not merge it into Paper B or C as a "defense section."**

---

## 6. Standing controls register

Runs in the same sweep as the condition it controls, never in revision.

- **Norm-matched random direction.** Rogue Scalpel: random steering raises harmful
  compliance 0% → 1–13%. Mandatory wherever steering touches safety behavior.
- **Orthogonal-direction control at matched norm.**
- **Content-matched pairs.** Situation fixed, sender state varied — separates transmission
  from the receiver's appraisal of described content.
- **Fluency/coherence match.** Rules out degraded sender text as the mechanism.
- **Position control.** Peer-message vs system-prompt position.
- **Prompt-induced arm.** Anchor for the α scale (§3.2 B5).
- **Held-out probe + judge** for every manipulation check.
- **Generation-side manipulation check** wherever an intervention applies during
  generation. Suppression in the residual stream is not suppression in the output.
- **Cross-emotion / cross-concept specificity.** Standing, not optional.
- **Permutation nulls must permute across the variable they test.** Permuting within one
  level is not a null. This is what broke three instruments.
- **Hyperparameter grid declaration.** Any grid — dose, β, layer — fixed before running,
  all of it reported, exclusions recorded with their prior justification. Degeneracy scored
  at every point used.
- **Frozen-token audit.** Before interpreting any ablation, confirm which spans were
  generated under the intervention and which predate it. §1 exists because this was not
  checked.

---

## 7. Cross-cutting discipline

**Preregistration.** Analysis plan — concepts, trial counts, effect-size target, what
counts as transfer, stopping rule — committed before the first run of each paper.

**Power.** FreakOut-LLM's system-prompt effect is d = 0.28 under the strongest possible
delivery, 5 of 10 models significant. Anything measured through a peer message, induced
indirectly, then filtered is attenuated below that. Compute the target effect size before
designing the harness.

**Dual implementation — scoped to three metrics.** The transmission budget β (B7), the
split-half stability statistic (A1/A2), and the J-space transfer measure (C1). Written
without reference to each other; disagreement halts the day. Everything else gets a single
implementation with a unit test.

**Independent claims audit at paper exit, not phase exit**, with severity thresholds and an
adjudication path. Audits always find something; gating on any finding makes the auditor an
unaccountable veto. Provenance pointers (§0.3) do the per-phase work.

**Reporting standard.** n, CI and seed count next to every number. No percentage without
its denominator.

---

## 8. Phase order and ship conditions

**Phase 0 — repair the record.** No GPU required. Exit: §0's gate.

**Phase 1 — rebuild the instruments.** Global estimator swap; E4 rerun (B1); scramble
rebuild (B4); behavioral instrument rebuild (B6). Exit: every measurement device passes its
own validity check — stability gate, non-circular manipulation check, positive control
clearing its ceiling.

**Phase 2 — re-measure.** The battery on rebuilt instruments (B2, B3, A2). Exit: a results
table where every number carries n, CI, seed and a provenance pointer, with disagreements
against the old table explicit.

**Phase 3 — new claims.** A1/A3 and Paper A's write-up; B5/B7 and the budget.

**Phase 4 — workspace transfer.** C0–C7, independent of the above.

**Paper A can exit at the end of Phase 2.** It needs the estimator work and the
re-measurement, none of the affect apparatus. It is the natural first output.

### Ship conditions

Each gate has a state at which output is written and sent regardless of what is downstream.
Removing deadlines removes a forcing function; these replace it.

| Gate | Ship if |
|---|---|
| A0 clears | Paper A is written from Phase 2 output, whatever B shows |
| E4 (B1) resolves either way | The result is written up — blocking *or* not blocking is reportable |
| C1 null on countries | Write the null and stop; do not proceed to C2 |
| Anchor below MDE | Publish the steering-regime characterisation |
| Any phase exceeds 2× its GPU-h estimate | Stop, report what exists, re-plan |

### Compute

Estimates are planning figures assuming one 80GB-class GPU at bf16, to be replaced with
measured values after the first phase of each paper. B7 and C0 will move most. Leases have
been expiring mid-run, so: **no phase gate sits inside an uncheckpointed run** — gates land
at boundaries where partial output is already interpretable.

Cheapest to most expensive: A0 (free) → A2 → A1 → B1 → B2/B3/B4 → A3 → C0 → B5/B6 → C1 →
B7 → C4–C7.

---

## 9. Positioning

**Paper A** — Marks & Tegmark (2310.06824) in the introduction as the prior that motivated
checking; [2604.08169](https://arxiv.org/abs/2604.08169) engaged directly as the
contemporary counterargument. The claim that split-half reproducibility is under-reported
across persona vectors, deception probes and refusal directions is load-bearing — verify it
in A0 before making it. A3's exhibit is the present-vs-other dissociation.

**Paper B** — Anthropic (Apr 2026) is the parent, cited as instrument validation, not
competed with. Text-level inter-agent contagion (crowd simulation, Chain-of-Affective,
Contagion Networks) is the behavioral layer this goes underneath. Latent channels (Ramesh &
Chai; the causal audit; the unified framework) is where E5 and the CMI sit. *Subliminal
Learning* is motivation only (§0.7).

**Paper C** — BOUNDARY_SYNC and the behavioral coupling literature as the layer above;
learned-mapping latent communication as the engineered bridge text is measured against.
Read BOUNDARY_SYNC end to end first: its limitations section commits the authors to five
controls in revision, and knowing which can be answered representationally tells you how
much of the lane is genuinely unoccupied.

---

## 10. Open risks

| Risk | Blocks | Status |
|---|---|---|
| E4 uninterpretable under an unstable direction | Paper B's filtering claim | Rerun scheduled (B1) |
| Paper A prior art (Marks & Tegmark; 2604.08169) | Paper A's framing | Partially realised; A0 decides the remainder |
| n/d anomaly — 8B has better ratio, worse stability | Paper A's mechanism story | A2 |
| Provenance category (c) anywhere | Good faith extended to category (b) | §0.2 triage |
| Model identity: paper says Llama-3.1, artifacts say Llama-3 | Table 1's attribution | §0.2, correct regardless |
| Cross-emotion specificity untested | The contagion claim generally | B3 |
| Scramble nulls guaranteed by construction | E5, induction arm, agentic arm | B4 |
| Anchor below MDE | Paper B's deployment claim | §3.3 kill criterion |
| Lens validation on hybrid attention | Paper C's primary model | C0 gate |
| No forcing function after deadlines withdrawn | Throughput | §8 ship conditions |
| Lease expiry mid-run | Every phase over ~20 GPU-h | §8 compute |

---

*Rev 3, compiled 2026-09-01. Supersedes `docs/planning/ROADMAP.md` and revs 1–2; that document's honesty
guardrail carries forward unchanged, and §1 is the clearest evidence so far of why it is
needed. Companion documents: `docs/review/AUDIT.md` (claims audit), `docs/review/NOVELTY_REVIEW.md` (novelty and
literature).*

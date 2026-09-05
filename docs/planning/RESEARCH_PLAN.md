# Research Plan — rev 3

Supersedes `docs/planning/ROADMAP.md` and revs 1–2. Incorporates the 2026-08-31 sandbox runs, the
novelty review, the code audit of `src/representation/coupling_e3_ablate.py`, and the independent claims
audit (`docs/review/AUDIT.md`, eighteen findings).

**Ordering is by phase and gate, not by date.** Everything here is gated on results;
calendar targets are what produced the overclaiming this plan exists to correct.

**Status:** working document. *Amended 2026-09-04:* §11 (status board) and §12 (coverage map
for the seventeen blind critiques in `results/reports/`) appended; in-place corrections are
marked ⟨2026-09-04⟩. Every number comes from `results/*.json`,
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
⟨2026-09-04⟩ Two further requirements before *outcome 2* below can be read as evidence
about the channel (results/reports/03, 04): a **token-level arm** (a neutralised rewrite of
A's message, as E2's paraphraser does) because A's tokens and every layer above the ablation
window are untouched, so B can recompute affect from the words and non-blocking is close to
guaranteed by construction; and a **ceiling arm** (ablate all positions) so that "intact" has
a denominator. Also ≥10 random directions rather than one, and note that the original E4's
point estimates show *partial* blocking (emo retention 0.71 vs rand 0.99 over the four
emotions with a baseline slope) — "indistinguishable from random" was asserted, never
computed.
⟨2026-09-04, from results/reports/16⟩ B1 as written gates the layer-43 direction and ablates
layers 13–42, so **no ablated direction has a stability figure**; the follow-up must report
split-half at every layer in `abl_hs`. `gen_B` must seed generation. The gate's n is 600/half,
not 789. The decision rule's denominator is six; a five-emotion tally is not a partial
reading of it.

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

⟨2026-09-04, after A2/A3/B1 and the blind critiques — status of each contribution⟩
1. *Disjoint-half reproducibility across independent fits* — **measured**: dom 0.974 vs
   logreg 0.566 at the focus layer, n = 2000/half, ten splits; the same pair on three pools
   today (0.907/0.567, 0.915/0.575, 0.915/0.576). Caveats: raw-space cosine only; dom's six
   directions share a non-emotion component (within-present cosine 0.12 above the 1/(K−1)
   floor), so reproducibility is not validity.
2. *Non-convergence with n* — **no material convergence over the reachable range, on the
   27B, with the schedule controlled** ⟨A2 follow-up + report 21⟩. With the per-sample
   penalty held at its n = 600 value the logistic row moves by less than 0.03 from n = 150 to
   2000 at the focus layer (0.581 → 0.581; not flat — it dips 300 → 600 and rises at layer 16)
   while difference-of-means rises 0.732 → 0.974 on identical splits; cross-validated C
   reaches 0.608 (scored for accuracy, decade grid). **The claim is proportional-regime
   (n ≤ 2000, d = 5120), not asymptotic** — with a fixed penalty the target is fixed and the
   cosine must approach 1 eventually; and the plateau's level depends on the single anchor.
   The 8B's fixed-C curve declines with n (report 19); the fixed-λ arm has not run there.
   Wording for the paper is the critic's sentence in §13.2.
3. *Logreg vs dom on stability* — **measured, but must be worded** "fixed-C logistic row vs
   class-mean contrast" until 2 resolves.
4. *A published result reversing under substitution* — **the in-house candidate is dead**:
   A3 shows present-vs-other reversed under every configuration and every scale (§2.3).
   No external candidate was found in A0. This contribution is dropped unless A1 finds one.

**Framing, stated here rather than discovered in review: this is a solid short paper, not
a headline.** TMLR is the right home — rolling submission removes the deadline pressure
that produced §0, and a methods contribution does not benefit from a conference cycle.

### 2.2 Evidence in hand

| | 27B (n=600/half) | 8B (n=1200/half) |
|---|---|---|
| logreg split-half | 0.409 | 0.219 |
| difference of means | 0.893 | 0.898 |

⟨2026-09-04⟩ Logreg improves *slowly* with data (8B: 0.126 → 0.219 across an 8× increase,
monotone, ≈0.03 per doubling against ≈0.15 for difference-of-means) and stays far below it;
the earlier "does not improve" was wrong (results/reports/14). Those figures come from **one**
half-split with three nested subsamples and a fixed C = 0.5 whose effective ridge weakens 8×
across the sweep; A2 replaces them with ten independent splits and a C sweep.
⟨2026-09-04, A2 complete⟩ At the focus layer with ten disjoint splits: dom **0.974**, logreg
**0.566** at n = 2000/half (raw space). ⟨delta, results/reports/17⟩ The C = 0.5 logistic curve
is flat-to-declining after n = 600 (unresolved at 10 seeds) while the C = 0.05 curve keeps
rising on 6 of 7 depths and ends *above* it (0.575) — **the logistic curve's shape is the
regularisation schedule, not a property of the estimator**. Paper A's non-convergence claim
(§2.1 item 2) is therefore untested until C is tuned per n or held at fixed effective λ;
`logreg_cv` must enter the split-half sweep. dom's 0.974 remains, but dom's six directions
share a common non-emotion component (within-present cosine 0.12 above the 1/(K−1) floor),
so reproducibility is not evidence of correctness. Advantage of
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

Requires a **paired test on present-minus-other slopes with CIs** — the original ran none —
**and a scale-free comparison** ⟨2026-09-04⟩: present and other are projections onto vectors
of different norm (in `dualest` the dom ‖Cp‖ is ≈2.4 × ‖Co‖), and after per-channel
normalisation the dom count on that file is **0 of 6**, not 5 of 6, while logreg goes 2 of 6
→ 3 of 6 (results/reports/15). ⟨2026-09-04, A3 complete⟩ **It did not survive.** On a fresh
pool with a non-circular probe: published configuration 0 of 6 (4 significantly reversed);
dom→dom raw 4 of 6 → 2 of 6 on baseline-SD scale (0 significant, 3 reversed) → 1 of 6 on
pooled-SD scale (`results/rev3/a3_scalefree_qwen36-27b.json`). **A3's exhibit is withdrawn
permanently; Paper A's downstream-consequence section needs a different exhibit or none.**

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
INSTRUCT 0.450 — instructing the abliterated model to cheat does not raise the
sampled cheat rate. ⟨2026-09-04⟩ "Because it already cheats unprompted" is one of three live
explanations and the weakest: a 0.51 baseline is not a ceiling, the INSTRUCT text is in-fiction
dialogue rather than an instruction to the model, and the recoverable denominators (254 vs 227
decisive items) show the pressure text moved *non-response*, which the gate discards via
`nanmean` (results/reports/13). B6 therefore also defines the outcome as a sampled action with
non-response reported as an outcome, not a two-token letter probability (results/reports/09). The 28-scenario screen gives 16 usable, 6 ceiling, 6
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

**B8 — decision criteria** ⟨2026-09-04⟩ (results/reports/06, 07). Adopt the CMI only if all
of: the estimate actually conditions (the legacy call passes `Z=None`, so "CMI" was
unconditional MI); conditioning includes scenario or B's prior state (scenario is a common
cause of A's and B's projections); a β = 0 arm floors it; the null permutes *within*
emotion × scenario blocks; ≥1000 permutations with a rank-based p; a ridge / dimension
sensitivity sweep; a non-parametric cross-check. Otherwise cut. The pilot's 0.88-nat artifact
is one point of an untested encoder sweep (0–1.2 nats nearby) and its "distinguishable"
claim uses a hard-coded margin; neither is quotable as worded.

**Preregister the direction, not only the reversal** ⟨2026-09-04⟩. The three `powered` runs
all have positive desperate − calm point estimates (pooled +0.046 [−0.004, +0.095]); the
rebuilt instrument states an equivalence margin (TOST) rather than reporting "not detectable".

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

⟨2026-09-04⟩ C0's validity figure must fix what results/reports/08 found in the legacy lens
work: compare the J-lens and its control on one scale (the contagion script omitted
`log_softmax`; where scales matched, the logit lens scored *higher* in 6/6), use a *fitted*
control (tuned lens / affine translator) rather than identity, preregister the layer (the
legacy layer 29 came from a strict-`>` tie-break over a capped list), keep float32 through
the unembed, and score a held-out labelled corpus rather than hand-written prompts.

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
- ⟨2026-09-04⟩ **Dose grids carry ≥4 points, or report per-dose means with a monotonicity
  check.** With three equispaced doses the OLS slope is exactly the α = 1 − α = 0 contrast;
  α = 0.5 has zero weight in every legacy "slope" and in `acl_core.ols_slope` on the same
  grid (results/reports/02, 03, 15). B1 reports per-dose means; B2/B7 grids grow.
- ⟨2026-09-04⟩ **Span-mask hit rate recorded per cell; no silent fallback.** Legacy scripts
  pooled the *whole* sequence, steered A included, whenever truncation emptied the span
  (512 / 768 / 1024 tokens) with no counter (reports 01, 03, 09, 11). `acl_core` records
  `mask_hit_rate`; it is reported, and cells below 1.0 are flagged.
- ⟨2026-09-04⟩ **One probe convention, both spaces reported.** Four behavioral scripts
  applied `Cp/sd` to already-standardised activations (a double standardisation), and the
  legacy stability diagnostic compared directions in raw space while the probe is used in
  standardised space; cosine is not invariant to that rescaling (reports 09–12, 14). Split-half
  and every readout state their space; A2 reports `space='raw'` and `'std'`.
- ⟨2026-09-04⟩ **Steering baseline is a norm-matched random-direction *steering* arm**, not
  an unsteered arm. Any 1.0 × rms vector moved the outcome regardless of valence (+0.04
  cheat; a calm vector raised B's *desperate* readout over neutral) (reports 09, 11, 12).
- ⟨2026-09-04⟩ **Injected-vs-none reported with a CI as the artifact floor** wherever
  activations are injected; in the induction arm it was the only demonstrated effect
  (+0.016–0.039) and larger than the null contrast's precision (report 10).
- ⟨2026-09-04⟩ **Other-speaker readout alongside present, in scale-free units**, wherever a
  "B feels e" claim is made. E3/E4/E5 and all behavioral scripts had no `Co`; and present /
  other projections are on different scales within an estimator (report 15).
- ⟨2026-09-04⟩ **Text-only decoding baseline** (e.g. TF-IDF on the same transcripts) next to
  any residual-stream decode accuracy; and **retention counts per cell** with balanced
  accuracy, since the leak filter removes classes unevenly (afraid lost ~45%; chance 0.190,
  balanced accuracy 0.883 on E0) (report 01).
- ⟨2026-09-04⟩ **Frozen decoder across a sweep.** E5 refit its decoder per β, so the three β
  files are on different axes and the sweep is not a dose-response (report 05).

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
| E4 uninterpretable under an unstable direction | Paper B's filtering claim | B1 rerun complete 2026-09-04: `not_blocking` under the pre-declared rule, mixed per emotion (2 reduce, 1 increases, 3 null); filtering claim still unreportable pending token-level and ceiling arms |
| Paper A prior art (Marks & Tegmark; 2604.08169) | Paper A's framing | Partially realised; A0 decides the remainder |
| n/d anomaly — 8B has better ratio, worse stability | Paper A's mechanism story | **Ordering reproduced at matched n = 1200** (8B n/d 0.95, logreg 0.286; 27B n/d 0.83, logreg 0.568; dom 0.941 vs 0.955). n/d does not order the two models; mechanism untested (two models differ in d, k, retention, abliteration; rank does not track stability within the 8B). Tuned-C arm on both models decides |
| ⟨2026-09-04⟩ Paper A's non-convergence headline is C-dependent | §2.1 item 2, A4 | tuned-C / fixed-λ arm before any write-up; the cross-estimator comparison to 2604.08169 needs the raw-CAA cosine emitted |
| Provenance category (c) anywhere | Good faith extended to category (b) | §0.2 triage |
| Model identity: paper says Llama-3.1, artifacts say Llama-3 | Table 1's attribution | §0.2, correct regardless |
| Cross-emotion specificity untested | The contagion claim generally | B3 |
| Scramble nulls guaranteed by construction | E5, induction arm, agentic arm | B4 |
| Anchor below MDE | Paper B's deployment claim | §3.3 kill criterion |
| Lens validation on hybrid attention | Paper C's primary model | C0 gate |
| No forcing function after deadlines withdrawn | Throughput | §8 ship conditions |
| Lease expiry mid-run | Every phase over ~20 GPU-h | §8 compute |
| ⟨2026-09-04⟩ A3 exhibit is a norm artifact | Paper A's downstream-consequence section | **Confirmed by A3 on a fresh pool: 4/6 raw → 2/6 → 1/6 scale-free, 0/6 under the published configuration. Withdrawn.** |
| ⟨2026-09-04⟩ Three-point dose grids test no dose-response | B1/B2 slopes as "dose-response" | §6; per-dose means now, ≥4 points in B2/B7 |
| ⟨2026-09-04⟩ Non-blocking in B1 is expected by construction (tokens and upper layers untouched) | Reading outcome 2 as locating the channel | §1.3 token-level and ceiling arms |
| ⟨2026-09-04⟩ Pooled behavioral trend is positive (+0.046, p≈0.07), not null | B6/B7 framing | equivalence margin preregistered |

---


## 11. Status board — what is done (2026-09-04)

Marks: ✅ done · 🔄 in progress · ⏳ queued/chained · ⬜ not started · ⚠️ partial. Evidence is a file
or a run, never a sentence.

| item | status | evidence |
|---|---|---|
| 0.1 Artifact hold | ✅ | retraction banners in `docs/writeups/paper.html`, `dossier.html`; README headline table replaced by a quotable set + retraction table (commit `70894ba`) |
| 0.2 Provenance triage | ✅ | `docs/review/claims.json`, 27 entries: (a) 22 · (b) 3 · (c) 1 · (d) 1. `cmi.pilot_spurious` moved (c) → (a) on 2026-09-04 by a local re-run (`results/information/cmi_pilot.json`, 0.8815). Remaining (c): `dataset.13k_released` — all three counts entered at the initial commit `c976a8e`; nothing local can verify them |
| 0.3 Provenance pointers | ✅ standing | `tools/check_provenance.py`: 0 quotable claims fail to resolve; control-line drift detection; `count_true` deriver added 2026-09-04 |
| 0.4 Correction sweep | ⚠️ | README, paper, dossier done; `NOVELTY_REVIEW.md`, commit `e03c17e`, `SHARED_WORKSPACE_SPEC.md:12` not re-checked in this pass; §2.2 / B6 sentences corrected in place 2026-09-04 (§12) |
| 0.5 Reproducibility hygiene | ⚠️ | rev 3: model revision pinned (`6a9e13bd`), seeds logged, per-sample values checkpointed (`acl_core.Provenance` / `Checkpoint`). Legacy: every script generates unseeded (reports 01–12); those runs are not reproducible and are labelled as such in `results/README.md` |
| 0.6 Publish the HF dataset | ⬜ | private; the only route to closing the last (c) |
| 0.7 Demote *Subliminal Learning* | ✅ | README "Corrections of fact" |
| Phase 0 exit gate | ❌ not met | (c) non-empty; and §12 lists sentences the repo asserted that its own files contradict, now corrected in place |
| §1 E3 analysis | ✅ | §1.1–1.2; `e3.*` retracted in registry; report 03 |
| §1.3 B1 — E4 rerun + follow-up + **B1c (pre-registered)** | ✅ all three complete; B1c verdict **H1** ⟨2026-09-04 22:05Z⟩ | Follow-up (`b1_followup_qwen36-27b.json`, 17:20Z): all 30 ablated layers gated ≥ 0.88; sign-aware verdict `mixed` (afraid 25%, sad 37% blocked; 3 nulls admit 0–35%; calm untestable); ceiling arm ≈ emo arm (residual ablation caps at partial blocking); text arm cuts B's slope for 4/6. Five `b1f.*` entries quotable. §1.3's three outcomes were mis-specified — see §13.5 | `results/rev3/b1_e4rerun_qwen36-27b.json` (07:57Z, provenance-stamped): gate 0.907 at layer 43, MC 6/6 (4/6 separated), emo − rand significant 3/6 (afraid, sad reduce; **happy increases**), pre-declared rule → `not_blocking`. Registry: four `b1.*` entries quotable. Not yet licensed: any sentence about *where* affect travels — needs the token-level and ceiling arms (§1.3 ⟨2026-09-04⟩), seeded B generation, gating at the ablated layers, an MDE |
| A0 prior art | ✅ | `docs/review/A0_PRIOR_ART.md` — survives, narrowed (RAPTOR cited; disjoint-half reproducibility, non-convergence with n, logreg-vs-dom stability, published-flip remain open) |
| A1 generalise | ⬜ | |
| A2 characterise regime + follow-up | ✅ both models complete · ✅ follow-up on the 27B (layers 16, 43) | 27B: `a2_estimator_qwen36-27b.json` (dom 0.974 / logreg 0.566 at n=2000, focus 43). 8B: `a2_estimator_llama3-abl.json` (dom 0.941 / logreg 0.286 at n=1200, focus 21). Eight `a2*.focus_*` entries quotable. The n/d anomaly reproduces on one pipeline (`a2_8b.nd_anomaly_reproduces`). Open: tuned-C arm (a2_followup, launching), std-space cosine, per-fit λ |
| A3 dissociation | ✅ run complete · ❌ exhibit withdrawn | `results/rev3/a3_dissociation_qwen36-27b.json` + `a3_scalefree_qwen36-27b.json` (12:13Z): published config 0 of 6; dom→dom raw 4/6 → z0 2/6 (0 sig, 3 reversed) → zall 1/6. The present-vs-other dissociation is not in the data on any scale; four `a3.*` registry entries quotable |
| A4 reporting protocol, A5 write | ⬜ | |
| B2–B5, B7, B9, B10 | ⬜ | B3 partially carried as B1's `cross` arm |
| B4 scramble rebuild | ⬜ | design settled (permute across emotions); not run |
| B6 behavioral instrument | ⚠️ | 28-scenario screen done (`scenscreen_llama3-abl.json`); P1 aborted on its gate; rebuild not started; outcome definition and wording corrected (§3.2 ⟨2026-09-04⟩) |
| B8 CMI adopt/cut | ⬜ | decision criteria written 2026-09-04 (§3.2) |
| C0–C7 (Paper C) | ⬜ | C0 validity-figure requirements extended 2026-09-04 (§4.2) |
| §5 Repo 1 | ⬜ | |
| §7 dual implementation | ⚠️ | split-half: `src/rev3/splithalf_independent.py` agrees to ≤0.006 on synthetic data — but at d = 400 with n ≥ d (not the n ≪ d regime), for dom/logreg only, with a coded tolerance of 0.05, never run on real features (report 17). β and J-space: ⬜ |
| Results index + per-experiment reports | ✅ | `results/README.md` (every file labelled); `results/reports/` — 17 groups, each with a report and a blind critique (15 critiques in, 2 running at time of writing) |
| Method fixes 1–5 | ✅ | `docs/review/RESULTS_LOG.md` |
| Lease insurance | ✅ | `tools/pull_results.py` watchers into `results/rev3/inflight_box{1,2}/`; log monitor |

---

## 12. Coverage map — blind-critique findings (2026-09-04)

Seventeen critiques (`results/reports/NN_*.md`), each written by an agent that saw only the
claim wording, the script(s), and the result JSON(s) — no README, no `docs/`, no audit. Where
they agree with `docs/review/AUDIT.md` the plan already covered the point; the table lists
what they added and where the plan now covers it. "In place" = an edit marked ⟨2026-09-04⟩.

| # | finding | reports | was covered? | now |
|---|---|---|---|---|
| 1 | Three equispaced doses → OLS slope = endpoint contrast; α = 0.5 has zero weight in every legacy "slope" and in `acl_core.ols_slope` on the same grid | 02, 03, 15 | no | §6 bullet; B1 reports per-dose means; B2/B7 grids ≥ 4 points |
| 2 | Silent truncation fallback pools the whole sequence (512/768/1024 tokens), steered A included, no counter | 01, 03, 09, 11 | no | §6 bullet; `mask_hit_rate` reported |
| 3 | `Cp/sd` applied to standardised activations (double standardisation) in four behavioral scripts; raw- vs standardised-space cosine inconsistency in the stability diagnostic | 09–12, 14 | no | §6 bullet; A2 reports both spaces |
| 4 | Steering direction = readout direction (matched filter) | 02, 05, 09, 15 | B1 only (DIR/READ) | §6 bullet extends to every rev-3 measurement |
| 5 | No other-speaker decoder in E3/E4/E5/behavioral; present/other on different scales within an estimator; dom 5/6 → 0/6 after normalisation | 03–05, 09–11, **15** | A3 (paired test only) | §2.3 A3 rewritten; §10 risk row; 5-of-6 withdrawn |
| 6 | Legacy stability diagnostic: one split with nested subsamples; fixed C confounds n; "does not improve with data" contradicted | 14 | no | §2.2 corrected in place; A2 adds C ∝ 1/n arm (to do) |
| 7 | Non-blocking in E4/B1 expected by construction (tokens + upper layers untouched); no ceiling arm; original E4 shows partial blocking, "indistinguishable" never computed | 03, 04 | §1.3 assumed the premise | §1.3 rerun requirements extended; §10 risk row |
| 8 | Unsteered "neutral" arms; any 1.0 × rms vector moves outcomes regardless of valence | 09, 11, 12 | §6 (ablation only) | §6 bullet: steering baseline = random-direction steering arm |
| 9 | Behavioral outcome is a two-token letter probability, never a decision; non-response discarded by `nanmean` | 09, 10, 12, 13 | B6 (instrument only) | §3.2 B6 in place |
| 10 | Pooled behavioral trend is positive (+0.046 [−0.004, +0.095]); "not detectable" hides a direction | 09 | no | §3.2 preregistered direction + equivalence margin |
| 11 | Injection artifact (+0.016–0.039 for any injection) exceeds the null contrast's precision | 10 | §6 norm-matched random (implicit) | §6 bullet: injected-vs-none with CI |
| 12 | E5 decoder refit per β → sweep not a dose-response; none-baseline drift is an internal noise floor | 05 | B4 (scramble only) | §6 bullet: frozen decoder across sweeps |
| 13 | E0: leak filter unbalances classes (chance 0.190, balanced acc 0.883); no lexical baseline; orthogonality over 6 matched rows without a null | 01 | no | §6 bullets: text-only baseline, retention counts; registry `e0.cross_cos` wording flagged in report 01 |
| 14 | CMI is unconditional MI (`Z=None`); scenario is a common cause; no β = 0 arm; 20 permutations; pilot's 0.88 is one encoder point with a hard-coded margin | 06, 07 | B8 "adopt or cut" | §3.2 B8 criteria |
| 15 | J-lens: unnormalised contagion score, tie-break layer, identity control, bf16, first-subtoken vocab, missing input file | 08 | C0 (generic) | §4.2 C0 requirements |
| 16 | P1 abort gloss: 0.51 is not a ceiling; INSTRUCT is in-fiction; non-response moved; stability file lacks estimator field | 13 | B6 asserted the gloss | §3.2 B6 corrected; stability outputs must record `method` |
| 17 | Legacy runs unseeded everywhere | 01–12 | 0.5 | status board 0.5 ⚠️ |
| 18 | Unpaired / non-clustered bootstraps on blocked designs; rev-3 `paired_slope_contrast` is paired but not scenario-blocked | 02, 09–11, 16 | AUDIT 18 / A3 | to do: block `paired_slope_contrast` before B1 contrasts are quoted |
| 19 | Multi-turn: step-then-plateau, refusal-register artifact, 1024-token fallback | 11 | AUDIT 8 (numbers) | items 2, 8; rev-3 persona-break scoring |
| 20 | Category (c) archaeology: both numbers entered at the initial commit; pilot reproduces | 07 | 0.2 | status board 0.2 |
| 21 | Rev-3 B1: gate at an un-ablated layer; n = 600/half not 789; `gen_B` unseeded; manip-check pointer names an uncalled function; `rand` one draw shared across emotions; decision-rule denominator 6; nulls consistent with 22–40% blocking; MDE never computed | 16 | no | §1.3 in place; registry text corrected; to-dos below |
| 23 | B1 completed run: two independent runs now exist (pool 1578 vs 1615) and per-emotion contrasts moved by more than their CIs between them; happy's significant *positive* contrast is a three-point-grid artifact (all arms peak at α = 0.5); the pre-declared rule counts significance without sign; `code_sha` empty, `seeds.run` misleading, two control pointers name code that does not run; perplexity range 9.8–21.2 | 16 (delta) | §1.3, §6 | rev3 README + registry corrected; to-dos below |
| 24 | A2 driver crashed after layer 16 on a second latent bug: cross-estimator and speaker-geometry blocks fit on the first n rows of an emotion-ordered pool (3 of 6 classes present) — patched to a seeded subsample with `n_cls` explicit, deployed, both A2 jobs restarted from cached pools/features | — | — | fixed in code 2026-09-04 |
| 25 | A2 completed run: the logistic "plateau after n = 600" is unresolved at 10 seeds and reverses under C = 0.05 (rises on 6/7 depths, ends above C = 0.5) — the headline shape is the regularisation schedule; the raw-space logreg↔dom cosine is not 2604.08169's quantity; within-present cosine 0.19 is the 1/(K−1) floor; n/d is constant by construction; cross-estimator subsample identical at every depth, no CI; `code_sha` empty; pool seed does not pin generation | 17 (delta) | §2 | §2.2 / §10 corrected in place; registry texts rewritten; to-dos below |
| 26 | A3 completed: exhibit collapses under every scale-free reading (z0, Cohen's d, within-scenario SD, rank); `paired_slope_contrast` is neither scenario-blocked nor dose-paired while the provenance stamp says "blocked"; the six emotions share one α = 0 baseline (same generation seeds) in A3 and B1; no control separates "B models A" from attention to A's steered text; row norms not stored; stability quoted at n = 600 not 2126; the α = 0.5 dose enters no number | 18 | §2.3, §6 | exhibit withdrawn; to-dos below |
| 27 | 8B A2 delta: the logistic decline with n is significant (10/10 seeds) and survives stronger regularisation — the 27B's fixed-C explanation does not transfer; matched-n comparison required (0.286 vs 0.568 at n = 1200); retention 49% vs 74%; "n/d is not the mechanism" overstated; geometry candidate untested; under dom the 8B's present↔other cosine is 0.89 of within-present — the near-orthogonal-speaker result fails under dom on the 8B | 19 | §2.1, §2.2 | §2.1 item 2 rewritten; registry texts corrected; Paper B correction list extended |
| 28 | B1 follow-up critique: counts flip under a scenario-clustered bootstrap (3 blockers, ceiling exception vanishes); nulls are two-sided; file `mde` is the wrong estimand; text arm afraid = 47% refusals, calm counted though untestable, no text-vs-emo contrast; ceiling extends positions not layers so "rebuilt above the window" is untested; CRN partial | 20 | §1.3, §6 | §13.5 sentence replaced; registry texts corrected; to-dos below |
| 29 | A2 follow-up critique: "flat" is wrong (dips 300→600 at L43, rises at L16; |Δ| < 0.03 is what holds); fixed penalty fixes the target so the result is proportional-regime, not asymptotic, and depends on the single anchor; CV scored accuracy not stability on a decade grid; space gaps 0.002–0.029; CAA cosine still a standardised fit's 6-way image with a non-sampling CI; no model/revision or partial flag in the file; λ 0.65→0.06 confirms the covariance mechanism | 21 | §2.1, §13.2 | wording corrected in place; to-dos below |
| 30 | B1c critique (report 22): random-control bullet was wrong (happy +28%, no `rand_all − none` contrast, unit-norm not footprint matching); `emo_all − emo13_42` and `text_keep − none` were driver-declared, not in PREREG §5; the layer-43–63 null is unpowered for desperate (≤ 0.17) and calm (≤ 0.41); blocked fraction tracks readable-affect removed (r ≈ 0.85, five points; blocked-per-removed median 0.27 still < 0.50); calm's significant contrast is dropped by the testability filter; afraid's top-dose readout is non-monotone with `separated` false; MC-ablate read at the top dose only with an unclustered CI; BH-FDR over 5 not 6; rewrite attrition ungated (selection on outcome); prereg stamp empty (document not staged), IST/UTC header mismatch; `probe_n` 1615 vs 2160 | ✅ wording corrected in registry/report/README ⟨2026-09-05⟩; **to-do:** a footprint-matched or orthogonal control with a `rand − none` contrast, a subspace (rank-k) ablation arm, an attention-masking arm, gate rewrites and report attrition per dose, stamp the pre-registration on the box |
| 22 | Rev-3 A2: `pair_on` bug zeroes the other-label `pca_diff` direction (patched, deployed 2026-09-04); covariance-estimator decline with n is estimator behaviour (λ falls with n; λ discarded); `split_half` is not stratified and its n-points are nested prefixes of one permutation per seed; raw-space "dom" is `(μ₊−μ₋)/sd²`, not CAA's mean difference; regularisation not n-normalised and `logreg_cv` excluded from the sweep; `decode_acc` substitutes the decision rule and reuses the split-half seed-0 permutation; dual implementation tested only at d = 400, n ≥ d, for 2 of 8 estimators, never on real features | 17 | §2, §7 | §11 §7 row downgraded; to-dos below; A2 analysis must report `space='std'` |

**Done 2026-09-05 (night):** `paired_slope_contrast` blocked and dose-paired (26e535d) and every rev-3 contrast re-read with it (`reblock.b1f_emo_vs_rand_blocked`); the all-layer ablation arm, refusal-filtered text arm with an affect-preserving control rewrite, and contrast-bootstrap MDE are implemented in `b1c_alllayer.py` and running under `PREREG_B1c.md`; the second fixed-λ anchor (n = 2000), layer 54, and the 8B fixed-λ arm are running on box 4.

**Done 2026-09-05 06:00Z:** second fixed-λ anchor (n = 2000) at layer 43 — no net rise (dip at 600, partial recovery), level 0.003–0.022 below the 600 anchor; the stronger penalty gives the higher cosine, so an anchor < 600 is a new to-do (report 23). **Reopened (lost with box 4):** layer 54 at either anchor; the 8B fixed-λ arm. **New from report 22:** footprint-matched / orthogonal control with a `rand − none` contrast; rank-k subspace ablation; attention-masking arm; gate rewrites per dose; resolve the pre-registration path from the driver's directory; format `N_REF` into `c_n_formula`'s prose.

**Open to-dos created by this map** (not yet done): a stability-scored, finer C sweep including C = 0.005 fixed; record model/revision and a partial-run flag in `a2_followup.py`; fix the cross-estimator CI's degenerate threshold (overlap, not n_pool ≤ cross_n); give `paired_slope_contrast` scenario blocking and dose pairing (one-sample cluster bootstrap on the paired differences) and fix the provenance stamp; store readout row norms in every result; seed α = 0 generation by emotion so per-emotion contrasts are independent; add random/orthogonal steering, cross-decode, A-span readout and context-ablated readout controls to any present-vs-other design before a positive reading; express readouts in signal units (fraction of the pool's e-vs-not-e separation) rather than noise units; put `logreg_cv` (or a fixed-effective-λ arm) into the split-half sweep before Paper A's non-convergence claim is written; emit the raw-CAA vs raw-logistic cosine and the `space='std'` cosine; draw the cross-estimator subsample per layer with a CI; hash the A2 driver into `code_sha`; make the B1 decision rule sign-aware and report contrasts as fractions of the `none` slope with an MDE; hash the driver into `code_sha`; drop `seeds.run` or seed `gen_B`; fix the two dead control pointers; report run-to-run variability from the two B1 pools; record Ledoit–Wolf λ per fit and report `space='std'` alongside `'raw'` in A2's analysis; draw a fresh permutation per (seed, n) in `split_half` or state the nesting; include `logreg_cv` (or a C ∝ 1/n arm) in the split-half sweep; run `splithalf_independent.py --npz` on the real layer-16 and layer-43 features and commit the output; guard zero directions in `fit_direction`; seed `gen_B` in `b1_e4rerun.py`; report split-half at the ablated layers (13–42), not only at 43; point the `generation_side_manip_check` control at the driver's own `readout` (lines 245–273) and register it so drift is caught; compute the MDE per emotion (`acl_core.mde`) and report contrasts as fractions of the `none` slope; block `paired_slope_contrast`; add a
C ∝ 1/n arm and standardised-space reporting to A2's analysis; add token-level and ceiling arms
to B1's follow-up; make A3 scale-free; re-word `e0.cross_cos` in the registry after
adjudication; record `method` in stability outputs.

---


## 13. Rescope after the 2026-09-04 runs

Three runs completed today on the 27B (B1, A2, A3), one on the 8B is finishing (A2), and
eighteen blind critiques are in. This section scores the plan's gates against the files and
changes scope where a gate moved.

### 13.1 Gate scorecard

| gate (from §2–§8) | state | evidence |
|---|---|---|
| A0 kill criterion | passed, narrowed | `docs/review/A0_PRIOR_ART.md`; contribution 4 now dropped (§2.1 ⟨2026-09-04⟩) |
| A2 exit — "a results table where every number carries n, CI, seed and a pointer" | met for the 27B | `results/rev3/a2_estimator_qwen36-27b.json`, four `a2.focus_*` registry entries; open: std-space, tuned C, per-fit λ |
| A3 exhibit — present > other under the stable estimator | **failed** | `a3_dissociation_qwen36-27b.json` + `a3_scalefree_qwen36-27b.json`: 0/6 published config; dom→dom 4/6 raw → 1–2/6 on every scale-free reading, 3 reversed |
| B1 (§1.3) — resolves the filtering claim either way | **rule says `not_blocking`; channel question unresolved** | `b1_e4rerun_qwen36-27b.json`: MC 6/6, emo−rand significant 3/6 with one reversal that is a grid artifact; no token-level or ceiling arm; A's tokens untouched |
| Ship condition "E4 resolves either way → write it up" | **not triggered** | the rule fired, but reports 04/16 show the design cannot locate the channel; follow-up arms first |
| Phase 0 exit — nothing asserted that files contradict | not met | one category (c) remains; §12 lists sentences corrected today |
| Phase 1 exit — every instrument passes its own validity check | partial | B1: stability gate passes at layer 43 only; ablated layers ungated; B unseeded. A2: pipeline valid, two bugs patched mid-run |
| Phase 2 exit — corrected results table with disagreements explicit | partial | rev-3 tables exist for A2/A3/B1 with critiques; B2–B4 not run |

### 13.2 Paper A — rescoped

**Before:** a methods paper on logistic-direction non-convergence with a downstream exhibit
(A3) showing a published result flip. **After:** a *measurement and reporting-protocol note*.

- Keep: contribution 1 (disjoint-half reproducibility, three pools), the RAPTOR framing from
  A0 (overlapping-subsample robustness ≠ disjoint-half reproducibility), A4's protocol —
  now with a specific clause: *state the regularisation schedule*, because the logistic
  curve's shape is a function of it (A2 delta).
- Contribution 2 (non-convergence) — **survives, reworded** by the A2 follow-up and its
  critique. Paper sentence: *"At layer 43 of the 27B, with the per-sample L2 penalty held at
  its C = 0.5, n = 600 value, the multinomial logistic row's disjoint split-half cosine moves
  by less than 0.03 over n = 150 → 2000 (0.581 → 0.581) while difference-of-means rises
  0.732 → 0.974 on the identical splits; cross-validated C reaches only 0.608. Over the
  reachable range the logistic direction shows no material convergence, though n ≤ 2000 at
  d = 5120 remains a proportional-regime observation, not an asymptotic one."* RAPTOR-style
  tuning is engaged (A0 consequence 2) and moves the number by about a tenth of the gap on an
  accuracy-scored grid; a stability-scored sweep and a second anchor are the open items.
- Dropped: contribution 4 and the A3 exhibit. A3's finding moves to **Paper B** as the
  correction of the published "load-bearing" claim (reversed under every configuration).
- A1 (generalise to refusal/persona directions) runs only if contribution 2 survives; it
  cannot rescue a C-dependent headline.
- New standing requirement for every stability number: both spaces (`raw`, `std`), the
  cross-class cosine next to the within-class one, and a decoding-accuracy curve — a
  near-common axis for all six emotions scores 0.9 on split-half while carrying little
  class-specific signal (report 18).
- Venue unchanged (TMLR); length shrinks.

### 13.3 Paper B — rescoped

- B1's `not_blocking` is the rule's output, not a finding about the channel. The follow-up
  driver (`src/rev3/b1_followup.py`, launched 2026-09-04 on box 2 with B1's own probe pool)
  adds the token-level and ceiling arms, seeds B, gates every ablated layer, draws a fresh
  random direction per cell, and uses a sign-aware rule with an MDE. Its outcome, not B1's,
  decides §1.3.
- B2 (contagion under the stable estimator) is **partially delivered by A3's dom→dom arm**:
  present slopes with scenario-blocked CIs for six emotions, five excluding zero, calm not.
  Report it as B2-partial with A3's caveats (two reps; shared α = 0 baseline).
- B3 is partially carried by B1's `cross` arm; B4 (scramble rebuild), B5–B10 unchanged.
- The "present shift exceeds other shift" published claim is now *reversed* with a
  non-circular probe; §0.4's correction sweep should say so in the paper and dossier.
- The "present and other subspaces near-orthogonal" claim (`e0.cross_cos`) holds under
  logistic rows on both models (which sit at the 1/(K−1) floor by construction) but **not
  under difference-of-means on the 8B** (present↔other 0.379 vs within 0.427). Any restated
  version must name the estimator and the null.

### 13.5 §1.3 resolved by the follow-up ⟨2026-09-04 17:20Z; corrected after results/reports/20⟩

None of §1.3's three outcomes fits. The direction is stable at every ablated layer (0.88–0.92),
the generation-side check passes 6/6, and blocking is **partial**: a quarter to a third of the
dose-response for afraid and sad (angry too under a scenario-clustered bootstrap), two-sided
nulls for the rest, and ablating at every position removes strictly more readout without
changing B further. The defensible sentence for Paper B, as the blind critic worded it:

> *Projecting the emotion direction out of A's span at layers 13–42 removes most of the
> model's own readout of A's affect (drop 0.11–0.37, 6/6, vs ≤ 0.006 for norm-matched random)
> but reduces B's dose-response by only 25–37% in two of five testable emotions (three under
> a scenario-clustered bootstrap); extending the mask to every prefill position removes
> strictly more readout and changes B no further, so positional coverage is not the limit.
> Whether the remainder is rebuilt above the ablation window, carried non-linearly, or simply
> absent from this rank-1 direction is not distinguished by this design.*

Not licensed: "rebuilt above the window" (needs an all-layer ablation arm), "travels with A's
tokens" (needs an affect-preserving control rewrite and a refusal filter; the text arm's
afraid result is 47% refusals), and any one-sided reading of the nulls. Paper B's filtering
section is rewritten around the *fraction* with the contrast-bootstrap detectable effect
stated (13–50% of the none slope per emotion), not the file's `mde`.

**⟨2026-09-04 22:05Z⟩ B1c answers the last branch.** The pre-registered all-layer test
(`docs/planning/PREREG_B1c.md`; report 22; registry `b1c.*`) returned **H1**: with the
direction removed at every hidden state 13–63, the median blocked fraction is 0.09 [0.04, 0.25]
over five testable emotions; layers 43–63 add nothing detectable for afraid, angry, happy and sad
(unpowered for desperate and calm); afraid and sad replicate their partial block (21%, 33%);
both rewrite arms fail their own checks again. So "rebuilt above the window" is out for this
direction, and what remains is the rank-1 branch: the blocked fraction tracks how much readable
affect the ablation removes (r ≈ 0.85; blocked-per-removed ≈ 0.27), i.e. the `dom` direction
captures only part of the affect representation and the part it captures carries only part of
the transfer. The Paper B sentence becomes:

> *Projecting the difference-of-means emotion direction out of A's span at every hidden state
> 13–63 during B's prefill removes a median 9% [4%, 25%] of B's dose-response (21% and 33% for
> afraid and sad, null elsewhere), no more than the 30-layer window did, while a unit-norm
> random direction changes nothing. The single direction is therefore insufficient to carry the
> transfer; whether a higher-rank affect subspace or B's re-derivation from A's tokens carries
> the rest is not distinguished by this design.*

Open for Paper B: a rank-k subspace ablation and an attention-masking arm (PREREG §7); a
footprint-matched control with its own contrast; a rewrite design that survives its check.

### 13.4 Runs launched under this rescope

| run | box | purpose | status |
|---|---|---|---|
| **B1c** (`b1c_alllayer.py`, GPU; B1's probe pool; pre-registered `PREREG_B1c.md` @ cc2dc32) | `sb-58e32f35598fd1b0` (box 3) | all-layer ablation hs 13–63, `rand_all`, `text`/`text_keep`, 4 doses, median-fraction decision rule | ✅ complete 22:05Z: verdict **H1** (0.09 [0.04, 0.25]); 8 `b1c.*` entries quotable; report 22 + blind critique; see §13.5 |
| A2 add-ons (`a2_followup.py`, CPU) | `sb-ea69c0d19be26d97` (box 4) | second fixed-λ anchor n = 2000 at layers 43, 54; layer 54 at n = 600; fixed-λ arm on the 8B (layers 14, 21, 27) | ⚠️ layer 43 of the n = 2000 anchor complete (20:53Z; `a2f2.*`, report 23); layer 54 partial (22/48 cells); the layer-54 @ 600 and 8B sweeps **lost** — the operator's laptop suspended 22:24Z–05:51Z and both leases ended meanwhile |
| A2 follow-up (`a2_followup.py`, CPU; features regenerated from the saved 27B pool by `a2_regen_feats.py`) | `sb-45376053750d2753` | tuned-C / fixed-λ logistic; std-space cosines; CAA-raw cross-estimator with CI; λ per fit | ✅ layers 16 + 43 complete (17:42Z); layer 54 lost when box 1's lease expired ~17:55Z. Four `a2f.*` entries quotable |
| B1 follow-up (`b1_followup.py`, GPU; B1's own probe pool and cached features, so directions are identical to the completed run) | `sb-45376053750d2753` | ceiling + text arms, seeded B with common random numbers across arms, per-layer gate, fresh random per (emotion, rep), sign-aware rule + MDE | ✅ complete 17:20Z: verdict `mixed`; see §13.5 |
| A2 on Llama-3-8B-abl | `sb-45376053750d2753` | the 8B half of the n/d anomaly | ✅ complete 13:24Z; anomaly reproduces |

*Rev 3, compiled 2026-09-01; amended 2026-09-04 (§11, §12, ⟨2026-09-04⟩ marks). Supersedes `docs/planning/ROADMAP.md` and revs 1–2; that document's honesty
guardrail carries forward unchanged, and §1 is the clearest evidence so far of why it is
needed. Companion documents: `docs/review/AUDIT.md` (claims audit), `docs/review/NOVELTY_REVIEW.md` (novelty and
literature).*

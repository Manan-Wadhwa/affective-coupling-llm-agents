# Pre-registration — B1c: does the mid-stack emotion direction carry a minority of agent-to-agent transfer?

**Status:** committed before any code for this experiment was written (RESEARCH_PLAN §7).
**Author of record:** the orchestrator, 2026-09-05; to be countersigned by the PI before launch.
**Supersedes:** nothing. Extends B1 (`results/rev3/b1_e4rerun_qwen36-27b.json`) and its
follow-up (`b1_followup_qwen36-27b.json`) on the same probe pool.

## 1. The question

The B1 follow-up showed that projecting the difference-of-means emotion direction out of A's
tokens at hidden states 13–42 during B's prefill removes most of the model's own readout of A's
affect (drop 0.11–0.37 on the forced-choice readout, 6/6) but reduces B's transmitted
dose-response by only 25–37% for two of five testable emotions (three under a scenario-clustered
bootstrap), with two-sided nulls for the rest — and that extending the ablation to every prefill
*position* removes strictly more readout without reducing B's dose-response further
(`b1f.ceiling_equals_emo`). The blind critique (results/reports/20) was right that this cannot
distinguish three explanations:

  (i) B rebuilds the affect from A's tokens in layers above the window (43–63);
  (ii) the affect was never carried by this rank-1 direction at any layer;
  (iii) the affect is carried non-linearly.

B1c separates (i) from (ii)+(iii) by ablating at *every* layer from 13 to the top, and separates
"direction" from "tokens" with a token-level arm that has the control the follow-up lacked.

## 2. Hypotheses, stated before the data

- **H1 (rank-1 insufficiency).** With the emotion direction projected out at all layers 13–63
  during B's prefill, B's dose-response is still at least half of its unablated value: the
  median blocked fraction over testable emotions is **≤ 0.50**.
- **H1′ (filterable channel).** All-layer ablation blocks most of the transfer: median blocked
  fraction **≥ 0.80**.
- The region 0.50–0.80 is **indeterminate** and will be reported as such, not resolved
  post hoc.

Either H1 or H1′ is a publishable result. H1 supports the sentence *"a rank-1 emotion direction
does not carry agent-to-agent affect transfer; the transfer is not linearly filterable at any
depth"*. H1′ supports *"the transfer is carried by the emotion direction and can be filtered"*,
which is the repo's original headline with an experiment behind it for the first time.

## 3. Design

Fixed and identical to the completed runs: model `Qwen/Qwen3.6-27B` at revision `6a9e13bd…`;
probe pool `probe_qwen36-27b.jsonl` (k = 60, 2160 generations, sha `70129f86…`) split into the
same DIR/READ halves by the same seed; steering direction `dom` at layer 42, scaled by `rms`;
readout probe fit on READ; 29 scenarios; estimator `dom`.

**Doses.** α ∈ {0, 0.33, 0.67, 1.0} — four points, so the slope is no longer the endpoint
contrast (RESEARCH_PLAN §6 ⟨2026-09-04⟩). Reps: 3 → n = 87 per (emotion, dose, arm).

**Arms (six).**

| arm | what | purpose |
|---|---|---|
| `none` | no ablation | baseline |
| `emo13_42` | emotion direction out of A's span, hs 13–42 | replicates the follow-up's `emo` |
| `emo_all` | emotion direction out of A's span, **hs 13–63** | the test of H1 vs H1′ |
| `rand_all` | fresh norm-matched random unit direction per (emotion, rep), hs 13–63 | the control for `emo_all`; `removed_norm` recorded per arm and per position |
| `text` | A's message replaced by an unsteered neutral rewrite; refusals and empty/identical rewrites **excluded**, count reported | token-level ablation |
| `text_keep` | A's message rewritten with the instruction to keep its tone and emotion but change every specific | affect-preserving control for `text`: separates "affect removed" from "context changed" |

B's replies are seeded with a common seed across arms (the follow-up's rule). The readout arm
`ceiling` is dropped: positional coverage was answered by the follow-up.

**Gate.** Split-half stability of the `dom` direction at every layer 13–63 (n = 600/half,
10 splits); the run proceeds only if the class mean is ≥ 0.80 at every ablated layer, and every
per-class value is reported.

## 4. Outcomes

- **Primary:** B's present-e score from the READ probe at layer 43, computed on B's *generated*
  reply in a separate forward pass with all hooks off (as `score_B` does now). Because the
  ablation acts only during generation, this outcome is valid under all-layer ablation.
- **Secondary (text-side):** the forced-choice readout applied to B's reply — "which emotion is
  B feeling" from the model's own next-token distribution, hooks off.
- **Manipulation checks, each pre-declared as pass/fail:**
  1. MC-steer: A's readout rises with α (slope CI excludes 0) for the emotion.
  2. MC-ablate on A's span: `emo_all` lowers A's readout at least as much as `emo13_42`
     does; `rand_all` lowers it by < 0.02. If `emo_all` fails this for ≥ 3 emotions the
     instrument failed and no channel claim is made.
  3. Text quality: degenerate + refusal fraction ≤ 0.10 per cell and median perplexity ≤ 2×
     the `none` arm's; cells failing are excluded and counted.
  4. `text` rewrites: A's readout on the rewrite ≤ 0.5 × its readout on the original for the
     emotion; emotions failing this have their `text` arm reported as *uninterpretable*, not
     as evidence.

## 5. Analysis, fixed in advance

- Per-arm dose-response slope: OLS over the four doses on per-scenario means; CI by
  **scenario-blocked** bootstrap (29 clusters).
- Contrasts (`emo_all − rand_all`, `emo13_42 − rand_all`, `text − text_keep`, `text − none`):
  paired across arms **and doses**, scenario-blocked — a one-sample cluster bootstrap on the
  per-scenario paired slope differences. `acl_core.paired_slope_contrast` will be given
  `block=` and dose pairing *before* the run; the change is recorded in the results log.
- **Testable emotions:** those whose `none` slope CI excludes zero. Others are reported but do
  not enter the decision.
- **Blocked fraction** = −(arm − control contrast) / `none` slope, with the CI-implied range,
  two-sided; the denominator's own CI is reported next to it.
- **Detectable effect** per emotion from the contrast's own bootstrap (not `acl_core.mde`).
- **Decision:** median blocked fraction of `emo_all` vs `rand_all` over testable emotions,
  with a bootstrap CI over scenarios. H1 if median ≤ 0.50 and the upper CI bound ≤ 0.65; H1′
  if median ≥ 0.80 and the lower bound ≥ 0.60; otherwise indeterminate. Per-emotion counts
  (significant negative / positive / null) are reported alongside; they do not override the
  median rule.
- **Sign-aware.** Positive contrasts count as anti-blocking, never toward blocking.
- Multiplicity: the decision is one test (the median). Per-emotion contrasts are descriptive
  and reported with BH-FDR over the six.

## 6. Stopping rule and what gets reported regardless

- Compute: 6 arms × 4 doses × 3 reps × 6 emotions = 432 arm-rows ≈ 2 h on one 96 GB GPU, plus
  ~15 min of gating. The run is checkpointed per arm-row and pulled every 10 minutes.
- If the sandbox dies with ≥ 4 emotions complete, the analysis is run on the complete emotions
  and labelled partial; the decision rule still applies only when all six are complete.
- Everything above is reported whichever way it comes out, including "indeterminate" and
  including a failed manipulation check.

## 7. What this does not test

The 8B; any layer below 13; whether transfer that survives all-layer ablation is carried
non-linearly or by attention to tokens (that needs an attention-masking arm, out of scope
here); the β budget (B7).

## 8. Provenance requirements

Result file carries: model revision, `code_sha` of the driver and core, the probe pool sha, the
seed rule, per-layer gates, per-arm `removed_norm` and mask sizes, per-row `gen_seed`, the
exclusion counts from §4.3–4.4, and this document's git commit hash.

## Addendum, 2026-09-05, before launch (driver written, not yet run)

Recorded so that it cannot be read as a post-hoc discovery:

1. The §5 decision resamples *scenarios*, not emotions. With an even split of blocked
   fractions across emotions (e.g. three near 0.7, three near 0.2) and little scenario-level
   heterogeneity, the median sits at the midpoint of the two clusters and is estimated
   tightly, so the rule can return **H1** on a median near 0.45 even though half the emotions
   are strongly blocked. This is the pre-registered rule and it stands; the per-emotion
   counts reported alongside are what would show such a split, and the write-up must show
   them next to the verdict.
2. Ablation covers hidden states 13–63 (decoder blocks 12–62). Hidden state 64 is the last
   block's output, which feeds only the final norm and head, so projecting there at A's
   positions cannot change anything B attends to.
3. "degenerate + refusal ≤ 0.10" is applied to their union (`is_degenerate` already includes
   refusal); both fractions are recorded separately.
4. Slopes and contrasts are computed on the balanced panel of (rep, scenario) keys present at
   every dose in every arm being compared, because a dose-paired contrast is undefined for a
   scenario missing a dose; `n_scenario_keys` is recorded per arm.
5. The §4.4 rewrite check applies to `text` only; `text_keep` is the affect-preserving control
   and its readout ratio (expected ≈ 1) is reported, not gated.
6. `instrument_failed` uses the full MC-ablate check (emo_all drop ≥ emo13_42 drop AND
   rand_all drop < 0.02); the count under the emo_all-only reading is reported alongside.

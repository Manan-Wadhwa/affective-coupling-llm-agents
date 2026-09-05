# PREREG B1d — rank-k subspace ablation: does a linear affect subspace along A's span carry the transfer?

Written 2026-09-05 (IST; provenance timestamps in the result file are UTC), before the driver
`src/rev3/b1d_subspace.py` exists and before any generation. Successor to `PREREG_B1c.md`,
whose verdict was H1 (report 22): removing the rank-1 difference-of-means direction at every
hidden state 13–63 leaves a median 91% of the transfer, and the blocked fraction tracks how
much readable affect the ablation removes (r ≈ 0.85, blocked-per-removed ≈ 0.27). So the
rank-1 direction is insufficient, and the open branch is whether a *higher-rank linear*
affect subspace along A's span carries the transfer, or whether the channel is not linear
along A's span at all (tokens re-read by B, or non-linear features).

## 1. The question

If the emotion-class-mean subspace (rank 5, the span of the six present-emotion class means
minus their grand mean) is projected out of A's token span at every hidden state 13–63 during
B's prefill, does B's dose-response to A's steered affect collapse?

## 2. Hypotheses, stated before the data

- **H2 (linear subspace carries it):** median blocked fraction of `sub_all` vs `perm_all` over
  testable emotions ≥ 0.60 with the lower CI bound ≥ 0.40.
- **H2′ (not even the class-mean subspace carries it):** median ≤ 0.35 with the upper CI bound
  ≤ 0.50, *while the subspace ablation does remove the readable affect on A's span* (§4.2).
- **indeterminate** otherwise.
- **instrument_failed:** the subspace ablation fails §4.2 for ≥ 3 of the 6 emotions. This is
  reported as its own finding — the readable affect on A's span is not confined to the rank-5
  class-mean subspace — and the blocked fractions are then descriptive only.

Prediction on record: B1c's per-emotion readout residuals (0.16–0.79 of readable affect
removed by rank 1) and blocked-per-removed ≈ 0.27 extrapolate to a blocked fraction near 0.3
if the same ratio holds for a subspace that removes everything readable; H2 needs that ratio
to double. We do not know which way it goes.

## 3. Design

- Model, pool, directions, gate, doses, reps, scenarios, sampling seeds: identical to B1c
  (`Qwen/Qwen3.6-27B` @ 6a9e13bd, B1's probe pool sha 70129f86…, DIR/READ halves, `dom`
  present direction, split-half gate ≥ 0.80 at every ablated layer with n = 600 per half and
  10 seeds, doses {0, 0.33, 0.67, 1.0}, 3 reps, 29 scenarios, B sampled with common random
  numbers across arms).
- **Subspace, per layer L (hidden states 13–63):** on the DIR half in raw space, the six
  present-emotion class means minus their grand mean; orthonormal basis of their span from the
  SVD, keeping the top 5 right singular vectors (rank 5, the null direction dropped). The same
  subspace is used for every steered emotion — it is the linear affect representation, not a
  per-emotion direction.
- **Subspace stability gate (new):** for every ablated layer, the mean cosine of the principal
  angles between the DIR-half and READ-half subspaces must be ≥ 0.70; every value is reported;
  the run aborts before any generation if any layer fails (as B1c's gate did).
- **Arms (five):** `none`; `emo_all` (B1c's rank-1 arm, hs 13–63, the bridge); `sub_all`
  (rank-5 class-mean subspace, hs 13–63); `perm_all` (the same estimator on the same DIR half
  with the present-emotion labels permuted, a fresh permutation per (emotion, rep); rank 5,
  hs 13–63 — the footprint-matched control the B1c critique asked for: same data, same
  estimator noise, no emotion information); `randsub_all` (a Gaussian random orthonormal
  5-frame, fresh per (emotion, rep) and per layer; hs 13–63 — the naive control).
- Ablation: `acl_core.Ablate` extended to rank-k frames (`proj = (h @ Q) @ Qᵀ`); the rank-1
  path is unchanged and bit-identical (verified before the run). `removed_norm`, `k` and
  `n_layers_ablated` are recorded per row.
- No rewrite arms: three designs in a row failed their own checks; the token channel is out of
  scope here.
- Compute: 5 arms × 4 doses × 3 reps × 6 emotions = 360 arm-rows ≈ 2 h on one 96 GB GPU.

## 4. Outcomes and manipulation checks

- 4.1 Primary outcome as in B1c: B's present-emotion projection (`B_present_e`) per arm-row;
  secondary: the forced-choice readout on B's reply with hooks off.
- 4.2 **Subspace instrument check** (per emotion, top dose, A-span forced-choice readout
  `ctx_readout_e` with the intervention live): floor-corrected share removed
  = (none − arm) / (none − none at dose 0). `sub_all` passes if its share ≥ 0.80.
  `perm_all` and `randsub_all` are inert if their share ≤ 0.20 each; a control that is not
  inert is reported (`control_not_inert`) and does not override the verdict.
- 4.3 MC-steer as in B1c (A's readout rises with dose under `none`).
- 4.4 Quality exclusions as in B1c (degenerate + refusal ≤ 0.10, perplexity ratio ≤ 2).

## 5. Analysis, fixed in advance

- Slopes: OLS over four doses on per-scenario means; scenario-blocked bootstrap CIs.
- Contrasts, scenario-blocked and dose-paired (`acl_core.paired_slope_contrast(block=,
  pair_doses=True)`, 5000 draws): `sub_all − perm_all` (decision), `sub_all − randsub_all`,
  `emo_all − perm_all`, `sub_all − emo_all`, `perm_all − none`, `randsub_all − none`,
  `emo_all − none`.
- Testable emotions: `none` slope CI excludes zero. Others reported, not decided on.
- Blocked fraction = −contrast / `none` slope, sign-aware, CI-implied range; the decision is
  the median over testable emotions of the `sub_all − perm_all` fraction with a bootstrap over
  scenarios (numerator and denominator recomputed per draw), thresholds in §2. Per-emotion
  counts (BH-FDR over the testable emotions) are descriptive.
- The extra rank beyond 1 is read from `sub_all − emo_all` (descriptive, with its CI as an
  extra blocked-fraction bound, as report 22 did).

## 6. Stopping rule and reporting

Checkpointed per arm-row and pulled every 15 min. If the box dies with ≥ 4 emotions complete,
the analysis runs on the complete emotions and is labelled partial; the decision applies only
with all six. Everything is reported whichever way it comes out.

## 7. What this does not test

Any layer below 13; ranks other than 5; non-linear readouts; attention masking; the 8B.

## 8. Provenance

The result file carries the model revision, driver and core `code_sha`, the probe pool sha,
per-layer gates (both), per-arm `removed_norm`, `k` and mask sizes, per-row `gen_seed`,
`perm_seed`/`rand_dir_seed`, and this document's sha256 and git commit. The driver resolves
this document relative to its own directory (B1c's relative lookup failed) and the document is
staged next to the driver on the box.

## Addendum, 2026-09-05 06:30Z, before launch (driver written and verified, not yet run)

1. Driver `src/rev3/b1d_subspace.py` at commit 940e418 (sha256 864ef287…) passed its selftest
   locally and on the box, and an independent verifier's pre-launch review (prereg fidelity,
   drift against B1c, GPU path, a real forward-pass smoke test with a rank-5 frame on the 27B:
   rank-1 path bit-identical, rank-5 frame loses orthonormality only at the bf16 level,
   2.5e-4). No code change was requested.
2. The probe pool is B1c's file (sha 70129f86…); its features are recomputed on this box, so
   `emo_all` equals B1c's arm up to forward-pass numerics, not bit-exactly.
3. Outputs, checkpoint and log are written under `/marimo/results` (persists across lease
   renewals); the pre-registration is staged at `/marimo/acl/docs/planning/PREREG_B1d.md`.
4. The selftest plants the control arms as identical arrays, so it cannot detect a swapped
   control in the decision panel; the verifier checked that panel by reading (`("none",
   "sub_all", "perm_all")`). Recorded so the limitation is on file before the data.

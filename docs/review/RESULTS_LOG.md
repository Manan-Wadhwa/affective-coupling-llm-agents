# Rev-3 rebuild — running results log

Append-only record of what the rev-3 rebuild has actually measured. Every entry carries a
provenance pointer. Entries marked **PARTIAL** are from an in-progress checkpoint and are
not quotable; they are here so a lease expiry cannot erase the observation.

Rule carried from `RESEARCH_PLAN.md` §0.3 and enforced by `tools/check_provenance.py`:
a number does not enter this file without a pointer that resolves.

**Hardware.** 2 × NVIDIA RTX PRO 6000 Blackwell Server Edition (95 GiB each), driver
580.126.20, torch 2.11.0+cu130, transformers 5.14.1, Python 3.13.11.

**Model pin (§0.5).** `Qwen/Qwen3.6-27B` at revision
`6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`. 64 decoder blocks, hidden 5120, hybrid
Gated-DeltaNet: `layer_types` is 3 × `linear_attention` to 1 × `full_attention`.
`focus = round(0.67 × 64) = 43`, matching the published convention.

**Kernel note, recorded because it affects reproduction.** `causal_conv1d` does not build
on cu130 here, so `transformers.models.qwen3_5` falls back to
`torch_recurrent_gated_delta_rule` / `torch_chunk_gated_delta_rule` rather than the fused
FLA path. Both sandboxes were deliberately left on the same fallback so numbers are
comparable across them. Throughput: ~38 s per batch of 64 dialogues at 240 new tokens.

---

## 2026-09-02 · B1 — E4 rerun (§1.3), Qwen3.6-27B

Driver `src/rev3/b1_e4rerun.py`; core `src/lib/acl_core.py`; sandbox `sb-327d7f6b`.
Checkpoint pulled to `results/rev3/b1_cells_qwen36-27b.json` continuously by
`tools/pull_results.py`.

### B1.1 — direction stability gate — **PASSES**

Probe pool n = 1578 kept of 2160 generated (leak 271, unparsed 311), split into disjoint
DIR (789) and READ (789) halves.

| estimator | split-half cosine, n=789/half, 10 splits | 95% CI |
|---|---|---|
| difference of means (used) | **0.909** | [0.906, 0.912] |
| logistic regression | 0.577 | — |

Pre-declared gate was 0.80. **This is the number the whole rerun turns on.** The published
E4 ablated along a direction whose split-half cosine was 0.394, which is why §1.3 judged it
uninterpretable rather than null: ablating along a direction that is ~60% noise *is*
approximately ablating along a random direction with respect to any independent fit. At
0.909 that objection no longer applies, so whatever E4 shows now means something.

Separately, this is the first committed measurement in the family of the category-(b)
post-fix stability gate (claimed 0.967, artifact lost with a lease). It does not reproduce
0.967 exactly — different n, different pool, ten splits instead of three — and it is logged
as a new measurement, not as a recovery of the old one.

### B1.2 — manipulation checks, `desperate` — **BOTH PASS**

n = 87 per cell at α ∈ {0, 0.5} (3 reps × 29 scenarios), n = 29 at α = 1.0 so far.
Readout = the model's own next-token distribution over the six emotion words when asked, in
its own output, which emotion A is feeling. This is the generation-side check §1.3 requires;
suppression in the residual stream is not suppression in the output.

**MC-steer** — does steering actually reach A's *text*?

| α | 0.0 | 0.5 | 1.0 |
|---|---|---|---|
| readout of `desperate` on A's message | 0.108 | 0.745 | 0.960 |

**MC-ablate** — does projecting the direction out of A's tokens remove readable affect,
and is the removal *specific*?

| arm | α=0.5 readout | α=1.0 readout | mean residual norm removed |
|---|---|---|---|
| `none` | 0.745 | 0.960 | 0.00 |
| `emo` (ablate desperate) | **0.565** | **0.839** | 0.33–0.55 |
| `rand` (norm-matched random) | 0.743 | 0.959 | 0.67–0.70 |
| `orth` (norm-matched orthogonal) | 0.743 | 0.958 | 0.77–0.79 |
| `cross` (ablate calm's direction) | 0.685 | 0.924 | 0.32–0.53 |

Emotion-specific drop at α=1.0 is **+0.121** against **+0.001** for the norm-matched
random control, and the bootstrap CIs of the `emo` and `rand` readouts do not overlap.
Neither norm-matched control moves the readout at all.

**Worth recording, because it complicates the phrase "norm-matched".** The emotion
direction removes *less* residual norm (0.33) than a random direction does (0.68) — the
projection magnitude depends on alignment with the activation, and the emotion direction is
less aligned with the mean activation than a random one is. The arms are matched in
direction norm (all unit) but not in norm removed. `removed_norm` is therefore reported
per arm rather than assumed equal, and no claim of exact norm matching is made.

Degeneracy is 0.00–0.03 and median perplexity 12.7–17.4 across every arm and dose, so
nothing here is bought with degraded text.

### B1.3 — does the ablation block contagion? `desperate` only — **PARTIAL**

B's own present-`desperate` score, held-out READ probe (never used to build the ablation
direction):

| arm | α=0.0 | α=0.5 | α=1.0 |
|---|---|---|---|
| `none` | −154.6 | −96.2 | −48.8 |
| `emo` | −150.8 | −101.5 | −61.1 |
| `rand` | −152.4 | −101.3 | −47.9 |
| `orth` | −154.1 | −101.4 | −61.1 |
| `cross` | −147.4 | −97.4 | −73.1 |

`emo` − `rand` slope difference **−14.88, CI [−41.40, +12.08]** — crosses zero, not
significant. Contagion rises strongly with dose in every arm including the ablated one.

**Reading, held loosely until all six emotions are in.** This is tracking toward the second
of §1.3's three outcomes: *still random under a stable direction with a passing manipulation
check → lexical affect ablation does not block contagion, which is the original claim now
actually evidenced.* The distinction from the published run is the whole point — that run
could not tell "does not block" apart from "the instrument was noise", because its direction
was at 0.394 and it had no generation-side check. This one can.

**Not a verdict.** One emotion of six. `src/rev3/b1_analyze.py` refuses to print a verdict
on a partial grid, deliberately: "manipulation check passed for k of 6" means nothing when
fewer than six have been run.

---

## 2026-09-02 · A0 — prior-art verdict (§2.3) — **Paper A survives, narrowed**

Full reasoning in `docs/review/A0_PRIOR_ART.md`. Summary: the problem statement is occupied
by RAPTOR ([2602.00158](https://arxiv.org/abs/2602.00158)) and must be cited, but RAPTOR's
stability metric is mean pairwise cosine over **overlapping 20% training-data ablations**
(two 80% subsamples share ~64% of their examples), not reproducibility across independent
fits; its baselines are xRFM and GCS, not difference-of-means; and its n/p sweep is for
accuracy, not direction stability. Disjoint-half reproducibility, non-convergence of the
direction with n, the logreg-vs-difference-of-means stability comparison, and a published
result reversing under estimator substitution all remain unoccupied.

---

## 2026-09-02 · Phase 0 — record repaired

- `docs/review/claims.json` + `tools/check_provenance.py`: 22 claims classified (a)/(b)/(c)/(d),
  each with script, config and output pointer; control-dependent claims additionally carry
  the *lines* implementing the control, and the checker reports DRIFTED if those lines stop
  containing the construct the claim depends on. Current state: **0 quotable claims fail to
  resolve**; 2 category-(b); 2 category-(c); 1 category-(d).
- `README.md` rewritten — headline table replaced by a quotable set plus an explicit
  retraction table; model identity corrected (Llama-3.1 → Llama-3); dataset claim corrected.
- Retraction banners added to `docs/writeups/paper.html` and `docs/writeups/dossier.html`.

---

## Method fixes made during the rebuild (each found by testing, not inspection)

1. **`pca_diff` was measuring nuisance, not signal.** Re-centering the difference cloud
   cancels the class offset algebraically; leaving it uncentered returns the dominant
   nuisance axis whenever within-class spread exceeds the class offset, which is the normal
   regime for residual streams. Implemented as sign-randomised **matched-pair** PCA (Zou et
   al.), pairing on the other-speaker emotion. Cos-to-truth on a synthetic with a known
   answer: **0.11 → 0.81**. The dependence on matched pairing is itself reportable — the
   estimator is worthless without it.
2. **Ledoit–Wolf made the A2 grid infeasible.** sklearn forms and inverts a d×d matrix;
   at d=5120 that is ~18 s per fit and the grid needs thousands. Every quantity reduces to
   the n×n Gram (‖S‖_F = ‖G‖_F, and Woodbury for the precision-apply). **Exact to 2e-15
   against sklearn, 18 s → 0.27 s.** `acl_core.lw_shrinkage_and_apply`.
3. **Decode accuracy had no intercept**, so argmax over one-vs-rest rows with unequal norms
   was comparing score offsets rather than directions, unfairly penalising the mass-mean
   family. Per-class midpoint intercept added.
4. **Pool caches could silently mis-pair prompts with generations.** Records are keyed by
   position in `pool_jobs(k, seed)`, so resuming a k=160 cache under k=60 pairs every cached
   generation with a different prompt, with no downstream symptom beyond wrong labels. The
   cache now carries a `_meta` header and `generate_pool` refuses on mismatch.
5. **§7 dual implementation satisfied for the split-half statistic.**
   `src/rev3/splithalf_independent.py` shares no code with `acl_core` — scipy L-BFGS instead
   of sklearn, raw-space instead of standardized-space difference-of-means, stratified
   instead of global splitting. Agreement on synthetic data: **≤0.006 at n ≥ 300**.

---

## In flight

*Table rewritten 2026-09-04; the 2026-09-02 sandboxes (`sb-327d7f6b`, `sb-50fc9327`)
expired before either sweep finished, taking their work directories with them.*

| Run | Box | State (2026-09-04) |
|---|---|---|
| B1 — E4 rerun, 27B, from scratch | `sb-45376053750d2753` | launched 06:13Z; pool generating |
| A2 — estimator battery, Llama-3-8B-abliterated, k=220 | `sb-45376053750d2753` | chained; starts on `B1_DONE` |
| A2 — estimator battery, 27B, k=160, all 7 depths | `sb-7aa3283deeba45cd` | launched 06:13Z; pool generating |
| A3 — present-vs-other paired test | `sb-7aa3283deeba45cd` | chained; starts on `A2_DONE`, reuses A2's pool (`--pool-cache`, `--probe-k 160`) |

---

## 2026-09-04 · re-read of the committed rev-3 checkpoints — **PARTIAL, no verdict**

No new runs. The cells files committed at `70894ba` were re-read and the derived summary
regenerated; `results/rev3/README.md` now labels every file in that folder with its grid
completeness and lists the numbers that have no committed source.

- **B1 is at 47/54 cells, not 7/54** as the 2026-09-02 entry and the previously committed
  `b1_partial_summary.json` state. Five emotions are complete; `angry` has 2 reps at α = 0.
  `b1_partial_summary.json` regenerated from the 47-cell file with `src/rev3/b1_analyze.py`
  (deterministic; two runs byte-identical).
- **Manipulation checks pass on point estimates for 5/5** measured emotions, with `calm`'s
  emo/rand CIs overlapping (specificity not established for it) and `afraid`'s A-readout
  non-monotone in α (0.576 at 0.5, 0.458 at 1.0).
- **Blocking contrast (emo − rand slope) excludes zero for 2 of 5** (`afraid` −25.7
  [−44.6, −6.3]; `sad` −28.1 [−51.4, −4.7]) and crosses zero for `desperate`, `happy`,
  `calm`. `desperate`'s value moved from −14.9 [−41.4, +12.1] (n = 29 at α = 1) to −7.7
  [−24.1, +9.2] (n = 87). The §B1.3 reading above, written from `desperate` alone, does not
  hold across the five: the grid is mixed. None of the three §1.3 outcomes is reached.
- **The B1.1 gate figure (0.909 / logreg 0.577) has no committed file.** It is written only
  to `b1_e4rerun_<tag>.json` at sweep completion, which has not been pulled. It is in
  exactly the position of the two category-(b) gates in `claims.json` and must not be
  quoted until the file lands.
- **A2 has one depth of seven** (`hidden_states[16]`, depth 0.25; 66 cells), not the focus
  layer. Layer-16 split-half: `dom` 0.50 → 0.96 across n = 75 → 2000; `logreg` 0.31 → 0.50,
  flattening between 1200 and 2000; `ridge` 0.25 → 0.31; `mass_mean_cov` and `lda_shrunk`
  *fall* with n (0.21 → 0.16, 0.25 → 0.20). Full tables in `results/rev3/README.md`.
- **Category (c) shrank from 2 to 1.** `src/information/cmi_pilot.py` is a CPU-only
  synthetic; run locally it writes `results/information/cmi_pilot.json` in ~6 s and gives
  the beta=0 lossy-conditioner estimate **0.8815** — the dossier's "0.88 nats" — byte-identical
  across two runs. Archaeology: the number entered at `c976a8e` (2026-07-25, the initial
  commit) and no run had ever been committed. Re-classified (a), left `pending` for
  adjudication. The remaining (c) is `dataset.13k_released`: all three counts (~12k / 12,333
  / ≈13k) entered together at `c976a8e` and nothing local can verify any of them.
- **Registry now covers rev 3.** Five `claims.json` entries added (the B1 gate as category
  (b) with no file; the partial B1 counts and the A2 layer-16 pair as `pending`); a
  `count_true` deriver added to the checker. 27 claims, 0 quotable failing to resolve.
- **Noted on the analyzer:** the `emo − X` contrasts in `b1_analyze.py` come from
  `acl_core.paired_slope_contrast`, which resamples samples within dose — paired, not
  scenario-blocked. The per-arm slope CIs are blocked. Recorded in `results/rev3/README.md`.
- **Reruns launched 2026-09-04** on two fresh sandboxes, from scratch (the old work dirs and
  probe pools expired with their leases): box 1 runs B1 (Qwen3.6-27B, 3 reps) then A2 on
  Llama-3-8B-abliterated (k=220); box 2 runs A2 (Qwen3.6-27B, k=160) then A3 reusing A2's
  pool. Same model revision `6a9e13bd` loaded on both. Checkpoints are pulled every 10 min
  into `results/rev3/inflight_box{1,2}/`.
- **Per-experiment reports with blind critiques** (`results/reports/`, 2026-09-04): seventeen
  groups; each critique was written by an agent given only the claim wording, the script and
  the JSON. Findings not in AUDIT.md are mapped to the plan in RESEARCH_PLAN §12 and, where a
  plan sentence was contradicted by a file, corrected in place (⟨2026-09-04⟩ marks). The
  largest: the present-vs-other dissociation under difference-of-means is 0 of 6 after
  scale normalisation (report 15); three-point dose grids make every legacy "slope" an
  endpoint contrast (reports 02, 03, 15); the legacy stability diagnostic used one split with
  nested subsamples (report 14).
- **Correction to the 2026-09-02 B1.1 entry (found by results/reports/16):** the split-half
  gate is computed at `n_per_half = min(N//2, 600)` = **600**, not 789 (789 is the DIR/READ
  half size). And it is measured at layer 43, which is not among the 30 ablated layers
  (13–42); the stability of the directions actually projected out is unmeasured. Also: the
  driver's `gen_B` calls `model.generate` without a seed, so B's replies in both the partial
  checkpoint and the running rerun are not reproducible; and the `generation_side_manip_check`
  control pointer names `acl_core.forced_choice_readout`, which the driver does not call.
- **A2 driver bug found by results/reports/17 and patched (2026-09-04):** `a2_estimator.py`
  paired the `pca_diff` estimator on the label it was fitting for the "other" label set, so
  that direction was all zeros and `dec/16/other/pca_diff` (0.193) is a class-0 base rate.
  Fixed to pair on the opposite label set; verified on synthetic data; deployed to both
  sandboxes. Box 2's A2 was restarted at 07:01Z from its pool cache (4352/5760 generations
  kept); box 1's chained Llama A2 will use the patched file. The covariance-corrected
  estimators' falling reproducibility with n reproduces on synthetic fixed-truth data, so it
  is estimator behaviour (Ledoit–Wolf λ shrinking with n), not a model finding.

---

## 2026-09-04 · B1 — E4 rerun, Qwen3.6-27B — **COMPLETE**, verdict `not_blocking`

Driver `src/rev3/b1_e4rerun.py`; sandbox `sb-45376053750d2753`; 06:13–07:57Z. Output
`results/rev3/b1_e4rerun_qwen36-27b.json` (provenance-stamped) + full cells + analyzer summary.
Registry: `b1.stability_gate_dom`, `b1.mc_ablate_pass_6of6`, `b1.emo_vs_rand_sig_3of6`,
`b1.verdict_not_blocking` — all resolve. Tables in `results/rev3/README.md`.

- **Gate:** dom split-half **0.907** [0.904, 0.911] at n = 600/half
  (logreg 0.567); passes 0.80. A new measurement, at layer 43, which is not an ablated
  layer. The 2026-09-02 entry's 0.909 stays unsourced and is superseded.
- **Manipulation checks:** 6/6 on point estimates, 4/6 on CI separation (afraid, calm fail).
- **Blocking:** emo − rand excludes zero for 3/6 — afraid and sad reduced, **happy increased**;
  desperate, calm, angry cross zero (calm has no transmission; desperate/angry consistent
  with up to a third blocked). Pre-declared rule (≥ 4 of 6) → `not_blocking`.
- **Reading, stated no more strongly than the file allows:** the rule's outcome is the plan's
  §1.3 outcome 2 *in form*, but the sentence "lexical affect ablation does not block
  contagion" is not licensed: it is a mixed result on an instrument that leaves A's tokens
  and layers 43–63 untouched, with no MDE. The token-level and ceiling arms (§1.3
  ⟨2026-09-04⟩) are the next run, not a write-up.
- **B1 delta critique (results/reports/16):** the rerun is a second independent run, not a
  resume (pool 1615 vs 1578); afraid's and happy's contrasts moved by more than their CIs
  between the two runs. happy's +31.2 is a three-point-grid artifact (all arms peak at
  α = 0.5). Perplexity range corrected to 9.8–21.2. Provenance: `code_sha` empty, `seeds.run`
  misleading (B unseeded), two control pointers dead.
- **A2 crashed after layer 16 (box 2, 08:02Z) on a second latent bug** — the cross-estimator
  and speaker-geometry blocks sliced the first n rows of the emotion-ordered pool, so only
  three classes were present and class indexing failed. Patched to a seeded subsample with
  `n_cls` explicit; both A2 jobs restarted from their cached pools and features.

---

## 2026-09-04 · A2 — estimator battery, Qwen3.6-27B — **COMPLETE** (7 depths)

Driver `src/rev3/a2_estimator.py` (two bugs patched mid-run, above); sandbox
`sb-7aa3283deeba45cd`; 08:11–11:42Z. Output `results/rev3/a2_estimator_qwen36-27b.json`
(provenance-stamped). Registry: `a2.focus_dom_split_half`, `a2.focus_logreg_split_half`,
`a2.focus_logreg_vs_dom_cos`, `a2.focus_decode_gap` — all resolve. Tables in
`results/rev3/README.md`.

- **Focus layer 43, n = 2000/half, 10 splits, raw space:** difference-of-means
  **0.974**, logistic **0.566**; logistic stops improving after
  n = 600 (0.575 → 0.568 → 0.566). The same shape on 5 of 7 depths; dom 0.966–0.979 on all 7.
- **Cross-estimator:** logreg ↔ dom 0.592 (0.54–0.60 on all depths) — the
  quantity 2604.08169 reports as 0.98–0.99; ridge groups with the covariance-corrected
  estimators (0.91), not with logreg.
- **Decode:** logreg 0.951 vs dom 0.883 (one split, no CI).
- **Regime:** n/d 0.83 everywhere; Fisher 0.09 → 0.19 and effective rank 327 → 230 across
  depth while the split-half curves barely move — the §2.2 anomaly is not explained by n/d.
- **Caveats carried from results/reports/17:** raw-space cosine only; n-grid points nested
  within a seed; C not n-normalised and `logreg_cv` not swept; decode on one split with a
  substituted rule. `stability.postfix_gate_27b` (category b) now has its re-measurement but
  the 0.967 sentence itself stays pending until retired.
- A3 started 11:43Z on the same box, reusing A2's pool (chain verified: "resuming with
  5760/5760 generations cached").
- **A2 delta critique (results/reports/17):** the logistic "plateau after n = 600" is not
  resolvable at 10 seeds and *reverses* under C = 0.05, which rises on 6 of 7 depths and ends
  above C = 0.5 at the focus layer — the curve's shape is the regularisation schedule. The
  raw-space logreg↔dom cosine (0.59) is `w/sd` vs `(μ₊−μ₋)/sd²`, not the raw-CAA quantity
  2604.08169 reports, so the two are not comparable yet. Within-present cosine 0.19 is the
  1/(K−1) floor; dom's 0.32 shows a shared non-emotion component. n/d is constant across
  depth by construction. Registry texts and README corrected; Paper A's non-convergence claim
  is untested until C is tuned per n.

---

## 2026-09-04 · A3 — present-vs-other dissociation, Qwen3.6-27B — **COMPLETE; exhibit does not survive**

Driver `src/rev3/a3_dissociation.py`; sandbox `sb-7aa3283deeba45cd`; 11:42–12:13Z; A2's pool.
Output `results/rev3/a3_dissociation_qwen36-27b.json`; scale-free re-read
`results/rev3/a3_scalefree_qwen36-27b.json` (`src/rev3/a3_scalefree.py`). Registry:
`a3.exhibit_dom_raw`, `a3.exhibit_dom_scalefree`, `a3.published_config_reversal`,
`a3.direction_stability`. Tables in `results/rev3/README.md`.

- **Published configuration (logreg→logreg), non-circular probe:** present > other **0 of 6**;
  other > present significant for 4 of 6 (raw), 3 of 6 (z0). The reversal the E2-CI critique
  found on the old file reproduces on a fresh pool.
- **dom → dom:** raw 4/6 (2 sig) → z0 2/6 (0 sig, 3 sig reversed) → zall 1/6.
  The raw count was the rows' norms, as results/reports/15 predicted. **The 5-of-6 exhibit
  is withdrawn; there is no present-vs-other dissociation on this data on any scale.**
- Stability on this pool (n = 600/half): dom 0.915, logreg 0.576 — third
  independent measurement today, all consistent.
- Consequence for Paper A: §2.3 A3's "downstream consequence" section has no exhibit. What
  remains for Paper A is the estimator-reproducibility result itself, now with the
  C-dependence caveat from the A2 delta critique.
- **A3 blind critique (results/reports/18):** every count reproduces; the collapse of the
  exhibit holds under Cohen's d, within-scenario SD and rank as well as z0 (dom→dom 1/6,
  0 sig, 3 reversed). Corrections: "norm artifact" overstated (row norms not stored; score-SD
  ratio 1.65–2.4×); `zall` is biased and dropped from the claim; `paired_slope_contrast` is
  neither scenario-blocked nor dose-paired (published-config reversals 3/6 after FDR, not 4);
  the six emotions share one α = 0 baseline; the design cannot separate B-models-A from
  attention to A's text. Registry, README, plan and `a3_scalefree.py` docstring corrected.

---

## 2026-09-04 · A2 — estimator battery, Llama-3-8B-abliterated — **COMPLETE** (7 depths)

Sandbox `sb-45376053750d2753`; 07:58–13:24Z. Output `results/rev3/a2_estimator_llama3-abl.json`.
Registry: `a2_8b.focus_dom_split_half`, `a2_8b.focus_logreg_split_half`,
`a2_8b.nd_anomaly_reproduces`, `a2_8b.focus_logreg_vs_dom_cos`. Tables in `results/rev3/README.md`.

- **Focus layer 21, n = 1200/half:** dom **0.941**, logreg **0.286** (C = 0.5 curve
  peaks at n = 300–600 and falls at 1200 on 5 of 7 depths). logreg↔dom 0.382.
- **The n/d anomaly reproduces on one pipeline:** better n/d on the 8B (0.95 vs 0.83), worse
  logistic reproducibility (0.29 vs 0.57), similar dom. n/d is not the mechanism; the
  regularisation schedule × activation geometry (lower effective rank, weaker SNR on the 8B)
  is the live candidate, untested until the tuned-C arm runs on both models.
- All of the day's planned runs are complete: B1, A2 (27B), A3, A2 (8B). Follow-ups
  (`b1_followup.py`, `a2_followup.py`) written, self-tested, under independent verification.
- **8B A2 delta critique (results/reports/19):** the logistic decline with n on the 8B is
  significant (600 → 1200 negative on 6/7 depths, 10/10 seeds pooled) and `logreg_c005`
  declines harder at all 7 — not a fixed-C artifact, unlike the 27B; the two models disagree
  on mechanism. Matched n = 1200: logreg 0.286 vs 0.568, dom 0.941 vs 0.955. "n/d is not the
  mechanism" reworded to "n/d does not order these two models". New: under dom the 8B's
  present↔other cosine is 0.89 of within-present — the near-orthogonal-speaker result fails
  under the stable estimator on this model. Registry, README, plan corrected.

---

## 2026-09-04 · B1 follow-up — **COMPLETE**, verdict `mixed` (sign-aware rule)

Driver `src/rev3/b1_followup.py` (verified same day); sandbox `sb-45376053750d2753`;
15:58–17:20Z; B1's own pool and features (directions identical; focus gate reproduces).
Output `results/rev3/b1_followup_qwen36-27b.json`. Registry: `b1f.verdict_mixed`,
`b1f.blocked_fraction`, `b1f.ceiling_equals_emo`, `b1f.text_arm`, `b1f.ablated_layers_gated`.

- **Ablated layers gated:** all 30 at 0.882–0.920; none below 0.80.
- **Blocking (emo − rand, B seeded, common random numbers):** afraid +0.25 [+0.16, +0.35],
  sad +0.37 [+0.19, +0.56] of the none slope; desperate, happy, angry null (0–35% admitted);
  calm untestable. happy's earlier +31 reversal is gone.
- **Ceiling arm:** ablating everywhere ≈ ablating A's span (5/6) — residual ablation at
  layers 13–42 caps at partial blocking; B recomputes affect above the window.
- **Text arm:** neutral rewrite of A's message cuts B's slope for 4/6 (angry −68 vs −24 under
  residual ablation); its own manipulation check fails for afraid and calm (paraphraser leak).
- **Reading for §1.3:** none of the three pre-declared outcomes as worded. Nearest defensible:
  *the residual emotion direction at mid-stack carries a minority of transmission (≤ ~35%,
  detectable for 2 of 5); the rest travels with A's tokens and is rebuilt above layer 42.*
- **B1 follow-up critique (results/reports/20):** counts flip on the variance choice — under
  a scenario-clustered, dose-paired bootstrap angry is a third blocker and the ceiling arm's
  one exception disappears (6/6 cross zero). Nulls are two-sided (amplification equally
  admitted). The file's `mde` is the wrong estimand. The text arm's afraid result is 47%
  refusals at α = 1. "B rebuilds above the window" is untested (ceiling extends positions,
  not layers). §13.5 sentence replaced; registry corrected; `paired_slope_contrast` blocking
  is now the single most load-bearing to-do.

---

## 2026-09-04 · A2 follow-up (tuned C / fixed λ / both spaces / CAA-raw cross-estimator) — **layers 16 and 43; layer 54 lost**

Driver `src/rev3/a2_followup.py` (verified same day); sandbox `sb-45376053750d2753`;
15:58–17:42Z for two layers; the box's lease expired ~17:55Z with layer 54 at 10/48 cells.
Output `results/rev3/a2_followup_qwen36-27b.json`. Registry: `a2f.focus_logreg_fixed_lambda_flat`,
`a2f.focus_tuned_c_does_not_close_gap`, `a2f.std_space_matches_raw`, `a2f.caa_raw_vs_logreg_raw`.

- **Fixed effective penalty:** logistic flat 0.581 → 0.581 from n = 150 to 2000 at the focus
  layer (3/10 seeds positive, mean -0.000); dom 0.732 → 0.974 on the same splits. The
  non-convergence claim stands on the 27B once the schedule is controlled; the afternoon's
  "it is the schedule" reading applied to the fixed-C curve only.
- **Tuned C:** 0.608 at n = 2000 — does not close the gap.
- **Both spaces:** standardised within +0.02–0.03 of raw, same shape.
- **Like-for-like cross-estimator (CAA raw ↔ logreg raw):** 0.427 at focus, 0.498 at layer 16,
  vs 0.98–0.99 in 2604.08169 (different trait/design).
- Both sandboxes are now dead; every rev-3 output is local.
- **A2 follow-up critique (results/reports/21):** every number verified and the C_n identity
  holds bit-for-bit at n = 600 on the real data. "Flat" withdrawn — the fixed-penalty curve
  moves < 0.03 over n = 150 → 2000 but dips at the focus layer and rises at layer 16; with a
  fixed penalty the target is fixed, so this is no material convergence in the proportional
  regime, not non-convergence in the limit, and depends on the single anchor. CV scored
  accuracy, not stability. Space gaps 0.002–0.029. CAA cosine still not the published
  quantity; its CI has no sampling content. Registry, README and plan reworded; Paper A's
  sentence is now the critic's.

---

## 2026-09-05 · B1c launched (pre-registered; `docs/planning/PREREG_B1c.md` @ cc2dc32)

Driver `src/rev3/b1c_alllayer.py` written from the pre-registration, self-tested, and passed
by an independent verifier with no must-fix items. Known limitations recorded before the run:
the A-span readout does not record its mask hit rate; cells whose every rewrite was excluded
vanish from the summary's `text_excluded_n` (still in rows); BH-FDR runs over testable
emotions, not all six; the `text` and `text_keep` prompts differ in framing beyond the
affect instruction; the per-layer gate thresholds the class mean, so one weak class at one
layer cannot fail it (per-class values are reported); feature extraction for 51 layers
overwrites the 31-layer cache. Runtime estimate 2.6–3.2 h on one 96 GB GPU. Box 4 runs the
A2 add-ons (layer 54; a second fixed-penalty anchor at n = 2000; the fixed-penalty arm on the
8B) in parallel.

---

## 2026-09-05 · Blocked, dose-paired re-read of every rev-3 contrast (CPU, no new data)

`src/rev3/reblock_contrasts.py` → `results/rev3/reblocked_contrasts.json`, using the
`block=`/`pair_doses=` path added to `acl_core.paired_slope_contrast` at 26e535d. Registry:
`reblock.b1f_emo_vs_rand_blocked`. Point estimates are unchanged by construction; what moves
is the intervals:

| contrast | unblocked → blocked+paired (significant negative / positive, of 6) |
|---|---|
| B1 emo − rand | 2 / 1 → 2 / 1 (unchanged; happy's reversal remains on this run) |
| B1 follow-up emo − rand | 2 / 0 → **3** / 0 (angry −24.1 [−44.9, −3.4]) |
| B1 follow-up emo − ceiling | 0 / 1 → 0 / **0** (angry's exception vanishes; 6/6 cross zero) |
| B1 follow-up emo − none | 3 / 0 → 4 / 0 |
| A3 dom→dom present − other | 0 / 2 → **1** / 2 (happy −46.6 [−93.3, −2.1]) |
| A3 logreg→logreg | 4 / 0 → 4 / 0 |

Reports 18 and 20's hand computations are confirmed to the digit. The blocked intervals are
wider for the B1 family (scenario effects are real) and similar for A3.

## 2026-09-04 22:05Z — B1c pre-registered all-layer ablation complete (box 3)
Verdict **H1** under PREREG_B1c §5: median blocked fraction 0.092 [0.038, 0.254] over five testable emotions (calm untestable). Significant blocking for afraid (21%) and sad (33%) only; `emo_all` vs `emo13_42` null for all six emotions, so layers 43–63 add nothing. Random control inert at twice the removed norm. All 51 ablated layers gated ≥ 0.88. MC-steer 6/6, MC-ablate failures 0, but the affect stays largely readable on A's span after ablation for four emotions (desperate 0.955 → 0.819). Rewrite arms uninterpretable 6/6. The driver's prereg stamp is empty (document not staged to the box); ordering rests on git times 19:02Z/19:22Z vs run start 19:32Z. Files `results/rev3/b1c_alllayer_qwen36-27b.json`, `b1c_cells_qwen36-27b.json`; registry `b1c.*` (8 entries); report `results/reports/22_rev3_b1c_alllayer.md`.

## 2026-09-05 06:00Z — box 4 second fixed-penalty anchor (N_REF = 2000): layer 43 complete, layer 54 lost; laptop suspend
Layer 43 (`a2_followup_qwen36-27b-nref2000.json`, stamped 20:53Z): `logreg_lam` raw 0.574, 0.578, 0.554, 0.556, 0.566 at n = 150…2000 — no net rise 150 → 2000 (a significant dip at 600 and partial recovery after it); sits 0.003–0.022 below the 600-anchor curve at every n (paired over shared splits). Corrected after the delta critique in report 23. Registry `a2f2.lam_anchor2000_no_rise`, `a2f2.anchor_level_shift`. Layer 54 reached 22 of 48 cells (plain logreg peak 0.573 at n = 600 then 0.554 at 2000) before the operator's laptop suspended at 22:24Z; both leases (boxes 3 and 4) ended before it woke at 05:51Z, so layer 54 at anchor 600 and the 8B fixed-penalty arm are lost again. File defects: empty `model`/`model_revision`, stale `c_n_formula` prose. Report `results/reports/23_rev3_a2_anchor2000.md`.

## 2026-09-05 06:35Z — A2 follow-up, 8B fixed-penalty arm complete (box 6, 638 s)
`a2_followup_llama3-abl.json`, layers 14/21/27, anchor 600. With the per-sample penalty fixed the logistic split-half cosine peaks at n = 150 and FALLS from there at every depth (focus layer 21: 0.369 → 0.276 from n = 150 to 1200, 0 of 10 paired splits positive); tuned C rises to 0.37–0.41; dom to 0.94; CAA-raw ↔ logreg-raw 0.33–0.37. So the 8B's ordering violation is not a schedule artefact. Registry `a2f8.lam_declines`, `a2f8.cross_estimator`; report 24. A first two-layer run reproduces bit for bit. Empty model/revision fields as before (rev dd67dd05 in the regen log).

## 2026-09-05 07:48Z — 27B layer 54 at both fixed-penalty anchors complete (box 6)
`a2_followup_qwen36-27b-l54.json` (anchor 600) and `-l54-nref2000.json`. Fixed-penalty logistic peaks at n = 150–300 (0.61 / 0.60), drops once 300 → 600 (−0.035, significant) and is flat after (0.56 / 0.55 at n = 2000; net 150 → 2000 −0.042 at both anchors). Not the 8B's monotone decline; layer 16 rises. Corrected after the delta critique in report 25; plain logreg peaks at 600 and falls to 0.554; tuned C 0.58; dom 0.98. Anchor gap ≤ 0.019. Cross-estimator CAA ↔ logreg 0.439. Registry `a2f54.*` (3); report 25. Box 6's queue is finished (8B 3-layer, layer 54 × 2 anchors) in 1 h 45 min total.

## 2026-09-05 08:05Z — A2 margin diagnostic (box 6): every logistic fit is a perfect separator at every n
`a2_margin_qwen36-27b-l43.json`, `-l54.json` and `a2_margin_llama3-abl-l21.json` (8B, same regime; cos(lam, dom) 0.91 → 0.43): training accuracy 1.000, log-loss ≤ 0.002, ≤ 25 iterations for n = 75…2000 per half under C_n, C = 0.5 and C = 50 alike; cos(lam, weak) ≥ 0.975; the weak-penalty direction is no more reproducible (0.52 vs 0.58); cos(lam, dom) falls 0.97 → 0.63 with n while dom's split-half rises 0.57 → 0.97. The follow-up's logistic directions are max-margin separators in the n ≪ d regime, not likelihood-driven estimates; the critics' "likelihood takes over" reading is not what happens. Registry `a2m.*` (3); report 26. Script `src/rev3/a2_margin_diag.py` (selftest reproduces the follow-up's fits to 1e-9).

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

| Run | Box | State |
|---|---|---|
| B1 — E4 rerun, 27B | `sb-327d7f6b` | sweep, 7/54 cells; `desperate` complete |
| A2 — estimator battery, 27B | `sb-50fc9327` | pool 3968/5760, then features, then the CPU grid |
| A3 — present-vs-other paired test | `sb-50fc9327` | chained; starts when A2 frees the GPU, reuses A2's pool at k=160 |
| A2 — estimator battery, Llama-3-8B-abliterated | `sb-327d7f6b` | chained; starts on `B1_DONE`, k=220 to match the committed run |

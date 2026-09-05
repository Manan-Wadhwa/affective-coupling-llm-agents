# results/rev3 — what is actually in these files

Rev-3 rebuild outputs (RESEARCH_PLAN rev 3, Phase 1–2). Everything here is a **checkpoint
pulled from a running sandbox** by `tools/pull_results.py`, not a finished run. Nothing in
this folder is quotable yet; the labels below say exactly how far each grid has got and
which numbers in the write-ups have a committed file behind them and which do not.

Numbers in this file are transcribed from the JSONs named next to them (or from
`b1_partial_summary.json`, which `src/rev3/b1_analyze.py` derives deterministically from
the cells file — regenerating it twice gives byte-identical output). The A2 tables are
what `src/rev3/a2_analyze.py` prints from the cells file. Nothing here is computed by hand. Last re-read: 2026-09-04, against commit `70894ba`.

| file | status | producing script | what it is |
|---|---|---|---|
| `b1_followup_qwen36-27b.json` | **RESULT · COMPLETE** (provenance-stamped, code_sha set) | `src/rev3/b1_followup.py` | follow-up arms ceiling/text, seeded B, per-layer gate, sign-aware verdict `mixed` |
| `b1f_cells_qwen36-27b.json` | **CHECKPOINT · COMPLETE 270 arm-rows** | `src/rev3/b1_followup.py` | |
| `b1_e4rerun_qwen36-27b.json` | **RESULT · COMPLETE** (provenance-stamped) | `src/rev3/b1_e4rerun.py` | driver output: gate, verdict, per-emotion summary, all rows |
| `b1_summary_qwen36-27b.json` | **DERIVED · COMPLETE 54/54** | `src/rev3/b1_analyze.py` | analyzer output on the full cells |
| `b1_cells_qwen36-27b.json` | **CHECKPOINT · COMPLETE 54/54** (was 47/54 at `70894ba`) | `src/rev3/b1_e4rerun.py` (core `src/lib/acl_core.py`) | per-cell raw arrays for the E4 rerun: one entry per (emotion, α, rep) holding five arm rows, each with per-scenario `B_present_e`, `B_other_e`, `A_readout_e`, `ctx_readout_e`, `removed_norm`, degeneracy, refusal, distinct-2, perplexity |
| `b1_partial_summary.json` | **SUPERSEDED** partial-grid derivation (47/54), kept for the record | `src/rev3/b1_analyze.py results/rev3/b1_cells_qwen36-27b.json --json …` | per-emotion dose slopes with scenario-blocked bootstrap CIs, arm contrasts, MC-steer, MC-ablate |
| `a3_dissociation_qwen36-27b.json` | **RESULT · COMPLETE** (provenance-stamped) | `src/rev3/a3_dissociation.py` | present/other slopes and paired contrasts under three estimator configurations; per-sample rows |
| `a3_scalefree_qwen36-27b.json` | **DERIVED · COMPLETE** | `src/rev3/a3_scalefree.py` | the same contrasts with each channel scaled by its own SD (z0, zall) |
| `a2_followup_qwen36-27b.json` | **RESULT · 2 of 3 layers** (16, 43; provenance-stamped, code_sha set) | `src/rev3/a2_followup.py` | tuned-C / fixed-λ / both-space split-half; CAA-raw cross-estimator with CI; λ per n |
| `a2f_cells_qwen36-27b.json` | **CHECKPOINT · layers 16, 43 complete; 54 partial (10/48)** | `src/rev3/a2_followup.py` | |
| `a2_estimator_llama3-abl.json` | **RESULT · COMPLETE** (provenance-stamped) | `src/rev3/a2_estimator.py` | the 8B half of the n/d question: 7 depths, same fields as the 27B file |
| `a2_cells_llama3-abl.json` | **CHECKPOINT · COMPLETE 462 cells** | `src/rev3/a2_estimator.py` | |
| `a2_estimator_qwen36-27b.json` | **RESULT · COMPLETE** (provenance-stamped) | `src/rev3/a2_estimator.py` | 7 depths: split-half, decode, cross-estimator matrix, speaker geometry, regime diagnostics, headline |
| `a2_cells_qwen36-27b.json` | **CHECKPOINT · COMPLETE 462 cells** (was layer 16 only at `70894ba`, different pool) | `src/rev3/a2_estimator.py`; read with `src/rev3/a2_analyze.py` | decode and split-half cells |

**Not in this folder, and therefore not committed anywhere:**

- ~~`b1_e4rerun_qwen36-27b.json`~~ — landed 2026-09-04 (see B1 below). The 2026-09-02
  log's 0.909 gate remains unsourced; the rerun's 0.907 replaces it.
- ~~`a2_estimator_qwen36-27b.json`~~ — landed 2026-09-04 11:42Z (see A2 below).

---

## B1 — E4 rerun, Qwen3.6-27B — **COMPLETE (54/54 cells), 2026-09-04 rerun**

**Files:** `b1_e4rerun_qwen36-27b.json` (driver output, provenance-stamped: model rev
`6a9e13bd`, acl_core `23c758a9`, 06:13–07:57Z, 6281 s) · `b1_cells_qwen36-27b.json` (full
checkpoint, 270 arm-rows; the 47-cell partial it replaces is in git history at `70894ba`) ·
`b1_summary_qwen36-27b.json` (from `b1_analyze.py`; every contrast matches the driver's own
`summary` block to 1e-6). `b1_partial_summary.json` is the superseded partial-grid derivation,
kept for the record.

Config: doses {0, 0.5, 1.0}; arms none / emo / rand / orth / cross; 3 reps × 29 scenarios
= 87 per (emotion, dose, arm); estimator `dom`; probe pool kept 1615 (DIR 807 / READ 808);
focus 43; ablated hidden states 13–42 (30 blocks); rms 81.3. B's replies are generated
**unseeded** (`gen_B` calls `model.generate` directly) — recorded in results/reports/16.

### B1.1 — direction-stability gate (`direction_stability`)

| estimator | split-half cosine, n = 600/half, 10 splits, raw space | 95% CI |
|---|---|---|
| difference of means (used) | **0.907** | [0.904, 0.911] |
| logistic regression | 0.567 | [0.560, 0.573] |

Per class (dom): happy 0.929, calm 0.948, sad 0.900, angry 0.942, afraid 0.859, desperate 0.867. Gate 0.80: **passes**.
This is a new measurement on a new pool, not a recovery of the 2026-09-02 log's 0.909. It is
measured at **layer 43**, the steering and read layer; the 30 directions actually projected
out (layers 13–42) have no stability figure (results/reports/16).

### B1.2 — manipulation checks (α = 1.0, n = 87 per arm)

| emotion | MC-steer slope [CI] | α=1 readout: none / emo / rand / orth / cross | drop_emo | drop_rand | separated | passes |
|---|---|---|---|---|---|---|
| desperate | 0.847 [+0.790, +0.900] | 0.955 / 0.823 / 0.955 / 0.955 / 0.937 | +0.133 | +0.000 | yes | yes |
| afraid | 0.147 [+0.030, +0.257] | 0.397 / 0.289 / 0.396 / 0.404 / 0.398 | +0.109 | +0.001 | **no** | yes |
| happy | 0.457 [+0.350, +0.560] | 0.744 / 0.564 / 0.742 / 0.743 / 0.760 | +0.181 | +0.002 | yes | yes |
| calm | 0.487 [+0.394, +0.576] | 0.764 / 0.637 / 0.760 / 0.758 / 0.714 | +0.127 | +0.004 | **no** | yes |
| sad | 0.506 [+0.429, +0.580] | 0.545 / 0.174 / 0.540 / 0.550 / 0.585 | +0.371 | +0.005 | yes | yes |
| angry | 0.560 [+0.470, +0.648] | 0.598 / 0.387 / 0.603 / 0.590 / 0.610 | +0.212 | -0.004 | yes | yes |

`passes` (sign test) 6/6; `separated` (emo's CI upper below rand's CI lower) 4/6. afraid's
MC-steer is the weakest (0.147) and its α = 1 modal readout is *desperate*, not afraid.

### B1.3 — blocking contrast (B's present score vs α; emo − X paired, not scenario-blocked)

| emotion | none slope [CI] | emo | rand | emo − rand [CI] | emo − none | emo − orth | emo − cross | blocked fraction of none slope [CI-implied] |
|---|---|---|---|---|---|---|---|---|
| desperate | +79.4 [+52, +106] | +71.6 | +81.3 | -9.8 [-26.1, +5.8] | -7.8 | -12.8 | +1.3 | +0.12 [-0.07, +0.33] |
| afraid | +182.6 [+146, +217] | +148.3 | +198.8 | -50.5 [-72.4, -29.1] **sig** | -34.3* | -58.4* | -22.9* | +0.28 [+0.16, +0.40] |
| happy | +48.6 [+9, +84] | +80.2 | +49.0 | +31.2 [+9.4, +53.3] **sig** | +31.6* | +20.0 | +20.0 | -0.64 [-1.10, -0.19] |
| calm | +46.4 [-0, +96] n.s. | +13.4 | +26.9 | -13.4 [-50.5, +23.5] | -33.0 | -17.7 | -15.3 | undefined (none slope n.s.) |
| sad | +102.1 [+72, +135] | +67.9 | +97.1 | -29.1 [-49.0, -10.0] **sig** | -34.2* | -32.5* | -24.8* | +0.29 [+0.10, +0.48] |
| angry | +195.3 [+156, +235] | +171.8 | +194.1 | -22.4 [-53.1, +7.3] | -23.5 | -20.8 | -51.3* | +0.11 [-0.04, +0.27] |

\* = interval excludes zero.

**What the file supports.** emo − rand excludes zero for **3 of 6**: afraid and sad in the
blocking direction, **happy in the opposite direction** (ablating happy's direction *raised*
B's dose-response relative to random). Desperate, calm, angry cross zero; calm has no
transmission to block (none slope n.s.); desperate and angry are consistent with up to a
third of the slope blocked. The driver's pre-declared rule (`b1_e4rerun.py:399-403`:
`blocking` needs ≥ 4 of 6 significant) returns **`not_blocking`** with `n_mc_pass` 6.

**What it does not support.** "Lexical affect ablation does not block contagion" as a
sentence: two emotions show a reduction, one an increase, three are underpowered nulls (no
MDE computed), and A's tokens plus layers 43–63 are untouched, so B can recompute affect from
the words (results/reports/04, 16; RESEARCH_PLAN §1.3 ⟨2026-09-04⟩ token-level and ceiling
arms). Nearest defensible sentence (results/reports/16, delta): *projecting the difference-of-means
emotion direction out of A's tokens at 30 mid-stack layers removes most of the model's own
readout of A's affect (paired specific drop 0.11–0.37, 6/6) yet reduces transmission to B's
reply by a detectable amount in only 2 of 6 emotions (afraid, sad); the remaining nulls are
underpowered (compatible with up to ~33% blocking), one emotion has no measurable transmission
(calm), and one contrast reverses sign under a dose grid whose response peaks at the dose the
slope discards.*

Text quality across all 270 arm-rows: degenerate ≤ 0.07, refusal ≤ 0.07, per-cell median
perplexity 9.8–21.2 (observed ranges, not pre-declared thresholds).

**Two independent runs, not one.** The 2026-09-02 partial (47/54, pool n = 1578) and this run
(pool n = 1615) share no cells — the checkpoint key includes the pool size, so the rerun
started clean. Between them afraid's emo − rand moved −25.7 → −50.5 and happy's +11.1 (n.s.)
→ +31.2 (sig), i.e. more than the reported CI half-widths: those CIs condition on one probe
pool and one random-direction draw and understate run-to-run variability.

**Happy's positive contrast is a dose-grid artifact.** All five arms peak at α = 0.5 (none
225 / 311 / 274; emo 212 / 308 / 292); the three-point OLS slope equals y(1) − y(0) and
discards the peak, so "+31.2" means "emo decayed less from the midpoint", not "ablation
increased transmission".

---

## B1 follow-up — same directions, new arms — **COMPLETE, 2026-09-04 17:20Z**

**Files:** `b1_followup_qwen36-27b.json` (provenance-stamped, `code_sha` set, seed rule
recorded, 4941 s) · `b1f_cells_qwen36-27b.json`. Driver `src/rev3/b1_followup.py`, written
and independently verified the same day (results/reports/16 addendum). Same probe pool
(n = 1615, hash-verified) and cached features as the completed B1, so the steering and
ablation directions are identical (the focus gate reproduces to every printed digit).

Changes from B1: B's replies seeded with **common random numbers across arms**; a fresh
random direction per (emotion, rep); the stability gate at **every ablated layer** (min
0.882 at hs 14, median 0.898, max 0.920 at hs 37 — none below 0.80); two new arms
— **ceiling** (emotion direction projected out at *all* prefill positions) and **text** (A's
steered message replaced by an unsteered neutral rewrite, no hooks); orth and cross dropped;
a **sign-aware pre-declared rule** with a per-emotion MDE.

### Manipulation check at α = 1 (readout on A's span; for `text`, on the rewrite)

| emotion | none / emo / rand / ceiling / text | drop emo | drop ceiling | drop text | separated |
|---|---|---|---|---|---|
| desperate | 0.955 / 0.823 / 0.955 / 0.722 / 0.668 | +0.133 | +0.233 | +0.287 | yes |
| afraid | 0.397 / 0.289 / 0.396 / 0.224 / 0.386 | +0.109 | +0.174 | +0.011 | **no** |
| happy | 0.744 / 0.564 / 0.742 / 0.432 / 0.560 | +0.181 | +0.312 | +0.185 | yes |
| calm | 0.764 / 0.637 / 0.762 / 0.531 / 0.706 | +0.127 | +0.233 | +0.058 | yes |
| sad | 0.545 / 0.174 / 0.540 / 0.089 / 0.438 | +0.371 | +0.456 | +0.107 | yes |
| angry | 0.598 / 0.387 / 0.604 / 0.331 / 0.480 | +0.212 | +0.267 | +0.118 | yes |

`passes` 6/6, `separated` 5/6 (afraid fails). The rewrite strips readable affect strongly
for desperate and happy, barely for afraid and calm — the same paraphraser leak report 02
found in E2-CI. Ablating everywhere (ceiling) still leaves 0.09–0.72 readable at layer 43.

### Dose-response and contrasts (paired; * = CI excludes zero)

| emotion | none | emo | rand | ceiling | text | emo − rand | emo − ceiling | ceiling − none | text − none | blocked fraction (emo vs rand) | MDE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| desperate | +78.2 | +79.8 | +77.2 | +78.0 | +69.1 | +2.6 [-14, +19] | +1.8 [-8, +12] | -0.2 [-17, +16] | -9.1 [-28, +10] | -0.03 [-0.24, +0.18] | 35 |
| afraid | +186.3 | +131.3 | +178.3 | +129.9 | +131.9 | -47.0 [-65, -30]* | +1.4 [-10, +13] | -56.4 [-76, -37]* | -54.4 [-78, -30]* | +0.25 [+0.16, +0.35] | 54 |
| happy | +57.9 | +58.2 | +58.2 | +60.1 | +67.2 | -0.0 [-20, +20] | -2.0 [-20, +18] | +2.3 [-19, +23] | +9.3 [-21, +41] | +0.00 [-0.35, +0.34] | 61 |
| calm | +37.9 n.s. | +4.5 | +27.3 | +4.6 | -4.3 | -22.8 [-49, +2] | -0.1 [-21, +19] | -33.3 [-62, -6]* | -42.2 [-79, -7]* | undefined | 83 |
| sad | +98.3 | +68.2 | +104.8 | +79.1 | +74.0 | -36.6 [-55, -18]* | -10.9 [-24, +2] | -19.2 [-39, +1] | -24.2 [-45, -3]* | +0.37 [+0.19, +0.56] | 51 |
| angry | +201.6 | +180.1 | +204.2 | +162.1 | +133.1 | -24.1 [-52, +3] | +18.0 [+1, +36]* | -39.5 [-66, -14]* | -68.5 [-104, -35]* | +0.12 [-0.01, +0.26] | 71 |

**Pre-declared rule → `mixed`:** n_block 2 (afraid, sad), n_antiblock 0, n_untestable 1
(calm), MC 6/6. The completed B1's happy sign-reversal is gone (−0.0 [−20, +20]).

**What the file supports** (corrected per results/reports/20). Projecting the emotion
direction out of A's span at layers 13–42 removes most of the model's own readout of A's
affect (drop 0.11–0.37, 6/6, vs ≤ 0.006 for norm-matched random) but reduces B's
dose-response by only 25–37% in two of five testable emotions under the driver's bootstrap
— three (adding angry) under a scenario-clustered one. Extending the mask to every prefill
position removes strictly more readout and changes B no further, so positional coverage is
not the limit. The token-level rewrite reduces B's dose-response for sad and angry; its
afraid result is refusal-contaminated (47% of rewrites at α = 1) and calm is untestable.

**What it does not support.** "Lexical affect ablation does not block contagion" (it blocks a
quarter to a third for two or three emotions); "B rebuilds affect above the window" (the
ceiling arm extends positions, not layers — the design cannot distinguish rebuilding above
layer 42 from the direction never carrying it or from non-linear carriage); "the remainder
travels with A's tokens" (the text arm has no affect-preserving control and its manipulation
check fails for afraid); the nulls are two-sided (up to ~35% blocking *or* 24–35%
amplification); the file's `mde` is the wrong estimand — from the contrasts' own bootstraps
the detectable difference is 13–50% of the none slope (96% for calm); common random numbers
share the seed but the streams diverge, so variance reduction is partial.

---

## A2 — estimator battery, Qwen3.6-27B — **COMPLETE (7 of 7 depths), 2026-09-04**

**Files:** `a2_estimator_qwen36-27b.json` (driver output, provenance-stamped: model rev
`6a9e13bd`, acl_core `23c758a9`, 08:11–11:42Z, 12,683 s of which the CPU grid was ~3.5 h) ·
`a2_cells_qwen36-27b.json` (full checkpoint, 462 cells; `a2_analyze.py` reads it). The
layer-16-only checkpoint committed at `70894ba` came from a different pool (n = 4261) and its
`dec/16/other/pca_diff` cell was a zero-direction artifact (pairing bug, patched 2026-09-04;
a second bug in the cross-estimator block was patched the same day — see RESULTS_LOG).

Config: k = 160 dialogues per 6 × 6 cell, 5760 generated, **4253 kept** (leak 654, unparsed
853); depths {16, 22, 29, 35, 43, 48, 54} of 64 (focus 43 = depth 0.67); n per half
{75, 150, 300, 600, 1200, 2000}; 10 disjoint splits per cell; nine estimators; cosine in
**raw space** (the steering direction, C/sd) — standardised-space cosine was not run.
`logreg_cv` is excluded from the split-half sweep by design.

### Headline (`headline`, focus layer 43, n = 2000/half)

| quantity | logreg | difference of means |
|---|---|---|
| disjoint split-half cosine | **0.566** [0.561, 0.572] | **0.974** [0.972, 0.975] |
| held-out present decode (one 70/30 split) | 0.951 | 0.883 |
| logreg ↔ dom direction cosine | 0.592 | |

### Split-half cosine at n = 2000/half, every depth

| hs | depth | block type | logreg | logreg_c005 | ridge | dom | dom_norm | pca_diff | mass_mean_cov | lda_shrunk |
|---|---|---|---|---|---|---|---|---|---|---|
| 16 | 0.25 | full | 0.523 | 0.508 | 0.336 | 0.966 | 0.967 | 0.933 | 0.175 | 0.211 |
| 22 | 0.344 | linear | 0.537 | 0.542 | 0.335 | 0.973 | 0.974 | 0.954 | 0.165 | 0.210 |
| 29 | 0.453 | linear | 0.577 | 0.575 | 0.354 | 0.971 | 0.972 | 0.949 | 0.176 | 0.223 |
| 35 | 0.547 | linear | 0.589 | 0.595 | 0.357 | 0.974 | 0.975 | 0.953 | 0.177 | 0.230 |
| 43 | 0.672 | linear | 0.566 | 0.575 | 0.349 | 0.974 | 0.974 | 0.954 | 0.176 | 0.232 |
| 48 | 0.75 | full | 0.561 | 0.573 | 0.351 | 0.974 | 0.975 | 0.953 | 0.181 | 0.232 |
| 54 | 0.844 | linear | 0.554 | 0.558 | 0.342 | 0.979 | 0.979 | 0.964 | 0.163 | 0.227 |

### Split-half vs n: logreg (left six) and dom (right six)

| hs | 75 | 150 | 300 | 600 | 1200 | 2000 | 75 | 150 | 300 | 600 | 1200 | 2000 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 16 | 0.307 | 0.380 | 0.428 | 0.476 | 0.498 | 0.523 | 0.497 | 0.680 | 0.803 | 0.891 | 0.942 | 0.966 |
| 22 | 0.368 | 0.444 | 0.491 | 0.543 | 0.537 | 0.537 | 0.560 | 0.732 | 0.842 | 0.915 | 0.956 | 0.973 |
| 29 | 0.362 | 0.444 | 0.501 | 0.561 | 0.569 | 0.577 | 0.537 | 0.712 | 0.829 | 0.906 | 0.951 | 0.971 |
| 35 | 0.390 | 0.469 | 0.531 | 0.586 | 0.588 | 0.589 | 0.562 | 0.732 | 0.844 | 0.915 | 0.956 | 0.974 |
| 43 | 0.392 | 0.471 | 0.525 | 0.575 | 0.568 | 0.566 | 0.562 | 0.732 | 0.842 | 0.915 | 0.955 | 0.974 |
| 48 | 0.384 | 0.461 | 0.511 | 0.572 | 0.571 | 0.561 | 0.561 | 0.730 | 0.842 | 0.915 | 0.955 | 0.974 |
| 54 | 0.410 | 0.483 | 0.523 | 0.573 | 0.564 | 0.554 | 0.617 | 0.772 | 0.871 | 0.931 | 0.965 | 0.979 |

### Decode, cross-estimator agreement, speaker geometry

| hs | present logreg | present dom | other logreg | other dom | logreg~dom | ridge~mass_mean_cov | present↔other cos (logreg / dom) | within-present cos (logreg / dom) |
|---|---|---|---|---|---|---|---|---|
| 16 | 0.910 | 0.796 | 0.784 | 0.502 | 0.543 | 0.906 | 0.057 / 0.211 | 0.190 / 0.359 |
| 22 | 0.925 | 0.843 | 0.814 | 0.560 | 0.564 | 0.902 | 0.043 / 0.178 | 0.191 / 0.347 |
| 29 | 0.940 | 0.856 | 0.875 | 0.633 | 0.561 | 0.910 | 0.044 / 0.123 | 0.196 / 0.342 |
| 35 | 0.949 | 0.868 | 0.878 | 0.648 | 0.595 | 0.908 | 0.045 / 0.126 | 0.193 / 0.337 |
| 43 | 0.951 | 0.883 | 0.860 | 0.589 | 0.592 | 0.908 | 0.033 / 0.096 | 0.193 / 0.320 |
| 48 | 0.952 | 0.878 | 0.846 | 0.583 | 0.591 | 0.908 | 0.036 / 0.124 | 0.193 / 0.339 |
| 54 | 0.939 | 0.883 | 0.827 | 0.560 | 0.599 | 0.882 | 0.032 / 0.142 | 0.190 / 0.353 |

### Regime diagnostics (present labels; n/d = 0.83 at every depth)

| hs | Fisher ratio | participation ratio | effective rank | top-1 var frac | class SNR | ‖x‖ |
|---|---|---|---|---|---|---|
| 16 | 0.093 | 81.5 | 327 | 0.057 | 1.72 | 49.7 |
| 22 | 0.127 | 74.3 | 319 | 0.068 | 1.83 | 51.2 |
| 29 | 0.129 | 67.8 | 293 | 0.071 | 1.80 | 61.0 |
| 35 | 0.144 | 65.5 | 293 | 0.077 | 1.84 | 66.3 |
| 43 | 0.156 | 59.7 | 265 | 0.082 | 1.84 | 81.2 |
| 48 | 0.157 | 58.1 | 252 | 0.081 | 1.84 | 89.3 |
| 54 | 0.189 | 48.6 | 230 | 0.099 | 1.83 | 112.4 |

**What the file supports, stated as literally as it allows.**

- On this pool, at every one of seven depths, the difference-of-means direction reproduces
  across disjoint halves at 0.97–0.98 by n = 2000 and is still rising; the multinomial
  logistic row (C = 0.5) reaches 0.52–0.59. ⟨2026-09-04, delta⟩ Its curve peaks at n = 600 on
  4 of 7 depths and is flat-to-declining after (pooled t ≈ −1.8 at 10 seeds, unresolved),
  **but the C = 0.05 row keeps rising on 6 of 7 depths and ends above it at the focus layer
  (0.575 vs 0.566)** — the shape depends on the untuned constant, as results/reports/14
  predicted. Ridge (α = 1000) sits at 0.34–0.36. The two covariance-corrected estimators sit
  at 0.16–0.23 and fall with n — estimator behaviour, per results/reports/17.
- The raw-space logistic and difference-of-means directions agree at 0.54–0.60 on every
  depth — on the *same* 2000-row subsample each time, with no CI. ⟨delta⟩ This is `w/sd` vs
  `(μ₊−μ₋)/sd²`, not the raw-CAA-vs-raw-logistic cosine 2604.08169 reports (0.98–0.99), so
  the two are not comparable until a rerun emits the raw quantity. Ridge agrees with the
  covariance-corrected pair (0.88–0.91), not with the logistic family.
- Logistic decodes better at every depth (0.91–0.95 vs 0.80–0.88), on one split without a CI.
- Present↔other cosine is 0.03–0.06 under logreg and 0.10–0.21 under dom at every depth.
  ⟨delta⟩ The within-present cosine of ≈0.19 under logreg is the algebraic floor 1/(K−1) for
  six one-vs-rest contrasts that sum to zero, not a measurement; dom's 0.32 sits 0.12 above
  it, i.e. dom's six directions share a non-emotion component. Normalised by each
  estimator's own floor the present/other asymmetry is ≈1.8×, not 2–4×, and "near-orthogonal"
  still needs a null the file does not contain.
- The regime diagnostics do not rescue the n ≪ d premise: n/d is 0.83 at every depth *by
  construction* (same n, same d); Fisher ratio doubles from 0.09 to 0.19 and effective rank
  falls 327 → 230 with depth, yet the depth with the most favourable regime (54) has the
  worst logistic trend and the best difference-of-means. Whatever drives the logistic curve,
  it is not n/d.

**What it does not support.** A plateau of the logistic curve (it is C-dependent); whether a
tuned C closes the gap (`logreg_cv` not swept; C fixed so the effective penalty weakens 27×
across the grid); anything in
standardised space; any comparison of these cosines to a permuted floor; "better classifier"
in the strict sense (single split, substituted decision rule); and the shape of the logistic
curve beyond n = 2000, which is where the pool ran out (halves cover 94% of it).

---

## A3 — present-vs-other dissociation, Qwen3.6-27B — **COMPLETE, 2026-09-04**

**Files:** `a3_dissociation_qwen36-27b.json` (driver output with per-sample `rows`, provenance-stamped, 11:42–12:13Z) · `a3_scalefree_qwen36-27b.json` (from `src/rev3/a3_scalefree.py`, deterministic) · cells in `inflight_box2/a3_cells_qwen36-27b.json`.

Reuses A2's pool (n = 4253; DIR 2126 / READ 2127, so steering directions and readout probes
come from disjoint halves). Three configurations: steering estimator → readout estimator =
logreg→logreg (the published configuration), dom→logreg, dom→dom. Doses {0, 0.5, 1.0},
2 reps × 29 scenarios = 58 per cell. Direction stability on this pool at n = 600/half:
dom 0.915 [0.911, 0.919], logreg 0.576 [0.567, 0.585].

### The exhibit, on three scales

`raw` is the driver's own comparison (projections onto unnormalised rows). `z0` divides each
channel by its own SD at α = 0; `zall` by its pooled SD over all doses. Contrast = present
slope − other slope, paired; * = CI excludes zero.

**logreg → logreg (published configuration):** raw: 0/6 present>other (0 sig), 4 sig other>present · z0: 0/6 present>other (0 sig), 3 sig other>present · zall: 1/6 present>other (0 sig), 2 sig other>present

| emotion | present (raw) | other (raw) | contrast raw | contrast z0 | contrast zall |
|---|---|---|---|---|---|
| desperate | +2.60 | +3.61 | -1.01 [-1.93, -0.08]* | -0.58 [-1.00, -0.14]* | -0.26 [-0.59, +0.08] |
| afraid | +3.66 | +3.94 | -0.28 [-1.57, +0.99] | -0.01 [-0.44, +0.43] | +0.06 [-0.31, +0.43] |
| happy | +1.77 | +5.00 | -3.23 [-4.41, -2.07]* | -1.41 [-1.95, -0.88]* | -0.97 [-1.36, -0.58]* |
| calm | +1.55 | +3.12 | -1.57 [-2.72, -0.45]* | -0.48 [-0.94, -0.02]* | -0.52 [-0.93, -0.10]* |
| sad | +3.74 | +4.51 | -0.77 [-1.88, +0.31] | -0.46 [-0.95, +0.01] | -0.17 [-0.52, +0.18] |
| angry | +2.56 | +3.84 | -1.28 [-2.20, -0.37]* | -0.30 [-0.87, +0.27] | -0.17 [-0.52, +0.18] |

**dom → logreg:** raw: 1/6 present>other (1 sig), 5 sig other>present · z0: 1/6 present>other (1 sig), 4 sig other>present · zall: 1/6 present>other (1 sig), 5 sig other>present

| emotion | present (raw) | other (raw) | contrast raw | contrast z0 | contrast zall |
|---|---|---|---|---|---|
| desperate | +1.57 | +3.78 | -2.20 [-3.23, -1.24]* | -1.12 [-1.58, -0.67]* | -0.64 [-1.02, -0.28]* |
| afraid | +6.04 | +4.56 | +1.48 [+0.31, +2.60]* | +0.63 [+0.24, +1.01]* | +0.38 [+0.08, +0.66]* |
| happy | +0.99 | +4.97 | -3.98 [-5.32, -2.66]* | -1.76 [-2.36, -1.16]* | -1.11 [-1.56, -0.66]* |
| calm | +0.76 | +1.97 | -1.21 [-2.35, -0.10]* | -0.39 [-0.85, +0.05] | -0.44 [-0.84, -0.05]* |
| sad | +2.98 | +6.05 | -3.07 [-4.20, -1.89]* | -1.48 [-1.98, -0.96]* | -0.53 [-0.85, -0.21]* |
| angry | +2.12 | +5.00 | -2.87 [-4.00, -1.72]* | -1.27 [-1.97, -0.55]* | -0.56 [-0.94, -0.17]* |

**dom → dom:** raw: 4/6 present>other (2 sig), 0 sig other>present · z0: 2/6 present>other (0 sig), 3 sig other>present · zall: 1/6 present>other (0 sig), 3 sig other>present

| emotion | present (raw) | other (raw) | contrast raw | contrast z0 | contrast zall |
|---|---|---|---|---|---|
| desperate | +76.22 | +59.71 | +16.52 [-6.96, +40.11] | -0.69 [-1.06, -0.31]* | -0.46 [-0.77, -0.15]* |
| afraid | +210.56 | +104.56 | +106.00 [+66.17, +144.42]* | +0.24 [-0.29, +0.79] | +0.24 [-0.11, +0.59] |
| happy | +48.39 | +95.03 | -46.64 [-96.25, +2.63] | -1.06 [-1.62, -0.51]* | -0.98 [-1.43, -0.53]* |
| calm | +34.50 | +17.68 | +16.81 [-43.31, +77.53] | -0.04 [-0.65, +0.55] | -0.09 [-0.55, +0.36] |
| sad | +87.78 | +88.91 | -1.13 [-36.26, +37.02] | -1.57 [-2.23, -0.92]* | -0.51 [-0.88, -0.13]* |
| angry | +215.70 | +106.47 | +109.23 [+67.91, +151.22]* | +0.29 [-0.25, +0.85] | -0.14 [-0.46, +0.19] |

**What the files support.** On this pool, at the focus layer, with a probe fit on a half the
steering directions never saw: under the published configuration B's *other*-speaker
readout rises at least as fast as its *present* readout for every emotion (0 of 6 the other
way; 4 of 6 significantly reversed on the raw scale). Under difference-of-means for both
steering and readout the raw count is 4 of 6 (afraid and angry significant), and that count
is the rows' norms: once each channel is on its own scale it is 2 of 6 with none significant
(z0) or 1 of 6 (zall), with three emotions significantly reversed. **The present-vs-other
dissociation the plan held out as Paper A's exhibit is not in this data on any scale.**

**What they do not support** (results/reports/18). Anything about *why* the other-speaker
readout moves more: the design has no random/orthogonal steering control, no cross-decode
figure, no readout on A's span, no context-ablated readout, and B's reply is pooled inside a
transcript that contains A's steered text, so "other rises more" is the expected signature of
attention to that text. "Norm artifact" is not established — row norms are not stored, only a
1.65–2.4× score-SD ratio. `zall` is a biased normaliser (pooled SD is inflated by the dose
response) and should be ignored; z0, Cohen's d, within-scenario SD and rank all agree. The
paired contrast is neither scenario-blocked nor dose-paired (counts unchanged when redone;
published-config reversals 3 of 6 after FDR). The six emotions share one α = 0 baseline, so
they are not independent votes. Stability is quoted at n = 600, not the 2126 used. Two reps;
generation seeded through `C.gen`, so reproducible in principle.

---

## A2 — estimator battery, Llama-3-8B-abliterated — **COMPLETE (7 of 7 depths), 2026-09-04**

**Files:** `a2_estimator_llama3-abl.json` (provenance-stamped: model rev `dd67dd05`, acl_core `23c758a9`, 07:58–13:24Z including two restarts, 18,793 s), `a2_cells_llama3-abl.json` (462 cells). Same driver, same patched code as the 27B run.

Config: k = 220 (matching the committed `stabdiag_llama3-abl-big` run), 7920 generated,
**3898 kept** (leak 2322 — 29%, vs 11% on the 27B — unparsed 1700); depths {8, 11, 14, 18, 21,
24, 27} of 32 (focus 21); n per half up to **1200** (2000 not reachable on this pool);
10 splits; raw space.

### Headline (focus layer 21, n = 1200/half)

| quantity | logreg | difference of means |
|---|---|---|
| disjoint split-half cosine | **0.286** [0.277, 0.295] | **0.941** [0.939, 0.943] |
| held-out present decode (one 70/30 split) | 0.803 | 0.731 |
| logreg ↔ dom direction cosine | 0.382 | |

### Split-half cosine at n = 1200/half, every depth

| hs | depth | logreg | logreg_c005 | ridge | dom | dom_norm | pca_diff | mass_mean_cov | lda_shrunk |
|---|---|---|---|---|---|---|---|---|---|
| 8 | 0.25 | 0.266 | 0.297 | 0.233 | 0.925 | 0.929 | 0.822 | 0.104 | 0.133 |
| 11 | 0.344 | 0.308 | 0.321 | 0.243 | 0.925 | 0.928 | 0.781 | 0.114 | 0.149 |
| 14 | 0.438 | 0.338 | 0.347 | 0.253 | 0.934 | 0.936 | 0.792 | 0.111 | 0.143 |
| 18 | 0.562 | 0.310 | 0.327 | 0.248 | 0.940 | 0.941 | 0.748 | 0.106 | 0.139 |
| 21 | 0.656 | 0.286 | 0.309 | 0.239 | 0.941 | 0.943 | 0.784 | 0.103 | 0.138 |
| 24 | 0.75 | 0.287 | 0.312 | 0.236 | 0.940 | 0.941 | 0.757 | 0.098 | 0.132 |
| 27 | 0.844 | 0.289 | 0.314 | 0.238 | 0.941 | 0.943 | 0.840 | 0.101 | 0.131 |

### Split-half vs n: logreg (left five) and dom (right five)

| hs | 75 | 150 | 300 | 600 | 1200 | 75 | 150 | 300 | 600 | 1200 |
|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 0.229 | 0.257 | 0.284 | 0.284 | 0.266 | 0.472 | 0.634 | 0.780 | 0.869 | 0.925 |
| 11 | 0.246 | 0.283 | 0.303 | 0.306 | 0.308 | 0.471 | 0.641 | 0.782 | 0.869 | 0.925 |
| 14 | 0.264 | 0.305 | 0.328 | 0.344 | 0.338 | 0.497 | 0.667 | 0.798 | 0.881 | 0.934 |
| 18 | 0.265 | 0.300 | 0.318 | 0.320 | 0.310 | 0.520 | 0.691 | 0.816 | 0.892 | 0.940 |
| 21 | 0.267 | 0.297 | 0.306 | 0.306 | 0.286 | 0.526 | 0.697 | 0.820 | 0.895 | 0.941 |
| 24 | 0.265 | 0.289 | 0.309 | 0.307 | 0.287 | 0.521 | 0.690 | 0.816 | 0.893 | 0.940 |
| 27 | 0.265 | 0.289 | 0.307 | 0.302 | 0.289 | 0.527 | 0.695 | 0.820 | 0.896 | 0.941 |

### Decode, cross-estimator, speaker geometry, diagnostics (n/d = 0.95 at every depth)

| hs | present logreg | present dom | other logreg | other dom | logreg~dom | present↔other cos (logreg / dom) | Fisher | eff. rank | class SNR |
|---|---|---|---|---|---|---|---|---|---|
| 8 | 0.771 | 0.684 | 0.571 | 0.379 | 0.363 | 0.031 / 0.328 | 0.060 | 126 | 1.44 |
| 11 | 0.799 | 0.718 | 0.628 | 0.451 | 0.403 | 0.027 / 0.305 | 0.069 | 190 | 1.51 |
| 14 | 0.827 | 0.724 | 0.615 | 0.469 | 0.418 | 0.031 / 0.221 | 0.088 | 231 | 1.55 |
| 18 | 0.811 | 0.724 | 0.580 | 0.439 | 0.390 | 0.027 / 0.291 | 0.100 | 235 | 1.56 |
| 21 | 0.803 | 0.731 | 0.579 | 0.426 | 0.382 | 0.025 / 0.379 | 0.106 | 240 | 1.57 |
| 24 | 0.816 | 0.725 | 0.573 | 0.405 | 0.383 | 0.028 / 0.426 | 0.105 | 237 | 1.55 |
| 27 | 0.813 | 0.724 | 0.582 | 0.408 | 0.380 | 0.027 / 0.417 | 0.108 | 238 | 1.55 |

**What the file supports** (corrected per results/reports/19). The C = 0.5 logistic row
reproduces at 0.27–0.34 on every depth; its curve peaks at n = 300–600 and **declines at
n = 1200 on six of seven depths, significantly** (paired over ten seeds: focus −0.020, 9/10
negative; pooled 10/10) — and the C = 0.05 row declines *harder* at all seven. Unlike the
27B, this is not a fixed-C artifact: on this model the logistic direction genuinely gets less
reproducible with more data. Difference-of-means reaches 0.925–0.942 and is still rising.
**The §2.2 ordering violation reproduces at matched n = 1200:** 8B logreg 0.286 vs 27B 0.568,
dom 0.941 vs 0.955, with the 8B holding the better n/d (0.95 vs 0.83) — n/d does not order
the two models. That is two models differing in d, k, retention (49% vs 74%), tokenizer and
abliteration; it is an ordering, not a mechanism, and the geometry candidate (effective rank
126–240 vs 230–327, class SNR 1.4–1.6 vs 1.7–1.8) is named, not tested — rank does not track
stability within the 8B. Under dom the present↔other cosine at the focus layer (0.379) is
0.89 of the within-present cosine (0.427): **the near-orthogonal-speaker result does not hold
under the stable estimator on this model** (logreg sits at the 1/(K−1) floor).

**What it does not support.** Everything the 27B section lists, plus: the tuned-C arm has not
run on the 8B; the leak filter removed 29% of generations here, so the pool's class balance
should be checked before any per-class claim.

---

## A2 follow-up — tuned C, fixed effective penalty, both spaces, like-for-like cross-estimator — **layers 16 and 43 complete; layer 54 lost to the lease**

**Files:** `a2_followup_qwen36-27b.json` (provenance-stamped, `code_sha` set, written after
layer 43 at 17:42Z) · `a2f_cells_qwen36-27b.json` (layer 54: 10 of 48 cells). Driver
`src/rev3/a2_followup.py`, written and independently verified 2026-09-04; features regenerated
from A2's saved pool (`a2_regen_feats.py`), so the pool is A2's (n = 4253).

Estimators: `logreg` (C = 0.5, as A2), `logreg_cv` (C tuned per fit), `logreg_lam`
(**C_n = 0.5 × 600 / n** — the per-sample penalty held at its n = 600 value for every n),
`dom`. Ten disjoint splits per cell; cosine in raw space (`C/sd`) **and** standardised space.

### Split-half cosine vs n per half — raw (left six) | standardised (right six)

| hs | estimator | 75 | 150 | 300 | 600 | 1200 | 2000 | 75 | 150 | 300 | 600 | 1200 | 2000 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 16 | logreg | 0.307 | 0.380 | 0.428 | 0.476 | 0.498 | 0.523 | 0.325 | 0.397 | 0.444 | 0.491 | 0.511 | 0.535 |
| 16 | logreg_cv | 0.328 | 0.417 | 0.448 | 0.484 | 0.504 | 0.539 | 0.347 | 0.436 | 0.463 | 0.499 | 0.517 | 0.550 |
| 16 | logreg_lam | 0.411 | 0.481 | 0.480 | 0.476 | 0.491 | 0.504 | 0.432 | 0.500 | 0.496 | 0.491 | 0.504 | 0.516 |
| 16 | dom | 0.497 | 0.680 | 0.803 | 0.891 | 0.942 | 0.966 | 0.518 | 0.697 | 0.814 | 0.898 | 0.946 | 0.968 |
| 43 | logreg | 0.392 | 0.471 | 0.525 | 0.575 | 0.568 | 0.566 | 0.420 | 0.496 | 0.548 | 0.597 | 0.589 | 0.588 |
| 43 | logreg_cv | 0.419 | 0.506 | 0.559 | 0.578 | 0.601 | 0.608 | 0.448 | 0.531 | 0.582 | 0.600 | 0.622 | 0.629 |
| 43 | logreg_lam | 0.496 | 0.581 | 0.596 | 0.575 | 0.578 | 0.581 | 0.525 | 0.606 | 0.619 | 0.597 | 0.599 | 0.603 |
| 43 | dom | 0.562 | 0.732 | 0.842 | 0.915 | 0.955 | 0.974 | 0.590 | 0.751 | 0.854 | 0.922 | 0.959 | 0.976 |

CIs at n = 2000, layer 43, raw: logreg [0.561, 0.572], logreg_cv [0.590, 0.625], logreg_lam [0.577, 0.585], dom [0.972, 0.975].

### Cross-estimator cosines at n = 2000 (10 independent subsamples)

| hs | CAA raw ↔ logreg raw | logreg raw ↔ dom raw | logreg std ↔ dom std | CAA raw ↔ dom raw | CAA raw ↔ dom std |
|---|---|---|---|---|---|
| 16 | +0.498 [+0.494, +0.503] | +0.549 [+0.544, +0.554] | +0.556 [+0.551, +0.561] | +0.926 [+0.925, +0.926] | +0.970 [+0.969, +0.970] |
| 43 | +0.427 [+0.421, +0.431] | +0.603 [+0.596, +0.609] | +0.616 [+0.609, +0.623] | +0.722 [+0.721, +0.723] | +0.869 [+0.868, +0.869] |

Ledoit–Wolf λ per n: layer 16 {'75': 0.65, '150': 0.431, '300': 0.273, '600': 0.163, '1200': 0.091, '2000': 0.056}; layer 43 {'75': 0.637, '150': 0.444, '300': 0.281, '600': 0.166, '1200': 0.093, '2000': 0.057}.

**What the file supports** (wording per results/reports/21).
- With the per-sample penalty held at its n = 600 value, the logistic row **moves by less
  than 0.03 over n = 150 → 2000** at both depths (not flat: it dips 300 → 600 at layer 43,
  t −4.6, and rises at layer 16, t +3.8) while difference-of-means on identical splits rises
  0.73 → 0.97. No material convergence over the reachable range. Because the fixed penalty
  fixes the population target, the cosine must approach 1 eventually: this is a
  proportional-regime observation (n ≤ 2000, d = 5120), not an asymptotic one, and the
  plateau's level depends on the single anchor run.
- Cross-validated C moves the logistic row by +0.042 (0.608), about a tenth of the gap — but
  CV scored accuracy, not stability, on a decade-spaced grid; C = 0.005 fixed was never run.
- Standardised-space values are always higher than raw, never by more than 0.03, same shape.
- The CAA-raw ↔ raw-logistic cosine is 0.43 at the focus layer, 0.50 at layer 16, vs
  0.98–0.99 in 2604.08169 — closer to that paper's quantity than A2's 0.59 but still a
  standardised fit's image and a 6-way row, and the 10-draw CI has no sampling content.
- Ledoit–Wolf λ falls 0.65 → 0.06 with n at both depths (one seed), confirming the mechanism
  behind the covariance estimators' decline.

**What it does not support.** Anything at layer 54 (lost); non-convergence *in the limit*;
anything about *validity* of the dom direction; generalisation beyond one pool; the 8B has
not had this arm. The file records no model or revision (it reads features) and no
partial-run flag.

---

## B1c — pre-registered all-layer ablation — **COMPLETE, 2026-09-04 22:05Z; verdict H1**

Pre-registration `docs/planning/PREREG_B1c.md` (122af4c 19:02Z, addendum cc2dc32 19:22Z; run start 19:32Z — the document was on the box at 19:32:17Z, sha identical to cc2dc32, but the driver's own prereg stamp is empty because its relative lookup missed it; see report 22). File `b1c_alllayer_qwen36-27b.json` (432 arm-rows, 29 scenarios, 6 arms × 4 doses × 3 reps × 6 emotions), checkpoint `b1c_cells_qwen36-27b.json`. Same probe pool and directions as B1 (sha 70129f86…); B sampled with common random numbers across arms. Gate ≥ 0.80 at all 51 ablated hidden states (0.882–0.927). Registry `b1c.*`.

**Decision (PREREG §5): H1** — median blocked fraction of `emo_all − rand_all` over the five testable emotions 0.092, scenario-bootstrap CI [0.038, 0.254] (H1 needs median ≤ 0.50 and upper ≤ 0.65). MC-steer 6/6, MC-ablate failures 0, no quality exclusions.

| emotion | none | emo13_42 | emo_all | rand_all | emo_all − rand_all (blocked, dose-paired) | blocked | emo_all − emo13_42 | A-span readout top dose none → emo_all (floor) |
|---|---|---|---|---|---|---|---|---|
| desperate | +84 [+59, +111] | +82 | +77 | +82 | −5.0 [−22.5, +12.1] | 0.06 | −4.1 [−14.3, +3.9] | 0.955 → 0.819 (0.108) |
| afraid | +212 [+169, +253] | +154 | +160 | +205 | **−44.9 [−58.0, −31.8]** | 0.21 | +5.8 [−2.4, +14.1] | 0.397 → 0.281 (0.250) |
| happy | +48 [+8, +88] | +53 | +59 | +61 | −2.3 [−19.0, +13.8] | 0.05 | +6.2 [−2.6, +15.8] | 0.744 → 0.551 (0.287) |
| calm · untestable | +44 [−3, +99] | +9 | +1 | +43 | −41.6 [−74.9, −10.7] | (0.94) | −8.2 [−17.9, +1.4] | 0.764 → 0.632 (0.277) |
| sad | +98 [+65, +132] | +65 | +70 | +101 | **−31.7 [−59.4, −6.1]** | 0.33 | +4.4 [−7.8, +15.3] | 0.545 → 0.165 (0.039) |
| angry | +196 [+155, +235] | +173 | +178 | +197 | −18.1 [−38.5, +1.3] | 0.09 | +5.7 [−1.8, +13.7] | 0.598 → 0.382 (0.038) |

- Layers 43–63 add nothing: `emo_all − emo13_42` includes zero for all six emotions.
- `emo13_42 − rand_all` significant for afraid, sad and angry (−23.8 [−44.9, −3.4]); BH-FDR q (emo_all) afraid 0.001, sad 0.048, angry 0.115.
- Random control (descriptive, no pre-registered contrast): `rand_all` within 5% of `none` for five emotions, +28% for happy (CIs overlapping); it removes ~2× the residual norm (1.15–1.26 vs 0.35–0.66) because it is unit-norm-matched, not footprint-matched.
- Instrument caveat (critique): blocked fraction tracks the share of readable affect the ablation removes (r ≈ 0.85, five points; blocked-per-removed median 0.27); calm's contrast −41.6 [−74.9, −10.7] is significant but excluded by the testability filter; afraid's top-dose readout is non-monotone and its MC `separated` flag is false.
- Secondary readout on B's reply: `emo_all − rand_all` significant for sad only (−0.09 [−0.16, −0.03]).
- Rewrite arms uninterpretable 6/6 (neutral rewrite keeps 67–98% of A's readable affect; `text_keep − none` significant for desperate +31, happy −51, calm −66, sad −69; afraid loses 26%/20% of rewrites to refusals).
- Reading: the rank-1 `dom` direction is insufficient to carry the transfer at any layer 13–63; what carries the rest is untested (PREREG §7). Report and blind critique: `results/reports/22_rev3_b1c_alllayer.md`.

## Provenance gaps introduced by these files (to close, not to hide)

1. ~~B1 stability gate 0.909 — no committed artifact~~ — closed by the rerun's `b1_e4rerun_qwen36-27b.json` (0.907; a new measurement).
2. RESULTS_LOG §B1.3 desperate figures are from the 7-cell state; superseded by the
   47-cell numbers here and in the regenerated `b1_partial_summary.json`.
3. ~~`docs/review/claims.json` has no entries for rev-3 numbers~~ — closed 2026-09-04:
   entries `b1.stability_gate_dom` (category (b), no file), `b1.mc_ablate_pass_5of5`,
   `b1.emo_vs_rand_sig_2of5`, `a2.l16_dom_2000`, `a2.l16_logreg_2000` (all `pending`,
   partial grids) now resolve under `tools/check_provenance.py`.
4. ~~A2 is one layer of seven and not the focus layer~~ — closed 2026-09-04: all seven depths in `a2_estimator_qwen36-27b.json`.
5. The `emo − X` contrast intervals are paired but not scenario-blocked (see B1.3).

## Reruns launched 2026-09-04

Both jobs were relaunched from scratch on fresh sandboxes (the old ones expired with their
work directories, including the probe pool that fixed B1's directions, so the 47-cell
checkpoint above cannot be consistently resumed — `acl_core.Checkpoint` keys on pool size
and refuses to merge). In-progress pulls land in `inflight_box1/` (B1 → A2 on
Llama-3-8B-abliterated) and `inflight_box2/` (A2 → A3 on Qwen3.6-27B); finished outputs
are promoted into this folder when they carry a provenance stamp.

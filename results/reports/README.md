# results/reports — per-experiment reports with blind critiques

One file per experiment group. Each file has two authors with different information:

- **Report** (`## What was run`, `## What the file shows`, `## What can be inferred`,
  `## Status`) — written from the script, the result file, and the repo's existing audit
  and claims registry. Every inference bullet is tagged `[supported by file]` or
  `[needs: …]`.
- **Independent critique** — written by a separate agent that was given only the claim
  wording, the script(s), and the result JSON(s). It was blocked from README, `docs/`,
  other results, and git history, and told it had no knowledge of any prior audit. Its
  verdicts are its own; where they disagree with `docs/review/AUDIT.md` the report says so
  in a short `## Reconciliation` note rather than editing either side.

Verdict vocabulary used by the critiques: SUPPORTED · SUPPORTED-WITH-CAVEATS ·
NOT SUPPORTED · UNDETERMINABLE.

Built 2026-09-04. Index below is filled in as each group completes.

| # | group | files | critique verdicts | agrees with AUDIT? |
|---|---|---|---|---|
| 01 | E0 present/other gate | `core/e0_qwen36-27b.json` | C1 SUPPORTED-WITH-CAVEATS · C2 NOT SUPPORTED as worded | AUDIT silent on E0; critique adds class imbalance (chance 0.190), no lexical baseline, no null for "near-orthogonal" |
| 02 | E2-CI contagion + paraphrase | `representation/e2ci_qwen36-27b.json` | C1 SWC · C2 NOT SUPPORTED · C3 SWC | agrees with AUDIT 3/18; adds endpoint-slope, pooled paraphrase check, five significant reversals |
| 03 | E3 read-time ablation | `representation/e3v2_qwen36-27b.json` | C1 SWC (arithmetic) / NS (affect) · C2 NOT SUPPORTED · C3 SWC | agrees with AUDIT 9 and plan §1.1; adds intercept shifts, no ceiling arm, truncation fallback |
| 04 | E4 gen-time ablation (original) | `representation/e4gentime_qwen36-27b.json` | C1 NOT SUPPORTED · C2 UNDETERMINABLE | agrees with plan §1.3; adds that the point estimates show *partial* blocking and non-blocking is expected by construction |
| 05 | E5 activation passing | `representation/e5actpass_*.json` | C1 NOT SUPPORTED · C2 NOT SUPPORTED | agrees with AUDIT 1/12; adds per-run decoder refit, none-baseline drift, joint criterion ≤ 3/6 |
| 06 | CMI coupling estimate | `information/cmi_passed_qwen36-27b.json` | NOT SUPPORTED | agrees with AUDIT 2; adds `Z=None` (unconditional MI), scenario common cause, 20 permutations |
| 07 | CMI pilot (synthetic) | `information/cmi_pilot.json` | C1 SWC · C2 NOT SUPPORTED · C3 SWC | no prior audit of content; analytic reproduction; encoder-sweep and decision-rule gaps |
| 08 | Jacobian-lens verbalization | `verbalization/*.json` | C1 NOT SUPPORTED · C2 NOT SUPPORTED · C3 SWC | agrees with AUDIT 10/11/17; adds missing `log_softmax`, tie-break layer, identity control, bf16 |
| 09 | Behavioral powered | `behavioral/behavioral_powered_*.json` | C1 NOT SUPPORTED as worded · C2 NOT SUPPORTED | agrees with AUDIT 6; adds letter-probability outcome, `Cp/sd` double standardisation, pooled positive trend |
| 10 | Behavioral induction | `behavioral/behavioral_induction_*.json` | C1 NOT SUPPORTED · C2 NOT SUPPORTED | agrees with AUDIT 7; adds injection artifact floor, identical scramble items |
| 11 | Multi-turn accumulation | `behavioral/behavioral_multiturn_qwen36-27b.json` | all three NOT SUPPORTED | agrees with AUDIT 8; adds unperturbed neutral arm, 1024-token fallback, refusal register |
| 12 | Behavioral pilots | `behavioral/behavioral_{coupling,verify}_*.json` | C1 NOT SUPPORTED · C2 NOT SUPPORTED | agrees with AUDIT 7 on the 0.0; explains it mechanically; pilots contradict each other 116× |
| 13 | Scenario screen, P1 abort, pre-fix 8B direction | `behavioral/scenscreen_*`, `p1dose_ABORTED_*`, `estimator/p1dose_stability_*` | C1 NOT SUPPORTED as glossed · C2 SWC · C3 UNDETERMINABLE | AUDIT 5 only; "already cheats unprompted" is not established; stability file lacks estimator |
| 14 | Split-half stability diagnostic | `estimator/stabdiag_*.json` | C1 SWC · C2 SWC · C3 NOT SUPPORTED | **not in AUDIT**; one split with nested subsamples; fixed C confounds n; "does not improve" contradicted |
| 15 | Dual-estimator battery | `estimator/dualest_qwen36-27b.json` | C1 SWC · C2 SWC · C3 SWC / UNDETERMINABLE · C4 NOT SUPPORTED | AUDIT 4 only; present>other under dom is 0/6 after scale normalisation; logreg count 2/6 not 1/6 |
| 16 | Rev-3 B1 partial (47/54) → completed rerun (54/54, verdict `not_blocking`; delta critique in) | `rev3/b1_*` | after delta: C1 SWC · C2 SWC · C3 NOT SUPPORTED as inference · C4 NOT SUPPORTED as worded | no prior audit; gate at an un-ablated layer, n = 600 not 789, unseeded B generation, denominator 6 |
| 17 | Rev-3 A2 partial (layer 16) → completed run (7 depths; delta in) | `rev3/a2_*` | after delta: C1 SWC (weakened) · C2 NOT SUPPORTED as worded · C3 UNDETERMINABLE | no prior audit; `pair_on` bug (patched), covariance decline is estimator behaviour, nested n-points, dual implementation never run in-regime |

| 18 | Rev-3 A3 dissociation (complete) + scale-free re-read | `rev3/a3_*` | C1 SWC · C2 SWC · C3 SUPPORTED as stated, not relevant as used | no prior audit; exhibit collapse robust to four normalisations; unblocked/unpaired contrast, shared α=0 baseline, no B-models-A control |
| 19 | Rev-3 A2 on Llama-3-8B-abl (complete) | `rev3/a2_*_llama3-abl.json` | C1 SWC · C2 SWC (ordering) / NOT SUPPORTED (mechanism) | no prior audit; logistic decline is real on the 8B and survives stronger C — models disagree on mechanism; speaker orthogonality fails under dom on the 8B |
| 20 | Rev-3 B1 follow-up (ceiling + text arms, seeded, per-layer gate) | `rev3/b1_followup_*`, `b1f_cells_*` | C1 SUPPORTED · C2 SWC · C3 NOT SUPPORTED as inference · C4 NOT SUPPORTED as worded · C5 NOT SUPPORTED as worded | no prior audit; counts flip under clustering; text arm refusal-contaminated; ceiling extends positions not layers |
| 21 | Rev-3 A2 follow-up (tuned C, fixed λ, both spaces, CAA-raw cross) — layers 16, 43 | `rev3/a2_followup_*`, `a2f_cells_*` | C1 NOT SUPPORTED as worded / SUPPORTED in substance · C2 SWC · C3 NOT SUPPORTED as worded · C4 SWC | no prior audit; "flat" withdrawn, proportional-regime caveat, CV scored accuracy, CAA still not the published quantity |
| 22 | Rev-3 B1c pre-registered all-layer ablation (hs 13–63) + rewrite arms — verdict H1 | `rev3/b1c_alllayer_*`, `b1c_cells_*` | C1 SUPPORTED (H1, median 0.09 [0.04, 0.25]) · C2 SUPPORTED for 4 emotions, unpowered for desperate/calm, driver-declared not pre-registered · C3 SUPPORTED (sampling replication) · C4 NOT SUPPORTED as worded (happy random arm +28%; no contrast) → descriptive · C5 SUPPORTED, rewritten around r ≈ 0.85 and blocked-per-removed 0.27 · C6 SUPPORTED · C7 provenance gap (prereg stamp empty; IST/UTC) |
| 23 | Rev-3 A2 follow-up, second fixed-penalty anchor (N_REF = 2000) — layer 43; layer 54 partial | `rev3/a2_followup_qwen36-27b-nref2000.json`, `a2f_cells_*-nref2000.json` | C1 SUPPORTED as 'no net rise', NOT as 'flat' (significant dip and recovery) · C2 SUPPORTED, range/counts corrected at n = 75 and 1200 · C3 descriptive, anchor-blind · C4 defects confirmed + selftest witness, checkpoint prose |
| 24 | Rev-3 A2 follow-up on the 8B — fixed per-sample penalty, tuned C, cross-estimator (layers 14, 21, 27) | `rev3/a2_followup_llama3-abl.json`, `a2f_cells_llama3-abl.json` | C1 SUPPORTED with 'from n = 150' (peak at 150) · C2 SUPPORTED (schedule ruled out, not n/d) · C3 numbers right, n corrected to 2000 with overlap caveat · C4 SUPPORTED at matched n · defects: cross_n default, synthetic equivalence check, empty model fields |
| 25 | Rev-3 A2 follow-up, 27B layer 54 at anchors 600 and 2000 | `rev3/a2_followup_qwen36-27b-l54*.json` | C1, C2, C4 VERIFIED to the digit · C3 REJECTED (one significant drop then flat, not the 8B's shape; layer 16 rises, so 'no net rise at every depth' was false) → corrected · three driver builds across the layer comparison |
| 26 | Rev-3 A2 margin diagnostic — separability, penalty sensitivity, divergence from difference-of-means (27B L43/L54, 8B L21) | `rev3/a2_margin_*.json` | C1 SUPPORTED (bounds corrected to extremes) · C2 HALF: likelihood-takeover refuted, max-margin limit NOT shown · C3 SUPPORTED as an upper bound · C4 numbers SUPPORTED, mechanism a story · C5 SUPPORTED (1e-15 on real data) · defects: 5 seeds/no CI, one weak C, missing n_cls assert |
| 27 | Rev-3 B1d pre-registered rank-5 subspace ablation (none / emo_all / sub_all / perm_all / randsub_all) — instrument_failed | `rev3/b1d_subspace_*`, `b1d_frame_geometry.json` | C1–C6 VERIFIED (gate max corrected to 0.931) · C2 'significant' withdrawn (BH-FDR q 0.051 each; scenario-only CI) · C7 CORRECTED (fitted, not footprint, is the confound; B1e runs the right control) · geometry added post hoc · verdict robust to own-floor shares, attribution not |
| 28 | Rev-3 B1e pre-registered fitted label-free rank-1 control (none / emo_all / permdir_all / pc1_all) — H3 | `rev3/b1e_footprint_*` | numbers VERIFIED incl. bit-identity (two CI/shift statements corrected) · rule FAITHFUL · control NOT a strong null (removes less than B1c's inert random direction; cosine unstamped) → footprint caveat stands · stats: point-rule vs sad's range, afraid's secondary contradicts · C5 (B1d retrospective) WITHDRAWN |

Abbreviations: SWC = SUPPORTED-WITH-CAVEATS. Every verdict is the critic's, quoted from the
report; the "agrees with AUDIT?" column is the orchestrator's comparison.

**Where this feeds back:** RESEARCH_PLAN §12 maps each finding to plan coverage and lists the
open to-dos; in-place plan corrections are marked ⟨2026-09-04⟩. Registry entries were not
changed by the critiques — adjudication of wording changes (e.g. `e0.cross_cos`) is the
authors' call.

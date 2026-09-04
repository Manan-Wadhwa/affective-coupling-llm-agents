# results/ — index and status of every committed result file

One row per file. The **label** column is not a judgement made here: it is copied from
`docs/review/claims.json` (`claim_status`) and `docs/review/AUDIT.md` (finding numbers), or
from a status field inside the file itself. Run `python tools/check_provenance.py` to
re-verify the pointers; nothing in this index re-adjudicates a claim.

Labels:

| label | meaning |
|---|---|
| **QUOTABLE** | at least one `claims.json` entry pointing at this file has `claim_status: quotable`, and the checker resolves it |
| **RETRACTED-CLAIM** | the file resolves and its numbers are real, but every sentence built on it in `claims.json` is `retracted` — the *interpretation* is withdrawn, not the file |
| **PENDING** | a `claims.json` entry is awaiting a re-run before it can be quoted either way |
| **UNREGISTERED** | no `claims.json` entry points at this file; its numbers are not cleared for quoting and have not been audited by the checker |
| **SUPERSEDED** | kept for the record; a later script or a rev-3 rerun replaces it |
| **ABORTED / PARTIAL / CHECKPOINT** | the file says so itself |
| **RAW** | generations, not a result |

Last built 2026-09-04 against commit `70894ba`. Per-experiment reports with blind critiques: [`reports/`](reports/README.md). Producing-script and parameter columns
were read from each file's own metadata and its script's argparse/output lines; "not
recorded" means the file carries no such field.

---

## core/

| file | script | model | n / params | contains | label | registry / audit |
|---|---|---|---|---|---|---|
| `e0_qwen36-27b.json` | `src/core/coupling_e0.py` | Qwen/Qwen3.6-27B | n=395, focus=43, layers 22/32/43/51 | per-layer present/other decode acc, LOO, cross-decode leak, present↔other cos, within-present cos, confusion; `focus_result` | **QUOTABLE** | `e0.present_acc` (0.899), `e0.cross_cos` (0.044) |

## representation/

| file | script | model | n / params | contains | label | registry / audit |
|---|---|---|---|---|---|---|
| `e2ci_qwen36-27b.json` | `src/representation/coupling_e2_ci.py` | Qwen3.6-27B | α∈{0,0.5,1.0}, repeats=3, n_per_cell=87, decoder_n=272, focus=43 | per-emotion present/other/paraphrase slopes + CI, `raw_scores` | **PENDING** (`e2ci.contagion_5of6`) and **RETRACTED-CLAIM** (`e2ci.present_gt_other`: holds for 1 of 6 in this file, claim said 6) | AUDIT 3, 4, 16, 18 |
| `e3v2_qwen36-27b.json` | `src/representation/coupling_e3_ablate.py` | Qwen3.6-27B | decoder_n=261, n_abl_layers=30, focus=43 | per-emotion A/B slopes under none/emo/rand, `A_manip_removed`, `B_contagion_lost`, 18 rows | **RETRACTED-CLAIM** — B never regenerates under ablation (plan §1.1); arithmetic resolves, "channel cannot be severed" does not follow | `e3.affect_removed`, `e3.text_channel_intact`; AUDIT 9, 18 |
| `e4gentime_qwen36-27b.json` | `src/representation/coupling_e4_gentime.py` | Qwen3.6-27B | decoder_n=266, n_abl_layers=30, focus=43 | same field set as E3; no A-affect manipulation check in file | **UNREGISTERED · SUPERSEDED** by rev-3 B1 (plan §1.3: direction split-half 0.394, judged uninterpretable, not null) | AUDIT 9, 18 |
| `e5actpass_qwen36-b02.json` | `src/representation/coupling_e5_actpass.py --beta 0.2` | Qwen3.6-27B | β=0.2, decoder_n=269 | per-emotion none/full/ablated/scramble means, transmit / emo_specific / vs_scramble | **UNREGISTERED** — one of three β values swept; β chosen on outcome | AUDIT 1, 18 |
| `e5actpass_qwen36-b03.json` | same, `--beta 0.3` | Qwen3.6-27B | β=0.3, decoder_n=266 | same | **RETRACTED-CLAIM** — the 6/6 file; scramble permutes within one emotion (specificity untestable); `emo_specific` negative for desperate and calm | `e5.beta03_all6`, `e5.emotion_causally_responsible` (claim said 6, file gives 3); AUDIT 1, 12, 15, 18 |
| `e5actpass_qwen36-27b.json` | same, `--beta 0.5` | Qwen3.6-27B | β=0.5, decoder_n=257 | same | **UNREGISTERED** — one of three β values | AUDIT 1, 12, 15, 18 |

## information/

| file | script | model | n / params | contains | label | registry / audit |
|---|---|---|---|---|---|---|
| `cmi_passed_qwen36-27b.json` | `src/information/cmi_passed.py` (hardcoded path) | Qwen3.6-27B | β=0.3, n=174 | `cmi_inject_nats` 0.264, scramble null 0.067 ± 0.017, `coupling_nats` 0.197, z 11.5 | **RETRACTED-CLAIM** — pools six emotions and permutes across emotion blocks; statistic mostly recovers emotion identity | `cmi.coupling_nats`; AUDIT 2, 18 |

| `cmi_pilot.json` | `src/information/cmi_pilot.py` (CPU-only synthetic linear-Gaussian system; default `--out` is this path) | none (synthetic) | N_est=2000, N_truth=400k, seeds 0–11 | beta sweep (truth / full-conditioner / lossy-conditioner / shuffled null), the beta=0 confound (`confound.beta0_est_lossy_ARTIFACT` 0.8815), TE hygiene, sample-size, nonlinear | **PENDING** — reproduced locally 2026-09-04 (6 s, byte-identical across two runs); was category (c) because no run had ever been committed. Promotion to quotable is an adjudication call, see claims.json note | `cmi.pilot_spurious`; AUDIT 2 |

## verbalization/

| file | script | model | n / params | contains | label | registry / audit |
|---|---|---|---|---|---|---|
| `jlens_contagion_qwen36-27b.json` | `src/verbalization/jlens_contagion.py` (hardcoded) | Qwen3.6-27B | layer=29; n not recorded (AUDIT: 16) | per-emotion jlens/logit readouts at α=0, 1 and their contagion deltas | **RETRACTED-CLAIM** — unnormalised scales, no CIs, layer chosen on the lens's own output | `jlens.beats_logit_lens`; AUDIT 10, 18 |
| `jlens_coupling_qwen36-27b.json` | `src/verbalization/jlens_coupling.py` (hardcoded) | Qwen3.6-27B | layer=29 | per-emotion inject/none readouts, jlens vs logit coupling | **UNREGISTERED · SUPERSEDED** by `jlens_contagion`; its input `results/generations/responses_qwen36.jsonl` is not in the repo | AUDIT 10, 11, 17, 18 |
| `jlens_probe.json` | `src/verbalization/jlens_probe.py` (hardcoded) | not recorded (script hardcodes Qwen3.6-27B) | mid_layer=31; 4 prompts | top J-lens vs logit-lens tokens per prompt | **UNREGISTERED** | AUDIT 10 |
| `jlens_readout.json` | `src/verbalization/jlens_readout.py` (hardcoded) | not recorded (script hardcodes Qwen3.6-27B) | 8 prompts; layer sweep at 0.45–0.75 depth | `best{jlens,logit,layer}` + `by_layer` per prompt | **UNREGISTERED** | AUDIT 10 |

## behavioral/

| file | script | model | n / params | contains | label | registry / audit |
|---|---|---|---|---|---|---|
| `behavioral_powered_qwen36-27b.json` | `src/behavioral/behavioral_powered.py` | Qwen3.6-27B | reps=4, n_per_cond=32 | cheat / B-desperate means by condition, manip and behavioral contrasts + CI; `manip_sig` False, `behavioral_sig` False | **RETRACTED-CLAIM** — CI at n=32 rules nothing out; no MDE | `behavioral.honesty_floor_ruled_out`; AUDIT 6, 8 |
| `behavioral_powered_llama3-abl.json` | same | Llama-3-8B-Instruct-abliterated | reps=4, n_per_cond=32 | same; `manip_sig` True, `behavioral_sig` False; behavioral CI [−0.080, +0.131] | **UNREGISTERED** — this is the CI AUDIT 6 quotes | AUDIT 6 |
| `behavioral_powered_llama3-inst.json` | same | Llama-3-8B-Instruct | reps=4, n_per_cond=32 | same; both `_sig` False | **UNREGISTERED** | — |
| `behavioral_induction_qwen36-27b.json` | `src/behavioral/behavioral_induction.py` | Qwen3.6-27B | β=0.3, n_per_cond=32 | means for desperate/calm/scramble/none, manip and behavioral contrasts + CI | **RETRACTED-CLAIM** — scramble permutes within one emotion, so ≈0 by construction | `induction.scramble_control` (claim said True, file gives False); AUDIT 7 |
| `behavioral_induction_llama3-abl.json` | same | Llama-3-8B-Instruct-abliterated | β=0.3, n_per_cond=32 | same; `manip_sig` False | **UNREGISTERED** — same broken scramble | AUDIT 7 |
| `behavioral_induction_llama3-inst.json` | same | Llama-3-8B-Instruct | β=0.3, n_per_cond=32 | same; `manip_sig` False | **UNREGISTERED** — same broken scramble | AUDIT 7 |
| `behavioral_multiturn_qwen36-27b.json` | `src/behavioral/behavioral_multiturn.py` | Qwen3.6-27B | turns=8, reps=4, n_per_turn=32 | per-turn means, per-condition cheat/B-desperate slopes + CI, desp−calm accumulation contrast (+0.008 [−0.011, +0.024], n.s.) | **RETRACTED-CLAIM** — neutral arm accumulates too; calm's slope exceeds desperate's; cheat 0.085 desperate vs 0.234 calm, opposite in sign to `behavioral_powered_qwen36-27b.json` | `behavioral.multiturn_accumulation`; AUDIT 8 |
| `behavioral_coupling_qwen36-27b.json` | `src/behavioral/behavioral_coupling.py` (hardcoded) | Qwen3.6-27B | n not recorded | cheat_prob by condition; desp−calm −0.001, desp−neutral +0.038 | **UNREGISTERED · SUPERSEDED** (README: early pilot, kept for the record) | — |
| `behavioral_verify_qwen36-27b.json` | `src/behavioral/behavioral_verify.py` (hardcoded) | Qwen3.6-27B | n not recorded | two-arm pilot; `arm2_manip_desperate_shift` = 0.0 exactly | **UNREGISTERED · SUPERSEDED** | AUDIT 7 |
| `scenscreen_llama3-abl.json` | `src/behavioral/scen_screen.py` | Llama-3-8B-Instruct-abliterated | reps=8, 28 scenarios | per-scenario baseline cheat, non-response, band; `n_usable` 16, `mean_baseline` 0.404 | **UNREGISTERED** — cited in RESEARCH_PLAN §3.2 B6 (16 usable / 6 ceiling / 6 floor); numbers match the file | — |
| `p1dose_ABORTED_llama3-abl.json` | `src/behavioral/behavioral_p1_dose.py` | not recorded | threshold 0.2 | `reason` "INSTRUCT ceiling check failed"; baseline 0.510, INSTRUCT 0.450, THREAT 0.540 | **ABORTED** (own `reason` field) — positive control failed; cited in plan §3.2 B6 | — |

## estimator/

| file | script | model | n / params | contains | label | registry / audit |
|---|---|---|---|---|---|---|
| `stabdiag_qwen36-27b.json` | `src/estimator/stability_diag.py --k 70` | Qwen3.6-27B | focus=43, seed=0, pool 1853, halves 926/927, d=5120 | split-half cos for logreg and diff-of-means at n/half 150/300/600: logreg 0.300/0.348/0.409, dom 0.687/0.808/0.893 | **QUOTABLE** | `stabdiag.logreg_27b_600`, `stabdiag.dom_27b_600` |
| `stabdiag_llama3-abl-big.json` | same, `--k 220` | Llama-3-8B-Instruct-abliterated | focus=21, seed=0, pool 3927, halves 1963/1964, d=4096 | n/half 150/300/600/1200: logreg 0.126/0.183/0.190/0.219, dom 0.425/0.694/0.822/0.898 | **QUOTABLE** | `stabdiag.dom_8b_1200` |
| `stabdiag_llama3-abl.json` | same, `--k 70` | Llama-3-8B-Instruct-abliterated | focus=21, seed=0, pool 1180, halves 590/590 | same field set, smaller pool | **UNREGISTERED** — the `-big` run is the registered one | — |
| `dualest_qwen36-27b.json` | `src/estimator/dual_estimator_battery.py --k 80` | Qwen3.6-27B | focus=43, seed=0, pool 2170; α∈{0,0.5,1,**2**} | E0 battery and contagion slopes under `logreg` and `dom` on identical activations/folds | **QUOTABLE** for the dissociation (decode 0.926 vs 0.850; stability 0.394 vs 0.940). Its contagion table includes α=2, which `coupling_e2_ci.py` excludes by docstring; the happy α=2 collapse (3.55 → 0.44) is in **this** file, not `e2ci` | `dualest.stability_dissociation`; AUDIT 4 |
| `p1dose_stability_llama3-abl.json` | `src/behavioral/behavioral_p1_dose.py` | Llama-3-8B-Instruct-abliterated | seed=0, focus=21, k=10, n1=176, n2=164 | `cos_v1_v2` 0.195, `stable` False | **UNREGISTERED** — the committed *pre-fix* 8B gate. The post-fix 0.965 / 0.967 figures have **no file** (claims `stability.postfix_gate_8b/_27b`, category (b), PENDING) | AUDIT 5 |

## generations/

| file | script | lines | record keys | label | registry / audit |
|---|---|---|---|---|---|
| `responses_qwen36_sample300.jsonl` | `src/core/coupling_e0.py` via `$RESP_LOG` | 300 | `tag, prompt, output` (all `e0_dialogue`) | **RAW** sample | AUDIT 14 — the "≈13k archived generations" claim is retracted (`dataset.13k_released`, category (c)); the repo holds these 300 |
| `responses_multiturn_qwen36_sample200.jsonl` | `src/behavioral/behavioral_multiturn.py` via `$RESP_LOG` | 200 | `cond, rep, turn, A, B, conv` | **RAW** sample | AUDIT 14 — and these 200 |

## rev3/ — the rebuild (see `rev3/README.md` for every table)

| file | label | one line |
|---|---|---|
| `b1_e4rerun_qwen36-27b.json` | **RESULT · COMPLETE** | E4 rerun; gate 0.907; rule → `not_blocking` (sign-aware re-read → `mixed`) |
| `b1_followup_qwen36-27b.json` | **RESULT · COMPLETE** | ceiling + text arms, seeded B, all ablated layers gated; `mixed`; emo−rand blocks 25–37% for two (three blocked-bootstrap) |
| `b1c_alllayer_qwen36-27b.json` | **RESULT · COMPLETE · PRE-REGISTERED** | all-layer ablation (hs 13–63) + text/text_keep arms; verdict **H1**, median blocked 0.09 [0.04, 0.25]; layers 43–63 add nothing; rewrite arms uninterpretable 6/6 |
| `b1c_cells_qwen36-27b.json` | RAW · checkpoint | per-arm-row checkpoint of the above |
| `b1_cells_*`, `b1f_cells_*`, `b1_summary_*`, `b1_partial_summary.json` | CHECKPOINT / DERIVED / SUPERSEDED | per-cell arrays and analyzer outputs |
| `a2_estimator_qwen36-27b.json` | **RESULT · COMPLETE** | 7 depths; focus dom 0.974 vs logreg 0.566 |
| `a2_estimator_llama3-abl.json` | **RESULT · COMPLETE** | 7 depths; focus dom 0.941 vs logreg 0.286; n/d ordering reproduces |
| `a2_followup_qwen36-27b.json` | **RESULT · 2 of 3 layers** | tuned C, fixed penalty (< 0.03 movement over n), both spaces, CAA-raw cross 0.43 |
| `a2_cells_*`, `a2f_cells_*` | CHECKPOINT | |
| `a3_dissociation_qwen36-27b.json` | **RESULT · COMPLETE** | present > other: published config 0/6; dom→dom 4/6 raw |
| `a3_scalefree_qwen36-27b.json` | **DERIVED** | the same on per-channel scales: 1–2/6, 0 significant — exhibit withdrawn |
| `reblocked_contrasts.json` | **DERIVED** (2026-09-05) | every rev-3 contrast recomputed with the scenario-blocked, dose-paired bootstrap; which significance calls change |
| `inflight_box*/` | staging | pools and response logs from the sandboxes (committed); feature caches and result duplicates ignored |

**Numbers in `docs/review/RESULTS_LOG.md` with no committed file:** the B1 direction-stability
gate (0.909 / logreg 0.577). It is written only to `b1_e4rerun_<tag>.json` at sweep
completion, which never happened on the original sandbox. Registered as `b1.stability_gate_dom`,
category (b). Both rev-3 jobs were relaunched from scratch on 2026-09-04; see `rev3/README.md`.

---

## Referenced somewhere, absent from the repo

| path | referenced by | status |
|---|---|---|
| `results/generations/responses_qwen36.jsonl` | `src/verbalization/jlens_coupling.py` (input) | not committed; only the 300-line sample exists |
| post-fix stability-gate files (27B 0.967, 8B 0.965) | README, RESULTS_LOG, claims `stability.postfix_gate_*` | category (b) — lost with sandbox leases; A2 is the re-measurement |
| `b1_e4rerun_qwen36-27b.json`, `a2_estimator_qwen36-27b.json` | `src/rev3/*.py` final outputs | runs not finished as of `70894ba` |
| any Llama-3.1 artifact | earlier text | none exists; every Llama file here is Llama-3 (`model_identity.llama31`, category (d)) |
| HF dataset `punctualprocrastinator/coupling-27b-results` | README | private; counts quoted for it disagree and are not to be cited |

## Cross-file cautions

- `dualest_qwen36-27b.json` sweeps α up to 2.0; `e2ci_qwen36-27b.json` caps at 1.0 by
  pre-declared docstring. Do not compare their contagion rows without saying which grid.
- `behavioral_powered_qwen36-27b.json` and `behavioral_multiturn_qwen36-27b.json` put the
  desperate-vs-calm cheat difference in opposite directions on the same model. Neither is
  significant. RESEARCH_PLAN §3.2 preregisters the reversal for the rebuilt instrument.
- The three `e5actpass_*` files are one β sweep, not three replications.
- `stabdiag_llama3-abl.json` and `stabdiag_llama3-abl-big.json` are two pool sizes of the
  same diagnostic; the registered 8B number comes from `-big`.

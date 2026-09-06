# Claims-and-numbers check, round 2 (paper_rev3.tex, 2026-09-06 draft)

Scope: every number and quantitative claim in `docs/writeups/paper_rev3.tex`, `table_arm_vs_none.tex` and `table_b1e_draws.tex`, traced to `docs/review/claims.json` and to fields of the committed result files under `results/rev3/`. Values were read with python (scripts kept in the session scratchpad: `verify_ablation.py`, `verify_estimator.py`, plus two inline passes). Precision: a number is marked Y when the file value rounds to the printed value; "approx" when it rounds differently by one unit in the last digit or comes from a different version/convention; N when it is wrong at the precision printed.

Files read: `paper_revision_stats.json`, `b1c_alllayer_{qwen36-27b,llama3-abl}.json`, `b1d_subspace_qwen36-27b.json`, `b1d_subspace_qwen36-27b-randk23.json`, `b1d_randk23_summary.json`, `b1d_frame_geometry.json`, `b1e_footprint_{qwen36-27b,llama3-abl}.json`, `b1e_footprint_qwen36-27b-draw{1..10}.json`, `b1e_draws_summary.json`, `b1_followup_qwen36-27b.json`, `a2_estimator_{qwen36-27b,llama3-abl}.json`, `a2_followup_qwen36-27b{,-nref2000,-l54,-l54-nref2000}.json`, `a2_followup_llama3-abl.json`, `a2_margin_{qwen36-27b-l43,qwen36-27b-l54,llama3-abl-l21}-v2.json`, `a2_lowd_{qwen36-27b-l43,llama3-abl-l21}-v2.json`, `reblocked_contrasts.json`.

## A. Checker output summary

`.venv/bin/python tools/check_provenance.py` (exit 0): 95 claims checked, 8 with problems, 0 quotable claims that fail. Categories (a) 91, (b) 2, (c) 1, (d) 1.

The eight failures are all `retracted` or `pending` legacy entries and none is cited by the paper:

| id | status | problem |
|---|---|---|
| e2ci.present_gt_other | retracted | value mismatch: claim 6, file 1 |
| e5.emotion_causally_responsible | retracted | value mismatch: claim 6, file 3 |
| behavioral.honesty_floor_ruled_out | retracted | value mismatch: claim True, file False |
| induction.scramble_control | retracted | value mismatch: claim True, file False |
| stability.postfix_gate_27b | pending (b) | no output file |
| stability.postfix_gate_8b | pending (b) | no output file |
| dataset.13k_released | retracted (c) | no output file |
| model_identity.llama31 | retracted (d) | no output file |

The checker prints the standing warning that category (c) is non-empty (`dataset.13k_released`); this does not touch any number in the paper but the footnote's "machine-checked" statement is true only for the quotable subset.

`\rid` coverage: the paper carries 50 distinct `\rid` tags (46 exact ids, 4 wildcards `a2_8b.*`, `a2f54.*`, `b1e10.*`, `b1f.*`). All resolve to registry ids and all resolved ids are `quotable`. No failing id is cited. 22 quotable ids are not cited by the paper (E0/A3/B1-original/stabdiag entries and `b1d.verdict_instrument_failed`, `b1c.gate_all_layers`, `b1e8.permdir_inert`, `a2f.std_space_matches_raw`, `a2f2.anchor_level_shift`, `reblock.b1f_emo_vs_rand_blocked`); of these, `b1d.verdict_instrument_failed` and `b1c.gate_all_layers` back sentences the paper does make (B1d verdict; "every layer from 13 upward passes the gate") and could be tagged.

## B. Number-by-number table

Abbreviations: prs = `paper_revision_stats.json`; b1c/b1d/b1e = the 27B driver files; k23 = `b1d_randk23_summary.json`; drw = `b1e_draws_summary.json`; a2 = `a2_estimator_*`; a2f = `a2_followup_*`; m = `a2_margin_*-v2`; l = `a2_lowd_*-v2`.

| # | Location | Number | Source | Match | Note |
|---|---|---|---|---|---|
| 1 | Abstract | 5/6 emotions respond, 27B | b1c counts.n_testable = 5 | Y | |
| 2 | Abstract | 5/6 on Llama-3-8B | b1c8.transfer_replicates (llama counts.n_testable = 5) | Y | |
| 3 | Abstract, Contrib., §3 | logistic 0.57 at n=2000 (27B) | a2.focus_logreg_split_half = 0.5664 | Y | |
| 4 | Abstract, Contrib., §3 | logistic 0.29 (8B) | a2_8b.focus_logreg_split_half = 0.2860 | Y | |
| 5 | Abstract, Contrib., §3 | dom 0.97 / 0.94 | 0.9736 / 0.9413 | Y | |
| 6 | Abstract, §3 | fixed per-sample penalty does not close gap | a2f L43 logreg_lam 2000 = 0.581 | Y | |
| 7 | Abstract, §3 | cosine 0.97 → 0.63 | m L43 cos_lam_dom 75 = 0.971, 2000 = 0.633 | Y | |
| 8 | Abstract, §3 | 100 PCs: reproduces at 0.90 | l 27B by_dim.100.2000.sh_logreg_std = 0.901 | Y | |
| 9 | Abstract, §3 | projections correlate 0.90–0.96 | l full proj_corr 0.956 (75) → 0.902 (2000) | Y | |
| 10 | Abstract | median 91% of response left (vs random or vs baseline) | 1 − 0.0922 (b1c decision.median); 1 − 0.0895 (prs emo_vs_none_5) | Y | |
| 11 | Abstract | 83% with untestable emotion included | vs baseline all-6: 1 − 0.167 = 83%; vs random all-6: 1 − 0.152 = 85% | approx | 83% holds only for the baseline comparison; sentence attaches it to both. See D1. |
| 12 | Abstract | rank-5 affect subspace fails instrument check | b1d verdict = instrument_failed; counts.n_sub_fail = 3 | Y | |
| 13 | Abstract, Fig 4 | permuted rank-5 blocks three emotions | b1d.perm_control_blocks = 3 (afraid, sad, angry) | Y | |
| 14 | Abstract, Intro, §6 | "as strongly as the affect subspace" | prs b1d sub_all vs none: sad 0.54, angry 0.40; perm_all: 0.27, 0.23; sub−perm sad −25.7 [−49.0, −3.3], angry −33.8 [−65.3, −4.5] | N | Overclaim, see E1 |
| 15 | Abstract, §5 | random 5-frame of half its footprint | b1d randsub_all 3.10–3.24 vs perm 6.49–6.89 | Y | |
| 16 | Abstract, §5, §6 | random 23-frame of same footprint inert | k23 6.67–6.87; q ≥ 0.714; reading = fitting | Y | |
| 17 | Abstract, §5 | permuted rank-1 inert | b1e.permdir_inert = 0/6 sig | Y | |
| 18 | Abstract, §5, Tab 4 | sad block holds in 10/10, median 35%, 24–56% | drw sad emo−perm: n_sig_neg 10, median 0.351, 0.238–0.556 | Y | |
| 19 | Abstract, §5, Tab 4 | afraid 18%, 7–23%, check passes 4 | drw afraid: 0.175, 0.066–0.225; n_mc_steer_sig = 4 | Y | |
| 20 | Abstract, §4 | other-speaker projection rises "as much as its own" | prs other slopes 59/119/65/93/96 vs present 84/212/48/98/196 | approx | Section 4 says "comparably"; present is larger for 4 of 5. See E3. |
| 21 | Intro | "only sad loses a third in every one of ten draws" | drw sad emo−perm min 0.238, emo−none min 0.228 | N | A third at the median; 24% in the lowest draw. See E2. |
| 22 | Intro | afraid "a fifth in the draws where its check passes" | drw afraid emo−perm in gate-passing draws 1,4,9,10: 0.182, 0.187, 0.225, 0.143 | Y | mean 0.184 |
| 23 | Intro | "seven interpretive sentences withdrawn; eighteen corrections" | repository documents, not a result file | n/a | not checkable against results |
| 24 | Prior work | 27B logistic plateau 0.55–0.61 | a2f L43/L54 at 2000: logreg 0.566/0.554, lam 0.581/0.563, cv 0.608/0.585 | Y | |
| 25 | Prior work | 8B falls to 0.28–0.38 | a2f 8B at 1200: lam 0.276–0.318, logreg 0.286–0.338, cv 0.374–0.412 | Y | |
| 26 | Prior work | dom rises to 0.94–0.98 | 0.941 (8B), 0.974 (L43), 0.979 (L54) | Y | |
| 27 | Prior work, §3 | 0.43 and 0.33–0.37 (CAA-raw vs logreg-raw) | a2f L43 0.427; 8B L14/21/27 0.371/0.336/0.329 | Y | |
| 28 | Prior work | RAPTOR 0.87–0.98; Braun 5–30 / 200–500 / >0.99; Herbster 0.98–0.99 | citations | n/a | not in scope |
| 29 | §2 | 29 scenarios; layer 42; doses 0/0.33/0.67/1; three replicates | b1c config n_scenarios 29, steer_layer 42, doses, reps 3 | Y | |
| 30 | §2, App. | Qwen revision 6a9e13bd; Llama revision dd67dd05 | model_revision stamps in b1c/b1e/a2 files (both models) | Y | |
| 31 | §2 | 808 / 807 / 1,615 items | read_half_n 808, dir_half_n 807, probe_n 1615 | Y | |
| 32 | §2, App. | 5,000 bootstrap draws, 29 clusters | acl_core.analyze default n_boot=5000; n_blocks 29 in every contrast | Y | default in code, not stamped in the files |
| 33 | §2 | 6×6 cells, 160 per cell (vs 60) | a2 27B config k=160; b1c probe_k 60 | approx | 8B estimator pool used k=220 (a2 8B config), not 160. See D4. |
| 34 | §2 | 4,253 kept of 5,760; 3,898 | a2 n_pool 4253 / 3898; 36×160 = 5760 | Y | 5760 is arithmetic, not stamped |
| 35 | §2 | 1,949; curves stop at 1,200 | 3898/2; a2f 8B target_n reached 1200 | Y | |
| 36 | §2 | deployed directions' gate 0.88–0.93 | b1c direction_stability.ablated_layer_summary min 0.882 max 0.927 | Y | |
| 37 | Fig 1 cap. | 27B plateau near 0.55–0.6 | as #24 | Y | |
| 38 | Fig 1 cap. | 8B "fall to 0.28–0.33" for C=0.5 and fixed penalty | 8B focus layer at 1200: 0.286 (C=0.5), 0.276 (lam); 0.33 matches only L14's C=0.5 (0.338), not drawn | approx | |
| 39 | Fig 1 cap., §3 | tuned C 0.61 and 0.38 | a2f cv 2000 = 0.608; 8B L21 cv 1200 = 0.384 | Y | |
| 40 | §3 | 0.974 / 0.566 at focus | a2 headline | Y | |
| 41 | §3 | layer 54: 0.979 / 0.554 | a2f L54 dom 2000 = 0.9789; logreg 2000 = 0.5541 | Y | |
| 42 | §3 | 8B focus 0.941 / 0.286 | a2 8B headline | Y | |
| 43 | §3 | tuned C 0.37–0.41 across three 8B depths | cv 1200: 0.374 / 0.384 / 0.412 | Y | |
| 44 | §3 | C_n = 0.5 N_ref / n | a2f config c_ref 0.5, n_ref 600 | Y | |
| 45 | §3 | 0.581 at both ends, 600 anchor | lam 150 = 0.5812, 2000 = 0.5810 | Y | |
| 46 | §3 | dip at 600 of −0.021, paired-significant, 1 of 10 positive | anchor 600: 300→600 −0.0212, t −4.55, 0/10 positive; anchor 2000: −0.0237, 1/10 positive | approx | "1 of 10 positive" belongs to the 2000 anchor; the 600 anchor's dip is 0/10. See D6. |
| 47 | §3 | partial recovery | 600→2000 +0.0058 (7/10) anchor 600; +0.0124 (8/10) anchor 2000 | Y | |
| 48 | §3 | anchor 2000: no net rise | 150→2000 −0.0077, 3/10 positive | Y | |
| 49 | §3 | layer 54: one drop 300→600 then flat | lam 0.608 → 0.573 → 0.566 → 0.563 | Y | |
| 50 | §3 | rise at layer 16 | lam 300 = 0.480 → 2000 = 0.504 (+0.024) | Y | |
| 51 | §3 | 8B peak at 150, fall to 0.276 at 1200 | lam L21 0.369 → 0.276 | Y | |
| 52 | §3 | n ≤ 2000, d = 5120, 27-fold; n ≤ 1200, d = 4096 | a2 by_layer.43 d = 5120; 8B d = 4096; 2000/75 = 26.7 | Y | |
| 53 | Fig 2 cap., §3 | training accuracy 1.000 in every cell, arm, seed; C = 0.5, 50, 500 | m v2 all arms min train_acc 1.0; config c_weak 50, c_weak2 500 | Y | |
| 54 | §3 | worst single-fit log-loss 0.0055 | m 8B L21 1200 lam_logloss max = 0.00560 | approx | v1 (5-seed) value; the cited v2 file gives 0.0056 |
| 55 | §3 | at most 80 iterations | m 8B L21 1200 lam_n_iter max = 81.0 (per-seed means 79.5–81) | N | v2 file exceeds 80 |
| 56 | §3 | weak penalty leaves direction unchanged, cosine ≥ 0.96 | cos_lam_weak min 0.9641 (8B 1200); 27B ≥ 0.976 | Y | |
| 57 | §3 | 0.52 against 0.58 at n=2000 | sh_weak2 0.5216 vs sh_lam 0.5810 | Y | |
| 58 | §3 | 0.91 → 0.43 on 8B | m v2 8B cos_lam_dom 75 = 0.920, 1200 = 0.435 | approx | 0.91 is the v1 value (0.913); v2 file gives 0.92 |
| 59 | §3 | held-out 0.82→0.94 vs 0.79→0.88 | l full heldout_acc_logreg 0.819→0.938; dom 0.788→0.884 | Y | |
| 60 | §3 | n/d' up to 20; train acc 0.999, 0.989, 0.974 | l by_dim.100 train_acc 600/1200/2000 | Y | |
| 61 | §3 | 0.90 vs 0.94 (std), from 0.39 and 0.33 | l 100: sh_logreg_std 0.393→0.901; sh_dom_std 0.327→0.938 | Y | |
| 62 | §3 | cosine to dom 0.91–0.94 across n; full width 0.90→0.62 | l 100 cos_logreg_dom_std 0.910–0.937; full 0.898→0.617 | Y | |
| 63 | §3 | held-out 0.92 vs 0.94 | 0.921 vs 0.938 | Y | |
| 64 | §3 | 8B: train acc 0.90 at 1200; 0.81 vs 0.85 | l 8B 100: 0.904; sh_logreg_std 0.810; sh_dom_std 0.852 | Y | |
| 65 | §4 | none slopes +48 to +212 | prs happy 47.9, afraid 211.8 | Y | |
| 66 | §4 | calm +44 [−3, +99] | prs calm 44.2 [−3.1, 98.9] | Y | |
| 67 | §4 | happy +71 [+50, +94] then +0 [−24, +25] | prs shape 70.9 [49.8, 94.2]; 0.1 [−24.0, 24.7] | Y | |
| 68 | §4 | calm +38 [−3, +78] then −34 [−77, +13] | 38.3 [−2.8, 78.4]; −34.0 [−77.2, 12.8] | Y | |
| 69 | §4 | desperate, sad, angry rise at both steps | last-step: desperate 9.9 [−8.2, 28.7], sad 21.5 [−3.0, 45.3], angry 48.8 [6.5, 90.9] | approx | point estimates rise; only angry's last step excludes zero. See E4. |
| 70 | §4 | B1 follow-up: same five testable, afraid/sad blocked | b1_followup counts n_testable 5, blocking [afraid, sad], verdict mixed | Y | |
| 71 | §4 | B1c–B1e agree bit for bit on shared arms | none/emo_all slopes identical across b1c/b1d/b1e (e.g. 211.772, 159.924) | Y | |
| 72 | §4 | manipulation check passes 6/6 in each run | n_mc_steer_pass 6 (b1c, b1d, b1e); b1f n_mc_pass 6 | Y | |
| 73 | §4 | 8B: same five respond; manipulation half as strong | llama b1c n_testable 5; mc_steer mean +0.25 vs +0.50 (registry note) | Y | second half from registry note only |
| 74 | §4 | other slopes +59, +119, +65, +93, +96 | prs other_probe_none_slope 59.3/119.2/65.4/93.1/95.9 | Y | |
| 75 | §4 | present +84, +212, +48, +98, +196 | prs none_slope 84.1/211.8/47.9/97.5/196.0 | Y | |
| 76 | §5 preamble | every layer from 13 passes ≥ 0.80; window 0.2× depth | b1c gate min 0.882; layers_below_gate []; 13/64 = 0.20 | Y | |
| 77 | Tab 1 B1c row | 0.06, 0.21*, 0.05, (0.94)·, 0.33*, 0.09 | b1c decision.per_emotion_fraction; calm 0.939 sig; q afraid 0.001, sad 0.048 | Y | |
| 78 | Tab 1 B1d row | 0.14, −0.13·, 0.05, (1.56)·, 0.26·, 0.17· | b1d decision.per_emotion_fraction; calm 1.563 sig; q 0.0507 for afraid/sad/angry | Y | |
| 79 | Tab 1 B1e row | 0.06, 0.25*, −0.14, (1.02)·, 0.29*, 0.09 | b1e blocked_fraction.emo_all_vs_permdir_all; q afraid 0.001, sad 0.028 | Y | |
| 80 | Tab 2 B1c rows (3×6) | as printed | prs b1c arm_vs_none blocked_fraction, sig, bh_q | Y | all 18 cells and marks match; blocked fraction = −diff/none verified to 1e-9 |
| 81 | Tab 2 B1d rows (3×6) | as printed | prs b1d | Y | all 18 cells and marks match |
| 82 | Tab 2 rank-23 row | +0.06, −0.03, +0.03, (+0.69)·, +0.07, +0.00 | k23 +0.062, −0.029, +0.027, +0.685, +0.070, −0.0002 | approx | calm 0.6849 rounds to 0.68; angry −0.0002 printed as +0.00 |
| 83 | Tab 2 B1e rows (2×6) | as printed | prs b1e | Y | all 12 cells and marks match |
| 84 | Tab 3 B1c emo | 0.46–0.66 (1); readable 0.16–0.79 | b1c emo_all removed_norm 0.463–0.659; floor-corrected share 0.161–0.787 | Y | |
| 85 | Tab 3 B1c emo | dose-0 shift ≤ 0.01 | prs b1c dose0_readout_shift emo_all: happy −0.071, sad −0.020 | N | true bound is ≤ 0.08; registry (b1d.readout_baseline_shift) says "within 0.08" |
| 86 | Tab 3 B1c random | 1.15–1.26 (2.0); ≤ 0.01; ≤ 0.03 | 1.155–1.262; mean ratio 2.05; shifts ≤ 0.003; share ≤ 0.017 | Y | |
| 87 | Tab 3 B1d affect | 6.9 (10.5) | sub_all 6.873–6.941; ratio mean/mean 11.6; 6.9/0.66 = 10.5 | approx | ratio convention differs from B1e rows (mean/mean gives 11.6). See D2. |
| 88 | Tab 3 B1d affect | +0.39 / −0.25; 0.11–1.25 | prs sub_all shifts 0.392, −0.253; share 0.114–1.249 | Y | |
| 89 | Tab 3 B1d permuted | 6.5–6.9 (10); ≤ 0.08; 0.07–0.52 | 6.486–6.889 (mean ratio 11.2); happy shift −0.0801; share 0.065–0.524 | Y | happy's shift is 0.0801, at the boundary |
| 90 | Tab 3 B1d random 5 | 3.1–3.2 (5); ≤ 0.01; ≤ 0.03 | 3.098–3.240 (5.4); ≤ 0.004; ≤ 0.025 | Y | |
| 91 | Tab 3 rank-23 | 6.7–6.9 (10)†; ≤ 0.01; ≤ 0.13 | k23 6.673–6.873 (11.4 vs B1d emo mean); ≤ 0.006; afraid 0.125 | Y | none arms identical to B1d (verified) |
| 92 | Tab 3 B1e permdir | 0.50 (0.85; afraid 0.84, sad 0.77); ≤ 0.01; ≤ 0.09 | 0.502; 0.846; 0.836; 0.767; ≤ 0.008; ≤ 0.087 | Y | |
| 93 | Tab 3 B1e pc1 | 3.47 (5.8); +0.55 / −0.28 | 3.471; 5.84; 0.554 / −0.277 | Y | |
| 94 | Tab 3 B1e pc1 | readable affect removed 0.71–1.5 "(range over emotions)" | b1e share_removed pc1_all: desperate 0.006, afraid 0.787, happy 1.442, calm 1.543, sad 0.713, angry 0.866 | N | range over emotions is 0.006–1.54; desperate's share is confounded by its +0.55 dose-0 shift but the cell claims a range |
| 95 | Fig 4 cap. | perm frame blocks afraid/sad/angry at q<0.05; random dir, random 5-frame, permdir block nothing; pc1 blocks angry | prs bh_q: perm 0.001/0.035/0.013; rand_all ≥ 0.264; randsub ≥ 0.511; permdir ≥ 0.935; pc1 angry 0.001 | Y | |
| 96 | §5 B1c | median 9% [4%, 25%] | b1c decision 0.0922 [0.038, 0.254] | Y | |
| 97 | §5 B1c | per-emotion 0.06, 0.21, 0.05, 0.33, 0.09; 15% with calm | b1c; prs decision_contrast_all_6 = 0.152 | Y | |
| 98 | §5 B1c | baseline median 9% (17% with calm) | prs emo_vs_none_5 = 0.0895; all_6 = 0.167 | Y | |
| 99 | §5 B1c | afraid 21% q 0.001; sad 33% q 0.048 | b1c counts.bh_fdr_q | Y | |
| 100 | §5 B1c | baseline afraid q 0.001; sad q 0.115 | prs bh_q emo_all | Y | |
| 101 | §5 B1c | extra blocking < 0.08 for afraid, angry, happy, sad | −ci_lo/none: 0.011, 0.009, 0.054, 0.0797; desperate 0.17, calm 0.41 | Y | sad at 0.0797, at the boundary |
| 102 | §5 B1c | random inert all six; happy +13.5 [−0.1, +28.4]; q ≥ 0.26 | prs rand_all: 13.5 [−0.1, 28.4]; min q 0.264 | Y | |
| 103 | §5 B1c | r ≈ 0.85, p ≈ 0.07 on five points | recomputed from b1c ctx_readout_by_dose and decision fractions: r 0.849, p 0.069 | Y | |
| 104 | §5 B1d | 16–21% of variance | frame_geometry var_true 0.1635–0.2053 | Y | |
| 105 | §5 B1d | ≥ 80% removed for only 3 of 6 | counts.n_sub_fail 3; sub_pass True for happy, calm, sad | Y | |
| 106 | §5 B1d | desperate 0.11 → 0.50 | ctx_readout_by_dose['0.0'] 0.108 → 0.500 | Y | |
| 107 | §5 B1d | affect blocks sad 54%, angry 40% at q = 0.001 | prs sub_all 0.538, 0.399; bh_q 0.0005, 0.0005 | approx | q is 0.0005 (bootstrap floor); "q ≤ 0.001" would be exact |
| 108 | §5 B1d | permuted: afraid 23%, sad 27%, angry 23%; q 0.001, 0.035, 0.013 | prs perm_all 0.234, 0.274, 0.226; q 0.001, 0.0353, 0.013 | Y | |
| 109 | §5 B1d | a third of its span in common | frame_geometry perm_vs_true 0.316–0.347 | Y | |
| 110 | §5 B1d | dose-0 shifts below 0.08 (permuted) | happy −0.0801 | approx | at the boundary |
| 111 | §5 B1d | random 5-frame q ≥ 0.51 | prs randsub bh_q min 0.511 | Y | |
| 112 | §5 B1d | 23-frame 6.7–6.9 vs 6.5–6.9; fractions −0.03 to +0.07; q ≥ 0.71 | k23 | Y | |
| 113 | §5 B1d | pre-registered contrast q = 0.051 | b1d counts.bh_fdr_q 0.0507 | Y | |
| 114 | §5 B1d | afraid +29.3, angry −60.6 | b1d sub_all_vs_emo_all 29.3 [8.6, 49.6]; −60.6 [−95.6, −27.4]; 2 of 6 sig | Y | |
| 115 | §5 B1e, App. | control cosine 0.004 to emotion direction | no field in b1e file (only pc1_vs_emo_cos, pc1_dir_read_cos) | UNSUPPORTED | paper and registry both say it is not stamped |
| 116 | §5 B1e | 0.85× (0.84 afraid, 0.77 sad) | b1e footprint ratio 0.846; per-emotion 0.836, 0.767 | Y | |
| 117 | §5 B1e | changes no dose-response, removes no readable affect | permdir_all_vs_none sig 0/6; share ≤ 0.087 | Y | "no readable affect" with afraid at 8.7%; Table 3 prints ≤ 0.09 |
| 118 | §5 B1e | afraid 25% [17, 33]; sad 29% [7, 56]; q 0.001, 0.028 | b1e blocked_fraction 0.248 [0.169, 0.325]; 0.291 [0.072, 0.556]; bh_q | Y | |
| 119 | §5 B1e | baseline 24% and 28% | prs b1e emo_all 0.245, 0.284 | Y | |
| 120 | §5 B1e | permuted within 7% (afraid), 13% (sad) throughout | drw permdir−none max 0.065, min −0.132 | Y | |
| 121 | §5 B1e | pc1 blocks angry 61% (q 0.001) and calm | b1e pc1_all_vs_none angry −120.4, fraction 0.614, q 0.001; calm −43.5 [−91.1, −0.04] sig | Y | |
| 122 | §5 B1e | pc1 removes 79% and 71% of afraid/sad readable affect | share_removed pc1_all 0.787, 0.713 | Y | |
| 123 | §5 B1e | dose-0 up to 0.55 | pc1 desperate 0.554 | Y | |
| 124 | §5 B1e | 10× against 0.85× | B1d convention vs mean/mean | approx | see D2 |
| 125 | §5 B1e | secondary readout agrees for sad, not afraid | b1e *_secondary: sad −0.074 [−0.124, −0.030] sig; afraid +0.022 n.s. | Y | |
| 126 | §5 8B | afraid 15%, sad 31% vs random | llama b1c blocked_fraction 0.149, 0.309 | Y | both also clear BH-FDR (q 0.005, 0.001); "by unadjusted CIs" understates |
| 127 | §5 8B | instrument_failed: upper window unmeasurable | llama b1c verdict; n_mc_ablate_fail 4 | Y | |
| 128 | §5 8B | B1e 8B: sad 36% q 0.002 secondary agreeing; afraid 14% q 0.08, no block vs baseline | llama b1e 0.361, q 0.002, secondary sig; 0.143, q 0.08; emo_all_vs_none afraid −10.2 [−25.0, 4.8] n.s. | Y | |
| 129 | §5 8B | control matched on average; B1c random reproduced contrasts | ratio 0.955; −18.1/−31.5 vs −17.4/−36.8 | Y | |
| 130 | §6 | "top principal direction blocks a different one" | pc1 blocks angry; angry is also blocked by affect (0.40*) and permuted (0.23*) frames | approx | wording inconsistency, see D7 |
| 131 | §8 | "carries a third of sad's response in every draw and on both models" | drw min 0.238; 8B sad vs baseline 0.27 (vs permuted 0.36) | N | see E2 |
| 132 | App. repro | torch 2.11.0+cu130, transformers 5.14.1 | _provenance stamps | Y | |
| 133 | App. repro | 27B experiment ≈ 2 h; 432 or 288 arm-rows; 19–21 s | b1c 2.55 h / 432 / 21.2 s; b1d 1.98 h / 360 / 19.8 s; b1e 1.67 h / 288 / 20.9 s | approx | B1d's 360 rows omitted; B1c is 2.5 h |
| 134 | App. repro | 8B experiment under thirty minutes | llama b1c 2831 s = 47 min; llama b1e 1679 s = 28 min | N | B1c on the 8B took 47 min |
| 135 | App. repro | seeds: pool 0, CRC32 rule, split seeds 0–9, bootstrap seed 0 | b1c seeds {pool 0, run 0}; gen_seed_rule in config; a2f split_seeds 0–9; analyze seed=0 | Y | |
| 136 | App. repro | draws: split seeds 1–10, afraid/sad, arms none/emo/permdir; prereg stamped | draw files split_seed 1–10, emotions [afraid, sad], 3 arms, prereg sha 4cb36929 found | Y | |
| 137 | App. repro | rank-23 run: none + random frame; prereg addendum stamped | randk23 file arms [none, randsub_all], randsub_k 23, sha c3bbe034 found | Y | |
| 138 | App. repro | 5 × (6.7/3.15)² ≈ 23 | 22.6 | Y | |
| 139 | App. repro | 110 new tokens, temperature 0.9 | b1c/b1e config max_new 110, temp 0.9 | approx | a2 estimator pool used max_new 240 (a2 config); stated as a global hyperparameter |
| 140 | App. repro | lbfgs, max_iter 3000; C set; CV decade grid | a2_followup.py:161; margin config max_iter 3000, c_weak 50/500 | Y | |
| 141 | App. details B1c | cc2dc32 at 19:22Z precedes run start 19:32Z | git: 2026-09-05T00:52:06+05:30 = 19:22Z; started_utc 19:32:20Z | Y | |
| 142 | App. details B1c | pool 1,615 vs registered 2,160; FDR over five | probe_n 1615; bh_q over 5 | Y | |
| 143 | App. details B1d | permuted frame removed 6.5% of afraid's readable affect | share_removed perm_all afraid 0.0655 | Y | |
| 144 | App. details B1d | 23-frame shifts calm −30 [−55, −4]; calm slope +44 [−3, +99] | k23 −30.3 [−55.4, −4.3], p 0.024 | Y | |
| 145 | App. details B1e | control removes less than B1c's random direction (1.22) | b1c rand_all mean removed_norm 1.220 | Y | |
| 146 | App. details B1e | sad's range starts at 0.07 | 0.0717 | Y | |
| 147 | App. details draws | draw 4 on a different machine | draw 4 started 08:44Z while draw 3 (08:38–09:07Z) was running; hostname not stamped | approx | consistent, not stamped |
| 148 | App. details draws | failing-draw slopes −0.02 to +0.09, two include zero by < 0.01 | draw files afraid mc_steer: −0.0211 … +0.0933; draw 2 ci_lo −0.0076, draw 6 −0.004 | Y | |
| 149 | App. details draws | verdict split four H3 / six instrument_failed; contrast significant in nine | drw verdicts; n_sig_negative 9 | Y | |
| 150 | App. details draws | permuted vs baseline significant in one of twenty emotion-draws (afraid, draw 6, +6.5%) | afraid draw 6 +0.065 sig; sad draw 3 −0.132 also sig (positive contrast; drw sad n_sig_positive = 1) | N | two of twenty |
| 151 | App. details 8B | sad 0.86× | llama b1e sad ratio 0.860 | Y | |
| 152 | App. details estimator | C = 50 and 500 identical to three decimals | cos_weak_weak2 0.99999; sh 0.5224 vs 0.5216 | Y | |
| 153 | App. details estimator | three driver builds; shared cells agree bit for bit | logreg/cv/dom identical between anchor-600 and anchor-2000 files at L43 and L54 | Y | |
| 154 | Tab 4 afraid emo−perm | 0.25 / 0.18 / [0.07, 0.23] / 9/10 / 4/10 | drw 0.248 / 0.175 / 0.066–0.225 / 9 / 4 | Y | |
| 155 | Tab 4 afraid emo−none | 0.24 / 0.19 / [0.08, 0.23] / 10/10 | 0.245 / 0.189 / 0.081–0.227 / 10 | Y | |
| 156 | Tab 4 afraid perm−none | 0.00 / 0.02 / [−0.04, 0.07] / 1/10 | −0.003 / 0.018 / −0.039–0.065 / 1 | Y | |
| 157 | Tab 4 sad emo−perm | 0.29 / 0.35 / [0.24, 0.56] / 10/10 / 10/10 | 0.291 / 0.351 / 0.238–0.556 / 10 / 10 | Y | |
| 158 | Tab 4 sad emo−none | 0.28 / 0.35 / [0.23, 0.44] / 10/10 | 0.284 / 0.346 / 0.228–0.438 / 10 | Y | |
| 159 | Tab 4 sad perm−none | −0.01 / −0.01 / [−0.13, 0.09] / 0/10 | −0.007 / −0.007 / −0.132–0.086 / 0 sig-negative | Y | one draw is significantly positive (see #150) |
| 160 | Contributions | logistic 0.57 / 0.29 vs dom 0.97 / 0.94 | as #3–5 | Y | |
| 161 | Contributions | verdicts H1, instrument_failed, H3 | b1c/b1d/b1e verdict fields | Y | |
| 162 | Limitations | ten re-splits for B1e's two emotions only | drw emotions [afraid, sad] | Y | |

Numbers/claims checked: 162 line items (several rows bundle a full table row or a pair of values; roughly 260 individual printed numbers).

## C. Mismatches and unsupported numbers (severity-ranked)

Hard mismatches (the file gives a different value at the printed precision):

1. **Table 3, B1c emotion direction, dose-0 shift "≤ 0.01"** (#85). `paper_revision_stats.json b1c_27b.per_emotion.happy.dose0_readout_shift.emo_all = −0.071`, sad −0.020. The bound is ≤ 0.08, which is also what the registry note for `b1d.readout_baseline_shift` says ("within 0.08"). Moderate: the row is used to argue the rank-1 arm leaves the instrument intact; 0.07 is still small but the printed bound is wrong by 7×.
2. **Table 3, B1e top principal direction, readable affect removed "0.71–1.5" as a range over emotions** (#94). `b1e_footprint_qwen36-27b.json summary.desperate.mc_footprint.share_removed.pc1_all = 0.006`. The range over the six emotions is 0.006–1.54. Desperate's share is uninterpretable because its dose-0 readout shifts by +0.55, but the cell claims a range, not a range over interpretable emotions. Moderate.
3. **Appendix, "the permuted direction against baseline reaches significance in one of the twenty emotion-draws"** (#150, #159). Two of twenty are significant: afraid draw 6 (blocked +6.5%) and sad draw 3 (`b1e_draws_summary.json across_draws.sad.permdir_all_vs_none.n_sig_positive = 1`, blocked fraction −0.132, i.e. the control significantly amplified sad's response in that draw). Table 4's "0/10" is correct as a count of significant negatives; the appendix sentence is not. Low-moderate.
4. **Appendix, "the 8B one under thirty minutes"** (#134). `b1c_alllayer_llama3-abl.json _provenance.elapsed_s = 2830.8` (47 min). Only the 8B B1e run (28 min) is under thirty. Low.
5. **§3, "at most 80 iterations"** (#55). `a2_margin_llama3-abl-l21-v2.json by_layer.21.1200.lam_n_iter.max = 81.0`. Low.

Rounding / version / convention discrepancies (approx):

6. "worst single-fit log-loss 0.0055" — the cited v2 file gives 0.0056 (#54); "0.91 → 0.43 on the 8B" — v2 gives 0.92 → 0.435 (#58). Both are the five-seed v1 values carried into a sentence tagged with the v2 claim.
7. "q = 0.001" for the affect subspace's sad and angry blocks against baseline — file gives 0.0005 (#107).
8. Table 2 rank-23 calm "+0.69" — file 0.6849 (#82).
9. "dose-0 readout shifts below 0.08" for the permuted frame — happy is −0.0801 (#110); "extra blocking under 0.08" — sad is 0.0797 (#101). Both sit on the boundary and should be stated as ≤ 0.08.
10. Figure 1 caption "fall to 0.28–0.33 (8B)": the drawn 8B focus-layer curves end at 0.286 and 0.276; 0.33 corresponds to layer 14, which the figure does not show (#38).

Unsupported by any committed file:

11. **The permuted-label direction's cosine 0.004 to the emotion direction** (#115). No field in `b1e_footprint_qwen36-27b.json` (the file stamps only `pc1_vs_emo_cos` and `pc1_dir_read_cos`). The paper's appendix and the registry note both disclose that it was read from the pre-launch verifier and not stamped. It remains a number in the paper with no committed source. Moderate, disclosed.
12. "Seven of our own interpretive sentences were withdrawn ... eighteen more corrections" (#23) are document counts with no result-file pointer; not checkable here.
13. The steering manipulation on the 8B being "half as strong and less specific" (#73) rests on registry notes (mean readout gain +0.25 vs +0.50), not on a quoted file field; the claim is plausible from `mc_steer.slope` values but no per-model summary field was located.

## D. Internal inconsistencies

1. **Abstract vs §5 B1c, the "with calm" median.** Abstract: "median 91% ... against a random-direction control or against baseline (83% with the untestable emotion included)". The 83% (= 1 − 0.167) is the baseline figure; against the random control the six-emotion median is 0.152, i.e. 85% (§5 B1c says "the median with calm included is 15%"). The parenthetical applies to only one of the two comparisons it is attached to.
2. **Table 3 footprint-ratio convention.** The B1e rows use mean-over-emotions / mean-over-emotions (`footprint.ratio_permdir_over_emo = 0.846`, `ratio_pc1_over_emo = 5.84`). The B1d rows print 10.5 / 10 / 5 / 10, which correspond to dividing by the largest per-emotion emotion-direction footprint (6.9/0.66 = 10.5); mean/mean gives 11.6, 11.2, 5.4 and 11.4. The same column mixes two conventions, and the text's "10× against 0.85×" inherits it.
3. **Appendix arm-row count.** "432 or 288 arm-rows" omits B1d's 360 (`b1d_subspace_qwen36-27b.json rows`), the rank-23 run's 144 and each draw's 72.
4. **§2 pool description vs the 8B estimator file.** "160 dialogues per cell" is the 27B setting (`a2_estimator_qwen36-27b.json config.k = 160`); the 8B pool used k = 220 (`a2_estimator_llama3-abl.json config.k = 220`; 36 × 220 = 7,920 generated, 3,898 kept).
5. **Appendix hyperparameters vs the estimator pool.** "110 new tokens per message" holds for the ablation runs (`max_new_A/B = 110`) but the estimator pool was generated with `max_new = 240` (a2 config), and §2 says that pool used "the same generator as the probe pool".
6. **§3 fixed-penalty sentence mixes the two anchors.** "a dip at 600 of −0.021 that is significant in a paired test over the ten shared splits, 1 of 10 positive" is written about the 600 anchor; on that anchor the dip is −0.0212 with 0 of 10 seeds positive (t = −4.55). "1 of 10 positive" is the anchor-2000 dip (−0.0237), which the registry entry `a2f2.lam_anchor2000_no_rise` describes.
7. **§6 "the top principal direction blocks a different one."** pc1 blocks angry (0.61*). Angry is also blocked by the affect subspace (0.40*) and by the permuted frame (0.23*), so it is not a different emotion from the three the previous clause names; it is different only from the rank-1 direction's pair (afraid, sad).
8. **Table 3 "≤ 0.01" for the B1c emotion direction vs the registry's "within 0.08"** (see C1).

Everything else checked for consistency agrees: the abstract, tables and sections quote the same values for the B1c/B1d/B1e contrasts; the median statements match the per-emotion table values; every Table 2 row is derivable from `paper_revision_stats.json` or `b1d_randk23_summary.json` with blocked fraction = −diff/none_slope (verified to 1e-9 for every cell); Table 4 matches `b1e_draws_summary.json` in all 18 cells; the rank-23 numbers match `b1d_randk23_summary.json`; the none arms of the rank-23 run are bit-identical to B1d's.

## E. Overclaims relative to the numbers

1. **"blocks three emotions as strongly as the affect subspace does" (abstract), "blocks the response as strongly as the affect subspace itself" (intro), "as strongly as the affect subspace blocks two of them" (§6).** Against baseline the affect subspace blocks sad by 54% and angry by 40%; the permuted frame blocks them by 27% and 23% (`paper_revision_stats.json b1d_27b`). The affect-minus-permuted contrast is negative with an unadjusted CI excluding zero for both (sad −25.7 [−49.0, −3.3], angry −33.8 [−65.3, −4.5]; q = 0.051). The permuted frame blocks about half as much for these two, and more than the affect subspace only for afraid (23% vs 11%). "Blocks three emotions at the FDR threshold, which the random controls do not" is what the numbers support; "as strongly" is not.
2. **"only sad loses a third in every one of ten direction draws" (intro); "the rank-1 direction carries a third of sad's response in every draw and on both models" (conclusion).** Across draws the block against the permuted control ranges 24–56% (median 35%) and against baseline 23–44%; the lowest draws lose a quarter, not a third. On the 8B the block is 36% against the permuted control but 27% against baseline, and the appendix itself notes a third of the 36% is control drift. "A quarter to a half in every draw, a third at the median" is what the files support.
3. **"the receiver's 'other-speaker' projection rises with dose as much as its own" (abstract).** Other-probe slopes are 59/119/65/93/96 against present-probe 84/212/48/98/196: smaller for four of five testable emotions (by half for afraid and angry), larger for happy. §4's "comparably" and the contribution list's "moves comparably" are accurate; "as much as" is not.
4. **"desperate, sad and angry rise at both steps" (§4).** Point estimates rise, but the last-step intervals include zero for desperate (+9.9 [−8.2, +28.7]) and sad (+21.5 [−3.0, +45.3]); only angry's excludes it. Read beside the surrounding CIs for happy and calm, the sentence implies a significant rise it does not have.
5. **"removes no readable affect" for the permuted rank-1 direction (§5 B1e).** Afraid's share removed is 0.087 (Table 3 prints ≤ 0.09). Mild; "removes at most 9%" is the supported statement.

Statements that were checked and are supported as worded: "survival of the rank-1 block against a permuted-label rank-1 control (H3)" (verdict field, with the draws' 4/6 split stated in the same paragraph); "inert" for the random direction, random 5-frame, rank-23 frame and permuted rank-1 direction (all q ≥ 0.26, 0.51, 0.71, 0.93 respectively); "instrument_failed" for B1d; "the same five emotions respond" on the 8B; "only sad replicates across models on both outcomes"; "every fit ... training accuracy 1.000".

## F. Verdict

Of 162 checked items (about 260 printed numbers), 145 match the committed files at the printed precision, 11 are rounding/version/convention discrepancies, 5 are wrong at the printed precision (Table 3's "≤ 0.01" and "0.71–1.5" cells, "one of twenty", "under thirty minutes", "at most 80 iterations"), and one number (the 0.004 cosine) has no committed source; none of the mismatches changes a verdict or a headline contrast, and every `\rid` tag resolves to a quotable registry entry. The paper's tables are derivable from the files as required, but two abstract/intro sentences ("as strongly as the affect subspace", "a third of sad's response in every draw") and the abstract's "as much as its own" claim more than the per-emotion numbers support and should be narrowed before submission.

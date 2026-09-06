# Numbers check, final (revision notes items 25 and 26)

Scope: material added or changed under revision_notes.md items 25–26 only. Paper: `docs/writeups/paper_rev3.tex` (abstract, §1, §5 B1c/B1d/B1e paragraphs, §6, §7, Conclusion, Appendix B, Appendix C, Table 3) and the three appendix table files. Sources: `results/rev3/b1d_randk23_summary.json`, `b1e_draws_summary.json`, `b1c_fdr_six_family.json`, `paper_appendix_tables.json`, `paper_revision_stats.json`, `b1d_subspace_qwen36-27b.json`, `b1d_subspace_qwen36-27b-randk23.json`, `b1e_footprint_qwen36-27b.json` (plus the ten `b1e_footprint_qwen36-27b-draw{1..10}.json` files the draws summary points to, for the Appendix B gate slopes). Blocked fractions are `-diff / none_slope`; footprint ratios are mean removed_norm over the six emotions divided by the same mean for `emo_all` (mean/mean convention of note 25(o)).

## Table

| # | location | number in text | source field | file value | match |
|---|---|---|---|---|---|
| A1 | abstract | permuted rank-5 frame "blocks three emotions" | paper_revision_stats b1d_27b.bh_q_over_testable.perm_all | afraid 0.001, sad 0.035, angry 0.013; desperate/happy 0.754 | Y |
| A2 | abstract | "same false-discovery threshold as the affect subspace" | ...bh_q_over_testable.sub_all | sad 0.0005, angry 0.0005 (afraid 0.079) | Y |
| A3 | abstract | "half to two-thirds of its size" | b1d_27b.per_emotion.{sad,angry}.arm_vs_none.{perm_all,sub_all}.diff | sad −26.74/−52.43 = 0.51; angry −44.34/−78.15 = 0.57 | Y |
| A4 | abstract | "random 5-frame of half its footprint" | b1d_subspace summary.*.{randsub_all,perm_all}.removed_norm (means) | 3.187/6.662 = 0.48 | Y |
| A5 | abstract | "random 23-frame matched to its footprint" | randk23 per_emotion.*.removed_norm.randsub_k vs ref_perm_all | 6.67–6.87 vs 6.49–6.89 (means 6.78 vs 6.66) | Y |
| A6 | abstract | three controls "inert on every testable emotion" | b1d_27b bh_q randsub_all; randk23 bh_q_over_testable; b1e_27b bh_q permdir_all | min 0.511; min 0.714; all 0.935 | Y |
| A7 | abstract | sad block "holds in all ten draws" | b1e_draws_summary across_draws.sad.emo_all_vs_permdir_all.n_sig_negative | 10 | Y |
| A8 | abstract | sad median 35% | ...median_blocked_fraction | 0.3514 | Y |
| A9 | abstract | sad range 24–56% | ...min / max | 0.2380 / 0.5561 | Y |
| A10 | abstract | afraid 18% | across_draws.afraid.emo_all_vs_permdir_all.median_blocked_fraction | 0.1755 | Y |
| A11 | abstract | afraid 7–23% | ...min / max | 0.0658 / 0.2250 | Y |
| A12 | abstract | afraid "manipulation check passes in only four" | across_draws.afraid.n_mc_steer_sig | 4 | Y |
| I1 | §1 | "sad loses at least a quarter in every one of ten direction draws" | across_draws.sad.emo_all_vs_permdir_all.min (emo_all_vs_none.min) | 0.238 (0.228) | N |
| I2 | §1 | "afraid a fifth at the median" | across_draws.afraid.emo_all_vs_permdir_all.median_blocked_fraction | 0.1755 = 18% | N |
| I3 | §1 | "manipulation check passing in four draws of ten" | across_draws.afraid.n_mc_steer_sig | 4 | Y |
| D1 | §5 B1d | affect subspace blocks sad 54% | b1d_27b sad sub_all diff / none_slope | 52.43/97.52 = 0.538 | Y |
| D2 | §5 B1d | angry 40% | b1d_27b angry sub_all | 78.15/195.99 = 0.399 | Y |
| D3 | §5 B1d | "at q<0.001" (sad, angry) | bh_q_over_testable.sub_all | 0.0005, 0.0005 | Y |
| D4 | §5 B1d | permuted frame "removes 7–52% of the readable affect" | b1d_subspace summary.*.mc_subspace.share_removed.perm_all | 0.065 (afraid) … 0.524 (happy) | Y |
| D5 | §5 B1d | "a third of its span in common with the affect frame" | not in any listed source file | — | unverified |
| D6 | §5 B1d | permuted frame blocks afraid 23% | b1d_27b afraid perm_all | 49.52/211.77 = 0.234 | Y |
| D7 | §5 B1d | sad 27% | b1d_27b sad perm_all | 26.74/97.52 = 0.274 | Y |
| D8 | §5 B1d | angry 23% | b1d_27b angry perm_all | 44.34/195.99 = 0.226 | Y |
| D9 | §5 B1d | q = 0.001, 0.035, 0.013 | bh_q_over_testable.perm_all | 0.001, 0.0353, 0.013 | Y |
| D10 | §5 B1d | permuted frame "dose-0 readout shifts of at most 0.08" | b1d_27b per_emotion.*.dose0_readout_shift.perm_all | max abs 0.0801 (happy) | Y |
| D11 | §5 B1d | "random 5-frame with half its footprint" | as A4 | 0.48 | Y |
| D12 | §5 B1d | "blocks nothing (q ≥ 0.51)" | bh_q_over_testable.randsub_all | min 0.511 | Y |
| D13 | §5 B1d | 23-frame "6.7–6.9 against 6.5–6.9 per position" | randk23 removed_norm.randsub_k / ref_perm_all | 6.67–6.87 / 6.49–6.89 | Y |
| D14 | §5 B1d | "blocked fractions −0.03 to +0.07" | randk23 per_emotion.*.randsub_vs_none.blocked_fraction (testable five) | −0.029, 0.062, 0.027, 0.070, −0.000 | Y |
| D15 | §5 B1d | "q ≥ 0.71" | randk23 bh_q_over_testable | min 0.714 | Y |
| E1 | §5 B1e | cosine 0.004 | unstamped (Appendix B says so) | — | unverifiable |
| E2 | §5 B1e | "0.85× its footprint on average" | b1e_footprint footprint.ratio_permdir_over_emo | 0.8456 | Y |
| E3 | §5 B1e | 0.84 afraid | summary.afraid permdir_all/emo_all removed_norm | 0.536/0.642 = 0.836 | Y |
| E4 | §5 B1e | 0.77 sad | summary.sad | 0.503/0.656 = 0.767 | Y |
| E5 | §5 B1e | "removed at all 51 layers" | direction_stability.gated_layers; summary.*.emo_all.n_layers_ablated | 51; 51 | Y |
| E6 | §5 B1e | "changes no emotion's dose-response" | b1e_27b bh_q permdir_all | all 0.935 | Y |
| E7 | §5 B1e | "removes at most 9% of the readable affect" | summary.*.mc_footprint.share_removed.permdir_all | max 0.0867 (afraid) | Y |
| E8 | §5 B1e | draw 0 "above afraid's ten-draw range" | reference_blocked_fraction 0.248 vs max 0.225 | above | Y |
| E9 | §5 B1e | draw 0 "inside sad's" | 0.291 vs [0.238, 0.556] | inside | Y |
| E10 | §5 B1e | afraid 18% at the median (7–23%) | as A10/A11 | 0.1755, 0.0658–0.2250 | Y |
| E11 | §5 B1e | afraid "significant in nine draws" | across_draws.afraid.emo_all_vs_permdir_all.n_sig_negative | 9 (draw 8 not) | Y |
| E12 | §5 B1e | check "passes in only four" | n_mc_steer_sig | 4 (draws 1, 4, 9, 10) | Y |
| E13 | §5 B1e | "H3 in four draws and instrument_failed in six" | b1e_draws_summary.verdicts | H3: 1,4,9,10; IF: 2,3,5,6,7,8 | Y |
| E14 | §5 B1e | sad 35% (24–56%) | as A8/A9 | 0.3514, 0.2380–0.5561 | Y |
| E15 | §5 B1e | sad "significant in all ten" | n_sig_negative | 10 | Y |
| E16 | §5 B1e | sad check "passing in all ten" | across_draws.sad.n_mc_steer_sig | 10 | Y |
| E17 | §5 B1e | permuted direction "within 7% (afraid)" of baseline | across_draws.afraid.permdir_all_vs_none min/max | −0.039 / +0.065 | Y |
| E18 | §5 B1e | "13% (sad)" | across_draws.sad.permdir_all_vs_none min/max | −0.132 / +0.086 | Y |
| E19 | §5 B1e | top principal direction "5.8× the footprint" | footprint.ratio_pc1_over_emo | 5.844 | Y |
| E20 | §5 B1e | "in footprint (11× against 0.85×)" | perm_all mean/mean; ratio_permdir_over_emo | 11.22; 0.846 | Y |
| E21 | §5 B1e | matched 23-frame "removes as much and blocks nothing" | randk23 removed_norm means; bh_q | 6.78 vs 6.66; q ≥ 0.714 | Y |
| E22 | §5 B1e | "blocking arms ... within one word" of baseline (affect subspace, permuted-label frame, top principal direction) | paper_appendix_tables coherence.{B1d.sub_all,B1d.perm_all,B1e.pc1_all}.n_words.diff_vs_none | −1.19 [−1.83, −0.54]; −0.98 [−1.74, −0.26]; +2.09 [+1.42, +2.75] | N |
| E23 | §5 B1e | "... and 0.1 perplexity units" | coherence.*.perplexity.diff_vs_none | +0.094; +0.122; −3.70 (11.23 vs 14.92) | N |
| C1 | §6 | permuted frame "blocks three emotions at the false-discovery threshold" | as A1 | 3 at q<0.05 | Y |
| C2 | §6 | "half to two-thirds of the affect subspace's size on the two it also blocks" | as A3 | 0.51 (sad), 0.57 (angry) | Y |
| C3 | §6 | top principal direction "blocks angry ... neither of the rank-1 direction's pair" | b1e_27b bh_q pc1_all | angry 0.001; afraid 0.931, sad 0.493 | Y |
| C4 | §6 | "(rank 23) is inert on the five testable emotions" | randk23 bh_q_over_testable | min 0.714 | Y |
| C5 | §6 | "it moves calm, the untestable one" | randk23 calm randsub_vs_none | −30.3 [−55.4, −4.3], sig; testable false | Y |
| L1 | §7 | "ten re-splits for B1e's two emotions only" | n_draws; draws[*].emotions | 10; [afraid, sad] | Y |
| K1 | Conclusion | rank-1 direction "carries a quarter to a half of sad's response in every draw" | across_draws.sad.emo_all_vs_permdir_all min/max (emo_all_vs_none) | 0.238–0.556 (0.228–0.438) | N |
| B1 | App. B (B1c) | six-emotion family afraid q=0.001 | b1c_fdr_six_family q_six_family.afraid | 0.0012 | Y |
| B2 | App. B | sad 0.030 | q_six_family.sad | 0.0296 | Y |
| B3 | App. B | calm 0.011 | q_six_family.calm | 0.0108 | Y |
| B4 | App. B | angry 0.093 | q_six_family.angry | 0.093 | Y |
| B5 | App. B | five-family 0.001, 0.037, --, 0.103 | q_five_family | 0.001, 0.0370, (no calm), 0.1033 | Y |
| B6 | App. B | "changes no verdict" | afraid and sad < 0.05 in both families | yes | Y |
| B7 | App. B (B1d) | 23-frame shifts calm "−30 [−55, −4]" | randk23 calm randsub_vs_none diff, ci | −30.30 [−55.39, −4.26] | Y |
| B8 | App. B | "calm's own slope is +44 [−3, +99]" | randk23 calm none_slope, none_slope_ci | 44.24 [−3.09, 98.91] | Y |
| B9 | App. B (draws) | failing draws' A-span slope "−0.02 to +0.09" | draw{2,3,5,6,7,8}.summary.afraid.mc_steer.slope | −0.021, −0.0002, +0.063, +0.071, +0.087, +0.093 | Y |
| B10 | App. B | "intervals that include zero, two of them by less than 0.01" | ...mc_steer.ci lower bound | draw 2 −0.0076, draw 6 −0.0040; others −0.037 to −0.119 | Y |
| B11 | App. B | "(four H3, six instrument_failed)" | verdicts | 4 / 6 | Y |
| B12 | App. B | "significantly negative in nine draws" | n_sig_negative | 9 | Y |
| B13 | App. B | "two of the twenty emotion-draws" | afraid permdir_vs_none n_sig_negative 1 + sad n_sig_positive 1 | 2 | Y |
| B14 | App. B | "afraid, draw 6, +6.5%" | draws[seed 6].afraid.contrasts.permdir_all_vs_none | +0.0655, sig | Y |
| B15 | App. B | "sad, draw 3, −13%" | draws[seed 3].sad.contrasts.permdir_all_vs_none | −0.132, sig | Y |
| B16 | App. B Table 4 caption | "split seeds 1–10" | draws[*].split_seed | 1..10 | Y |
| T5-0 | Table 5 caption | "Calm (u) is untestable" | per_emotion.calm.testable | false | Y |
| T5-1 | Table 5 none row | +84 [+59,+111]; +212 [+169,+253]; +48 [+8,+88]; +44 [−3,+99]; +98 [+64,+132]; +196 [+155,+235] | b1c_27b per_emotion.*.none_slope(_ci) | 84.14 [58.55,111.34]; 211.77 [169.18,252.79]; 47.86 [8.22,87.52]; 44.24 [−3.09,98.91]; 97.52 [64.45,132.07]; 195.99 [154.93,235.44] | Y |
| T5-2 | Table 5 B1c emotion direction hs 13–63 | −7 [−26,+11]; −52 [−66,−37]; +11 [−2,+25]; −43 [−74,−13]; −28 [−58,−0]; −18 [−40,+2] | b1c_27b arm_vs_none.emo_all | −6.73 [−25.79,11.06]; −51.85 [−66.30,−37.48]; 11.27 [−1.84,24.69]; −43.23 [−73.90,−13.20]; −27.72 [−57.85,−0.37]; −17.54 [−40.10,2.38] | Y |
| T5-3 | Table 5 B1c hs 13–42 | −3 [−23,+18]; −58 [−71,−44]; +5 [−10,+20]; −35 [−61,−9]; −32 [−57,−8]; −23 [−44,−4] | arm_vs_none.emo13_42 | −2.64 [−23.28,17.51]; −57.60 [−70.89,−44.49]; 5.12 [−9.65,20.46]; −35.07 [−61.40,−9.14]; −32.11 [−57.35,−8.26]; −23.26 [−44.23,−3.62] | Y |
| T5-4 | Table 5 B1c random direction | −2 [−13,+9]; −7 [−16,+3]; +14 [−0,+28]; −2 [−19,+16]; +4 [−7,+15]; +1 [−14,+15] | arm_vs_none.rand_all | −1.70 [−12.90,9.46]; −6.91 [−16.15,3.42]; 13.52 [−0.14,28.42]; −1.68 [−19.01,16.13]; 3.97 [−6.70,14.502]; 0.53 [−13.66,14.78] | Y |
| T5-5 | Table 5 B1d affect subspace | −14 [−34,+4]; −23 [−45,−0]; +3 [−33,+36]; −58 [−95,−24]; −52 [−77,−27]; −78 [−113,−44] | b1d_27b arm_vs_none.sub_all | −13.76 [−33.57,4.37]; −22.54 [−45.28,−0.41]; 2.53 [−32.61,35.73]; −58.49 [−95.04,−23.52]; −52.43 [−77.44,−27.02]; −78.15 [−113.15,−44.14] | Y |
| T5-6 | Table 5 B1d permuted-label subspace | −2 [−17,+12]; −50 [−70,−27]; +5 [−23,+33]; +11 [−21,+40]; −27 [−49,−5]; −44 [−80,−12] | arm_vs_none.perm_all | −2.23 [−16.56,12.09]; −49.52 [−70.22,−27.33]; 4.93 [−22.77,32.91]; 10.67 [−20.74,40.497]; −26.74 [−49.04,−4.53]; −44.34 [−80.36,−11.90] | Y |
| T5-7 | Table 5 B1d random 5-frame | −4 [−14,+5]; −4 [−17,+9]; +8 [−5,+21]; +5 [−8,+19]; +1 [−13,+16]; −12 [−26,+4] | arm_vs_none.randsub_all | −4.35 [−14.44,4.72]; −4.25 [−17.40,9.05]; 8.26 [−4.53,21.31]; 5.13 [−7.91,18.62]; 1.30 [−13.43,15.997]; −11.68 [−26.30,3.68] | Y |
| T5-8 | Table 5 B1d random 23-frame | −5 [−18,+8]; +6 [−7,+19]; −1 [−21,+18]; −30 [−55,−4]; −7 [−24,+11]; +0 [−22,+22] | randk23 per_emotion.*.randsub_vs_none | −5.22 [−17.80,7.67]; 6.04 [−6.84,19.40]; −1.28 [−20.75,17.64]; −30.30 [−55.39,−4.26]; −6.85 [−24.08,10.65]; 0.03 [−22.33,22.36] | Y |
| T5-9 | Table 5 B1e permuted-label direction | −2 [−12,+8]; +1 [−10,+11]; +5 [−7,+17]; +2 [−11,+14]; +1 [−11,+13]; −1 [−18,+17] | b1e_27b arm_vs_none.permdir_all | −2.06 [−12.36,8.19]; 0.65 [−9.84,10.98]; 4.76 [−7.24,17.28]; 1.82 [−11.20,13.79]; 0.67 [−11.07,12.86]; −0.79 [−18.17,17.04] | Y |
| T5-10 | Table 5 B1e top principal direction | −14 [−35,+7]; +2 [−21,+25]; −14 [−47,+17]; −43 [−91,−0]; +13 [−17,+43]; −120 [−154,−87] | arm_vs_none.pc1_all | −13.53 [−35.06,7.16]; 1.505 [−20.69,25.36]; −14.25 [−47.08,17.05]; −43.46 [−91.13,−0.04]; 12.92 [−17.08,42.57]; −120.38 [−154.11,−86.79] | Y |
| T6-1 | Table 6 B1c none | 46 / 0.3% / 0.3% / 1.00 / 14.92, no dots | coherence.B1c.none | 45.88 / 0.34% / 0.34% / 0.998 / 14.925 | Y |
| T6-2 | Table 6 B1c emotion direction hs 13–63 | 46 / 0.5% / 0.5% / 1.00 / 15.19, no dots | coherence.B1c.emo_all (all CIs cover 0) | 45.51 / 0.48% / 0.48% / 0.998 / 15.189 | Y |
| T6-3 | Table 6 B1c hs 13–42 | 46 / 0.5% / 0.5% / 1.00 / 15.42· | coherence.B1c.emo13_42 (perplexity CI [+0.12,+0.89]) | 45.55 / 0.48% / 0.48% / 0.998 / 15.423, dot on perplexity only | Y |
| T6-4 | Table 6 B1c random direction | 46 / 0.3% / 0.3% / 1.00 / 14.72, no dots | coherence.B1c.rand_all | 46.03 / 0.34% / 0.34% / 0.998 / 14.722 | Y |
| T6-5 | Table 6 B1d none | as T6-1 | coherence.B1d.none | identical to B1c.none | Y |
| T6-6 | Table 6 B1d affect subspace | 45· / 0.5% / 0.5% / 1.00 / 15.02 | coherence.B1d.sub_all (words CI [−1.83,−0.54]) | 44.69 / 0.48% / 0.48% / 0.998 / 15.019, dot on words only | Y |
| T6-7 | Table 6 B1d permuted-label subspace | 45· / 0.3% / 0.3% / 1.00· / 15.05 | coherence.B1d.perm_all (words CI [−1.74,−0.26]; distinct2 CI [−0.0010,−0.0001]) | 44.89 / 0.34% / 0.34% / 0.997 / 15.047, dots on words and distinct-2 | Y |
| T6-8 | Table 6 B1d random 5-frame | 46· / 0.4% / 0.4% / 1.00 / 14.46· | coherence.B1d.randsub_all (words CI [+0.001,+0.77]; ppl CI [−0.76,−0.18]) | 46.27 / 0.38% / 0.38% / 0.998 / 14.459, dots on words and perplexity | Y |
| T6-9 | Table 6 B1e none | as T6-1 | coherence.B1e.none | identical | Y |
| T6-10 | Table 6 B1e permuted-label direction | 46 / 0.4% / 0.4% / 1.00 / 14.87, no dots | coherence.B1e.permdir_all | 45.67 / 0.43% / 0.43% / 0.998 / 14.867 | Y |
| T6-11 | Table 6 B1e top principal direction | 48· / 0.0%· / 0.0%· / 1.00· / 11.23· | coherence.B1e.pc1_all (all five CIs exclude 0) | 47.97 / 0.00% / 0.00% / 0.996 / 11.226 | Y |
| T6-12 | Table 6 B1d-23 none | as T6-1 | coherence.B1d-23.none | identical | Y |
| T6-13 | Table 6 B1d-23 random 23-frame | 46 / 0.3% / 0.3% / 1.00 / 14.80, no dots | coherence.B1d-23.randsub_all | 45.65 / 0.34% / 0.34% / 0.998 / 14.801 | Y |
| T6-c1 | Table 6 caption | "72 (emotion, dose, replicate) arm-rows" | coherence.*.*.n_cells | 72 | Y |
| T6-c2 | Table 6 caption | blocking arms "differ from baseline by one word" | sub_all, perm_all n_words | 45 vs 46 as printed (raw −1.19, −0.98) | Y |
| T6-c3 | Table 6 caption | "and by 0.1 in perplexity" | sub_all, perm_all perplexity | 15.02−14.92 = 0.10; 15.05−14.92 = 0.13 (raw +0.094, +0.122) | N |
| T6-c4 | Table 6 caption | top principal direction "lowers perplexity to 11.2 and lengthens replies" | B1e.pc1_all perplexity mean; n_words diff, ci | 11.23; +2.09 [+1.42, +2.75] | Y |
| T6-c5 | Table 6 caption | "All 27B runs share the none arm bit for bit" | the four none rows | identical means in all five metrics | Y |
| T3-1 | Table 3 | affect subspace ratio 11.6 | mean sub_all / mean emo_all removed_norm | 6.914/0.594 = 11.64 | Y |
| T3-2 | Table 3 | permuted-label subspace 11.2 | mean perm_all / mean emo_all | 6.662/0.594 = 11.22 | Y |
| T3-3 | Table 3 | random 5-frame 5.4 | mean randsub_all / mean emo_all | 3.187/0.594 = 5.37 | Y |
| T3-4 | Table 3 | random 23-frame 11.4 (†, B1d emotion direction) | randk23 mean randsub_all / B1d mean emo_all | 6.780/0.594 = 11.42 | Y |
| T3-5 | Table 3 | permuted-label direction 0.85 (afraid 0.84, sad 0.77) | b1e footprint.ratio_permdir_over_emo; per-emotion | 0.846; 0.836; 0.767 | Y |
| T3-6 | Table 3 | top principal direction 5.8 | footprint.ratio_pc1_over_emo | 5.844 | Y |
| T3-7 | Table 3 | absolute footprints 0.46–0.66 / 6.9 / 6.5–6.9 / 3.1–3.2 / 6.7–6.9 / 0.50 / 3.47 | removed_norm per arm | 0.463–0.659 / 6.87–6.94 / 6.49–6.89 / 3.10–3.24 / 6.67–6.87 / 0.502 / 3.471 | Y |
| T4-1 | Table 4 afraid, emotion − permuted | 0.25 / 0.18 / [0.07, 0.23] / 9/10 / 4/10 | across_draws.afraid.emo_all_vs_permdir_all; n_mc_steer_sig | 0.2479 / 0.1755 / [0.0658, 0.2250] / 9 / 4 | Y |
| T4-2 | Table 4 afraid, emotion − none | 0.24 / 0.19 / [0.08, 0.23] / 10/10 | across_draws.afraid.emo_all_vs_none | 0.2448 / 0.1895 / [0.0814, 0.2274] / 10 | Y |
| T4-3 | Table 4 afraid, permuted − none | 0.00 / 0.02 / [−0.04, 0.07] / 1/10 | across_draws.afraid.permdir_all_vs_none | −0.0031 / 0.0184 / [−0.0388, 0.0655] / 1 | Y |
| T4-4 | Table 4 sad, emotion − permuted | 0.29 / 0.35 / [0.24, 0.56] / 10/10 / 10/10 | across_draws.sad.emo_all_vs_permdir_all; n_mc_steer_sig | 0.2912 / 0.3514 / [0.2380, 0.5561] / 10 / 10 | Y |
| T4-5 | Table 4 sad, emotion − none | 0.28 / 0.35 / [0.23, 0.44] / 10/10 | across_draws.sad.emo_all_vs_none | 0.2843 / 0.3462 / [0.2280, 0.4375] / 10 | Y |
| T4-6 | Table 4 sad, permuted − none | −0.01 / −0.01 / [−0.13, 0.09] / 0/10 | across_draws.sad.permdir_all_vs_none | −0.0069 / −0.0068 / [−0.1322, 0.0862] / 0 sig-negative (1 sig-positive, draw 3) | Y |

Items checked: 118 (114 match, 2 unverifiable from the listed sources, 6 mismatched cells in 5 places).

## Mismatches, ranked

1. **§5, B1e paragraph, coherence sentence** (E22, E23). "The blocking arms leave B's replies as long and as fluent as baseline within one word and 0.1 perplexity units (Table 6): the arms that block (affect subspace, permuted-label frame, top principal direction) ..." The sentence counts the top principal direction among the blocking arms, and that arm lengthens replies by 2.1 words [1.4, 2.7] and lowers perplexity by 3.70 (11.23 against 14.92); the affect subspace is 1.19 words shorter [−1.83, −0.54], and the permuted frame's perplexity is +0.12 (0.13 at the table's precision). Corrected: "The affect subspace and the permuted-label frame leave B's replies within 1.2 words and 0.13 perplexity units of baseline (Table 6); the top principal direction lengthens replies by two words and lowers perplexity by 3.7."
2. **Table 6 caption** (T6-c3). "differ from baseline ... by 0.1 in perplexity": the permuted-label subspace differs by 0.13 at the printed precision (15.05 vs 14.92; raw +0.122), the affect subspace by 0.10 (raw +0.094). Corrected: "by at most 0.13 in perplexity" (or "by 0.1 and 0.13").
3. **§1** (I1). "only sad loses at least a quarter in every one of ten direction draws": the smallest draw is 0.238 against the permuted control (0.228 against baseline). Corrected: "at least 24%" (or "about a quarter or more").
4. **Conclusion** (K1). "carries a quarter to a half of sad's response in every draw": the range is 0.24–0.56 against the permuted control (0.23–0.44 against baseline); the top exceeds a half. Corrected: "24–56% of sad's response".
5. **§1** (I2). "afraid a fifth at the median": the median is 0.175 (18%), closer to a sixth. Corrected: "18% at the median".

Unverifiable from the listed sources (not counted as mismatches): D5 "a third of its span in common with the affect frame" (no perm-vs-affect frame overlap statistic in b1d_subspace_qwen36-27b.json, its randk23 companion, or paper_revision_stats.json); E1 cosine 0.004 (Appendix B already states it is unstamped).

Outside scope, noticed in passing: the caption of `tab:permnull` (Appendix C) still contains the literal placeholder "PERMREADING".

---
Applied 2026-09-06 (after this check): items 1–5 as proposed (§5 coherence sentence now gives 1.2 words / 0.13 perplexity for the two blocking frames and +2 words / −3.7 perplexity for the top principal direction; Table 6 caption "at most 1.2 words and 0.13"; §1 "at least 24%" and "18% at the median"; Conclusion "24–56% of sad's response across the ten draws"). The unverifiable "a third of its span in common" is replaced by its source quantity, mean principal cosine 0.32–0.35 to the affect frame (`b1d.frame_geometry`, `results/rev3/b1d_frame_geometry.json` 13/30/43/63.perm_vs_true). PERMREADING is the slot for the permutation null, filled when the four runs land.

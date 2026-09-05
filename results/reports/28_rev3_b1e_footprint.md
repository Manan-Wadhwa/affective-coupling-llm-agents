# 28 · Rev-3 B1e — a fitted, label-free rank-1 control: is B1c's afraid/sad blocking emotion-specific? (verdict H3)

**Files:** `results/rev3/b1e_footprint_qwen36-27b.json` (288 arm-rows; provenance-stamped; pre-registration stamped inside: `PREREG_B1e.md` sha da043b93…, found = true; code_sha e2008fad, core c1d065ca), `b1e_cells_qwen36-27b.json` · **Script:** `src/rev3/b1e_footprint.py` (commit b9e4904 after the verifier's three fixes) · **Pre-registration:** `PREREG_B1e.md` at 0228b41 with addenda 1 (motivation corrected after report 27) and 2 (measured footprint, H3 caveat) committed before launch; run 09:38:26Z → 11:18:49Z on box 5. · **Model:** Qwen3.6-27B rev 6a9e13bd; B1c's probe pool.

## What was run
Four arms × four doses × three reps × six emotions over 29 scenarios: `none`; `emo_all` (B1c's rank-1 difference-of-means direction, hs 13–63); `permdir_all` (the same estimator on permuted labels, the steered emotion's class index, one permutation per emotion and rep, hs 13–63 — fitted, label-free, cosine 0.004 to the emotion direction); `pc1_all` (the top principal direction of A's span per layer, no label). Decision per PREREG §2 on the two pre-specified emotions afraid and sad: H3 if `emo_all − permdir_all` is significantly negative with blocked fraction ≥ 0.10 for both.

## What the file shows
**Verdict: H3.** Footprint (removed norm per position summed over 51 layers): emo 0.59, permdir 0.50 (0.85×), pc1 3.47 (5.8×). MC-steer 6/6; permdir's dose-0 readout shift < 0.01 for every emotion; no quality exclusions.

| emotion | none | emo_all | permdir_all | pc1_all | emo − permdir | permdir − none | pc1 − none | share of readable affect removed: emo / permdir / pc1 |
|---|---|---|---|---|---|---|---|---|
| desperate | +84 | +77 | +82 | +71 | -4.7 [-22.4, +12.7] | -2.1 [-12.4, +8.2] | -13.5 [-35.1, +7.2] | 0.16 / 0.00 / 0.01 |
| afraid | +212 | +160 | +212 | +213 | -52.5 [-68.9, -35.8] | +0.7 [-9.8, +11.0] | +1.5 [-20.7, +25.4] | 0.79 / 0.09 / 0.79 |
| happy | +48 | +59 | +53 | +34 | +6.5 [-10.2, +23.5] | +4.8 [-7.2, +17.3] | -14.2 [-47.1, +17.0] | 0.42 / -0.00 / 1.44 |
| calm (untestable) | +44 | +1 | +46 | +1 | -45.1 [-72.2, -19.2] | +1.8 [-11.2, +13.8] | -43.5 [-91.1, -0.0] | 0.27 / -0.00 / 1.54 |
| sad | +98 | +70 | +98 | +110 | -28.4 [-54.2, -7.0] | +0.7 [-11.1, +12.9] | +12.9 [-17.1, +42.6] | 0.75 / -0.01 / 0.71 |
| angry | +196 | +178 | +195 | +76 | -16.8 [-37.3, +2.9] | -0.8 [-18.2, +17.0] | -120.4 [-154.1, -86.8] | 0.39 / 0.01 / 0.87 |

Blocked fractions of the none slope for `emo_all − permdir_all`: afraid 0.25 [0.17, 0.33], sad 0.29 [0.07, 0.56] (BH-FDR q 0.001 and 0.028); desperate 0.06, happy −0.14, angry 0.09 null. `none` and `emo_all` reproduce B1c and B1d bit for bit.

## What can be inferred
- Against a control fitted by the same estimator on the same data with the labels destroyed, the emotion direction's blocking of afraid and sad is emotion-specific (H3). [supported by file]
- The fitted label-free direction is inert for every emotion and removes no readable affect; it is the control B1d's rank-5 permuted frame was not (that frame overlapped the affect frame by a third). [supported by file]
- Footprint does not explain the afraid/sad block: the control removes 0.85× the norm, and the top principal direction, at 5.8×, leaves afraid and sad untouched while crushing angry (−120, 61% of its slope) and calm. Readable-affect removal and transfer blocking dissociate: pc1 removes 79% of afraid's readable affect without blocking it. [supported by file]
- Retrospective reading of B1d: the permuted rank-5 frame blocked afraid and sad through its overlap with the affect frame and angry through high-variance removal. [inference from B1d's geometry and this file]
- Scope: two pre-specified emotions; sad's interval is wide (0.07–0.56); the readout on A's span is still forced-choice; one pool and one direction estimate. [scope]

## Status
Complete; pre-registered decision H3. Discharges the caveat B1d placed on B1c: the rank-1 afraid/sad blocking is emotion-specific. Feeds RESEARCH_PLAN §13.3 and the deadline paper's claim 5.

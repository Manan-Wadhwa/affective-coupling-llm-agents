# 30 · B1e's design on Llama-3-8B-abliterated — H3 with a footprint-matched control

**Files:** `results/rev3/b1e_footprint_llama3-abl.json` (288 arm-rows; provenance-stamped, revision dd67dd05; PREREG_B1e stamped inside, sha da043b93…), `b1e_cells_llama3-abl.json` · **Script:** `src/rev3/b1e_footprint.py --model failspy/Llama-3-8B-Instruct-abliterated` on box 6, 11:30–11:58Z (1679 s). · **Status:** a replication of the pre-registered B1e design on a second model (PREREG_B1e names the 27B; the rule and thresholds apply unchanged); the pool and directions are the 8B B1c run's (report 29).

## What the file shows
**Verdict: H3.** Footprint (removed norm per position, layer mean): emo 0.129, permdir 0.124 (**0.96×**), pc1 0.188 (1.45×; the 8B's top principal direction is unstable across halves, |cos| median 0.08, so pc1 is a weak control here). MC-steer 6/6; permdir's dose-0 shift ≤ 0.02 everywhere; no quality exclusions.

| emotion | none | emo_all | permdir_all | pc1_all | emo − permdir | permdir − none | share of readable affect removed: emo / permdir |
|---|---|---|---|---|---|---|---|
| desperate | +57 | +54 | +57 | +48 | -3.0 [-13.2, +7.1] | -0.4 | 0.47 / 0.00 |
| afraid | +122 | +111 | +129 | +128 | -17.4 [-32.7, -1.7] | +7.1 | 1.07 / -0.02 |
| happy | +66 | +59 | +66 | +57 | -7.1 [-22.5, +7.2] | +0.4 | 0.17 / 0.00 |
| calm (untestable) | +36 | +32 | +34 | +49 | -1.6 [-21.2, +19.7] | -2.0 | 0.52 / 0.05 |
| sad | +102 | +75 | +112 | +91 | -36.8 [-57.1, -16.7] | +9.6 | 1.36 / 0.01 |
| angry | +102 | +94 | +97 | +98 | -3.1 [-22.6, +16.3] | -4.8 | 0.07 / -0.02 |

Blocked fractions of the none slope: sad 0.36 [0.16, 0.56] (BH-FDR q 0.002; secondary readout on B's reply agrees, −0.075, significant); afraid 0.14 [0.01, 0.27] (q 0.08; secondary null); desperate, happy, angry null.

## What can be inferred
- On the 8B the fitted label-free control is footprint-matched (0.96×) and inert, and the emotion direction still blocks sad by a third against it: for sad, on this model, the footprint caveat of report 28 is discharged. [supported by file]
- Afraid passes the pre-registered point rule (H3 needs both emotions) but its range touches 0.01 and it does not clear the descriptive FDR line; call it marginal. [supported by file]
- Across the two models: sad blocks against a fitted label-free control on both (29% / 36%); afraid on both under the rule (25% / 14%) but marginal on the 8B; permdir is inert on both. [supported by files]
- The pc1 arm says little on the 8B (unstable direction, small footprint). [scope]

## Status
Complete. Registry `b1e8.*`; delta critique appended below when it lands.

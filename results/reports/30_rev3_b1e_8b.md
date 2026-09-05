# 30 · B1e's design on Llama-3-8B-abliterated — H3 with a footprint-matched control

**Files:** `results/rev3/b1e_footprint_llama3-abl.json` (288 arm-rows; provenance-stamped, revision dd67dd05; PREREG_B1e stamped inside, sha da043b93…), `b1e_cells_llama3-abl.json` · **Script:** `src/rev3/b1e_footprint.py --model failspy/Llama-3-8B-Instruct-abliterated` on box 6, 11:30–11:58Z (1679 s). · **Status:** a replication of the pre-registered B1e design on a second model (PREREG_B1e names the 27B; the rule and thresholds apply unchanged); the pool and directions are the 8B B1c run's (report 29).

## What the file shows
**Verdict: H3.** Footprint (removed norm per position, layer mean): emo 0.129, permdir 0.124 (**0.96×**), pc1 0.188 (1.45×; the 8B's top principal direction is unstable across halves, |cos| median 0.08, so pc1 is a weak control here). MC-steer 6/6; permdir's dose-0 shift ≤ 0.02 everywhere; permdir − none diffs within [−4.8, +9.6] (CIs within [−22.5, +22.6]); its readable-affect shares ≤ 0.051; no quality exclusions.

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
- Under the rule the verdict is H3, and sad's contrast against the control is large. But the control is matched only on average: per emotion it removes 0.86× for sad (the 27B's ratio again), 1.17× for afraid. Both directions remove close to the isotropic norm (0.136), and B1c's random direction on this model removed more (0.168) and was equally inert, with emotion-minus-random −18.1 / −31.5 reproducing this run's −17.4 / −36.8. On the 8B, then, B1e adds no discriminating power over B1c, and the footprint caveat is not discharged. [supported by file; corrected after the critique]
- Afraid passes the pre-registered point rule but its emotion direction does not block against baseline here (emo_all − none −10.2 [−25.0, +4.8]); the control drifts upward (+7.1), so the pass is a contrast against a raised control; range from 0.01, q 0.08, secondary null. Call afraid unreplicated. About a third of sad's −36.8 is likewise control drift over its −27.2 against baseline. [supported by file; the critique's reading adopted]
- Across the two models the strong emotion swaps (27B afraid q 0.001 / sad 0.028; 8B sad 0.002 / afraid 0.08). Only sad replicates on both models on both outcomes. [supported by files]
- The pc1 arm says little on the 8B (unstable direction, small footprint). This run is an out-of-prereg replication: PREREG_B1e fixes the 27B, layer 42 and hs 13–63, and the stamped document does not cover focus 21 / hs 6–31. The bridge arm inherits the 8B B1c run, whose own verdict was instrument_failed. emo_all drives A's readout below the unsteered floor for afraid and sad (shares 1.07, 1.36); afraid's steering span is the weakest in either model. [scope]

## Status
Complete. Registry `b1e8.*`; delta critique appended below when it lands.


### Delta critique — B1e on the 8B (the same blind reviewer as report 28)

# B1e on Llama-3-8B-abliterated — delta critique

**1. Numbers.** C1 verifies exactly (`footprint` 0.1294/0.1236/0.1880, ratios 0.9551/1.4529; MC-steer 6/6; max permdir dose-0 shift +0.0178; 0 exclusions; gate min 0.822; BH arithmetic correct). C3 verifies; sad's `..._secondary` is −0.0752 [−0.1586, **−0.0001**], knife-edge. C5 verifies (`pc1_dir_read_abs_cos` median 0.0749; pc1 significant only for desperate, −9.6 [−17.6,−1.6], unadjusted across ~30 contrasts). Two C2 errors: permdir−none *diffs* span [−4.8,+9.6] and *CIs* [−22.5,+22.6] — "[−12,+14]" is neither; calm's permdir share is **0.0506**, above the stated ≤0.05.

**The finding neither C2 nor C3 states:** `emo_all_vs_none` is **not significant for afraid** (−10.2 [−25.0,+4.8]), and permdir−none drifts *upward* (+7.1 afraid, +9.6 sad). So afraid's entire H3 pass is a contrast against a control that raised the slope, with no demonstrated block against baseline; sad's −36.8 is ~35% control drift over its −27.2 vs `none`.

**2. C4 not licensed.** The pooled 0.955 averages opposite-signed mismatches: per-emotion `permdir_all.removed_norm / emo_all` is **sad 0.860** (the 27B's 0.85 again), afraid 1.165, calm 0.774. Sad — the emotion C4 claims discharged — is not matched. removed_norm is a per-layer *mean*, so equal means can hide unequal per-layer profiles, which are not recorded. Nor does matching norms make inertness informative: emo (0.129) and permdir (0.124) both sit near isotropic (8.71/√4096 ≈ 0.136), while b1c-llama's random direction removed 0.168 (1.30×) and was *equally* inert — its emo−rand is −18.1 (afraid) / −31.5 (sad), reproducing B1e's −17.4/−36.8. B1e adds no discriminating power over B1c here.

**3. afraid.** Report per model: it passes the point rule (0.143) but its range starts at 0.014, q 0.08, secondary null, emo−none null. The models *swap* which emotion is strong (27B afraid q 0.001 / sad 0.028; 8B sad 0.002 / afraid 0.08) — instability, not convergence. Only **sad** replicates on both models on both outcomes; call afraid unreplicated.

**4. 8B-only.** `share_removed` exceeds 1 for afraid (1.071) and sad (1.358) — emo_all drives A's readout below the unsteered floor; afraid's steering span is 0.0739 with MC-steer 0.066 [0.012,0.120], the weakest manipulation in either model. Pool 1079 (vs 1615), gate min 0.822.

**5. Defects.** PREREG_B1e §3 fixes the 27B, layer 42, hs 13–63; this run is focus 21, hs 6–31 — the stamped sha da043b93 attests a document that does not cover it, so label it an out-of-prereg replication. The bridge arm inherits b1c-llama, whose own verdict was `instrument_failed`. `none`/`emo_all` are bit-identical to b1c-llama (144/144 cells). The "summed over the ablated layers" mislabel and the missing permdir geometry carry over.

### Reconciliation (author)

- **C2 corrected** (diff and CI ranges; calm's share 0.051).
- **C4 withdrawn:** matched on average only; sad's per-emotion ratio is 0.86×; both directions sit at the isotropic norm and B1c's random direction already reproduced the contrasts, so this run adds no discriminating power on the 8B.
- **afraid reclassified as unreplicated:** no block against baseline on the 8B, a raised control, and the models swap which emotion is strong. Only sad replicates on both models on both outcomes.
- **Labelled an out-of-prereg replication**; the stamped pre-registration does not cover the 8B's layers.
- Registry `b1e8.*` rewritten; the proposal, README and draft now say "sad replicates; afraid does not".

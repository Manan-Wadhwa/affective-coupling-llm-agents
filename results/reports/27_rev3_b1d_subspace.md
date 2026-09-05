# 27 · Rev-3 B1d — pre-registered rank-5 subspace ablation: the instrument failed, and the control taught something

**Files:** `results/rev3/b1d_subspace_qwen36-27b.json` (360 arm-rows; provenance-stamped, pre-registration stamped inside it: `docs/planning/PREREG_B1d.md` sha 784cd649…, found = true; code_sha 4ed3821a, core c1d065ca), `b1d_cells_qwen36-27b.json` · **Script:** `src/rev3/b1d_subspace.py` (commit 940e418, verified before launch: report in the night log) · **Pre-registration:** `PREREG_B1d.md` committed c3a38b4 (06:1xZ) with addendum fd20f51 (06:30Z); run 06:32:18Z → 08:31:11Z (1 h 59 min on box 5). · **Model:** Qwen3.6-27B rev 6a9e13bd; B1c's probe pool (sha 70129f86…).

## What was run
Six emotions × five arms × four doses × three reps over 29 scenarios. Arms: `none`; `emo_all` (B1c's rank-1 direction, hs 13–63); `sub_all` (rank-5 class-mean affect subspace from the DIR half, hs 13–63); `perm_all` (the same estimator with the emotion labels permuted, one permutation per emotion and rep); `randsub_all` (Gaussian orthonormal 5-frame per layer). Gates: dom split-half ≥ 0.80 at every ablated layer (0.88–0.93) and subspace principal-angle cosine DIR vs READ ≥ 0.70 (0.856–0.931). Decision contrast `sub_all − perm_all`; instrument check (§4.2): `sub_all` must remove ≥ 80% of the floor-corrected readable affect on A's span at the top dose, controls ≤ 20%.

## What the file shows

**Verdict: instrument_failed** (PREREG §2): `sub_all` passes §4.2 for 3 of 6 emotions; the permuted control is not inert for 4 of 6; `randsub_all` is inert everywhere. MC-steer 6/6; no quality exclusions.

| emotion | none | emo_all | sub_all | perm_all | randsub_all | share removed: sub / perm / rand | sub − perm | perm − none | sub − emo | dose-0 readout none → sub |
|---|---|---|---|---|---|---|---|---|---|---|
| desperate | +84 | +77 | +70 | +82 | +80 | 0.11 / 0.07 / 0.00 | −11.5 [−28.0, +5.1] | −2.2 [−16.6, +12.1] | −7.0 [−24.7, +11.3] | 0.108 → 0.500 |
| afraid | +212 | +160 | +189 | +162 | +208 | 0.61 / 0.07 / 0.01 | **+27.0 [+3.8, +48.2]** | **−49.5 [−70.2, −27.3]** | **+29.3 [+8.6, +49.6]** | 0.250 → 0.291 |
| happy | +48 | +59 | +50 | +53 | +56 | 1.25 / 0.52 / 0.01 | −2.4 [−41.8, +36.0] | +4.9 [−22.8, +32.9] | −8.7 [−36.0, +17.1] | 0.287 → 0.120 |
| calm (untestable) | +44 | +1 | −14 | +55 | +49 | 1.19 / 0.39 / 0.01 | −69.2 [−104.8, −32.6] | +10.7 [−20.7, +40.5] | −15.3 [−45.8, +16.4] | 0.277 → 0.024 |
| sad | +98 | +70 | +45 | +71 | +99 | 1.06 / 0.50 / 0.03 | **−25.7 [−49.0, −3.3]** | **−26.7 [−49.0, −4.5]** | −24.7 [−50.7, +2.6] | 0.039 → 0.013 |
| angry | +196 | +178 | +118 | +152 | +184 | 0.38 / 0.30 / −0.01 | **−33.8 [−65.3, −4.5]** | **−44.3 [−80.4, −11.9]** | **−60.6 [−95.6, −27.4]** | 0.038 → 0.052 |

Slopes are present-emotion projections of B's reply per unit dose; contrasts are scenario-blocked and dose-paired. Descriptive median blocked fraction of `sub_all − perm_all` over the five testable emotions: 0.137 [−0.027, 0.286]. Removed norm per masked position summed over layers: `sub_all` 6.9, `perm_all` 6.5–6.9, `randsub_all` 3.1–3.2, `emo_all` 0.46–0.66. The `none` and `emo_all` arms reproduce B1c's slopes bit for bit (all twelve). Secondary readout on B's reply: `sub_all − perm_all` significant for sad (−0.05) and angry (−0.11) only.

## What can be inferred
- Under the pre-registered rule the outcome is instrument_failed: the rank-5 class-mean subspace does not remove the readable affect on A's span for desperate, afraid and angry, and where it "removes" it (happy, calm, sad) it pushes the readout below the unsteered floor. [supported by file]
- The instrument and the readout share a target: with the subspace gone, the forced-choice readout changes at dose 0, where nothing is steered (desperate 0.11 → 0.50, calm 0.28 → 0.02). The §4.2 share, which uses the none-arm floor, is therefore not interpretable for this arm; the instrument failure is partly the check's, not only the ablation's. [supported by file; reading ours]
- The label-permuted control is not inert. It removes 30–52% of readable affect for four emotions and, more importantly, reduces B's dose-response significantly for afraid (−49.5), sad (−26.7) and angry (−44.3), as much as the rank-1 emotion direction did. Both data-derived frames remove ten times the norm the direction removed and twice what a random 5-frame removes, so they lie along the residual stream's high-variance directions. Large-footprint ablations of A's span reduce transfer without emotion information. [supported by file; the high-variance reading is inferred from the norms]
- The emotion-specific increment over that footprint-matched control is small: median 0.14 with a CI that includes zero; afraid goes the other way (the affect subspace blocks less than the permuted one, and less than the rank-1 direction). H2 is not supported; H2′ cannot be declared because the instrument failed. [supported by file]
- Consequence for B1c: its afraid/sad rank-1 blocking (footprint 0.66, random control inert at 1.2) stands as the cleanest evidence, but B1d shows that footprint alone can block, so B1c's specificity now needs a footprint-matched rank-1 control (the top principal direction of A's span, or a permuted-label rank-1 direction) before it is called emotion-specific. [inference; needs that control]
- What carries the transfer remains open; the linear rank-5 class-mean subspace is not a clean handle on it, and the readout used as the manipulation check is not independent of the subspace. [scope]

## Status
Complete; pre-registered decision instrument_failed. Feeds RESEARCH_PLAN §13.3 (Paper B): the subspace branch is closed as "no clean instrument", the footprint confound is new, and the next controlled step is a footprint-matched rank-1 control for B1c.

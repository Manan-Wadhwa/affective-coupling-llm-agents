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

Slopes are present-emotion projections of B's reply per unit dose; contrasts are scenario-blocked and dose-paired. Descriptive median blocked fraction of `sub_all − perm_all` over the five testable emotions: 0.137 [−0.027, 0.286] (a scenario-only bootstrap; the emotion-to-emotion spread is not in it). The pre-registered BH-FDR q for the three per-emotion contrasts whose unadjusted CIs exclude zero (afraid, sad, angry) is 0.0507 each, so none clears 0.05 under the registered rule. Removed norm per masked position summed over layers: `sub_all` 6.9, `perm_all` 6.5–6.9, `randsub_all` 3.1–3.2, `emo_all` 0.46–0.66. The `none` and `emo_all` arms reproduce B1c's slopes bit for bit (all twelve). Secondary readout on B's reply: `sub_all − perm_all` significant for sad (−0.05) and angry (−0.11) only.


## Frame geometry (post hoc, `src/rev3/b1d_frame_geometry.py`, cached probe features, B1d's split and seeds)
| hidden state | affect frame vs top-5 PCs | affect frame vs top-20 | perm frame vs top-5 | perm vs top-20 | perm vs affect frame | variance captured: affect / perm / top-5 / random-5 |
|---|---|---|---|---|---|---|
| 13 | 0.71 | 0.94 | 0.38 | 0.61 | 0.32 | 0.164 / 0.068 / 0.219 / 0.0010 |
| 30 | 0.77 | 0.94 | 0.38 | 0.62 | 0.33 | 0.178 / 0.069 / 0.219 / 0.0010 |
| 43 | 0.81 | 0.95 | 0.39 | 0.63 | 0.34 | 0.199 / 0.077 / 0.235 / 0.0010 |
| 63 | 0.74 | 0.96 | 0.38 | 0.61 | 0.35 | 0.205 / 0.075 / 0.233 / 0.0011 |

The affect subspace is largely the pool's top-variance subspace (cosine 0.71–0.81 to the top-5 PCs, 16–21% of the variance). The permuted frame is a fitted high-variance frame (7% of the variance, 65× a random frame) that overlaps the affect frame by about a third — not the same subspace, not orthogonal to it; `sub_all − perm_all` therefore subtracts part of what it should isolate and is biased toward zero. The critic's recomputation of the §4.2 share with each arm's own dose-0 floor (0.58 / 0.89 / 0.88 / 0.67 / 1.00 / 0.40) leaves three failures, so the verdict is robust while the per-emotion attribution is not.

## What can be inferred
- Under the pre-registered rule the outcome is instrument_failed: the rank-5 class-mean subspace does not remove the readable affect on A's span for desperate, afraid and angry, and where it "removes" it (happy, calm, sad) it pushes the readout below the unsteered floor. [supported by file]
- The instrument and the readout share a target: with the subspace gone, the forced-choice readout changes at dose 0, where nothing is steered (desperate 0.11 → 0.50, calm 0.28 → 0.02). The §4.2 share, which uses the none-arm floor, is therefore not interpretable for this arm; the instrument failure is partly the check's, not only the ablation's. [supported by file; reading ours]
- The label-permuted control is not inert. It removes 30–52% of readable affect for four emotions and, more importantly, reduces B's dose-response significantly for afraid (−49.5), sad (−26.7) and angry (−44.3), as much as the rank-1 emotion direction did. Both data-derived frames remove ten times the norm the direction removed and twice what a random 5-frame removes, so they lie along the residual stream's high-variance directions. Large-footprint ablations of A's span reduce transfer without emotion information. [supported by file; the high-variance reading is inferred from the norms]
- The emotion-specific increment over that footprint-matched control is small: median 0.14 with a CI that includes zero; afraid goes the other way (the affect subspace blocks less than the permuted one, and less than the rank-1 direction). H2 is not supported; H2′ cannot be declared because the instrument failed. [supported by file]
- Consequence for B1c, corrected by the critique: B1c's random direction already removed twice the norm the emotion direction did and was inert, so norm alone is not what makes a control bite; what distinguishes `perm_all` from `randsub_all` is that it is *fitted* (it captures 65× a random frame's variance and overlaps the affect frame by a third, see the geometry below). B1c's afraid/sad specificity therefore needs a fitted, label-free rank-1 control — the same estimator on permuted labels — which is what PREREG_B1e runs. [inference; needs that control]
- What carries the transfer remains open; the linear rank-5 class-mean subspace is not a clean handle on it, and the readout used as the manipulation check is not independent of the subspace. [scope]

## Status
Complete; pre-registered decision instrument_failed. Feeds RESEARCH_PLAN §13.3 (Paper B): the subspace branch is closed as "no clean instrument", the footprint confound is new, and the next controlled step is a footprint-matched rank-1 control for B1c.


### Blind critique — B1d (independent agent; saw only the driver, the core functions it calls, both pre-registrations and the two result files)

# Blind review — B1d rank-5 subspace

## 1. Numbers

Every number in C1–C6 checks out except one. Verdict, counts (3 sub-fails, 4 non-inert
controls, MC-steer 6/6, 0 exclusions, 360 rows), all four shares per emotion, median 0.13696
CI [−0.0274, +0.2859] with its five per-emotion fractions (I re-ran `_decide` from `rows` and
reproduced it exactly), every contrast, removed norm and dose-0 readout. C6: `none`/`emo_all`
are byte-identical to B1c — not just slopes but `B_present_e`, `ctx_readout_e`, `perplexity`,
the gate table, `rms`; prereg sha 784cd649… matches the repo file; driver sha256 864ef287…
is the addendum's value.

**One error.** C1's "subspace gate 0.856–0.906": `direction_stability.subspace.summary` is min
.8558, **median .9201, max .9312**. 0.906 is not any statistic there.

## 2. Rule fidelity

Faithful. Verdict override L611–618, testability L593–595, §4.2 formula L708–721, frames
L271–293, one permutation per cell reused at every layer L1490–1496, fresh random frame per
layer L1506, CRN on part "B" L222, both gates fatal L1417–1461; `Ablate`'s 2-D branch
(acl_core L806–809) is `(h@Q)@Qᵀ`, rank-1 path untouched. Three notes, none decision-changing:
(a) the `permutation_control` pointer names `::perm_frame` (L296–304), which `main` never
calls — L1494 uses `class_mean_frame` on `yperm` directly; the "cannot drift" pointer has
drifted. (b) `blocked_fraction` divides by a single-arm-panel `none` slope while its numerator
comes from a two-arm panel; harmless only because the run is balanced. (c) The addendum
predicted `emo_all` would match B1c "not bit-exactly". It matched bit-exactly. Same-box
determinism is the benign reading, but nothing in the file separates "re-ran, same floats"
from "reused B1c's cells". Confirm from the log.

## 3. The instrument

`perm_all` removes 6.5–6.9 norm units vs `randsub_all`'s 3.1–3.2 — a fitted frame captures
~4.6× the variance an isotropic 5-frame does. That is the file's evidence that a
permuted-label class-mean frame is essentially the pool's **top-variance subspace**: with
labels shuffled the six class means are grand mean + noise, so the SVD of their differences
returns the leading PCs. What would settle it, and is absent: per-layer principal angles
between the perm frame and (i) the DIR half's top-5 PCs, (ii) the true class-mean frame. The
selftest asserts perm-vs-true < 0.5 on a *synthetic* fixture, never on the real pool. If the
overlap is high, `sub_all − perm_all` subtracts part of the effect it should isolate, biased
toward 0 — the direction the data went. Over-strong, not merely footprint-matched.

The dose-0 shift is worse than C4 says. Not floor drift: at α=0 `sub_all` **rewrites** the
readout, (happy .287, calm .277, sad .039, angry .038, afraid .250, desperate .108) → (.120,
.024, .013, .052, .291, **.500**), entropy 1.55→1.26, while `perm_all` barely moves. The
ablation injects a fixed desperate/afraid bias on unsteered text. Readout and ablation share a
target, and §4.2's numerator charges that bias to "affect removed" — which is why happy/calm/
sad exceed 1.0 and desperate collapses to .11. Recomputing with each arm's own floor
(1 − arm_span/none_span): .577/.889/.882/.670/1.004/.403 — still 3 failures, verdict
unchanged, but the failing set becomes desperate/calm/angry. Report the verdict as robust and
the per-emotion attribution as not.

## 4. Statistics

The CI is **scenario-only**: one shared draw (L768), 5 fixed emotions, median across them
(L772). Emotion spread (sd 0.132, range 0.391) matches the half-width 0.157 and is nowhere in
the interval. The median-realising emotion switches across draws (angry 2027, desperate 1772,
sad 761, happy 419, afraid 21); `happy`'s fraction has bootstrap sd 10.4 because its `none`
slope nearly crosses zero — the median absorbs that by luck. Prereg-faithful, but
[−0.027, 0.286] is not "a typical emotion's blocked fraction".

Afraid's anti-blocking is probably real: identical in `sub_all−emo_all` (+29.3) and
`sub_all−perm_all` (+27.0) on the primary outcome, `randsub_all` null, and not a
small-denominator artefact (afraid has the largest `none` slope, 211.8). Multiplicity is the
caveat: 42 uncorrected intervals, and the pre-registered BH-FDR q for afraid, sad and angry is
**0.0507 each — none clears 0.05**. C2 should not call them "significant" without that line.

## 5. Is C7 licensed?

Mostly, with one correction. `perm_all−none` (afraid −49.5, sad −26.7) ≈ `emo_all−none`
(−51.9, −27.7) and `emo_all−perm_all` is null, so rank-1 blocking is indistinguishable from a
same-footprint mislabelled subspace. But B1c already ran a rank-1 control: `rand_all` removed
**1.15–1.26** norm units, ~2× `emo_all`'s 0.46–0.66, and was inert (afraid 204.9 vs none
211.8; sad 101.5 vs 97.5), with `emo_all−rand_all` = −44.9 [−58.0, −31.8]. A norm-matched
rank-1 control exists and `emo_all` beats it. What is missing is a **rank-1 permuted-label**
direction — same estimator, same DIR half, shuffled labels — because the perm-vs-randsub gap
shows fitted-vs-random, not rank, is what makes a control bite. Reword C7; the rest stands.

## 6. Residual defects

`prereg.git_commit` is empty, so the addendum's commit 940e418 is unverifiable from the
result; the stamped path `/marimo/acl/...` is outside the repo. No frame geometry is recorded
(only `removed_norm`), which is why §3 cannot be settled from the file — §8 should have
required it. `calm` carries the largest contrast (−69.2) and a blocked fraction of 1.56 while
untestable; correctly excluded, but say so wherever that fraction is quoted.

### Reconciliation (author)

- **Gate range** corrected (0.856–0.931) in the registry; the report already carried it.
- **BH-FDR line** added everywhere the three per-emotion contrasts are quoted; they are unadjusted-CI findings, not significant under the registered rule.
- **CI scope** stated (scenario-only). The afraid anti-blocking is kept as "probably real" per the critic.
- **C7 corrected:** norm alone is not the confound; a fitted label-free direction is the missing control. B1e's pre-registration wording is amended in an addendum (the design was already the right one).
- **Geometry** computed as the critic asked (table above; registry `b1d.frame_geometry`): the perm frame overlaps the affect frame by a third, so the decision contrast is biased toward zero.
- **Own-floor recomputation** reported: verdict robust, attribution not.
- **Defects** recorded: pointer drift (`::perm_frame` named, `class_mean_frame` used), empty `prereg.git_commit`, no frame geometry in the result file (now supplied post hoc, unstamped), calm's 1.56 fraction excluded wherever quoted. The bit-exact B1c match is a fresh run (360 rows over 6724 s in the log).

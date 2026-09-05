# 28 · Rev-3 B1e — a fitted, label-free rank-1 control: is B1c's afraid/sad blocking emotion-specific? (verdict H3)

**Files:** `results/rev3/b1e_footprint_qwen36-27b.json` (288 arm-rows; provenance-stamped; pre-registration stamped inside: `PREREG_B1e.md` sha da043b93…, found = true; code_sha e2008fad, core c1d065ca), `b1e_cells_qwen36-27b.json` · **Script:** `src/rev3/b1e_footprint.py` (commit b9e4904 after the verifier's three fixes) · **Pre-registration:** `PREREG_B1e.md` at 0228b41 with addenda 1 (motivation corrected after report 27) and 2 (measured footprint, H3 caveat) committed before launch; run 09:38:26Z → 11:18:49Z on box 5. · **Model:** Qwen3.6-27B rev 6a9e13bd; B1c's probe pool.

## What was run
Four arms × four doses × three reps × six emotions over 29 scenarios: `none`; `emo_all` (B1c's rank-1 difference-of-means direction, hs 13–63); `permdir_all` (the same estimator on permuted labels, the steered emotion's class index, one permutation per emotion and rep, hs 13–63 — fitted, label-free, cosine 0.004 to the emotion direction); `pc1_all` (the top principal direction of A's span per layer, no label). Decision per PREREG §2 on the two pre-specified emotions afraid and sad: H3 if `emo_all − permdir_all` is significantly negative with blocked fraction ≥ 0.10 for both.

## What the file shows
**Verdict: H3.** Footprint (removed norm per position, averaged over the 51 layers): emo 0.59, permdir 0.50 (0.85× on average; 0.84× afraid, 0.77× sad), pc1 3.47 (5.8×); B1c's inert random direction removed 1.22. MC-steer 6/6; permdir's dose-0 readout shift < 0.01 for every emotion, pc1's is +0.55 (desperate), −0.27 (happy), −0.28 (calm); no quality exclusions.

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
- Under the pre-registered rule the verdict is H3: against a control fitted by the same estimator on the same data with the labels destroyed, the emotion direction blocks afraid and sad. The rule tests the point fraction; sad's range (0.07–0.56) dips below 0.10, and the secondary readout on B's reply contradicts afraid (+0.02, null, wrong sign) while agreeing for sad. [supported by file, with those two qualifications]
- The fitted label-free direction is inert for every emotion and removes no readable affect. Its inertness is close to predictable: it removes 0.41× what B1c's inert random direction removed, and the file does not record what it captures (no variance or decode number). It is a fitted, label-free null with a footprint at or below the emotion direction's, not a footprint-matched one. [supported by file; the critique's qualification adopted]
- The footprint caveat of addendum 2 is **not** discharged: the control removes less than the emotion direction, and the 0.85×–5.8× band, where B1d's rank-5 frame blocked afraid and sad, is unsampled. The top principal direction at 5.8× leaves afraid and sad untouched while crushing angry (−120, 61%) and calm, but it is a different kind of direction and it breaks the readout for three emotions. What it does show is a dissociation: it removes 79% of afraid's readable affect without blocking it, so removing readable affect is not what blocks transfer. [supported by file; interpretation narrowed after the critique]
- The retrospective reading of B1d offered in the first version of this report is withdrawn: B1d's permuted frame removed only 6.5% of afraid's readable affect yet blocked its transfer by 49.5, so overlap with the readable-affect direction does not explain it. B1d's rank-5 permuted frame blocks and B1e's rank-1 permuted direction does not; they differ in rank and in footprint (6.5 vs 0.5), and which matters is untested. [open]
- Scope: two pre-specified emotions; sad's interval is wide (0.07–0.56); the readout on A's span is still forced-choice; one pool and one direction estimate. [scope]

## Status
Complete; pre-registered decision H3, with the footprint caveat still standing: the rank-1 afraid/sad blocking survives a fitted label-free control of smaller footprint; a matched-footprint control (the 0.85×–5.8× band) has not been run. Feeds RESEARCH_PLAN §13.3 and the deadline paper's claim 5.


### Blind critique — B1e (independent agent; saw only the driver, the core functions, the pre-registration and the three result files)

# B1e — adversarial review

## 1. Numbers
Verified against the JSON: `decision.per_emotion` (afraid −52.501 [−68.931,−35.817], frac 0.2479 [0.169,0.325]; sad −28.399 [−54.208,−6.989], 0.2912 [0.072,0.556]), `counts.bh_fdr_q` 0.001/0.028 (BH arithmetic on the 5 testable checks out), `counts.n_mc_steer_pass` 6, `footprint.ratio_permdir_over_emo` 0.8456, `ratio_pc1_over_emo` 5.8438, `summary.*.mc_footprint.share_removed` (permdir ≤ 0.0867; pc1 afraid 0.787/sad 0.713/angry 0.866), `pc1_all_vs_none` angry −120.38 [−154.11,−86.79] frac 0.614, afraid +1.50, sad +12.92, calm −43.46 [−91.13,**−0.04**]. Gate passed (min 0.882), 0 quality exclusions, 288 rows × 29 scenarios, n_blocks 29. Bit-identity confirmed: `none` and `emo_all` are *exactly* equal to B1c and B1d on all five metrics and on `removed_norm`/`gen_seed`/`scenario` (0 differing cells of 72 each). Prereg sha da043b93 matches disk; committed b9e4904 09:38:04Z, run started 09:38:26Z.

Two corrections. **C2's "[−12,+17]" is wrong**: angry's `permdir_all_vs_none` CI is [−18.17,+17.04]; [−12.36,+17.28] is the union over the *other five*. **C1's "dose-0 shift < 0.01 everywhere"** is true only of permdir; `pc1_all` shifts dose-0 by **+0.554 (desperate), −0.271 (happy), −0.277 (calm)**, with `share_removed` 1.44/1.54 — pc1 breaks the readout instrument for three emotions. Nothing in the decision or counts surfaces this; §2's trigger gates permdir only.

## 2. Rule fidelity
Clean. Verdict order (incomplete → instrument → H3 → H3′ → indeterminate) is `_decide` L804-818; MC-steer uses `passes` (L806), incomplete spans all six emotions (L804) — both addendum-2 fixes present. permdir: one permutation per (emotion, rep) drawn at L1613 and reused across all 51 layers (L1614); 18 distinct `perm_seed`s, formula reproduces exactly. pc1 sign L309-311; `pair_on` is ignored by `dom` (acl_core L556), so permdir's omission of it is harmless. Stale docs, not divergences: `_mc_steer` docstring L680-682 still says the trigger is `sig`; `footprint.note` (L512) says removed_norm is "summed over the ablated layers" — `Ablate` appends per layer (acl_core L812) and `gen_B` takes `np.mean` (L1729), so it is a *mean*. Ratios unaffected.

## 3. The control
Not a strong null. Its cosine (0.004) is chance in d≈5120 and is **not in the result file at all** — only pc1's geometry is recorded, so C2's headline number is unverifiable from the run. `dom` on permuted labels is a difference of two sample means in standardized space, i.e. noise, then divided by `sd`, which tilts it toward *low*-variance coordinates: it removes 0.502 against B1c's random direction at **1.22** (b1c rows) — permdir is 0.41× an arm B1c already showed inert. Its inertness is close to predictable. The JSON shows what it removes (norm; ≤9% of readable affect) and cannot show what it *captures* (no variance, no decode accuracy). 0.85× does not discharge addendum 2's caveat — the control removes less, per-emotion 0.84 (afraid) / 0.77 (sad). pc1 helps only partly: it is a different object (top-variance axis, |cos| to emotion directions median 0.41, max 0.82; `pc1_vs_emo_cos` hs43 afraid −0.32), it disturbs the instrument elsewhere, and the footprint band 0.85×–5.8× is unsampled — which is where B1d's 11× rank-5 frame blocked afraid/sad.

## 4. Statistics
The intersection-union rule on two named emotions is conservative and needs no adjustment; BH q adds nothing (same data, and afraid's p=0.0002 is the bootstrap **floor** 1/5000, so q=0.001 is a floor, not an estimate). The ≥0.10 test is on the point only: sad's range lower end is 0.072, below the threshold. `blocked_fraction.range` divides by the point denominator and never propagates `denominator_ci` ([64,132] for sad), so the interval understates uncertainty. 29 clusters, percentile (not BCa) bootstrap. C3's pc1 claims are unadjusted across ~30 contrasts, and calm's is knife-edge (upper −0.042) on an untestable emotion with a broken readout. **The secondary outcome contradicts afraid**: `emo_all_vs_permdir_all_secondary` = +0.022 [−0.021,+0.063], wrong sign and null; sad's agrees. afraid's steering span is only 0.1475 and MC-steer 0.159 [0.056,0.267].

## 5. C5
Overstated. Its own C3 dissociation undercuts it: pc1 removes 79%/71% of afraid/sad readable affect and blocks *nothing*, so "emo_all blocks by removing affect" is not established. The B1d story fails on B1d's own file: `perm_all`'s `share_removed` is **0.065** with `perm_inert: true`, so "one-third overlap with the affect frame" is not what blocked it; and `frame_geometry` does not exist anywhere in the b1d JSON. B1d's verdict was `instrument_failed`.

## 6. Residual
Also: `control_pointers` for `emotion_direction`/`pc1_control` name DIRS (L1583-1617) though the vectors are built at L1496-1499; stamp `git_commit` empty.

### Reconciliation (author)

- **C1/C2 numbers corrected:** angry's permdir CI is [−18, +17]; the dose-0 statement is true of permdir only, and pc1's shifts (+0.55 / −0.27 / −0.28) are now stated.
- **The control is not a strong null (accepted):** it removes less than B1c's inert random direction; the file does not stamp its cosine or what it captures; the footprint caveat stands, and the 0.85×–5.8× band is the open gap. H3 is reported as the rule's verdict with these qualifications, not as "footprint discharged".
- **Statistics (accepted):** sad's range lower end 0.07; denominator CI not propagated; afraid's secondary outcome contradicts; p = 0.0002 is the bootstrap floor.
- **C5 withdrawn:** the B1d retrospective (overlap with the affect frame) is wrong on B1d's own numbers; the report now says rank and footprint are the untested differences. The frame geometry cited lives in `b1d_frame_geometry.json`, not in the B1d result file.
- **Defects recorded:** `removed_norm` is a per-layer mean, not a sum (wording fixed in the registry for B1d and B1e); `_mc_steer` docstring stale; control pointers name the dict sites; `git_commit` empty in the stamp.

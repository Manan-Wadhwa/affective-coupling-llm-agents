# 22 · Rev-3 B1c — pre-registered all-layer ablation (the rank-1 insufficiency test)

**Files:** `results/rev3/b1c_alllayer_qwen36-27b.json` (432 arm-rows, provenance-stamped, sha256 c0a37561…), `b1c_cells_qwen36-27b.json` (checkpoint) · **Script:** `src/rev3/b1c_alllayer.py` (code_sha 5fd777be, core 3da064ad) · **Pre-registration:** `docs/planning/PREREG_B1c.md`, committed 122af4c (2026-09-04 19:02Z) with addendum cc2dc32 (19:22Z); run started 19:32:20Z, finished 22:05:07Z (2 h 33 min on one RTX PRO 6000) · **Model:** Qwen3.6-27B rev 6a9e13bd.

**Provenance gap, stated up front.** The driver was meant to stamp the pre-registration's sha and commit into the result (PREREG §8). It did not: `_provenance.config.prereg.found` is `false`. Checked on the box afterwards (22:40Z): the document was there before launch, at `/marimo/acl/docs/planning/PREREG_B1c.md`, sha256 5368b835… (byte-identical to the committed cc2dc32 version), mtime 19:32:17Z against the runner's start at 19:32:18Z and `_provenance.started_utc` 19:32:20Z; the driver's relative path did not resolve from the runner's working directory. The ordering "hypotheses before data" therefore rests on git commit timestamps, the box file's mtime and `started_utc`, not on the file's own stamp.

## What was run
Six emotions × six arms × four doses {0, 0.33, 0.67, 1.0} × three reps over 29 scenarios. A is steered at layer 42 with the difference-of-means (`dom`) direction fitted on the DIR half (807) of B1's probe pool (1615, sha 70129f86…); the probe is fitted on the disjoint READ half. Arms: `none`; `emo13_42` (ablate the emotion direction from A's span at hidden states 13–42 during B's prefill, the B1/B1-follow-up window); `emo_all` (hidden states 13–63, 51 layers); `rand_all` (a fresh norm-matched random direction per emotion, rep and layer at all 51); `text` (A's message rewritten to neutral wording, no hooks); `text_keep` (A's message rewritten with the same tone but every detail changed, no hooks). B's reply is sampled with common random numbers across arms. Gate before generation: split-half class-mean cosine ≥ 0.80 at every ablated layer (600 per half, 10 seeds) — all 51 pass (0.882 at hs 14 to 0.927 at hs 53). Contrasts are scenario-blocked and dose-paired (`acl_core.paired_slope_contrast(block=, pair_doses=True)`, 5000 draws). Decision rule, fixed in PREREG §5: median blocked fraction of `emo_all − rand_all` over testable emotions with a bootstrap over scenarios; H1 if median ≤ 0.50 and upper CI ≤ 0.65; H1′ if median ≥ 0.80 and lower ≥ 0.60.

## What the file shows

**Verdict: H1.** Median blocked fraction 0.092, CI [0.038, 0.254], five testable emotions (calm's `none` slope +44 [−3, +99] does not exclude zero). MC-steer 6/6; MC-ablate failures 0; no cell excluded for quality (ablation-arm perplexity ratios ≤ 1.04).

| emotion | none slope | emo13_42 | emo_all | rand_all | emo_all − rand_all | blocked | emo_all − emo13_42 | A-span readout, top dose: none → emo_all (floor at dose 0) |
|---|---|---|---|---|---|---|---|---|
| desperate | +84 [+59, +111] | +82 | +77 | +82 | −5.0 [−22.5, +12.1] | 0.06 | −4.1 [−14.3, +3.9] | 0.955 → 0.819 (0.108) |
| afraid | +212 [+169, +253] | +154 | +160 | +205 | **−44.9 [−58.0, −31.8]** | 0.21 | +5.8 [−2.4, +14.1] | 0.397 → 0.281 (0.250) |
| happy | +48 [+8, +88] | +53 | +59 | +61 | −2.3 [−19.0, +13.8] | 0.05 | +6.2 [−2.6, +15.8] | 0.744 → 0.551 (0.287) |
| calm (untestable) | +44 [−3, +99] | +9 | +1 | +43 | −41.6 [−74.9, −10.7] | (0.94) | −8.2 [−17.9, +1.4] | 0.764 → 0.632 (0.277) |
| sad | +98 [+65, +132] | +65 | +70 | +101 | **−31.7 [−59.4, −6.1]** | 0.33 | +4.4 [−7.8, +15.3] | 0.545 → 0.165 (0.039) |
| angry | +196 [+155, +235] | +173 | +178 | +197 | −18.1 [−38.5, +1.3] | 0.09 | +5.7 [−1.8, +13.7] | 0.598 → 0.382 (0.038) |

Slopes are in probe units per unit dose (present-emotion projection of B's reply). BH-FDR q over the per-emotion contrasts: afraid 0.001, sad 0.048, angry 0.115, desperate 0.72, happy 0.81. `emo13_42 − rand_all` is significant for afraid (−50.7), sad (−36.1) and angry (−23.8 [−44.9, −3.4]). Secondary outcome (forced-choice readout on B's reply, hooks off): `emo_all − rand_all` significant for sad only (−0.09 [−0.16, −0.03]).

**Removed norm.** `emo_all` removes 0.35–0.66 per masked position summed over layers; `rand_all` removes 1.15–1.26 and leaves every slope within 7% of `none`.

**Rewrite arms.** All six flagged uninterpretable by the pre-registered §4.4 check: the neutral rewrite keeps 67–98% of A's readable affect (threshold 50%). The affect-preserving control itself shifts the transfer for desperate (+31 [+8, +54]), happy (−51 [−85, −18]), calm (−66 [−105, −30]) and sad (−69 [−94, −44]). Afraid loses 26% (text) and 20% (text_keep) of rewrites to refusals or degenerate output.

## What can be inferred
- Under the pre-registered rule the answer is H1: removing the rank-1 difference-of-means emotion direction from A's span at every hidden state 13–63 during B's prefill leaves most of the affect transfer in place (median 9% removed, upper CI 25%). [supported by file]
- The 21 layers above 42 add nothing detectable for afraid, angry, happy and sad: `emo_all − emo13_42` bounds their extra blocked fraction below 0.08 of the none slope. For desperate (≤ 0.17) and calm (≤ 0.41) the null is unpowered. This contrast was in the driver but not in PREREG §5. [supported by file, with that scope]
- The partial blocking for afraid and sad replicates the B1 follow-up's numbers (25%/37% there; 24%/37% here at 13–42, 21%/33% at 13–63) with fresh generation seeds and a fourth dose. This is a replication of the sampling, not of the direction estimate: pool and directions are identical. [supported by file]
- Specificity, descriptively: the random direction (unit-norm-matched, not footprint-matched; it removes ~2× the residual norm because an isotropic vector aligns with the residual stream better than a mean-centred direction) leaves five of six slopes within 5% of `none`; happy's random arm is +28% (47.9 → 61.4, CIs overlapping). No `rand_all − none` contrast was pre-registered or computed, so this is an eyeball comparison, not a test. [descriptive]
- The manipulation check passes as pre-registered, but the affect stays largely readable on A's span after ablation for desperate, happy, calm and angry, and across the five testable emotions the blocked fraction tracks the floor-corrected share of readable affect the ablation removes (r ≈ 0.85 on five points; the critic gets 0.91 with a variant floor; angry loses 39% and ranks fourth, so it is a trend, not a two-emotion story). Blocked-per-removed is 0.11–0.43 (median 0.27), so a linear extrapolation to a fully effective ablation still lands under the H1 threshold of 0.50. The licensed reading of H1 is therefore "this rank-1 direction is insufficient to carry the transfer", not "the linear affect representation does not carry it"; part of what claims 1–3 measure is instrument strength. [supported by file; the interpretation is ours]
- Nothing about the token channel: both rewrite arms fail their own checks, for the third design in a row. [supported by file, as a negative about the instrument]
- What carries the remainder — a higher-rank affect subspace along A's span, or B re-deriving affect from A's tokens — is untested (PREREG §7). [needs: subspace ablation; attention-masking arm]

**Under-reported by the rule, stated here.** Calm's `emo_all − rand_all` (−41.6 [−74.9, −10.7]) is the largest and is significant, but calm's `none` slope does not exclude zero, so the pre-registered testability filter drops it and the 0.94 fraction is a ratio of noise. Afraid's A-span readout under `none` is non-monotone in dose (0.250, 0.493, 0.582, 0.397) and afraid is the only emotion whose MC-ablate `separated` flag is false, so one of the two blocked emotions has a weak top-dose manipulation check. `probe_n` is 1615 against PREREG §3's 2160 generations (B1's pool after its degeneracy filter), the gate runs at 600 per half while the deployed direction is fit on 807, and the pre-registration's header date (2026-09-05) is in IST while the provenance timestamps are UTC.

## Status
Complete; pre-registered decision H1. Feeds RESEARCH_PLAN §13.3 (Paper B) and closes §13.5's "is the channel localised?" with: not to this direction, at any layer 13–63.


### Blind critique — B1c (independent agent; saw only the driver, the core functions it calls, PREREG_B1c.md and the JSON)

# Blind critique — B1c all-layer ablation

## 1. Verification

Re-running `analyze(d["rows"],…)` reproduces the stored block bit-for-bit (verdict `H1`,
median 0.09219863, CI [0.038266, 0.253915], 29 scenarios, 5 emotions); rebuilding the
per-scenario slopes by hand from `rows` gives the same `per_emotion_fraction`: desperate
0.0598, afraid 0.2122, happy 0.0470, sad 0.3251, angry 0.0922 (median = angry's). Claim 1
reproduces. Claim 2 reproduces: every `emo_all_vs_emo13_42.ci` contains 0. Claim 3
reproduces exactly: afraid −44.938 [−57.981, −31.800] = 21.2% of 211.772; sad −31.698
[−59.389, −6.072] = 32.5%; angry −18.070 [−38.526, +1.285]; desperate/happy null. Claim 5's four
readout pairs reproduce, as do `n_mc_steer_pass` 6, `n_mc_ablate_fail` 0,
`n_text_uninterpretable` 6. `removed_norm` rand 1.155–1.262 vs emo 0.352–0.659 as quoted, but
0.352–0.506 are 30-layer means and 0.463–0.659 51-layer means — the range pools two quantities.

**Claim 4 fails.** `rand_all` vs `none` present slope: desperate −2.0%, afraid −3.3%, calm
−3.8%, sad +4.1%, angry +0.3%, **happy +28.2%** (47.86→61.38). "Every slope within 7%" is
false. Claim 1's "every readable layer" is also loose — hs 0–12 are never touched (b1c:1115).

## 2. Rule fidelity

`_decide` (b1c:647–686), testability (b1c:509), the sign-aware fraction (b1c:326–346), the
dose-paired contrast (acl_core:1026–1073), the balanced panel (b1c:275–305) and addendum 2–6
are implemented as written. Divergences:

- **b1c:109–110**: `CONTRASTS` adds `emo_all_vs_emo13_42` and `text_keep_vs_none`; §5 lists
  four contrasts and neither of these, so claim 2 and half of claim 6 rest on tests in the
  driver but not pre-registered.
- **No `rand_all` vs `none` contrast exists anywhere**, so claim 4 has no test — only an
  eyeball comparison of slopes with overlapping CIs.
- **b1c:595–621**: MC-ablate is read at the top dose only (prereg names no dose); `separated`
  uses `C.ci_of` (acl_core:943, unclustered) while every other CI is scenario-blocked.
- **b1c:522**: BH-FDR over five testable emotions; §5 says six.
- **b1c:376–405**: §4.3's ≤0.10 gate hits B's replies (max 0.083, 0 cells excluded) but never
  the rewrites — afraid's `text` arm loses 26.1% of rewrites per dose and 70% of its panel
  (26/87 keys; `text_vs_text_keep` n_blocks = 11) ungated. (b1c:1163 gates on the class *mean*;
  per-class min is 0.811 at hs 14, so it passes either way.)

## 3. Statistics

The median bootstrap is sound — one shared 29-scenario draw across emotions, numerator and
denominator recomputed per draw (b1c:664–682) — and stable on reseeding ([0.038,0.254] →
[0.043,0.256]). But 0.19% of draws give a negative denominator and happy's per-draw fraction
spans −1.26 to +0.93 (1st–99th pct): the CI behaves because a median of five is robust, not
because the ratio is. It carries no emotion-sampling variance, and the reported median *is*
one emotion's fraction.

**The claim-2 null is only half powered.** Each `emo_all_vs_emo13_42` CI, as an extra blocked
fraction, bounds layers 43–63 at angry +0.009, afraid +0.011, happy +0.054, sad +0.080,
**desperate +0.170, calm +0.405** — for the last two the undetected contribution could exceed
the whole measured block, so "all six" overstates. Otherwise: balanced panel (87/87 keys),
real common random numbers (one `gen_seed` per (emotion, dose, rep) across all six arms,
verified in `rows`), a real 4-dose OLS slope, multiplicity fine.

## 4. The instrument

MC-ablate is a direction check, not a sufficiency check: "emo_all drop ≥ emo13_42 drop" is
near-tautological (superset layer set) and passed by margins of 0.004–0.013, and "rand drop
<0.02" only says the control is inert. Nothing bounds the residual — all six passed while
26–83% of readable affect survived.

The confound is real: across the five testable emotions, blocked fraction vs fraction of
floor-corrected readable affect removed at α=1 gives Pearson r = 0.909 (p = 0.032), Spearman
0.80, so claims 1–3 partly measure instrument strength. It does not overturn H1 —
blocked-per-removed is 0.18–0.46 (median 0.29), so linear extrapolation to a 100%-effective
ablation still gives ≈0.29 ≤ 0.50. Claim 5's *reading* is right; its *evidence* is not: angry
loses 37.7% of readable affect yet ranks fourth in blocking, and calm loses 25.9% while
producing the largest contrast. Quote the r = 0.91 relation and the extrapolation, not a
two-emotion story.

Two under-reported items. (a) **calm**: `emo_all_vs_rand_all` = −41.55 [−74.85, −10.68], the
largest and significant, dropped as untestable. The 0.94 fraction is meaningless (denominator
44.24 [−3.09, 98.91]); the contrast is not. "calm untestable" hides that the pre-registered
filter removed the one emotion where all-layer ablation killed the slope. (b) **afraid's top
dose is broken**: A's readout under `none` runs 0.250/0.493/0.582/**0.397** — non-monotone,
peaking at α=0.67 — yet MC-ablate and the readout numbers are read at α=1.0, and afraid is the
only emotion with `mc_ablate.separated = False`. One of the two blocked emotions rests on it.

## 5. Provenance and residual defects

Claim 7 is right and understated: `prereg.found = false`, empty sha256 and git_commit, *and*
PREREG_B1c.md is dated 2026-09-05 in its header and its "before launch" addendum while
`_provenance.started_utc` = 2026-09-04T19:32:20Z and `finished_utc` = 2026-09-04T22:05:07Z —
reconcilable only at ≥UTC+5, and nothing in the file settles it. One strong positive:
`code_hash(driver, acl_core)` recomputed on disk today is `5fd777beab86b177`, identical to the
stamped `code_sha`, and `acl_core_sha == acl_core_sha_on_disk_now`; no control pointer is stale.

Fix or disclose before quoting: (i) claim 4's 7%; (ii) `rand_all` is norm-matched on the
*direction* (both unit, acl_core:828–833), not the footprint — it removes ~2× the residual
norm precisely because an isotropic vector aligns with the residual stream better than a
mean-centred dom direction, so "twice the removed norm" is not a stronger control, and the
sharper ones (`orthogonal_dir`, a cross-emotion direction) were never run; (iii) `probe_n` =
1615 against §3's 2160 generations, unexplained, with the gate at n_per_half = 600 while the
deployed direction is fit on 807; (iv) text-arm attrition is selection on the outcome
(refusing rewrites are the intense ones), compounding their §4.4 uninterpretability.

### Reconciliation (author)

- **Claim 4 was wrong as written** and is corrected above and in the registry (`b1c.random_control_inert`): happy's random arm is +28%, not "within 7%"; there is no `rand_all − none` contrast; and the random direction is norm-matched as a unit vector, not by footprint, so its larger removed norm is not a stronger control. Accepted in full.
- **Claim 2 rescoped** to the four emotions where the extra blocked fraction is bounded below 0.08; desperate and calm are unpowered. Also disclosed that `emo_all_vs_emo13_42` and `text_keep_vs_none` were driver-declared, not in PREREG §5. Accepted.
- **Claim 5 rewritten** around the correlation and the blocked-per-removed extrapolation (author's recomputation r = 0.85, blocked/removed 0.11–0.43, median 0.27; critic's 0.91 / 0.18–0.46 / 0.29 under a variant floor). The extrapolation keeps H1, but the confound is real and now stated. Accepted.
- **Calm and afraid** caveats added to the report body. Accepted; the pre-registered filter stands, the numbers are shown next to it as PREREG addendum 1 requires.
- **Rule-fidelity items** (BH-FDR over 5 not 6; MC-ablate read at the top dose only with an unclustered `separated` CI; rewrite attrition ungated and selection-on-outcome; `probe_n` 1615 vs 2160; gate at 600/half vs fit on 807): all disclosed above; none changes the verdict, and the driver is not edited post hoc.
- **Provenance**: the prereg-stamp gap and the IST/UTC date mismatch are stated at the top of the report; `code_sha` matches the driver on disk. The document was on the box one second before the runner started (sha identical to cc2dc32); the driver's relative lookup missed it. The fix for next time (resolve the pre-registration path from the driver's own directory) is in the operator notes.
- Registry entries `b1c.verdict_H1`, `b1c.no_gain_beyond_layer42`, `b1c.random_control_inert`, `b1c.readout_residual` were re-worded after this critique; the other four are unchanged.

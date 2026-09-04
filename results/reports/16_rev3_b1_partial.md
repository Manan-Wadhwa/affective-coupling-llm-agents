# 16 · Rev-3 B1 — E4 rerun (partial checkpoint 47/54 → completed rerun 54/54)

**Files:** `results/rev3/b1_cells_qwen36-27b.json`, `results/rev3/b1_partial_summary.json` · **Scripts:** `src/rev3/b1_e4rerun.py`, `src/rev3/b1_analyze.py`, `src/lib/acl_core.py` · **Model:** Qwen/Qwen3.6-27B rev 6a9e13bd · **Registry:** `b1.stability_gate_dom` (b, no file), `b1.mc_ablate_pass_5of5`, `b1.emo_vs_rand_sig_2of5` — all PENDING

## What was run
See `results/rev3/README.md` for the full tables. Difference-of-means directions from a DIR half of the probe pool; a READ-half probe measures B; a generation-side forced-choice readout checks that steering reaches A's text and that ablation removes readable affect; arms none / emo / rand / orth / cross; degeneracy, refusal, perplexity and removed norm recorded per arm-row. 5 of 6 emotions complete at 3 doses × 3 reps.

## What the file shows
Manipulation checks pass on point estimates for 5/5 measured (calm's emo/rand CIs overlap; afraid's readout non-monotone in α). Blocking contrast emo − rand excludes zero for afraid (−25.7 [−44.6, −6.3]) and sad (−28.1 [−51.4, −4.7]); crosses zero for desperate, happy, calm. The direction-stability gate (0.909) is not in any committed file.

## What can be inferred
- With a direction that reproduces across halves (per the uncommitted gate) and a passing generation-side check, ablating A's affect reduces B's dose-response for two of five emotions and not detectably for three. [supported by file, partial grid]
- The `emo − X` contrast intervals are paired but not scenario-blocked. [supported by code]
- No verdict is defined until the grid is complete; the analyzer refuses one. [supported by code]
- Because A's tokens and layers above 43 are untouched (as in the original E4), a non-blocking outcome does not locate the channel. [needs: token-level arm]

## Status
Superseded by the from-scratch rerun launched 2026-09-04; this checkpoint remains the committed partial record.

### Independent critique (blind: saw only the scripts and the JSONs)

## 0. What the code actually computes

**Pool and halves.** `generate_pool` builds a 6×6 crossed corpus (`acl_core.py:369-452`), `pool_features` mean-pools the final-A-utterance span (`acl_core.py:454-482`). `b1_e4rerun.py:136-137` splits the kept items by a single permutation: `DIR = rs[:N//2]`, `READ = rs[N//2:]`. With `$.config.n = 1578` that is 789/789. Steering and ablation directions come from `dir_dec` (`:140-141`); B is scored with `read_dec[focus]` and `read_dec_other` (`:142-144`, used at `:286-287`).

**Is the B-probe independent?** Sample-disjoint, yes. Axis-independent, no — both halves estimate the same population direction, so if split-half cosine is 0.909 the read probe is ~91% aligned with the ablated direction. What saves it from circularity is not the split but the *span*: `score_B` (`:275-287`) pools only B's reply characters (`spans` at `:283`) with `ab.clear()` first (`:284`). Two consequences the authors do not state: (a) B's reply is pooled *in context*, so activations over B's tokens can carry A's affect by attention — "B's present score" is not purely a property of B's text; (b) it is an unnormalised dot product in standardised space, so the units are arbitrary and incomparable across emotions (desperate ≈ −150, happy ≈ +220 at dose 0).

**Ablate.** `acl_core.py:799-813`: at each of 30 decoder blocks (`abl_hs` 13..42, from `b1_e4rerun.py:124-125` with focus=43, n_layers=64) it removes `(h·d)d` at the masked positions, prefill only (`:806-807` returns unmodified on decode steps). `removed_norm` (`:810`) is the mean L2 of the removed component over masked positions. `random_dir_like`/`orthogonal_dir` (`:828-840`) return **unit** vectors — that is the only sense in which the arms are "norm-matched". `rand_dirs` is **one draw per layer, shared across all six emotions and all reps** (`b1_e4rerun.py:172`); `orth_dirs` is one draw per emotion (`:169`). `cross` reuses another emotion's real direction (`:160,173`).

**Readout.** `b1_e4rerun.py:245-273` is a private reimplementation of `acl_core.forced_choice_readout` (the provenance pointer at `:111-113` names a function the run never calls). It builds `conv + READOUT_Q`, masks A's reply span (`:262-263`), runs one forward pass, takes last-position logits (`:267`) and softmaxes **only over the six emotion first-tokens** (`:270-272`). So it is a renormalised 6-way share, not a probability of naming the emotion. The ablation *does* touch the positions the readout reads — A's span — at layers 13-42; the final position itself is unmasked, so the effect must travel by attention, and layers 43-63 are untouched.

**Statistics.** `slope_ci` (`:959-992`) bootstraps sample means per dose, or scenario blocks when `block=` is passed. `paired_slope_contrast` (`:994-1017`) has **no block argument at all** and resamples samples with a shared index across arms. With doses (0, 0.5, 1.0), `ols_slope` (`:953-957`) has centred x = (−0.5, 0, +0.5): **the slope is exactly y(1.0) − y(0.0)** — verified numerically (desperate/none slope +109.888 = y1−y0 +109.888). The α=0.5 cell carries zero weight in every reported slope and contrast; it is one third of the compute and does not enter a single number.

**Analyzer flags.** `b1_analyze.py:80` `separated` = emo CI upper < rand CI lower (unpaired, per-arm). `:81` `passes` = `drop_emo > 0 and drop_emo > drop_rand` — a bare sign test, no CI, no threshold.

---

## C1 — "split-half cosine 0.909 [0.906, 0.912] at n=789/half over 10 splits (logistic 0.577), gate 0.80"

**Where the number would live.** `b1_e4rerun.py:147-152` computes `stab`/`stab_logreg` and prints exactly `[B1 GATE] split-half(dom) X.XXX CI [...] | logreg X.XXX | gate 0.8`. It is stored at `:411-412` under `out["direction_stability"]` and written by `C.write_result` (`:421`) to `<outdir>/b1_e4rerun_<tag>.json` — default `/marimo/out`. Neither named file contains it: `results/rev3/b1_cells_qwen36-27b.json` has only `$.key`, `$.config`, `$.cells` (`Checkpoint.put`, `acl_core.py:1147-1149` writes nothing else), and `results/rev3/b1_partial_summary.json` has only `$.summary`, `$.n_cells`, `$.n_expected`, `$.doses`, `$.emotions` (`b1_analyze.py:137-139`). **The claim is unverifiable from the artefacts and was evidently transcribed from a console log.**

**Objections.**

1. **n=789/half is not what the code computes.** `:147` passes `n_per_half=min(N // 2, 600)`. With `$.config.n = 1578` that is **600**, not 789. 789 is the DIR/READ split size (`:136-137`). The two are being conflated; no configuration of this script can produce a 789/half split-half.
2. **The gated direction is not an ablated direction.** The gate is fitted on `feats[focus]`, focus = 43 (`$.config.focus`). The ablation uses `dir_dec[L]` for `L in abl_hs` = 13..42 (`:124-125,168`), which **excludes 43**. The comment at `:146` ("the direction we are about to ablate along") is false: layer 43 supplies the *steering* vector (`:171`) and the *scoring* probe (`:286`), not any ablated one. The stability of the 30 directions actually projected out is never measured.
3. **The CI is over 10 seeds, and the statistic is a mean over six classes.** `split_half` returns `ci_of` of the per-seed class-means (`acl_core.py:709-713`). A width of 0.006 over 10 draws says the split-half estimator is reproducible, not that every class is at 0.909. `angry` — the emotion with no usable data — could be the weak class and the headline would not move.
4. **`pair_on=yo` at `:147` is inert.** `fit_direction` reads `pair_on` only in the `pca_diff` branch (`acl_core.py:597`); for `dom` it is ignored, so the "matched" framing in the call is decorative. The `logreg` comparator is called without it (`:149`), so the two arms of the comparison differ in more than the estimator only cosmetically.
5. **Higher stability makes the B-probe *more* circular, not less.** At cos = 0.909 the "held-out" READ direction is nearly the DIR direction. The non-circularity argument (`:29-31`, `:114-115`) rests on span disjointness, not on the split.
6. The gate does not gate: `:154-156` prints a warning and runs anyway.

**To make it.** Ship `direction_stability` inside the checkpoint (or quote the run's own result file with its provenance stamp); report `per_class` and the `angry` entry; report split-half at the layers in `abl_hs`, not at `focus`; state `n_per_half` as the code sets it. Nearest defensible wording: "difference-of-means split-half cosine at layer 43 is 0.909 (10 splits, 600/half), versus 0.577 for logistic; the 30 ablated layers were not gated."

**Verdict: UNDETERMINABLE** — absent from both files, and the stated n contradicts the code, which gates a layer that is never ablated.

---

## C2 — "generation-side manipulation checks pass for 5 of the 5 emotions measured"

**Recomputed (matches `b1_partial_summary.json` to all printed digits).**

MC-steer, `$.summary.<e>.mc_steer.slope` — desperate 0.825, afraid 0.208, happy 0.482, calm 0.532, sad 0.565, all `sig: true`; `angry` absent.

MC-ablate at α=1.0, `$.summary.<e>.mc_ablate` — `drop_emo` 0.134 / 0.219 / 0.243 / 0.110 / 0.400 (desperate/afraid/happy/calm/sad) against `drop_rand` −0.001 / 0.002 / 0.001 / 0.000 / 0.007. `passes: true` ×5; `separated: false` for calm.

**Objections.**

1. **MC-steer for `afraid` is not monotone.** `$.cells.afraid/*` `A_readout_e` means: 0.250 → 0.577 → **0.458**. The top dose is *below* the mid dose. Because OLS on this grid is y(1.0) − y(0.0), the slope is positive and "sig" while the dose response has turned over. "raises … with dose" is false for afraid at the dose the ablation arms are all evaluated at.
2. **`afraid` steering does not produce afraid.** At α=1.0 the modal readout is *desperate* (0.517) over *afraid* (0.458) — `ctx_readout_full` on the `none` arm. The manipulation moved A's text into a neighbouring state. `afraid` is one of the two emotions C3 counts as blocking.
3. **"5 of 5 pass" uses the weaker of the two flags the code computes.** `passes` (`b1_analyze.py:81`) is a sign comparison with no uncertainty; `separated` (`:80`) is the CI test and is **false for calm** (`$.summary.calm.mc_ablate.separated`). By the analyzer's own stricter criterion it is 4 of 5. (In calm's favour: the per-arm CIs are unpaired. A paired per-sample test — same A text in both arms — gives calm's specific drop +0.109 CI [0.084, 0.137], comfortably clear of zero. The right fix is to run the paired test, not to quote the looser flag.)
4. **"norm-matched" does not survive `removed_norm`.** Mean `removed_norm` at α=1.0: emo 0.41–0.66, rand 0.66–0.70, orth 0.67–0.87, cross 0.33–0.54. The random and orthogonal arms remove **1.1–2.1× more** residual norm than the emo arm. The match is on unit direction length only. Here the mismatch runs *against* the authors (emo perturbs least and moves the readout most: mean TV distance from `none` is emo 0.221, cross 0.032, orth 0.005, rand 0.003), so the specificity conclusion survives — but the word "norm-matched" as a description of the intervention magnitude is wrong, and the driver's own docstring (`:21-24`) concedes it.
5. **A passing MC-ablate on A's readout does not show the emotion was removed from what B received.** `acl_core.frozen_token_audit` says it outright (`:942`): "a readout over a frozen span measures re-encoding, not transmission". The readout and the ablation are the *same forward pass over the same frozen A tokens at the same 30 layers*; it certifies that the linear feature is gone from layers 13-42 of A's span, which is close to definitional for a projection. B's generation is a different forward pass, and the residual readout is 0.20-0.80, not floor — the feature is attenuated, not removed.
6. **Cross is not null.** Paired none−cross at α=1.0: desperate +0.030 [0.020,0.042], calm +0.041 [0.025,0.060], and *negative* for afraid −0.023, happy −0.012, sad −0.033. One-vs-rest `dom` directions share a valence component; the arms are not mutually orthogonal.
7. `rand` is a single draw (`:172`) reused for all emotions, so its ~0 effect carries no draw-level error bar.

**To make it.** Report `separated`/paired CIs rather than `passes`; state the afraid non-monotonicity and its argmax; drop "norm-matched" or replace with "unit-norm, with removed norm reported"; and either restrict the claim to "the linearly-decodable emotion feature was removed from A's span at layers 13-42" or add a check on B's *received* representation (e.g. the readout run over B's prefill state, or an independent classifier on A's text).

**Verdict: SUPPORTED-WITH-CAVEATS** — the specific drops are large and clean, but "5 of 5" leans on the weaker flag, afraid's MC-steer is non-monotone and off-target, "norm-matched" is inaccurate, and the check licenses a claim about re-encoding, not about transmission.

---

## C3 — "blocking contrast excludes zero for 2 of 5 (afraid, sad), crosses zero for desperate, happy, calm"

**Recomputed, matches `$.summary.<e>.emo_vs_rand` exactly:** desperate −7.65 [−24.05, +9.23] `sig:false`; afraid −25.72 [−44.63, −6.34] `true`; happy +11.12 [−13.51, +36.05] `false`; calm −11.68 [−46.23, +22.47] `false`; sad −28.10 [−51.37, −4.66] `true`. The arithmetic is right.

**Objections.**

1. **"slope against dose" is a two-point difference.** With x = (0, 0.5, 1.0), OLS gives y(1.0) − y(0.0). Nothing tests dose-*response*; the α=0.5 cells are inert weight. This matters because two of the three "crosses zero" emotions are visibly non-monotone in B's present score (`$.summary.happy.none.mean_by_dose`: 218.3 / 304.8 / 251.7; `calm`: 107.5 / 127.1 / 112.8 — both peak at the ignored midpoint).
2. **`calm` has no transmission to block.** `$.summary.calm.none.present_slope` = +5.29 CI [−48.4, +60.6], not distinguishable from zero. Counting calm as "crosses zero" implies a null blocking effect where there is no channel; it should be excluded, not tallied.
3. **The nulls are underpowered and the report does not say so** (`acl_core.mde` exists at `:1019` and is never called). Expressed as a fraction of the `none`-arm slope, the contrast CIs are consistent with blocking up to **22%** (desperate), **40%** (happy). "Crosses zero" is not evidence of no blocking at these n.
4. **Multiplicity.** Four contrasts × five emotions = 20 tests, uncorrected. Beyond the two headline hits: `$.summary.happy.emo_vs_orth` = +23.57 [+1.47, +44.63] `sig:true` — ablating the emotion direction **increased** B's score relative to orthogonal — and `$.summary.calm.emo_vs_cross` = −40.03 [−69.63, −9.63] `sig:true`, blocking by a *different* emotion's direction on the emotion with no transmission. Selecting `emo_vs_rand` out of four available comparators is a forking path the analyzer's silence does not close.
5. **The contrast bootstrap ignores clustering** (`paired_slope_contrast`, `acl_core.py:994-1017`, no `block`), inconsistently with `present_slope` which is blocked (`b1_analyze.py:52`). n=87 per cell is 29 scenarios × 3 reps (`$.cells.*[*].scenario` length 29, `$.config.reps` 3). I re-ran the contrast with a scenario-blocked paired bootstrap: afraid −25.72 [−45.28, −6.37], sad −28.10 [−50.76, −5.73] — pairing already absorbs the scenario main effect, so this particular objection does not bite. It should still be fixed for consistency.
6. **Arms are magnitude-mismatched in the direction that matters here.** `rand` removes ~1.2-1.6× the norm the `emo` arm removes (§C2.4), so the contrast is emo vs a *stronger* generic perturbation; whatever generic disruption does to B's score is not cancelled.
7. **B's replies are unseeded.** `gen_B` calls `h.model.generate(..., do_sample=True)` directly (`b1_e4rerun.py:229-232`) with no `torch.manual_seed`, unlike `acl_core.gen` (`:305-306`). Every reported outcome is a non-reproducible sample despite `Provenance(seeds=...)` at `:101`.
8. Two of five is also two of *six* declared; and all five emotions share an identical α=0 baseline (the α=0 A-messages are byte-identical across emotions — `vec=None` at `:195`, same seed — visible as identical `ctx_readout_full` at α=0 for every emotion), so the five per-emotion results are correlated, not five independent trials.

**To make it.** Report effect sizes as a fraction of the `none` slope with an MDE per emotion; exclude emotions whose `none` slope does not clear zero; pre-register `emo_vs_rand` as *the* contrast or correct across the four; use ≥4 doses if "slope" is to mean anything; seed B.

**Verdict: SUPPORTED-WITH-CAVEATS** as a bare description of the numbers; **NOT SUPPORTED** as an inference, because "crosses zero" for calm is vacuous (no transmission) and for desperate/happy is consistent with 22-40% blocking.

---

## C4 — "degeneracy ≤ 0.07, refusal ≤ 0.07, perplexity 10-20 across every arm and dose"

**Recomputed over all 235 arm-rows in `$.cells`:** `degenerate_frac` max **0.0690**, `refusal_frac` max **0.0690**, `perplexity` min **10.41** / max **19.61**, `distinct2` min 0.990, `n_words` 35.0-53.5. Zero rows violate any of the three bounds. Per (arm, dose) aggregates: degeneracy 0.000-0.0138, refusal 0.000-0.0138, median-of-medians perplexity 13.97-16.06.

**Objections.**

1. **The bounds are post-hoc and sit on the observed maxima.** 0.0690 is 2/29 responses in one cell; "≤ 0.07" is the smallest round number above the largest value seen, and "10-20" brackets [10.41, 19.61]. Quoted as thresholds these read as pre-declared criteria; nothing in `b1_e4rerun.py` declares them (`DOSES`, `ARMS`, `EMOS`, `STABILITY_GATE`, `PROBE_K` at `:55-60` are the declared constants; no degeneracy gate exists). Report the maxima.
2. **`degenerate_frac` and `refusal_frac` are near-identical** (means 0.0050 vs 0.0048; equal in 14 of 15 arm×dose aggregates). `is_degenerate` (`acl_core.py:872-874`) is an OR over empty / refusal / persona-break / distinct2<0.5; since `distinct2` never falls below 0.990 and `n_words` never below 35, the composite is doing nothing except re-reporting refusal. The degeneracy screen has essentially no discriminating power on this data.
3. **`perplexity` is the median over each cell** (`:322`), so tail degradation is invisible by construction; a cell with 20% gibberish and a clean median passes.
4. **Perplexity is computed on B's short replies in isolation** (`acl_core.py:876-895`, `C.perplexity(h, breps)` at `:308`) — not in context — so it measures unconditional fluency, not coherence with A's turn, which is the thing an ablation would be expected to damage.
5. **Coverage.** The claim says "every arm and dose"; the checkpoint has 47 of 54 cells (`$.n_cells` 47, `$.n_expected` 54) and `angry` only at α=0, reps 0-1. "Every" is 87% of the grid.
6. Scope: this rules out *gross* text degradation. It does not rule out the readout drops in C2 arising from a subtler representational disruption that leaves surface fluency intact — and the emo arm is exactly where TV distance from `none` is 0.22 while perplexity is flat.

**To make it.** State the observed maxima rather than round thresholds; report a perplexity quantile (p90) alongside the median; compute perplexity in context; and scope "every arm and dose" to the 47/54 cells present.

**Verdict: SUPPORTED-WITH-CAVEATS** — the numbers hold on the cells present, but the thresholds are retrofitted to the data, the degeneracy composite is redundant with refusal, and "every" overstates a 47/54 grid.

---

## Cross-cutting: is a partial grid interpretable, and does the analyzer protect against forking paths?

`b1_analyze.py:129-131` refuses a verdict when `got < expect` or any emotion is incomplete, and prints "PARTIAL GRID -- no verdict". That is the right reflex, and it fires here (47/54, `angry` at `_complete: false`). But it protects nothing that matters:

- It withholds only the *aggregate* label. Every per-emotion number — `mc_ablate`, `emo_vs_rand`, `mc_steer` — is still written to `$.summary` in full, and all four claims quote those. A refusal to name the verdict while publishing the tally ("5 of 5", "2 of 5") is the verdict.
- The denominator moves. The pre-declared rule at `b1_e4rerun.py:399-403` is `n_mc_pass < len(EMOS)/2` and `n_block >= 4` **out of six**. C3's "2 of 5" is being scored against a rule written for 6; 2 of 6 could never reach the `blocking` threshold of 4. Reporting the fraction with a shrunken denominator makes a pre-declared failing result look like a partial one.
- `angry` is not missing at random with respect to interpretation: it is the only emotion with no α=1.0 cell, and the two cells that exist show `A_readout_e` at 0.042-0.051 — the lowest α=0 baseline in the set.

The honest partial statement is: *of six pre-declared emotions, five have complete grids; on those five the ablation removes the linearly-decodable emotion feature from A's span (paired specific drop 0.109-0.393, all CIs clear of zero), and the reduction in B's transmitted affect relative to a random-direction control clears zero in two, with the three nulls consistent with up to 22-40% blocking and one of them (calm) showing no transmission to block.*

## Reconciliation
No prior audit covers rev 3. Everything below is new and most of it bears on the rerun now executing, which uses the same code:
- **The gate's n is 600 per half, not 789** (`b1_e4rerun.py:147` caps at 600); the results log, the registry text and this report's sources repeated the 789. Corrected in the registry and logged.
- **The gated direction is never ablated.** Stability is measured at layer 43; the ablation projects out `dir_dec[L]` for L = 13..42. The 30 directions actually removed have no stability figure. The plan's §1.3 premise ("ablating along a stable direction") is therefore not yet established for B1 either.
- **B's replies are unseeded** in `gen_B` (direct `model.generate`, no `torch.manual_seed`), so the rerun is not reproducible despite its provenance stamp.
- **Control-pointer drift:** the `generation_side_manip_check` pointer names `acl_core.forced_choice_readout`; the driver uses a private reimplementation (`readout`, lines 245–273). The checker cannot catch this because no registry entry carries that pointer yet.
- **The pre-declared rule uses a denominator of six** (`n_block >= 4` of 6); "2 of 5" is not a partial version of that rule, it is below it.
- **afraid is off-target**: at α = 1 the modal readout is *desperate* (0.517) not afraid (0.458).
- `separated` (the CI flag) is false for calm, so the stricter count is 4 of 5; a paired per-sample test rescues calm (+0.109 [0.084, 0.137]) and should replace both flags.
- Nulls are consistent with 22–40% blocking; `acl_core.mde` exists and is never called; calm has no transmission to block and should not be tallied.
- `rand` is one draw shared across emotions; degeneracy is redundant with refusal; perplexity is a median on B's text out of context.
Verdicts: C1 UNDETERMINABLE, C2 SWC, C3 SWC as description / NOT SUPPORTED as inference, C4 SWC. The critic's closing "honest partial statement" is the wording this report adopts.


## Update 2026-09-04 · the from-scratch rerun completed (54/54)

Files: `results/rev3/b1_e4rerun_qwen36-27b.json`, `b1_summary_qwen36-27b.json`. Full tables in
`results/rev3/README.md`. Against the critique above: the gate is now in a committed file
(0.907, n = 600/half, layer 43, per-class afraid 0.859 and desperate 0.867 lowest);
`separated` fails for afraid and calm on the full grid; emo − rand is significant for 3/6 with
happy in the *wrong* direction (+31.2 [+9.4, +53.3]); the pre-declared rule returns
`not_blocking`. The seed, layer-gating and control-pointer defects the critique found are
unchanged in this run. The delta critique follows.

### Delta critique — completed B1 run (blind: saw only the scripts and the JSONs)

**Framing correction first: this is not the 47/54 partial completed.** `$.config.n` in the cells file is now **1615** (was 1578); `probe_n` 1615, halves 807/808. `Checkpoint.key` is a sha256 over the config *including* `n` (`acl_core.py:1124-1126`), so on restart the old file was rejected, not resumed (`:1131-1136`). The 47 cells were discarded and everything regenerated against a different probe pool. That is a windfall: two independent runs of the same design. `emo_vs_rand` moved afraid −25.7 → **−50.5**, happy +11.1 (ns) → **+31.2 (sig)**, desperate −7.6 → −9.8, sad −28.1 → −29.2, calm −11.7 → −13.4; MC-steer afraid 0.208 → 0.148. Two of six changed by more than their own CI half-width, and one crossed into significance. **The reported CIs understate run-to-run variability**, because they condition on one probe pool and one draw of `rand_dirs` (`b1_e4rerun.py:172`).

**1. C1.** The re-worded claim matches the file exactly: `$.direction_stability.used` = mean 0.9074, ci [0.9042, 0.9107], `n_per_half` 600, `n_seeds` 10, `space` "raw"; `logreg_for_comparison.mean` 0.5670. My "n=789" objection is resolved and the layer-43 caveat is now stated. Two new ones. (a) `per_class` is [happy .929, calm .948, sad .900, angry .942, **afraid .859, desperate .867**] — the gate tests only the mean (`:153`), so per-class is ungated; the two weakest classes are afraid (which carries the largest blocking effect) and desperate. The logreg per-class floor (afraid .381, desperate .396) is essentially the 0.394 the original E4 was faulted for — the two estimators disagree most exactly where the headline sits. (b) `space: "raw"` is the right choice (it is what `Steer`/`Ablate` consume), but `raw_direction` divides by a shared `sd` (`acl_core.py:655-657`), adding a class-independent tilt that inflates cosine. With no `space="std"` comparator and no label-permuted floor, 0.907-against-0.80 has no calibration. Still ungated: the 30 directions at `abl_hs` 13-42.

**2. C2/C3.** `passes` 6/6 but `separated` 4/6 (afraid, calm false) — again the unpaired test. Paired per-sample specific drops (same A text both arms) all exclude zero: afraid +0.108 [0.071, 0.144], calm +0.124 [0.097, 0.153], sad +0.366, angry +0.216. Use the paired test and MC-ablate is 6/6 honestly. But afraid's MC-*steer* is still non-monotone (0.250 → 0.509 → **0.397**), its slope CI is now [0.030, 0.257] — barely clear of zero — and its α=1.0 argmax is *desperate* (0.46) over afraid (0.40). Afraid is simultaneously the least stable direction, the weakest manipulation, and the largest blocking effect.

**Happy (+31.2) is a grid artifact, not an anti-blocking effect.** All five arms peak at the ignored midpoint (`none` 225.2/310.8/**273.9**; `emo` 212.0/307.7/**292.2**). Since OLS on (0, 0.5, 1.0) is exactly y(1.0) − y(0.0), the contrast is entirely "emo decayed less from the α=0.5 peak". Reading it as "ablation increased transmission" requires believing a slope that discards the peak.

**Angry does not rescue the null:** −22.4 [−53.1, +7.3] is consistent with blocking up to **27%** of its `none` slope (+195.3). Nulls as fractions of `none`: desperate −0.33, angry −0.27, calm −1.09 (and calm's `none` slope is +46.4 CI [−0.1, +96.3], **not** distinguishable from zero — still nothing to block).

**3. The verdict.** `verdict: "not_blocking"` follows mechanically from `b1_e4rerun.py:400-403`: `n_mc_pass` 6 ≥ 3, and `sum(sig)` counts **3** — but that count includes happy, whose contrast is *positive*. The rule counts significance without sign (`:402`), so "not_blocking" is reached partly by a result pointing the other way. It licenses only: *the pre-declared 4-of-6 threshold was not met.* It does not license a null, because no MDE was computed (`acl_core.mde:1019` never called) and the three nulls admit 27-33% blocking.

**"Lexical affect ablation does not block contagion" is not defensible from this file.** Nearest defensible: *"Projecting the difference-of-means emotion direction out of A's tokens at 30 mid-stack layers removes most of the model's own readout of A's affect (paired specific drop 0.108-0.366, 6/6) yet reduces transmission to B's reply by a detectable amount in only 2 of 6 emotions (afraid, sad); the remaining nulls are underpowered (compatible with up to ~33% blocking), one emotion has no measurable transmission (calm), and one contrast reverses sign under a dose grid whose response peaks at the dose the slope discards."*

**4. Provenance.** Helps: `model_revision` 6a9e13bd…, `acl_core_sha` = `…on_disk_now` (23c758a9a0a52c7b), library versions, 6280.7 s. Hurts: `code_sha: ""` — `Provenance` is built without it (`b1_e4rerun.py:95-118`) and `code_hash` is never called on the driver, so the script that produced this is unhashed. `seeds: {"pool": 0, "run": 0}` is **affirmatively misleading**: `gen_B` calls `generate` directly with `do_sample=True` and no `torch.manual_seed` (`:229-232`), so every B reply — the outcome variable — is unseeded. Two `control_pointers` name code that does not run: `readout -> acl_core.forced_choice_readout` (the driver reimplements it at `:245-273`) and `build_dirs`, which does not exist. `frozen_token_audit` is the literal dict passed at `:413-417` — a declaration, not a verification — and its own `note` ("a readout over a frozen span measures re-encoding, not transmission") remains the best short objection to MC-ablate.

**C4 now fails as worded.** Over 270 rows, `perplexity` spans **[9.84, 21.16]**, with three cells outside 10-20 (happy/1.0/2/emo 21.16; calm/1.0/0/rand 20.98; sad/1.0/2/orth 9.84). Degeneracy and refusal max 0.069 hold. Fix: "perplexity 9.8-21.2 (median per cell), degeneracy and refusal ≤ 0.07."

**5. Updated verdicts**

| Claim | Before | Now | One line |
|---|---|---|---|
| C1 stability gate | UNDETERMINABLE | **SUPPORTED-WITH-CAVEATS** | Numbers verified at `$.direction_stability.used`; afraid/desperate per-class 0.859/0.867 are ungated, the 30 ablated layers are unmeasured, and raw-space cosine has no permuted floor. |
| C2 manipulation checks | SUPPORTED-WITH-CAVEATS | **SUPPORTED-WITH-CAVEATS** (unchanged) | MC-ablate is 6/6 on a paired test (not the `separated` flag, 4/6); afraid's MC-steer stays non-monotone and off-target; "norm-matched" still contradicted by `removed_norm` (emo 0.39-0.61 vs rand 0.66-0.70). |
| C3 blocking contrast | SUPPORTED-WITH-CAVEATS / NOT SUPPORTED as inference | **NOT SUPPORTED as inference** | 3/6 "significant" includes a sign reversal that is a two-point-slope artifact; nulls admit 27-33% blocking; calm has no transmission; run-to-run drift exceeds the CIs. |
| C4 text quality | SUPPORTED-WITH-CAVEATS | **NOT SUPPORTED as worded** | Perplexity range is [9.84, 21.16], not 10-20; three cells violate it. |

## Reconciliation (delta)
Accepted in full; the record was corrected as follows.
- **Two independent runs exist, not one resumed run.** The checkpoint key includes the pool size, so the 2026-09-04 rerun discarded the 47-cell partial and regenerated everything. Comparing the two: afraid's emo − rand moved −25.7 → −50.5 and happy's +11.1 (n.s.) → +31.2 (sig). The reported CIs condition on one pool and one random-direction draw and therefore understate run-to-run variability. Recorded in the rev3 README and the results log.
- **Happy's positive contrast is a grid artifact**: every arm peaks at α = 0.5, which the three-point OLS slope discards. The registry entry `b1.emo_vs_rand_sig_3of6` now says so; the pre-declared rule counts significance without sign, so its "3 significant" includes this reversal.
- **Perplexity wording corrected** to the observed range 9.8–21.2 (three cells outside 10–20); degeneracy and refusal ≤ 0.07 hold.
- **Provenance defects**: `code_sha` is empty (the driver never hashes itself), `seeds` claims a run seed while B's generation is unseeded, and two control pointers name code that does not run (`forced_choice_readout`, `build_dirs`). Added to the plan's to-dos; the registry note for `b1.verdict_not_blocking` carries the caveat.
- The critic's "nearest defensible" sentence replaces the earlier one in the rev3 README.
Updated verdicts: C1 SWC · C2 SWC · C3 NOT SUPPORTED as inference · C4 NOT SUPPORTED as worded.

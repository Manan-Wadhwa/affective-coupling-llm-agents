# 29 · Rev-3 B1c's design on Llama-3-8B-abliterated — the transfer replicates; the rule returns instrument_failed on a tie

**Files:** `results/rev3/b1c_alllayer_llama3-abl.json` (432 arm-rows; provenance-stamped, revision dd67dd05; PREREG_B1c stamped inside), `b1c_cells_llama3-abl.json` · **Script:** `src/rev3/b1c_alllayer.py --model failspy/Llama-3-8B-Instruct-abliterated` on box 6, 10:33–11:20Z (2831 s; 5 s per arm-row). · **Status of the design:** a replication of a pre-registered design on a second model, not a new pre-registration; the 27B's rule and thresholds apply unchanged.

## What was run
As B1c: six arms (none, emo13_42 → here hs 6–20, emo_all → hs 6–31, rand_all, text, text_keep), four doses, three reps, 29 scenarios; a fresh probe pool for this model (2160 generated, 1079 kept; sha d1f52ac8…), DIR/READ halves, focus layer 21, steering at layer 20, rms 8.7; gate ≥ 0.80 at every ablated layer (0.82–0.88, 26 layers).

## What the file shows
**Verdict: instrument_failed** — the MC-ablate criterion "emo_all drop ≥ emo13_42 drop" fails for 4 of 6 emotions. The upper window (hs 21–31) is ablated and removes more norm per layer than hs 6–20 (0.165 vs 0.104; `n_layers_ablated` 26 vs 15 in every cell), yet it moves A's readout by nothing measurable: per-emotion mean differences of 2e−4 to 1e−3, a scenario bootstrap straddling zero for all six. Under that null a strict ≥ with no tolerance fails each emotion with p ≈ 0.5 and reaches three failures about two thirds of the time. The pre-registered rule is applied as written; the blocked fractions below are descriptive.

| emotion | none | emo13_42 | emo_all | rand_all | emo_all − rand_all | readout drop: emo13_42 / emo_all / rand | MC-ablate |
|---|---|---|---|---|---|---|---|
| desperate | +57 [+43, +72] | +54 | +54 | +54 | -0.2 [-9.0, +9.3] | 0.176 / 0.176 / 0.003 | FAIL |
| afraid | +122 [+97, +147] | +111 | +111 | +130 | -18.1 [-28.7, -7.5] | 0.079 / 0.079 / -0.000 | FAIL |
| happy | +66 [+42, +87] | +61 | +59 | +65 | -6.3 [-19.2, +5.3] | 0.045 / 0.044 / -0.000 | FAIL |
| calm (untestable) | +36 [-4, +76] | +43 | +32 | +37 | -4.5 [-20.4, +12.9] | 0.174 / 0.174 / -0.002 | pass |
| sad | +102 [+74, +129] | +72 | +75 | +106 | -31.5 [-48.2, -14.8] | 0.259 / 0.259 / -0.002 | pass |
| angry | +102 [+71, +132] | +98 | +94 | +85 | +9.2 [-11.1, +31.8] | 0.018 / 0.018 / -0.001 | FAIL |

Descriptive median blocked fraction 0.096 [−0.05, 0.21] over five testable emotions; afraid 15% and sad 31% reach significance, the same two as on the 27B (21%, 33%); the other three are too wide to tell agreement from disagreement, and angry's point estimate flips sign (+9 vs −18). MC-steer 6/6, but the manipulation is half as strong as on the 27B (A's readout gains +0.25 on average vs +0.50) and less specific: at the top dose 'desperate' is the modal readout for afraid, sad and angry, and afraid's readout (0.115) sits below the 1/6 chance floor. Random direction inert while removing more norm than the emotion direction (0.167 vs 0.137). Rewrite arms: only afraid, calm and sad fail the §4.4 check; but `text_keep` collapses relative to `none` for five of six, so changing the context alone removes most of the transfer. Gate: the class mean passes everywhere, the desperate class alone is 0.64–0.76.

## What can be inferred
- The phenomenon generalises: B's dose-response to A's steered affect is positive with CIs excluding zero for the same five emotions on a second model family and size. [supported by file]
- The rank-1 partial block for afraid and sad replicates on the 8B in size and in which emotions it touches. [supported by file, descriptive under the rule's verdict]
- The pre-registered MC-ablate criterion is not robust to a null: when the extra layers change the readout by nothing measurable, a strict ≥ on nested windows fails by chance. On this model the extra layers are ablated and remove more, and still change the readout not at all. [supported by file; a defect of the criterion, corrected after the critique from 'floating-point tie']
- Not comparable in size: the pool is smaller (50% kept vs 75%), the steering manipulation is half as strong and less specific, afraid's readout is below chance, and slopes are unnormalised probe scores. Only the pattern (which emotions transfer; which two block) is comparable. [scope; the critique's reading adopted]

## Status
Complete. Generality evidence for the deadline paper's claim 1 and claim 3 as a pattern, not a size. To-dos: a tolerance band or equivalence test in the MC-ablate criterion; per-class gate reporting; the driver's prereg lookup (empty stamp here as on the 27B).


### Blind critique — B1c on the 8B (independent agent; saw only the driver, the core functions, PREREG_B1c and the two result files)

# Blind critique 29 — B1c on Llama-3-8B-abliterated

**(1) Numbers.** All of C1–C3 checks out. `summary.<e>.none.present_slope`: desperate
+57.1 [+42.5,+71.7], afraid +121.7, happy +65.5, sad +102.0, angry +101.7, calm +35.6
[−4.2,+75.5]; `n_mc_steer_pass` 6; `direction_stability.ablated_layer_summary` .8216/.8721/
.8833 over 26 layers, `layers_below_gate` []. `emo_all_vs_rand_all.diff` afraid −18.106
[−28.71,−7.49] bf .149, sad −31.505 [−48.15,−14.84] bf .309; `decision.median` .0961
[−.0502,.2053]; `n_mc_ablate_fail` 4. Re-running `analyze(rows)` reproduces the stored
decision and verdict bit-for-bit.

**(2) The "tie" is misdescribed, and it is not a bug.** The readouts are **not** identical:
max |emo_all − emo13_42| over rows is 0.0538; per-emotion mean drops differ by 2e−4…1e−3,
four orders above float32 eps. Nor is the dirs dict unapplied — `n_layers_ablated` is 15 vs
26 in all 72 cells, `mask_hit_rate` 1.0, and emo_all's `removed_norm` exceeds emo13_42's in
**72/72** cells; the implied upper window (hs 21–31) removes **0.165/layer** against
0.104/layer in hs 6–20. The upper layers are ablated, remove more, and still move A's
readout by nothing measurable: a scenario bootstrap of the paired per-scenario difference
at α=1 straddles zero for all six emotions (largest |mean| 0.0013). On the 27B all six are
negative with CIs excluding zero (−0.0036 to −0.0129). So: a real null of the upper window,
decided by a strict `≥` with no tolerance band (b1c_alllayer.py:615) on nested windows —
under that null each emotion fails with p≈0.5, so ≥3 failures arrives ~66% of the time by
chance. "instrument_failed" is a defect of the pre-registered check, not fp noise and not a
pipeline fault. C2's conclusion survives; its stated reason does not.

**(3) The dose is not comparable, and the file says so.** `rms` 8.7 vs 81 is activation
scale and cancels. MC-steer does not: A's readout at α=1 reaches 0.115–0.741 on the 8B
(Δ +0.074…+0.372, mean +0.245) vs 0.397–0.955 on the 27B (Δ +0.147…+0.847, mean +0.501) —
half the manipulation. Worse, `ctx_readout_full` at α=1 makes **desperate** the modal
answer for afraid, sad *and* angry on the 8B (only afraid on the 27B): weaker *and* less
specific. afraid's 0.115 is below the 1/6 chance floor, so its drop (0.115→0.036) is a
below-chance quantity. Slopes are unnormalised probe scores, so "+57 vs +84" across models
is not a comparison.

**(4) Other.** The 0.80 gate is on the **class mean**. Per class, desperate is 0.640–0.764
at every ablated layer (0.755 at focus) against a 0.811 floor on the 27B; PREREG §3
requires per-class values reported for exactly this reason, and C1 omits it. `rand_all`
removes *more* norm than `emo_all` (0.167 vs 0.137; 27B 1.23 vs 0.66) — unmatched
perturbations (conservative, but unstated). Refusals ≤0.034, 0 quality exclusions. Pool
1079/2160 (50%) vs 1615/2160 (75%); the file records neither 2160 nor the leak/unparse
split.

**(5) C3 is wrong as written.** Only 3/6 are `text_uninterpretable` on the 8B (afraid .823,
calm .829, sad 1.247); desperate .486, happy .370, angry .452 **pass** §4.4. The right
objection: `text_keep` also collapses (−23…−75 vs none, 5/6 sig), so "context changed"
alone removes most of the transfer.

**(6) "Same two emotions"** is a true count over a 5-way pattern, but angry's estimate
flips sign (+9.2 vs −18.1), calm is significant on the 27B only, and the three 8B nulls
have CIs covering both 0 and the 27B point. Say "the same two reach significance; the rest
are too wide to tell agreement from disagreement".

**(7) Provenance.** `prereg` is empty in **both** files (`git_commit` "", `found` false) —
§8 unmet; git shows 122af4ce (19:02Z) and cc2dc329 (19:22Z) do predate the 27B start
(19:32:20Z). `predecessors` still lists the 27B runs. PREREG §7 lists "the 8B" under what
this does *not* test, and hs 6–20/6–31 come from `lo = round(0.2*n_layers)` at run time —
deterministic, never pre-specified. `acl_core_sha` differs (3da064ad→c1d065ca), but only by
the rank-k branch (acl_core.py:809–812), bypassed for 1-D dirs; driver bytes identical.

### Reconciliation (author)

- **(2) accepted, wording corrected:** not a floating-point tie but a real null of the upper window met by a strict ≥; the criterion's defect is a missing tolerance band. Registry `b1c8.verdict_and_block` rewritten.
- **(3) accepted:** the manipulation is half as strong and less specific on the 8B; "same five emotions transfer" is kept as a pattern statement, size comparisons dropped.
- **(4) accepted:** per-class gate values (desperate 0.64–0.76) and the random direction's larger footprint now stated.
- **(5) C3 corrected:** three of six rewrite arms fail §4.4; the stronger point is that `text_keep` collapses for five of six.
- **(6) adopted verbatim:** "the same two reach significance; the rest are too wide to tell agreement from disagreement".
- **(7) recorded:** empty prereg stamp in both B1c files (driver lookup bug); the 8B was listed under "not tested" in PREREG_B1c §7, so this is a replication of the design, not a pre-registered test; windows derived at run time.

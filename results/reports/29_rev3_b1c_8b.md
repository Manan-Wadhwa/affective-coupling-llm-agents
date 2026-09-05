# 29 · Rev-3 B1c's design on Llama-3-8B-abliterated — the transfer replicates; the rule returns instrument_failed on a tie

**Files:** `results/rev3/b1c_alllayer_llama3-abl.json` (432 arm-rows; provenance-stamped, revision dd67dd05; PREREG_B1c stamped inside), `b1c_cells_llama3-abl.json` · **Script:** `src/rev3/b1c_alllayer.py --model failspy/Llama-3-8B-Instruct-abliterated` on box 6, 10:33–11:20Z (2831 s; 5 s per arm-row). · **Status of the design:** a replication of a pre-registered design on a second model, not a new pre-registration; the 27B's rule and thresholds apply unchanged.

## What was run
As B1c: six arms (none, emo13_42 → here hs 6–20, emo_all → hs 6–31, rand_all, text, text_keep), four doses, three reps, 29 scenarios; a fresh probe pool for this model (2160 generated, 1079 kept; sha d1f52ac8…), DIR/READ halves, focus layer 21, steering at layer 20, rms 8.7; gate ≥ 0.80 at every ablated layer (0.82–0.88, 26 layers).

## What the file shows
**Verdict: instrument_failed** — the MC-ablate criterion "emo_all drop ≥ emo13_42 drop" fails for 4 of 6 emotions. The two windows produce the same A-span readout (drops 0.176 vs 0.176, 0.079 vs 0.079 …): layers 21–31 add nothing to the readout, so the ≥ comparison turns on floating-point noise. The pre-registered rule is applied as written and the blocked fractions below are descriptive.

| emotion | none | emo13_42 | emo_all | rand_all | emo_all − rand_all | readout drop: emo13_42 / emo_all / rand | MC-ablate |
|---|---|---|---|---|---|---|---|
| desperate | +57 [+43, +72] | +54 | +54 | +54 | -0.2 [-9.0, +9.3] | 0.176 / 0.176 / 0.003 | FAIL |
| afraid | +122 [+97, +147] | +111 | +111 | +130 | -18.1 [-28.7, -7.5] | 0.079 / 0.079 / -0.000 | FAIL |
| happy | +66 [+42, +87] | +61 | +59 | +65 | -6.3 [-19.2, +5.3] | 0.045 / 0.044 / -0.000 | FAIL |
| calm (untestable) | +36 [-4, +76] | +43 | +32 | +37 | -4.5 [-20.4, +12.9] | 0.174 / 0.174 / -0.002 | pass |
| sad | +102 [+74, +129] | +72 | +75 | +106 | -31.5 [-48.2, -14.8] | 0.259 / 0.259 / -0.002 | pass |
| angry | +102 [+71, +132] | +98 | +94 | +85 | +9.2 [-11.1, +31.8] | 0.018 / 0.018 / -0.001 | FAIL |

Descriptive median blocked fraction 0.096 [−0.05, 0.21] over five testable emotions; afraid 15% and sad 31% significant, the same two emotions as on the 27B (21%, 33%); MC-steer 6/6; random direction inert (rand vs none within ±17 everywhere). The rewrite arms sit far below `none` for both `text` and `text_keep` (e.g. afraid +41 and +67 against +122), so they are uninterpretable here as on the 27B.

## What can be inferred
- The phenomenon generalises: B's dose-response to A's steered affect is positive with CIs excluding zero for the same five emotions on a second model family and size. [supported by file]
- The rank-1 partial block for afraid and sad replicates on the 8B in size and in which emotions it touches. [supported by file, descriptive under the rule's verdict]
- The pre-registered MC-ablate criterion is not robust to a tie and returns instrument_failed when two windows give identical readouts; on this model the extra layers change the readout not at all. [supported by file; a defect of the criterion]
- Not comparable without care: the pool is smaller, the readout floors differ (afraid's A-span readout only 0.04 → 0.12 at the top dose), and rms is 8.7 vs 81. [scope]

## Status
Complete. Generality evidence for the deadline paper's claim 1 and claim 3; the tie defect goes to the §12 to-dos (use a strict margin or a two-sided equivalence in the MC criterion).

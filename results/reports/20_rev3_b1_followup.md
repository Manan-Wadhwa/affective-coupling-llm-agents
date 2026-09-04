# 20 · Rev-3 B1 follow-up — ceiling and token-level arms, seeded, gated at every ablated layer

**Files:** `results/rev3/b1_followup_qwen36-27b.json`, `b1f_cells_qwen36-27b.json` · **Script:** `src/rev3/b1_followup.py` (written and independently verified 2026-09-04) · **Model:** Qwen/Qwen3.6-27B rev 6a9e13bd · **Registry:** `b1f.verdict_mixed`, `b1f.blocked_fraction`, `b1f.ceiling_equals_emo`, `b1f.text_arm`, `b1f.ablated_layers_gated`

## What was run
B1's exact probe pool and features (directions identical; focus gate reproduces to every digit). Arms none / emo / rand / ceiling / text; B seeded with common random numbers across arms; a fresh random direction per (emotion, rep); split-half gate at all 30 ablated layers; a pre-declared sign-aware rule with a per-emotion MDE. Tables in `results/rev3/README.md`.

## What the file shows
Ablated-layer gates 0.882–0.920. MC-ablate 6/6 (separated 5/6). emo − rand: afraid +0.25 [+0.16, +0.35], sad +0.37 [+0.19, +0.56] of the none slope; desperate/happy/angry null; calm untestable. emo − ceiling crosses zero for 5/6. text − none significant for 4/6. Verdict `mixed` (2 block, 0 anti-block, 1 untestable).

## What can be inferred
- Residual ablation of the emotion direction at layers 13–42 blocks a minority of transmission: detectably a quarter to a third for two emotions, not detectably for three (MDE admits up to ~35%). [supported by file]
- Ablating everywhere at those layers does no better than ablating A's span, so that is the ceiling for residual ablation there; B recomputes affect above the window. [supported by file]
- The token-level rewrite removes as much or more, but its manipulation check fails for afraid and calm, so the text arm is not yet a clean channel measurement. [supported by file]
- The completed B1's happy anti-block was noise: absent under seeded, common-random-number generation. [supported by file]
- Where the remaining transmission travels is not localised by this design. [needs: ablation spanning layers 43–63; an affect-preserving control rewrite]

## Status
Complete. Resolves RESEARCH_PLAN §1.3 as "partial blocking" (§13.5). Nothing committed.

### Independent critique — B1 follow-up (blind: saw only the scripts and the JSONs)

## 1. Verification of the machinery

**Seeding / CRN.** `gen_seed` (`b1_followup.py:170-185`) drops `arm` from the key when `part=="B"`. Confirmed in the data: in **54/54 cells all five arms share one `gen_seed`** (e.g. `$.cells["sad/1.0/0"][*].gen_seed` = 2002200872). It is seeded per batch immediately before `generate` (`:745`), and per forward in `readout` (`:793`) — defect (a) is fixed. But CRN ≠ identical draws: the arms differ in logits, so the streams diverge at the first differing token. At α=0, `none` vs `emo` `B_present_e` is **not** identical (afraid r=0.951, sad r=0.836; text r=0.63-0.91). "Common random numbers" is true of the seed, not of the realised noise — the variance reduction is real but partial, and the file does not quantify it.

**Per-layer gate.** `direction_stability_by_layer` holds all 30 ablated indices plus focus (`:622-631`). `$.ablated_layer_stability` min 0.8816 (hs 14), median 0.8984, max 0.9202 (hs 37); `$.layers_below_gate` = `[]`. `$.direction_stability.used` is byte-identical to the completed run (0.9073818377260926) — same pool, same directions, as claimed. Defect (b) fixed.

**Fresh random.** `rand_dirs_for` (`:659-667`): **18 distinct `rand_dir_seed`s over 18 (emotion, rep) keys**, `None` on every non-rand row. Defect (c) fixed — though 3 draws per emotion is still a small sample of direction space.

**Ceiling mask.** `M = enc["attention_mask"].bool()` (`:739`, `:789`) — every non-pad prefill position. `mask_n_positions` at α=1.0: ceiling **91.9** vs emo/rand **32.7**, i.e. A's span is 36% of the prompt and the ceiling covers the rest. Per-position `removed_norm` is *lower* for ceiling (0.353 vs emo 0.511) because template/instruction tokens carry little of the direction; **total** removed norm is 32.6 vs 16.7 — ~2× more. The authors are right that the pair must be read together. Two residual problems: `mask_hit_rate` is set to `1.0` by fiat for ceiling (`hits = len(batch)`, `:739`) so it verifies nothing; and `rand` still removes 1.5× the norm `emo` does over the same positions — "norm-matched" remains a statement about unit direction length only.

**Text arm.** `REWRITE_PROMPT` (`:118-120`) never names an emotion — no target leakage. B's prompt is rebuilt from the rewritten line (`:830-832`) and runs with no hooks (`:710-711`). `rewrite_failed` (`:824`) counts **only** `len(pr.split()) < 3` or byte-identity with the original. **A refusal passes both tests and is substituted into A's slot.** The driver records `rewrite_refusal_frac` (`:839`) but `analyze` never reads it. The consequence is dose-aligned:

| emotion | refusal frac α=0 / 0.5 / 1.0 | `rewrite_failed` α=1.0 |
|---|---|---|
| afraid | 0.000 / 0.138 / **0.471** | 0.138 |
| desperate, sad | 0.011 / 0.080 / 0.092 | ≤0.023 |
| happy, calm, angry | 0.000 | ≤0.069 |

At α=1.0, **47% of afraid's "neutral rewrites" are refusals** (61% flagged degenerate). The text arm's afraid slope is therefore substantially a measure of how often the rewriter broke, and it breaks *with dose*.

## 2. Recomputation (scenario-clustered, dose-paired bootstrap; 29 scenarios × 3 reps)

`paired_slope_contrast` (`acl_core.py:994-1017`) still has no block argument, though `analyze`'s docstring (`:244`) says "scenario-blocked". Clustering changes **two headline calls**:

- **angry `emo_vs_rand`: −24.10 → [−44.90, −3.41], sig=True** (file `$.summary.angry.emo_vs_rand.ci` = [−51.95, +2.70], sig=False). `n_block` becomes **3**, not 2, and angry is not a null.
- **angry `emo_vs_ceiling`: +18.00 → [−2.65, +39.39], sig=False** (file [+1.35, +35.69], sig=True). C3's sole exception disappears; emo−ceiling then crosses zero **6/6**.

Everything else is unchanged. That the counts in both C2 and C3 flip on a defensible variance choice is the strongest objection to both as worded.

**MDE.** `$.summary.<e>.mde` uses `C.mde(sd, 87)` on top-dose `none` scores (`:322-328`) — the wrong estimand (a two-arm mean difference, not a slope contrast) and the wrong n (87 samples, 29 independent scenarios). From each contrast's own bootstrap, the detectable emo−rand slope difference is: afraid 24.7 (**13%** of the none slope), sad 26.5 (27%), angry 39.1 (19%), desperate 23.4 (30%), happy 28.8 (**50%**), calm 36.4 (96%). CRN makes the design *more* sensitive than the reported MDE says (0.44-0.67×), so the file understates its own power while overstating what the nulls exclude.

## 3. Does emo ≈ ceiling mean "B rebuilds above the window"?

No. `$.summary.<e>.mc_ablate.drop_by_arm` shows ceiling is a strictly stronger lesion on the readout in all six (e.g. sad 0.456 vs 0.372; happy 0.312 vs 0.181) — so the equality of B-slopes is not because the two arms are the same intervention. But **`ceiling` extends coverage in positions, not in layers**: `abl_dec` is identical (`:592`, hs 13-42). An arm that saturates positions and changes nothing tells you the binding constraint is not positional; it is silent between "rebuilt at layers 43-63", "never carried by this one direction" (a rank-1 lesion of 5120 dims), and "carried non-linearly". "B rebuilds affect above the window" names one of three and tests none. The needed arm — ablate at *all* layers — does not exist in this run.

## 4. Is C5 licensed?

- **"~25-37% for two of five"** — `$.summary.afraid.blocked_fraction.emo_vs_rand` 0.252 [0.161, 0.347] and `sad` 0.372 [0.186, 0.563] are correct as printed, but the count is 3/5 under clustering, and `blocked_fraction` divides by a `none` slope whose own CI is ignored (`blocked_fraction:231-238`).
- **"minority"** — defensible for those two; not established for the three nulls, whose intervals are two-sided. C2's "consistent with 0-35% blocking" silently drops the negative half (desperate range [−0.241, +0.178], happy [−0.351, +0.345], angry [−0.013, +0.258]): they are equally consistent with up to 24-35% *amplification*.
- **"rebuilt above the window"** — untested (§3).
- **"travels with A's tokens"** — the text arm is the only evidence and it does not carry it: for afraid the rewrite's own readout drop is **0.011** (the words still read as afraid) while the slope falls 54, and 47% of those "rewrites" at α=1.0 are refusals; for calm the drop is 0.058 and calm is *untestable* by the driver's own rule (`:341-343`, `none` slope n.s.) yet is counted among C4's four. Separating token identity from context would need an arm holding A's tokens fixed while blocking B's attention to them, or a paraphrase whose readout drop is verified to match the emo arm's. Neither exists. The text arm also changes the context in which `score_B` pools B's reply (`:848-850`), so `text_vs_none` confounds transmission with the scoring context.
- **Nearest defensible sentence:** *"Projecting the emotion direction out of A's span at layers 13-42 removes most of the model's own readout of A's affect (drop 0.11-0.37, 6/6, vs ≤0.006 for norm-matched random) but reduces B's dose-response by only 25-37% in two of five testable emotions (three under a scenario-clustered bootstrap); extending the mask to every prefill position removes strictly more readout and changes B no further, so positional coverage is not the limit. Whether the remainder is rebuilt above the ablation window, carried non-linearly, or simply absent from this rank-1 direction is not distinguished by this design."*

## 5. Provenance and residual defects

`$._provenance.code_sha` = `b0568e500db40c0c` (set, via `C.code_hash` at `:535`), `gen_seed_rule` recorded in config, control pointers derived from the AST (`fn_lines`/`block_lines`, `:150-167`) so they cannot go stale — defect (f) fixed. Still open: contrasts unclustered; `blocks(e, arm, doses[0])` reused for all doses (`:278`); `separated` still an unpaired CI test (`:313`, false for afraid again); the OLS "slope" over (0, 0.5, 1.0) is still exactly y(1.0) − y(0.0), so the midpoint dose carries zero weight and non-monotonicity (happy: 221.6/315.4/279.4) is invisible; no `text_vs_emo` contrast although C4 asserts "for angry more than residual ablation does"; refusal-contaminated rewrites are neither excluded nor flagged in `analyze`; `--selftest` (`:399-444`) plants *identical* noise across arms, so it verifies plumbing, not the false-positive rate. Text quality is clean (degeneracy/refusal ≤0.069, perplexity 10.95-18.69).

## 6. Verdicts

| Claim | Verdict | One line |
|---|---|---|
| C1 all 30 layers gate 0.882-0.920 | **SUPPORTED** | Matches `$.ablated_layer_stability` and `$.layers_below_gate` = []; caveat: the gate is on the class mean, and afraid at hs 14 is 0.811 (`per_class`). |
| C2 emo−rand 0.25/0.37, nulls, "mixed" | **SUPPORTED-WITH-CAVEATS** | Numbers verified, but angry becomes a third blocker under a scenario-clustered bootstrap, and "0-35% blocking" reports one side of a two-sided interval. |
| C3 emo ≈ ceiling ⇒ partial blocking | **NOT SUPPORTED as inference** | The 5/6 description holds (6/6 clustered — the angry exception is an artefact of the unclustered CI), but ceiling extends positions, not layers, so "B rebuilds above the window" is untested. |
| C4 text arm reduces 4/6 | **NOT SUPPORTED as worded** | afraid is 47% refusals at α=1.0 with a dose-aligned failure rate that `rewrite_failed` (0.138) hides; calm is untestable by the paper's own rule; "more than residual ablation" has no contrast. |
| C5 proposed paper sentence | **NOT SUPPORTED as worded** | "minority" is defensible for two emotions; "rebuilt above the window" and "travels with A's tokens" are not tested by any arm here. |

## Reconciliation
Accepted in full; the record was corrected as follows.
- **Counts flip on the variance choice.** Under a scenario-clustered, dose-paired bootstrap angry becomes a third blocker (−24.1 [−44.9, −3.4]) and the ceiling arm's one exception vanishes (emo − ceiling crosses zero 6/6). The registry now states both readings. `paired_slope_contrast` is still unblocked — the same to-do as this morning, now load-bearing twice.
- **The nulls are two-sided.** "Consistent with 0–35% blocking" dropped the negative half; desperate, happy and angry are equally consistent with up to 24–35% amplification. Reworded.
- **The MDE in the file is the wrong estimand.** From each contrast's own bootstrap the detectable emo − rand difference is 13% (afraid), 19% (angry), 27% (sad), 30% (desperate), 50% (happy), 96% (calm) of the none slope; common random numbers make the design more sensitive than the recorded MDE says.
- **The text arm's afraid result is refusal-contaminated**: 47% of its "neutral rewrites" at α = 1 are refusals (61% flagged degenerate), and `rewrite_failed` (0.138) does not catch them; calm is untestable by the driver's own rule yet was counted; there is no text-vs-emo contrast behind "more than residual ablation". Registry text rewritten to two emotions (sad, angry) with the contamination stated.
- **"B rebuilds affect above the window" is untested.** The ceiling arm extends positions, not layers; equality of B-slopes is silent between "rebuilt at layers 43–63", "never carried by this rank-1 direction", and "carried non-linearly". The plan's §13.5 sentence is replaced by the critic's.
- Common random numbers share the seed but the streams diverge at the first differing token (α = 0 none-vs-emo r = 0.84–0.95); variance reduction is real but partial and unquantified. afraid's per-class gate at hs 14 is 0.811.
Verdicts: C1 SUPPORTED · C2 SWC · C3 NOT SUPPORTED as inference · C4 NOT SUPPORTED as worded · C5 NOT SUPPORTED as worded.

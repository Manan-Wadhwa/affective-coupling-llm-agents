# 23 · Rev-3 A2 follow-up — second fixed-penalty anchor (N_REF = 2000), layer 43; layer 54 partial

**Files:** `results/rev3/a2_followup_qwen36-27b-nref2000.json` (layer 43 complete, provenance-stamped 20:53Z; code_sha ccd27d64, acl_core 23c758a9), `a2f_cells_qwen36-27b-nref2000.json` (checkpoint: 22 of 48 layer-54 cells) · **Script:** `src/rev3/a2_followup.py --n-ref 2000 --layers 43,54` on box 4 (`sb-ea69c0d19be26d97`), CPU · **Companion:** report 21 (anchor 600, `a2_followup_qwen36-27b.json`), which shares the ten split seeds.

**Lost.** Layer 54 stopped being pulled at 22:24Z when the operator's laptop suspended; the box's lease ended before it woke (05:51Z). The two queued sweeps (layer 54 at anchor 600; the 8B fixed-penalty arm on layers 14, 21, 27) were never pulled and are lost. Whether they ran on the box is unknown.

## What was run
Same battery as report 21 with the per-sample penalty held at its n = 2000 value: `C_n = 0.5 · 2000 / n`, so the per-sample weight is 1/2000 at every n (the 600 anchor holds 1/600). `logreg_lam` at n = 2000 is therefore the same fit as plain `logreg` at n = 2000 (0.5664 in both).

## What the file shows
Layer 43, split-half cosine of the direction (10 disjoint splits; mean over six classes):

| n per half | logreg | logreg_cv | **logreg_lam, anchor 2000** | logreg_lam, anchor 600 (report 21) | paired Δ (2000 − 600), of 60 negative | dom |
|---|---|---|---|---|---|---|
| 150 | 0.471 | 0.506 | **0.574** [0.562, 0.586] | 0.581 | −0.007, 55/60 | 0.732 |
| 300 | 0.525 | 0.559 | **0.578** [0.565, 0.589] | 0.596 | −0.019, 60/60 | 0.842 |
| 600 | 0.575 | 0.578 | **0.554** [0.544, 0.565] | 0.575 | −0.021, 59/60 | 0.915 |
| 1200 | 0.568 | 0.601 | **0.556** [0.552, 0.561] | 0.578 | −0.022, 51/60 | 0.955 |
| 2000 | 0.566 | 0.608 | **0.566** [0.561, 0.572] | 0.581 | −0.015, 36/60 | 0.974 |

Standardised space: anchor 2000 = 0.599, 0.601, 0.576, 0.577, 0.588; anchor 600 = 0.606, 0.619, 0.597, 0.599, 0.603. Cross-estimator CAA-raw ↔ logreg-raw at n = 2000: 0.427 [0.421, 0.431], identical to report 21 (same cells).

Layer 54, partial (checkpoint, no provenance stamp): plain `logreg` raw 0.410, 0.484, 0.523, 0.573, 0.564, 0.554 at n = 75…2000 (peak at 600, then declining, the same ordering violation as layer 43); `logreg_cv` 0.438, 0.528, 0.567, 0.582, 0.584 up to n = 1200. No `logreg_lam` or `dom` cells reached layer 54.

## What can be inferred
- With the per-sample penalty fixed at a second, weaker value the logistic direction again shows no net gain in reproducibility from n = 150 to 2000 (paired −0.008, 3/10 positive). The curve is not flat: the dip 300 → 600 (−0.024, 1/10) and the recovery 600 → 2000 (+0.012, 8/10) are both significant. [supported by file; wording corrected after the critique]
- The plateau level is mildly anchor-dependent: the weaker penalty sits lower at every n (0.003 at n = 75, 0.007–0.022 from 150 up; 41–60 of 60 paired differences negative below n = 2000, 36 of 60 at 2000), so "≈ 0.58" in report 21 is an anchor-specific number; "0.55–0.60, never approaching difference-of-means" is the anchor-robust statement. The stronger penalty gives the higher cosine, so the stability-maximising penalty is unbracketed; anchors below 600 are untested. [supported by both files; corrected after the critique]
- Layer 54 plain-logreg replicates the peak-then-decline shape from a second depth, on 22 cells and without provenance; it says nothing about the anchor, because plain logreg does not depend on it and no `logreg_lam` or `dom` cell reached layer 54. [partial; descriptive only]
- The 8B fixed-penalty arm remains untested. [needs: a rerun]

## Defects to fix before quoting
`model` and `model_revision` are empty strings in the file (the driver reads them from the features' `source`, which carries none); the recorded `c_n_formula` sentence has the right formula (`0.5 * 2000 / n_fit`) but stale prose ("held at its n=600 value … C_600 == 0.5 exactly"). Both are in the registry notes. The critique adds: the stale prose is also baked into the checkpoint config; the driver's selftest asserts "C_n at n = 600" on a slice that moves with the anchor (lines 630–633); and the ragged layer-54 partial has `logreg` at n = 2000 but `logreg_cv` only to 1200.

## Status
Layer 43 complete; layer 54 partial; 8B not run. Feeds RESEARCH_PLAN §12 (to-do "second fixed-λ anchor" done; "8B fixed-λ arm" and "layer 54" reopened).


### Delta critique — N_REF=2000 anchor (blind: two follow-up JSONs, the checkpoint, the driver)

**(2) The anchor really changes the fit.** `--n-ref` sets the global (a2_followup.py:689–691) that `fit_logreg_lam` reads at call time (line 158). `logreg_lam` at n=600 differs between files by up to 0.0504 per seed (0.5539 vs 0.5752), and in the new file `logreg_lam@2000 == logreg@2000` to **exactly 0.0** per seed (C_2000 = 0.5). `logreg`, `logreg_cv`, `dom` are identical across files — the right invariance. But `code_sha` differs (ccd27d64… vs ae3621bb…): different driver builds, with those shared cells the only reassurance.

**C1 — numbers right; "no rise" holds end-to-end, fails in the middle.** 0.5741/0.5776/0.5539/0.5561/0.5664 ✓, range 0.0237 ✓, dip at 600 ✓. Paired over the 10 shared splits: 150→2000 −0.0077 (t −1.08, 3/10); but 300→600 −0.0237 (t −3.47, 1/10) and **600→2000 +0.0124 (t +2.01, 8/10)**, 1200→2000 +0.0102 (t +2.34). Non-monotone — significant dip, significant partial recovery — not flat.

**C2 — the range and one count are wrong.** Gaps −0.0071/−0.0188/−0.0213/−0.0218/−0.0146 for n=150…2000 ✓, but n=75 is −0.0029, so "0.007–0.022 **at every n**" is false. Negative counts per (seed×class): 41/60 (n=75), 55, 60, 59, **51/60 (n=1200)**, 36/60 ✓ — "55–60 for n≤1200" fails at n=75 and n=1200. Both plateaus 0.55–0.60 vs dom 0.97 ✓.

**(3) Reading.** With per-sample λ fixed both halves estimate the same population λ-minimiser, so the cosine should climb as n/d rises 0.015→0.39. It does not — that, not "fixed target", is the finding. A ~0.02 shift for a 3.3× λ change means the level is no penalty artefact; but two anchors is a 2-point check and both sit on the same side — the *stronger* penalty gives the *higher* cosine (0.581 vs 0.566 at n=2000) — so the λ-maximum is unbracketed and N_REF < 600 untested.

**C3 — correct but empty.** 0.4099/0.4835/0.5231/0.5730/0.5639/0.5541 ✓, peak at 600 ✓. But the 22 layer-54 cells hold only `logreg` and `logreg_cv` — **no `logreg_lam`, no `dom`** — and plain logreg is anchor-independent (same values as the A2 sweep), so the partial says nothing about the anchor.

**C4 + new defects.** ✓ `model`/`model_revision` empty; line 691 replaces only the literal "0.5 * 600 / n_fit", leaving "held at its C=0.5, n=600 value … C_600 == 0.5 exactly" — false here (C_600 = 1.667). New: (a) that prose is baked into the **checkpoint config**, hence the config hash — harmless, but now wrong in three places; (b) at L54 `logreg_cv` lacks n=2000 while `logreg` has it, inviting a 2000-vs-1200 read off a ragged partial; (c) the selftest slices `[:N_REF]` but asserts "C_n at n=600" (lines 630–633), so its witness silently moves with the anchor.

### Reconciliation (author)

- **C1 accepted:** "no rise" is now "no net rise, with a significant dip and partial recovery" in the registry (`a2f2.lam_anchor2000_no_rise`), this report and the log.
- **C2 corrected:** the gap range and the negative counts were misquoted at n = 75 and n = 1200; the registry text now lists all six gaps and counts. The critic's reading that the stability-maximising penalty is unbracketed is adopted as a to-do (anchor < 600).
- **C3 accepted** as stated; the partial is descriptive and anchor-blind.
- **C4 and new defects** recorded here and in RESEARCH_PLAN §12 to-dos (format `N_REF` into the prose, fix the selftest witness, flag ragged partials); the driver is not edited post hoc for these runs.

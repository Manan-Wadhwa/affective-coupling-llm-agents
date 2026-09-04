# 21 · Rev-3 A2 follow-up — tuned C, fixed effective penalty, both spaces, like-for-like cross-estimator

**Files:** `results/rev3/a2_followup_qwen36-27b.json`, `a2f_cells_qwen36-27b.json` · **Script:** `src/rev3/a2_followup.py` (written and verified 2026-09-04), features via `src/rev3/a2_regen_feats.py` from A2's saved pool · **Model:** Qwen/Qwen3.6-27B · **Registry:** `a2f.focus_logreg_fixed_lambda_flat`, `a2f.focus_tuned_c_does_not_close_gap`, `a2f.std_space_matches_raw`, `a2f.caa_raw_vs_logreg_raw`

## What was run
On A2's pool (n = 4253) at layers 16 and 43 (54 lost to the lease at 10/48 cells): split-half stability over ten disjoint splits for `logreg` (C = 0.5), `logreg_cv` (C tuned per fit), `logreg_lam` (C_n = 0.5·600/n, fixed per-sample penalty) and `dom`, in raw and standardised space; a ten-subsample cross-estimator block including CAA's raw mean difference; Ledoit–Wolf λ per n. Tables in `results/rev3/README.md`.

## What the file shows
Focus layer, n = 2000, raw: logreg 0.566, logreg_cv 0.608, logreg_lam 0.581 (flat from n = 150: 0.581), dom 0.974. CAA raw ↔ logreg raw 0.427. Standardised space within +0.02–0.03 of raw.

## What can be inferred
- Holding the per-sample penalty fixed removes the rise in the logistic curve: the direction does not become more reproducible with n on this pool. [supported by file]
- Tuning C per fit improves it by ~0.04 and does not approach difference-of-means. [supported by file]
- The space of the cosine does not change the picture. [supported by file]
- The counter-paper's near-identity of logistic and mean-difference directions does not hold here in its own units, for this trait and design. [supported by file, different trait]
- Whether the 8B behaves the same under fixed λ is untested. [needs: the same arm on the 8B]

## Status
Complete for two of three layers. Feeds RESEARCH_PLAN §2.1 item 2 (now "established on the 27B with the schedule controlled").

### Delta critique — A2 tuned-C follow-up (blind: saw only the scripts and the JSONs)

**(1) Verification.** All quoted numbers reproduce. `by_layer.43.stability.logreg_lam.{150,300,600,1200,2000}.raw.mean` = 0.5812, 0.5964, 0.5752, 0.5779, 0.5810, ci `[0.5769, 0.5848]`; layer 16 spans 0.4764–0.5040 ("0.48–0.50", borderline); dom raw 0.7322 → 0.9736. `logreg_cv` 0.6080 `[0.5902, 0.6253]`; `caa_raw|logreg_raw` 0.4268 (L43) / 0.4982 (L16).

**The C_n formula is right and the equivalence holds on the real data.** sklearn minimises `0.5‖w‖² + C·Σloss`; dividing by `C·n` gives penalty `1/(2C_n·n)` on the mean loss, and `C_n·n = 0.5·600 = 300` is constant (a2_followup.py:158). Max |per-seed difference| between `logreg_lam` and `logreg` at n=600 is **exactly 0.0** in both spaces at both layers. Note `equivalence_check` itself is only synthetic ("d=120"); this real-data identity is the stronger check.

**(2) It is not flat.** Paired over the 10 shared seeds, `logreg_lam` raw: L43 **300→600 = −0.0212 (t −4.55, 0/10 positive)** and 300→2000 = −0.0154 (t −2.86, 2/10); L43 150→2000 = −0.0002. L16 is a significant **rise**: 300→2000 = +0.0242 (t +3.80, 10/10). So the curve peaks at n=300 and dips at L43, and climbs at L16 — "flat" is wrong at both layers. What survives is the magnitude contrast: |Δ| ≤ 0.028 over a 13× range against dom's +0.1318 (300→2000, t +44.6) and +0.2414 (150→2000).

**Anchor.** N_REF=600 is arbitrary and only one λ was run. Holding per-sample λ fixed makes the target a *fixed* population minimiser, so the cosine must → 1 asymptotically; a 0.58 plateau at n=2000, d=5120 shows we are still in the proportional regime — that is slow convergence, not non-convergence. Anchoring at n=2000 weakens λ 3.3×, and A2 shows weaker C lowers stability at large n (0.5664 at C=0.5 vs 0.5749 at C=0.05), so the plateau's *level* is set by the anchor. Only its flatness could be anchor-free, and that is untested.

**(3) "Tuning does not close the gap" — overreaches.** The `warnings` block records 240× "The default value of the parameter 'scoring' will change from None, i.e. accuracy" — **CV selects C for accuracy, not direction stability**, and this study shows those diverge (mass_mean_cov: 0.937 decode / 0.176 cosine). The grid is decade-spaced; the trend runs toward smaller C (0.5→0.5664, 0.05→0.5749, CV→0.6080), and 0.005 was never run as a fixed value. `logreg_cv` also refits C per half, adding disagreement the fixed arms lack, so 0.608 may understate. Ridge is a weak data point (different loss). Also, tuning *does* move it +0.042 — ~10 % of the 0.41 gap.

**(4) Space.** Gaps run **0.0022 to 0.0292, with 27 of 48 cells outside [0.02, 0.03]** — dom at n=2000 is +0.002. "Within +0.02–0.03 for every estimator and n" is false; "std is always higher, never by more than 0.03, shrinking as the cosine → 1" is true, and the conclusion stands.

**(5) CAA.** Much closer, and the file's own `caa_raw|dom_raw = 0.7219` proves the earlier 0.59 compared two mis-specified objects. Still not the published quantity: `logreg_raw` is `C_std/sd` (line 303) — the raw-space image of a fit on *standardised* features, not a logistic fit on raw activations — and a 6-way softmax row is a contrast against five classes jointly, not CAA's binary one-vs-rest. **The CI is not a sampling CI**: 10 draws of n=2000 from 4253 overlap by ~941 items (47 %), yet `_agg_pairs` stamps `degenerate` only when `n_pool ≤ cross_n` (line 394), so nothing is flagged. Right idea, wrong threshold.

**(6) Provenance.** `code_sha` = `ae3621bbfc2d8e6a` — the gap I flagged twice is **fixed**. But `model` and `model_revision` are **empty**: the run reads `source.feats` and records nothing about which model produced the activations. And `config.layers = [16,43,54]` while `by_layer` has two — **no partial flag anywhere** in the file. `lw_lambda` now confirms the covariance mechanism directly: λ falls 0.650→0.056 (L16) and 0.637→0.057 (L43) across n=75→2000 at fixed d=5120 (one seed, no CI).

| Claim | Verdict | One line |
|---|---|---|
| **C1** | **NOT SUPPORTED as worded / SUPPORTED in substance** | Numbers exact and the C_n fix is verified bit-for-bit at n=600, but the curve is not flat (L43 300→600 t −4.55; L16 300→2000 t +3.80); one λ anchor sets the plateau level, untested. |
| **C2** | **SUPPORTED-WITH-CAVEATS** | True for this decade-spaced grid selected by *accuracy*; C=0.005 untested as a fixed value, per-half refitting inflates disagreement, and tuning does move it +0.042. |
| **C3** | **NOT SUPPORTED as worded** | Gaps span 0.002–0.029 with 27/48 outside the quoted band; the conclusion ("space was not the story") nonetheless holds. |
| **C4** | **SUPPORTED-WITH-CAVEATS** | 0.4268/0.4982 verified and a real improvement on A2's 0.59, but `logreg_raw` is still a standardised fit's image and the contrast is 6-way, not binary; the CI has no sampling content at 47 % draw overlap. |

**Nearest defensible sentence for Paper A:** *"At layer 43 of the 27B, with the per-sample L2 penalty held at its C=0.5, n=600 value, the multinomial logistic row's disjoint split-half cosine moves by less than 0.03 over n=150→2000 (0.581→0.581) while difference-of-means rises 0.732→0.974 on the identical splits; cross-validated C reaches only 0.608. Over the reachable range the logistic direction shows no material convergence, though n≤2000 at d=5120 remains a proportional-regime observation, not an asymptotic one."*

## Reconciliation
Accepted; substance stands, wording corrected.
- **"Flat" withdrawn.** Paired over seeds the fixed-penalty curve dips 300 → 600 at the focus layer (t −4.6) and rises at layer 16 (t +3.8); what survives is that it moves by less than 0.03 over n = 150 → 2000 while difference-of-means moves +0.24 on identical splits. Registry, README and plan now say "moves by less than 0.03", not "flat".
- **Proportional-regime caveat added.** With the per-sample penalty fixed the target is a fixed population minimiser, so the cosine must approach 1 eventually; a 0.58 plateau at n = 2000, d = 5120 is *no material convergence over the reachable range*, not non-convergence in the limit. One anchor (n = 600) was run; the plateau's level depends on it. Paper A's sentence uses the critic's wording.
- **"Tuning does not close the gap" softened**: cross-validation selected C for accuracy, not stability; the grid is decade-spaced with the trend toward smaller C and 0.005 never run as a fixed value; tuning moves the number by +0.042, about a tenth of the gap.
- **Space claim reworded**: gaps run 0.002–0.029, 27 of 48 cells outside the "0.02–0.03" band; "standardised is always higher, never by more than 0.03, shrinking as the cosine → 1" is what the file shows.
- **CAA comparison still not the published quantity**: the raw logistic direction here is the raw-space image of a standardised fit and a 6-way row, not a binary probe on raw activations; the 10-draw CI has no sampling content at 47% overlap and the degenerate flag's threshold is wrong.
- **Provenance**: `code_sha` set; `model`/`model_revision` empty (the driver reads features, not a model); no partial-run flag although `config.layers` lists three and `by_layer` holds two. λ 0.65 → 0.06 with n confirms the covariance-estimator mechanism directly.
Verdicts: C1 NOT SUPPORTED as worded / SUPPORTED in substance · C2 SWC · C3 NOT SUPPORTED as worded · C4 SWC.

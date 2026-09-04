# 19 · Rev-3 A2 — estimator battery on Llama-3-8B-abliterated

**Files:** `results/rev3/a2_estimator_llama3-abl.json`, `a2_cells_llama3-abl.json` · **Script:** `src/rev3/a2_estimator.py` (patched 2026-09-04) · **Model:** failspy/Llama-3-8B-Instruct-abliterated rev dd67dd05 · **Registry:** `a2_8b.focus_dom_split_half`, `a2_8b.focus_logreg_split_half`, `a2_8b.nd_anomaly_reproduces`, `a2_8b.focus_logreg_vs_dom_cos`

## What was run
The same sweep as report 17's completed run, on the 8B: k = 220, 3898 kept of 7920 (29% leak), seven depths of 32 (focus 21), n per half up to 1200, ten splits, nine estimators, raw-space cosine.

## What the file shows
Focus layer: dom 0.941, logreg 0.286 at n = 1200; logreg peaks at n = 300–600 and falls at 1200 on 5 of 7 depths; dom 0.925–0.941 on all depths; logreg↔dom 0.382; decode 0.803 vs 0.731; n/d 0.95 everywhere; effective rank 126–240; class SNR 1.4–1.6. Full tables in `results/rev3/README.md`.

## What can be inferred
- The estimator gap is larger on the 8B than the 27B and present at every depth. [supported by file]
- The plan's n/d anomaly reproduces on one pipeline: better n/d, worse logistic reproducibility, similar dom. n/d is not the mechanism. [supported by file]
- Whether the logistic curve's peak-then-fall is an estimator property or the fixed-C schedule is not separable here. [needs: tuned-C arm on the 8B]
- Why the 8B's activations give a lower-rank, lower-SNR regime is not identifiable from this file. [needs: model comparison beyond diagnostics]

## Status
Complete. The 8B re-measurement that `stability.postfix_gate_8b` was waiting for exists; that entry stays pending until its 0.965 sentence is retired.

### Delta critique — 8B A2 run (blind: saw only the scripts and the JSONs)

**1. Numbers.** All verified against `a2_estimator_llama3-abl.json`: `headline` = dom 0.9413 (ci `[0.9391,0.9434]`), logreg 0.2860, decode 0.8034/0.7308, logreg↔dom 0.3824 (range 0.3632–0.4181 ✓). n=2000 is indeed unreachable (2·2000 > n_pool 3898, a2_estimator.py:149). Two small slips: dom at n=1200 spans 0.9252–**0.9415** (hs27), so "0.925–0.941" quotes the focus layer as the max; and the fall at 1200 holds at **6 of 7** depths, not 5 — only hs11 rises (+0.0021). The claim understates its own evidence.

**Peak-then-fall is resolvable here, and it is a decline.** Paired over the 10 shared seeds, logreg 600→1200: hs21 −0.0204 (t −3.55, 9/10 negative), hs24 −0.0203 (t −3.17), hs27 −0.0137 (t −1.96); pooled over the six declining depths −0.0146, t −2.82, **10/10 seeds negative**. Unlike the 27B, this is not noise.

**And `logreg_c005` does not rescue it — it declines harder**: hs21 −0.0255 (t −6.24, 10/10), pooled −0.0210 (t −5.49, 10/10), negative at **all seven** depths. This directly contradicts the 27B, where `logreg_c005` rose at 6/7. So my "fixed-C artifact" objection does **not** transfer; C1 is stronger on this model. The cost is that the two models now disagree about the mechanism, which weakens calling them a replication.

**2. Matched-n.** Comparing 0.286 (n=1200) with 0.566 (n=2000) is not like-for-like, but the conclusion survives: at **n=1200 on both**, logreg 0.2860 vs 0.5678 (gap 0.282, versus 0.280 unmatched) and dom 0.9413 vs 0.9552 (gap 0.014, *better* for "similar" than the quoted 0.033). Restate at matched n; nothing else changes.

**3. Attrition.** The quoted 2322 leaks is only part of it: k=220 ⇒ 7920 jobs, 3898 kept = **49.2 % retention**, versus 73.8 % on the 27B. Class counts are in neither JSON (`cells` has no `yp`), so imbalance is not directly checkable. The available proxies show it is real but estimator-specific: `diagnostics_present.class_snr` at hs21 gives "desperate" 0.965 (8B) vs 1.388 (27B), spread 1.09 vs 0.69, and dom `per_class` at n=1200 puts "desperate" at 0.876 (8B) vs 0.939 (27B). But dom's per-class floor is still 0.876 while logreg's is 0.171, so differential attrition cannot account for the logreg gap.

**4. "n/d is not the mechanism."** The ordering violation is real and survives using split-half n rather than the pool n that `diagnostics` reports (acl_core.py:742): 0.293 vs 0.234 at n=1200. But it is a 2-point ordering across models differing in d, k, retention, tokenizer and abliteration — "does not order these two models" is defensible; "not the mechanism" is not. The geometry candidate is **named, not tested**: the effective-rank ranges overlap (8B 231–240 at hs14–27 vs 27B 230 at hs54), and within the 8B rank varies ×1.9 while logreg does not track it (hs8 rank 125.5/logreg 0.266; hs14 rank 231/logreg 0.338; hs21–27 rank ≈238/logreg ≈0.286). No correlation is computed anywhere.

**New, unasked:** `speaker_geometry.dom` at hs21 is present_other 0.3791 against within_present 0.4272 — ratio 0.89, versus 0.30 on the 27B, with logreg at 0.13 (near the algebraic 1/(K−1) floor). **The "near-orthogonal speaker" result collapses under dom on this model.**

**5. `_provenance`.** `code_sha` still `""` — with two models now being compared, nothing records that both ran the patched driver. `acl_core_sha` (23c758a9a0a52c7b) is identical to the 27B run and matches disk. `seeds.pool: 0` again cannot pin a temp-0.9 generation; k=220 vs 160 is recorded.

| Claim | Verdict | One line |
|---|---|---|
| **C1** | **SUPPORTED-WITH-CAVEATS** | Every number checks (with "5 of 7" understating 6, and the dom max being 0.942); the decline is significant paired at 10 seeds and, unlike the 27B, is *not* a C=0.5 artifact — `logreg_c005` falls harder at all 7 depths. |
| **C2** | **SUPPORTED-WITH-CAVEATS for the ordering, NOT SUPPORTED for the mechanism** | The n/d ordering violation holds at matched n=1200 (0.286 vs 0.568) and under split-half n, but rests on 2 models differing in d, k, 49 %-vs-74 % retention and abliteration; the geometry candidate is named only — effective-rank ranges overlap and rank does not track stability within the 8B. |

## Reconciliation
Accepted; corrections made:
- **The logistic decline with n is real on the 8B** — paired over ten seeds, 600 → 1200 is negative at 6 of 7 depths (10/10 seeds pooled), and `logreg_c005` declines *harder* at all 7. The fixed-C explanation that dissolved the 27B's plateau does not transfer. The two models disagree on mechanism; registry and plan now say so ("6 of 7", not 5).
- **Matched n:** at n = 1200 on both models logreg is 0.286 vs 0.568 and dom 0.941 vs 0.955; the ordering holds and the dom gap is smaller than the unmatched quote suggested.
- **Retention:** 49% on the 8B vs 74% on the 27B; class counts are not in the file; dom's per-class floor (0.876) vs logreg's (0.171) shows attrition cannot account for the logistic gap.
- **"n/d is not the mechanism" was too strong**: the ordering violation holds (also under split-half n), but it is two models differing in d, k, retention, tokenizer and abliteration. Reworded to "n/d does not order these two models"; the geometry candidate is named, not tested — effective rank does not track stability within the 8B.
- **New:** under dom on the 8B the present↔other cosine (0.379) is 0.89 of the within-present cosine (0.427); the "near-orthogonal speaker" result does not hold under the stable estimator on this model. Registered as a caveat on `a2_8b.focus_logreg_vs_dom_cos` and added to the plan's Paper B correction list.
Verdicts: C1 SWC · C2 SWC for the ordering, NOT SUPPORTED for the mechanism.

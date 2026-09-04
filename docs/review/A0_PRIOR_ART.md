# A0 — prior-art verdict for Paper A

**RESEARCH_PLAN rev 3 §2.3 makes A0 the kill criterion.** Not "does it replicate on other
directions" — it will, since n ≪ d instability is estimator statistics, not a property of
affect. The risk is prior publication. §2.1 already conceded part of the lane to Marks &
Tegmark. This is the verdict on the remainder.

**Verdict: Paper A survives, narrowed.** The problem statement is occupied and must be
cited as such. Three of the four claimed contributions remain unoccupied, and one of them
got *sharper* because a contemporary paper measured a weaker version of the same quantity.

---

## What is occupied

**Marks & Tegmark, *The Geometry of Truth*
([2310.06824](https://arxiv.org/abs/2310.06824)) §5.1.** Identifies the deficiency in
logistic regression, proposes mass-mean probing, reports comparable classification accuracy
with greater causal implication. The "better classifier, worse direction" dissociation is
theirs, published 2023. Paper A cites it as the prior that motivated checking, and claims
no credit for it.

**RAPTOR: Ridge-Adaptive Logistic Probes ([2602.00158](https://arxiv.org/abs/2602.00158)),
Gao et al., Northwestern.** This is the closest live work and the one that most constrains
the paper. It states the problem in the same terms we do — probe-then-steer pipelines need
concept vectors that are "accurate, directionally stable under ablation, and inexpensive",
and it names directional stability as requirement (ii): *"the learned direction must remain
consistent under minor training perturbations … ensuring the concept vector is reusable
rather than dataset-specific."* It proposes ridge-tuned logistic regression as the remedy
and gives a CGMT analysis of the high-dimensional few-shot regime.

**Paper A therefore cannot claim to have identified probe-direction instability.** That
framing is gone. Say so in the introduction rather than letting a reviewer say it.

## What is not occupied, and why the distinctions are real

**1 · Disjoint-half reproducibility, as against overlapping-subsample robustness.**
RAPTOR's stability metric (their Eq. 9) is the mean absolute pairwise cosine over K runs
under **20% training-data ablations**:

> Robust = 2/(K(K−1)) · Σ_{r<s} |⟨v_r, v_s⟩|

Any two 80% subsamples of one training set share roughly 64% of their examples. That
measures *sensitivity to perturbation*, which is a real and useful quantity, but it is not
reproducibility across independent fits, and the two come apart hard. RAPTOR reports
0.88–0.99 (their Table 2, across Llama-3.1-8B/70B, Qwen2.5-7B, on STSA / HateXplain /
Sarcasm). Our disjoint-half figure on the same estimator family is **0.41** (27B) and
**0.22** (8B). The gap between an overlapping-subsample number and a disjoint-half number
is itself the finding, and it is the reason a field reading RAPTOR's table would conclude
the problem is already solved.

**2 · Non-convergence of the direction with n.** RAPTOR does sweep the aspect ratio
δ = n/p — "6 fractions, 5 seeds", 12 settings — but for **held-out accuracy**, comparing
Acc_true against a theory-derived Acc_pred (median Spearman 0.86). The direction stability
is not swept against n anywhere. Marks & Tegmark argue logreg finds the *wrong* direction,
not that it fails to find a *consistent* one. The claim that accuracy converges while the
direction does not is unclaimed.

**3 · Logistic regression against difference-of-means on direction stability.** RAPTOR's
baselines are **xRFM and GCS**. The strings "difference-of-means", "difference of means",
"mass-mean" and "CAA" do not appear in the paper. The comparison Paper A is built on is
not made there.

**4 · A published result that reverses under estimator substitution.** No candidate found.

**The live counterargument still stands and must be engaged:**
[2604.08169](https://arxiv.org/abs/2604.08169) reports CAA mean-difference and logistic
directions as near-identical (0.99 Llama, 0.98 Qwen for compassion; 0.80 for Qwen
honesty). Note that this is **cross-estimator agreement** — do two different estimators
find the same direction — which is a different quantity from **within-estimator
reproducibility** — does one estimator find the same direction twice. Both are measured
in A2 (`cross_estimator` and `stability` in `a2_estimator.py`), so the disagreement can be
reported in that paper's own units.

**Adjacent but not competing:** *Understanding (Un)Reliability of Steering Vectors*
([2505.22637](https://arxiv.org/abs/2505.22637)) predicts steering success from directional
agreement *within the training contrast pairs* and does not fit twice; it does note in
limitations that "with few (5–30) randomly sampled training activations, steering vectors
vary so widely", which is corroboration, not preemption. *Geometric Stability*
([2601.09173](https://arxiv.org/abs/2601.09173)) introduces Shesha, which correlates
dissimilarity matrices built from complementary halves of the **feature dimensions** and is
compared against CKA and Procrustes — a representation-level RSA metric, not a probe
direction, and not a split over data.

---

## Consequences for the paper

1. **Reframe from discovery to measurement.** The contribution is not "probe directions are
   unstable" — RAPTOR says that. It is that *the field's standard reliability check is the
   wrong one*, and that the right one changes the answer.
2. **RAPTOR's remedy is a condition in our own battery.** `a2_estimator.py` already fits
   `ridge`, `logreg_c005` and `logreg_cv`, so A2 answers directly whether validation-tuned
   L2 closes the disjoint-half gap or merely the overlapping-subsample one. If ridge does
   close it, that is a genuine negative result for Paper A's headline and should be
   reported as such; if it does not, Paper A engages a contemporary method on its own turf.
3. **A4's reporting protocol gains a specific target.** Recommend disjoint-half cosine
   *alongside* decode accuracy, and state explicitly that overlapping-subsample robustness
   is not a substitute.
4. **The venue judgement in §2.1 is unchanged.** TMLR, short methods paper. A0 does not
   clear it for a headline and nothing found here suggests it should be one.

*A0 run 2026-09-02. Sources read directly from the PDFs, not from search summaries — the
first automated summary of 2601.09173 asserted that it measured probe split-half
reproducibility across disjoint data and compared logistic regression against
difference-of-means. It does neither.*

---

## Addendum 2026-09-05 — the in-house numbers on the two quantities A0 separated

A0 distinguished **cross-estimator agreement** (2604.08169's quantity) from **within-estimator
reproducibility** (Paper A's). Both are now measured on this data (`results/rev3/`):

- *Within-estimator, disjoint halves, n = 2000/half, ten splits, focus layer of the 27B:*
  difference-of-means 0.974, logistic (C = 0.5) 0.566, logistic with the per-sample penalty
  held fixed 0.581 (moving < 0.03 over n = 150 → 2000), cross-validated C 0.608
  (`a2.focus_*`, `a2f.*`). The 8B: 0.941 vs 0.286 at n = 1200 (`a2_8b.*`).
- *Cross-estimator, the like-for-like version — CAA's raw mean difference against the raw-space
  logistic direction:* **0.43** at the focus layer, 0.50 at depth 0.25
  (`a2f.caa_raw_vs_logreg_raw`), against 0.98–0.99 in 2604.08169 for compassion. Caveats from
  results/reports/21: the logistic direction here is the raw-space image of a fit on
  standardised features and a six-way row, not a binary probe; the ten-subsample CI has no
  sampling content. A disagreement in that paper's units, on a different trait and design; not
  a refutation.
- *RAPTOR's remedy engaged:* validation-tuned L2 moves the logistic row by about a tenth of
  the gap (0.566 → 0.608) on an accuracy-scored, decade-spaced grid; a stability-scored sweep
  is the open item (RESEARCH_PLAN §12 to-dos).

Consequence for the verdict above: contribution 2 (non-convergence with n) survives as a
proportional-regime statement — no material convergence over n ≤ 2000 at d = 5120 with the
schedule controlled — not as an asymptotic one; contribution 4 (a published flip) is dropped
(A3, results/reports/18).

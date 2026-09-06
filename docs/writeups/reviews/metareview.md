# Metareview — "Steered affect moves a conversational partner's activations"

**Venue:** IAB @ NeurIPS 2026 — non-archival, ≤9 pages plus references and unlimited appendix, explicitly welcoming negative results.

**Reviews:** ICLR bar 5/4/5, all reject; recalibrated to the IAB bar R1 7, R2 7, R3 8, all accept, confidence 4. Fix list and bundle were revised mid-review; findings are against the current files.

## 1. Fix verification (checked in the PDF, not taken on faith)

| # | Claimed | Verdict | Evidence |
|---|---|---|---|
| 2 | Random-vs-none computed | **PASS** | §5, Table 2: "happy +13.5 [−0.1, +28.4], q ≥ 0.26", others ≤ 0.04. |
| 3 | Random 5-frame reported | **PASS** | Tables 2–3: q ≥ 0.51, footprint 3.1–3.2 (5×). R2's missing arm now has numbers. |
| 4 | One table, one standard | **PASS** | I differenced all 18 testable Table 2 cells against Table 1: agree to ≤0.01 throughout; Fig 4 matches. |
| 5 | FDR corrected | **PASS** | §5: permuted rank-5 blocks at q = 0.001/0.035/0.013; q = 0.051 moved to the affect−permuted contrast; sad's q = 0.115 disclosed against interest. |
| 6 | Dose-0 shift, every arm | **PASS** | Table 3: affect +0.39/−0.25; permuted ≤ 0.08; random ≤ 0.01; pc1 +0.55/−0.28. |
| 8 | Other-speaker probe | **PASS** | §4: other slopes +59/+119/+65/+93/+96 vs present +84/+212/+48/+98/+196; "its own state" withdrawn. |
| 9 | Fig 3 bands + shape | **PASS** | Bands on all 24 curves; dose steps in titles (happy +71/+0, calm +38/−34). |
| 13 | Low-dimensional control | **PASS** | §3: at d′ = 100, training accuracy 0.974 at n = 2000, logistic reproduces at 0.90 vs 0.94. The earlier fix list said 0.71/0.77 — wrong in the authors' own summary until corrected mid-review. |
| 21 | Title recalibrated | **PARTIAL** | Meets R2's condition, but §4 is still headed "Transfer" and the body says "affect transfer" in three places, so R3's is half done. |
| 1 | Estimator pool described | **PARTIAL** | §2 gives 4,253 / 3,898, so n = 2000 is possible. But no class structure, construction, or overlap with the 1,615-item pool (R3 W1 asked all three), and 3,898 permits 1,900 per half, so the promised account of the 8B cap is missing. |
| — | *(new error)* | **FAIL** | Abstract: "median 91% … (85% against baseline)"; §5: "Against baseline the median is 9% (17% with calm)" → 91% or 83.5%. From Table 2 I get 91% and 84%. 85% is neither, and survived the refresh. |

## 2. Agreement, divergence, standing conditions

All three converged from the start: the estimator study is the real contribution; provenance and Appendix A are exceptional; "a control that bites" was confounded with footprint; the receiver readout was unvalidated. They diverged on the estimator half — R2 called perfect separation mere arithmetic (n < d), R3 called the mechanism measurement sharper. The d′ = 100 control settles that in R2's favour, and the authors then bound their own alarm: the logistic row *decodes better* (0.94 vs 0.88), projections correlating 0.90–0.96. The contribution narrows correctly — fine classifier, poor direction, in the regime probe pools occupy.

What flipped all three is the bar, not new evidence. Most blocking objections were answered by running experiments rather than rewording, and the paper volunteers the sentence that most undercuts it: the other-speaker projection rises comparably, so the probe cannot separate the receiver's state from its model of the sender's. At an archival venue that concession is fatal; at a workshop soliciting negative results, it *is* the paper.

**Conditions:** R2's (retitle) **met**. R3's (rename to what the probe licenses) **half met**. R1's (refit on the ten existing split seeds, report B1e's across-draw spread) **not met** — it sits on the authors' own "not done" list.

## 3. Open

**The controls claim.** Whether the rank-5 permuted frame blocks because it was fitted or because it is large is untested, and the paper says so. But Table 3 holds sharper evidence than the text draws out: a *rank-1* arm at 5.8× footprint (PC1) blocks angry while a *rank-5* arm at 5× is inert. That points at footprint over rank, and the claim that the 5×–10× band "has no fitted control in it" — true only because PC1 is label-free rather than label-fitted — reads more reassuringly than the data warrant.

Also absent: an ablation rerun with the logistic direction, cross-experiment multiplicity, and any coverage check on the 29-cluster bootstrap.

## 4. Decision

**Accept — long paper, oral. 8 / 10.** (At the ICLR bar I had reached accept (poster), 6/10; context only.)

Not a mechanical average of 7/7/8. Upward for scope fit and for what a talk buys here: a well-instrumented negative result about tools this audience uses daily, Tables 2–3 being a controls register they can adopt immediately. Downward for an arithmetic error still in the abstract of a paper whose virtue is checkable arithmetic. **The oral is contingent on conditions 1–3; otherwise poster.**

**Camera-ready conditions:**
1. Correct the abstract's "85% against baseline" to what Table 2 supports; name the family.
2. Finish R3's condition: retitle §4, drop "affect transfer" from the body, and drop "and which controls bite" from the subtitle unless 4 is run.
3. Meet R1's condition: refit on the ten existing split seeds, report B1e's across-draw spread for afraid and sad. A paper arguing single direction estimates are untrustworthy cannot rest every 27B ablation on one.
4. Optional but decisive: run the footprint-scaled random 5-frame; add it to Table 2.
5. §2: class structure, construction and overlap of the estimator pool with the 1,615-item pool; why the 8B stops at 1,200.
6. Reconcile §3's "falling from 0.90 to 0.62" with its "cosine 0.97 to 0.63"; print the 8B low-d pair the PDF omits.
7. Define "CI-implied ranges"; mark Fig 4's clipped whiskers; say in Fig 3's caption that separation comes from the paired contrast, not the drawn bands.
8. State cross-experiment multiplicity as uncontrolled.
9. §1–§8 close on page 9, the Reproducibility statement and Appendix A on page 10 — confirm the organizers count that statement with the appendix, or move it. The author line is no issue at a non-archival venue.

## 5. Note to authors

Before any archival version: **run the random 5-frame scaled to the permuted frame's footprint.** About two GPU-hours by your own timing, and the one arm that turns your central methodological claim — subtitle, abstract, Section 6 — from conjecture into result. All three reviewers named it independently at the archival bar, and it returns the moment this leaves a non-archival venue. If it comes back inert, the subtitle is earned. If it blocks, footprint is the variable and your rule becomes "match your control's footprint" — just as publishable, and more useful. R1's across-draw check is cheaper and should ship with this camera-ready; the footprint arm is what the next version stands on.

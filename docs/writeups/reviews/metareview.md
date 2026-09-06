# Metareview — "Steered affect moves a conversational partner's activations"

**Reviews:** R1 5/10, R2 4/10, R3 5/10, all confidence 4, all reject, each naming the runs that would reverse them. Fix list and bundle were both revised mid-review; findings are against the current files.

## 1. Fix verification (against the PDF, not the fix list)

| # | Claimed | Where | Verdict | Evidence |
|---|---|---|---|---|
| 2 | Random-vs-none computed | §5 B1c; Table 2 r3 | **PASS** | "happy +13.5 [−0.1, +28.4], q ≥ 0.26", others ≤ 0.04. Closes R1 W3 / R3 W3; happy's CI touches zero. |
| 3 | Random 5-frame reported | Table 2 r6; Table 3 | **PASS** | q ≥ 0.51; footprint 3.1–3.2 (5×). R2 W3's missing arm now has numbers. |
| 4 | One table, one standard | Tables 1–2, Fig 4 | **PASS** | I differenced all 18 testable Table 2 cells against Table 1: agree to ≤0.01 throughout. Fig 4 bars match Table 2. |
| 5 | FDR corrected | §5 B1d, B1c | **PASS** | Permuted rank-5 blocks at q = 0.001/0.035/0.013; q = 0.051 moved to the affect−permuted contrast; "sad does not (q = 0.115)" stated against interest. |
| 6 | Dose-0 shift, every arm | Table 3 | **PASS** | affect +0.39/−0.25; permuted ≤ 0.08; random ≤ 0.01; pc1 +0.55/−0.28. Exactly R1 W2's request. |
| 8 | Other-speaker probe | §4; abstract | **PASS** | Other slopes +59/+119/+65/+93/+96 vs present +84/+212/+48/+98/+196; "its own state" withdrawn. Costly and correct. |
| 9 | Fig 3 bands + shape | fig3.png | **PASS** | Bands on all 24 curves; steps in titles (happy +71/+0, calm +38/−34). |
| 13 | Low-dimensional control | §3 | **PASS** | d′ = 100: training accuracy 0.974 at n = 2000, logistic reproduces at 0.90 vs 0.94. The earlier fix list said 0.71/0.77 — a headline number wrong in the authors' own summary until corrected mid-review. |
| 21 | Title recalibrated | p. 1 | **PASS, partial** | Drops the transfer construct §4 withdraws; the subtitle still ends "and which controls bite", the one question the paper states is untested. |
| 1 | Estimator pool described | §2 | **PARTIAL** | 4,253 / 3,898 given, so n = 2000 is possible. But no class structure, construction, or overlap with the 1,615-item pool — R3 W1 asked all three — and 3,898 permits 1,900 per half, so the promised account of the 8B cap is absent. |
| — | *(new error)* | Abstract vs §5 | **FAIL** | Abstract: "median 91% … (85% against baseline)". §5: "Against baseline the median is 9% (17% with calm)" → 91% or 83.5%. From Table 2: 0.09 → 91%, 0.165 → 84%. 85% is neither, and persists. |

## 2. Agreement, divergence, what moved

All three converged: the estimator study is the real contribution; provenance and Appendix A are exceptional; "a control that bites" was confounded with footprint and rested on an unreported arm; the receiver readout was never validated.

R1's five demands: four met — random-vs-none, unified Table 1, per-arm shifts and footprints, and the numerical reconciliations (I confirmed the tuned-C range, the C = 50/500 split, the 0.63/0.43 conventions, the 8B axis, the anchor labels). R3's three: W3 fully, W1 and W5 partly. R2's four runs: two done, two not.

They diverged on the estimator half: R2 called perfect separation mere arithmetic (n < d) and demanded the low-dimensional control; R3 called the mechanism measurement genuinely sharper. The revision settles it in R2's favour and says so. The authors then bound the cost themselves — held-out projections correlate 0.90–0.96 and the logistic probe *decodes better*. That shrinks the contribution to "in the regime every probe study works in, the coefficient row is a tie-break, not a direction; report disjoint-half reproducibility" — narrower than v1 sold, still worth publishing.

## 3. Open

**The controls claim.** Whether the rank-5 permuted frame blocks because it was fitted or because it is large is untested, and the paper says so. But Table 3 holds sharper evidence than the text draws out: a *rank-1* arm at 5.8× footprint (PC1) blocks angry while a *rank-5* arm at 5× is inert. That points at footprint over rank, and the claim that the 5×–10× band "has no fitted control in it" — true only because PC1 is label-free rather than label-fitted — reads more reassuringly than the data warrant.

**Absent:** across-draw variability of the direction (R1 W4, mitigated but not answered by the deployed 0.88–0.93 cosine); an ablation rerun with the logistic direction (R2 Q7, the check showing the estimator result changes a conclusion); cross-experiment multiplicity; bootstrap coverage at 29 clusters; "CI-implied ranges" undefined; happy's whiskers clipped unmarked in Fig 4.

**Construct validity** is now a stated scope limit rather than a hidden flaw — answering R3 honestly, while conceding the affect half contributes little beyond "something transfers and a probe sees it."

## 4. Decision

**Accept (poster). 6 / 10.**

Not an average of 4/5/5. Two reviewers conditioned acceptance on named runs; most are now in the PDF and verifiable, and R2 offered 6–7 for a well-executed revision. Against oral: the subtitle still advertises the one question the paper cannot answer, the affect result is thin by the authors' own account, and a fresh arithmetic error sits in the abstract of a paper whose distinguishing virtue is checkable arithmetic.

**Camera-ready conditions (all required):**
1. Correct the abstract's "85% against baseline" to what Table 2 supports, and name the family.
2. Reconcile §3's "falling from 0.90 to 0.62" with its "cosine 0.97 to 0.63"; print the 8B low-d pair, which the fix list gives but the PDF does not.
3. Drop "and which controls bite" from the subtitle unless 4 is met, and revise the 5×–10× band sentence to note PC1 at 5.8× blocks where the random 5-frame at 5× does not.
4. Preferred over 3: run the footprint-scaled random 5-frame and add it to Table 2.
5. §2: class structure, construction and overlap of the estimator pool with the 1,615-item pool; why the 8B stops at n = 1,200.
6. Define "CI-implied ranges"; mark the clipped whiskers; say in Fig 3's caption that separation comes from the paired contrast, not the drawn bands.
7. Report B1e's across-draw spread over the existing ten split seeds, or state in the abstract that no interval reflects direction variability.
8. State cross-experiment multiplicity as uncontrolled.
9. The PDF carries an author line, repository paths and its own review history; resolve per venue policy.

## 5. Note to authors

One action above all: **run the random 5-frame scaled to the permuted frame's footprint.** By your own timing it costs about two GPU-hours. It is the single arm that turns your central methodological claim — the one in your subtitle, abstract and Section 6 — from conjecture into result, and all three reviewers named it independently. Everything else above is editing; this is the experiment. If it comes back inert, you have the controls paper you have been trying to write and the subtitle is earned. If it blocks, footprint is the variable and your rule becomes "match your control's footprint" — just as publishable, and more useful than the rule you state now.

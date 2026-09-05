Your review is AUTHOR-FACING ONLY: it will not be shown to the area chair and does not enter the decision.

---

## 1. Summary

The paper studies whether an emotional state induced in one LLM agent moves the internal state of a second agent it talks to, and finds that it does, dose-dependently, for five of six emotions on two models. Its real contribution, though, is methodological: it shows that the logistic-regression direction the field uses to read and write such states fails to reproduce across disjoint fits (0.57 on the 27B, 0.29 on the 8B) because every fit perfectly separates its training half, so the direction is the penalty's tie-break rather than an estimate; and that a *fitted, label-free* control blocks transfer where the field's norm-matched random control is inert. Three pre-registered ablations then use the reproducible difference-of-means direction to show that the rank-1 linear channel carries at most a fifth to a third of the transfer, for two emotions only.

---

## 2. Rubric

| Dimension | Score | Justification |
|---|---|---|
| (a) Novelty | **7** | The specific quantity — disjoint-half cosine, distinguished carefully from the overlapping-subsample robustness that Gao et al. report at 0.88–0.99 — is a genuinely new and well-chosen instrument, even though logistic instability itself is acknowledged as known (Marks & Tegmark, Braun et al.). |
| (b) Significance | **7** | The controls rule in Section 6 ("run the same estimator on permuted labels; match rank and footprint; the manipulation check must not share the ablation's target") is directly actionable for a large and growing ablation literature, though the affect-transfer finding itself is deliberately narrow. |
| (c) Technical soundness | **6** | The separability diagnosis is airtight and measured directly (training accuracy exactly 1.000 at every n, worst log-loss 0.0055, ≤80 iterations), but the headline ablation statistic — a ratio of OLS slopes over four dose points — is a fragile instrument on the visibly non-monotone curves in Figure 3. |
| (d) Experimental rigor | **7** | Common random numbers, scenario-blocked and dose-paired contrasts, three bit-identical replicate runs, a second model, and an explicit line between pre-registered and driver-declared contrasts are all well above the norm at this scale; a single probe pool and a single direction estimate underlying every 27B ablation is the main limit, and the authors say so. |
| (e) Statistical validity | **6** | The scenario-blocked bootstrap and the BH-FDR discipline are the right choices and are applied honestly (including the admission that B1d clears nothing at q = 0.051), but Table 1's bold and Figure 4's error bars mix FDR-adjusted and unadjusted significance under one visual mark. |
| (f) Clarity | **6** | Section 3's argument is beautifully staged — claim, objection, mechanism, "not claimed" — but the footprint quantity is expressed against three different denominators in Sections 5, 6 and Appendix A, and several numbers cannot be reconstructed from the tables. |
| (g) Reproducibility | **8** | Per-number claim identifiers, model revisions, torch/transformers versions, the CRC32 seed recipe with the arm field deliberately blanked for the receiver, bootstrap seed and draw count, and per-run wall-clock — this is close to best practice; the estimator study's data source is the one gap. |
| (h) Figures & presentation | **5** | Figure 4 and Table 1 agree cell-for-cell across all thirty entries, which is excellent discipline, but the figures are typeset far too small to read at print size, the colour mapping changes meaning between Figures 1 and 2, and Figure 3's caption contradicts its own calm panel. |
| (i) Honesty about limitations | **9** | Appendix A volunteers an empty pre-registration stamp caused by a path bug, a pool of 1,615 against a registered 2,160, an FDR that ran over five emotions instead of six, tolerance-limited penalty arms, and a control cosine that was never stamped into the result file — most authors would bury every one of these. |

---

## 3. Strengths

1. **The right quantity, defended against the right objection.** Section 3 does not just report that logistic directions are unstable; it anticipates the obvious rebuttal (a fixed *C* means a shrinking per-sample penalty as *n* grows), builds the C_n = 0.5·N_ref/n arm to kill it, and shows the curve still does not rise (0.581 at both ends of n = 150→2000 for the 600 anchor). Then it goes one level deeper and *measures* the mechanism rather than asserting it: training accuracy exactly 1.000 in every cell, arm and seed; a hundredfold weaker penalty giving cosine ≥ 0.96 to the same-half solution and no more reproducibility (0.52 against 0.58 at n = 2000). This is how a mechanism claim should be built.

2. **The control that bites is a genuinely good idea, and the paper resists over-selling it.** The B1e result — a rank-1 direction fitted by the same estimator on permuted labels, cosine 0.004 to the emotion direction, that is completely inert (Table 1's bottom row: 0.02, −0.00, −0.10, (−0.04), −0.01, 0.00) — is exactly the null a rank-1 ablation study needs. And the paper immediately turns around and reports that B1d's *rank-5* permuted frame is *not* inert (afraid 0.23, sad 0.27, angry 0.23, all bold), which is the inconvenient half of its own story.

3. **The "not claimed" paragraphs are a model of scientific temper.** Section 3 explicitly declines the max-margin identification and the support-vector mechanism because the weak arms are tolerance-limited. Section 5 flags that B1c's random-versus-none comparison was never computed, so the +28% happy move is "descriptively" reported. Section 7 lists the token channel as untested *and* explains why (three rewrite designs failed their own manipulation check). This is rare and it is the paper's best habit.

4. **Internal arithmetic holds up under checking.** Table 1's B1d "permuted subspace − none" fractions reproduce the raw block sizes quoted in Section 5 when multiplied by Figure 3's none-arm slopes (0.23 × 212 = 48.8 against the stated −49.5; 0.27 × 98 = 26.5 against −26.7; 0.23 × 196 = 45.1 against −44.3). Figure 4 matches Table 1 in all thirty cells. The 9% median for B1c is exactly the median of its own row over the five testable emotions. Even the compute budget checks: 288 arm-rows × 29 scenarios = 8,352 conversations, and 288 × ~20 s ≈ 1.6 h, consistent with "about two hours".

---

## 4. Growth areas

Each of these is framed as something to *add*, not something to retract — the findings survive all of them.

1. **Consider stating where the estimator study's data comes from, because Section 2 and Section 3 currently cannot both be true.** Section 2 says the probe is "fitted on one half of a 1,615-item probe pool while steering and ablation directions come from the disjoint other half" — about 807 items per half. Section 3 then reports split-half fits at "n = 2000 per half", which needs 4,000 items, and the Reproducibility statement lists the grid as n ∈ {75, …, 2000} per half. Appendix A confirms the ablation pool is 1,615 (against a registered 2,160). A reader who checks this will conclude either that the estimator study uses a different, larger, undescribed pool, or that the split-half fits at n ≥ 1200 overlap. One sentence in Section 2 naming the estimator pool and its size would remove the doubt entirely — and given how much of the paper's credibility rests on "disjoint halves share none", this is the single most load-bearing sentence you are missing.

2. **Consider reporting the dose-response with something other than a four-point OLS slope, or at least flagging the non-monotone panels, because Figure 3 shows the summary and the data disagreeing for two of six emotions.** Happy rises from ~221 to ~290 between dose 0 and 0.33 and then goes *flat or down* (290 → 279 → 279); the "+48 [+8, +88]" slope is a linear fit to a saturating step. Calm is worse: 102 → 140 → 174 → 139, an inverted U whose descent at the top dose is what pushes its CI across zero. The paper's framing — "calm's [CI] never does [exclude zero]", calm "untestable at this dose grid" — reads as *no effect*, when the panel shows a large effect that reverses. Appendix A flags only afraid's non-monotone top dose. Consider adding a monotonicity check or a saturating fit per emotion, because the honest story ("transfer saturates by dose 0.33 for happy, and reverses at dose 1.0 for calm") is more interesting than the one the slope tells, and it is already in your data.

3. **Consider making Table 1's bold and Figure 4's error bars mean one thing, because right now the same mark carries two different evidentiary standards.** Table 1's caption says "significant contrasts in bold". The B1c and B1e rows are bolded on BH-FDR-adjusted evidence (you quote q = 0.001, 0.028 for B1e). The B1d "permuted subspace − none" row is bolded on what Section 5 calls "unadjusted CIs", and the B1d decision-contrast row directly above it is unbolded precisely because "no per-emotion contrast clears the registered BH-FDR rule (q = 0.051)". Figure 4 then plots all five contrasts with identical error bars and no adjustment marker, so the yellow calm bar at 1.56 with a CI clearly off zero invites exactly the reading Section 5 disclaims. Consider a superscript or hatch distinguishing FDR-surviving from unadjusted contrasts, because your own most-quotable finding — the rank-5 fitted control blocks — is currently the one resting on the weaker standard, and a hostile reader will find that before you point it out.

4. **Consider fixing the footprint quantity to one definition with one denominator, because it currently appears with three and the reader cannot chain them.** Section 5 (B1c) says the random direction removes "twice the norm"; Appendix A says the same direction sits at "1.22". Section 5 (B1e) says the permuted-label control has "0.85× its footprint"; Appendix A says "0.84× afraid, 0.77× sad". Section 5 says B1d's frame and B1e's direction "differ in footprint (6.5 against 0.5 per position)"; Section 6 says the rank-5 frame has "ten times the emotion direction's footprint" — reachable only by re-basing 6.5/0.5 through the 0.85× factor. Since Section 6's whole rule turns on "match rank and footprint", the quantity that rule is built on should be defined once, in Section 2, in units, with every later number expressed against the emotion direction. This is a half-page edit that would materially strengthen the paper's central recommendation.

5. **Consider redrawing the figures at readable size and fixing the colour semantics, because Figures 1 and 3 are effectively unreadable in the PDF and Figures 1 and 2 contradict each other's legend.** On page 3, Figure 1's three panels occupy roughly a quarter of the text height and its four-entry legend is around 5 pt. On page 5, Figure 3's six panels are similarly compressed, and its legend box sits on top of the desperate panel's data. More seriously: in Figure 1 the magenta/pink series *is* difference-of-means, while in Figure 2 pink dotted is `cos(logistic, difference of means)` and green is difference-of-means — a reader who learns the colour code on page 3 will misread page 4 on first pass. Consider one full-width figure per page with a shared colour key, because the split-half result is your headline and it is currently the least legible thing in the paper.

6. **Consider softening or re-scoping Figure 3's caption, because it is contradicted by its own panel.** The caption says "the ablation arms separate from the none arm only for afraid and sad." The calm panel shows the two emotion-direction arms falling to ~96 and ~102 against a none arm of ~139 at dose 1.0 — visually the largest separation in the figure — and Table 1 records calm's B1c contrast at 0.94. Appendix A states plainly that "Calm's contrast is the largest and is dropped by the testability rule." Consider "separate to significance only for afraid and sad; calm's separation is large but untestable", because the current wording asks the reader to disbelieve their own eyes and undercuts the trust the rest of the paper works so hard to earn.

7. **Consider adding a sensitivity check on the blocked-fraction ratio, because its denominator is a bootstrapped slope that is near zero for two emotions.** Calm's blocked fractions run to 1.56 and 1.02 — the paper handles this by declaring calm untestable, which is right. But happy's none-arm slope is +48 with CI [+8, +88], a denominator whose lower bound is a sixth of its point estimate, and happy is nonetheless included in the five-emotion median that yields the headline "91% of the transfer survives". Consider reporting the median on the raw slope differences alongside the ratio, or a Fieller-type interval for the ratio, because the current headline number inherits happy's denominator instability without saying so.

---

## 5. Questions for the authors

1. How many items are in the pool used for the split-half estimator study, and is it the same 1,615-item pool as the ablations? If the same, how are 2,000 items per half drawn from disjoint halves?
2. Which `N_ref` anchor is plotted as "logistic, fixed per-sample penalty" in Figure 1 and as "logistic, fixed penalty" in Figure 2 — the 600 anchor or the 2000 anchor? The text describes two anchors; each figure plots one unlabelled curve.
3. Section 3 says "Weakening the penalty a hundredfold leaves the direction unchanged … (0.52 against 0.58 at n = 2000)", but Figure 2's caption describes "a 1000× weaker penalty" and its C = 500 curve is the one sitting at ~0.52 at n = 2000. Is the quoted 0.52 from the C = 50 arm or the C = 500 arm?
4. How are B1d's `rank5_vs_rank1` numbers ("+29.3" for afraid, "−60.6" for angry) defined? I could not reconstruct them from Table 1 and Figure 3: afraid's B1c decision contrast is 0.21 × 212 ≈ +44.5 and its B1d decision contrast is −0.13 × 212 ≈ −27.6, a difference of about −72.
5. Figure 1's caption says the three logistic variants "fall to 0.28 (8B)", but the tuned-*C* curve in the 8B panel ends near 0.38, matching the text's "0.37–0.41 on the 8B". Should the caption read "two of the three"?
6. Section 5 says B1c's random direction removes "twice the norm" and Appendix A gives it as 1.22. Which is right, and against what reference?
7. Appendix A says the "0.85×–5.8× footprint band, where B1d's frame blocked, is unsampled", but PC1 is reported at 5.8× and was tested. Is the claim that the band's *interior* is unsampled?
8. Given that calm's dose-response in Figure 3 is a clean inverted U rather than a null, would you consider reporting calm as a non-monotone effect rather than as an untestable one?
9. The estimator study measures cosine between fitted directions. Did you also measure whether the *downstream* readout differs — i.e. do two logistic directions at cosine 0.57 give materially different present-emotion projections, or is the instability geometric but behaviourally benign? This would tell readers how much the finding should actually worry them.

---

## 6. Overall score and confidence

**Overall: 6/10. Confidence: 4/5.**

This is a better paper than its polish suggests, and I want to be clear about which parts of my score are about substance and which are about presentation. The substance is good: Section 3 is a small, complete, well-defended scientific argument — a surprising instability, an obvious objection anticipated and eliminated, a mechanism measured rather than asserted, and an explicit boundary drawn around what the evidence does not license. The B1e control is a real methodological contribution, and Section 6's rule is the kind of thing that changes how a subfield writes its methods sections. Appendix A is, frankly, the most honest limitations section I have read in a while; the disclosure that the pre-registration stamp is empty because a driver resolved a path against the wrong directory is the sort of thing that buys enormous credibility precisely because nobody would have caught it. What holds the score at 6 rather than 7 or 8 is a cluster of things that are all fixable in a revision cycle: an unresolved arithmetic tension between the 1,615-item pool and the n = 2,000 split-half fits, which sits under the paper's central claim; a headline effect summary (four-point OLS slope, ratio to a near-zero denominator) that Figure 3 visibly strains; a significance mark that means two different things in the same table; a footprint quantity with three denominators; and figures too small to read at print size. None of these threaten the conclusions — I checked eleven numbers across text, table, figures and appendix and the substantive ones reconcile — but together they make the paper harder to trust on a first read than it deserves to be. My confidence is 4 rather than 5 because I could not inspect the claim registry or the result files that the per-number identifiers point at, and several of my questions are exactly the ones that registry would answer in a minute.

**Cross-check ledger** (eleven numbers checked; four mismatches, all reported above):

| Check | Text | Figure / Table | Verdict |
|---|---|---|---|
| Split-half at 27B L43, n = 2000 | 0.974 DoM / 0.566 logistic (§3) | Fig 1 left panel | ✅ match |
| Split-half at 8B L21, n = 1200 | 0.941 / 0.286 (§3) | Fig 1 right panel | ✅ match |
| Logistic→DoM drift | 0.97→0.63 (27B), 0.91→0.43 (8B) (§3) | Fig 2 dotted series | ✅ match |
| Weak-penalty reproducibility | 0.52 vs 0.58 at n = 2000 (§3) | Fig 2 left, C = 500 vs fixed | ✅ values match; **arm label disputed** (§3 says 100×, Fig 2 says 1000×) |
| None-arm slopes | +48 to +212; calm +44 [−3, +99] (§4) | Fig 3 panel titles | ✅ match |
| B1c median block | 9%, CI [4%, 25%] (§5) | median of Table 1 row 1 over 5 testable = 0.09 | ✅ match |
| B1d permuted-control blocks | −49.5, −26.7, −44.3 (§5) | Table 1 row 3 × Fig 3 slopes = 48.8, 26.5, 45.1 | ✅ match (rounding) |
| B1e emotion-dir block | afraid 25%, sad 29% (abstract, §5) | Table 1 row 4 = 0.25, 0.29 | ✅ match |
| All 30 blocked fractions | Table 1 | Fig 4 bar heights | ✅ match, cell for cell |
| Probe pool size | 1,615 items (§2, App. A) | vs n = 2,000 per disjoint half (§3, Repro.) | ❌ **mismatch — requires ≥ 4,000 items** |
| B1c random-direction footprint | "twice the norm" (§5) | "1.22" (App. A) | ❌ **mismatch** |
| B1e control footprint | "0.85×" (§5) | "0.84× afraid, 0.77× sad" (App. A) | ❌ **mismatch (0.77 is outside the stated figure)** |
| Fig 3 arm separation | "only for afraid and sad" (Fig 3 caption) | calm panel; Table 1 calm 0.94; App. A "calm's contrast is the largest" | ❌ **mismatch** |
| 8B logistic variants | "0.37–0.41 on the 8B" for tuned C (§3) | Fig 1 caption: "the three logistic variants … fall to 0.28 (8B)" | ❌ **caption overstates for one of three** |

---

## 7. Advice for your next paper

1. **Write the "one definition" table before you write the prose.** Three of the five mismatches above are the same failure: a quantity (footprint, norm removed, pool size) defined implicitly at first use and then re-expressed against a different denominator two sections later. Before drafting, make a private table with one row per derived quantity — name, formula, units, reference denominator — and check every number in the paper against it once at the end. This will cost you an hour and would have caught four of my seven growth areas.

2. **Build the figure at final size, from the start.** Render each figure at exactly the column width it will occupy and read it at 100% zoom on paper before you write its caption. Figures 1 and 3 are good figures shrunk into bad ones, and this is entirely a workflow problem, not a design problem. A corollary: fix the colour-to-series mapping globally in your plotting module, so that "difference of means" cannot be pink in one figure and green in the next.

3. **Write each caption from the panel, not from the claim.** Figure 3's caption states a conclusion ("only for afraid and sad") that its own calm panel contradicts, and Figure 1's caption generalises over three curves when one of them behaves differently. A useful discipline: after drafting a caption, cover the text and check that every sentence in it is verifiable from the pixels.

4. **Let the shape of the data pick the summary statistic.** Your dose grid has four points and two of six emotions are visibly non-monotone on it. A linear slope was the natural pre-registered choice, and pre-registration means you should report it — but nothing stops you from adding a registered-secondary monotonicity or saturation analysis. Framing calm as "untestable" when the panel shows a large reversing effect leaves a genuine finding on the table.

5. **Keep the Appendix A habit and make it more visible.** The candour there is the most persuasive thing in the submission and it is buried on page 8. Consider a short "what would change our mind" paragraph in the introduction that surfaces the two or three defects a skeptical reader most needs to know about. Reviewers reward authors who have clearly already been their own harshest critic, and you have been — you are currently hiding your best evidence of that.

6. **Close the loop from geometry to consequence.** Your finding is that a widely used direction estimator is unstable at cosine 0.57. The natural next question, and the one that would make this a much-cited paper rather than a well-respected one, is: what does a reader lose by using it? One extra experiment — refit the logistic direction on ten splits, run your full ablation pipeline with each, and report the spread in the blocked fraction — would convert a geometric observation into a downstream error bar that practitioners cannot ignore. You already have every component needed to run it.

This is careful, honest, well-instrumented work on a question that matters, and the methodological core is publishable. Spend the revision cycle on the pool arithmetic, the footprint definition, and the figures, and it will read as strongly as it actually is.

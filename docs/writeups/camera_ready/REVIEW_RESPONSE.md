# Camera-ready (IAB @ NeurIPS 2026): which reviews hold, and what changed

Build: `cd docs/writeups/camera_ready && ./build.sh` (figures: `python src/rev3/paper_figures_cr.py` from the repo root).
Final PDF: `docs/writeups/final/paper_camera_ready_2026-11-21.pdf` (main text pages 1–8; references from page 9; appendices A–D).

## Scope of edits

- **Abstract, Introduction, Setup:** kept as the authors wrote them. Only typo/grammar fixes, plus three
  necessary edits: (1) the intro sentence about "the monospace identifier after it" now points to
  Appendix D, because registry tags were moved out of the main text (reviewer q3Hp); (2) the Setup now
  names the *other-speaker projection* when it defines it, because later sections use the name;
  (3) "per scenario" in the slope definition is back to "on per-scenario means" (the original, precise wording).
- **Sections 3–8:** rewritten in plain language. Each term is defined before it is used. Registry tags moved
  to Appendix D. Detail tables moved to Appendix C.
- **Template:** official `neurips_2026.sty` with `[sglblindworkshop,final]` and
  `\workshoptitle{Interpreting Agent Behavior}`. The footer now reads "40th Conference on Neural Information
  Processing Systems (NeurIPS 2026). Workshop: Interpreting Agent Behavior." It previously said
  "Submitted to … Do not distribute."
- **Authors:** five authors, Redarc Labs, two co-first authors, two contact emails (manan@, shivam@).
- **Appendix A:** "the paper has not been peer reviewed" replaced; the paper has now been peer reviewed.

## Reviewer points

| # | Point (who) | Holds? | What we did |
|---|---|---|---|
| 1 | Template/footer refers to the main conference (Z1Vs, AC) | Yes | Fixed (see above). |
| 2 | Too dense: registry tags, B1c/B1d/B1e shorthand, statistical shorthand (q3Hp, Z1Vs, AC) | Yes | Plain-language rewrite. New Figure 1 setup schematic. New Table 1 "map of the evidence" (registered vs addendum vs unregistered replication vs failed instrument). Tags moved to Appendix D. |
| 3 | Figures too small, dense legends (q3Hp, Z1Vs) | Yes | All figures redrawn at print width; text prints at about 7 pt. Old Figures 1+2 merged into Figure 2. Figure 4 now includes the footprint-matched random 23-frame, and its legend gives each arm's rank and footprint. |
| 4 | Self vs other construct is unresolved (both, AC) | Yes | Section 4 now names three readings: B's own state, B's model of A, and A's words in B's context. The text avoids "affect transfer/contagion" and reports a "receiver-side, affect-related response". |
| 5 | No behavioural endpoint (both, AC) | Yes | Not fixable without new runs; stated as the first limitation after the construct. |
| 6 | Same/different system personas? (q3Hp) | Question | Answered in the Figure 1 caption: one model plays both roles with no system persona. Each turn is a plain-text scene prompt, and names rotate across replicates. |
| 7 | "Tie-break, not an estimate" is too absolute; n<d needs general-position qualifier (Z1Vs) | Partly | Section 3 now states Cover's result with its general-position condition. It explains that separation was checked directly, and limits the claim to the n<d regime. The title is kept as the authors chose it (see "Open items"). |
| 8 | Recommendations should depend on use (Z1Vs) | Yes | New "Recommendations, by use" list in Section 6: classification, scalar readout, steering/erasing, ablation controls. |
| 9 | Ridge-tuned logistic regression? (Z1Vs Q7) | Answered | Cross-validated L2 tuning reaches only 0.61 (27B) and 0.38 (8B). A low-dimensional, non-separable refit recovers 0.90, at 0.92 vs 0.94 held-out accuracy. Both are now stated explicitly. |
| 10 | Only five permutations; random frame inside the top-PC subspace; factorial rank/footprint/estimator (Z1Vs, q3Hp rank-2/3) | Yes | Not run. Stated in Section 6 as the experiments that would separate the factors, and in the limitations. |
| 11 | Blocked fraction divides by a point estimate (Z1Vs) | Partly | Significance is decided on the slope-unit contrast, never on the ratio. This is now said in Section 5. The registered decision medians recompute numerator and denominator in every bootstrap draw (Appendix A). Slope-unit table: Appendix Table 6. |
| 12 | Only sad is robust; afraid fragile (both, AC) | Yes | Stated in Section 5, Table 1 and the conclusion. |
| 13 | 8B is abliterated, weaker manipulation, not registered (Z1Vs) | Yes | Stated in the limitations and Table 1. |
| 14 | One probe pool; 29 fixed scenarios (Z1Vs) | Yes | Limitations: the re-splits measure sensitivity to partitioning, not to new pools or scenario families. |
| 15 | Paraphrase/regenerate A's text without affective cues (Z1Vs Q2) | Yes | Untested; the three rewrite designs failed their own check. Now said in the limitations. |
| 16 | "Permuted controls block just as effectively as the affect subspace" (q3Hp summary) | No, as worded | On the two shared emotions the permuted subspace blocks about half as much (0.51 and 0.57 of the affect subspace's size). It blocks more on afraid. The text says this. |
| 17 | "Rank-1 erasure leaves over 90% intact" (q3Hp summary) | Median only | Median 91%; afraid keeps 79% and sad 67%. The text reports per-emotion values. |

## Open items for the authors (not changed, because the abstract/introduction are theirs)

1. **Abstract, shuffled-label sentence.** "a region built using shuffled emotion labels reduced the effect for
   three emotions" is true for the reported shuffle. Over five shuffles only angry (5/5) and sad (4/5) recur;
   afraid recurs in 2/5 (Table 8). Suggested: "…reduced the effect for three emotions in the reported shuffle,
   and for angry and sad in most of five shuffles, whereas…".
2. **Abstract does not mention sad.** The metareview's main point is that sad is the only robust emotion.
   Suggested final sentence before the probe caveat: "Only sad's partial block survives ten re-estimations of
   every direction and a second model."
3. **Title.** Reviewer Z1Vs finds "returns a tie-break, not an estimate" too absolute. Optional softer variant:
   "…returns a tie-break, not an estimate, when examples are fewer than dimensions".
4. **Introduction terms.** It uses "controls register", "footprint", "rank-5 frame", H1/H3 before they are
   defined (all are defined in Sections 2 and 5). The contribution bullet says "every arm against baseline in
   one table (Section 6)": the main text now shows this as Figure 4, and the table is Appendix Table 3.

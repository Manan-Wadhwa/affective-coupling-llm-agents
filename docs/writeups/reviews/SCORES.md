# Final scores — recalibrated to the target venue

**Target venue:** the 1st Workshop on Interpreting Agent Behavior (IAB) at NeurIPS 2026 — non-archival; long papers up to 9 pages plus references and an unlimited appendix; scope "what do agents do and how; what do humans do in response; what happens when humans and agents interact"; the call welcomes position papers, tools, demos, preliminary findings and negative results. Deadline September 5, 2026 (tracker shows an extension to September 6, 11:59 UTC). Source: the workshop site's data file and aiworkshoptracker.com.

The three adversarial reviews and the mentor review were written cold on the first version at an ICLR main-track bar. After the revision (`revision_notes.md`), each reviewer was asked to recalibrate to the IAB bar without re-reviewing; the recalibrations are appended to the review files.

| reviewer | focus | first version (ICLR bar) | recalibrated (IAB bar, after revision) | remaining condition |
|---|---|---|---|---|
| R1 | statistics, pre-commitment, traceability | 5/10, conf 4, reject | **7/10, conf 4, accept** | refit the emotion direction on the ten split seeds and report the across-draw spread of B1e's afraid/sad blocked fractions (needs GPU runs; camera-ready) |
| R2 | machinery, confounds, controls | 4/10, conf 4, reject | **7/10, conf 4, accept** | retitle so the title does not assert the construct Section 4 withdraws — **done** |
| R3 | construct validity, calibration, figures | 5/10, conf 4, reject | **8/10, conf 4, accept (long paper)** | retitle to what the probe licenses — **done** |
| mentor (author-facing, not shown to the AC) | developmental | 6/10, conf 4 | **8/10, conf 4** | carry the withdrawn construct claim into title and abstract — **done** |
| area chair | metareview on R1–R3 + revision notes + recalibrations (`metareview.md`) | accept (poster), 6/10 at the ICLR bar (context) | **accept, long paper, oral, 8/10** — oral contingent on conditions 1–3, otherwise poster | conditions 1 (abstract median), 2 (retitle §4, purge "affect transfer"), 5 (pool description), 6 (cosine conventions, 8B low-d pair), 7 (CI-implied ranges, whiskers, Fig 3 caption), 8 (cross-experiment multiplicity stated) applied 2026-09-06; condition 9 (reproducibility statement in the appendix) as done; conditions 3 (direction-resampling spread) and 4 (footprint-scaled random 5-frame) need GPU time and are stated limitations, so the operative decision is **accept (poster)** until 3 is run |

What moved the scores, in the reviewers' own words: the random-versus-none contrast and the random 5-frame now exist and are inert; every arm sits against baseline under one significance standard; the footprint is defined once; the low-dimensional control turned the "n < d" objection into a result; the other-speaker probe withdrew the "receiver's own state" construct claim and the paper says so. What did not move: R1's request for direction-resampling spread, which remains a stated limitation.

Audits: all nine original citations real (two stale records fixed; two misstatements corrected; Soudry et al., Benjamini–Hochberg, Efron–Tibshirani and two contagion papers added); the statistical-consistency audit's 17 items are addressed in the revision or moved to Appendix B as stated limits.


## Camera-ready conditions, status 2026-09-06 09:00Z
- Condition 4 (footprint-scaled random frame): **run and added** (Tables 2–3, §5, §6, abstract; `b1d23.*`). Result: inert for all five testable emotions; the fitted control bites because it is fitted.
- Condition 3 (across-draw spread of B1e on afraid and sad): ten draws running on three boxes; six of ten in at 09:00Z; to be added before submission.

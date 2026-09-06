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
- Condition 3 (across-draw spread of B1e on afraid and sad): **run and added** (appendix Table, §5, abstract, intro, Limitations; `b1e10.*`). Sad's block holds in 10/10 draws (median 35%); afraid's manipulation check passes in 4/10, so the affect-specific claim is narrowed to sad.


## Round 2 (2026-09-06, after camera-ready conditions 3 and 4; both venues scored; reviews in `reviews/round2/`)

| reviewer | NeurIPS main track | IAB workshop | format | top weakness |
|---|---|---|---|---|
| R1 (statistics) | 4/10, conf 4 | 7/10, conf 4 | poster (spotlight if the permuted control is run as a distribution) | permuted-label control is one realisation; "as strongly" contradicted by Table 1 |
| R2 (interpretability) | 4/10, conf 4 | 6/10, conf 3 | poster | "bites because it is fitted" not licensed: covariance-matched random frame never run |
| R3 (agents / workshop fit) | 4/10, conf 4 | 6/10, conf 4 | poster | not about agent behaviour: no transcript-level outcome on B |
| citation validator | 14/14 real; 1 mischaracterised, 2 imprecise, 1 decorative; 6 missing needed | | | |
| claims-and-numbers checker | 162 items; 5 precision mismatches, 8 inconsistencies, 5 overclaims; no headline number wrong | | | |
| area chair | reject, 4/10 | poster, 6/10 (spotlight if conditions 2, 3, 12 met) | poster | fitted-control mechanism untested against a covariance-matched frame; rank-5 control is one draw; no transcript-level outcome |

All reviewers scored the same bundle (commit c8e1ea5). Fixes applied after the round: revision note 25.

Area chair conditions 1, 4–11 met from committed data (revision note 26); 2, 3, 12 need GPU runs. Final manuscript: `docs/writeups/final/paper_final_2026-09-06.pdf`.

Update 10:50Z: condition 3 met by the permutation null (revision note 27; reading 'unlucky draw'). Conditions 2 and 12 still need runs.

| final independent reviewer (all angles, after condition 3) | 4/10, conf 4 | 6/10, conf 3 | poster | covariance-matched random frame still the discriminating control; five permutations too few to serve as a null for the affect subspace; abstract's "same threshold as the affect subspace" overclaimed (fixed) |

Final reviewer (`reviews/round2/final_review.md`, on commit 9727837): 18 wording fixes proposed, 13 applied (revision note 28); title change and the 8B-choice sentence declined; two fixes need runs.

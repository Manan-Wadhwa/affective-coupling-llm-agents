# Blind review — paper_rev3.md

## Required edits, most serious first

1. **§5/Abstract** — "a *fitted* label-free control was not [inert]" / "a norm-matched random direction is inert where a fitted one is not." — False: B1e's control is fitted, label-free and **inert** (`b1e.permdir_inert`). — "A random direction was inert in every run; so was a fitted rank-1 label-free direction. Only a fitted rank-5 frame blocked; whether rank or footprint does that is untested."

2. **§3** — "none-arm slopes +84 to +212" — wrong: happy is **+47.9 [8.2, 87.5]** in `b1c_alllayer_qwen36-27b.json` (others 84.1/97.5/196.0/211.8); no id carries the range. — "none-arm slopes +48 to +212, CIs excluding zero (`b1c.verdict_H1`)".

3. **Header** — "blind critiques of reports 28 and 29 are pending" — false: reports 28–30 exist and their corrections are already in the notes. — "Critiques of reports 22–30 are in and incorporated."

4. **Abstract** — "readable-affect removal is not what blocks transfer" — contradicts §4's "blocked fraction tracks how much readable affect the ablation removes (r ≈ 0.85)"; the registry says they *dissociate*. — "…so removal and blocking dissociate for afraid and sad, though across B1c's five the blocked fraction tracks removal."

5. **§2** — "Depth-specific, never converging." — reasserts the asymptotic reading `a2f.focus_logreg_fixed_lambda_flat`'s note withdraws ("must approach 1 eventually"). — "Depth-specific, no convergence over the reachable range (n ≤ 2000, d = 5120) — proportional-regime, not asymptotic."

6. **§4 (B1d)** — "afraid (−49.5), sad (−26.7) and angry (−44.3) … (unadjusted CIs, BH-FDR q = 0.051)" — the q belongs to *sub_all − perm_all* (`b1d.median_fraction_descriptive`), not perm-vs-none. — Drop it here; append to the increment sentence: "no per-emotion contrast clears the registered BH-FDR rule (q = 0.051)."

7. **§4 (B1c)** — "a unit-norm random direction … changes nothing" — the note: "DESCRIPTIVE ONLY: no rand_all vs none contrast was … computed", and happy moves **+28%**. — "…leaves five of six slopes within 5% of the none arm and moves happy by +28% (CIs overlapping); no rand-vs-none contrast exists."

8. **§7** — "Pre-registration before the driver existed (B1c, B1d, B1e)" — false for B1c: driver 122af4c 19:02Z, prereg cc2dc32 19:22Z, run 19:32Z, `prereg.found = false`. — "Pre-registration before each run; B1d and B1e stamped in the file, B1c's unresolved and resting on commit times, mtime and started_utc."

9. **Abstract** — "replicated three times" — B1d and B1e reproduce B1c's arms *bit for bit* ("Pipeline replication, not a new measurement"); B1c shares B1's pool. — "…in one sampling replication and two bit-identical pipeline reproductions on one pool."

10. **Abstract** — "a fitted label-free subspace of the same footprint blocks as much as the emotion direction" — it matches the *affect subspace* (6.5–6.9 vs 6.9), ten times the emotion direction's 0.46–0.66 (`b1d.footprint`). — "…a subspace of the affect subspace's footprint, ten times the emotion direction's, blocks as much."

11. **§2** — "In the counter-paper's own units" — the note: "Closer to 2604.08169's quantity than A2's 0.59 **but still not it**". — "Closer to the counter-paper's units than A2's 0.59, though still not identical, the cosine is 0.43/0.33–0.37, not 0.98."

12. **§6** — incomplete: the 8B B1e run is out-of-prereg; B1e's 0.004 cosine is verifier-only, unstamped; B1c's stamp is empty and its pool (1,615) and BH-FDR set (5) depart from the registered 2,160 and six. — Add a sentence naming those three.

13. **§2** — "three penalties spanning 1000×" — four arms (C_n, 0.5, 50, 500), and C = 500 reproduces C = 50 to three decimals, so the span "bounds nothing" (`a2m.v2_ten_seeds`). — "under the fixed schedule and C = 0.5, 50 and 500 alike (the weak arms being one solution)".

14. **§4 (B1d)** — "rank 5 blocks *less* than rank 1 for afraid" — cherry-picked; the id: "two of six significant, in opposite directions". — "…less for afraid (+29.3), more for angry (−60.6)."

15. **§4 (B1e)** — "blocks angry by 61% and calm" — calm is untestable, knife-edge (−43 [−91, −0.04]), unadjusted over ~30. — "blocks angry by 61% (and calm, knife-edge on an untestable emotion)".

16. **§7** — "Two … interpretive sentences were rejected by review" — undercount; the notes record seven ('footprint discharged', 'max-margin separator itself', 'the 8B's shape', B1d's overlap explanation, 'no net rise at every depth', 'within 7%', 'floating-point tie'). — "Seven were withdrawn under review."

17. **§3** — "Happy's response is weak and non-monotone" — no id; the non-monotone readout on record is *afraid's*, and happy's α = 0.5 peak is from the superseded grid. — "Happy's is the weakest of the five (+47.9)."

18. **§4 (B1c)** — "layers 43–63 add nothing detectable for four emotions" — zero is included for **all six**; four are merely tightly bounded, and the contrast is driver-declared, not pre-registered. — "…changes no dose-response detectably, with an extra-blocking bound under 0.08 for four (driver-declared)."

## Numbers with no registry id
- §1: "revision 6a9e13bd" (resolves in the files, no entry); the dose grid and reps; "29 scenarios"; "1,615-item … (sha 70129f86…)"; "5,000 draws".
- §2: "Tuned C … 0.61 (27B) and **0.38** (8B)" — no id, the 8B value existing only as 0.37–0.41 in a note; "log-loss ≤ 0.006, ≤ 80 iterations"; "three penalties spanning 1000×"; "0.91 → 0.43 (8B)", cited to `a2m.logistic_departs_from_dom` but in `a2m.8b_same_regime`.
- §4: "q = 0.051" under the wrong id. Abstract: "median 91%" (complement of 0.092).
- §3: "none-arm slopes +84 to +212"; "non-monotone".

**18 edits are required before every sentence in the draft is supported.**

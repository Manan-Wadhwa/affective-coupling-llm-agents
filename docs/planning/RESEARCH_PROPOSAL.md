# Research proposal — where we stand after rev 3 (2026-09-05) and what we propose next

This document is the forward-looking companion to `RESEARCH_PLAN.md` (the operative plan
with its status board and rescopes) and `docs/review/claims.json` (the registry, 76
entries, every quotable number machine-checked). Every number below carries a registry id or
a report number in `results/reports/`; nothing is quoted from memory. Status labels: **established**
(quotable, critiqued, survived), **partial** (quotable but scoped), **refuted / withdrawn**,
**open**. B1e and the 8B replication landed at 11:19Z and 11:21Z and are folded in.

## 1. The question, restated

When two LLM agents converse, does A's affective state reach B's internal representation, and
through what channel? Rev 3 rebuilt the pipeline from one vendored core (`src/lib/acl_core.py`),
put a provenance pointer on every number, and re-measured. The programme now has two arms:

- **Paper A (methods):** how emotion directions are estimated and why the estimator matters.
- **Paper B (phenomenon):** affect transfer between agents and its channel.

Paper C (the Jacobian lens / verbalisation) has not started and is not proposed here.

## 2. What is established

### 2.1 Estimator reproducibility and its mechanism (Paper A)
- **Difference-of-means directions reproduce across independent fits; logistic directions do
  not, in the regime everyone works in.** Focus layer 43 of the 27B, n = 2000 per half, ten
  disjoint splits: dom 0.974, logistic 0.566 (`a2.focus_*`, report 16); 8B focus layer 21,
  n = 1200: dom 0.941, logistic 0.286 (`a2_8b.*`, report 19). **Established.**
- **Holding the per-sample penalty fixed does not make the logistic direction converge with
  n.** 27B layer 43: no net rise 150 → 2000 at two anchors (600 and 2000; `a2f.*`, `a2f2.*`,
  reports 21/23), with a significant dip at 600 and a partial recovery; layer 54: one drop at
  300 → 600 then flat (`a2f54.*`, report 25); layer 16 rises. 8B layer 21: peaks at 150 and
  falls to 0.276 at 1200 (`a2f8.lam_declines`, report 24). **Established as depth-specific;
  "no rise at every depth" is false (report 25's critique).**
- **Why: every logistic fit at every n used is a perfect separator.** Training accuracy 1.000
  in every cell, arm and seed; loss saturated; the direction is the penalty's tie-break among
  separating hyperplanes, a 100× (and 1000×) weaker penalty gives the same direction and no
  better reproducibility, and the separator drifts away from the mean-difference direction as
  n grows (cos 0.97 → 0.63 on the 27B, 0.91 → 0.43 on the 8B) while difference-of-means
  converges (`a2m.*`, report 26, ten-seed CIs). **Established.** Not claimed: that the fit sits
  at the max-margin limit (the weak arms are tolerance-limited), or a support-vector mechanism.
- **Paper A's item 2 is to be worded** as the report-26 critic put it: *"On separable
  activations — every logistic fit from n = 75 to 2000 attains training accuracy 1.0 — the
  multinomial direction is the penalty's tie-break among separating hyperplanes, not a
  likelihood estimate. Its split-half reproducibility plateaus at 0.55–0.60 (27B) and falls to
  0.28 (8B) across a 27-fold n increase; difference-of-means, an average rather than an
  extremum, rises to 0.94–0.98 on the same splits, and the two diverge (cos 0.97 → 0.43–0.63)."*
- The counter-paper's near-identity of CAA and logistic directions does not hold here in its
  own units: 0.43–0.44 on the 27B, 0.33–0.37 on the 8B (`a2f.caa_raw_vs_logreg_raw`,
  `a2f8.cross_estimator`, `a2f54.cross_estimator`). **Established, with the overlap caveat on
  the CI.**
- Withdrawn from Paper A: "a published result reverses under estimator substitution" (A3:
  reversed under every configuration; report 18); "logistic and mean-difference are
  near-orthogonal" as a general statement (`e0.cross_cos` holds only for logistic rows at the
  1/(K−1) floor).

### 2.2 Affect transfer and its channel (Paper B)
- **Transfer exists and is dose-dependent** in B's present-emotion projection for five of six
  emotions (calm never excludes zero), reproduced three times with identical directions
  (B1 follow-up, B1c, B1d; reports 20/22/27). **Established** (representation level; the
  behavioural reach is untested).
- **The rank-1 emotion direction is not the channel.** Removing it from A's span at every
  hidden state 13–63 during B's prefill leaves a median 91% of the transfer (H1, pre-registered;
  `b1c.*`, report 22). Partial blocking for afraid (21%) and sad (33%) replicates; blocking
  tracks how much readable affect the ablation removes (r ≈ 0.85). **Established.**
- **The rank-5 class-mean subspace is not a clean instrument** (B1d, pre-registered
  instrument_failed; `b1d.*`, report 27): it shifts the readout at dose 0, and a permuted-label
  subspace of the same footprint blocks transfer for afraid, sad and angry as much as the
  emotion direction did — because it is *fitted* (65× a random frame's variance, a third
  overlap with the affect frame), not because of norm. **Established as a confound.**
- **B1e returned H3 (report 28, `b1e.*`):** a fitted, label-free rank-1 direction (footprint
  0.85×) is inert for every emotion; the emotion direction blocks afraid by 25% and sad by 29%
  against it (q 0.001 / 0.028); the top principal direction at 5.8× blocks angry and calm but
  not afraid or sad. **Established under the rule, with the caveat standing (critique, report 28):** the blocking
  survives a fitted label-free control of smaller footprint; a matched-footprint control in the
  0.85×–5.8× band has not been run. Readable-affect removal dissociates from transfer blocking.
- **8B replication (report 29, `b1c8.*`):** the same five emotions transfer and the same two
  reach significance (15%, 31%); the other three are too wide to tell; the manipulation is
  half as strong there; the MC criterion returned instrument_failed under a null upper window
  (a strict ≥ with no tolerance band — a defect to fix).
- **Open, three designs in a row:** the token channel. Every rewrite arm failed its own check
  (the neutral rewrite keeps 67–98% of readable affect; the affect-preserving rewrite changes
  transfer itself). No claim about text vs activation is quotable.
- Refuted / withdrawn from the earlier record: the E3 "text channel cannot be severed" claim,
  the β = 0.3 activation-passing result, the 0.197-nat CMI coupling (`cmi.pilot_spurious`),
  the J-lens 6/6, the honesty-floor bound, "present shift exceeds other shift" (reversed under
  a non-circular probe), and 8-turn accumulation. See the README's retraction table.

## 3. Proposal — Paper A

**Claim set (all quotable now):** disjoint-half reproducibility as the missing quantity;
non-convergence under a fixed per-sample penalty in the n ≪ d regime, depth-specific; the
separability mechanism; the CAA-vs-logistic number in the counter-paper's units.

**What remains before writing:** (i) one pool with n > d is the only way to show the regime
boundary — none of ours reaches it (max n/d 0.39); propose a 4096-d model with a 10k pool, or a
reduced-dimension study (PCA to d′ < n) as the honest substitute; (ii) ten-seed CIs are now on
the margin diagnostic; the same for the follow-up's cross-estimator with a smaller `cross_n`
(overlap caveat); (iii) a second anchor below 600 (the stronger penalty gave the higher
cosine; the maximum is unbracketed); (iv) the `n_cls` assert and `N_REF` prose fixes in the
drivers. Exhibits: the split-half-vs-n figure at three depths and two models; the separator
divergence figure; the margin table.

## 4. Proposal — Paper B

**Claim set:** representation-level transfer with dose-response; the rank-1 direction carries
at most a fifth to a third of it and only for two emotions; a linear rank-5 affect subspace is
not a usable instrument; fitted ablations of A's span reduce transfer without emotion
information (the footprint/fitted confound), which the field's random-direction controls do
not catch.

**Decisive next runs, in order (each pre-registered before its driver exists):**
1. ~~B1e~~ done: H3.
2. **Attention-masking arm** (PREREG_B1c §7): mask B's attention to A's span instead of
   projecting — separates "B re-reads A's tokens" from "B reads A's residual stream".
3. **A manipulation check independent of the ablated subspace**: a held-out probe fitted on
   the READ half in a subspace orthogonal to the ablated one, or a text-side rater, so the
   instrument and the intervention no longer share a target (B1d's failure mode).
4. **Text channel, fourth design:** replace LLM rewrites with template-controlled messages of
   fixed affect and varied content, gated by the readout before use.
5. **B6 behavioural reach** only after 1–3, on the 16 usable scenarios of the screen.

**Risks:** the readout is generation-side and coarse (forced choice); calm is untestable at
this dose grid; leases and suspend events have cost two runs — outputs now go under
`/marimo/results` and pulls run every 10–15 minutes.

## 5. Practice adopted in rev 3 (keep)

Pre-registration before the driver exists, with addenda before launch; an independent
verifier with a real forward-pass smoke test before every GPU run; a blind critique after
every result, folded in with a reconciliation and corrected registry wording; every number
with a pointer; a night log with timestamps. Six critiques this cycle changed wording in
every report they touched and rejected two of my sentences outright (reports 22 and 25).

## 6. Two-week plan

Days 1–2: B1e result, report 28, critique; if H3′, B1c's specificity sentence is withdrawn.
Days 3–5: attention-masking driver + prereg; independent readout design. Days 6–8: run both;
Paper A figure drafts from the existing files. Days 9–12: text-channel fourth design; Paper A
draft. Days 13–14: Paper B rescope with all four results; decide B6.

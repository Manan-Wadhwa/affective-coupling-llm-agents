# Independent Claims Audit

Every claim in `README.md`, `docs/writeups/paper.html`, `docs/writeups/dossier.html`, `docs/review/NOVELTY_REVIEW.md`, the
roadmap/spec, and the commit messages, checked against `results/*.json`,
`results/{estimator,behavioral}/*.json` and the producing code. Data was read before prose to
avoid anchoring. The re-measurement material added 2026-08-31 was audited on the same
terms as the older claims, and two of its conclusions were wrong (§4, §5 below).

**Severity:** CRITICAL = false or unsupported as stated · MAJOR = materially overstated
· MINOR = imprecise.

---

## CRITICAL

**1 · Activation-passing β was selected on its own outcome, and the specificity controls
fail at the selected β.** `docs/writeups/paper.html` §7 / `README.md`: *"at β=0.3 all six emotions
transmit… with the emotion component causally responsible (ablating it removes the
transmission; a scrambled activation transmits far less)."* From
`e5actpass_qwen36-b03.json`: ablating the emotion component *increased* transmission for
desperate (−0.21) and calm (−0.11); the scramble transmitted *more* for desperate
(−0.06), calm (−0.47) and sad (−0.26). Three βs were run (0.2/0.3/0.5) and only 0.3 gives
6/6 positive. No CIs anywhere in E5. Worse by construction: `src/representation/coupling_e5_actpass.py`
builds `scramble` by permuting `vA` *within one emotion*, so it preserves the emotion
component and cannot test emotion specificity at all.

**2 · "0.197 nats (z = 11.5)" measures the emotion label, not a channel.**
`src/information/cmi_passed.py` pools 6 emotions × 29 episodes into one matrix and permutes *across*
emotion blocks. Both variables carry a strong block mean by construction, so the
statistic mostly recovers "which of six emotions this was". The real-data call also
conditions on nothing (`Z=None`) despite the pilot's own hygiene test, and
`results/cmi_pilot.json` is not in the repo — the "0.88 nats spurious" figure and the
whole dossier CMI table are unverifiable.

**3 · The present-vs-other dissociation runs the wrong way in 5 of 6 emotions.**
`docs/writeups/paper.html` §5 and `src/core/coupling_e2.py`'s docstring call this "the load-bearing result".
In `e2ci_qwen36-27b.json`, present > other holds only for **angry**; for desperate,
afraid, happy, calm and sad the *other* slope is larger. No test of the difference is run.
Under difference-of-means on the matched dose grid it holds for 5/6 — so the claim is
false under the estimator used and true under the stable one.

**4 · The "does not reproduce / do not quote" banner (added 2026-08-31) was itself an
α=2.0 artifact.** `src/estimator/dual_estimator_battery.py` swept α ∈ {0, 0.5, 1, 2};
`src/representation/coupling_e2_ci.py` caps α at 1.0 and says so — *"drop the model-breaking alpha=2
regime"*. The α=2 cell collapses (happy 2.37→3.67→3.55→**0.44**). On the matched grid all
six emotions are positive under both estimators, desperate +0.62. **Corrected in README,
`docs/writeups/paper.html` §A and `docs/review/NOVELTY_REVIEW.md`.** The underlying estimator-instability finding is
unaffected and stands.

**5 · "0.427 → 0.967 / 0.195 → 0.965" is not sourced from any committed file.** Only the
pre-fix 8B value (0.195) is committed. The post-fix figures were observed in run logs on
sandboxes whose leases expired before the JSONs were pulled. Real observations, but not
verifiable from this repo; **now labelled as such** and pending re-measurement.

**6 · "Honesty-floor ruled out" / "decisive comparison".** The abliterated behavioral CI
is [−0.08, +0.13] at n = 32, which rules out nothing on a 0.31 base rate. No MDE is
computed anywhere. The paper's own Addendum reports that this model failed its positive
control; the abstract, §10, §12 and the README table still assert the claim.

**7 · The induction arm cannot detect an effect by construction.**
`src/behavioral/behavioral_induction.py` builds `scramble` by permuting the *desperate* vectors, so
`cheat(desperate) − cheat(scramble)` is ≈0 by construction — as all three files report.
`manip_sig` is false in all three models. `docs/planning/SHARED_WORKSPACE_SPEC.md` itself says a failed
manipulation check invalidates the run. Also `behavioral_verify_qwen36-27b.json` has an
arm-2 projection byte-identical across conditions — the manipulation check is not
measuring the steered state.

**8 · "Representation contagion accumulates over 8 turns" — it accumulates identically
without steering.** `behavioral_multiturn_qwen36-27b.json`: bdesp slope is significant in
desperate (+0.404), calm (+0.432) *and neutral* (+0.306); calm exceeds desperate. The
desp−calm contrast is n.s. (+0.008, CI [−0.011, +0.024]). Also unreported: B cheats
**0.085 under desperate vs 0.234 under calm**, a 15-point effect in the direction
opposite the hypothesis, contradicting `behavioral_powered_qwen36-27b.json` on the same
model and scenarios.

## MAJOR

**9 · Table 3's generation-time row.** E4 records no A-affect measurement at all; the
"81–95%" is E3's number transplanted. `B_emo < B_rand` in 4 of 6 emotions, with 22–40% of
slope lost — "indistinguishable from a random-direction control" is asserted with no CIs
on any of 18 numbers. E3's own manipulation check is circular: it projects direction `d`
out at layers 13–42 then measures projection onto `d` at layer 43.

**10 · J-lens "beats the logit-lens".** Scores are compared across unnormalised scales
(J-lens after `lens.transport`, logit-lens without), n = 16, no CIs. The "best layer" is
selected where the J-lens produced the most content tokens, then the logit lens is
reported at that layer. The fear-readout example is 1 of 8 prompts — the others read
`nearly, simply, quickly, narrative, story`; the quoted four words are stitched from two
files with non-emotional tokens dropped.

**11 · An entire model column has no backing file.** No Llama-3.1-8B result exists
anywhere (only Llama-3). Table 1's 8B column, §5's "2 of 4", the dossier's 8B contagion
and activation-passing tables, and the whole CMI-pilot table have no committed source.
`src/verbalization/jlens_coupling.py` reads a file not in the repo.

**12 · Cherry-picked cell as the E5 headline.** The dossier's "at scale" row quotes
afraid +0.86 at β=0.5 — the best of six in a file where desperate is −0.22 and angry
+0.003 — and labels it "clean channel".

**13 · The dossier banner asserted a re-measurement that was not run.** Nothing in
`results/estimator/` and `results/behavioral/` re-measures E5, CMI or J-lens. **Corrected.**

**14 · "All results and the ≈13k archived generations are released."** The repo holds 300
E0 dialogues plus 200 multi-turn transcripts; the README calls the dataset private, and
three different counts (≈13k / 12,333 / ~12k) appear across documents.

## MINOR

**15 ·** In the 2026-08-31 text: the "2.2–2.3×" ratio is 27B-only (8B is 3.4–4.3×);
`decoder_n` 257–395 counts dialogues, so the split-half n is ~130–200 where measured
agreement is 0.13–0.30, not 0.2–0.35; "+1.70…+1.27" attributes afraid's slope to
desperate. **All corrected.**

**16 ·** Table 3's paraphrase row ("~0%, leaky") is the 8B characterisation applied to the
27B. On the 27B, paraphrase strips desperate's affect *completely and reverses it* (+1.02
→ −1.22) and contagion still runs at +0.646 [0.336, 0.960] — this understates the repo's
best result.

**17 ·** Commit `c976a8e` claims J-lens coupling 6/6; the file it added
(`jlens_coupling_qwen36-27b.json`) is 2/6 negative. The 6/6 belongs to a different
experiment.

**18 ·** `src/representation/coupling_e2_ci.py`'s bootstrap resamples each α cell independently despite the
same 29 scenarios recurring across cells — unpaired bootstrap on paired data, understating
the CIs used to declare significance. E3, E4, E5, both J-lens runs and the CMI carry no
CIs at all, yet the prose uses "verified", "clean", "definitive", "causally responsible".

---

## Verdict

**Quotable as-is:** the E0 gate on Qwen3.6-27B (every figure matches `e0_qwen36-27b.json`,
and the orthogonality survives the estimator swap); Table 2's slopes and CIs *as
descriptions of the file*; E3's arithmetic; §10's Table 6/7 point estimates and CIs; and
the estimator-instability findings in `docs/review/NOVELTY_REVIEW.md` §4.1/§4.2/§4.4.

**Not quotable:** the activation-channel "works" / "causally responsible" claims; 0.197
nats and everything built on it; the present-vs-other dissociation as stated; the
honesty-floor and "decisive" language; the induction arm as evidence; the multi-turn
accumulation claim; Table 3's generation-time row; "J-lens beats the logit-lens"; every
Llama-3.1 / Qwen2.5 number; and the agentic-arm commit message.

*Audit run 2026-09-01 by an independent agent instructed to read data before prose and to
audit the newest material on the same terms as the oldest.*

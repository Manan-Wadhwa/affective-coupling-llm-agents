# PREREG B1e — is B1c's rank-1 blocking emotion-specific, or footprint?

Written 2026-09-05 (IST; provenance timestamps are UTC), after B1d (report 27) and before the
driver `src/rev3/b1e_footprint.py` exists or any generation is run.

## 1. The question

B1c found that projecting the rank-1 difference-of-means emotion direction out of A's span at
hidden states 13–63 during B's prefill removes 21% (afraid) and 33% (sad) of B's dose-response,
with a unit-norm random direction inert. B1d then found that a label-permuted rank-5 subspace
with a large footprint (removed norm 6.5–6.9 against 0.66 for the direction) blocks transfer for
afraid, sad and angry as much as the emotion direction did, without emotion information. So the
random-direction control in B1c was footprint-poor. Question: does a rank-1 direction with the
emotion direction's footprint but no emotion information block afraid and sad the way the
emotion direction does?

## 2. Hypotheses, stated before the data

Pre-specified emotions: **afraid and sad** (B1c's two blockers; report 22). All six are run and
reported; the decision uses these two.

- **H3 (specific):** for both afraid and sad, the blocked, dose-paired contrast `emo_all −
  permdir_all` is significantly negative and its blocked fraction (relative to the `none`
  slope) is ≥ 0.10.
- **H3′ (footprint):** for neither afraid nor sad is `emo_all − permdir_all` significantly
  negative, while `permdir_all − none` is significantly negative for both (the permuted
  direction blocks as the emotion direction did).
- **indeterminate** otherwise (one of the two, or neither blocks at all).
- **instrument_failed:** MC-steer fails for either pre-specified emotion, or `permdir_all`
  shifts the dose-0 A-span readout by more than 0.15 from the `none` arm for either (the
  control then disturbs the readout the way B1d's subspace did and the comparison is not clean).

## 3. Design

- Everything identical to B1c/B1d: model and revision, B1c's probe pool (sha 70129f86…),
  DIR/READ halves, `dom` steering direction at layer 42, split-half gate ≥ 0.80 at every ablated
  layer, doses {0, 0.33, 0.67, 1.0}, 3 reps, 29 scenarios, common random numbers for B.
- **Arms (four):** `none`; `emo_all` (B1c's rank-1 direction, hs 13–63 — the bridge; expected
  to reproduce B1c/B1d bit for bit); `permdir_all` (the same `dom` estimator on the same DIR half
  with the present-emotion labels permuted, the class-index direction taken for the steered
  emotion, unit norm, hs 13–63 — one permutation per (emotion, rep), applied at every layer;
  the rank-1 analogue of B1d's `perm_all`); `pc1_all` (the top principal direction of the
  centred DIR-half features at each layer, unit norm, the same for every emotion, hs 13–63 —
  the largest-footprint rank-1 control, reported descriptively).
- The footprint (removed norm per masked position summed over layers) is recorded per row; the
  match between `emo_all` and `permdir_all` is checked after the run and reported, not gated:
  the permuted-label direction is expected to remove more than the emotion direction (B1d's
  rank-5 numbers imply ~1.3 per dimension against 0.66), which makes it a conservative control.
- Compute: 4 arms × 4 doses × 3 reps × 6 emotions = 288 arm-rows ≈ 1 h 35 min on one GPU.

## 4. Outcomes and checks

- Primary: B's present-emotion projection per arm-row; secondary: the forced-choice readout on
  B's reply, hooks off.
- MC-steer as in B1c. A-span readout shares removed at the top dose and dose-0 readouts
  reported per arm (§4.2 of PREREG_B1d), with the dose-0 shift used only as the
  instrument_failed trigger above.
- Quality exclusions as in B1c.

## 5. Analysis, fixed in advance

Slopes and contrasts as in B1c/B1d (scenario-blocked, dose-paired, 5000 draws). Contrasts:
`emo_all − permdir_all` (decision), `emo_all − pc1_all`, `permdir_all − none`, `pc1_all − none`,
`emo_all − none`. Blocked fractions relative to the `none` slope, sign-aware. The verdict is
the per-emotion rule of §2 on afraid and sad; all six emotions' contrasts are reported with
BH-FDR over the testable ones as descriptive.

## 6. What this does not test

Whether `emo_all`'s residual (79–67% of the transfer surviving) is carried linearly at all;
the 8B; layers below 13.

## 7. Provenance

As PREREG_B1d §8, with this document staged next to the driver and its sha and path stamped in
the result; outputs under `/marimo/results`.

# 07 · CMI pilot — estimator validation on a synthetic system

**File:** `results/information/cmi_pilot.json` (written 2026-09-04 by a local re-run; previously absent) · **Script:** `src/information/cmi_pilot.py` · **Model:** none (synthetic linear-Gaussian) · **Status in registry:** category (a), `pending` (`cmi.pilot_spurious`); was category (c)

## What was run
A synthetic two-agent linear-Gaussian system with a known hidden channel β from A to B and a message channel that a "lossy encoder" (rank 2 plus noise) reads out. The Gaussian log-det CMI estimator is run with N = 2000 samples over 12 sampling seeds, against a "truth" obtained by running the same estimator at N = 400,000. CPU-only; ~6 s; byte-identical across two runs.

## What the file shows
- Beta sweep (`beta_sweep[i]`): at β = 0 / 0.1 / 0.2 / 0.35 / 0.5 / 1.0, truth 0.000 / 0.049 / 0.187 / 0.513 / 0.919 / 2.360; estimate with the full conditioner 0.009 / 0.061 / 0.201 / 0.529 / 0.934 / 2.373; estimate with the lossy conditioner 0.882 / 0.925 / 1.013 / 1.214 / 1.473 / 2.513; shuffled null ≈ 0.008 throughout.
- `confound.beta0_est_lossy_ARTIFACT` 0.8815 — the dossier's "0.88 nats". `confound.genuine_exceeds_artifact` true (1.013 at β = 0.2 vs 0.882 at β = 0).
- `te_hygiene`: 1.214 with B's previous state conditioned vs 0.816 without.
- `sample_size`: null falls 0.073 → 0.003 from N = 250 to 5000.
- `nonlinear.fraction_recovered` 0.767.

## What can be inferred
- With a faithful conditioner the plug-in estimator tracks the large-N reference to within +0.01–0.015 nats across the sweep. [supported by file]
- With the chosen rank-2 lossy conditioner, the estimator reports 0.88 nats at zero true coupling. [supported by file]
- The size of that artifact at other encoder ranks or noise levels is not in the file; only one encoder configuration was run. [needs: encoder sweep]
- "A genuine channel can be told apart from the artifact" rests on one comparison (β = 0.2 vs β = 0) at that one encoder, with no decision rule usable without knowing the β = 0 value. [needs: a rule that does not require ground truth]
- The shuffled null is a bias floor, not a check that would catch the lossy-conditioner artifact. [supported by file: the artifact is >100 null-sd above the null at zero coupling]
- Nothing in a linear-Gaussian synthetic licenses the estimator on a transformer residual stream. [needs: non-Gaussian / high-d validation]

## Status
Registry: resolves (0.8815, tol 0.001), category (a), `pending`. Whether the dossier sentence may be quoted again is an adjudication decision. See the critique below before making it.

### Independent critique (blind: saw only the script and the JSON)

## 1. What the code actually computes

**Generative model** (`sample`, lines 57–68): `zA ~ N(0,I_6)` (60), `zBprev ~ N(0,I_6)` (61), `m_true = zA Wmᵀ + 0.6·ε` (62), `zBnext = zBprev Wbbᵀ + m_true Wbmᵀ + β·(g(zA) Wbaᵀ) + 0.5·ε` (64–65), `C_lossy = m_true Wencᵀ + 0.3·ε` (66). All weights are `N(0,1)/√d` (47–51). With `nonlinear=False` (63) the joint law is exactly Gaussian.

**Estimator** (`gaussian_cmi`, 29–41): plug-in log-det CMI, `0.5(ld(ΣXZ)+ld(ΣYZ)−ld(ΣZ)−ld(ΣXYZ))` (40), i.e. it *assumes* joint Gaussianity — the same family the simulator draws from. `RIDGE=1e-3` (26) is added to every covariance diagonal (34); dimensions balance (14+14−8−20=0), so ridge bias largely cancels. `max(0,·)` is applied only in the unconditional branch (38), not the conditional one (41).

**Conditioners** (`est`, 83–88): `full_te` = {zBprev, m_true}; `lossy_te` = {zBprev, C_lossy}; `lossy_note` = {C_lossy}. `Wenc` is `(k_enc=2, d=6)` (51) — a rank-2 truncation of a 6-dim message plus σ=0.3 noise.

**"Truth"** (109, 123): *not* analytic ground truth — it is the same estimator run at `N_TRUTH=400_000` (96) with faithful conditioning. It verifies the estimator against its own large-sample limit.

**Null** (79–81): `X` alone is permuted, so `X ⟂ (Y,Z)` and the true CMI is *exactly* 0. It measures the plug-in bias floor `d_x·d_y/(2N)=36/(2N)`, not a conditional null.

**Seeds** (96, 108–158): system weights are *always* `make_system(0,…)` — one draw of W. The 12 seeds vary only `sample`'s rng (58), so every reported sd is sampling noise at fixed W.

I reproduced the closed-form Gaussian CMI for this system independently: analytic truth = [0, 0.0494, 0.1886, 0.5187, 0.9288, 2.3849]; analytic lossy = [0.8819, 0.9218, 1.0065, 1.2032, 1.4601, 2.5011]. Both match the JSON, so the numbers are arithmetically right; the objections below are about what they license.

## 2. Does the JSON support the numbers?

- C1: `confound.beta0_est_lossy_ARTIFACT = 0.8815364…`, `confound.beta0_truth_full = 4.919e-05`, `confound.beta0_est_full = 0.008862…`. "0.88 nats spurious" is literally in the file. `beta_sweep[0].lossy_sd = 0.02059` (12 sampling seeds, one W).
- C2: `confound["beta0.2_est_lossy"] = 1.01305…` vs artifact 0.88154 → Δ=0.1315 nats; `confound.genuine_exceeds_artifact = true`. `beta_sweep[2].lossy_sd = 0.02295`.
- C3 recovery: `est_full` vs `truth_full` per beta: 0.00886/0.0000492, 0.06131/0.04886, 0.20115/0.18655, 0.52856/0.51314, 0.93396/0.91914, 2.37297/2.36006 — uniform +0.009…+0.015 upward bias. Null: `shuffled_null` = 0.00814→0.00875 across all six betas, `null_sd ≈ 0.0021`. `sample_size[*].null` = 0.0734, 0.0351, 0.0176, 0.00838, 0.00340 for N=250…5000 — these are `36/(2N)` to within 2%, i.e. the null is the analytic bias floor, not an empirical finding.

## 3. Objections

**C1 — the 0.88 is a property of the encoder, and the encoder is never varied.** `k_enc=2, sigma_enc=0.3, sigma_m=0.6` are defaults (44) and *no call site overrides them* (108, 110–112, 122, 124–125, 130, 139–140, 149–150, 157–158). Varying only the encoder in my analytic replication: artifact = 1.214 (k=1), 0.882 (k=2), 0.633 (k=3), 0.477 (k=4), 0.401 (k=5), 0.373 (k=6 with σ=0.3), and exactly 0 at k=6, σ→0. Sweeping σ_enc at k=2: 0.823 (σ=0) → 1.204 (σ=1). So the confound spans 0–1.2 nats over the obvious neighbourhood of the chosen point; 0.88 is one arbitrary coordinate. Also missing: the denominator. Total message-mediated information `I(zA;zBnext|zBprev)` at β=0 is 1.476 nats (my computation), so "0.88 nats" is "the rank-2 readout captured 40% of the channel" — the informative statement, and it is not in the JSON.

**C2 — no decision rule; the comparison is an oracle.** The test is `e_lossy_b02 > e_lossy + 0.02` (line 134), a hard-coded margin unrelated to any estimated sd (`lossy_sd ≈ 0.021–0.023`), evaluated at a single β=0.2. The separation is real *given the numbers* (Δ=0.1315 ≈ 6 per-seed sd, ≈21 sem), but "distinguishable" requires knowing `e_lossy` at β=0 — the artifact baseline — which on a real system is exactly the unknown. Worse, the artifact moves under the conditioner more than the signal moves under β: artifact at k=1 (1.214) exceeds the β=0.2 genuine reading at k=2 (1.013), and artifact at k=2 (0.882) exceeds the *true* coupling at β=0.35 (`beta_sweep[3].truth_full = 0.5131`). The sweep does show monotonicity in β under lossy conditioning (0.8815→2.5134), which is more evidence than the single β=0.2 test the JSON records — but monotonicity at fixed, known encoder fidelity is not identifiability.

**C3 — the "null" cannot detect the confound it was introduced to police.** At β=0 the lossy estimate is 0.8815 against a null of 0.0081 (`beta_sweep[0]`): >100 sd of "significant coupling" where the true coupling is zero. Reporting "the null reads ~0" alongside "the estimator recovers true coupling" invites the reader to treat the null as a validity check; it is only a bias floor, because shuffling X (79–81) breaks X–Z as well as X–Y. A conditional/local permutation preserving X–Z is what would be needed.

**Linearity.** The estimator's Gaussian assumption is satisfied by construction, so the pilot tests finite-sample behaviour, not model adequacy. The only stress test is `tanh` (63), a near-linear map over ±1 for standard-normal inputs, and `nonlinear.fraction_recovered = 0.7665` compares two *different* systems with no matched truth (tanh shrinks Var(g(zA)), so equal β is not equal true coupling) — the 77% conflates variance shrinkage with nonlinearity blindness. Nothing here licenses d≈10³–10⁴ non-Gaussian transformer residual streams, where the `d_x·d_y/(2N)` bias alone is prohibitive and any dimension reduction changes the estimand in the same way `Wenc` does.

## 4. What each claim would need

- **C1**: sweep `k_enc ∈ {1..6}` and `sigma_enc`, report the artifact as a curve and as a *fraction* of `I(zA;zBnext|zBprev)`; resample W over ≥20 system seeds so the sd covers system draw, not just sampling.
- **C2**: state a rule usable without ground truth — e.g. an estimated encoder-fidelity bound mapping to a maximum artifact, or a paired conditional-permutation test — then show ROC/power over (β × encoder fidelity), not one point and a `+0.02` constant. Nearest defensible claim as-is: *"at a fixed, known rank-2 conditioner, β=0.2 raises the estimate 0.13 nats (≈6 sd) above the β=0 artifact."*
- **C3**: qualify to faithful conditioning — *"under `full_te` the estimator tracks the analytic CMI to within the +0.009–0.015 nat plug-in bias; under the lossy conditioner it does not"* — and replace the marginal shuffle with a conditional null. Compute the analytic CMI in closed form instead of calling a 400k-sample run of the same estimator "truth".

## 5. Verdicts

- **C1 — SUPPORTED-WITH-CAVEATS**: the number is in `confound.beta0_est_lossy_ARTIFACT` and reproduces analytically, but it is one untested point in encoder space (0–1.2 nats nearby) with no denominator, from a single W draw.
- **C2 — NOT SUPPORTED as worded**: the recorded test is one β=0.2 point against a hard-coded 0.02 margin (line 134) and requires oracle knowledge of the β=0 artifact; encoder-rank variation moves the artifact more than β=0.2 moves the signal.
- **C3 — SUPPORTED-WITH-CAVEATS**: recovery holds only for `full_te` (`est_full` vs `truth_full`, +0.009–0.015 nats); under the realistic `lossy_te` it fails badly at low β, and the "~0 null" is the analytic `36/(2N)` bias floor from a marginal shuffle, which flags the 0.88-nat artifact as highly significant coupling.

## Reconciliation
No prior AUDIT finding addressed the pilot's content (only its provenance). The critique reproduces every number analytically, so the arithmetic is settled. Its substantive points bear directly on the pending adjudication: the 0.88 is one coordinate of an untested encoder sweep (0–1.2 nats nearby), the "distinguishable" claim uses a hard-coded 0.02 margin and oracle knowledge of the β = 0 artifact, and the null cannot flag the confound. Recommendation: quote C1 only with the encoder configuration stated and as a fraction of the total message-mediated information; do not reinstate C2 as worded.

"""CMI-estimator pilot — de-risk #2 for the workspace-coupling proposal.

Validates the coupling estimator C(A->B) = I(z_A ; z_B^next | z_B^prev, C) on a
SYNTHETIC two-agent linear-Gaussian system with KNOWN ground truth, before trusting
it on real agents. Answers:

  1. RECOVERY    — does the Granger/Gaussian CMI track true coupling across a beta sweep?
  2. NULL        — does the shuffled-pairing null read ~0 (bias floor)?
  3. THE CONFOUND — at beta=0 (message is the ONLY channel), does conditioning on a
                    LOSSY message-encoder falsely report residual coupling? Can a real
                    hidden channel (beta>0) be told apart from that artifact?
  4. TE HYGIENE  — does omitting z_B^prev inflate apparent coupling?
  5. SAMPLE SIZE — bias/variance vs #episodes (how much data E1/E2 actually need).
  6. NONLINEAR   — does the linear estimator miss nonlinear coupling?

Generative model (all linear in independent Gaussians -> jointly Gaussian):
  z_A ~ N(0,I);  z_Bprev ~ N(0,I)
  m_true = Wm z_A + eps_m            (the message; lossy in z_A if sigma_m>0)
  C_lossy = Wenc m_true + eps_enc    (compressed encoder readout of the message, rank k)
  z_Bnext = Wbb z_Bprev + Wbm m_true + beta*(Wba g(z_A)) + eps_b
    beta = DIRECT A->B coupling NOT through the message (the "hidden channel").
"""
import argparse, json
import numpy as np

RIDGE = 1e-3


def gaussian_cmi(X, Y, Z=None):
    """I(X;Y|Z) in nats for Gaussian variables via log-dets. Z=None -> I(X;Y)."""
    def cov(A):
        return np.atleast_2d(np.cov(A, rowvar=False))
    def logdet(A):
        A = A + RIDGE * np.eye(A.shape[0])
        return np.linalg.slogdet(A)[1]
    if Z is None or (hasattr(Z, "shape") and Z.shape[1] == 0):
        XY = np.hstack([X, Y])
        return max(0.0, 0.5 * (logdet(cov(X)) + logdet(cov(Y)) - logdet(cov(XY))))
    XZ, YZ, XYZ = np.hstack([X, Z]), np.hstack([Y, Z]), np.hstack([X, Y, Z])
    val = 0.5 * (logdet(cov(XZ)) + logdet(cov(YZ)) - logdet(cov(Z)) - logdet(cov(XYZ)))
    return val  # can be slightly negative from finite-sample; report raw


def make_system(seed, d=6, k_enc=2, sigma_m=0.6, sigma_b=0.5, sigma_enc=0.3,
                beta=0.0, nonlinear=False):
    rng = np.random.default_rng(seed)
    W = {"Wm": rng.standard_normal((d, d)) / np.sqrt(d),
         "Wbb": rng.standard_normal((d, d)) / np.sqrt(d),
         "Wbm": rng.standard_normal((d, d)) / np.sqrt(d),
         "Wba": rng.standard_normal((d, d)) / np.sqrt(d),
         "Wenc": rng.standard_normal((k_enc, d)) / np.sqrt(d),
         "d": d, "k": k_enc, "sm": sigma_m, "sb": sigma_b, "se": sigma_enc,
         "beta": beta, "nl": nonlinear}
    return W


def sample(W, N, seed):
    rng = np.random.default_rng(seed)
    d = W["d"]
    zA = rng.standard_normal((N, d))
    zBprev = rng.standard_normal((N, d))
    m_true = zA @ W["Wm"].T + W["sm"] * rng.standard_normal((N, d))
    gA = np.tanh(zA) if W["nl"] else zA
    zBnext = (zBprev @ W["Wbb"].T + m_true @ W["Wbm"].T
              + W["beta"] * (gA @ W["Wba"].T) + W["sb"] * rng.standard_normal((N, d)))
    C_lossy = m_true @ W["Wenc"].T + W["se"] * rng.standard_normal((N, W["k"]))
    return {"zA": zA, "zBprev": zBprev, "m_true": m_true,
            "C_lossy": C_lossy, "zBnext": zBnext}


def est(W, N, seed, cond="lossy_te", shuffle=False):
    """Estimate coupling under a conditioning choice. cond in:
       full_te   : condition on {zBprev, m_true}  (faithful message)
       lossy_te  : condition on {zBprev, C_lossy} (realistic lossy encoder)
       lossy_note: condition on {C_lossy} only    (omit zB own past)
    """
    D = sample(W, N, seed)
    X = D["zA"].copy()
    if shuffle:                                    # shuffled-pairing null
        r = np.random.default_rng(seed + 777)
        X = X[r.permutation(N)]
    Y = D["zBnext"]
    if cond == "full_te":
        Z = np.hstack([D["zBprev"], D["m_true"]])
    elif cond == "lossy_te":
        Z = np.hstack([D["zBprev"], D["C_lossy"]])
    elif cond == "lossy_note":
        Z = D["C_lossy"]
    return gaussian_cmi(X, Y, Z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/cmi_pilot.json")
    args = ap.parse_args()
    N_TRUTH, N_EST, SEEDS = 400_000, 2000, list(range(12))
    out = {}

    def avg(fn):
        v = [fn(s) for s in SEEDS]
        return float(np.mean(v)), float(np.std(v))

    print("=== 1. RECOVERY + 3. CONFOUND: beta sweep (N_est=2000, 12 seeds) ===")
    print(f"{'beta':>5} {'truth(full)':>12} {'est full_te':>13} {'est lossy_te':>14} "
          f"{'shuffled null':>14}")
    rec = []
    for beta in [0.0, 0.1, 0.2, 0.35, 0.5, 1.0]:
        W = make_system(0, beta=beta)
        truth = est(W, N_TRUTH, 0, "full_te")               # faithful-cond truth
        full_m, full_s = avg(lambda s: est(make_system(0, beta=beta), N_EST, s, "full_te"))
        lossy_m, lossy_s = avg(lambda s: est(make_system(0, beta=beta), N_EST, s, "lossy_te"))
        null_m, null_s = avg(lambda s: est(make_system(0, beta=beta), N_EST, s, "lossy_te", shuffle=True))
        rec.append({"beta": beta, "truth_full": truth, "est_full": full_m,
                    "est_lossy": lossy_m, "shuffled_null": null_m,
                    "lossy_sd": lossy_s, "null_sd": null_s})
        print(f"{beta:>5.2f} {truth:>12.4f} {full_m:>10.4f}±{full_s:.3f} "
              f"{lossy_m:>10.4f}±{lossy_s:.3f} {null_m:>10.4f}±{null_s:.3f}")
    out["beta_sweep"] = rec

    print("\n=== 3b. CONFOUND ISOLATED: beta=0 (message is the ONLY channel) ===")
    print("If the estimator is honest, faithful conditioning -> 0. Lossy encoder may fake coupling.")
    W0 = make_system(0, beta=0.0)
    t_full = est(W0, N_TRUTH, 0, "full_te")
    e_full = avg(lambda s: est(make_system(0, beta=0.0), N_EST, s, "full_te"))[0]
    e_lossy = avg(lambda s: est(make_system(0, beta=0.0), N_EST, s, "lossy_te"))[0]
    print(f"  beta=0 truth (condition on FULL message)   : {t_full:.4f}")
    print(f"  beta=0 est   (condition on FULL message)   : {e_full:.4f}   <- should be ~0")
    print(f"  beta=0 est   (condition on LOSSY encoder)  : {e_lossy:.4f}   <- SPURIOUS if >0")
    # is a genuine hidden channel distinguishable from the lossy artifact?
    e_lossy_b02 = avg(lambda s: est(make_system(0, beta=0.2), N_EST, s, "lossy_te"))[0]
    out["confound"] = {"beta0_truth_full": t_full, "beta0_est_full": e_full,
                       "beta0_est_lossy_ARTIFACT": e_lossy,
                       "beta0.2_est_lossy": e_lossy_b02,
                       "genuine_exceeds_artifact": bool(e_lossy_b02 > e_lossy + 0.02)}
    print(f"  genuine beta=0.2 (lossy cond)              : {e_lossy_b02:.4f}   "
          f"{'> artifact — DISTINGUISHABLE' if e_lossy_b02 > e_lossy + 0.02 else '<= artifact — CONFOUNDED'}")

    print("\n=== 4. TE HYGIENE: omitting z_Bprev (beta=0.35) ===")
    with_prev = avg(lambda s: est(make_system(0, beta=0.35), N_EST, s, "lossy_te"))[0]
    no_prev = avg(lambda s: est(make_system(0, beta=0.35), N_EST, s, "lossy_note"))[0]
    out["te_hygiene"] = {"with_zBprev": with_prev, "without_zBprev": no_prev}
    print(f"  condition on {{zBprev, C}} : {with_prev:.4f}")
    print(f"  condition on {{C}} only    : {no_prev:.4f}   "
          f"({'inflated' if no_prev > with_prev + 0.02 else 'similar'})")

    print("\n=== 5. SAMPLE SIZE: bias/variance vs #episodes (beta=0.35, lossy_te) ===")
    ss = []
    for N in [250, 500, 1000, 2000, 5000]:
        m, s = avg(lambda sd: est(make_system(0, beta=0.35), N, sd, "lossy_te"))
        nn, ns = avg(lambda sd: est(make_system(0, beta=0.35), N, sd, "lossy_te", shuffle=True))
        ss.append({"N": N, "est": m, "est_sd": s, "null": nn, "null_sd": ns})
        print(f"  N={N:>5}: est {m:.4f}±{s:.3f}   null {nn:.4f}±{ns:.3f}   "
              f"SNR {(m-nn)/max(s,1e-6):.1f}")
    out["sample_size"] = ss

    print("\n=== 6. NONLINEAR coupling: linear estimator blind spot (beta=0.5) ===")
    lin = avg(lambda s: est(make_system(0, beta=0.5, nonlinear=False), N_EST, s, "lossy_te"))[0]
    nl = avg(lambda s: est(make_system(0, beta=0.5, nonlinear=True), N_EST, s, "lossy_te"))[0]
    out["nonlinear"] = {"linear_coupling_est": lin, "nonlinear_coupling_est": nl,
                        "fraction_recovered": nl / lin if lin else float("nan")}
    print(f"  linear coupling    : {lin:.4f}")
    print(f"  nonlinear coupling : {nl:.4f}   (linear estimator recovers "
          f"{100*nl/lin:.0f}% of equal-strength nonlinear coupling)")

    import os
    os.makedirs("results", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

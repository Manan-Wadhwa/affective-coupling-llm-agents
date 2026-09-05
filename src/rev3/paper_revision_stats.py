"""paper_revision_stats.py -- the numbers the reviews asked for, from the committed result files.

For each B1 family file: every hooked arm against `none` (scenario-blocked, dose-paired),
blocked fractions relative to the none slope, per-dose scenario-bootstrap CIs for every arm
(figure bands), shape contrasts on the none arm (dose 0.33 - 0 and 1.0 - 0.67, scenario-
paired), the other-speaker probe's dose slope, the dose-0 A-span readout shift per arm, a
two-sided bootstrap p and BH-FDR q for every arm-vs-none contrast over the testable emotions,
and B1c's median blocked fraction with calm included. CPU only. Output:
results/rev3/paper_revision_stats.json.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import acl_core as C

FILES = {"b1c_27b": ("results/rev3/b1c_alllayer_qwen36-27b.json", ["emo13_42", "emo_all", "rand_all", "text", "text_keep"]),
         "b1d_27b": ("results/rev3/b1d_subspace_qwen36-27b.json", ["emo_all", "sub_all", "perm_all", "randsub_all"]),
         "b1e_27b": ("results/rev3/b1e_footprint_qwen36-27b.json", ["emo_all", "permdir_all", "pc1_all"]),
         "b1c_8b": ("results/rev3/b1c_alllayer_llama3-abl.json", ["emo13_42", "emo_all", "rand_all", "text", "text_keep"]),
         "b1e_8b": ("results/rev3/b1e_footprint_llama3-abl.json", ["emo_all", "permdir_all", "pc1_all"])}
EMOS = ["desperate", "afraid", "happy", "calm", "sad", "angry"]


def gather(rows, key, e, arm):
    P, B = {}, {}
    for r in rows:
        if r["emotion"] == e and r["arm"] == arm and r.get("scenario"):
            P.setdefault(float(r["alpha"]), []).extend(r[key]); B.setdefault(float(r["alpha"]), []).extend(r["scenario"])
    return {d: np.asarray(v, float) for d, v in P.items()}, {d: np.asarray(v) for d, v in B.items()}


def balanced(Pa, Ba, Pb, Bb, doses):
    """Keep, per dose, the scenario keys present in both arms (rep-expanded: keys are (rep-order, scenario));
    the drivers' rows list scenarios in the same order per (dose, rep), so align by position within scenario id."""
    out = {}
    for d in doses:
        a, ba, b, bb = Pa[d], Ba[d], Pb[d], Bb[d]
        # group by scenario id, pair the k-th occurrence in each arm
        ia = {}; ib = {}
        for i, s in enumerate(ba): ia.setdefault(int(s), []).append(i)
        for i, s in enumerate(bb): ib.setdefault(int(s), []).append(i)
        keep_a, keep_b, blk = [], [], []
        for s in sorted(set(ia) & set(ib)):
            k = min(len(ia[s]), len(ib[s]))
            keep_a += ia[s][:k]; keep_b += ib[s][:k]; blk += [s] * k
        out[d] = (a[keep_a], b[keep_b], np.asarray(blk))
    return out


def contrast(doses, Pa, Ba, Pb, Bb, seed=0):
    bal = balanced(Pa, Ba, Pb, Bb, doses)
    A = {d: bal[d][0] for d in doses}; Bv = {d: bal[d][1] for d in doses}; blk = {d: bal[d][2] for d in doses}
    r = C.paired_slope_contrast(doses, A, Bv, block=blk, pair_doses=True, seed=seed)
    return {k: r[k] for k in ("diff", "ci", "sig", "n_blocks")}


def boot_p(doses, Pa, Ba, Pb, Bb, n_boot=5000, seed=0):
    """Two-sided bootstrap p for the paired slope difference over scenarios (same construction as _decide)."""
    bal = balanced(Pa, Ba, Pb, Bb, doses); rng = np.random.default_rng(seed)
    # per-scenario paired slope difference
    scen = sorted(set(int(s) for d in doses for s in bal[d][2]))
    X = np.asarray(doses); Xc = X - X.mean()
    dd = []
    for s in scen:
        ma = [bal[d][0][bal[d][2] == s].mean() for d in doses]; mb = [bal[d][1][bal[d][2] == s].mean() for d in doses]
        sa = float(np.dot(Xc, ma) / np.dot(Xc, Xc)); sb = float(np.dot(Xc, mb) / np.dot(Xc, Xc)); dd.append(sa - sb)
    dd = np.asarray(dd); b = dd[rng.integers(0, len(dd), size=(n_boot, len(dd)))].mean(1)
    p = 2 * min(float((b <= 0).mean()), float((b >= 0).mean())); return float(min(1.0, max(p, 1.0 / n_boot))), len(dd)


def bh(ps):
    keys = list(ps); p = np.asarray([ps[k] for k in keys]); m = len(p); order = np.argsort(p); q = np.empty(m)
    prev = 1.0
    for rank, i in list(enumerate(order))[::-1]:
        val = min(prev, p[i] * m / (rank + 1)); q[i] = val; prev = val
    return dict(zip(keys, [float(x) for x in q]))


def dose_means_ci(P, B, doses, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed); out = {}
    for d in doses:
        v, b = P[d], B[d]; scen = sorted(set(int(s) for s in b)); m = np.asarray([v[b == s].mean() for s in scen])
        bs = m[rng.integers(0, len(m), size=(n_boot, len(m)))].mean(1)
        out[str(d)] = {"mean": float(v.mean()), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "n_scen": len(scen)}
    return out


def paired_dose_diff(P, B, d1, d0, n_boot=5000, seed=0):
    rng = np.random.default_rng(seed); scen = sorted(set(int(s) for s in B[d1]) & set(int(s) for s in B[d0]))
    diff = np.asarray([P[d1][B[d1] == s].mean() - P[d0][B[d0] == s].mean() for s in scen])
    bs = diff[rng.integers(0, len(diff), size=(n_boot, len(diff)))].mean(1)
    return {"diff": float(diff.mean()), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "n_scen": len(scen)}


def main():
    res = {}
    for tag, (path, arms) in FILES.items():
        d = json.load(open(path)); rows = d["rows"]; doses = [float(x) for x in d["doses"]]; S = d["summary"]
        out = {"file": path, "per_emotion": {}}
        for e in EMOS:
            Pn, Bn = gather(rows, "B_present_e", e, "none")
            if not Pn: continue
            ns = S[e]["none"]["present_slope"]; none_slope = float(ns["slope"])
            ent = {"none_slope": none_slope, "none_slope_ci": ns["ci"], "testable": bool(ns["sig"]), "arm_vs_none": {}, "bands": {}, "dose0_readout_shift": {}}
            ent["bands"]["none"] = dose_means_ci(Pn, Bn, doses)
            ent["shape"] = {"step_0_to_033": paired_dose_diff(Pn, Bn, 0.33, 0.0), "step_067_to_1": paired_dose_diff(Pn, Bn, 1.0, 0.67)}
            Po, Bo = gather(rows, "B_other_e", e, "none"); ent["other_probe_none_slope"] = C.slope_ci(doses, Po, block=Bo[doses[0]])
            cr0 = S[e]["none"]["ctx_readout_by_dose"]["0.0"]
            for arm in arms:
                Pa, Ba = gather(rows, "B_present_e", e, arm)
                if not Pa or S[e].get(arm) is None: continue
                c = contrast(doses, Pa, Ba, Pn, Bn); p, nb = boot_p(doses, Pa, Ba, Pn, Bn)
                c["p_boot"] = p; c["blocked_fraction"] = float(-c["diff"] / none_slope) if abs(none_slope) > 1e-9 else None
                c["blocked_range"] = sorted(-x / none_slope for x in c["ci"]) if abs(none_slope) > 1e-9 else None
                ent["arm_vs_none"][arm] = c
                ent["bands"][arm] = dose_means_ci(Pa, Ba, doses)
                ent["dose0_readout_shift"][arm] = float(S[e][arm]["ctx_readout_by_dose"]["0.0"] - cr0)
                if arm == "emo_all":
                    Poa, Boa = gather(rows, "B_other_e", e, arm); ent["other_probe_emo_slope"] = C.slope_ci(doses, Poa, block=Boa[doses[0]])
            out["per_emotion"][e] = ent
        testable = [e for e in EMOS if e in out["per_emotion"] and out["per_emotion"][e]["testable"]]
        out["bh_q_over_testable"] = {}
        for arm in arms:
            ps = {e: out["per_emotion"][e]["arm_vs_none"][arm]["p_boot"] for e in testable if arm in out["per_emotion"][e]["arm_vs_none"]}
            if ps: out["bh_q_over_testable"][arm] = bh(ps)
        res[tag] = out
        print(tag, "testable", testable, flush=True)
        for e in EMOS:
            if e not in out["per_emotion"]: continue
            en = out["per_emotion"][e]
            print(f"  {e:9s} none {en['none_slope']:+7.1f} other-probe none slope {en['other_probe_none_slope']['slope']:+7.1f} sig={en['other_probe_none_slope']['sig']} | shape 0->0.33 {en["shape"]["step_0_to_033"]['diff']:+6.1f} {[round(x,1) for x in en["shape"]["step_0_to_033"]['ci']]}  0.67->1 {en["shape"]["step_067_to_1"]['diff']:+6.1f} {[round(x,1) for x in en["shape"]["step_067_to_1"]['ci']]}")
            for arm, c in en["arm_vs_none"].items():
                q = out["bh_q_over_testable"].get(arm, {}).get(e)
                print(f"      {arm:11s} vs none {c['diff']:+7.1f} [{c['ci'][0]:+.1f},{c['ci'][1]:+.1f}] sig={c['sig']} p={c['p_boot']:.4f} q={q if q is None else round(q,3)} frac={c['blocked_fraction'] if c['blocked_fraction'] is None else round(c['blocked_fraction'],3)} dose0shift={en['dose0_readout_shift'][arm]:+.3f}")
    # B1c medians with and without calm
    b = res["b1c_27b"]; fr = {e: -b["per_emotion"][e]["arm_vs_none"]["emo_all"]["diff"] / b["per_emotion"][e]["none_slope"] for e in EMOS}
    S = json.load(open(FILES["b1c_27b"][0]))["summary"]
    dec = {e: S[e]["blocked_fraction"]["emo_all_vs_rand_all"]["point"] for e in EMOS}
    res["b1c_medians"] = {"decision_contrast_5_testable": float(np.median([dec[e] for e in EMOS if e != "calm"])), "decision_contrast_all_6": float(np.median(list(dec.values()))),
                          "emo_vs_none_5_testable": float(np.median([fr[e] for e in EMOS if e != "calm"])), "emo_vs_none_all_6": float(np.median(list(fr.values()))), "per_emotion_emo_vs_none": fr}
    print("b1c medians", res["b1c_medians"])
    json.dump(res, open("results/rev3/paper_revision_stats.json", "w"), indent=1)
    print("REVSTATS_DONE")


if __name__ == "__main__":
    main()

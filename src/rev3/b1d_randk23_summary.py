"""b1d_randk23_summary.py -- PREREG_B1d addendum 2: the footprint-matched random frame.

Reads the rank-k `randsub_all` run (arms none, randsub_all) and the rank-5 B1d run it is matched
to. For every emotion: none slope (testability), `randsub_all - none` recomputed from rows with
the scenario-blocked, dose-paired contrast, two-sided bootstrap p and BH-FDR q over the testable
emotions, blocked fraction with range, dose-0 readout shift, the arm's removed norm and the §4.2
share against the rank-5 run's `perm_all` and `randsub_all`. The addendum's fixed reading is
applied mechanically: "footprint" if the matched frame blocks afraid, sad or angry at q < 0.05
with a negative contrast, else "fitting". No decision verdict. CPU only.
"""
import argparse, hashlib, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_revision_stats import gather, contrast, boot_p, bh, EMOS

READING_EMOS = ("afraid", "sad", "angry")


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="results/rev3/b1d_subspace_qwen36-27b-randk23.json")
    ap.add_argument("--ref", default="results/rev3/b1d_subspace_qwen36-27b.json")
    ap.add_argument("--out", default="results/rev3/b1d_randk23_summary.json")
    a = ap.parse_args()
    d = json.load(open(a.file)); ref = json.load(open(a.ref))
    rows = d["rows"]; doses = [float(x) for x in d["doses"]]; S = d["summary"]; R = ref["summary"]
    out = {"file": a.file, "file_sha256_16": sha(a.file), "ref": a.ref, "ref_sha256_16": sha(a.ref),
           "randsub_k": d.get("randsub_k", d.get("_provenance", {}).get("config", {}).get("randsub_k")),
           "ref_sub_k": ref.get("sub_k"), "arms": d.get("arms"), "verdict_in_file": d.get("verdict"), "per_emotion": {}}
    for e in EMOS:
        if e not in S or S[e].get("randsub_all") is None or S[e].get("none") is None: continue
        Pn, Bn = gather(rows, "B_present_e", e, "none"); Pa, Ba = gather(rows, "B_present_e", e, "randsub_all")
        if not Pn or not Pa: continue
        ns = S[e]["none"]["present_slope"]; none_slope = float(ns["slope"])
        c = contrast(doses, Pa, Ba, Pn, Bn); p, nb = boot_p(doses, Pa, Ba, Pn, Bn)
        c["p_boot"] = p; c["n_scen"] = nb
        c["blocked_fraction"] = float(-c["diff"] / none_slope) if abs(none_slope) > 1e-9 else None
        c["blocked_range"] = sorted(-x / none_slope for x in c["ci"]) if abs(none_slope) > 1e-9 else None
        drv = S[e].get("randsub_all_vs_none") or {}
        ent = {"none_slope": none_slope, "none_slope_ci": ns["ci"], "testable": bool(ns["sig"]),
               "randsub_vs_none": c, "driver_contrast": {k: drv.get(k) for k in ("diff", "ci", "sig")},
               "dose0_readout_shift": float(S[e]["randsub_all"]["ctx_readout_by_dose"]["0.0"] - S[e]["none"]["ctx_readout_by_dose"]["0.0"]),
               "removed_norm": {"randsub_k": S[e]["randsub_all"].get("removed_norm"),
                                "ref_perm_all": (R.get(e, {}).get("perm_all") or {}).get("removed_norm"),
                                "ref_randsub_5": (R.get(e, {}).get("randsub_all") or {}).get("removed_norm"),
                                "ref_sub_all": (R.get(e, {}).get("sub_all") or {}).get("removed_norm")},
               "mc_subspace": S[e].get("mc_subspace"),
               "ref_perm_vs_none": {k: (R.get(e, {}).get("perm_all_vs_none") or {}).get(k) for k in ("diff", "ci", "sig")},
               "ref_none_slope": float((R.get(e, {}).get("none") or {}).get("present_slope", {}).get("slope", float("nan")))}
        out["per_emotion"][e] = ent
    testable = [e for e in out["per_emotion"] if out["per_emotion"][e]["testable"]]
    out["testable"] = testable
    out["bh_q_over_testable"] = bh({e: out["per_emotion"][e]["randsub_vs_none"]["p_boot"] for e in testable}) if testable else {}
    blocks = [e for e in READING_EMOS if e in out["bh_q_over_testable"] and out["bh_q_over_testable"][e] < 0.05
              and out["per_emotion"][e]["randsub_vs_none"]["diff"] < 0]
    out["reading"] = {"rule": "PREREG_B1d addendum 2: 'footprint' if the matched random frame blocks afraid, sad or angry at BH-FDR q<0.05 (negative contrast); else 'fitting'",
                      "blocks_at_fdr": blocks, "reading": "footprint" if blocks else "fitting"}
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"rank-{out['randsub_k']} random frame vs none  (file {out['file_sha256_16']}, ref rank-{out['ref_sub_k']} {out['ref_sha256_16']})  testable={testable}")
    for e, en in out["per_emotion"].items():
        c = en["randsub_vs_none"]; q = out["bh_q_over_testable"].get(e); rn = en["removed_norm"]
        print(f"  {e:9s} none {en['none_slope']:+7.1f} | randsub-none {c['diff']:+7.1f} [{c['ci'][0]:+.1f},{c['ci'][1]:+.1f}] p={c['p_boot']:.4f} q={'-' if q is None else round(q,3)} frac={None if c['blocked_fraction'] is None else round(c['blocked_fraction'],3)} | dose0 shift {en['dose0_readout_shift']:+.3f} | removed {rn['randsub_k']} (ref perm {rn['ref_perm_all']}, ref rand5 {rn['ref_randsub_5']}) | driver diff {en['driver_contrast']['diff']}")
    print("reading:", out["reading"]["reading"], "blocks_at_fdr", blocks)


if __name__ == "__main__":
    main()

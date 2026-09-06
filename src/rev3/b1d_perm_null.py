"""b1d_perm_null.py -- PREREG_B1d addendum 3: the permutation null for the permuted-label rank-5 frame.

Reads the fresh perm_all realisations (files b1d_subspace_qwen36-27b-perm<s>.json, or their
b1d_cells checkpoints if the analysis did not finish) and contrasts each against the B1d run's stored
none arm (same generation seeds; bit-identical) with the scenario-blocked, dose-paired bootstrap of
paper_revision_stats. Reports per realisation the six blocked fractions with CIs and the dose-0 readout
shift, and across realisations (original seed 0 included) the median, range and count significantly
negative. Applies the addendum's fixed reading mechanically. CPU only.
"""
import argparse, glob, hashlib, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_revision_stats import gather, contrast, boot_p, bh, EMOS

def rows_of(path):
    d = json.load(open(path))
    if "rows" in d: return d["rows"], d
    for k in ("cells", "rows_done", "done"):
        if k in d and isinstance(d[k], (list, dict)):
            v = d[k]; return (list(v.values()) if isinstance(v, dict) else v), d
    raise SystemExit(f"no rows in {path}: keys {list(d)[:8]}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="*", default=None)
    ap.add_argument("--ref", default="results/rev3/b1d_subspace_qwen36-27b.json")
    ap.add_argument("--out", default="results/rev3/b1d_perm_null.json")
    ap.add_argument("--tex", default="docs/writeups/table_perm_null.tex")
    a = ap.parse_args()
    ref = json.load(open(a.ref)); rrows = ref["rows"]; doses = [float(x) for x in ref["doses"]]; S = ref["summary"]
    files = a.files or sorted(glob.glob("results/rev3/b1d_subspace_qwen36-27b-perm*.json"))
    out = {"ref": a.ref, "realisations": {}, "across": {}}
    # original realisation from the reference run's own rows
    real = {"0": (rrows, {"source": a.ref})}
    for f in files:
        rows, d = rows_of(f); s = str(d.get("perm_seed_base", d.get("_provenance", {}).get("config", {}).get("perm_seed_base", f.rsplit("perm", 1)[1].split(".")[0])))
        real[s] = (rows, {"source": f, "sha256_16": hashlib.sha256(open(f, "rb").read()).hexdigest()[:16], "prereg": (d.get("prereg") or {}).get("sha256", "")[:8]})
    for s, (rows, meta) in real.items():
        ent = {"meta": meta, "per_emotion": {}}
        for e in EMOS:
            Pn, Bn = gather(rrows, "B_present_e", e, "none"); Pa, Ba = gather(rows, "B_present_e", e, "perm_all")
            if not Pn or not Pa: ent["per_emotion"][e] = None; continue
            none_slope = float(S[e]["none"]["present_slope"]["slope"])
            c = contrast(doses, Pa, Ba, Pn, Bn); p, nb = boot_p(doses, Pa, Ba, Pn, Bn)
            c["p_boot"] = p; c["blocked_fraction"] = float(-c["diff"] / none_slope); c["blocked_range"] = sorted(-x / none_slope for x in c["ci"])
            c["sig_negative"] = bool(c["sig"] and c["diff"] < 0); c["testable"] = bool(S[e]["none"]["present_slope"]["sig"])
            # dose-0 readout shift from rows (mean ctx_readout at dose 0, perm - none)
            r0 = [np.mean(r["ctx_readout_e"]) for r in rows if r["emotion"] == e and r["arm"] == "perm_all" and float(r["alpha"]) == 0.0]
            n0 = [np.mean(r["ctx_readout_e"]) for r in rrows if r["emotion"] == e and r["arm"] == "none" and float(r["alpha"]) == 0.0]
            c["dose0_readout_shift"] = float(np.mean(r0) - np.mean(n0)) if r0 and n0 else None
            ent["per_emotion"][e] = c
        testable = [e for e in EMOS if ent["per_emotion"].get(e) and ent["per_emotion"][e]["testable"]]
        ent["bh_q_over_testable"] = bh({e: ent["per_emotion"][e]["p_boot"] for e in testable})
        out["realisations"][s] = ent
    fresh = [s for s in out["realisations"] if s != "0"]
    for e in EMOS:
        vals = [out["realisations"][s]["per_emotion"][e] for s in out["realisations"] if out["realisations"][s]["per_emotion"].get(e)]
        fr = [v["blocked_fraction"] for v in vals]
        out["across"][e] = {"n": len(vals), "median": float(np.median(fr)) if fr else None, "min": float(min(fr)) if fr else None, "max": float(max(fr)) if fr else None,
                            "n_sig_negative": int(sum(v["sig_negative"] for v in vals)),
                            "n_sig_negative_fresh": int(sum(out["realisations"][s]["per_emotion"][e]["sig_negative"] for s in fresh if out["realisations"][s]["per_emotion"].get(e))),
                            "n_fresh": len(fresh)}
    blocked_all3 = sum(1 for s in fresh if all(out["realisations"][s]["per_emotion"].get(e) and out["realisations"][s]["per_emotion"][e]["sig_negative"] for e in ("afraid", "sad", "angry")))
    out["reading"] = {"n_fresh": len(fresh), "fresh_realisations_blocking_afraid_sad_angry": blocked_all3,
                      "rule": "addendum 3: 'estimator' if >= 3 of 4 fresh realisations block afraid, sad and angry (unadjusted CI below zero); 'unlucky draw' if <= 1; else 'mixed'",
                      "reading": "estimator" if blocked_all3 >= 3 else ("unlucky draw" if blocked_all3 <= 1 else "mixed")}
    json.dump(out, open(a.out, "w"), indent=1)
    def f(v): return "--" if v is None else (f"{v:+.2f}")
    L = [r"\begin{tabular}{lcccccc}", r"\toprule", r"permutation & desperate & afraid & happy & calm$^{u}$ & sad & angry \\", r"\midrule"]
    for s in sorted(out["realisations"], key=int):
        cells = []
        for e in EMOS:
            c = out["realisations"][s]["per_emotion"].get(e)
            if not c: cells.append("--"); continue
            v = f"{c['blocked_fraction']:+.2f}"; v = f"({v})" if e == "calm" else v
            if c["sig_negative"]: v += r"$^{\cdot}$"
            cells.append(v)
        L.append(f"{'0 (paper)' if s == '0' else s} & " + " & ".join(cells) + r" \\")
    L += [r"\midrule", "median & " + " & ".join((f"{out['across'][e]['median']:+.2f}" if out['across'][e]['median'] is not None else "--") for e in EMOS) + r" \\",
          "blocks (CI $<0$) & " + " & ".join(f"{out['across'][e]['n_sig_negative']}/{out['across'][e]['n']}" for e in EMOS) + r" \\", r"\bottomrule", r"\end{tabular}"]
    open(a.tex, "w").write("\n".join(L) + "\n")
    print(f"realisations {sorted(out['realisations'], key=int)}; reading: {out['reading']}")
    for e in EMOS:
        x = out["across"][e]; print(f"  {e:9s} median {x['median']:+.3f} range [{x['min']:+.3f}, {x['max']:+.3f}] sig-neg {x['n_sig_negative']}/{x['n']} (fresh {x['n_sig_negative_fresh']}/{x['n_fresh']})")
    print("wrote", a.out, a.tex)

if __name__ == "__main__":
    main()

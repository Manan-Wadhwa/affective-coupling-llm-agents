"""A3 — the present-vs-other dissociation, tested (RESEARCH_PLAN rev 3, §2.3 A3).

THE EXHIBIT. `coupling_e2.py`'s own docstring calls the present-vs-other dissociation
"the load-bearing result". In `e2ci_qwen36-27b.json` it holds for 1 of 6 emotions; under
difference-of-means on the matched dose grid it is reported to hold for 5 of 6. A published
central claim that is FALSE under the estimator used and TRUE under the stable one is a
far stronger exhibit for Paper A than any single emotion's slope -- and it converts Paper
B's retracted headline into Paper A's evidence.

What the original never ran, and this does:

  * a PAIRED test on present-minus-other slopes, with CIs. present and other are measured
    on the SAME B utterance from the same forward pass, so the contrast is paired and an
    unpaired interval is the wrong one. (AUDIT 18 separately found the published bootstrap
    resampled each alpha cell independently although the same 29 scenarios recur across
    cells -- unpaired bootstrap on paired data, understating the intervals used to declare
    significance. Here the bootstrap is scenario-BLOCKED.)

  * the estimator isolated as the only moving part. The steering direction is held FIXED
    (difference-of-means, the stable one) while the READOUT estimator varies over
    {logreg, dom}. Any change in the conclusion is then attributable to the readout alone.
    Varying both at once -- which is what comparing the published run to a dom run does --
    confounds the direction B was pushed along with the ruler used to measure B.

  * the published configuration reproduced alongside it (`logreg` steering + `logreg`
    readout), so the exhibit is anchored to a number the reader can find in the old file.

  * a NON-CIRCULAR readout. Steering directions come from one half of the dialogue pool,
    readout probes from the disjoint other half. The published run used one probe for both,
    so its readout was partly a function of its own steering direction.

Reuses the A2 activation cache when present: the dialogue pool and its features are the
expensive part and A2 has already paid for them on this machine.

Usage: python a3_dissociation.py --model Qwen/Qwen3.6-27B --tag qwen36-27b --reps 2
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import acl_core as C

DOSES = (0.0, 0.5, 1.0)          # matched grid; alpha=2 excluded by pre-data declaration
EMOS = ("desperate", "afraid", "happy", "calm", "sad", "angry")
# (steering estimator, readout estimator) -- the published config, and the corrected one
# with the readout varied while the steering is held fixed.
CONFIGS = (("logreg", "logreg"), ("dom", "logreg"), ("dom", "dom"))
PROBE_K = 60


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--outdir", default="/marimo/out")
    ap.add_argument("--workdir", default="/marimo/work")
    ap.add_argument("--pool-cache", default="", help="reuse an existing pool jsonl (A2's)")
    ap.add_argument("--probe-k", type=int, default=PROBE_K,
                    help="MUST match the k of --pool-cache; the cache is keyed by "
                         "position in pool_jobs(k, seed)")
    ap.add_argument("--gen-bs", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True); os.makedirs(a.workdir, exist_ok=True)

    prov = C.Provenance(
        script="src/rev3/a3_dissociation.py",
        config={"doses": DOSES, "emotions": EMOS, "configs": CONFIGS, "reps": a.reps,
                "probe_k": a.probe_k, "n_scenarios": len(C.SCENARIOS), "max_new": 110},
        model_id=a.model, model_revision="", seeds={"pool": a.seed, "run": a.seed},
        control_pointers={
            "paired_contrast": "acl_core.py::paired_slope_contrast -- present and other "
                               "resampled together, because both are read from the same "
                               "forward pass over the same B utterance",
            "blocked_bootstrap": "acl_core.py::slope_ci(block=...) -- resamples SCENARIOS, "
                                 "the unit of independence, not individual utterances",
            "estimator_isolated": "a3_dissociation.py::CONFIGS -- steering held fixed at "
                                  "dom while readout varies, so the contrast is not "
                                  "confounded with the direction B was pushed along",
            "non_circular_readout": "steering probe fitted on DIR half, readout probe on "
                                    "the disjoint READ half",
        })

    h = C.load(a.model)
    prov.model_revision = h.revision
    focus = h.focus(); steer_layer = focus - 1

    pool_cache = a.pool_cache or os.path.join(a.workdir, f"a3pool_{a.tag}.jsonl")
    items = C.generate_pool(h, a.probe_k, pool_cache, seed=a.seed, bs=a.gen_bs)
    feats, yp, yo = C.pool_features(h, items, [focus],
                                    cache=os.path.join(a.workdir, f"a3feat_{a.tag}.npz"))
    N = len(items)
    rs = np.random.default_rng(a.seed).permutation(N)
    DIR, READ = rs[:N // 2], rs[N // 2:]
    print(f"[A3] pool n={N}  DIR {len(DIR)}  READ {len(READ)}  focus {focus}", flush=True)

    X = feats[focus]
    steer_probe = {m: C.fit_direction(X[DIR], yp[DIR], m, seed=a.seed,
                                      pair_on=yo[DIR]) for m in ("logreg", "dom")}
    read_present = {m: C.fit_direction(X[READ], yp[READ], m, seed=a.seed,
                                       pair_on=yo[READ]) for m in ("logreg", "dom")}
    read_other = {m: C.fit_direction(X[READ], yo[READ], m, seed=a.seed)
                  for m in ("logreg", "dom")}
    stability = {m: C.split_half(X, yp, m, n_per_half=min(N // 2, 600), seeds=range(10),
                                 pair_on=yo) for m in ("logreg", "dom")}
    for m, s in stability.items():
        print(f"[A3] split-half({m}) {s['mean']:.3f} CI {s['ci']}", flush=True)

    rms = float(np.linalg.norm(X, axis=1).mean())
    steer = C.Steer(h, steer_layer)
    S = C.SCENARIOS
    rlog = C.ResponseLog(os.path.join(a.workdir, f"a3responses_{a.tag}.jsonl"))
    ck = C.Checkpoint(os.path.join(a.workdir, f"a3_cells_{a.tag}.json"),
                      {"doses": DOSES, "emos": EMOS, "cfg": CONFIGS, "reps": a.reps,
                       "n": N, "focus": focus, "probe_k": a.probe_k})

    def run_cell(est_steer, e, alpha, rep):
        """One (steering estimator, emotion, dose, rep): A speaks steered, B replies, and B's
        reply is scored for present-e and other-e under EVERY readout estimator at once --
        the same activations, so the readout is the only thing that differs."""
        ei = C.EMOTIONS.index(e)
        names = [C.NAMES[(i + rep) % len(C.NAMES)] for i in range(len(S))]
        prompts, ctx = [], []
        for (setting, bopen), (A, B) in zip(S, names):
            s = setting.format(A=A, B=B); bo = bopen.format(A=A, B=B)
            prompts.append(f"{s}\n{bo}\nContinue the conversation. Write only {A}'s next "
                           f"reply as one short paragraph, starting with '{A}:'.")
            ctx.append({"A": A, "B": B, "setting": s, "bopen": bo, "scenario": len(ctx)})
        d = C.raw_direction(steer_probe[est_steer], ei)
        vec = None if alpha == 0 else torch.tensor(
            alpha * rms * d, dtype=h.model.dtype, device=h.model.device)
        a_reps = C.gen(h, prompts, max_new=110, bs=a.gen_bs, seed=a.seed + rep,
                       first_line=True, log=rlog, tag=f"A_{est_steer}_{e}_a{alpha}_r{rep}",
                       hooks_on=(lambda: steer.on(vec)) if vec is not None else None,
                       hooks_off=steer.off)
        bprompts = []
        for c, ar in zip(ctx, a_reps):
            A = c["A"]
            ar = ar if ar.startswith(f"{A}:") else f"{A}: {ar}"
            c["ar"] = ar
            c["conv"] = f"{c['setting']}\n{c['bopen']}\n{ar}"
            bprompts.append(f"{c['conv']}\nWrite only {c['B']}'s next reply as one short "
                            f"paragraph, starting with '{c['B']}:'.")
        steer.off()
        b_reps = C.gen(h, bprompts, max_new=110, bs=a.gen_bs, seed=a.seed + 1000 + rep,
                       first_line=True, log=rlog, tag=f"B_{est_steer}_{e}_a{alpha}_r{rep}")
        texts, spans = [], []
        for c, br in zip(ctx, b_reps):
            B = c["B"]
            br = br if br.startswith(f"{B}:") else f"{B}: {br}"
            full = f"{c['conv']}\n{br}"
            texts.append(full); spans.append((full.rfind(f"{B}:") + len(f"{B}:"), len(full)))
        f = C.pool_spans(h, texts, spans, [focus], bs=8)[focus]
        out = {"scenario": [c["scenario"] for c in ctx],
               "degenerate": float(np.mean([C.is_degenerate(C.degeneracy(b)) for b in b_reps]))}
        for m in ("logreg", "dom"):
            Xs = (f - read_present[m]["mu"]) / read_present[m]["sd"]
            out[f"present_{m}"] = (Xs @ read_present[m]["C"].T)[:, ei].tolist()
            Xo = (f - read_other[m]["mu"]) / read_other[m]["sd"]
            out[f"other_{m}"] = (Xo @ read_other[m]["C"].T)[:, ei].tolist()
        return out

    rows = []
    t0 = time.time()
    steer_ests = list(dict.fromkeys(c[0] for c in CONFIGS))
    for est_steer in steer_ests:
        for e in EMOS:
            for alpha in DOSES:
                for rep in range(a.reps):
                    cid = f"{est_steer}/{e}/{alpha}/{rep}"
                    if ck.has(cid):
                        rows.append(dict(ck.get(cid), _cid=cid)); continue
                    r = run_cell(est_steer, e, alpha, rep)
                    r.update(steer_est=est_steer, emotion=e, alpha=alpha, rep=rep)
                    ck.put(cid, r); rows.append(dict(r, _cid=cid))
                    print(f"[{est_steer:>6} {e:>9} a{alpha} r{rep}] "
                          f"present(logreg) {np.mean(r['present_logreg']):+.2f} "
                          f"other(logreg) {np.mean(r['other_logreg']):+.2f} | "
                          f"present(dom) {np.mean(r['present_dom']):+.2f} "
                          f"other(dom) {np.mean(r['other_dom']):+.2f} | "
                          f"deg {r['degenerate']:.2f}  [{time.time()-t0:.0f}s]", flush=True)

    # ---------------- the test ----------------------------------------------------------
    def gather(est_steer, e, key):
        return {d: np.concatenate([np.array(r[key]) for r in rows
                                   if r["steer_est"] == est_steer and r["emotion"] == e
                                   and r["alpha"] == d] or [np.array([])]) for d in DOSES}

    def scen(est_steer, e):
        v = [r for r in rows if r["steer_est"] == est_steer and r["emotion"] == e
             and r["alpha"] == DOSES[0]]
        return np.concatenate([np.array(x["scenario"]) for x in v]) if v else None

    results = {}
    for est_steer, est_read in CONFIGS:
        key = f"{est_steer}->{est_read}"
        per_emo, n_pres_gt_other, n_sig = {}, 0, 0
        for e in EMOS:
            P = gather(est_steer, e, f"present_{est_read}")
            O = gather(est_steer, e, f"other_{est_read}")
            if any(len(v) == 0 for v in P.values()):
                continue
            blk = scen(est_steer, e)
            contrast = C.paired_slope_contrast(DOSES, P, O)
            per_emo[e] = {
                "present": C.slope_ci(DOSES, P, block=blk),
                "other": C.slope_ci(DOSES, O, block=blk),
                "present_minus_other": contrast,
                "present_mean_by_dose": {str(d): float(P[d].mean()) for d in DOSES},
            }
            n_pres_gt_other += int(contrast["diff"] > 0)
            n_sig += int(contrast["sig"] and contrast["diff"] > 0)
        results[key] = {
            "steering_estimator": est_steer, "readout_estimator": est_read,
            "present_gt_other_count": n_pres_gt_other,
            "present_gt_other_significant": n_sig,
            "n_emotions": len(per_emo), "by_emotion": per_emo,
        }
        print(f"\n[{key}] present>other in {n_pres_gt_other}/{len(per_emo)} emotions, "
              f"significant in {n_sig}", flush=True)

    out = {"model": a.model, "tag": a.tag, "focus": focus, "rms": rms, "probe_n": N,
           "dir_half_n": len(DIR), "read_half_n": len(READ), "doses": list(DOSES),
           "configs": [list(c) for c in CONFIGS], "direction_stability": stability,
           "results": results, "rows": rows,
           "exhibit": {k: {"present_gt_other": v["present_gt_other_count"],
                           "significant": v["present_gt_other_significant"],
                           "of": v["n_emotions"]} for k, v in results.items()}}
    C.write_result(os.path.join(a.outdir, f"a3_dissociation_{a.tag}.json"), out, prov)
    print("\n[A3 exhibit]", json.dumps(out["exhibit"], indent=2), flush=True)
    print("A3_DONE", flush=True)


if __name__ == "__main__":
    main()

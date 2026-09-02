"""B1 — the E4 rerun (RESEARCH_PLAN rev 3, §1.3). The highest-priority experiment.

WHAT WENT WRONG THE FIRST TIME. The published E4 reported generation-time affect ablation
as "indistinguishable from a random-direction control" and treated that as a finding. It
is more likely instrument failure: the direction it ablated along had a split-half cosine
of 0.394, so ablating along it IS approximately ablating along a random direction with
respect to any independent fit of the same direction. An ablation that is ~60% noise
should look random. That is arithmetic on a measured number, not a hypothesis.

WHAT THIS RUN CHANGES, point by point against §1.3's rerun requirements:

  1. difference-of-means directions          -- `--estimator dom`, and the split-half
                                                stability of the directions actually used
                                                is measured and gated BEFORE the sweep.
  2. a generation-side manipulation check    -- `readout()` asks the model, in its own
                                                output tokens, which emotion A is feeling,
                                                while the ablation is live on A's span.
                                                Suppression in the residual stream is not
                                                suppression in the output; only the latter
                                                licenses a claim about what B produced.
  3. norm-matched random AND orthogonal      -- arms `rand` and `orth`, both unit-norm,
     controls                                  routed through the identical code path;
                                                the projection norm actually removed is
                                                recorded per arm rather than assumed equal.
  4. the dose grid declared before running   -- DOSES below, all of it reported.
  5. degeneracy scored at every dose         -- distinct-n, refusal, persona-break and
                                                perplexity at every (emotion, dose, arm).

Non-circularity: the probe that MEASURES B is fitted on a disjoint half of the dialogue
pool from the probe that supplies the steering and ablation directions. The published run
used one probe for both, so the readout was partly a function of its own direction.

Also carries B3 (cross-emotion specificity) as an arm rather than a separate run: `cross`
ablates a DIFFERENT emotion's direction, so specificity is measured in the same sweep as
the effect it qualifies (§6: standing, not optional).

Outcomes, fixed in advance (§1.3):
  * specific blocking under a stable direction  -> the channel claim has an experiment
  * still random under a stable direction with
    a passing manipulation check                -> lexical affect ablation does not block
                                                   contagion; the original claim, now
                                                   actually evidenced
  * manipulation check fails                    -> E4 is unreportable, and Paper B loses
                                                   its filtering headline

Usage: python b1_e4rerun.py --model Qwen/Qwen3.6-27B --tag qwen36-27b --reps 3
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import acl_core as C

# ---- declared grid (§6 hyperparameter grid declaration) --------------------------------
DOSES = (0.0, 0.5, 1.0)          # alpha <= 1.0: the model-breaking alpha=2 regime is out,
                                 # per coupling_e2_ci.py's pre-data declaration
ARMS = ("none", "emo", "rand", "orth", "cross")
EMOS = ("desperate", "afraid", "happy", "calm", "sad", "angry")
STABILITY_GATE = 0.80            # a direction below this does not get interpreted
PROBE_K = 60                     # dialogues per 6x6 cell for the probe pool


def mask_for(h, texts, subs, enc, offs):
    """Boolean [B, T] mask selecting the tokens of `subs[b]` inside `texts[b]`.

    Lives here rather than in acl_core so that acl_core stays byte-identical to the copy
    the A2 run imported -- a provenance stamp that reports a hash the run did not actually
    execute is exactly the failure mode §0.3 exists to catch.
    """
    M = torch.zeros_like(enc["attention_mask"], dtype=torch.bool)
    hits = 0
    for b, (t, sub) in enumerate(zip(texts, subs)):
        pos = t.find(sub)
        span = (pos, pos + len(sub)) if pos >= 0 else (-1, -1)
        hits += pos >= 0
        M[b] = torch.tensor(C.span_mask(offs[b].tolist(), span,
                                        enc["attention_mask"][b].tolist()),
                            device=enc["attention_mask"].device)
    return M, hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--estimator", default="dom")
    ap.add_argument("--outdir", default="/marimo/out")
    ap.add_argument("--workdir", default="/marimo/work")
    ap.add_argument("--gen-bs", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True); os.makedirs(a.workdir, exist_ok=True)

    prov = C.Provenance(
        script="src/rev3/b1_e4rerun.py",
        config={"doses": DOSES, "arms": ARMS, "emotions": EMOS, "reps": a.reps,
                "estimator": a.estimator, "probe_k": PROBE_K,
                "stability_gate": STABILITY_GATE, "max_new_A": 110, "max_new_B": 110,
                "temp": 0.9, "top_p": 0.95, "n_scenarios": len(C.SCENARIOS)},
        model_id=a.model, model_revision="", seeds={"pool": a.seed, "run": a.seed},
        control_pointers={
            "norm_matched_random": "b1_e4rerun.py::build_dirs -> acl_core.random_dir_like "
                                   "(unit norm, same Ablate code path as the emo arm); the "
                                   "projection norm actually removed is reported per arm as "
                                   "removed_norm, not assumed equal",
            "orthogonal_control": "acl_core.orthogonal_dir -- random direction with the "
                                  "emotion component projected out, unit norm",
            "cross_emotion_specificity": "ARM 'cross' ablates a DIFFERENT emotion's "
                                         "direction (see build_dirs: cross_of)",
            "generation_side_manip_check": "b1_e4rerun.py::readout -> "
                                           "acl_core.forced_choice_readout, run WITH the "
                                           "ablation hooks live on A's token span",
            "non_circular_measurement": "probe fitted on READ half; steering/ablation "
                                        "directions fitted on the disjoint DIR half",
            "frozen_token_audit": "B's reply is regenerated under the intervention; A's "
                                  "message predates it. See out['frozen_token_audit'].",
        })

    h = C.load(a.model)
    prov.model_revision = h.revision
    focus = h.focus(); steer_layer = focus - 1
    lo = round(0.2 * h.n_layers)
    abl_dec = [d for d in range(h.n_layers) if lo <= d + 1 <= focus - 1]
    abl_hs = [d + 1 for d in abl_dec]
    hs_needed = sorted(set(abl_hs) | {focus})
    print(f"[B1] focus {focus} steer_layer {steer_layer} ablating {len(abl_dec)} blocks "
          f"(hs {abl_hs[0]}..{abl_hs[-1]})", flush=True)

    # ---------------- probe pool, split into DIR and READ halves ------------------------
    items = C.generate_pool(h, PROBE_K, os.path.join(a.workdir, f"probe_{a.tag}.jsonl"),
                            seed=a.seed, bs=a.gen_bs)
    feats, yp, yo = C.pool_features(h, items, hs_needed,
                                    cache=os.path.join(a.workdir, f"probefeat_{a.tag}.npz"))
    N = len(items)
    rs = np.random.default_rng(a.seed).permutation(N)
    DIR, READ = rs[:N // 2], rs[N // 2:]
    print(f"[B1] probe pool n={N}: DIR half {len(DIR)}, READ half {len(READ)}", flush=True)

    dir_dec = {L: C.fit_direction(feats[L][DIR], yp[DIR], a.estimator, seed=a.seed,
                                  pair_on=yo[DIR]) for L in hs_needed}
    read_dec = {L: C.fit_direction(feats[L][READ], yp[READ], a.estimator, seed=a.seed,
                                   pair_on=yo[READ]) for L in hs_needed}
    read_dec_other = C.fit_direction(feats[focus][READ], yo[READ], a.estimator, seed=a.seed)

    # ---- GATE: is the direction we are about to ablate along reproducible at all? ------
    stab = C.split_half(feats[focus], yp, a.estimator, n_per_half=min(N // 2, 600),
                        seeds=range(10), pair_on=yo)
    stab_logreg = C.split_half(feats[focus], yp, "logreg", n_per_half=min(N // 2, 600),
                               seeds=range(10))
    print(f"[B1 GATE] split-half({a.estimator}) {stab['mean']:.3f} CI {stab['ci']}  |  "
          f"logreg {stab_logreg['mean']:.3f}  | gate {STABILITY_GATE}", flush=True)
    gate_pass = bool(stab["mean"] >= STABILITY_GATE)
    if not gate_pass:
        print("[B1 GATE] *** DIRECTION BELOW GATE — the sweep still runs, but the "
              "ablation arms are not interpretable as emotion-specific ***", flush=True)

    # ---------------- directions ---------------------------------------------------------
    rng = np.random.default_rng(a.seed)
    cross_of = {e: EMOS[(EMOS.index(e) + 3) % len(EMOS)] for e in EMOS}   # fixed, declared

    def tt(v):
        return torch.tensor(v, dtype=h.model.dtype, device=h.model.device)

    emo_dirs, rand_dirs, orth_dirs, steer_dir = {}, {}, {}, {}
    for e in EMOS:
        ei = C.EMOTIONS.index(e)
        emo_dirs[e] = {L: tt(C.raw_direction(dir_dec[L], ei)) for L in abl_hs}
        orth_dirs[e] = {L: tt(C.orthogonal_dir(C.raw_direction(dir_dec[L], ei), rng))
                        for L in abl_hs}
        steer_dir[e] = tt(C.raw_direction(dir_dec[focus], ei))
    rand_dirs = {L: tt(C.random_dir_like(C.raw_direction(dir_dec[L], 0), rng)) for L in abl_hs}
    cross_dirs = {e: emo_dirs[cross_of[e]] for e in EMOS}

    steer = C.Steer(h, steer_layer)
    ab = C.Ablate(h, abl_dec)
    rms = float(np.linalg.norm(feats[focus], axis=1).mean())
    print(f"[B1] rms {rms:.1f}; cross map {cross_of}", flush=True)

    # ---------------- the sweep ----------------------------------------------------------
    S = C.SCENARIOS
    rlog = C.ResponseLog(os.path.join(a.workdir, f"responses_{a.tag}.jsonl"))
    ck = C.Checkpoint(os.path.join(a.workdir, f"b1_cells_{a.tag}.json"),
                      {"doses": DOSES, "arms": ARMS, "emos": EMOS, "reps": a.reps,
                       "est": a.estimator, "n": N, "focus": focus, "probe_k": PROBE_K})

    def gen_A(e, alpha, rep):
        names = [C.NAMES[(i + rep) % len(C.NAMES)] for i in range(len(S))]
        prompts, ctx = [], []
        for (setting, bopen), (A, B) in zip(S, names):
            s = setting.format(A=A, B=B); bo = bopen.format(A=A, B=B)
            prompts.append(f"{s}\n{bo}\nContinue the conversation. Write only {A}'s next "
                           f"reply as one short paragraph, starting with '{A}:'.")
            ctx.append({"A": A, "B": B, "setting": s, "bopen": bo, "scenario": len(ctx)})
        vec = None if alpha == 0 else (alpha * rms) * steer_dir[e]
        ab.clear()
        outs = C.gen(h, prompts, max_new=110, bs=a.gen_bs, seed=a.seed + rep, first_line=True,
                     hooks_on=(lambda: steer.on(vec)) if vec is not None else None,
                     hooks_off=steer.off, log=rlog, tag=f"A_{e}_a{alpha}_r{rep}")
        for c, ar in zip(ctx, outs):
            A = c["A"]
            ar = ar if ar.startswith(f"{A}:") else f"{A}: {ar}"
            c["ar"] = ar
            c["conv"] = f"{c['setting']}\n{c['bopen']}\n{ar}"
            c["bprompt"] = (f"{c['conv']}\nWrite only {c['B']}'s next reply as one short "
                            f"paragraph, starting with '{c['B']}:'.")
        return ctx

    def dirs_for(arm, e):
        return {"none": None, "emo": emo_dirs[e], "rand": rand_dirs,
                "orth": orth_dirs[e], "cross": cross_dirs[e]}[arm]

    def gen_B(ctx, arm, e, bs=30):
        """B regenerates while A's token span is ablated during B's PREFILL."""
        dirs = dirs_for(arm, e)
        h.tok.padding_side = "left"
        out, removed, hitrate = [], [], []
        for i in range(0, len(ctx), bs):
            batch = ctx[i:i + bs]
            texts = [C.chat(h, c["bprompt"]) for c in batch]
            enc = h.tok(texts, return_tensors="pt", padding=True, return_offsets_mapping=True)
            offs = enc.pop("offset_mapping")
            enc = {k: v.to(h.model.device) for k, v in enc.items()}
            if arm != "none":
                M, hits = mask_for(h, texts, [c["ar"] for c in batch], enc, offs)
                hitrate.append(hits / len(batch))
                ab.set(arm, M, dirs)
            try:
                with torch.no_grad():
                    g = h.model.generate(**enc, max_new_tokens=110, do_sample=True,
                                         temperature=0.9, top_p=0.95,
                                         pad_token_id=h.tok.pad_token_id)
            finally:
                # only the intervened arms have a projection to report; reading this on the
                # 'none' arm would pick up the previous arm's stale accumulator
                if arm != "none" and ab.removed_norm:
                    removed.append(float(np.mean(ab.removed_norm)))
                ab.clear()
            out += [o.strip().split("\n")[0].strip() for o in
                    h.tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)]
        rlog.write([c["conv"] for c in ctx], out, f"B_{e}_{arm}")
        return out, (float(np.mean(removed)) if removed else 0.0), \
               (float(np.mean(hitrate)) if hitrate else 1.0)

    def readout(ctx, arm, e, bs=30):
        """GENERATION-SIDE manipulation check: with the ablation live on A's span, does the
        model still name A's emotion in its OWN output tokens? Returns [N, 6] probs."""
        dirs = dirs_for(arm, e)
        h.tok.padding_side = "left"
        ids = []
        for em in C.EMOTIONS:
            cand = [h.tok.encode(x, add_special_tokens=False)
                    for x in (em, " " + em, em.capitalize())]
            ids.append(sorted({c[0] for c in cand if c}))
        res = []
        for i in range(0, len(ctx), bs):
            batch = ctx[i:i + bs]
            texts = [C.chat(h, f"{c['conv']}\n\n{C.READOUT_Q.format(A=c['A'])}") for c in batch]
            enc = h.tok(texts, return_tensors="pt", padding=True, return_offsets_mapping=True)
            offs = enc.pop("offset_mapping")
            enc = {k: v.to(h.model.device) for k, v in enc.items()}
            if arm != "none":
                M, _ = mask_for(h, texts, [c["ar"] for c in batch], enc, offs)
                ab.set(arm, M, dirs)
            try:
                with torch.no_grad():
                    lg = h.model(**enc).logits[:, -1, :].float()
            finally:
                ab.clear()
            cols = torch.stack([lg[:, torch.tensor(g, device=lg.device)].logsumexp(-1)
                                for g in ids], -1)
            res.append(cols.softmax(-1).cpu().numpy())
        return np.concatenate(res, 0)

    def score_B(ctx, breps, bs=8):
        """B's own present-e and other-e on its reply, under the HELD-OUT read probe."""
        texts, spans = [], []
        for c, br in zip(ctx, breps):
            B = c["B"]
            br = br if br.startswith(f"{B}:") else f"{B}: {br}"
            full = f"{c['conv']}\n{br}"
            texts.append(full)
            spans.append((full.rfind(f"{B}:") + len(f"{B}:"), len(full)))
        ab.clear()
        f = C.pool_spans(h, texts, spans, [focus], bs=bs)[focus]
        Xs = (f - read_dec[focus]["mu"]) / read_dec[focus]["sd"]
        return (Xs @ read_dec[focus]["C"].T), (Xs @ read_dec_other["C"].T)

    rows = []
    t_start = time.time()
    for e in EMOS:
        ei = C.EMOTIONS.index(e)
        for alpha in DOSES:
            for rep in range(a.reps):
                cid = f"{e}/{alpha}/{rep}"
                if ck.has(cid):
                    rows.extend(ck.get(cid)); print(f"[skip] {cid}", flush=True); continue
                ctx = gen_A(e, alpha, rep)
                # MC-steer: did steering actually put the emotion into A's TEXT?
                a_read = readout(ctx, "none", e)
                cell = []
                for arm in ARMS:
                    breps, removed, hit = gen_B(ctx, arm, e)
                    pres, oth = score_B(ctx, breps)
                    # MC-ablate: is the affect still readable from the ablated context?
                    ab_read = readout(ctx, arm, e)
                    deg = [C.degeneracy(b) for b in breps]
                    ppl = C.perplexity(h, breps)
                    cell.append({
                        "emotion": e, "alpha": alpha, "rep": rep, "arm": arm,
                        "scenario": [c["scenario"] for c in ctx],
                        "B_present_e": pres[:, ei].tolist(),
                        "B_other_e": oth[:, ei].tolist(),
                        "A_readout_e": a_read[:, ei].tolist(),
                        "ctx_readout_e": ab_read[:, ei].tolist(),
                        "ctx_readout_full": ab_read.mean(0).tolist(),
                        "removed_norm": removed, "mask_hit_rate": hit,
                        "degenerate_frac": float(np.mean([C.is_degenerate(d) for d in deg])),
                        "refusal_frac": float(np.mean([d["refusal"] for d in deg])),
                        "distinct2": float(np.mean([d["distinct2"] for d in deg])),
                        "n_words": float(np.mean([d["n_words"] for d in deg])),
                        "perplexity": float(np.median(ppl)),
                    })
                    r = cell[-1]
                    print(f"[{e:>9} a{alpha} r{rep} {arm:>5}] present {np.mean(r['B_present_e']):+.2f} "
                          f"ctx_readout(e) {np.mean(r['ctx_readout_e']):.3f} "
                          f"removed {removed:.2f} deg {r['degenerate_frac']:.2f} "
                          f"ppl {r['perplexity']:.1f}", flush=True)
                ck.put(cid, cell)
                rows.extend(cell)
                print(f"[B1] {len(rows)} cells, {time.time()-t_start:.0f}s elapsed", flush=True)

    # ---------------- analysis -----------------------------------------------------------
    def cellsel(e, arm, alpha, key):
        v = [r for r in rows if r["emotion"] == e and r["arm"] == arm and r["alpha"] == alpha]
        return np.concatenate([np.array(x[key]) for x in v]) if v else np.array([])

    def blocks(e, arm, alpha):
        v = [r for r in rows if r["emotion"] == e and r["arm"] == arm and r["alpha"] == alpha]
        return np.concatenate([np.array(x["scenario"]) for x in v]) if v else np.array([])

    summary = {}
    for e in EMOS:
        s = {}
        for arm in ARMS:
            ps = {d: cellsel(e, arm, d, "B_present_e") for d in DOSES}
            os_ = {d: cellsel(e, arm, d, "B_other_e") for d in DOSES}
            if any(len(v) == 0 for v in ps.values()):
                continue
            s[arm] = {
                "present_slope": C.slope_ci(DOSES, ps, block=blocks(e, arm, DOSES[0])),
                "other_slope": C.slope_ci(DOSES, os_, block=blocks(e, arm, DOSES[0])),
                "present_minus_other": C.paired_slope_contrast(DOSES, ps, os_),
                "mean_by_dose": {str(d): float(ps[d].mean()) for d in DOSES},
                "degenerate_by_dose": {
                    str(d): float(np.mean([r["degenerate_frac"] for r in rows
                                           if r["emotion"] == e and r["arm"] == arm
                                           and r["alpha"] == d])) for d in DOSES},
            }
        # the contrasts §1.3 is actually asking about
        if "emo" in s and "none" in s:
            s["emo_vs_none"] = C.paired_slope_contrast(
                DOSES, {d: cellsel(e, "emo", d, "B_present_e") for d in DOSES},
                {d: cellsel(e, "none", d, "B_present_e") for d in DOSES})
        if "emo" in s and "rand" in s:
            s["emo_vs_rand"] = C.paired_slope_contrast(
                DOSES, {d: cellsel(e, "emo", d, "B_present_e") for d in DOSES},
                {d: cellsel(e, "rand", d, "B_present_e") for d in DOSES})
        if "emo" in s and "orth" in s:
            s["emo_vs_orth"] = C.paired_slope_contrast(
                DOSES, {d: cellsel(e, "emo", d, "B_present_e") for d in DOSES},
                {d: cellsel(e, "orth", d, "B_present_e") for d in DOSES})
        if "emo" in s and "cross" in s:
            s["emo_vs_cross"] = C.paired_slope_contrast(
                DOSES, {d: cellsel(e, "emo", d, "B_present_e") for d in DOSES},
                {d: cellsel(e, "cross", d, "B_present_e") for d in DOSES})
        # MC-steer: does A's readout rise with dose? (steering reached the OUTPUT)
        s["mc_steer"] = C.slope_ci(DOSES,
                                   {d: cellsel(e, "none", d, "A_readout_e") for d in DOSES})
        # MC-ablate at the top dose: did ablation remove readable affect vs the controls?
        top = DOSES[-1]
        mc = {}
        for arm in ARMS:
            v = cellsel(e, arm, top, "ctx_readout_e")
            mc[arm] = {"mean": float(v.mean()) if len(v) else None,
                       "ci": C.ci_of(v) if len(v) else None, "n": int(len(v))}
        drop_emo = mc["none"]["mean"] - mc["emo"]["mean"]
        drop_rand = mc["none"]["mean"] - mc["rand"]["mean"]
        mc["drop_emo"] = drop_emo
        mc["drop_rand"] = drop_rand
        mc["specific_drop"] = drop_emo - drop_rand
        mc["passes"] = bool(drop_emo > 0 and drop_emo > drop_rand)
        s["mc_ablate"] = mc
        summary[e] = s
        print(f"\n[{e}] emo_vs_rand diff {s.get('emo_vs_rand',{}).get('diff')} "
              f"CI {s.get('emo_vs_rand',{}).get('ci')} | MC-ablate passes {mc['passes']} "
              f"(emo drop {drop_emo:+.3f} vs rand {drop_rand:+.3f})", flush=True)

    n_mc_pass = sum(1 for e in EMOS if summary[e]["mc_ablate"]["passes"])
    verdict = ("manipulation_check_failed" if n_mc_pass < len(EMOS) / 2 else
               "blocking" if sum(1 for e in EMOS
                                 if summary[e].get("emo_vs_rand", {}).get("sig")) >= 4
               else "not_blocking")

    out = {
        "model": a.model, "tag": a.tag, "focus": focus, "steer_layer": steer_layer,
        "n_abl_blocks": len(abl_dec), "abl_hs_range": [abl_hs[0], abl_hs[-1]],
        "probe_n": N, "dir_half_n": len(DIR), "read_half_n": len(READ),
        "estimator": a.estimator, "rms": rms, "doses": list(DOSES), "arms": list(ARMS),
        "cross_map": cross_of,
        "direction_stability": {"used": stab, "logreg_for_comparison": stab_logreg,
                                "gate": STABILITY_GATE, "gate_pass": gate_pass},
        "frozen_token_audit": C.frozen_token_audit({
            "generated": ["B's reply (regenerated under the ablation, arms emo/rand/orth/cross)",
                          "the readout answer token (produced with the ablation live)"],
            "frozen": ["A's message (generated BEFORE any ablation, under steering only)",
                       "the scenario setting and B's opening line (fixed stimuli)"]}),
        "verdict": verdict, "n_mc_pass": n_mc_pass,
        "summary": summary, "rows": rows,
    }
    C.write_result(os.path.join(a.outdir, f"b1_e4rerun_{a.tag}.json"), out, prov)
    print(f"\n[B1] VERDICT: {verdict}  (manipulation check passed for {n_mc_pass}/{len(EMOS)})",
          flush=True)
    print("B1_DONE", flush=True)


if __name__ == "__main__":
    main()

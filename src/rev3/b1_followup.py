#!/usr/bin/env python3
"""B1-FOLLOWUP — the second pass at the E4 rerun (RESEARCH_PLAN rev 3, §1.3).

WHAT THE COMPLETED B1 RUN LEFT UNRESOLVED
(results/rev3/b1_e4rerun_qwen36-27b.json, verdict "not_blocking", n_mc_pass 6/6 — an
independent review of that file found six defects, each of which this driver answers.)

  (a) NOT REPRODUCIBLE. `gen_B` and `readout` called `h.model.generate` / the forward pass
      directly with no `torch.manual_seed`, so B's replies cannot be regenerated even from
      the same pool file, the same directions and the same prompts.
  (b) THE GATE MEASURED THE WRONG LAYERS. Split-half stability was computed only at the
      focus layer (hidden_states 43); the ablation projects at hidden_states 13..42, so
      none of the 30 directions actually removed had a stability figure attached to it.
  (c) THE RANDOM CONTROL HAD n=1. One draw per layer was shared across all six emotions
      and all reps, so "indistinguishable from random" rested on a single random vector.
  (d) NO CEILING AND NO TOKEN-LEVEL ARM. Nothing said what the readout looks like when the
      direction is removed at EVERY position rather than only over A's span, and a residual
      ablation leaves A's *words* untouched, so B can simply recompute the affect from the
      text — an alternative that arm structure could not exclude.
  (e) THE DECISION RULE IGNORED SIGN AND SCALE. It counted significant emo-vs-rand
      contrasts without asking whether they were negative (blocking) or positive
      (anti-blocking), reported no MDE although `acl_core.mde` exists, and never expressed
      a contrast as a fraction of the none-arm slope it was supposed to be cancelling.
  (f) THE PROVENANCE STAMP WAS WRONG. `_provenance.code_sha` was the empty string, and
      `control_pointers` named `acl_core.forced_choice_readout` and `build_dirs` — the
      driver called neither (it had its own inlined `readout`, and no `build_dirs`).

WHAT THIS DRIVER CHANGES, point for point

  (a) -> Every sampling call is preceded by `torch.manual_seed(gen_seed(...))`, with the
         seed a pure function of (--seed, emotion, alpha, rep, arm, part):
             gen_seed = crc32(f"{seed}|{emotion}|{alpha:g}|{rep}|{arm}|{part}") & 0x7fffffff
         and `+ batch_start_index` per batch, matching `acl_core.gen`'s convention. For
         part "B" the `arm` field is left EMPTY, so B's reply in all five arms is sampled
         from ONE stream (common random numbers): shared sampling noise then cancels inside
         each paired arm contrast rather than widening its CI. "readout" and "rewrite" keep
         `arm`. The integer actually used is written into every arm-row as `gen_seed`.
  (b) -> `direction_stability_by_layer` holds a full `split_half` dict for EVERY ablated
         hidden_states index and for the focus layer; min/median/max across the ablated
         layers are printed, and any layer under the gate is listed in `layers_below_gate`.
         The focus layer remains the (recorded, non-fatal) abort criterion, unchanged, so
         the gate number stays comparable with the completed run.
  (c) -> A FRESH norm-matched random direction per (emotion, rep), drawn from a generator
         seeded by (--seed, canonical emotion index, rep); the seed is recorded per row as
         `rand_dir_seed`.
  (d) -> Two new arms. `ceiling` projects the SAME emotion direction out of every prefill
         position (mask = the attention mask), which upper-bounds what a residual ablation
         over A's span could ever do. `text` deletes the affect at the token level instead:
         A's steered line is rewritten by the UNSTEERED model into neutral wording and B
         then replies with no hooks at all, so its readout drop measures how much readable
         affect lives in the words themselves.
         `removed_norm` is a mean over MASKED positions, and the ceiling arm masks every
         attended token while emo/rand mask only A's span — so the two are not comparable
         on their own. Every row therefore also carries `mask_n_positions` (mean masked
         positions per sample); read the pair, never removed_norm alone. `mask_hit_rate` is
         None, not 1.0, for the arms that install no hook at all (none, text).
  (e) -> The verdict counts SIGNED contrasts, only among emotions whose none-arm slope
         excludes zero; `mde` is reported per emotion; and every contrast is also given as
         `blocked_fraction`, the share of the none-arm slope it removes, with the range
         implied by the contrast CI.
  (f) -> `Provenance(code_sha=C.code_hash(this file, acl_core.py))`, and every
         `control_pointers` entry names a function and line range in THIS file.

Held constant on purpose, so this run's directions are IDENTICAL to the completed B1 run
whenever the same pool file is present: the probe pool path `{workdir}/probe_{tag}.jsonl`
and PROBE_K=60, the feature cache `{workdir}/probefeat_{tag}.npz`, the DIR/READ
permutation (`np.random.default_rng(seed).permutation(N)`), the default estimator `dom`,
the focus/steer/ablation layer construction, DOSES and EMOS. A's messages are still
generated through `acl_core.gen` with `seed=--seed + rep`, so A's half of the stimulus is
byte-comparable with the completed run as well.

PRE-DECLARED DECISION RULE (fixed before this run; not to be amended after the numbers)
  Testable emotion  : the none-arm present-e dose slope excludes zero.
  n_block           : testable emotions with a SIGNIFICANT and NEGATIVE emo_vs_rand slope
                      contrast (the emo arm flattens the dose response relative to a
                      norm-matched random direction).
  n_antiblock       : testable emotions with a significant POSITIVE emo_vs_rand contrast.
  n_untestable      : emotions whose none-arm slope does not exclude zero.
  VERDICT           : "manipulation_check_failed" if fewer than half of the emotions that
                      were MEASURED pass MC-ablate for the emo arm (drop_emo > 0 and
                      drop_emo > drop_rand), and always if none were measured — the
                      denominator is what was measured, not what was planned, so a grid
                      that half died cannot buy a pass with its missing cells;
                      else "blocking"      if n_block >= 4;
                      else "mixed"         if n_block >= 1 or n_antiblock >= 1;
                      else "not_blocking".
  A "not_blocking" verdict is only reportable next to the per-emotion `mde`: a null is a
  null about an effect size, not about an effect.

Usage: python b1_followup.py --model Qwen/Qwen3.6-27B --tag qwen36-27b --reps 3
       python b1_followup.py --selftest        # analysis only, no model, no GPU
"""
import argparse, ast, json, os, sys, time, zlib

for _c in (os.path.dirname(os.path.abspath(__file__)),
           os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib")):
    if os.path.exists(os.path.join(_c, "acl_core.py")):
        sys.path.insert(0, _c)
        break
import numpy as np
try:                                    # absent on the CPU box where --selftest runs
    import torch
except Exception:                       # pragma: no cover
    torch = None
import acl_core as C

# ---- declared grid (§6 hyperparameter grid declaration) --------------------------------
# Identical to b1_e4rerun.py except for ARMS: same doses, same emotions, same probe size,
# same gate, so the directions are the same objects and only the arm structure differs.
DOSES = (0.0, 0.5, 1.0)
ARMS = ("none", "emo", "rand", "ceiling", "text")
EMOS = ("desperate", "afraid", "happy", "calm", "sad", "angry")
STABILITY_GATE = 0.80
PROBE_K = 60

# verbatim from src/representation/coupling_e2_ci.py (the paraphraser control, ~line 70);
# the token-level arm is that control moved inside the ablation sweep.
REWRITE_PROMPT = ('Rewrite this line in plain, emotionally neutral wording, '
                  'keeping only the literal information: "{ar}". '
                  "Start with '{A}:'. Output only the rewritten line.")


# ---- self-locating source pointers -----------------------------------------------------
# The completed run's provenance named a `build_dirs` that did not exist (defect (f)), and
# hand-written line numbers went stale the moment this file was edited. Both pointers below
# are therefore READ OUT OF THIS FILE at import time rather than typed.
_SRC_PATH = os.path.abspath(__file__)
try:
    _SRC = open(_SRC_PATH, encoding="utf-8").read()
except Exception:                                        # pragma: no cover
    _SRC = ""


def _fn_spans() -> dict:
    """{name: (first_line, last_line)} for every def in this file, nested ones included."""
    out = {}
    try:
        tree = ast.parse(_SRC)
    except Exception:                                    # pragma: no cover
        return out
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out[node.name] = (node.lineno, node.end_lineno)
    return out


_FN_SPANS = _fn_spans()


def fn_lines(name: str) -> str:
    """'L624-L663' for a function of this file, straight from the AST — never typed by
    hand, so a control pointer cannot drift away from the code it names."""
    s = _FN_SPANS.get(name)
    return f"L{s[0]}-L{s[1]}" if s else f"L?-L? ({name} NOT FOUND — pointer is stale)"


def block_lines(name: str) -> str:
    """Span of the code between the '# <<NAME_START>>' / '# <<NAME_END>>' sentinel comments,
    for the two stretches of `main` that are controls but are not their own function."""
    tags = {"# <<" + f"{name}_{w}" + ">>": w for w in ("START", "END")}
    hit = {}
    for i, ln in enumerate(_SRC.splitlines(), 1):
        if ln.strip() in tags:
            hit[tags[ln.strip()]] = i
    if "START" in hit and "END" in hit:
        return f"L{hit['START'] + 1}-L{hit['END'] - 1}"
    return f"L?-L? ({name} sentinels NOT FOUND — pointer is stale)"


def gen_seed(seed: int, emotion: str, alpha: float, rep: int, arm: str, part: str = "") -> int:
    """Deterministic per-generation torch seed (defect (a)).

    crc32 of "seed|emotion|alpha|rep|arm|part", masked to 31 bits. `part` separates the
    several generate() calls that share an arm-row ("B" replies, "rewrite", "readout"), so
    two different calls in the same cell never sample from the same stream. Callers add the
    batch start index, exactly as `acl_core.gen` does, so batching does not change the draw.

    ONE EXCEPTION, on purpose: for part "B" the `arm` field is left EMPTY, so B's reply is
    sampled from the same stream in all five arms — common random numbers. Every arm
    contrast here is paired, and sampling noise that is shared between the two arms cancels
    in the difference instead of widening its CI. The readout and the rewrite keep `arm` in
    the key, because those calls differ between arms by construction.
    """
    key = f"{seed}|{emotion}|{alpha:g}|{rep}|{'' if part == 'B' else arm}|{part}"
    return int(zlib.crc32(key.encode()) & 0x7FFFFFFF)


def rand_dir_seed(seed: int, emotion: str, rep: int) -> int:
    """Seed for the fresh random control of one (emotion, rep) cell (defect (c)).

    Derived from (--seed, the CANONICAL emotion index in acl_core.EMOTIONS, rep) so it is
    stable under a resumed run and independent of the order EMOS happens to be written in.
    """
    ei = C.EMOTIONS.index(emotion)
    return int(zlib.crc32(f"rand|{seed}|{ei}|{rep}".encode()) & 0x7FFFFFFF)


def mask_for(h, texts, subs, enc, offs):
    """Boolean [B, T] mask selecting the tokens of `subs[b]` inside `texts[b]`.

    Lives here rather than in acl_core so that acl_core stays byte-identical to the copy
    every other rev-3 run imported.
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


# ---------------------------------------------------------------------------------------
# analysis — importable with no torch and no model, so it is testable (`--selftest`)
# ---------------------------------------------------------------------------------------

def blocked_fraction(contrast: dict | None, none_slope: dict | None) -> dict:
    """How much of the none-arm dose slope a contrast removes (defect (e)).

    -diff / slope(none): 1.0 means the arm cancelled the whole dose response, 0.0 means it
    did nothing, negative means it AMPLIFIED it. The range is the same transform applied to
    the two ends of the contrast CI (re-sorted, because dividing by a negative denominator
    flips them). `denominator_sig` is carried alongside because a fraction of a slope that
    does not itself exclude zero is a ratio of two noise estimates, not a result.
    """
    if not contrast or not none_slope:
        return {"point": None, "range": [None, None], "defined": False}
    den = float(none_slope.get("slope", 0.0))
    if not np.isfinite(den) or abs(den) < 1e-9:
        return {"point": None, "range": [None, None], "defined": False,
                "denominator": den, "denominator_sig": bool(none_slope.get("sig"))}
    lo, hi = sorted(-float(c) / den for c in contrast["ci"])
    return {"point": float(-float(contrast["diff"]) / den), "range": [lo, hi],
            "defined": True, "denominator": den,
            "denominator_sig": bool(none_slope.get("sig"))}


def analyze(rows, doses, emos, arms=ARMS, n_boot: int = 5000) -> dict:
    """Summarise the arm-rows. Returns {'per_emotion', 'counts', 'verdict', ...}.

    Per emotion and arm this is b1_analyze.py's summary (scenario-blocked present/other
    slope CIs, mean_by_dose, degenerate_by_dose), plus the contrasts this design is for
    (emo_vs_none, emo_vs_rand, ceiling_vs_none, text_vs_none, emo_vs_ceiling), mc_steer,
    mc_ablate over emo/rand/ceiling/text, and — new here — `mde` and `blocked_fraction`.
    The verdict is the pre-declared, SIGN-AWARE rule in the module docstring.
    """
    doses = [float(d) for d in doses]
    arms = tuple(arms)

    def sel(e, arm, d, key):
        v = [r for r in rows if r["emotion"] == e and r["arm"] == arm
             and float(r["alpha"]) == d]
        return np.concatenate([np.asarray(x[key], dtype=float) for x in v]) if v \
            else np.array([])

    def blocks(e, arm, d):
        v = [r for r in rows if r["emotion"] == e and r["arm"] == arm
             and float(r["alpha"]) == d]
        return np.concatenate([np.asarray(x["scenario"]) for x in v]) if v else None

    def present(e, arm):
        return {d: sel(e, arm, d, "B_present_e") for d in doses}

    per_emotion = {}
    for e in emos:
        s, have_all = {}, True
        for arm in arms:
            ps = present(e, arm)
            if any(len(v) == 0 for v in ps.values()):
                have_all = False
                continue
            os_ = {d: sel(e, arm, d, "B_other_e") for d in doses}
            s[arm] = {
                "present_slope": C.slope_ci(doses, ps, n_boot=n_boot,
                                            block=blocks(e, arm, doses[0])),
                "other_slope": C.slope_ci(doses, os_, n_boot=n_boot,
                                          block=blocks(e, arm, doses[0])),
                "mean_by_dose": {str(d): float(ps[d].mean()) for d in doses},
                "degenerate_by_dose": {
                    str(d): float(np.mean([r["degenerate_frac"] for r in rows
                                           if r["emotion"] == e and r["arm"] == arm
                                           and float(r["alpha"]) == d] or [np.nan]))
                    for d in doses},
            }
        for x, y in (("emo", "none"), ("emo", "rand"), ("ceiling", "none"),
                     ("text", "none"), ("emo", "ceiling")):
            if x in s and y in s:
                s[f"{x}_vs_{y}"] = C.paired_slope_contrast(doses, present(e, x),
                                                           present(e, y), n_boot=n_boot)
        if "none" in s:
            s["mc_steer"] = C.slope_ci(doses, {d: sel(e, "none", d, "A_readout_e")
                                               for d in doses}, n_boot=n_boot)

        # MC-ablate at the top dose: is the affect still readable OUT of the model?
        top = doses[-1]
        mc = {}
        for arm in arms:
            v = sel(e, arm, top, "ctx_readout_e")
            mc[arm] = {"mean": float(v.mean()) if len(v) else None,
                       "ci": C.ci_of(v) if len(v) else None, "n": int(len(v))}
        if mc.get("none", {}).get("mean") is not None:
            drops = {arm: mc["none"]["mean"] - mc[arm]["mean"]
                     for arm in ("emo", "rand", "ceiling", "text")
                     if mc.get(arm, {}).get("mean") is not None}
            mc["drop_by_arm"] = drops
            if "emo" in drops and "rand" in drops:
                mc["drop_emo"], mc["drop_rand"] = drops["emo"], drops["rand"]
                mc["specific_drop"] = drops["emo"] - drops["rand"]
                # the control CIs must not overlap, or "specific" is not established
                mc["separated"] = bool(mc["emo"]["ci"][1] < mc["rand"]["ci"][0])
                mc["passes"] = bool(drops["emo"] > 0 and drops["emo"] > drops["rand"])
            else:
                mc["passes"] = None
        else:
            mc["passes"] = None
        s["mc_ablate"] = mc

        # MDE: the smallest top-dose mean difference this n could have resolved (defect (e))
        top_none = sel(e, "none", top, "B_present_e")
        sd = float(top_none.std(ddof=1)) if len(top_none) > 1 else float("nan")
        s["mde"] = {"sd": sd, "n": int(len(top_none)),
                    "value": float(C.mde(sd, len(top_none))) if len(top_none) > 1
                    else float("nan"),
                    "units": "probe-score units, two-arm difference in mean B_present_e "
                             f"at dose {top} (alpha=0.05, power=0.8)"}
        ns = s.get("none", {}).get("present_slope")
        s["blocked_fraction"] = {
            "emo_vs_rand": blocked_fraction(s.get("emo_vs_rand"), ns),
            "emo_vs_ceiling": blocked_fraction(s.get("emo_vs_ceiling"), ns),
        }
        s["_complete"] = have_all
        per_emotion[e] = s

    # ---- the pre-declared, sign-aware verdict (defect (e)) -----------------------------
    testable, block, antiblock = [], [], []
    for e in emos:
        ns = per_emotion[e].get("none", {}).get("present_slope")
        if not ns or not ns.get("sig"):
            continue
        testable.append(e)
        c = per_emotion[e].get("emo_vs_rand")
        if c and c.get("sig"):
            (block if c["diff"] < 0 else antiblock).append(e)
    measured = [e for e in emos
                if per_emotion[e].get("mc_ablate", {}).get("passes") is not None]
    n_mc_pass = sum(1 for e in measured if per_emotion[e]["mc_ablate"]["passes"])
    counts = {"n_block": len(block), "n_antiblock": len(antiblock),
              "n_untestable": len(emos) - len(testable), "n_testable": len(testable),
              "n_mc_pass": n_mc_pass, "n_mc_measured": len(measured),
              "blocking_emotions": block, "antiblocking_emotions": antiblock,
              "testable_emotions": testable}
    # the denominator is the emotions actually MEASURED, not the emotions planned: a grid
    # that half died must not clear the manipulation check with its missing cells
    verdict = ("manipulation_check_failed"
               if (not measured or n_mc_pass < len(measured) / 2) else
               "blocking" if counts["n_block"] >= 4 else
               "mixed" if (counts["n_block"] >= 1 or counts["n_antiblock"] >= 1) else
               "not_blocking")
    return {"per_emotion": per_emotion, "counts": counts, "verdict": verdict,
            "doses": doses, "emotions": list(emos), "arms": list(arms), "n_boot": n_boot}


def print_summary(res):
    per, cnt = res["per_emotion"], res["counts"]
    print(f"\n{'emotion':>10} {'MC-abl':>7} {'sep':>5} {'drop_emo':>9} {'drop_rand':>10} "
          f"{'drop_ceil':>10} {'drop_text':>10} {'emo-vs-rand':>22} {'blockfrac':>10} "
          f"{'mde':>7}")
    for e in res["emotions"]:
        s = per.get(e, {})
        mc = s.get("mc_ablate", {})
        evr = s.get("emo_vs_rand")
        d = mc.get("drop_by_arm", {})
        bf = s.get("blocked_fraction", {}).get("emo_vs_rand", {}).get("point")
        txt = (f"{evr['diff']:+.2f} CI[{evr['ci'][0]:+.2f},{evr['ci'][1]:+.2f}]"
               if evr else "--")
        print(f"{e:>10} {str(mc.get('passes')):>7} {str(mc.get('separated')):>5} "
              f"{_f(d.get('emo')):>9} {_f(d.get('rand')):>10} {_f(d.get('ceiling')):>10} "
              f"{_f(d.get('text')):>10} {txt:>22} "
              f"{(f'{bf:+.2f}' if bf is not None else '--'):>10} "
              f"{_f(s.get('mde', {}).get('value')):>7}")
    print(f"\nMC-ablate passes {cnt['n_mc_pass']}/{cnt['n_mc_measured']} · "
          f"blocking {cnt['n_block']} · anti-blocking {cnt['n_antiblock']} · "
          f"untestable {cnt['n_untestable']}/{len(res['emotions'])}")
    print(f"VERDICT: {res['verdict']}")


def _f(x, nd=3):
    return "--" if x is None or (isinstance(x, float) and not np.isfinite(x)) \
        else f"{x:+.{nd}f}"


# ---------------------------------------------------------------------------------------
# --selftest — analysis on synthetic rows with a planted effect (no model, no torch)
# ---------------------------------------------------------------------------------------

def _synth_rows(doses, emos, arms, n_scen=12, reps=2, seed=7,
                blocked=("desperate", "afraid"), reduction=0.30):
    """Rows with a KNOWN answer: the emo arm's dose slope is `reduction` smaller than the
    none arm's for the emotions in `blocked`, and exactly equal to it for the rest.

    Every arm in one (emotion, dose, rep) cell shares its noise draw, so the null arms are
    identical rather than merely similar — a paired contrast of two identical arms has diff
    0 and a degenerate CI, which makes "not significant" deterministic instead of a 1-in-20
    coin flip per emotion. Only the planted difference is left for the test to find.
    """
    rng = np.random.default_rng(seed)
    slope = {"none": 1.0, "emo": 1.0, "rand": 1.0, "ceiling": 1.0, "text": 1.0}
    read_top = {"none": 0.70, "emo": 0.40, "rand": 0.68, "ceiling": 0.20, "text": 0.30}
    rows = []
    for e in emos:
        for d in doses:
            for rep in range(reps):
                nz = rng.normal(0, 0.15, n_scen)          # shared across arms
                nzo = rng.normal(0, 0.15, n_scen)
                nzr = rng.normal(0, 0.02, n_scen)
                a_read = 0.2 + 0.5 * d + rng.normal(0, 0.05, n_scen)
                for arm in arms:
                    k = slope[arm] * (1.0 - reduction
                                      if (arm == "emo" and e in blocked) else 1.0)
                    base = read_top["none"] if d == 0 else read_top[arm]
                    rows.append({
                        "emotion": e, "alpha": float(d), "rep": rep, "arm": arm,
                        "scenario": list(range(n_scen)),
                        "B_present_e": (k * d + nz).tolist(),
                        "B_other_e": (0.05 * d + nzo).tolist(),
                        "A_readout_e": a_read.tolist(),
                        "ctx_readout_e": (base + nzr).tolist(),
                        "ctx_readout_full": [1.0 / 6] * 6,
                        "removed_norm": 0.0 if arm in ("none", "text") else 3.2,
                        "mask_hit_rate": None if arm in ("none", "text") else 1.0,
                        "mask_n_positions": None if arm in ("none", "text")
                        else (740.0 if arm == "ceiling" else 46.0),
                        "degenerate_frac": 0.02,
                        "refusal_frac": 0.0, "distinct2": 0.9, "n_words": 40.0,
                        "perplexity": 12.0,
                        "gen_seed": gen_seed(0, e, d, rep, arm, "B"),
                        "rand_dir_seed": rand_dir_seed(0, e, rep) if arm == "rand" else None,
                        **({"rewrite_failed": 0.0, "rewrite_degenerate_frac": 0.0,
                            "rewrite_refusal_frac": 0.0} if arm == "text" else {}),
                    })
    return rows


def selftest() -> int:
    t0 = time.time()
    blocked = ("desperate", "afraid")
    rows = _synth_rows(DOSES, EMOS, ARMS, blocked=blocked, reduction=0.30)
    print(f"[selftest] {len(rows)} synthetic arm-rows; planted: emo slope -30% vs none "
          f"for {blocked}, all other arms identical to none", flush=True)
    # n_boot below the 5000 default only to keep this under a minute on a CPU box; the
    # planted contrasts are either exactly 0 or -0.30 with a hair-width CI, so the
    # significance calls do not depend on the bootstrap size.
    res = analyze(rows, DOSES, EMOS, n_boot=1200)
    print_summary(res)
    per, cnt = res["per_emotion"], res["counts"]

    assert cnt["n_untestable"] == 0, cnt
    assert cnt["n_mc_pass"] == len(EMOS), cnt          # planted drop_emo > drop_rand
    assert cnt["n_block"] == len(blocked), cnt
    assert sorted(cnt["blocking_emotions"]) == sorted(blocked), cnt
    assert cnt["n_antiblock"] == 0, cnt
    assert res["verdict"] == "mixed", res["verdict"]
    for e in EMOS:
        s = per[e]
        assert s["_complete"] and set(ARMS) <= set(s), e
        assert np.isfinite(s["mde"]["value"]) and s["mde"]["value"] > 0, (e, s["mde"])
        for name in ("emo_vs_rand", "emo_vs_ceiling"):
            bf = s["blocked_fraction"][name]
            assert bf["defined"] and np.isfinite(bf["point"]), (e, name, bf)
            assert all(np.isfinite(x) for x in bf["range"]), (e, name, bf)
        got = s["blocked_fraction"]["emo_vs_rand"]["point"]
        want = 0.30 if e in blocked else 0.0
        assert abs(got - want) < 0.05, (e, got, want)
        assert s["mc_ablate"]["separated"] is True, e
        assert s["mc_steer"]["sig"] is True, e
        # the ceiling arm was planted with the none-arm slope, so it must not read as
        # blocking either -- a rule that fires on an arm with no planted effect is broken
        assert s["ceiling_vs_none"]["sig"] is False, e
    # and the rule must MOVE when the data does (a verdict that cannot change is not a
    # rule). Smaller grid + fewer boots purely for speed.
    small = dict(n_scen=8, reps=2)
    flat = analyze(_synth_rows(DOSES, EMOS, ARMS, blocked=EMOS, reduction=0.30, **small),
                   DOSES, EMOS, n_boot=600)
    assert flat["counts"]["n_block"] == len(EMOS) and flat["verdict"] == "blocking", \
        flat["counts"]
    none_ = analyze(_synth_rows(DOSES, EMOS, ARMS, blocked=(), reduction=0.30, **small),
                    DOSES, EMOS, n_boot=600)
    assert none_["counts"]["n_block"] == 0 and none_["verdict"] == "not_blocking", \
        none_["counts"]
    print(f"[selftest] verdict moves blocking/mixed/not_blocking with the planted effect "
          f"({time.time()-t0:.1f}s)")
    print("SELFTEST_OK", flush=True)
    return 0


# ---------------------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--tag")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--estimator", default="dom")
    ap.add_argument("--outdir", default="/marimo/out")
    ap.add_argument("--workdir", default="/marimo/work")
    ap.add_argument("--gen-bs", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true",
                    help="run analyze() on synthetic rows and exit; loads no model")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if not a.model or not a.tag:
        ap.error("--model and --tag are required (or pass --selftest)")
    os.makedirs(a.outdir, exist_ok=True); os.makedirs(a.workdir, exist_ok=True)

    prov = C.Provenance(
        script="src/rev3/b1_followup.py",
        config={"doses": DOSES, "arms": ARMS, "emotions": EMOS, "reps": a.reps,
                "estimator": a.estimator, "probe_k": PROBE_K,
                "stability_gate": STABILITY_GATE, "max_new_A": 110, "max_new_B": 110,
                "max_new_rewrite": 110, "temp": 0.9, "top_p": 0.95,
                "n_scenarios": len(C.SCENARIOS),
                "gen_seed_rule": "crc32('seed|emotion|alpha|rep|arm|part') & 0x7fffffff, "
                                 "+ batch start index; `arm` is EMPTY for part 'B' so all "
                                 "five arms sample B's reply with common random numbers, "
                                 "and kept for parts 'readout' and 'rewrite'",
                "predecessor": "results/rev3/b1_e4rerun_qwen36-27b.json"},
        model_id=a.model, model_revision="", seeds={"pool": a.seed, "run": a.seed},
        code_sha=C.code_hash(os.path.abspath(__file__), os.path.abspath(C.__file__)),
        control_pointers={
            # Every span below is read out of THIS file's AST at import time by fn_lines /
            # block_lines -- never typed, so it cannot go stale the way the completed run's
            # `build_dirs` pointer did (defect (f)). A name that stops existing renders as
            # "NOT FOUND -- pointer is stale" instead of quietly pointing at nothing.
            "norm_matched_random": f"b1_followup.py::rand_dirs_for "
                                   f"({fn_lines('rand_dirs_for')}) -> "
                                   f"acl_core.random_dir_like, seeded per (emotion, rep) "
                                   f"by ::rand_dir_seed ({fn_lines('rand_dir_seed')}); the "
                                   f"seed is in each row as rand_dir_seed and the "
                                   f"projection norm actually removed as removed_norm "
                                   f"(read it with mask_n_positions)",
            "ceiling_control": f"b1_followup.py::gen_B ({fn_lines('gen_B')}) and ::readout "
                               f"({fn_lines('readout')}), arm 'ceiling': mask = the full "
                               f"attention mask, same Ablate hooks and same emo direction, "
                               f"so it upper-bounds a span-limited residual ablation",
            "token_level_control": f"b1_followup.py::rewrite_A ({fn_lines('rewrite_A')}), "
                                   f"REWRITE_PROMPT copied from "
                                   f"src/representation/coupling_e2_ci.py; B replies with "
                                   f"NO hooks and the readout is taken on the rewritten "
                                   f"message",
            "generation_side_manip_check": f"b1_followup.py::readout "
                                           f"({fn_lines('readout')}) — the forced-choice "
                                           f"emotion question answered from the model's "
                                           f"own next-token distribution with the ablation "
                                           f"live on A's span",
            "reproducible_sampling": f"b1_followup.py::gen_seed ({fn_lines('gen_seed')}), "
                                     f"called immediately before every h.model.generate in "
                                     f"::gen_B ({fn_lines('gen_B')}), ::rewrite_A "
                                     f"({fn_lines('rewrite_A')}) and before the readout "
                                     f"forward in ::readout ({fn_lines('readout')}); part "
                                     f"'B' drops `arm` from the key so every arm samples "
                                     f"B's reply with common random numbers",
            "per_layer_stability_gate": f"b1_followup.py main ({block_lines('GATE')}): "
                                        f"acl_core.split_half at EVERY ablated "
                                        f"hidden_states index, stored in "
                                        f"out['direction_stability_by_layer'] and "
                                        f"out['layers_below_gate']",
            "non_circular_measurement": f"b1_followup.py main ({block_lines('POOL')}): "
                                        f"probe fitted on the READ half, steering/ablation "
                                        f"directions on the disjoint DIR half",
            "sign_aware_decision_rule": f"b1_followup.py::analyze ({fn_lines('analyze')}) "
                                        f"and ::blocked_fraction "
                                        f"({fn_lines('blocked_fraction')}); the rule is "
                                        f"stated in this file's module docstring and was "
                                        f"fixed before the run",
            "frozen_token_audit": "arms none/emo/rand/ceiling regenerate B under the "
                                  "intervention while A's message predates it; arm 'text' "
                                  "regenerates A's message too. See "
                                  "out['frozen_token_audit'].",
        })

    h = C.load(a.model)
    prov.model_revision = h.revision
    focus = h.focus(); steer_layer = focus - 1
    lo = round(0.2 * h.n_layers)
    abl_dec = [d for d in range(h.n_layers) if lo <= d + 1 <= focus - 1]
    abl_hs = [d + 1 for d in abl_dec]
    hs_needed = sorted(set(abl_hs) | {focus})
    print(f"[B1F] focus {focus} steer_layer {steer_layer} ablating {len(abl_dec)} blocks "
          f"(hs {abl_hs[0]}..{abl_hs[-1]})", flush=True)

    # <<POOL_START>>
    # ---------------- probe pool, split into DIR and READ halves ------------------------
    # Same path, same k, same seed, same permutation as b1_e4rerun.py: with the pool file
    # from that run present in --workdir, the directions below are the SAME vectors.
    items = C.generate_pool(h, PROBE_K, os.path.join(a.workdir, f"probe_{a.tag}.jsonl"),
                            seed=a.seed, bs=a.gen_bs)
    feats, yp, yo = C.pool_features(h, items, hs_needed,
                                    cache=os.path.join(a.workdir, f"probefeat_{a.tag}.npz"))
    N = len(items)
    rs = np.random.default_rng(a.seed).permutation(N)
    DIR, READ = rs[:N // 2], rs[N // 2:]
    print(f"[B1F] probe pool n={N}: DIR half {len(DIR)}, READ half {len(READ)}", flush=True)

    dir_dec = {L: C.fit_direction(feats[L][DIR], yp[DIR], a.estimator, seed=a.seed,
                                  pair_on=yo[DIR]) for L in hs_needed}
    read_dec = {L: C.fit_direction(feats[L][READ], yp[READ], a.estimator, seed=a.seed,
                                   pair_on=yo[READ]) for L in hs_needed}
    read_dec_other = C.fit_direction(feats[focus][READ], yo[READ], a.estimator, seed=a.seed)
    # <<POOL_END>>

    # <<GATE_START>>
    # ---- GATE: stability of EVERY direction we ablate along, not just the focus one -----
    # (defect (b)). pair_on is passed so the focus-layer entry is computed by exactly the
    # call b1_e4rerun.py made and stays comparable; it only affects the pca_diff estimator.
    stab_by_layer = {}
    for L in hs_needed:
        stab_by_layer[L] = C.split_half(feats[L], yp, a.estimator,
                                        n_per_half=min(N // 2, 600), seeds=range(10),
                                        pair_on=yo)
        print(f"[B1F gate] hs {L:>3} split-half({a.estimator}) "
              f"{stab_by_layer[L]['mean']:.3f} CI {stab_by_layer[L]['ci']}", flush=True)
    abl_means = [stab_by_layer[L]["mean"] for L in abl_hs]
    layers_below_gate = [L for L in hs_needed
                         if not (stab_by_layer[L]["mean"] >= STABILITY_GATE)]
    stab = stab_by_layer[focus]
    stab_logreg = C.split_half(feats[focus], yp, "logreg", n_per_half=min(N // 2, 600),
                               seeds=range(10))
    print(f"[B1F GATE] ablated layers hs {abl_hs[0]}..{abl_hs[-1]}: min "
          f"{np.min(abl_means):.3f} median {np.median(abl_means):.3f} max "
          f"{np.max(abl_means):.3f}; {len(layers_below_gate)} layer(s) below the "
          f"{STABILITY_GATE} gate: {layers_below_gate}", flush=True)
    print(f"[B1F GATE] focus hs {focus}: split-half({a.estimator}) {stab['mean']:.3f} CI "
          f"{stab['ci']}  |  logreg {stab_logreg['mean']:.3f}", flush=True)
    gate_pass = bool(stab["mean"] >= STABILITY_GATE)      # focus layer = abort criterion
    if not gate_pass:
        print("[B1F GATE] *** FOCUS DIRECTION BELOW GATE — the sweep still runs, but the "
              "ablation arms are not interpretable as emotion-specific ***", flush=True)
    # <<GATE_END>>

    # ---------------- directions ---------------------------------------------------------
    def tt(v):
        return torch.tensor(v, dtype=h.model.dtype, device=h.model.device)

    emo_dirs, steer_dir = {}, {}
    for e in EMOS:
        ei = C.EMOTIONS.index(e)
        emo_dirs[e] = {L: tt(C.raw_direction(dir_dec[L], ei)) for L in abl_hs}
        steer_dir[e] = tt(C.raw_direction(dir_dec[focus], ei))

    _rand_cache = {}

    def rand_dirs_for(e, rep):
        """A FRESH norm-matched random direction per (emotion, rep) — defect (c)."""
        key = (e, rep)
        if key not in _rand_cache:
            s = rand_dir_seed(a.seed, e, rep)
            rr = np.random.default_rng(s)
            _rand_cache[key] = ({L: tt(C.random_dir_like(C.raw_direction(dir_dec[L], 0), rr))
                                 for L in abl_hs}, s)
        return _rand_cache[key]

    steer = C.Steer(h, steer_layer)
    ab = C.Ablate(h, abl_dec)
    rms = float(np.linalg.norm(feats[focus], axis=1).mean())
    print(f"[B1F] rms {rms:.1f}", flush=True)

    # ---------------- the sweep ----------------------------------------------------------
    S = C.SCENARIOS
    rlog = C.ResponseLog(os.path.join(a.workdir, f"responses_b1f_{a.tag}.jsonl"))
    ck = C.Checkpoint(os.path.join(a.workdir, f"b1f_cells_{a.tag}.json"),
                      {"doses": DOSES, "arms": ARMS, "emos": EMOS, "reps": a.reps,
                       "est": a.estimator, "n": N, "focus": focus, "probe_k": PROBE_K,
                       "driver": "b1_followup"})

    def gen_A(e, alpha, rep):
        """A's steered message. Unchanged from b1_e4rerun.py, seed included, so A's half of
        the stimulus is comparable with the completed run; acl_core.gen seeds each batch."""
        names = [C.NAMES[(i + rep) % len(C.NAMES)] for i in range(len(S))]
        prompts, ctx = [], []
        for (setting, bopen), (A, B) in zip(S, names):
            s = setting.format(A=A, B=B); bo = bopen.format(A=A, B=B)
            prompts.append(f"{s}\n{bo}\nContinue the conversation. Write only {A}'s next "
                           f"reply as one short paragraph, starting with '{A}:'.")
            ctx.append({"A": A, "B": B, "setting": s, "bopen": bo, "scenario": len(ctx)})
        vec = None if alpha == 0 else (alpha * rms) * steer_dir[e]
        ab.clear()
        outs = C.gen(h, prompts, max_new=110, bs=a.gen_bs, seed=a.seed + rep,
                     first_line=True,
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

    def dirs_for(arm, e, rep):
        """Directions per arm. 'none' and 'text' run with NO hooks at all; 'ceiling' uses
        the emo direction and differs only in which positions the mask selects."""
        if arm in ("none", "text"):
            return None, None
        if arm == "rand":
            return rand_dirs_for(e, rep)
        return emo_dirs[e], None                       # 'emo' and 'ceiling'

    def gen_B(ctx, arm, e, alpha, rep, bs=30):
        """B regenerates while the direction is projected out during B's PREFILL.

        arm 'emo'/'rand' mask A's token span; 'ceiling' masks EVERY attended position;
        'none'/'text' install no hook. Seeded per batch (defect (a)).

        Returns removed_norm AND mask_n_positions, because removed_norm is a mean over the
        masked positions: the ceiling arm averages it over the whole prefill and emo/rand
        over A's span only, so the number is uninterpretable without the count it was
        averaged over. mask_hit_rate is None for the un-hooked arms rather than a 1.0 that
        would read as "the span was found" when nothing was ever looked for."""
        dirs, _seed = dirs_for(arm, e, rep)
        base = gen_seed(a.seed, e, alpha, rep, arm, "B")
        h.tok.padding_side = "left"
        out, removed, hitrate, npos = [], [], [], []
        for i in range(0, len(ctx), bs):
            batch = ctx[i:i + bs]
            texts = [C.chat(h, c["bprompt"]) for c in batch]
            enc = h.tok(texts, return_tensors="pt", padding=True, return_offsets_mapping=True)
            offs = enc.pop("offset_mapping")
            enc = {k: v.to(h.model.device) for k, v in enc.items()}
            if dirs is not None:
                if arm == "ceiling":
                    M, hits = enc["attention_mask"].bool(), len(batch)
                else:
                    M, hits = mask_for(h, texts, [c["ar"] for c in batch], enc, offs)
                hitrate.append(hits / len(batch))
                npos.append(float(M.sum().item()) / len(batch))
                ab.set(arm, M, dirs)
            torch.manual_seed(base + i)
            try:
                with torch.no_grad():
                    g = h.model.generate(**enc, max_new_tokens=110, do_sample=True,
                                         temperature=0.9, top_p=0.95,
                                         pad_token_id=h.tok.pad_token_id)
            finally:
                # only the intervened arms have a projection to report; reading this on an
                # un-hooked arm would pick up the previous arm's stale accumulator
                if dirs is not None and ab.removed_norm:
                    removed.append(float(np.mean(ab.removed_norm)))
                ab.clear()
            out += [o.strip().split("\n")[0].strip() for o in
                    h.tok.batch_decode(g[:, enc["input_ids"].shape[1]:],
                                       skip_special_tokens=True)]
        rlog.write([c["conv"] for c in ctx], out, f"B_{e}_a{alpha}_r{rep}_{arm}")
        return out, (float(np.mean(removed)) if removed else 0.0), \
            (float(np.mean(hitrate)) if hitrate else None), base, \
            (float(np.mean(npos)) if npos else None)

    def readout(ctx, arm, e, alpha, rep, bs=30):
        """GENERATION-SIDE manipulation check: with the intervention live, does the model
        still name A's emotion in its OWN output distribution? Returns [N, 6] probs.

        Deterministic (a single forward), but seeded anyway so the call is reproducible
        under any stochastic kernel."""
        dirs, _seed = dirs_for(arm, e, rep)
        base = gen_seed(a.seed, e, alpha, rep, arm, "readout")
        h.tok.padding_side = "left"
        ids = []
        for em in C.EMOTIONS:
            cand = [h.tok.encode(x, add_special_tokens=False)
                    for x in (em, " " + em, em.capitalize())]
            ids.append(sorted({c[0] for c in cand if c}))
        res = []
        for i in range(0, len(ctx), bs):
            batch = ctx[i:i + bs]
            texts = [C.chat(h, f"{c['conv']}\n\n{C.READOUT_Q.format(A=c['A'])}")
                     for c in batch]
            enc = h.tok(texts, return_tensors="pt", padding=True, return_offsets_mapping=True)
            offs = enc.pop("offset_mapping")
            enc = {k: v.to(h.model.device) for k, v in enc.items()}
            if dirs is not None:
                if arm == "ceiling":
                    M = enc["attention_mask"].bool()
                else:
                    M, _ = mask_for(h, texts, [c["ar"] for c in batch], enc, offs)
                ab.set(arm, M, dirs)
            torch.manual_seed(base + i)
            try:
                with torch.no_grad():
                    lg = h.model(**enc).logits[:, -1, :].float()
            finally:
                ab.clear()
            cols = torch.stack([lg[:, torch.tensor(g, device=lg.device)].logsumexp(-1)
                                for g in ids], -1)
            res.append(cols.softmax(-1).cpu().numpy())
        return np.concatenate(res, 0)

    def rewrite_A(ctx, e, alpha, rep):
        """TOKEN-LEVEL arm (defect (d)): replace A's steered line with a neutral rewrite by
        the UNSTEERED, un-ablated model, then let B answer that with no hooks at all.

        A rewrite that comes back empty (or byte-identical to the original) is NOT dropped
        -- the original line is kept so the conversation stays well-formed -- but it is
        counted in `rewrite_failed`, which is therefore the fraction of this row's
        scenarios for which the 'text' arm is really a copy of the 'none' arm.
        """
        base = gen_seed(a.seed, e, alpha, rep, "text", "rewrite")
        ab.clear(); steer.off()
        prompts = [REWRITE_PROMPT.format(ar=c["ar"], A=c["A"]) for c in ctx]
        outs = C.gen(h, prompts, max_new=110, bs=a.gen_bs, seed=base, first_line=True,
                     log=rlog, tag=f"REWRITE_{e}_a{alpha}_r{rep}")
        new, fails, texts = [], 0, []
        for c, pr in zip(ctx, outs):
            A = c["A"]
            pr = pr.strip()
            if pr and not pr.startswith(f"{A}:"):
                pr = f"{A}: {pr}"
            bad = (len(pr.split()) < 3) or (pr.strip() == c["ar"].strip())
            fails += bool(bad)
            ar = c["ar"] if bad else pr
            texts.append(ar)
            c2 = dict(c)
            c2["ar"] = ar
            c2["conv"] = f"{c['setting']}\n{c['bopen']}\n{ar}"
            c2["bprompt"] = (f"{c2['conv']}\nWrite only {c['B']}'s next reply as one short "
                             f"paragraph, starting with '{c['B']}:'.")
            new.append(c2)
        deg = [C.degeneracy(t) for t in texts]
        return new, {"rewrite_failed": fails / max(len(ctx), 1),
                     "rewrite_failed_n": int(fails),
                     "rewrite_degenerate_frac": float(np.mean([C.is_degenerate(d)
                                                               for d in deg])),
                     "rewrite_refusal_frac": float(np.mean([d["refusal"] for d in deg])),
                     "rewrite_n_words": float(np.mean([d["n_words"] for d in deg]))}

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
                a_read = readout(ctx, "none", e, alpha, rep)
                cell = []
                for arm in ARMS:
                    actx, extra = ctx, {}
                    if arm == "text":
                        actx, extra = rewrite_A(ctx, e, alpha, rep)
                    breps, removed, hit, gseed, npos = gen_B(actx, arm, e, alpha, rep)
                    pres, oth = score_B(actx, breps)
                    # MC-ablate: is the affect still readable from the intervened context?
                    ab_read = readout(actx, arm, e, alpha, rep)
                    deg = [C.degeneracy(b) for b in breps]
                    ppl = C.perplexity(h, breps)
                    _d, rseed = dirs_for(arm, e, rep)
                    cell.append({
                        "emotion": e, "alpha": alpha, "rep": rep, "arm": arm,
                        "scenario": [c["scenario"] for c in actx],
                        "B_present_e": pres[:, ei].tolist(),
                        "B_other_e": oth[:, ei].tolist(),
                        "A_readout_e": a_read[:, ei].tolist(),
                        "ctx_readout_e": ab_read[:, ei].tolist(),
                        "ctx_readout_full": ab_read.mean(0).tolist(),
                        "removed_norm": removed, "mask_hit_rate": hit,
                        "mask_n_positions": npos,
                        "degenerate_frac": float(np.mean([C.is_degenerate(d) for d in deg])),
                        "refusal_frac": float(np.mean([d["refusal"] for d in deg])),
                        "distinct2": float(np.mean([d["distinct2"] for d in deg])),
                        "n_words": float(np.mean([d["n_words"] for d in deg])),
                        "perplexity": float(np.median(ppl)),
                        "gen_seed": gseed, "rand_dir_seed": rseed,
                        **extra,
                    })
                    r = cell[-1]
                    print(f"[{e:>9} a{alpha} r{rep} {arm:>7}] present "
                          f"{np.mean(r['B_present_e']):+.2f} ctx_readout(e) "
                          f"{np.mean(r['ctx_readout_e']):.3f} removed {removed:.2f} "
                          f"over {'--' if npos is None else f'{npos:.0f}'} pos "
                          f"deg {r['degenerate_frac']:.2f} ppl {r['perplexity']:.1f} "
                          f"seed {gseed}"
                          + (f" rewrite_failed {r['rewrite_failed']:.2f}"
                             if arm == "text" else ""), flush=True)
                ck.put(cid, cell)
                rows.extend(cell)
                print(f"[B1F] {len(rows)} arm-rows, {time.time()-t_start:.0f}s elapsed",
                      flush=True)

    # ---------------- analysis -----------------------------------------------------------
    res = analyze(rows, DOSES, EMOS)
    print_summary(res)

    out = {
        "model": a.model, "tag": a.tag, "focus": focus, "steer_layer": steer_layer,
        "n_abl_blocks": len(abl_dec), "abl_hs_range": [abl_hs[0], abl_hs[-1]],
        "abl_hs": abl_hs, "probe_n": N, "dir_half_n": len(DIR), "read_half_n": len(READ),
        "estimator": a.estimator, "rms": rms, "doses": list(DOSES), "arms": list(ARMS),
        "predecessor": "results/rev3/b1_e4rerun_qwen36-27b.json",
        "direction_stability": {"used": stab, "logreg_for_comparison": stab_logreg,
                                "gate": STABILITY_GATE, "gate_pass": gate_pass,
                                "gate_layer": focus},
        "direction_stability_by_layer": {str(L): stab_by_layer[L] for L in hs_needed},
        "ablated_layer_stability": {"min": float(np.min(abl_means)),
                                    "median": float(np.median(abl_means)),
                                    "max": float(np.max(abl_means)),
                                    "per_layer_mean": {str(L): stab_by_layer[L]["mean"]
                                                       for L in abl_hs}},
        "layers_below_gate": layers_below_gate,
        "frozen_token_audit": C.frozen_token_audit({
            "generated": ["B's reply (regenerated under the intervention, arms "
                          "emo/rand/ceiling)",
                          "the readout answer token (produced with the intervention live)",
                          "A's rewritten message and B's reply to it (arm 'text', both "
                          "produced with NO hooks)"],
            "frozen": ["A's steered message (generated BEFORE any ablation; rewritten, not "
                       "ablated, in arm 'text')",
                       "the scenario setting and B's opening line (fixed stimuli)"]}),
        "verdict": res["verdict"], "counts": res["counts"],
        "summary": res["per_emotion"], "rows": rows,
    }
    C.write_result(os.path.join(a.outdir, f"b1_followup_{a.tag}.json"), out, prov)
    print(f"\n[B1F] VERDICT: {res['verdict']}  (MC-ablate passed for "
          f"{res['counts']['n_mc_pass']}/{res['counts']['n_mc_measured']}; blocking "
          f"{res['counts']['n_block']}, anti-blocking {res['counts']['n_antiblock']}, "
          f"untestable {res['counts']['n_untestable']})", flush=True)
    print("B1F_DONE", flush=True)


if __name__ == "__main__":
    main()

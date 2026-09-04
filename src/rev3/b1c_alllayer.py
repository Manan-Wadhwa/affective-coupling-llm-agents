#!/usr/bin/env python3
"""B1c — ALL-LAYER ablation: does the mid-stack emotion direction carry a minority of
agent-to-agent transfer? (docs/planning/PREREG_B1c.md, committed before this file existed.)

This driver implements that pre-registration and nothing else. Where the pre-registration
says a quantity is fixed in advance, it is written here as a constant, not chosen at run
time.

HYPOTHESES, stated before the data (PREREG_B1c §2, verbatim)

  - **H1 (rank-1 insufficiency).** With the emotion direction projected out at all layers
    13-63 during B's prefill, B's dose-response is still at least half of its unablated
    value: the median blocked fraction over testable emotions is **<= 0.50**.
  - **H1' (filterable channel).** All-layer ablation blocks most of the transfer: median
    blocked fraction **>= 0.80**.
  - The region 0.50-0.80 is **indeterminate** and will be reported as such, not resolved
    post hoc.

  Either H1 or H1' is a publishable result. H1 supports the sentence *"a rank-1 emotion
  direction does not carry agent-to-agent affect transfer; the transfer is not linearly
  filterable at any depth"*. H1' supports *"the transfer is carried by the emotion
  direction and can be filtered"*, which is the repo's original headline with an experiment
  behind it for the first time.

DECISION RULE, fixed in advance (PREREG_B1c §5, verbatim)

  - Per-arm dose-response slope: OLS over the four doses on per-scenario means; CI by
    **scenario-blocked** bootstrap (29 clusters).
  - Contrasts (`emo_all - rand_all`, `emo13_42 - rand_all`, `text - text_keep`,
    `text - none`): paired across arms **and doses**, scenario-blocked — a one-sample
    cluster bootstrap on the per-scenario paired slope differences.
    `acl_core.paired_slope_contrast` will be given `block=` and dose pairing *before* the
    run; the change is recorded in the results log.
  - **Testable emotions:** those whose `none` slope CI excludes zero. Others are reported
    but do not enter the decision.
  - **Blocked fraction** = -(arm - control contrast) / `none` slope, with the CI-implied
    range, two-sided; the denominator's own CI is reported next to it.
  - **Detectable effect** per emotion from the contrast's own bootstrap (not `acl_core.mde`).
  - **Decision:** median blocked fraction of `emo_all` vs `rand_all` over testable
    emotions, with a bootstrap CI over scenarios. H1 if median <= 0.50 and the upper CI
    bound <= 0.65; H1' if median >= 0.80 and the lower bound >= 0.60; otherwise
    indeterminate. Per-emotion counts (significant negative / positive / null) are reported
    alongside; they do not override the median rule.
  - **Sign-aware.** Positive contrasts count as anti-blocking, never toward blocking.
  - Multiplicity: the decision is one test (the median). Per-emotion contrasts are
    descriptive and reported with BH-FDR over the six.

  Additionally (§4.2): if MC-ablate fails for >= 3 emotions the instrument failed and no
  channel claim is made — the verdict is "instrument_failed" and overrides the median rule.
  The verdict string is one of: "H1", "H1prime", "indeterminate", "instrument_failed".

ARMS (PREREG_B1c §3; six, `ceiling` dropped because the follow-up answered positional
coverage)

  none       no ablation                                                     baseline
  emo13_42   emotion direction out of A's span, hs 13-42                     replicates
                                                                             b1_followup's
                                                                             `emo`
  emo_all    emotion direction out of A's span, hs 13..n_layers-1            the H1 vs H1'
                                                                             test
  rand_all   FRESH norm-matched random direction per (emotion, rep), same    control for
             layers as emo_all, redrawn at every ablated layer               emo_all
  text       A's message replaced by an unsteered NEUTRAL rewrite; refusals, token-level
             empty (<3 words) and identical rewrites EXCLUDED and counted    ablation
  text_keep  A's message rewritten to keep tone and emotion but change every affect-
             specific; same exclusions, its own counters                     preserving
                                                                             control

  `none`, `text` and `text_keep` install no hook at all. B's replies are seeded with a
  common seed across arms (b1_followup's rule), so shared sampling noise cancels inside
  each paired arm contrast.

OUTCOMES (§4)
  primary    B_present_e   — the READ probe at the focus layer on B's GENERATED reply, in a
                             separate forward pass with all hooks off (`score_B`).
  secondary  B_readout_e   — the forced-choice readout applied to B's reply, asking about
                             B, hooks off.
  MC         ctx_readout_e — the forced-choice readout on A's span with the arm's ablation
                             LIVE, so `emo_all` / `emo13_42` / `rand_all` are comparable.

Usage: python b1c_alllayer.py --model Qwen/Qwen3.6-27B --tag qwen36-27b --reps 3
       python b1c_alllayer.py --selftest      # analysis only, no model, no GPU, < 90 s
"""
import argparse, ast, hashlib, json, os, subprocess, sys, time, zlib

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

# ---- declared grid (PREREG_B1c §3; fixed in advance, not chosen at run time) ------------
DOSES = (0.0, 0.33, 0.67, 1.0)          # four points: the slope is no longer an endpoint
ARMS = ("none", "emo13_42", "emo_all", "rand_all", "text", "text_keep")
NO_HOOK_ARMS = ("none", "text", "text_keep")
REWRITE_ARMS = ("text", "text_keep")
EMOS = ("desperate", "afraid", "happy", "calm", "sad", "angry")
STABILITY_GATE = 0.80
PROBE_K = 60
PREREG = "docs/planning/PREREG_B1c.md"

# the six contrasts of §5 (the last two are the token-arm's own baselines)
CONTRASTS = (("emo_all", "rand_all"), ("emo13_42", "rand_all"), ("emo_all", "emo13_42"),
             ("text", "text_keep"), ("text", "none"), ("text_keep", "none"))
DECISION_CONTRAST = "emo_all_vs_rand_all"

# §4.3 text-quality thresholds and §4.4 rewrite threshold, pre-declared
MAX_BAD_FRAC = 0.10
MAX_PPL_RATIO = 2.0
REWRITE_READOUT_RATIO = 0.5
# §5 decision thresholds
H1_MEDIAN, H1_UPPER = 0.50, 0.65
H1P_MEDIAN, H1P_LOWER = 0.80, 0.60
MC_ABLATE_FAIL_LIMIT = 3
RAND_DROP_MAX = 0.02

# verbatim from src/representation/coupling_e2_ci.py (the paraphraser control), the same
# string b1_followup.py used, so the `text` arm is comparable across the two runs.
REWRITE_PROMPT = ('Rewrite this line in plain, emotionally neutral wording, '
                  'keeping only the literal information: "{ar}". '
                  "Start with '{A}:'. Output only the rewritten line.")
# §3: the affect-PRESERVING control. Fixed in advance, quoted here exactly as run.
REWRITE_KEEP_PROMPT = ('Rewrite the following line so that it keeps exactly the same tone '
                       'and emotional state but changes every specific detail (names, '
                       'objects, places, numbers): "{ar}". '
                       'Reply with only the rewritten line.')
REWRITE_PROMPTS = {"text": REWRITE_PROMPT, "text_keep": REWRITE_KEEP_PROMPT}


# ---- self-locating source pointers (b1_followup's mechanism, unchanged) -----------------
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
    """Span between the '# <<NAME_START>>' / '# <<NAME_END>>' sentinels, for the stretches
    of `main` that are controls but are not their own function."""
    tags = {"# <<" + f"{name}_{w}" + ">>": w for w in ("START", "END")}
    hit = {}
    for i, ln in enumerate(_SRC.splitlines(), 1):
        if ln.strip() in tags:
            hit[tags[ln.strip()]] = i
    if "START" in hit and "END" in hit:
        return f"L{hit['START'] + 1}-L{hit['END'] - 1}"
    return f"L?-L? ({name} sentinels NOT FOUND — pointer is stale)"


def file_sha256(path: str) -> str:
    """Full sha256 of a file, or '' if it is not there — the probe pool's identity (§8)."""
    try:
        hsh = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                hsh.update(chunk)
        return hsh.hexdigest()
    except Exception:
        return ""


def prereg_stamp() -> dict:
    """Path, content hash and (best effort) git commit of the pre-registration (§8)."""
    root = os.path.abspath(os.path.join(os.path.dirname(_SRC_PATH), "..", ".."))
    path = os.path.join(root, PREREG)
    commit = ""
    try:
        commit = subprocess.run(["git", "-C", root, "log", "-1", "--format=%H", "--",
                                 PREREG], capture_output=True, text=True,
                                timeout=10).stdout.strip()
    except Exception:
        commit = ""
    return {"path": PREREG, "sha256": file_sha256(path), "git_commit": commit,
            "found": os.path.exists(path)}


def gen_seed(seed: int, emotion: str, alpha: float, rep: int, arm: str, part: str = "") -> int:
    """Deterministic per-generation torch seed (b1_followup's rule, unchanged).

    crc32 of "seed|emotion|alpha|rep|arm|part", masked to 31 bits, plus the batch start
    index. For part "B" the `arm` field is left EMPTY, so B's reply is sampled from ONE
    stream in all six arms — common random numbers, so sampling noise shared between two
    arms cancels inside each paired contrast instead of widening its CI. "readout",
    "readout_B" and "rewrite" keep `arm`, because those calls differ between arms by
    construction.
    """
    key = f"{seed}|{emotion}|{alpha:g}|{rep}|{'' if part == 'B' else arm}|{part}"
    return int(zlib.crc32(key.encode()) & 0x7FFFFFFF)


def rand_dir_seed(seed: int, emotion: str, rep: int) -> int:
    """Seed for the fresh random control of one (emotion, rep) cell.

    Derived from (--seed, the CANONICAL emotion index in acl_core.EMOTIONS, rep) so it is
    stable under a resumed run and independent of the order EMOS happens to be written in.
    """
    ei = C.EMOTIONS.index(emotion)
    return int(zlib.crc32(f"rand|{seed}|{ei}|{rep}".encode()) & 0x7FFFFFFF)


def mask_for(h, texts, subs, enc, offs):
    """Boolean [B, T] mask selecting the tokens of `subs[b]` inside `texts[b]`.

    Lives here rather than in acl_core so that acl_core stays comparable with the copy
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

METRICS = ("B_present_e", "B_other_e", "A_readout_e", "ctx_readout_e", "B_readout_e")


def _rowkey(r) -> tuple:
    return (r["emotion"], r["arm"], float(r["alpha"]), int(r["rep"]))


def _index(rows, e, arm, d) -> dict:
    """{(rep, scenario): (row, position)} for one (emotion, arm, dose) cell.

    The `text` arms DROP excluded scenarios from their rows, so their key sets are smaller
    than the other arms'. Everything downstream works off these keys rather than off array
    positions, which is what makes a ragged text arm safe to compare.
    """
    out = {}
    for r in rows:
        if r["emotion"] == e and r["arm"] == arm and float(r["alpha"]) == float(d):
            for j, s in enumerate(r.get("scenario") or []):
                out[(int(r["rep"]), int(s))] = (r, j)
    return out


def panel(rows, e, arms, doses, metrics=METRICS):
    """A BALANCED panel: the (rep, scenario) keys present in EVERY (arm, dose) cell asked
    for, in one order, for every metric.

    Balance matters twice. `acl_core.slope_ci` takes ONE block array for all four doses, so
    the doses must line up; and §5's dose-paired contrast is only defined when a scenario
    has a value at every dose in both arms. Restricting to the intersection is the honest
    way to get there when the text arms have excluded different scenarios at different
    doses. Returns None when any requested cell is missing entirely.
    """
    doses = [float(d) for d in doses]
    idx = {(arm, d): _index(rows, e, arm, d) for arm in arms for d in doses}
    if any(len(v) == 0 for v in idx.values()):
        return None
    ks = sorted(set.intersection(*[set(v) for v in idx.values()]))
    if not ks:
        return None
    data = {}
    for key in metrics:
        data[key] = {}
        for arm in arms:
            data[key][arm] = {}
            for d in doses:
                vals = []
                for k in ks:
                    r, j = idx[(arm, d)][k]
                    v = r.get(key)
                    vals.append(float(v[j]) if isinstance(v, (list, tuple)) and j < len(v)
                                else np.nan)
                data[key][arm][d] = np.asarray(vals, float)
    return {"data": data, "blocks": np.asarray([k[1] for k in ks]), "keys": ks}


def block_slopes(doses, per_dose, blocks):
    """(scenario ids, per-scenario OLS slope over doses of that scenario's mean).

    The same unit `acl_core.paired_slope_contrast(pair_doses=True)` resamples; computed
    here as well because §5's DECISION needs the per-scenario numerator and denominator
    together, under one shared scenario draw.
    """
    doses = [float(d) for d in doses]
    uniq = sorted(set(np.asarray(blocks).tolist()))
    keep, sl = [], []
    for u in uniq:
        m = [per_dose[d][np.asarray(blocks) == u] for d in doses]
        if all(len(x) for x in m):
            keep.append(u)
            sl.append(C.ols_slope(doses, [x.mean() for x in m]))
    return np.asarray(keep), np.asarray(sl, float)


def blocked_fraction(contrast: dict | None, none_slope: dict | None) -> dict:
    """How much of the none-arm dose slope a contrast removes (§5).

    -diff / slope(none): 1.0 means the arm cancelled the whole dose response, 0.0 means it
    did nothing, negative means it AMPLIFIED it (sign-aware: that is anti-blocking, never
    blocking). `range` is the same transform applied to the two ends of the contrast CI,
    re-sorted because dividing by a negative denominator flips them. The denominator's own
    CI travels with the fraction, because a fraction of a slope that does not itself
    exclude zero is a ratio of two noise estimates, not a result.
    """
    if not contrast or not none_slope:
        return {"point": None, "range": [None, None], "defined": False}
    den = float(none_slope.get("slope", 0.0))
    base = {"denominator": den, "denominator_ci": none_slope.get("ci"),
            "denominator_sig": bool(none_slope.get("sig"))}
    if not np.isfinite(den) or abs(den) < 1e-9 or not np.isfinite(contrast.get("diff",
                                                                               np.nan)):
        return {"point": None, "range": [None, None], "defined": False, **base}
    lo, hi = sorted(-float(c) / den for c in contrast["ci"])
    return {"point": float(-float(contrast["diff"]) / den), "range": [lo, hi],
            "defined": True, **base}


def detectable_effect(contrast: dict | None) -> dict:
    """§5: the detectable effect comes from the CONTRAST'S OWN bootstrap — the half-width
    of its CI — not from `acl_core.mde`, which assumes an unclustered two-sample t."""
    if not contrast or not np.all(np.isfinite(contrast.get("ci", [np.nan, np.nan]))):
        return {"half_width": None, "units": None}
    lo, hi = contrast["ci"]
    return {"half_width": float((hi - lo) / 2.0), "ci": [float(lo), float(hi)],
            "units": "slope units (probe score per unit alpha), scenario-blocked "
                     "dose-paired bootstrap half-width"}


def bh_fdr(pvals) -> list:
    """Benjamini-Hochberg q-values. Descriptive only (§5: the decision is the median)."""
    p = np.asarray(pvals, float)
    n = len(p)
    if n == 0:
        return []
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank in range(n - 1, -1, -1):
        i = order[rank]
        prev = min(prev, p[i] * n / (rank + 1))
        q[i] = prev
    return [float(x) for x in q]


def quality_flags(rows, doses, emos) -> dict:
    """§4.3 text quality, per CELL (emotion, arm, dose, rep), each pre-declared pass/fail.

    NOTE on "degenerate + refusal <= 0.10": `acl_core.is_degenerate` already fires on the
    refusal flag, so `degenerate_frac` IS the union of the two sets and adding them would
    double-count. The union is what is thresholded; both numbers are recorded so the other
    reading is recoverable from the result file.
    """
    base = {}
    for r in rows:
        if r["arm"] == "none":
            base[(r["emotion"], float(r["alpha"]), int(r["rep"]))] = r.get("perplexity")
    out = {}
    for r in rows:
        e, arm, d, rep = _rowkey(r)
        bad = float(r.get("degenerate_frac") or 0.0)
        ref = float(r.get("refusal_frac") or 0.0)
        ppl = r.get("perplexity")
        ppl0 = base.get((e, d, rep))
        pfail = bool(ppl is not None and ppl0 not in (None, 0) and np.isfinite(ppl)
                     and np.isfinite(ppl0) and ppl > MAX_PPL_RATIO * ppl0)
        dfail = bool(bad > MAX_BAD_FRAC)
        out[_rowkey(r)] = {
            "degenerate_frac": bad, "refusal_frac": ref,
            "quality_bad_frac": bad,        # the union; see the docstring
            "perplexity": ppl, "perplexity_none": ppl0,
            "perplexity_ratio": (float(ppl / ppl0) if (ppl and ppl0) else None),
            "degenerate_fail": dfail, "perplexity_fail": pfail,
            "fail": bool(dfail or pfail)}
    return out


def analyze(rows, doses, emos, arms=ARMS, n_boot: int = 5000, seed: int = 0,
            apply_quality_exclusion: bool = True) -> dict:
    """PREREG_B1c §4-§5 in one importable, CPU-only function.

    Returns {'per_emotion', 'counts', 'decision', 'verdict', 'quality', ...}. Every slope
    is scenario-blocked; every contrast is scenario-blocked AND dose-paired through the new
    `acl_core.paired_slope_contrast(block=..., pair_doses=True)`; the verdict is the median
    rule of §5, with §4.2's instrument-failure override.
    """
    doses = [float(d) for d in doses]
    arms = tuple(arms)
    rows = [r for r in rows if r.get("scenario")]

    qual = quality_flags(rows, doses, emos)
    excluded = sorted(k for k, v in qual.items() if v["fail"])
    good = [r for r in rows if not (apply_quality_exclusion and qual[_rowkey(r)]["fail"])]

    per_emotion, dec_in = {}, {}
    for e in emos:
        s = {"_quality_excluded_cells": [list(k) for k in excluded if k[0] == e]}
        have = []
        for arm in arms:
            P = panel(good, e, (arm,), doses)
            if P is None:
                s[arm] = None
                continue
            have.append(arm)
            D, blk = P["data"], P["blocks"]
            rs = [r for r in good if r["emotion"] == e and r["arm"] == arm]
            s[arm] = {
                "present_slope": C.slope_ci(doses, D["B_present_e"][arm], n_boot=n_boot,
                                            seed=seed, block=blk),
                "other_slope": C.slope_ci(doses, D["B_other_e"][arm], n_boot=n_boot,
                                          seed=seed, block=blk),
                "readout_slope": C.slope_ci(doses, D["B_readout_e"][arm], n_boot=n_boot,
                                            seed=seed, block=blk),
                "mean_by_dose": {str(d): float(np.nanmean(D["B_present_e"][arm][d]))
                                 for d in doses},
                "readout_mean_by_dose": {str(d): float(np.nanmean(D["B_readout_e"][arm][d]))
                                         for d in doses},
                "ctx_readout_by_dose": {str(d): float(np.nanmean(D["ctx_readout_e"][arm][d]))
                                        for d in doses},
                "degenerate_by_dose": {
                    str(d): float(np.mean([r["degenerate_frac"] for r in rs
                                           if float(r["alpha"]) == d] or [np.nan]))
                    for d in doses},
                "n_scenario_keys": len(P["keys"]),
                "n_layers_ablated": _first(rs, "n_layers_ablated"),
                "removed_norm": _mean_of(rs, "removed_norm"),
                "mask_n_positions": _mean_of(rs, "mask_n_positions"),
                "text_excluded_frac": _mean_of(rs, "text_excluded_frac"),
                "text_excluded_n": _sum_of(rs, "text_excluded_n"),
            }

        # ---- §5 contrasts: scenario-blocked AND dose-paired --------------------------
        for x, y in CONTRASTS:
            name = f"{x}_vs_{y}"
            P = panel(good, e, (x, y), doses, metrics=("B_present_e", "B_readout_e"))
            if P is None:
                s[name] = None
                continue
            blk = P["blocks"]
            s[name] = C.paired_slope_contrast(doses, P["data"]["B_present_e"][x],
                                              P["data"]["B_present_e"][y],
                                              n_boot=n_boot, seed=seed,
                                              block=blk, pair_doses=True)
            s[name + "_secondary"] = C.paired_slope_contrast(
                doses, P["data"]["B_readout_e"][x], P["data"]["B_readout_e"][y],
                n_boot=n_boot, seed=seed, block=blk, pair_doses=True)

        ns = (s.get("none") or {}).get("present_slope")
        s["blocked_fraction"] = {n: blocked_fraction(s.get(n), ns) for n in
                                 ("emo_all_vs_rand_all", "emo13_42_vs_rand_all")}
        s["detectable_effect"] = {f"{x}_vs_{y}": detectable_effect(s.get(f"{x}_vs_{y}"))
                                  for x, y in CONTRASTS}

        # ---- §4 manipulation checks ---------------------------------------------------
        s["mc_steer"] = _mc_steer(good, e, doses, n_boot, seed)
        s["mc_ablate"] = _mc_ablate(good, e, doses, arms)
        s["mc_text"] = _mc_text(good, e, doses)

        # ---- §5 decision inputs: numerator and denominator on ONE scenario panel -------
        Pd = panel(good, e, ("none", "emo_all", "rand_all"), doses,
                   metrics=("B_present_e",))
        if Pd is not None:
            d0 = Pd["data"]["B_present_e"]
            sc, sl_e = block_slopes(doses, d0["emo_all"], Pd["blocks"])
            _, sl_r = block_slopes(doses, d0["rand_all"], Pd["blocks"])
            _, sl_n = block_slopes(doses, d0["none"], Pd["blocks"])
            dec_in[e] = {"scen": sc, "dd": sl_e - sl_r, "nn": sl_n}
            s["decision_input"] = {"n_scenarios": int(len(sc)),
                                   "mean_slope_diff": float(sl_e.mean() - sl_r.mean()),
                                   "mean_none_slope": float(sl_n.mean())}
        s["_complete"] = bool(set(arms) <= set(have))
        per_emotion[e] = s

    # ---- testable emotions, sign-aware counts, BH-FDR (descriptive) --------------------
    testable, block_e, anti_e, null_e, pvals = [], [], [], [], {}
    rng = np.random.default_rng(seed)
    for e in emos:
        ns = (per_emotion[e].get("none") or {}).get("present_slope")
        if not ns or not ns.get("sig"):
            continue
        testable.append(e)
        c = per_emotion[e].get(DECISION_CONTRAST)
        if c and c.get("sig"):
            (block_e if c["diff"] < 0 else anti_e).append(e)
        else:
            null_e.append(e)
        if e in dec_in and len(dec_in[e]["dd"]) > 1:
            d = dec_in[e]["dd"]
            b = d[rng.integers(0, len(d), size=(max(n_boot, 200), len(d)))].mean(1)
            p = 2.0 * min(float((b <= 0).mean()), float((b >= 0).mean()))
            pvals[e] = float(min(1.0, max(p, 1.0 / len(b))))
    q = bh_fdr([pvals[e] for e in testable if e in pvals])
    fdr = dict(zip([e for e in testable if e in pvals], q))

    n_mc_fail = sum(1 for e in emos
                    if per_emotion[e].get("mc_ablate", {}).get("passes") is False)
    n_mc_fail_emo_all = sum(1 for e in emos
                            if per_emotion[e].get("mc_ablate", {}).get("passes_emo_all")
                            is False)

    decision = _decide(dec_in, testable, n_boot, seed)
    if n_mc_fail >= MC_ABLATE_FAIL_LIMIT:
        verdict = "instrument_failed"
    elif decision["median"] is None:
        verdict = "indeterminate"
    elif decision["median"] <= H1_MEDIAN and decision["ci"][1] <= H1_UPPER:
        verdict = "H1"
    elif decision["median"] >= H1P_MEDIAN and decision["ci"][0] >= H1P_LOWER:
        verdict = "H1prime"
    else:
        verdict = "indeterminate"

    counts = {"n_testable": len(testable), "n_untestable": len(emos) - len(testable),
              "n_block": len(block_e), "n_antiblock": len(anti_e), "n_null": len(null_e),
              "testable_emotions": testable, "blocking_emotions": block_e,
              "antiblocking_emotions": anti_e, "null_emotions": null_e,
              "n_mc_ablate_fail": n_mc_fail,
              "n_mc_ablate_fail_emo_all_only": n_mc_fail_emo_all,
              "n_mc_steer_pass": sum(1 for e in emos if (per_emotion[e].get("mc_steer")
                                                         or {}).get("sig")),
              "n_text_uninterpretable": sum(
                  1 for e in emos
                  if per_emotion[e].get("mc_text", {}).get("text_uninterpretable")),
              "bh_fdr_q": fdr, "p_values": pvals,
              "note": "counts are DESCRIPTIVE; the decision is the median rule (§5)"}
    return {"per_emotion": per_emotion, "counts": counts, "decision": decision,
            "verdict": verdict, "doses": doses, "emotions": list(emos),
            "arms": list(arms), "n_boot": n_boot,
            "quality": {"per_cell": {"/".join(map(str, k)): v for k, v in qual.items()},
                        "excluded_cells": [list(k) for k in excluded],
                        "n_excluded_cells": len(excluded),
                        "applied": bool(apply_quality_exclusion),
                        "thresholds": {"max_bad_frac": MAX_BAD_FRAC,
                                       "max_perplexity_ratio": MAX_PPL_RATIO}}}


def _first(rs, key):
    for r in rs:
        if r.get(key) is not None:
            return r[key]
    return None


def _mean_of(rs, key):
    v = [float(r[key]) for r in rs if r.get(key) is not None]
    return float(np.mean(v)) if v else None


def _sum_of(rs, key):
    v = [float(r[key]) for r in rs if r.get(key) is not None]
    return float(np.sum(v)) if v else None


def _mc_steer(rows, e, doses, n_boot, seed) -> dict | None:
    """§4.1 — did steering actually put the emotion into A's TEXT? Slope CI excludes 0."""
    P = panel(rows, e, ("none",), doses, metrics=("A_readout_e",))
    if P is None:
        return None
    out = C.slope_ci(doses, P["data"]["A_readout_e"]["none"], n_boot=n_boot, seed=seed,
                     block=P["blocks"])
    out["passes"] = bool(out["sig"] and out["slope"] > 0)
    return out


def _mc_ablate(rows, e, doses, arms) -> dict:
    """§4.2 — on A's span, at the top dose: `emo_all` must lower A's readout AT LEAST AS
    MUCH as `emo13_42`, and `rand_all` must lower it by < 0.02."""
    top = float(doses[-1])
    P = panel(rows, e, tuple(a for a in arms if a not in REWRITE_ARMS), doses,
              metrics=("ctx_readout_e",))
    mc = {"dose": top}
    if P is None:
        mc["passes"] = None
        return mc
    D = P["data"]["ctx_readout_e"]
    for arm in D:
        v = D[arm][top]
        mc[arm] = {"mean": float(np.nanmean(v)), "ci": C.ci_of(v), "n": int(len(v))}
    if "none" not in mc:
        mc["passes"] = None
        return mc
    drops = {arm: mc["none"]["mean"] - mc[arm]["mean"] for arm in D if arm != "none"}
    mc["drop_by_arm"] = drops
    if {"emo_all", "emo13_42", "rand_all"} <= set(drops):
        mc["passes_emo_all"] = bool(drops["emo_all"] >= drops["emo13_42"])
        mc["passes_rand"] = bool(drops["rand_all"] < RAND_DROP_MAX)
        mc["passes"] = bool(mc["passes_emo_all"] and mc["passes_rand"])
        mc["separated"] = bool(mc["emo_all"]["ci"][1] < mc["rand_all"]["ci"][0])
    else:
        mc["passes"] = None
    return mc


def _mc_text(rows, e, doses) -> dict:
    """§4.4 — A's readout on the NEUTRAL rewrite must be <= 0.5x its readout on the
    original; an emotion that fails has its `text` arm reported as UNINTERPRETABLE, not as
    evidence. `text_keep` is the affect-PRESERVING control, so the same threshold would be
    backwards for it: its ratio is reported descriptively and gates nothing.
    """
    top = float(doses[-1])
    out = {"dose": top, "threshold": REWRITE_READOUT_RATIO}
    for arm in REWRITE_ARMS:
        P = panel(rows, e, (arm,), doses, metrics=("A_readout_e", "ctx_readout_e"))
        if P is None:
            out[arm] = None
            continue
        orig = float(np.nanmean(P["data"]["A_readout_e"][arm][top]))
        rew = float(np.nanmean(P["data"]["ctx_readout_e"][arm][top]))
        out[arm] = {"readout_original": orig, "readout_rewrite": rew,
                    "ratio": float(rew / orig) if orig else None}
    t = out.get("text")
    out["text_uninterpretable"] = bool(
        t and t["ratio"] is not None and t["ratio"] > REWRITE_READOUT_RATIO)
    return out


def _decide(dec_in, testable, n_boot, seed) -> dict:
    """§5 DECISION: the median blocked fraction of `emo_all` vs `rand_all` over testable
    emotions, with a bootstrap CI OVER SCENARIOS.

    One scenario draw is shared by all emotions (the same 29 scenarios are run for every
    emotion), and numerator and denominator are recomputed inside each draw, so the CI is
    the CI of the ratio-of-means and not of a ratio of independent estimates.
    """
    use = [e for e in testable if e in dec_in and len(dec_in[e]["scen"]) > 1]
    out = {"median": None, "ci": [float("nan"), float("nan")], "n_emotions": len(use),
           "emotions": use, "per_emotion_fraction": {}, "n_scenarios": 0,
           "rule": f"H1 if median <= {H1_MEDIAN} and upper <= {H1_UPPER}; "
                   f"H1prime if median >= {H1P_MEDIAN} and lower >= {H1P_LOWER}; "
                   f"else indeterminate. instrument_failed overrides when MC-ablate "
                   f"fails for >= {MC_ABLATE_FAIL_LIMIT} emotions."}
    if not use:
        return out
    common = sorted(set.intersection(*[set(dec_in[e]["scen"].tolist()) for e in use]))
    if len(common) < 2:
        return out
    D, N = [], []
    for e in use:
        pos = {int(s): i for i, s in enumerate(dec_in[e]["scen"].tolist())}
        D.append([dec_in[e]["dd"][pos[int(s)]] for s in common])
        N.append([dec_in[e]["nn"][pos[int(s)]] for s in common])
    D, N = np.asarray(D, float), np.asarray(N, float)
    point = -D.mean(1) / np.where(np.abs(N.mean(1)) < 1e-12, np.nan, N.mean(1))
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(common), size=(n_boot, len(common)))
    num = -D[:, pick].mean(2)                       # [n_emotions, n_boot]
    den = N[:, pick].mean(2)
    frac = num / np.where(np.abs(den) < 1e-12, np.nan, den)
    med = np.nanmedian(frac, axis=0)
    med = med[np.isfinite(med)]
    lo, hi = (np.percentile(med, [2.5, 97.5]) if len(med) > 1
              else (float("nan"), float("nan")))
    out.update({"median": float(np.nanmedian(point)), "ci": [float(lo), float(hi)],
                "n_scenarios": len(common),
                "per_emotion_fraction": {e: float(p) for e, p in zip(use, point)}})
    return out


def print_summary(res):
    per, cnt, dec = res["per_emotion"], res["counts"], res["decision"]
    print(f"\n{'emotion':>10} {'MCabl':>6} {'MCstr':>6} {'d_all':>7} {'d_1342':>7} "
          f"{'d_rand':>7} {'emo_all-rand_all':>26} {'blockfrac':>10} {'det':>7} "
          f"{'txtOK':>6}")
    for e in res["emotions"]:
        s = per.get(e, {})
        mc, d = s.get("mc_ablate", {}), s.get("mc_ablate", {}).get("drop_by_arm", {})
        c = s.get(DECISION_CONTRAST)
        bf = s.get("blocked_fraction", {}).get(DECISION_CONTRAST, {}).get("point")
        de = s.get("detectable_effect", {}).get(DECISION_CONTRAST, {}).get("half_width")
        txt = (f"{c['diff']:+.3f} CI[{c['ci'][0]:+.3f},{c['ci'][1]:+.3f}]"
               if c else "--")
        print(f"{e:>10} {str(mc.get('passes')):>6} "
              f"{str((s.get('mc_steer') or {}).get('passes')):>6} "
              f"{_f(d.get('emo_all')):>7} {_f(d.get('emo13_42')):>7} "
              f"{_f(d.get('rand_all')):>7} {txt:>26} "
              f"{(f'{bf:+.3f}' if bf is not None else '--'):>10} "
              f"{_f(de):>7} "
              f"{str(not s.get('mc_text', {}).get('text_uninterpretable')):>6}")
    print(f"\ntestable {cnt['n_testable']}/{len(res['emotions'])} · blocking "
          f"{cnt['n_block']} · anti-blocking {cnt['n_antiblock']} · null {cnt['n_null']} · "
          f"MC-ablate failures {cnt['n_mc_ablate_fail']} · quality-excluded cells "
          f"{res['quality']['n_excluded_cells']}")
    m = dec["median"]
    print(f"DECISION: median blocked fraction (emo_all vs rand_all) over "
          f"{dec['n_emotions']} testable emotions = "
          f"{('%.3f' % m) if m is not None else '--'} "
          f"CI[{dec['ci'][0]:.3f},{dec['ci'][1]:.3f}] over {dec['n_scenarios']} scenarios")
    print(f"VERDICT: {res['verdict']}")


def _f(x, nd=3):
    return "--" if x is None or (isinstance(x, float) and not np.isfinite(x)) \
        else f"{x:+.{nd}f}"


# ---------------------------------------------------------------------------------------
# --selftest — synthetic rows with a planted answer (no model, no torch)
# ---------------------------------------------------------------------------------------

def _legacy_paired_slope_contrast(doses, a_per_sample, b_per_sample, n_boot=5000, seed=0):
    """A VERBATIM copy of `acl_core.paired_slope_contrast` as it stood BEFORE `block=` and
    `pair_doses=` were added. The selftest asserts the live default path still returns
    exactly this, so the backward-compatibility claim is checked rather than asserted."""
    doses = list(doses)
    A = [np.asarray(a_per_sample[d], float) for d in doses]
    B = [np.asarray(b_per_sample[d], float) for d in doses]
    sa = C.ols_slope(doses, [y.mean() for y in A])
    sb = C.ols_slope(doses, [y.mean() for y in B])
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        ma, mb = [], []
        for ya, yb in zip(A, B):
            n = min(len(ya), len(yb))
            ix = rng.integers(0, n, n)
            ma.append(ya[ix].mean()); mb.append(yb[ix].mean())
        draws.append(C.ols_slope(doses, ma) - C.ols_slope(doses, mb))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"slope_a": sa, "slope_b": sb, "diff": sa - sb,
            "ci": [float(lo), float(hi)], "sig": bool(lo > 0 or hi < 0),
            "n_per_dose": {str(d): int(min(len(x), len(y))) for d, x, y in zip(doses, A, B)}}


def _contrast_selftest(n_boot=1500) -> None:
    """Part A of the deliverable: the new keyword arguments of
    `acl_core.paired_slope_contrast`.

      (1) with the defaults the function is bit-identical to the pre-change one;
      (2) with scenario effects planted, the blocked+dose-paired path returns a CI at least
          as wide as the unblocked one — because the unblocked bootstrap treats the reps of
          one scenario as independent samples and so under-counts the clustered variance;
      (3) with NO scenario effect the two agree, point estimate exactly and CI closely,
          which is what makes (2) a statement about clustering and not about the code.
    """
    doses = list(DOSES)
    rng = np.random.default_rng(3)
    n_scen, reps = 24, 3

    # (1) default path == legacy path, exactly
    a = {d: rng.normal(0.4 * d, 0.3, 60) for d in doses}
    b = {d: rng.normal(0.2 * d, 0.3, 60) for d in doses}
    new = C.paired_slope_contrast(doses, a, b, n_boot=n_boot)
    old = _legacy_paired_slope_contrast(doses, a, b, n_boot=n_boot)
    assert new == old, ("default path is NOT bit-identical to the pre-change function",
                        new, old)
    print("[selftest] paired_slope_contrast defaults: bit-identical to the legacy body "
          f"(diff {new['diff']:+.4f}, CI {[round(x, 4) for x in new['ci']]})")

    def make(scen_slope_sd):
        """n_scen scenarios x reps samples per dose; arm A's slope exceeds arm B's by a
        per-scenario amount with sd `scen_slope_sd`, on top of iid per-sample noise."""
        blocks = np.repeat(np.arange(n_scen), reps)
        c = rng.normal(0.0, scen_slope_sd, n_scen)          # per-scenario slope difference
        A, B = {}, {}
        for d in doses:
            eps = rng.normal(0, 0.20, n_scen * reps)
            base = rng.normal(0, 0.50, n_scen * reps)       # cancels in the pairing
            B[d] = base + 0.5 * d
            A[d] = base + 0.5 * d + np.repeat(c, reps) * d + eps
        return A, B, blocks

    def width(x):
        return x["ci"][1] - x["ci"][0]

    A, B, blocks = make(0.45)
    unb = C.paired_slope_contrast(doses, A, B, n_boot=n_boot)
    blk = C.paired_slope_contrast(doses, A, B, n_boot=n_boot, block=blocks,
                                  pair_doses=True)
    assert blk["n_blocks"] == n_scen, blk
    assert set(blk) == set(unb) | {"n_blocks"}, (sorted(blk), sorted(unb))
    assert width(blk) >= width(unb), (
        "blocked CI must be at least as wide as the unblocked one when scenario effects "
        "are planted", width(blk), width(unb))
    print(f"[selftest] planted scenario slope effects: blocked CI width {width(blk):.4f} "
          f">= unblocked {width(unb):.4f} (ratio {width(blk)/width(unb):.2f}), "
          f"n_blocks {blk['n_blocks']}")

    A0, B0, blocks0 = make(0.0)
    unb0 = C.paired_slope_contrast(doses, A0, B0, n_boot=n_boot)
    blk0 = C.paired_slope_contrast(doses, A0, B0, n_boot=n_boot, block=blocks0,
                                   pair_doses=True)
    assert abs(blk0["diff"] - unb0["diff"]) < 1e-9, (blk0["diff"], unb0["diff"])
    rel = abs(width(blk0) - width(unb0)) / max(width(unb0), 1e-9)
    assert rel < 0.30, ("with no scenario effect the two schemes must agree", rel,
                        width(blk0), width(unb0))
    print(f"[selftest] no scenario effect: diff identical ({blk0['diff']:+.6f}), CI widths "
          f"agree to {rel*100:.0f}%")


def _synth_rows(doses, emos, arms, frac_by_emotion, n_scen=10, reps=2, seed=7,
                text_excluded=2, blocking_sd=0.0):
    """Rows with a KNOWN answer.

    `frac_by_emotion[e]` is the blocked fraction planted for `emo_all` vs `rand_all`: the
    none, rand_all and text_keep arms all get slope 1.0, `emo_all` gets 1 - f, so
    -(slope(emo_all) - slope(rand_all)) / slope(none) == f exactly.

    Every arm in one (emotion, dose, rep) cell shares its noise draw, so an arm with no
    planted effect is IDENTICAL to its control rather than merely similar and its contrast
    is deterministically null. A per-scenario slope offset (shared by all arms, and CENTRED
    so the none-arm slope is exactly 1.0) gives the scenario-blocked machinery real
    clustering to handle. The `text` arms drop `text_excluded` scenarios per row, which
    exercises the balanced-panel intersection.

    `blocking_sd` > 0 makes HOW MUCH is blocked itself vary by scenario (f -> f(1 + w_s),
    w centred), which is what a genuinely mixed result looks like: the median blocked
    fraction is still f, but its scenario bootstrap widens. With blocking_sd = 0 and an
    even three-versus-three split the median sits exactly at the midpoint of the two
    clusters and is estimated tightly, so §5's rule returns H1 rather than indeterminate --
    a property of the pre-registered statistic, which resamples scenarios and not emotions.
    """
    rng = np.random.default_rng(seed)
    # ctx_readout: none highest; emo_all drops most, emo13_42 less, rand_all barely
    read_top = {"none": 0.70, "emo13_42": 0.45, "emo_all": 0.30, "rand_all": 0.69,
                "text": 0.30, "text_keep": 0.66}
    scen_off = rng.normal(0, 0.25, n_scen)                  # per-scenario slope offset
    scen_off -= scen_off.mean()                             # -> none slope exactly 1.0
    # how blockable a scenario is, is a property of the SCENARIO, so the multiplier is
    # shared by all six emotions -- which is exactly why a scenario bootstrap can be wide
    w = rng.normal(0, blocking_sd, n_scen) if blocking_sd else np.zeros(n_scen)
    w -= w.mean()                                           # -> mean fraction exactly f
    rows = []
    for e in emos:
        f = float(frac_by_emotion[e])
        blocked = f * (1.0 + w)                             # per-scenario blocked amount
        slope = {a: 1.0 for a in arms}
        slope["emo_all"] = 1.0 - f
        for d in doses:
            for rep in range(reps):
                nz = rng.normal(0, 0.15, n_scen)            # shared across arms
                nzo = rng.normal(0, 0.15, n_scen)
                nzr = rng.normal(0, 0.02, n_scen)
                a_read = 0.2 + 0.5 * d + rng.normal(0, 0.05, n_scen)
                for arm in arms:
                    keep = list(range(n_scen))
                    if arm in REWRITE_ARMS and text_excluded:
                        drop = {(rep + i) % n_scen for i in range(text_excluded)}
                        keep = [i for i in keep if i not in drop]
                    k = np.asarray(keep)
                    base = read_top["none"] if d == 0 else read_top[arm]
                    sl = ((1.0 - blocked) if arm == "emo_all"
                          else np.full(n_scen, slope[arm]))
                    rows.append({
                        "emotion": e, "alpha": float(d), "rep": rep, "arm": arm,
                        "scenario": keep,
                        "B_present_e": ((sl + scen_off) * d + nz)[k].tolist(),
                        "B_other_e": (0.05 * d + nzo)[k].tolist(),
                        "A_readout_e": a_read[k].tolist(),
                        "ctx_readout_e": (base + nzr)[k].tolist(),
                        "B_readout_e": (0.15 + 0.30 * slope[arm] * d + nzr)[k].tolist(),
                        "ctx_readout_full": [1.0 / 6] * 6,
                        "removed_norm": 0.0 if arm in NO_HOOK_ARMS else 3.2,
                        "mask_hit_rate": None if arm in NO_HOOK_ARMS else 1.0,
                        "mask_n_positions": None if arm in NO_HOOK_ARMS else 46.0,
                        "n_layers_ablated": (0 if arm in NO_HOOK_ARMS else
                                             30 if arm == "emo13_42" else 51),
                        "degenerate_frac": 0.02, "refusal_frac": 0.0,
                        "distinct2": 0.9, "n_words": 40.0, "perplexity": 12.0,
                        "gen_seed": gen_seed(0, e, d, rep, arm, "B"),
                        "rand_dir_seed": (rand_dir_seed(0, e, rep)
                                          if arm == "rand_all" else None),
                        **({"text_excluded_n": n_scen - len(keep),
                            "text_excluded_frac": (n_scen - len(keep)) / n_scen,
                            "rewrite_degenerate_frac": 0.0,
                            "rewrite_refusal_frac": 0.0} if arm in REWRITE_ARMS else {}),
                    })
    return rows


def selftest() -> int:
    t0 = time.time()
    _contrast_selftest()

    # --- the pre-registered indeterminate zone: three emotions at 0.70, three at 0.20, ---
    # --- and the amount blocked varying by scenario, so the median's scenario bootstrap ---
    # --- is genuinely wide: median 0.45 <= 0.50 but the upper bound clears 0.65. --------
    plant = {e: (0.70 if i < 3 else 0.20) for i, e in enumerate(EMOS)}
    rows = _synth_rows(DOSES, EMOS, ARMS, plant, blocking_sd=1.3)
    print(f"\n[selftest] {len(rows)} synthetic arm-rows; planted blocked fraction "
          f"{ {e: plant[e] for e in EMOS} } (scenario sd 1.3x), text_keep == none",
          flush=True)
    # n_boot below the 5000 default only to keep this under 90 s on a CPU box; the planted
    # contrasts are 0 or a fixed fraction with a hair-width CI, so no significance call
    # here depends on the bootstrap size.
    res = analyze(rows, DOSES, EMOS, n_boot=500)
    print_summary(res)
    per, cnt, dec = res["per_emotion"], res["counts"], res["decision"]

    assert cnt["n_untestable"] == 0, cnt
    assert cnt["n_mc_ablate_fail"] == 0, cnt          # emo_all >= emo13_42, rand < 0.02
    assert cnt["n_mc_steer_pass"] == len(EMOS), cnt
    assert cnt["n_text_uninterpretable"] == 0, cnt    # 0.30 <= 0.5 * 0.70
    for e in EMOS:
        s = per[e]
        assert s["_complete"] and all(s.get(a) for a in ARMS), (e, s["_complete"])
        assert s["emo_all"]["n_layers_ablated"] == 51 and \
            s["emo13_42"]["n_layers_ablated"] == 30, e
        assert s["text"]["text_excluded_n"] and s["text_keep"]["text_excluded_n"], e
        bf = s["blocked_fraction"][DECISION_CONTRAST]
        assert bf["defined"] and abs(bf["point"] - plant[e]) < 0.06, (e, bf, plant[e])
        assert bf["denominator_ci"] is not None and all(np.isfinite(bf["range"])), (e, bf)
        de = s["detectable_effect"][DECISION_CONTRAST]["half_width"]
        assert de is not None and np.isfinite(de) and de > 0, (e, de)
        # text_keep was planted equal to none: it must not read as blocking
        assert s["text_keep_vs_none"]["sig"] is False, (e, s["text_keep_vs_none"])
        assert s["mc_ablate"]["passes"] is True and s["mc_ablate"]["separated"] is True, e
        for name in ("emo_all_vs_rand_all", "text_vs_text_keep"):
            assert s[name]["n_blocks"] > 1, (e, name, s[name])
    med = dec["median"]
    assert abs(med - 0.45) < 0.08, med                # median of {.7,.7,.7,.2,.2,.2}
    assert res["verdict"] == "indeterminate", (res["verdict"], dec)
    print(f"[selftest] indeterminate zone reproduced: median {med:.3f} "
          f"CI[{dec['ci'][0]:.3f},{dec['ci'][1]:.3f}] -> {res['verdict']}")

    # --- the rule must MOVE with the data: all three verdicts on a re-planted grid -------
    small = dict(n_scen=8, reps=2)
    h1 = analyze(_synth_rows(DOSES, EMOS, ARMS, {e: 0.15 for e in EMOS}, **small),
                 DOSES, EMOS, n_boot=300)
    assert h1["verdict"] == "H1", (h1["verdict"], h1["decision"])
    h1p = analyze(_synth_rows(DOSES, EMOS, ARMS, {e: 0.92 for e in EMOS}, **small),
                  DOSES, EMOS, n_boot=300)
    assert h1p["verdict"] == "H1prime", (h1p["verdict"], h1p["decision"])
    mid = analyze(_synth_rows(DOSES, EMOS, ARMS, {e: 0.65 for e in EMOS}, **small),
                  DOSES, EMOS, n_boot=300)
    assert mid["verdict"] == "indeterminate", (mid["verdict"], mid["decision"])
    print(f"[selftest] verdict moves H1 ({h1['decision']['median']:.2f}) / H1prime "
          f"({h1p['decision']['median']:.2f}) / indeterminate "
          f"({mid['decision']['median']:.2f}) with the planted fraction")

    # --- §4.2 instrument-failure override -----------------------------------------------
    bad = _synth_rows(DOSES, EMOS, ARMS, {e: 0.92 for e in EMOS}, **small)
    for r in bad:                                     # emo_all now drops LESS than emo13_42
        if r["arm"] == "emo_all" and float(r["alpha"]) > 0:
            r["ctx_readout_e"] = [0.60] * len(r["scenario"])
    binst = analyze(bad, DOSES, EMOS, n_boot=200)
    assert binst["verdict"] == "instrument_failed", (binst["verdict"], binst["counts"])
    assert binst["counts"]["n_mc_ablate_fail"] >= MC_ABLATE_FAIL_LIMIT, binst["counts"]
    print(f"[selftest] MC-ablate failure for {binst['counts']['n_mc_ablate_fail']} "
          f"emotions overrides a median of "
          f"{binst['decision']['median']:.2f} -> instrument_failed")

    print(f"[selftest] {time.time()-t0:.1f}s")
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

    pool_path = os.path.join(a.workdir, f"probe_{a.tag}.jsonl")
    prov = C.Provenance(
        script="src/rev3/b1c_alllayer.py",
        config={"doses": DOSES, "arms": ARMS, "emotions": EMOS, "reps": a.reps,
                "driver": "b1c", "estimator": a.estimator, "probe_k": PROBE_K,
                "stability_gate": STABILITY_GATE, "max_new_A": 110, "max_new_B": 110,
                "max_new_rewrite": 110, "temp": 0.9, "top_p": 0.95,
                "n_scenarios": len(C.SCENARIOS),
                "contrasts": [f"{x}_vs_{y}" for x, y in CONTRASTS],
                "decision_contrast": DECISION_CONTRAST,
                "decision_thresholds": {"H1_median_max": H1_MEDIAN,
                                        "H1_upper_max": H1_UPPER,
                                        "H1prime_median_min": H1P_MEDIAN,
                                        "H1prime_lower_min": H1P_LOWER,
                                        "mc_ablate_fail_limit": MC_ABLATE_FAIL_LIMIT},
                "text_quality": {"max_bad_frac": MAX_BAD_FRAC,
                                 "max_perplexity_ratio": MAX_PPL_RATIO,
                                 "rewrite_readout_ratio": REWRITE_READOUT_RATIO},
                "rewrite_prompts": REWRITE_PROMPTS,
                "gen_seed_rule": "crc32('seed|emotion|alpha|rep|arm|part') & 0x7fffffff, "
                                 "+ batch start index; `arm` is EMPTY for part 'B' so all "
                                 "six arms sample B's reply with common random numbers, "
                                 "and kept for parts 'readout', 'readout_B' and 'rewrite'",
                "rand_dir_seed_rule": "crc32('rand|seed|EMOTIONS.index(e)|rep') & "
                                      "0x7fffffff; one fresh norm-matched direction per "
                                      "(emotion, rep) AND per ablated layer",
                "prereg": prereg_stamp(),
                "probe_pool_path": pool_path,
                "predecessors": ["results/rev3/b1_e4rerun_qwen36-27b.json",
                                 "results/rev3/b1_followup_qwen36-27b.json"]},
        model_id=a.model, model_revision="", seeds={"pool": a.seed, "run": a.seed},
        code_sha=C.code_hash(os.path.abspath(__file__), os.path.abspath(C.__file__)),
        control_pointers={
            # Every span below is read out of THIS file's AST at import time by fn_lines /
            # block_lines — never typed, so a pointer cannot drift away from the code it
            # names; a name that stops existing renders as "NOT FOUND — pointer is stale".
            "per_layer_gate": f"b1c_alllayer.py main ({block_lines('GATE')}): "
                              f"acl_core.split_half at EVERY ablated hidden_states index "
                              f"(13..n_layers-1) plus the focus layer, n_per_half="
                              f"min(N//2,600), 10 seeds; the run EXITS without generating "
                              f"anything if any ablated layer's class mean is below "
                              f"{STABILITY_GATE} (PREREG §3 'the run proceeds only if')",
            "all_layer_ablation": f"b1c_alllayer.py main ({block_lines('LAYERS')}) and "
                                  f"::dirs_for ({fn_lines('dirs_for')}): ONE acl_core."
                                  f"Ablate over decoder blocks 12..n_layers-2 (hs "
                                  f"13..n_layers-1); the per-arm `dirs` dict holds only "
                                  f"that arm's layers and Ablate._mk leaves a hidden state "
                                  f"untouched when `dirs.get(hs_index)` is None, so "
                                  f"emo13_42 and emo_all/rand_all share one hook set",
            "random_control": f"b1c_alllayer.py::rand_dirs_for "
                              f"({fn_lines('rand_dirs_for')}) -> acl_core.random_dir_like, "
                              f"a FRESH draw per (emotion, rep) AND per ablated layer, "
                              f"seeded by ::rand_dir_seed ({fn_lines('rand_dir_seed')}); "
                              f"the seed is in every row as rand_dir_seed and the norm "
                              f"actually removed as removed_norm (read it with "
                              f"mask_n_positions and n_layers_ablated)",
            "text_exclusion": f"b1c_alllayer.py::rewrite_A ({fn_lines('rewrite_A')}): a "
                              f"rewrite whose acl_core.degeneracy refusal flag is set, or "
                              f"that has fewer than 3 words, or that is identical to the "
                              f"original, DROPS that scenario from the arm-row (the "
                              f"`scenario` id list shrinks with it) and is counted in "
                              f"text_excluded_n / text_excluded_frac",
            "text_keep_control": f"b1c_alllayer.py REWRITE_KEEP_PROMPT + ::rewrite_A "
                                 f"({fn_lines('rewrite_A')}): the same pipeline and the "
                                 f"same exclusion rules with the affect-PRESERVING "
                                 f"instruction, so `text` minus `text_keep` separates "
                                 f"'affect removed' from 'context changed'",
            "blocked_paired_contrast": f"acl_core.paired_slope_contrast(block=..., "
                                       f"pair_doses=True) -> acl_core."
                                       f"_blocked_paired_slope_contrast, called from "
                                       f"b1c_alllayer.py::analyze ({fn_lines('analyze')}) "
                                       f"on the balanced scenario panel built by ::panel "
                                       f"({fn_lines('panel')}); one-sample bootstrap over "
                                       f"scenarios of the per-scenario paired slope "
                                       f"difference, 5000 draws, seed 0",
            "decision_rule": f"b1c_alllayer.py::_decide ({fn_lines('_decide')}), called "
                             f"from ::analyze ({fn_lines('analyze')}); the median blocked "
                             f"fraction of {DECISION_CONTRAST} over testable emotions with "
                             f"a bootstrap CI over scenarios, thresholds "
                             f"{H1_MEDIAN}/{H1_UPPER}/{H1P_MEDIAN}/{H1P_LOWER} fixed in "
                             f"this file's module docstring and in PREREG_B1c §5 before "
                             f"the run; ::blocked_fraction "
                             f"({fn_lines('blocked_fraction')}) is sign-aware",
            "secondary_outcome": f"b1c_alllayer.py::readout ({fn_lines('readout')}) with "
                                 f"who='B' and the reply appended to the conversation, "
                                 f"hooks OFF — stored per row as B_readout_e",
            "manipulation_checks": f"b1c_alllayer.py::_mc_steer ({fn_lines('_mc_steer')}), "
                                   f"::_mc_ablate ({fn_lines('_mc_ablate')}), ::_mc_text "
                                   f"({fn_lines('_mc_text')}) and ::quality_flags "
                                   f"({fn_lines('quality_flags')}) — PREREG §4.1-§4.4",
            "reproducible_sampling": f"b1c_alllayer.py::gen_seed ({fn_lines('gen_seed')}), "
                                     f"called immediately before every h.model.generate in "
                                     f"::gen_B ({fn_lines('gen_B')}), ::rewrite_A "
                                     f"({fn_lines('rewrite_A')}) and before the readout "
                                     f"forward in ::readout ({fn_lines('readout')})",
            "non_circular_measurement": f"b1c_alllayer.py main ({block_lines('POOL')}): "
                                        f"probe fitted on the READ half, steering and "
                                        f"ablation directions on the disjoint DIR half",
            "frozen_token_audit": "arms none/emo13_42/emo_all/rand_all regenerate B under "
                                  "the intervention while A's message predates it; arms "
                                  "text/text_keep regenerate A's message too. See "
                                  "out['frozen_token_audit'].",
        })

    h = C.load(a.model)
    prov.model_revision = h.revision

    # <<LAYERS_START>>
    # ---- the layer sets (PREREG §3: emo13_42 is hs 13-42, emo_all/rand_all hs 13-63) ----
    # `lo` and the mid window are built exactly as b1_followup.py built them, so emo13_42
    # is the follow-up's `emo` arm and not a near-miss. The all-layer set runs to
    # n_layers-1: hs n_layers is the output of the LAST block, which feeds only the final
    # norm and the LM head, so projecting there cannot change what B attends to — it would
    # be a no-op arm dressed as an ablation.
    focus = h.focus(); steer_layer = focus - 1
    lo = round(0.2 * h.n_layers)
    abl_hs_mid = list(range(lo, focus))                     # 13..42 at n_layers=64
    abl_hs_all = list(range(lo, h.n_layers))                # 13..63 at n_layers=64
    abl_dec_all = [L - 1 for L in abl_hs_all]               # decoder blocks 12..62
    hs_needed = sorted(set(abl_hs_all) | {focus})
    # <<LAYERS_END>>
    print(f"[B1C] focus {focus} steer_layer {steer_layer} n_layers {h.n_layers}; "
          f"emo13_42 hs {abl_hs_mid[0]}..{abl_hs_mid[-1]} ({len(abl_hs_mid)} blocks), "
          f"emo_all/rand_all hs {abl_hs_all[0]}..{abl_hs_all[-1]} "
          f"({len(abl_hs_all)} blocks)", flush=True)

    # <<POOL_START>>
    # ---------------- probe pool, split into DIR and READ halves ------------------------
    # Same path, same k, same seed, same permutation as b1_e4rerun.py and b1_followup.py:
    # with the pool file from those runs present in --workdir the directions are the SAME
    # vectors, which is what makes emo13_42 a replication rather than a new measurement.
    items = C.generate_pool(h, PROBE_K, pool_path, seed=a.seed, bs=a.gen_bs)
    feats, yp, yo = C.pool_features(h, items, hs_needed,
                                    cache=os.path.join(a.workdir, f"probefeat_{a.tag}.npz"))
    N = len(items)
    rs = np.random.default_rng(a.seed).permutation(N)
    DIR, READ = rs[:N // 2], rs[N // 2:]
    pool_sha = file_sha256(pool_path)
    print(f"[B1C] probe pool n={N}: DIR half {len(DIR)}, READ half {len(READ)}; "
          f"sha256 {pool_sha[:16]}...", flush=True)

    dir_dec = {L: C.fit_direction(feats[L][DIR], yp[DIR], a.estimator, seed=a.seed,
                                  pair_on=yo[DIR]) for L in hs_needed}
    read_dec = {L: C.fit_direction(feats[L][READ], yp[READ], a.estimator, seed=a.seed,
                                   pair_on=yo[READ]) for L in hs_needed}
    read_dec_other = C.fit_direction(feats[focus][READ], yo[READ], a.estimator, seed=a.seed)
    # <<POOL_END>>

    # <<GATE_START>>
    # ---- GATE (PREREG §3): split-half at EVERY ablated layer, 13..n_layers-1 ------------
    # "the run proceeds only if the class mean is >= 0.80 at every ablated layer, and every
    # per-class value is reported". So a failure here is FATAL, not a warning: the gate
    # results are written and the process exits before a single generation is spent.
    stab_by_layer = {}
    for L in hs_needed:
        stab_by_layer[L] = C.split_half(feats[L], yp, a.estimator,
                                        n_per_half=min(N // 2, 600), seeds=range(10),
                                        pair_on=yo)
        print(f"[B1C gate] hs {L:>3} split-half({a.estimator}) "
              f"{stab_by_layer[L]['mean']:.3f} CI {stab_by_layer[L]['ci']} "
              f"per-class {[round(x, 3) for x in stab_by_layer[L]['per_class']]}",
              flush=True)
    abl_means = [stab_by_layer[L]["mean"] for L in abl_hs_all]
    layers_below_gate = [L for L in abl_hs_all
                         if not (stab_by_layer[L]["mean"] >= STABILITY_GATE)]
    stab = stab_by_layer[focus]
    stab_logreg = C.split_half(feats[focus], yp, "logreg", n_per_half=min(N // 2, 600),
                               seeds=range(10))
    gate_pass = not layers_below_gate
    gate_block = {
        "gate": STABILITY_GATE, "gate_pass": gate_pass,
        "gated_layers": abl_hs_all, "layers_below_gate": layers_below_gate,
        "n_per_half": min(N // 2, 600), "n_seeds": 10,
        "by_layer": {str(L): stab_by_layer[L] for L in hs_needed},
        "per_class_by_layer": {str(L): stab_by_layer[L]["per_class"] for L in hs_needed},
        "ablated_layer_summary": {"min": float(np.min(abl_means)),
                                  "median": float(np.median(abl_means)),
                                  "max": float(np.max(abl_means))},
        "focus": {"layer": focus, "used": stab, "logreg_for_comparison": stab_logreg},
    }
    print(f"[B1C GATE] hs {abl_hs_all[0]}..{abl_hs_all[-1]}: min {np.min(abl_means):.3f} "
          f"median {np.median(abl_means):.3f} max {np.max(abl_means):.3f}; "
          f"{len(layers_below_gate)} layer(s) below the {STABILITY_GATE} gate: "
          f"{layers_below_gate}", flush=True)
    if not gate_pass:
        C.write_result(os.path.join(a.outdir, f"b1c_alllayer_{a.tag}.json"),
                       {"model": a.model, "tag": a.tag, "aborted": "gate",
                        "focus": focus, "n_layers": h.n_layers,
                        "abl_hs_all": abl_hs_all, "abl_hs_mid": abl_hs_mid,
                        "probe_n": N, "probe_pool_sha256": pool_sha,
                        "direction_stability": gate_block,
                        "verdict": "gate_failed", "rows": []}, prov)
        print(f"[B1C GATE] *** ABORTING: {len(layers_below_gate)} ablated layer(s) below "
              f"the pre-registered {STABILITY_GATE} split-half gate "
              f"({layers_below_gate}). PREREG_B1c §3 says the run proceeds ONLY if every "
              f"ablated layer clears it, so no generation was performed. The per-layer "
              f"gate results are in {a.outdir}/b1c_alllayer_{a.tag}.json ***", flush=True)
        print("B1C_GATE_FAILED", flush=True)
        sys.exit(2)
    # <<GATE_END>>

    # ---------------- directions ---------------------------------------------------------
    def tt(v):
        return torch.tensor(v, dtype=h.model.dtype, device=h.model.device)

    emo_dirs_all, emo_dirs_mid, steer_dir = {}, {}, {}
    for e in EMOS:
        ei = C.EMOTIONS.index(e)
        emo_dirs_all[e] = {L: tt(C.raw_direction(dir_dec[L], ei)) for L in abl_hs_all}
        emo_dirs_mid[e] = {L: emo_dirs_all[e][L] for L in abl_hs_mid}
        steer_dir[e] = tt(C.raw_direction(dir_dec[focus], ei))

    _rand_cache = {}

    def rand_dirs_for(e, rep):
        """A FRESH norm-matched random direction per (emotion, rep) AND per ablated layer.

        One generator per (emotion, rep) draws `len(abl_hs_all)` independent unit vectors,
        so `rand_all` is not one vector reused down the stack — it is the same number of
        independent directions the emotion arm removes, which is what makes the two
        comparable on `removed_norm`.
        """
        key = (e, rep)
        if key not in _rand_cache:
            s = rand_dir_seed(a.seed, e, rep)
            rr = np.random.default_rng(s)
            _rand_cache[key] = ({L: tt(C.random_dir_like(C.raw_direction(dir_dec[L], 0), rr))
                                 for L in abl_hs_all}, s)
        return _rand_cache[key]

    steer = C.Steer(h, steer_layer)
    # ONE Ablate over 13..n_layers-1; the per-arm `dirs` dict decides which of those hooks
    # actually project (Ablate._mk returns the hidden state unchanged when the layer is
    # missing from `dirs`), so emo13_42 and emo_all differ only in that dict.
    ab = C.Ablate(h, abl_dec_all)
    rms = float(np.linalg.norm(feats[focus], axis=1).mean())
    print(f"[B1C] rms {rms:.1f}; one Ablate over {len(abl_dec_all)} decoder blocks",
          flush=True)

    # ---------------- the sweep ----------------------------------------------------------
    S = C.SCENARIOS
    rlog = C.ResponseLog(os.path.join(a.workdir, f"responses_b1c_{a.tag}.jsonl"))
    ck = C.Checkpoint(os.path.join(a.workdir, f"b1c_cells_{a.tag}.json"),
                      {"doses": DOSES, "arms": ARMS, "emos": EMOS, "reps": a.reps,
                       "est": a.estimator, "n": N, "focus": focus, "probe_k": PROBE_K,
                       "abl_hs_all": abl_hs_all, "abl_hs_mid": abl_hs_mid,
                       "driver": "b1c"})

    def gen_A(e, alpha, rep):
        """A's steered message. Unchanged from b1_followup.py / b1_e4rerun.py, seed
        included, so A's half of the stimulus stays comparable across the three runs."""
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
        """Directions per arm — the ONLY place the arms differ mechanically.

        'none', 'text' and 'text_keep' install no hook at all. 'emo13_42' passes a dirs
        dict holding hs 13..focus-1, 'emo_all' one holding hs 13..n_layers-1, 'rand_all'
        the fresh random draws over the same all-layer set. Ablate leaves any hidden state
        that is not a key of `dirs` untouched.
        """
        if arm in NO_HOOK_ARMS:
            return None, None
        if arm == "rand_all":
            return rand_dirs_for(e, rep)
        if arm == "emo13_42":
            return emo_dirs_mid[e], None
        return emo_dirs_all[e], None                   # 'emo_all'

    def gen_B(ctx, arm, e, alpha, rep, bs=30):
        """B regenerates while the arm's direction is projected out of A's TOKEN SPAN
        during B's PREFILL. Seeded per batch; part 'B' shares one stream across arms.

        Returns removed_norm AND mask_n_positions, because removed_norm is a mean over the
        masked positions and over the ablated LAYERS: emo13_42 averages it over 30 layers
        and emo_all over 51, so the number is uninterpretable without both counts.
        mask_hit_rate is None for the un-hooked arms rather than a 1.0 that would read as
        "the span was found" when nothing was ever looked for.
        """
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

    def readout(ctx, arm, e, alpha, rep, who="A", replies=None, bs=30):
        """The forced-choice readout: from the model's OWN next-token distribution, which
        of the six emotions is `who` feeling? Returns [N, 6] probabilities.

        who='A' is the §4.2 manipulation check and runs with the arm's ablation LIVE on A's
        span. who='B' is the §4 SECONDARY OUTCOME: B's generated reply is appended to the
        conversation, the question asks about B, and NO hooks are installed — the outcome
        is read off a clean forward pass, exactly as the primary one is.

        Deterministic (a single forward), but seeded anyway so the call is reproducible
        under any stochastic kernel.
        """
        dirs = None if who == "B" else dirs_for(arm, e, rep)[0]
        base = gen_seed(a.seed, e, alpha, rep, arm,
                        "readout" if who == "A" else "readout_B")
        h.tok.padding_side = "left"
        ids = []
        for em in C.EMOTIONS:
            cand = [h.tok.encode(x, add_special_tokens=False)
                    for x in (em, " " + em, em.capitalize())]
            ids.append(sorted({c[0] for c in cand if c}))
        convs = []
        for i, c in enumerate(ctx):
            if who == "B":
                B = c["B"]
                br = replies[i]
                br = br if br.startswith(f"{B}:") else f"{B}: {br}"
                convs.append((f"{c['conv']}\n{br}", B))
            else:
                convs.append((c["conv"], c["A"]))
        res = []
        for i in range(0, len(convs), bs):
            batch = convs[i:i + bs]
            texts = [C.chat(h, f"{cv}\n\n{C.READOUT_Q.format(A=w)}") for cv, w in batch]
            enc = h.tok(texts, return_tensors="pt", padding=True, return_offsets_mapping=True)
            offs = enc.pop("offset_mapping")
            enc = {k: v.to(h.model.device) for k, v in enc.items()}
            if dirs is not None:
                M, _ = mask_for(h, texts, [c["ar"] for c in ctx[i:i + bs]], enc, offs)
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

    def rewrite_A(ctx, arm, e, alpha, rep):
        """TOKEN-LEVEL arms: replace A's steered line with a rewrite by the UNSTEERED,
        un-ablated model, then let B answer that with NO hooks at all.

        arm 'text'      -> REWRITE_PROMPT, neutral wording, the affect REMOVED.
        arm 'text_keep' -> REWRITE_KEEP_PROMPT, tone and emotional state kept and every
                           specific detail changed — the control that separates "affect
                           removed" from "context changed".

        PREREG §3/§4.3: a rewrite that is a REFUSAL (acl_core.degeneracy's refusal flag),
        EMPTY (fewer than three words) or IDENTICAL to the original is EXCLUDED — the
        scenario is dropped from this arm-row entirely, including from its `scenario` id
        list, so the row never silently contains a copy of the `none` arm. b1_followup kept
        those scenarios and only counted them; this run drops them, because a copy of the
        none arm inside the text arm biases the text contrast toward zero.
        """
        base = gen_seed(a.seed, e, alpha, rep, arm, "rewrite")
        ab.clear(); steer.off()
        prompts = [REWRITE_PROMPTS[arm].format(ar=c["ar"], A=c["A"]) for c in ctx]
        outs = C.gen(h, prompts, max_new=110, bs=a.gen_bs, seed=base, first_line=True,
                     log=rlog, tag=f"REWRITE_{arm}_{e}_a{alpha}_r{rep}")
        new, keep_idx, texts = [], [], []
        n_ref = n_empty = n_same = 0
        for i, (c, pr) in enumerate(zip(ctx, outs)):
            A = c["A"]
            raw = pr.strip()
            body = raw[len(A) + 1:].strip() if raw.startswith(f"{A}:") else raw
            orig = c["ar"]
            obody = orig[len(A) + 1:].strip() if orig.startswith(f"{A}:") else orig.strip()
            d = C.degeneracy(body)
            refusal, empty = bool(d["refusal"]), len(body.split()) < 3
            same = body == obody
            n_ref += refusal; n_empty += empty; n_same += same
            if refusal or empty or same:
                continue
            ar = f"{A}: {body}"
            texts.append(body)
            keep_idx.append(i)
            c2 = dict(c)
            c2["ar"] = ar
            c2["conv"] = f"{c['setting']}\n{c['bopen']}\n{ar}"
            c2["bprompt"] = (f"{c2['conv']}\nWrite only {c['B']}'s next reply as one short "
                             f"paragraph, starting with '{c['B']}:'.")
            new.append(c2)
        n_ex = len(ctx) - len(new)
        deg = [C.degeneracy(t) for t in texts]
        stats = {"text_excluded_n": int(n_ex),
                 "text_excluded_frac": float(n_ex / max(len(ctx), 1)),
                 "text_excluded_refusal": int(n_ref),
                 "text_excluded_empty": int(n_empty),
                 "text_excluded_identical": int(n_same),
                 "text_kept_n": int(len(new)),
                 "rewrite_degenerate_frac": float(np.mean([C.is_degenerate(x)
                                                           for x in deg])) if deg else None,
                 "rewrite_refusal_frac": float(np.mean([x["refusal"]
                                                        for x in deg])) if deg else None,
                 "rewrite_n_words": float(np.mean([x["n_words"]
                                                   for x in deg])) if deg else None}
        return new, keep_idx, stats

    def score_B(ctx, breps, bs=8):
        """PRIMARY OUTCOME: B's present-e and other-e on its own reply, under the HELD-OUT
        read probe, in a separate forward pass with ALL HOOKS OFF — which is what makes it
        valid under an all-layer ablation (PREREG §4)."""
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
                    actx, extra, keep = ctx, {}, list(range(len(ctx)))
                    if arm in REWRITE_ARMS:
                        actx, keep, extra = rewrite_A(ctx, arm, e, alpha, rep)
                    if not actx:
                        print(f"[{e:>9} a{alpha} r{rep} {arm:>9}] EVERY rewrite excluded "
                              f"— empty arm-row", flush=True)
                        cell.append({"emotion": e, "alpha": alpha, "rep": rep, "arm": arm,
                                     "scenario": [], "B_present_e": [], "B_other_e": [],
                                     "A_readout_e": [], "ctx_readout_e": [],
                                     "B_readout_e": [], **extra})
                        continue
                    breps, removed, hit, gseed, npos = gen_B(actx, arm, e, alpha, rep)
                    pres, oth = score_B(actx, breps)
                    # MC-ablate: is the affect still readable from the intervened context?
                    ab_read = readout(actx, arm, e, alpha, rep, who="A")
                    # SECONDARY outcome: the same question about B, on B's reply, hooks off
                    b_read = readout(actx, arm, e, alpha, rep, who="B", replies=breps)
                    deg = [C.degeneracy(b) for b in breps]
                    ppl = C.perplexity(h, breps)
                    _d, rseed = dirs_for(arm, e, rep)
                    cell.append({
                        "emotion": e, "alpha": alpha, "rep": rep, "arm": arm,
                        "scenario": [c["scenario"] for c in actx],
                        "B_present_e": pres[:, ei].tolist(),
                        "B_other_e": oth[:, ei].tolist(),
                        "A_readout_e": a_read[keep, ei].tolist(),
                        "ctx_readout_e": ab_read[:, ei].tolist(),
                        "ctx_readout_full": ab_read.mean(0).tolist(),
                        "B_readout_e": b_read[:, ei].tolist(),
                        "B_readout_full": b_read.mean(0).tolist(),
                        "removed_norm": removed, "mask_hit_rate": hit,
                        "mask_n_positions": npos,
                        "n_layers_ablated": (0 if arm in NO_HOOK_ARMS
                                             else len(_d) if _d else 0),
                        "degenerate_frac": float(np.mean([C.is_degenerate(d) for d in deg])),
                        "refusal_frac": float(np.mean([d["refusal"] for d in deg])),
                        "distinct2": float(np.mean([d["distinct2"] for d in deg])),
                        "n_words": float(np.mean([d["n_words"] for d in deg])),
                        "perplexity": float(np.median(ppl)),
                        "gen_seed": gseed, "rand_dir_seed": rseed,
                        **extra,
                    })
                    r = cell[-1]
                    print(f"[{e:>9} a{alpha} r{rep} {arm:>9}] present "
                          f"{np.mean(r['B_present_e']):+.2f} B_readout "
                          f"{np.mean(r['B_readout_e']):.3f} ctx_readout "
                          f"{np.mean(r['ctx_readout_e']):.3f} removed {removed:.2f} "
                          f"over {'--' if npos is None else f'{npos:.0f}'} pos x "
                          f"{r['n_layers_ablated']} layers deg "
                          f"{r['degenerate_frac']:.2f} ppl {r['perplexity']:.1f} "
                          f"seed {gseed}"
                          + (f" excluded {r['text_excluded_n']}"
                             if arm in REWRITE_ARMS else ""), flush=True)
                ck.put(cid, cell)
                rows.extend(cell)
                print(f"[B1C] {len(rows)} arm-rows, {time.time()-t_start:.0f}s elapsed",
                      flush=True)

    # ---------------- analysis -----------------------------------------------------------
    res = analyze(rows, DOSES, EMOS)
    print_summary(res)

    out = {
        "model": a.model, "tag": a.tag, "focus": focus, "steer_layer": steer_layer,
        "n_layers": h.n_layers,
        "abl_hs_mid": abl_hs_mid, "abl_hs_all": abl_hs_all,
        "n_abl_blocks_mid": len(abl_hs_mid), "n_abl_blocks_all": len(abl_hs_all),
        "probe_n": N, "dir_half_n": len(DIR), "read_half_n": len(READ),
        "probe_pool_path": pool_path, "probe_pool_sha256": pool_sha,
        "estimator": a.estimator, "rms": rms, "doses": list(DOSES), "arms": list(ARMS),
        "prereg": prereg_stamp(),
        "predecessors": ["results/rev3/b1_e4rerun_qwen36-27b.json",
                         "results/rev3/b1_followup_qwen36-27b.json"],
        "direction_stability": gate_block,
        "frozen_token_audit": C.frozen_token_audit({
            "generated": ["B's reply (regenerated under the intervention, arms "
                          "emo13_42/emo_all/rand_all)",
                          "the readout answer token on A's span (produced with the "
                          "intervention live)",
                          "the readout answer token on B's reply (hooks OFF)",
                          "A's rewritten message and B's reply to it (arms text and "
                          "text_keep, both produced with NO hooks)"],
            "frozen": ["A's steered message (generated BEFORE any ablation; rewritten, not "
                       "ablated, in arms text/text_keep)",
                       "the scenario setting and B's opening line (fixed stimuli)"]}),
        "verdict": res["verdict"], "decision": res["decision"], "counts": res["counts"],
        "quality": res["quality"], "summary": res["per_emotion"], "rows": rows,
    }
    C.write_result(os.path.join(a.outdir, f"b1c_alllayer_{a.tag}.json"), out, prov)
    d = res["decision"]
    print(f"\n[B1C] VERDICT: {res['verdict']}  (median blocked fraction "
          f"{d['median'] if d['median'] is None else round(d['median'], 3)} "
          f"CI[{d['ci'][0]:.3f},{d['ci'][1]:.3f}] over {d['n_emotions']} testable "
          f"emotions; MC-ablate failures {res['counts']['n_mc_ablate_fail']})", flush=True)
    print("B1C_DONE", flush=True)


if __name__ == "__main__":
    main()

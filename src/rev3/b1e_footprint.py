#!/usr/bin/env python3
"""B1e — FOOTPRINT-MATCHED RANK-1 control: is B1c's rank-1 blocking emotion-specific, or is
it what removing *any* rank-1 direction of that size from A's span does?
(docs/planning/PREREG_B1e.md, committed before this file existed.)

This driver implements that pre-registration and nothing else. Where the pre-registration
says a quantity is fixed in advance, it is written here as a constant, not chosen at run
time. It is a copy of `b1d_subspace.py` with the rank-5 machinery removed (subspace,
permuted frame, random frame, the DIR/READ subspace gate) and two rank-1 controls put in
its place: a permuted-label `dom` direction and the top principal direction of the DIR
half. Everything else — pool, DIR/READ split, per-layer rank-1 gate, doses, reps,
scenarios, common random numbers for B, checkpointing, quality flags, panel machinery,
blocked and dose-paired contrasts — is B1c/B1d's, unchanged.

HYPOTHESES, stated before the data (PREREG_B1e §2, verbatim; ASCII transliteration of
the two non-ASCII glyphs, "-" for the minus sign and "H3'" for H3-prime)

  Pre-specified emotions: **afraid and sad** (B1c's two blockers; report 22). All six are
  run and reported; the decision uses these two.

  - **H3 (specific):** for both afraid and sad, the blocked, dose-paired contrast `emo_all
    - permdir_all` is significantly negative and its blocked fraction (relative to the
    `none` slope) is >= 0.10.
  - **H3' (footprint):** for neither afraid nor sad is `emo_all - permdir_all`
    significantly negative, while `permdir_all - none` is significantly negative for both
    (the permuted direction blocks as the emotion direction did).
  - **indeterminate** otherwise (one of the two, or neither blocks at all).
  - **instrument_failed:** MC-steer fails for either pre-specified emotion, or
    `permdir_all` shifts the dose-0 A-span readout by more than 0.15 from the `none` arm
    for either (the control then disturbs the readout the way B1d's subspace did and the
    comparison is not clean).

  The verdict string is one of: "H3", "H3prime", "indeterminate", "instrument_failed".

DECISION RULE, fixed in advance (PREREG_B1e §2 and §5)

  - Per-arm dose-response slope: OLS over the four doses on per-scenario means; CI by
    **scenario-blocked** bootstrap (29 clusters).
  - Contrasts, scenario-blocked AND dose-paired
    (`acl_core.paired_slope_contrast(block=..., pair_doses=True)`, 5000 draws), in this
    order: `emo_all - permdir_all` (decision), `emo_all - pc1_all`, `permdir_all - none`,
    `pc1_all - none`, `emo_all - none`.
  - **Blocked fraction** = -(arm - control contrast) / `none` slope, sign-aware, with the
    CI-implied range; the denominator's own CI is reported next to it.
  - **Decision:** the PER-EMOTION rule of §2 applied to the two PRE-SPECIFIED emotions
    (afraid, sad) only. Unlike B1d there is no median-over-emotions rule: §2 names the two
    emotions in advance and asks a yes/no question of each.
  - **Testable emotions** (`none` slope CI excludes zero), the sign-aware blocking /
    anti-blocking / null counts and BH-FDR over the testable emotions for the decision
    contrast are DESCRIPTIVE and reported for all six; they do not override §2's rule.

ARMS (PREREG_B1e §3; four, all rank 1 — the whole point is that the three hooked arms
remove objects of the SAME rank at the SAME layers and differ only in what the direction
knows about emotion)

  none          no ablation                                                    baseline
  emo_all       B1c's rank-1 `dom` emotion direction out of A's span,           the bridge to
                hs 13..n-1                                                      reports 22/27
  permdir_all   the SAME `dom` estimator on the SAME DIR half with the          footprint-
                present-emotion labels PERMUTED, the class-index direction      matched
                for the steered emotion, unit norm, same layers; ONE            control
                permutation per (emotion, rep)                                  (the decision)
  pc1_all       the top principal direction of the centred DIR-half features    largest-
                at each layer, unit norm, THE SAME for every emotion            footprint
                                                                                control

  Only `none` installs no hook. B's replies are seeded with a common seed across arms
  (b1_followup's rule), so shared sampling noise cancels inside each paired arm contrast.

DIRECTIONS (§3), all unit-norm 1-D vectors in RAW activation space, per hidden state L in
13..n_layers-1, all fitted on the DIR half only
  emo_all      `acl_core.raw_direction(fit_direction(feats[L][DIR], yp[DIR], 'dom'), i_e)`
               — exactly B1c's and B1d's object, so `emo_all` is the bridge arm.
  permdir_all  the same two calls with `yp[DIR]` replaced by ONE permutation of it drawn
               per (emotion, rep) from `perm_seed`, re-used at every ablated layer.
  pc1_all      the top right singular vector of the MEAN-CENTRED DIR-half features, sign
               fixed so the largest-magnitude component is positive.
  The cosine of pc1 against each emotion direction per layer, and pc1's own DIR-vs-READ
  cosine per layer, are recorded at the top level of the result as DESCRIPTIVE geometry.
  Neither is gated: §3 says the footprint match "is checked after the run and reported,
  not gated".

GATE (§3), fatal before a single generation is spent
  rank-1   `acl_core.split_half` mean cosine >= 0.80 at every ablated hidden state.
  (B1d's rank-5 DIR/READ subspace gate is gone with the rank-5 arms; there is no gate on
  the rank-1 controls, because a permuted-label direction is *supposed* to be unstable and
  gating it would select the permutations that happen to look like signal.)

OUTCOMES (§4)
  primary    B_present_e   — the READ probe at the focus layer on B's GENERATED reply, in a
                             separate forward pass with all hooks off (`score_B`).
  secondary  B_readout_e   — the forced-choice readout applied to B's reply, asking about
                             B, hooks off.
  MC         ctx_readout_e — the forced-choice readout on A's span with the arm's ablation
                             LIVE, so all three hooked arms are comparable (§4.2); the
                             DOSE-0 value of this readout is what the instrument_failed
                             trigger of §2 is computed from.

Usage: python b1e_footprint.py --model Qwen/Qwen3.6-27B --tag qwen36-27b --reps 3
       python b1e_footprint.py --selftest   # analysis + directions only, no model, no GPU
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

# ---- declared grid (PREREG_B1e §3; fixed in advance, not chosen at run time) ------------
DOSES = (0.0, 0.33, 0.67, 1.0)          # four points: the slope is no longer an endpoint
ARMS = ("none", "emo_all", "permdir_all", "pc1_all")
NO_HOOK_ARMS = ("none",)
EMOS = ("desperate", "afraid", "happy", "calm", "sad", "angry")
STABILITY_GATE = 0.80                   # rank-1 split-half gate, unchanged from B1c/B1d
PROBE_K = 60
PREREG = "docs/planning/PREREG_B1e.md"

# the five contrasts of §5, in the pre-registered order
CONTRASTS = (("emo_all", "permdir_all"), ("emo_all", "pc1_all"),
             ("permdir_all", "none"), ("pc1_all", "none"), ("emo_all", "none"))
DECISION_CONTRAST = "emo_all_vs_permdir_all"

# §2 decision inputs: the two emotions named in advance and the two thresholds
PRESPEC_EMOTIONS = ("afraid", "sad")    # B1c's two blockers (report 22)
H3_MIN_FRACTION = 0.10                  # H3 needs the blocked fraction point >= this
DOSE0_SHIFT_MAX = 0.15                  # instrument_failed above this dose-0 readout shift

# §4.4 quality thresholds, pre-declared (as in B1c/B1d)
MAX_BAD_FRAC = 0.10
MAX_PPL_RATIO = 2.0


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
    """Full sha256 of a file, or '' if it is not there — the probe pool's identity (§7)."""
    try:
        hsh = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                hsh.update(chunk)
        return hsh.hexdigest()
    except Exception:
        return ""


def prereg_stamp() -> dict:
    """Path, content hash and (best effort) git commit of the pre-registration (§7).

    PREREG_B1e §7 inherits PREREG_B1d §8: "The driver resolves this document relative to
    its own directory (B1c's relative lookup failed) and the document is staged next to the
    driver on the box." So the search starts NEXT TO THE DRIVER and only then falls back to
    the repo root; the path actually used is recorded, so a stamp with `found` false is
    never mistaken for a stamp of the wrong file.
    """
    root = os.path.abspath(os.path.join(os.path.dirname(_SRC_PATH), "..", ".."))
    here = os.path.dirname(_SRC_PATH)
    cands = [os.path.join(here, PREREG),                 # first: next to the driver
             os.path.join(root, PREREG),                 # then: the repo root
             os.path.join(here, os.path.basename(PREREG))]   # staged as a bare file
    path = next((p for p in cands if os.path.exists(p)), cands[0])
    commit = ""
    try:
        commit = subprocess.run(["git", "-C", root, "log", "-1", "--format=%H", "--",
                                 PREREG], capture_output=True, text=True,
                                timeout=10).stdout.strip()
    except Exception:
        commit = ""
    return {"path": os.path.relpath(path, root) if path.startswith(root) else path,
            "abs_path": path, "searched": cands,
            "sha256": file_sha256(path), "git_commit": commit,
            "found": os.path.exists(path)}


def gen_seed(seed: int, emotion: str, alpha: float, rep: int, arm: str, part: str = "") -> int:
    """Deterministic per-generation torch seed (b1_followup's rule, unchanged).

    crc32 of "seed|emotion|alpha|rep|arm|part", masked to 31 bits, plus the batch start
    index. For part "B" the `arm` field is left EMPTY, so B's reply is sampled from ONE
    stream in all four arms — common random numbers, so sampling noise shared between two
    arms cancels inside each paired contrast instead of widening its CI. "readout" and
    "readout_B" keep `arm`, because those calls differ between arms by construction.
    """
    key = f"{seed}|{emotion}|{alpha:g}|{rep}|{'' if part == 'B' else arm}|{part}"
    return int(zlib.crc32(key.encode()) & 0x7FFFFFFF)


def perm_seed(seed: int, emotion: str, rep: int) -> int:
    """Seed for the label permutation of one (emotion, rep) cell (§3 `permdir_all`).

    Derived from (--seed, the CANONICAL emotion index in acl_core.EMOTIONS, rep) so it is
    stable under a resumed run and independent of the order EMOS happens to be written in.
    ONE permutation of the DIR half's labels is drawn per cell and re-used at EVERY ablated
    layer, so `permdir_all` is one coherent mislabelling of the pool rather than 51
    independent ones — the same rule B1d's rank-5 `perm_all` used, at rank 1.
    """
    ei = C.EMOTIONS.index(emotion)
    return int(zlib.crc32(f"perm|{seed}|{ei}|{rep}".encode()) & 0x7FFFFFFF)


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
# the three rank-1 directions (PREREG_B1e §3) — pure numpy, raw space, no torch needed
# ---------------------------------------------------------------------------------------

def permuted_direction(X: np.ndarray, y_perm: np.ndarray, emotion: str,
                       seed: int = 0) -> np.ndarray:
    """§3 `permdir_all`: the SAME `dom` estimator on the SAME rows with the labels PERMUTED.

    Identical data, identical class counts, identical estimator noise, identical rank and
    layers — and no emotion information. `acl_core.raw_direction` divides the standardized
    coefficient row by `sd` and normalises, so what comes back is a UNIT raw-space vector,
    the same object type `emo_all` installs.

    `y_perm` is passed in rather than permuted here on purpose: §3 fixes ONE permutation
    per (emotion, rep), re-used at every ablated layer, and that permutation is drawn by
    the caller from `perm_seed` so it is recorded in the row and reproducible.

    The estimator is hard-coded to "dom" (PREREG_B1e §3: "the same `dom` estimator"); the
    --estimator flag is checked against it in `main` and the run refuses to start if they
    disagree, so this control can never silently stop matching `emo_all`.
    """
    dec = C.fit_direction(np.asarray(X), np.asarray(y_perm), "dom", seed=seed)
    return C.raw_direction(dec, C.EMOTIONS.index(emotion))


def pc1_direction(X: np.ndarray) -> np.ndarray:
    """§3 `pc1_all`: the top principal direction of the CENTRED features at one layer.

    Unit norm, and the SAME vector for every steered emotion — it knows nothing about the
    labels at all, and it is the largest-footprint rank-1 object available: projecting it
    out removes more norm from A's span than any other unit direction can.

    The sign is fixed so the largest-magnitude component is positive. A principal axis is
    only defined up to sign, and LAPACK's choice varies with the build; without this fix
    the recorded cosines against the emotion directions would flip sign between boxes for
    no reason. It changes nothing about the ablation itself, because `Ablate` removes
    (h . v) v, which is invariant under v -> -v.
    """
    X = np.asarray(X, dtype=np.float64)
    Xc = X - X.mean(0)
    v = np.linalg.svd(Xc, full_matrices=False)[2][0]      # top right singular vector
    j = int(np.argmax(np.abs(v)))
    if v[j] < 0:
        v = -v
    return np.ascontiguousarray(v / (np.linalg.norm(v) + 1e-12), dtype=np.float64)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """cos(a, b) for two raw-space vectors; the descriptive geometry of §3."""
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


# ---------------------------------------------------------------------------------------
# analysis — importable with no torch and no model, so it is testable (`--selftest`)
# ---------------------------------------------------------------------------------------

METRICS = ("B_present_e", "B_other_e", "A_readout_e", "ctx_readout_e", "B_readout_e")


def _rowkey(r) -> tuple:
    return (r["emotion"], r["arm"], float(r["alpha"]), int(r["rep"]))


def _index(rows, e, arm, d) -> dict:
    """{(rep, scenario): (row, position)} for one (emotion, arm, dose) cell.

    Everything downstream works off these keys rather than off array positions. B1e has no
    ragged (rewrite) arms, so in a complete run every cell carries all 29 scenarios; the
    key-based indexing is kept because a run resumed from a checkpoint written by an older
    lease can still hold a short row, and a silent positional mismatch there would be
    unrecoverable from the result file.
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
    way to get there whenever the cells are not identical. Returns None when any requested
    cell is missing entirely.
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
    here as well because the per-emotion p-value of §5 needs the per-scenario numerator and
    denominator together, under one shared scenario draw.
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
    """Benjamini-Hochberg q-values. Descriptive only (§5: the decision is §2's per-emotion
    rule on the two pre-specified emotions)."""
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
    """§4.4 quality, per CELL (emotion, arm, dose, rep), each pre-declared pass/fail.

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


def footprint_summary(rows, arms=ARMS) -> dict:
    """§3's footprint check, computed AFTER the run and reported, never gated.

    The mean `removed_norm` per arm over every row (quality exclusions not applied: this
    describes what the hooks did, not what the outcome analysis used), and the two ratios
    that say whether the controls are as large as the emotion direction. §3 predicts
    permdir/emo > 1 ("the permuted-label direction is expected to remove more than the
    emotion direction ... which makes it a conservative control") and pc1/emo larger still;
    a ratio BELOW 1 would mean the control is footprint-poor in exactly the way B1c's
    random direction was, and the whole comparison would have to be read with that caveat.
    """
    mean = {}
    for arm in arms:
        v = [float(r["removed_norm"]) for r in rows
             if r.get("arm") == arm and r.get("removed_norm") is not None]
        mean[arm] = float(np.mean(v)) if v else None
    base = mean.get("emo_all")

    def ratio(arm):
        x = mean.get(arm)
        if x is None or base is None or abs(base) < 1e-12:
            return None
        return float(x / base)

    return {"mean_removed_norm": mean,
            "ratio_permdir_over_emo": ratio("permdir_all"),
            "ratio_pc1_over_emo": ratio("pc1_all"),
            "n_rows_by_arm": {arm: int(sum(1 for r in rows if r.get("arm") == arm))
                              for arm in arms},
            "note": "mean removed_norm per masked position summed over the ablated layers, "
                    "averaged over ALL rows of the arm (before quality exclusion); all "
                    "three hooked arms are rank 1 over the same layers, so these are "
                    "directly comparable (PREREG_B1e §3, reported not gated)"}


def analyze(rows, doses, emos, arms=ARMS, n_boot: int = 5000, seed: int = 0,
            apply_quality_exclusion: bool = True) -> dict:
    """PREREG_B1e §4-§5 in one importable, CPU-only function.

    Returns {'per_emotion', 'counts', 'decision', 'verdict', 'footprint', 'quality', ...}.
    Every slope is scenario-blocked; every contrast is scenario-blocked AND dose-paired
    through `acl_core.paired_slope_contrast(block=..., pair_doses=True)`; the verdict is
    §2's per-emotion rule on PRESPEC_EMOTIONS, with §2's instrument-failure override.
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
                "k": _first(rs, "k"),
                "n_layers_ablated": _first(rs, "n_layers_ablated"),
                "removed_norm": _mean_of(rs, "removed_norm"),
                "mask_hit_rate": _mean_of(rs, "mask_hit_rate"),
                "mask_n_positions": _mean_of(rs, "mask_n_positions"),
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
        # the decision fraction, the descriptive pc1 twin, and each control against `none`
        s["blocked_fraction"] = {n: blocked_fraction(s.get(n), ns) for n in
                                 ("emo_all_vs_permdir_all", "emo_all_vs_pc1_all",
                                  "permdir_all_vs_none", "pc1_all_vs_none")}
        s["detectable_effect"] = {f"{x}_vs_{y}": detectable_effect(s.get(f"{x}_vs_{y}"))
                                  for x, y in CONTRASTS}

        # ---- §4 manipulation checks ---------------------------------------------------
        s["mc_steer"] = _mc_steer(good, e, doses, n_boot, seed)
        s["mc_footprint"] = _mc_footprint(good, e, doses, arms)

        # ---- per-emotion p-value input: numerator and denominator on ONE panel ---------
        Pd = panel(good, e, ("none", "emo_all", "permdir_all"), doses,
                   metrics=("B_present_e",))
        if Pd is not None:
            d0 = Pd["data"]["B_present_e"]
            sc, sl_e = block_slopes(doses, d0["emo_all"], Pd["blocks"])
            _, sl_p = block_slopes(doses, d0["permdir_all"], Pd["blocks"])
            _, sl_n = block_slopes(doses, d0["none"], Pd["blocks"])
            dec_in[e] = {"scen": sc, "dd": sl_e - sl_p, "nn": sl_n}
            s["decision_input"] = {"n_scenarios": int(len(sc)),
                                   "mean_slope_diff": float(sl_e.mean() - sl_p.mean()),
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

    decision = _decide(per_emotion)
    verdict = decision["verdict"]

    counts = {"n_testable": len(testable), "n_untestable": len(emos) - len(testable),
              "n_block": len(block_e), "n_antiblock": len(anti_e), "n_null": len(null_e),
              "testable_emotions": testable, "blocking_emotions": block_e,
              "antiblocking_emotions": anti_e, "null_emotions": null_e,
              "n_mc_steer_pass": sum(1 for e in emos if (per_emotion[e].get("mc_steer")
                                                         or {}).get("sig")),
              "n_permdir_dose0_bad": sum(
                  1 for e in emos
                  if (per_emotion[e].get("mc_footprint") or {}).get("permdir_dose0_ok")
                  is False),
              "bh_fdr_q": fdr, "p_values": pvals,
              "note": "counts are DESCRIPTIVE over all six emotions; the verdict is §2's "
                      "per-emotion rule on the pre-specified emotions "
                      f"{list(PRESPEC_EMOTIONS)} only"}
    return {"per_emotion": per_emotion, "counts": counts, "decision": decision,
            "verdict": verdict, "doses": doses, "emotions": list(emos),
            "arms": list(arms), "n_boot": n_boot,
            "footprint": footprint_summary(rows, arms),
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


def _mc_steer(rows, e, doses, n_boot, seed) -> dict | None:
    """§4.3 — did steering actually put the emotion into A's TEXT? Slope CI excludes 0.

    §2's instrument_failed trigger is "MC-steer fails", which is read here as the slope CI
    failing to exclude zero (`sig`); `passes` (sig AND the slope positive) is recorded next
    to it, and a missing check counts as a failure rather than as a pass by default.
    """
    P = panel(rows, e, ("none",), doses, metrics=("A_readout_e",))
    if P is None:
        return None
    out = C.slope_ci(doses, P["data"]["A_readout_e"]["none"], n_boot=n_boot, seed=seed,
                     block=P["blocks"])
    out["passes"] = bool(out["sig"] and out["slope"] > 0)
    return out


def _mc_footprint(rows, e, doses, arms) -> dict:
    """§4.2 FOOTPRINT / READOUT CHECK, per emotion, on A's span with the intervention LIVE
    (`ctx_readout_e`). Two quantities, both from the same balanced panel.

    (1) At the TOP dose, the floor-corrected share removed, exactly as B1d computed it:

        share_removed[arm] = (none_top - arm_top) / (none_top - none_at_dose_0)

    The denominator is the FLOOR CORRECTION: the readout on an unsteered A is not 0, it is
    whatever the model guesses, so an ablation can only ever remove the part that steering
    put there. In B1e this is DESCRIPTIVE for all three hooked arms — there is no §4.2
    pass/fail threshold on it, because the question is no longer "did the ablation remove
    the readable affect" but "does a footprint-matched direction block the transfer".

    (2) At DOSE 0, the readout per arm and its shift from the `none` arm,

        dose0_shift[arm] = arm_at_dose_0 - none_at_dose_0

    which is what §2's instrument_failed trigger is computed from: `permdir_all` must not
    move the unsteered readout by more than DOSE0_SHIFT_MAX in either direction. A control
    that drags the dose-0 readout around is disturbing the measurement itself (B1d's rank-5
    subspace did), and then `emo_all - permdir_all` is not a clean comparison of two
    directions. `pc1_all`'s shift is reported the same way but does not gate anything.
    """
    top, zero = float(doses[-1]), float(doses[0])
    mc = {"dose": top, "floor_dose": zero, "dose0_shift_max": DOSE0_SHIFT_MAX}
    P = panel(rows, e, tuple(arms), doses, metrics=("ctx_readout_e",))
    if P is None:
        mc["permdir_dose0_ok"] = None
        mc["passes"] = None
        return mc
    D = P["data"]["ctx_readout_e"]
    for arm in D:
        v = D[arm][top]
        mc[arm] = {"mean": float(np.nanmean(v)), "ci": C.ci_of(v), "n": int(len(v))}
    if "none" not in mc:
        mc["permdir_dose0_ok"] = None
        mc["passes"] = None
        return mc

    # (2) the dose-0 readouts and their shifts — the §2 instrument trigger
    dose0 = {arm: float(np.nanmean(D[arm][zero])) for arm in D}
    none_zero = dose0["none"]
    mc["dose0_readout"] = dose0
    mc["dose0_shift"] = {arm: float(dose0[arm] - none_zero) for arm in dose0}
    sh = mc["dose0_shift"].get("permdir_all")
    mc["permdir_dose0_ok"] = (None if sh is None or not np.isfinite(sh)
                              else bool(abs(sh) <= DOSE0_SHIFT_MAX))
    mc["passes"] = mc["permdir_dose0_ok"]

    # (1) the floor-corrected shares at the top dose — descriptive
    none_top = mc["none"]["mean"]
    span = none_top - none_zero
    mc["none_at_floor"] = none_zero
    mc["steering_span"] = float(span)
    if not np.isfinite(span) or abs(span) < 1e-9:
        # nothing for the ablation to remove: the share is undefined, not zero
        mc["share_removed"] = {}
        return mc
    hooked = [arm for arm in ("emo_all", "permdir_all", "pc1_all") if arm in mc]
    mc["share_removed"] = {arm: float((none_top - mc[arm]["mean"]) / span)
                           for arm in hooked}
    mc["drop_by_arm"] = {arm: float(none_top - mc[arm]["mean"]) for arm in hooked}
    return mc


def _decide(per_emotion, prespec=PRESPEC_EMOTIONS) -> dict:
    """§2 VERDICT: the per-emotion rule, on the PRE-SPECIFIED emotions only.

    B1d decided on a median over whichever emotions turned out testable; B1e names afraid
    and sad in advance (B1c's two blockers) and asks the same yes/no question of each, so
    there is no aggregation step and no way for a fourth emotion's noise to move the
    verdict. The other four emotions are run and fully reported, and their contrasts,
    counts and BH-FDR q-values sit in `counts` as descriptive.

    Order of the checks is the order of §2: instrument first (a broken instrument makes the
    rest uninterpretable), then H3, then H3', else indeterminate. A missing MC-steer counts
    as a failure; a permdir_dose0_ok that is None (the panel could not be built) is NOT
    counted as False, because §2's trigger is a measured shift above 0.15, not an absent
    measurement — such a run lands in indeterminate through the H3/H3' tests instead.

    An emotion that is not COMPLETE (some arm has no rows at all — a run killed mid-sweep,
    or a mid-run analysis of a checkpoint) short-circuits to "indeterminate" with the
    emotion named in `incomplete_emotions`, ahead of every other check. PREREG_B1d §6,
    which B1e inherits, says the decision applies only with the full grid; without this
    guard an unfinished run would report "instrument_failed", which is a claim about the
    instrument rather than about the missing data.
    """
    ev = {}
    for e in prespec:
        s = per_emotion.get(e) or {}
        c = s.get(DECISION_CONTRAST)
        pn = s.get("permdir_all_vs_none")
        bf = (s.get("blocked_fraction") or {}).get(DECISION_CONTRAST) or {}
        mc = s.get("mc_steer") or {}
        fp = s.get("mc_footprint") or {}
        pt = bf.get("point")
        ev[e] = {
            "decision_sig_negative": bool(c and c.get("sig") and c["diff"] < 0),
            "diff": (float(c["diff"]) if c else None),
            "ci": ([float(x) for x in c["ci"]] if c else None),
            "blocked_fraction": (float(pt) if pt is not None else None),
            "blocked_fraction_range": bf.get("range"),
            "meets_min_fraction": bool(pt is not None and float(pt) >= H3_MIN_FRACTION),
            "permdir_vs_none_sig_negative": bool(pn and pn.get("sig") and pn["diff"] < 0),
            "mc_steer_sig": bool(mc.get("sig")),
            "mc_steer_passes": bool(mc.get("passes")),
            "permdir_dose0_ok": fp.get("permdir_dose0_ok"),
            "permdir_dose0_shift": (fp.get("dose0_shift") or {}).get("permdir_all"),
            "complete": bool(s.get("_complete")),
        }
    missing = [e for e in per_emotion if not (per_emotion[e] or {}).get("_complete")]
    bad = [e for e in prespec
           if (not ev[e]["mc_steer_passes"]) or ev[e]["permdir_dose0_ok"] is False]
    if missing:
        verdict = "indeterminate"
    elif bad:
        verdict = "instrument_failed"
    elif all(ev[e]["decision_sig_negative"] and ev[e]["meets_min_fraction"]
             for e in prespec):
        verdict = "H3"
    elif (not any(ev[e]["decision_sig_negative"] for e in prespec)) and \
            all(ev[e]["permdir_vs_none_sig_negative"] for e in prespec):
        verdict = "H3prime"
    else:
        verdict = "indeterminate"
    return {"prespec_emotions": list(prespec), "per_emotion": ev, "verdict": verdict,
            "instrument_failures": bad, "incomplete_emotions": missing,
            "rule": f"indeterminate if either of {list(prespec)} is incomplete; else "
                    f"instrument_failed if MC-steer is not significant for either of "
                    f"{list(prespec)} or |dose-0 readout shift of permdir_all| > "
                    f"{DOSE0_SHIFT_MAX} for either; else H3 if {DECISION_CONTRAST} is "
                    f"significantly negative with blocked fraction >= {H3_MIN_FRACTION} "
                    f"for BOTH; else H3prime if it is significantly negative for NEITHER "
                    f"while permdir_all_vs_none is significantly negative for BOTH; else "
                    f"indeterminate (PREREG_B1e §2)"}


def print_summary(res):
    per, cnt, dec = res["per_emotion"], res["counts"], res["decision"]
    print(f"\n{'emotion':>10} {'MCstr':>6} {'shEmo':>7} {'shPerm':>7} {'shPc1':>7} "
          f"{'d0Perm':>7} {'emo_all-permdir_all':>28} {'blockfrac':>10} {'det':>7} "
          f"{'d0OK':>6}")
    for e in res["emotions"]:
        s = per.get(e, {})
        mc = s.get("mc_footprint", {})
        sh = mc.get("share_removed", {})
        d0 = mc.get("dose0_shift", {})
        c = s.get(DECISION_CONTRAST)
        bf = s.get("blocked_fraction", {}).get(DECISION_CONTRAST, {}).get("point")
        de = s.get("detectable_effect", {}).get(DECISION_CONTRAST, {}).get("half_width")
        txt = (f"{c['diff']:+.3f} CI[{c['ci'][0]:+.3f},{c['ci'][1]:+.3f}]"
               if c else "--")
        star = "*" if e in PRESPEC_EMOTIONS else " "
        print(f"{e + star:>10} {str((s.get('mc_steer') or {}).get('passes')):>6} "
              f"{_f(sh.get('emo_all')):>7} {_f(sh.get('permdir_all')):>7} "
              f"{_f(sh.get('pc1_all')):>7} {_f(d0.get('permdir_all')):>7} {txt:>28} "
              f"{(f'{bf:+.3f}' if bf is not None else '--'):>10} "
              f"{_f(de):>7} {str(mc.get('permdir_dose0_ok')):>6}")
    fp = res.get("footprint", {})
    print(f"\ntestable {cnt['n_testable']}/{len(res['emotions'])} · blocking "
          f"{cnt['n_block']} · anti-blocking {cnt['n_antiblock']} · null {cnt['n_null']} · "
          f"MC-steer pass {cnt['n_mc_steer_pass']} · permdir dose-0 shifts too far "
          f"{cnt['n_permdir_dose0_bad']} · quality-excluded cells "
          f"{res['quality']['n_excluded_cells']}")
    print(f"FOOTPRINT: mean removed_norm "
          f"{ {k: (None if v is None else round(v, 3)) for k, v in fp.get('mean_removed_norm', {}).items()} }"
          f"; permdir/emo {_f(fp.get('ratio_permdir_over_emo'), 2)} · pc1/emo "
          f"{_f(fp.get('ratio_pc1_over_emo'), 2)}")
    print(f"DECISION (§2, pre-specified emotions {dec['prespec_emotions']}):")
    for e, v in dec["per_emotion"].items():
        print(f"  {e:>10}  decision sig-neg {str(v['decision_sig_negative']):>5} · "
              f"blocked fraction {_f(v['blocked_fraction'])} (>= {H3_MIN_FRACTION}: "
              f"{str(v['meets_min_fraction']):>5}) · permdir-none sig-neg "
              f"{str(v['permdir_vs_none_sig_negative']):>5} · MC-steer sig "
              f"{str(v['mc_steer_sig']):>5} · dose-0 shift "
              f"{_f(v['permdir_dose0_shift'])} ok {str(v['permdir_dose0_ok']):>5}")
    print(f"VERDICT: {res['verdict']}")


def _f(x, nd=3):
    return "--" if x is None or (isinstance(x, float) and not np.isfinite(x)) \
        else f"{x:+.{nd}f}"


# ---------------------------------------------------------------------------------------
# --selftest — synthetic rows with a planted answer, plus the direction algebra (no torch)
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
    """The scenario-blocked, dose-paired contrast machinery §5 depends on.

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


SIG_K = 5                               # rank of the planted class-mean signal in the fixture


def _synth_features(d=256, n_per=500, seed=1, sig=6.0, nuis=6.0, n_nuis=20):
    """A pool with the geometry the real one has: a low-dimensional class-mean signal
    inside a much larger, much stronger NUISANCE subspace, plus isotropic noise.

    The nuisance block is the whole point of the fixture for B1e. Without it the top
    principal direction of the features would BE the emotion direction, `pc1_all` would be
    an emotion arm in disguise, and the "pc1 differs from emo" assertion below would be
    testing nothing. Residual streams are not like that — the dominant variance is
    position/length/topic nuisance — and neither is this fixture.

    The nuisance block gets a DECAYING spectrum rather than an isotropic one, for the same
    reason: with 20 equal-variance nuisance directions the top principal direction is
    degenerate and pc1 is whatever the sampling noise of one half of the pool picks out of
    the tie, which is not what a residual stream looks like and would make the DIR-vs-READ
    stability line below meaningless.

    Columns are scaled to unit variance so that `acl_core.fit_direction`'s per-dimension
    `sd` is a CONSTANT and `raw_direction` is exactly proportional to the class-mean
    contrast, which keeps the direction assertions exact rather than approximate.
    """
    rng = np.random.default_rng(seed)
    B = np.linalg.qr(rng.standard_normal((d, SIG_K + n_nuis)))[0]
    S, Nu = B[:, :SIG_K], B[:, SIG_K:]
    U = rng.standard_normal((6, SIG_K))
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    spec = np.linspace(2.0, 0.4, n_nuis)                 # decaying nuisance spectrum
    X, y = [], []
    for c in range(6):
        Z = rng.standard_normal((n_per, d))
        Z = Z + (rng.standard_normal((n_per, n_nuis)) * nuis * spec) @ Nu.T
        X.append(Z + sig * (U[c] @ S.T))
        y += [c] * n_per
    X = np.concatenate(X)
    X = X / (X.std(0) + 1e-12)
    return X, np.asarray(y), S


def _direction_selftest() -> None:
    """PREREG_B1e §3: the three rank-1 objects, and the geometry that makes the controls
    controls rather than relabelled copies of the emotion direction."""
    X, y, _S = _synth_features()
    d = X.shape[1]

    # ---- the emotion direction (unchanged from B1c/B1d) and the permuted-label control --
    dec = C.fit_direction(X, y, "dom")
    for e in PRESPEC_EMOTIONS:
        i = C.EMOTIONS.index(e)
        v_emo = C.raw_direction(dec, i)
        rng = np.random.default_rng(perm_seed(0, e, 0))
        v_perm = permuted_direction(X, rng.permutation(y), e)
        assert v_perm.shape == (d,), (e, v_perm.shape)
        # 1e-6, not 1e-9: acl_core.raw_direction divides by (norm + 1e-9), so a unit
        # vector out of it is unit to ~1e-9 BELOW one, never exactly one.
        assert abs(np.linalg.norm(v_perm) - 1.0) < 1e-6, (e, np.linalg.norm(v_perm))
        assert abs(np.linalg.norm(v_emo) - 1.0) < 1e-6, (e, np.linalg.norm(v_emo))
        cs = abs(cosine(v_perm, v_emo))
        assert cs < 0.5, ("the permuted-label direction must NOT recover the emotion "
                          "direction", e, cs)
        # one permutation per (emotion, rep): the same seed must give the same vector back
        rng2 = np.random.default_rng(perm_seed(0, e, 0))
        assert np.allclose(v_perm, permuted_direction(X, rng2.permutation(y), e)), e
        print(f"[selftest] permdir({e}) unit, |cos| with the emotion direction {cs:.3f} "
              f"< 0.5, reproducible from perm_seed {perm_seed(0, e, 0)}")

    # ---- pc1: unit, the top eigenvector of the covariance, and sign-stable --------------
    v_pc = pc1_direction(X)
    assert v_pc.shape == (d,) and abs(np.linalg.norm(v_pc) - 1.0) < 1e-9, v_pc.shape
    Xc = X - X.mean(0)
    w, V = np.linalg.eigh(Xc.T @ Xc / (len(Xc) - 1))
    top = V[:, int(np.argmax(w))]
    cs = abs(cosine(v_pc, top))
    assert cs > 0.999, ("pc1_direction is not the top eigenvector of the covariance", cs)
    j = int(np.argmax(np.abs(v_pc)))
    assert v_pc[j] > 0, ("the sign fix did not fire", v_pc[j])
    assert np.array_equal(v_pc, pc1_direction(X)), "pc1_direction is not deterministic"
    emo_cos = [abs(cosine(v_pc, C.raw_direction(dec, i))) for i in range(6)]
    assert max(emo_cos) < 0.5, ("pc1 must not be the emotion direction in disguise",
                                emo_cos)
    print(f"[selftest] pc1 unit, |cos| with the top eigh eigenvector {cs:.6f} > 0.999, "
          f"largest-magnitude component positive, max |cos| with an emotion direction "
          f"{max(emo_cos):.3f}")

    # the sign fix on data whose natural top PC points the "wrong" way
    rng = np.random.default_rng(11)
    u = np.zeros(8); u[3] = -1.0; u[0] = 0.2
    u = u / np.linalg.norm(u)
    Xs = np.outer(rng.normal(size=400), u) + 0.01 * rng.standard_normal((400, 8))
    v = pc1_direction(Xs)
    assert v[int(np.argmax(np.abs(v)))] > 0, v
    assert abs(abs(cosine(v, u)) - 1.0) < 1e-2, cosine(v, u)
    assert np.array_equal(v, pc1_direction(Xs))
    assert np.array_equal(v, pc1_direction(-Xs)), "the sign fix must not depend on the sign"
    print(f"[selftest] sign fix deterministic: axis recovered at |cos| "
          f"{abs(cosine(v, u)):.4f} with its largest component positive, identical on -X")

    # DIR/READ halves: pc1 is a property of the model, not of half the pool
    idx = np.random.default_rng(3).permutation(len(y))
    DIR, READ = idx[:len(y) // 2], idx[len(y) // 2:]
    g = abs(cosine(pc1_direction(X[DIR]), pc1_direction(X[READ])))
    assert g > 0.9, ("the fixture's pc1 should be stable across halves", g)
    print(f"[selftest] pc1 DIR-vs-READ |cos| {g:.3f} (descriptive in the run, not gated)")


# planted §4.2 shares: `none` rises from the floor with dose and every hooked arm removes
# some of that rise; the numbers are descriptive in B1e, so they only have to be distinct.
READ_FLOOR = 0.20
READ_TOP = {"none": 0.70, "emo_all": 0.45, "permdir_all": 0.40, "pc1_all": 0.35}


def _synth_rows(doses, emos, arms, frac_by_emotion, n_scen=10, reps=2, seed=7,
                blocking_sd=0.0, read_top=None, read_floor=None, permdir_frac=0.0,
                pc1_frac=None, a_read_slope=0.5, a_read_noise=0.05):
    """Rows with a KNOWN answer.

    Arm slopes are planted directly, so every §2 branch is reachable from the parameters:

        slope[none]        = 1.0
        slope[permdir_all] = 1.0 - permdir_frac          (how much the CONTROL blocks)
        slope[pc1_all]     = 1.0 - pc1_frac              (defaults to permdir_frac)
        slope[emo_all]     = slope[permdir_all] - f      (f = frac_by_emotion[e])

    so `emo_all - permdir_all` is exactly -f and `permdir_all - none` exactly
    -permdir_frac. Every arm in one (emotion, dose, rep) cell shares its noise draw, so an
    arm with no planted effect is IDENTICAL to its control rather than merely similar and
    its contrast is deterministically null. A per-scenario slope offset (shared by all
    arms, and CENTRED so the none-arm slope is exactly 1.0) gives the scenario-blocked
    machinery real clustering to handle.

    `ctx_readout_e` is planted at `read_floor[arm]` at dose 0 and at `read_top[arm]` above
    it, so both halves of §4.2 are exact: the floor-corrected share removed at the top
    dose, and the dose-0 shift `read_floor[arm] - read_floor['none']` that §2's
    instrument_failed trigger reads.

    `a_read_slope` = 0 makes MC-steer fail (A's readout flat in dose), the other
    instrument_failed trigger; pair it with `a_read_noise` = 0 to make the failure
    DETERMINISTIC — with noise left in, a flat readout still clears its slope CI in about
    one emotion in twenty by chance, and a selftest that depends on which ones is a
    selftest that fails on someone else's machine. `blocking_sd` > 0 makes HOW MUCH is
    blocked itself vary by scenario (f -> f(1 + w), w centred), widening the bootstrap.
    """
    rng = np.random.default_rng(seed)
    read_top = dict(READ_TOP if read_top is None else read_top)
    read_floor = dict({a: READ_FLOOR for a in arms} if read_floor is None else read_floor)
    pc1_frac = permdir_frac if pc1_frac is None else pc1_frac
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
        slope = {"none": 1.0, "permdir_all": 1.0 - permdir_frac,
                 "pc1_all": 1.0 - pc1_frac}
        for d in doses:
            for rep in range(reps):
                nz = rng.normal(0, 0.15, n_scen)            # shared across arms
                nzo = rng.normal(0, 0.15, n_scen)
                nzr = rng.normal(0, 0.02, n_scen)
                a_read = np.full(n_scen, 0.2 + a_read_slope * d)
                if a_read_noise:
                    a_read = a_read + rng.normal(0, a_read_noise, n_scen)
                for arm in arms:
                    keep = list(range(n_scen))
                    base = read_floor[arm] if d == 0 else read_top[arm]
                    sl = ((slope["permdir_all"] - blocked) if arm == "emo_all"
                          else np.full(n_scen, slope[arm]))
                    rows.append({
                        "emotion": e, "alpha": float(d), "rep": rep, "arm": arm,
                        "scenario": keep,
                        "B_present_e": ((sl + scen_off) * d + nz).tolist(),
                        "B_other_e": (0.05 * d + nzo).tolist(),
                        "A_readout_e": a_read.tolist(),
                        "ctx_readout_e": (base + nzr).tolist(),
                        "ctx_readout_full": [1.0 / 6] * 6,
                        "B_readout_e": (0.15 + 0.30 * float(np.mean(sl)) * d
                                        + nzr).tolist(),
                        "k": (0 if arm in NO_HOOK_ARMS else 1),
                        "removed_norm": (0.0 if arm in NO_HOOK_ARMS else
                                         {"emo_all": 0.66, "permdir_all": 0.90,
                                          "pc1_all": 1.40}[arm]),
                        "mask_hit_rate": None if arm in NO_HOOK_ARMS else 1.0,
                        "mask_n_positions": None if arm in NO_HOOK_ARMS else 46.0,
                        "n_layers_ablated": 0 if arm in NO_HOOK_ARMS else 51,
                        "degenerate_frac": 0.02, "refusal_frac": 0.0,
                        "distinct2": 0.9, "n_words": 40.0, "perplexity": 12.0,
                        "gen_seed": gen_seed(0, e, d, rep, arm, "B"),
                        "perm_seed": (perm_seed(0, e, rep) if arm == "permdir_all"
                                      else None),
                    })
    return rows


def selftest() -> int:
    t0 = time.time()
    _direction_selftest()
    _contrast_selftest()

    # --- H3: the emotion direction blocks MORE than its footprint-matched twin -----------
    plant = {e: 0.30 for e in EMOS}
    rows = _synth_rows(DOSES, EMOS, ARMS, plant)
    print(f"\n[selftest] {len(rows)} synthetic arm-rows; planted blocked fraction 0.30 "
          f"for emo_all vs permdir_all, permdir_all planted equal to none", flush=True)
    # n_boot below the 5000 default only to keep this fast on a CPU box; the planted
    # contrasts are 0 or a fixed fraction with a hair-width CI, so no significance call
    # here depends on the bootstrap size.
    res = analyze(rows, DOSES, EMOS, n_boot=500)
    print_summary(res)
    per, cnt, dec = res["per_emotion"], res["counts"], res["decision"]

    assert cnt["n_untestable"] == 0, cnt
    assert cnt["n_mc_steer_pass"] == len(EMOS), cnt
    assert cnt["n_permdir_dose0_bad"] == 0, cnt
    for e in EMOS:
        s = per[e]
        assert s["_complete"] and all(s.get(a) for a in ARMS), (e, s["_complete"])
        # every hooked arm is rank 1 in B1e -- that IS the design
        for arm in ("emo_all", "permdir_all", "pc1_all"):
            assert s[arm]["k"] == 1 and s[arm]["n_layers_ablated"] == 51, (e, arm, s[arm])
        assert s["none"]["k"] == 0, (e, s["none"]["k"])
        # the NUMERATOR is exact: emo_all and permdir_all share their noise draw in every
        # cell, so the planted slope difference survives the dose-paired contrast with no
        # residual at all. The FRACTION carries the `none` denominator's own sampling
        # error (a slope estimated from 10 scenarios x 2 reps), which is where the 0.10
        # slack below comes from -- and that error is itself reported as denominator_ci.
        assert abs(s[DECISION_CONTRAST]["diff"] + plant[e]) < 1e-9, (e,
                                                                     s[DECISION_CONTRAST])
        bf = s["blocked_fraction"][DECISION_CONTRAST]
        assert bf["defined"] and abs(bf["point"] - plant[e]) < 0.10, (e, bf, plant[e])
        assert bf["denominator_ci"] is not None and all(np.isfinite(bf["range"])), (e, bf)
        for name in ("emo_all_vs_pc1_all", "permdir_all_vs_none", "pc1_all_vs_none"):
            assert name in s["blocked_fraction"], (e, name)
        de = s["detectable_effect"][DECISION_CONTRAST]["half_width"]
        assert de is not None and np.isfinite(de) and de > 0, (e, de)
        # permdir_all and pc1_all were planted equal to none: neither may read as blocking
        assert s["permdir_all_vs_none"]["sig"] is False, (e, s["permdir_all_vs_none"])
        assert s["pc1_all_vs_none"]["sig"] is False, (e, s["pc1_all_vs_none"])
        mc = s["mc_footprint"]
        assert mc["permdir_dose0_ok"] is True, (e, mc)
        assert abs(mc["dose0_shift"]["permdir_all"]) < 1e-6, (e, mc["dose0_shift"])
        assert abs(mc["share_removed"]["emo_all"] - 0.50) < 0.05, (e, mc["share_removed"])
        assert abs(mc["share_removed"]["pc1_all"] - 0.70) < 0.05, (e, mc["share_removed"])
        for name in [f"{x}_vs_{y}" for x, y in CONTRASTS]:
            assert s[name]["n_blocks"] > 1, (e, name, s[name])
    assert res["verdict"] == "H3", (res["verdict"], dec)
    assert all(dec["per_emotion"][e]["decision_sig_negative"] for e in PRESPEC_EMOTIONS)
    fp = res["footprint"]
    assert abs(fp["ratio_permdir_over_emo"] - 0.90 / 0.66) < 1e-6, fp
    assert abs(fp["ratio_pc1_over_emo"] - 1.40 / 0.66) < 1e-6, fp
    assert fp["mean_removed_norm"]["none"] == 0.0, fp
    print(f"[selftest] H3 reproduced on both pre-specified emotions "
          f"{list(PRESPEC_EMOTIONS)}; footprint ratios permdir/emo "
          f"{fp['ratio_permdir_over_emo']:.2f}, pc1/emo {fp['ratio_pc1_over_emo']:.2f}")

    # --- the rule must MOVE with the data: H3prime and indeterminate ---------------------
    small = dict(n_scen=8, reps=2)
    # H3': the permuted direction blocks just as much, so emo_all - permdir_all is null
    h3p = analyze(_synth_rows(DOSES, EMOS, ARMS, {e: 0.0 for e in EMOS},
                              permdir_frac=0.30, **small), DOSES, EMOS, n_boot=300)
    assert h3p["verdict"] == "H3prime", (h3p["verdict"], h3p["decision"])
    for e in PRESPEC_EMOTIONS:
        v = h3p["decision"]["per_emotion"][e]
        assert v["decision_sig_negative"] is False and v["permdir_vs_none_sig_negative"], \
            (e, v)
    # indeterminate: ONE of the two pre-specified emotions blocks, the other does not
    mixed = {e: (0.30 if e == "afraid" else 0.0) for e in EMOS}
    mid = analyze(_synth_rows(DOSES, EMOS, ARMS, mixed, **small), DOSES, EMOS, n_boot=300)
    assert mid["verdict"] == "indeterminate", (mid["verdict"], mid["decision"])
    assert mid["decision"]["per_emotion"]["afraid"]["decision_sig_negative"] is True
    assert mid["decision"]["per_emotion"]["sad"]["decision_sig_negative"] is False
    print(f"[selftest] verdict moves H3 / H3prime / indeterminate with the planted arm "
          f"slopes ({res['verdict']} / {h3p['verdict']} / {mid['verdict']})")

    # --- §2 instrument_failed, trigger 1: permdir_all shifts the dose-0 readout ----------
    shifted = _synth_rows(DOSES, EMOS, ARMS, plant, **small,
                          read_floor={"none": 0.20, "emo_all": 0.20,
                                      "permdir_all": 0.50, "pc1_all": 0.20})
    bshift = analyze(shifted, DOSES, EMOS, n_boot=200)
    assert bshift["verdict"] == "instrument_failed", (bshift["verdict"],
                                                      bshift["decision"])
    assert bshift["counts"]["n_permdir_dose0_bad"] == len(EMOS), bshift["counts"]
    sh = bshift["per_emotion"]["afraid"]["mc_footprint"]["dose0_shift"]["permdir_all"]
    assert abs(sh - 0.30) < 0.02, sh
    print(f"[selftest] a permdir_all dose-0 shift of {sh:+.2f} (> {DOSE0_SHIFT_MAX}) "
          f"overrides a planted 0.30 blocked fraction -> instrument_failed")

    # --- §2 instrument_failed, trigger 2: MC-steer flat ----------------------------------
    flat = analyze(_synth_rows(DOSES, EMOS, ARMS, plant, a_read_slope=0.0,
                               a_read_noise=0.0, **small), DOSES, EMOS, n_boot=200)
    assert flat["verdict"] == "instrument_failed", (flat["verdict"], flat["counts"])
    assert flat["counts"]["n_mc_steer_pass"] == 0, flat["counts"]
    assert set(flat["decision"]["instrument_failures"]) == set(PRESPEC_EMOTIONS), \
        flat["decision"]["instrument_failures"]
    print("[selftest] a flat A-span readout (MC-steer not significant) also overrides "
          "-> instrument_failed")

    # --- a PARTIAL run must read as indeterminate, not as a broken instrument ------------
    part = [r for r in _synth_rows(DOSES, EMOS, ARMS, plant, **small)
            if not (r["emotion"] == "sad" and r["arm"] == "permdir_all")]
    bpart = analyze(part, DOSES, EMOS, n_boot=200)
    assert bpart["verdict"] == "indeterminate", (bpart["verdict"], bpart["decision"])
    assert bpart["decision"]["incomplete_emotions"] == ["sad"], bpart["decision"]
    assert bpart["decision"]["per_emotion"]["afraid"]["complete"] is True, bpart["decision"]
    print("[selftest] a run missing one arm of one pre-specified emotion reports "
          "indeterminate with incomplete_emotions ['sad'], not instrument_failed")

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
                    help="run analyze() and the direction algebra on synthetic data and "
                         "exit; loads no model")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if not a.model or not a.tag:
        ap.error("--model and --tag are required (or pass --selftest)")
    if a.estimator != "dom":
        # PREREG_B1e §3 fixes `dom` for both `emo_all` and `permdir_all`; a different
        # --estimator would fit the two arms with different estimators and destroy the
        # footprint match that IS the experiment.
        ap.error(f"PREREG_B1e §3 fixes the `dom` estimator for emo_all and permdir_all; "
                 f"--estimator {a.estimator!r} would break the footprint match")
    os.makedirs(a.outdir, exist_ok=True); os.makedirs(a.workdir, exist_ok=True)

    pool_path = os.path.join(a.workdir, f"probe_{a.tag}.jsonl")
    prov = C.Provenance(
        script="src/rev3/b1e_footprint.py",
        config={"doses": DOSES, "arms": ARMS, "emotions": EMOS, "reps": a.reps,
                "driver": "b1e", "estimator": a.estimator, "probe_k": PROBE_K,
                "stability_gate": STABILITY_GATE,
                "max_new_A": 110, "max_new_B": 110,
                "temp": 0.9, "top_p": 0.95,
                "n_scenarios": len(C.SCENARIOS),
                "contrasts": [f"{x}_vs_{y}" for x, y in CONTRASTS],
                "decision_contrast": DECISION_CONTRAST,
                "prespec_emotions": list(PRESPEC_EMOTIONS),
                "decision_thresholds": {"h3_min_fraction": H3_MIN_FRACTION,
                                        "dose0_shift_max": DOSE0_SHIFT_MAX},
                "quality": {"max_bad_frac": MAX_BAD_FRAC,
                            "max_perplexity_ratio": MAX_PPL_RATIO},
                "gen_seed_rule": "crc32('seed|emotion|alpha|rep|arm|part') & 0x7fffffff, "
                                 "+ batch start index; `arm` is EMPTY for part 'B' so all "
                                 "four arms sample B's reply with common random numbers, "
                                 "and kept for parts 'readout' and 'readout_B'",
                "perm_seed_rule": "crc32('perm|seed|EMOTIONS.index(e)|rep') & 0x7fffffff; "
                                  "ONE permutation of the DIR half's labels per "
                                  "(emotion, rep), re-used at EVERY ablated layer",
                "prereg": prereg_stamp(),
                "probe_pool_path": pool_path,
                "predecessors": ["results/rev3/b1c_alllayer_qwen36-27b.json",
                                 "results/rev3/b1d_subspace_qwen36-27b.json"]},
        model_id=a.model, model_revision="", seeds={"pool": a.seed, "run": a.seed},
        code_sha=C.code_hash(os.path.abspath(__file__), os.path.abspath(C.__file__)),
        control_pointers={
            # Every span below is read out of THIS file's AST at import time by fn_lines /
            # block_lines — never typed, so a pointer cannot drift away from the code it
            # names; a name that stops existing renders as "NOT FOUND — pointer is stale".
            "per_layer_gate": f"b1e_footprint.py main ({block_lines('GATE')}): "
                              f"acl_core.split_half at EVERY ablated hidden_states index "
                              f"(13..n_layers-1) plus the focus layer, n_per_half="
                              f"min(N//2,600), 10 seeds; the run EXITS without generating "
                              f"anything if any ablated layer's class mean is below "
                              f"{STABILITY_GATE} (PREREG_B1e §3, identical to B1c/B1d). "
                              f"B1d's rank-5 DIR/READ subspace gate is GONE with the "
                              f"rank-5 arms and there is no gate on either rank-1 control",
            "emotion_direction": f"b1e_footprint.py main ({block_lines('DIRS')}): "
                                 f"acl_core.raw_direction(fit_direction(feats[L][DIR], "
                                 f"yp[DIR], 'dom'), EMOTIONS.index(e)) per ablated hidden "
                                 f"state — B1c's and B1d's object unchanged, so `emo_all` "
                                 f"is the bridge arm to reports 22 and 27",
            "permuted_direction_control": f"b1e_footprint.py::permuted_direction "
                                          f"({fn_lines('permuted_direction')}) seeded by "
                                          f"::perm_seed ({fn_lines('perm_seed')}), "
                                          f"assembled in main's ::permdir_dirs_for "
                                          f"({fn_lines('permdir_dirs_for')}): the SAME "
                                          f"`dom` estimator on the SAME DIR-half rows with "
                                          f"the present-emotion labels PERMUTED, the "
                                          f"class-index direction for the steered emotion, "
                                          f"unit norm — ONE permutation per (emotion, rep) "
                                          f"re-used at every ablated layer, so the "
                                          f"footprint (same data, same estimator noise, "
                                          f"SAME RANK 1, same layers) is matched and only "
                                          f"the emotion information is gone; the seed is "
                                          f"in every permdir_all row as perm_seed "
                                          f"(PREREG_B1e §3, the decision control)",
            "pc1_control": f"b1e_footprint.py::pc1_direction "
                           f"({fn_lines('pc1_direction')}), assembled in main "
                           f"({block_lines('DIRS')}): the top right singular vector of the "
                           f"MEAN-CENTRED DIR-half features at each ablated layer, unit "
                           f"norm, sign fixed so the largest-magnitude component is "
                           f"positive, THE SAME direction for every steered emotion — the "
                           f"largest-footprint rank-1 control, reported descriptively",
            "footprint_geometry": f"b1e_footprint.py::cosine ({fn_lines('cosine')}) in "
                                  f"main ({block_lines('DIRS')}) -> out['pc1_vs_emo_cos'] "
                                  f"and out['pc1_dir_read_cos'], plus ::footprint_summary "
                                  f"({fn_lines('footprint_summary')}) -> "
                                  f"out['footprint'] (mean removed_norm per arm and the "
                                  f"permdir/emo and pc1/emo ratios). PREREG_B1e §3: the "
                                  f"footprint match is checked AFTER the run and reported, "
                                  f"NOT gated",
            "ablation": f"b1e_footprint.py main ({block_lines('LAYERS')}) and ::dirs_for "
                        f"({fn_lines('dirs_for')}): ONE acl_core.Ablate over decoder "
                        f"blocks 12..n_layers-2 (hs 13..n_layers-1); all three hooked arms "
                        f"pass 1-D unit direction tensors, so Ablate takes the SAME rank-1 "
                        f"path (proj = (h.d) d) for every one of them and the arms differ "
                        f"ONLY in which direction is removed; per-row `k`, `removed_norm`, "
                        f"`mask_n_positions` and `n_layers_ablated` record what was "
                        f"actually taken out",
            "blocked_paired_contrast": f"acl_core.paired_slope_contrast(block=..., "
                                       f"pair_doses=True) -> acl_core."
                                       f"_blocked_paired_slope_contrast, called from "
                                       f"b1e_footprint.py::analyze ({fn_lines('analyze')}) "
                                       f"on the balanced scenario panel built by ::panel "
                                       f"({fn_lines('panel')}); one-sample bootstrap over "
                                       f"scenarios of the per-scenario paired slope "
                                       f"difference, 5000 draws, seed 0",
            "decision_rule": f"b1e_footprint.py::_decide ({fn_lines('_decide')}), called "
                             f"from ::analyze ({fn_lines('analyze')}); PREREG_B1e §2's "
                             f"PER-EMOTION rule on the pre-specified emotions "
                             f"{list(PRESPEC_EMOTIONS)} only — H3 needs "
                             f"{DECISION_CONTRAST} significantly negative with a blocked "
                             f"fraction >= {H3_MIN_FRACTION} for BOTH, H3prime needs it "
                             f"significant for NEITHER while permdir_all_vs_none is "
                             f"significantly negative for both, and instrument_failed "
                             f"overrides when MC-steer is not significant or the "
                             f"permdir_all dose-0 readout shift exceeds "
                             f"{DOSE0_SHIFT_MAX}; thresholds fixed in this file's module "
                             f"docstring and in PREREG_B1e §2 before the run; "
                             f"::blocked_fraction ({fn_lines('blocked_fraction')}) is "
                             f"sign-aware",
            "secondary_outcome": f"b1e_footprint.py::readout ({fn_lines('readout')}) with "
                                 f"who='B' and the reply appended to the conversation, "
                                 f"hooks OFF — stored per row as B_readout_e",
            "manipulation_checks": f"b1e_footprint.py::_mc_steer ({fn_lines('_mc_steer')}) "
                                   f"and ::_mc_footprint ({fn_lines('_mc_footprint')}) — "
                                   f"the latter reports PREREG_B1d §4.2's floor-corrected "
                                   f"share removed, (none_top - arm_top) / (none_top - "
                                   f"none at dose 0) on ctx_readout_e at the top dose "
                                   f"(descriptive for all three hooked arms in B1e) AND "
                                   f"the DOSE-0 readout per arm with its shift from the "
                                   f"none arm, which is the quantity PREREG_B1e §2's "
                                   f"instrument_failed trigger thresholds at "
                                   f"{DOSE0_SHIFT_MAX}; ::quality_flags "
                                   f"({fn_lines('quality_flags')}) is §4.4",
            "reproducible_sampling": f"b1e_footprint.py::gen_seed ({fn_lines('gen_seed')}),"
                                     f" called immediately before every h.model.generate in"
                                     f" ::gen_B ({fn_lines('gen_B')}) and before the "
                                     f"readout forward in ::readout ({fn_lines('readout')})",
            "non_circular_measurement": f"b1e_footprint.py main ({block_lines('POOL')}): "
                                        f"probe fitted on the READ half, steering "
                                        f"directions and ALL THREE ablation directions "
                                        f"(emotion, permuted-label, pc1) built on the "
                                        f"disjoint DIR half; the READ-half pc1 is computed "
                                        f"ONLY to report the descriptive DIR-vs-READ "
                                        f"cosine and is never installed as a hook",
            "frozen_token_audit": "arms emo_all/permdir_all/pc1_all regenerate B under the "
                                  "intervention while A's message predates it; no arm "
                                  "rewrites A's message (PREREG_B1e §3: no rewrite arms). "
                                  "See out['frozen_token_audit'].",
        })

    h = C.load(a.model)
    prov.model_revision = h.revision

    # <<LAYERS_START>>
    # ---- the layer set (PREREG_B1e §3: every hooked arm ablates hs 13..n_layers-1) ------
    # `lo` is built exactly as b1c_alllayer.py and b1d_subspace.py built it, so `emo_all`
    # is B1c's arm and not a near-miss. The set runs to n_layers-1: hs n_layers is the
    # output of the LAST block, which feeds only the final norm and the LM head, so
    # projecting there cannot change what B attends to — a no-op arm dressed as an ablation.
    focus = h.focus(); steer_layer = focus - 1
    lo = round(0.2 * h.n_layers)
    abl_hs_all = list(range(lo, h.n_layers))                # 13..63 at n_layers=64
    abl_dec_all = [L - 1 for L in abl_hs_all]               # decoder blocks 12..62
    hs_needed = sorted(set(abl_hs_all) | {focus})
    # <<LAYERS_END>>
    print(f"[B1E] focus {focus} steer_layer {steer_layer} n_layers {h.n_layers}; "
          f"all hooked arms hs {abl_hs_all[0]}..{abl_hs_all[-1]} "
          f"({len(abl_hs_all)} blocks); all arms rank 1", flush=True)

    # <<POOL_START>>
    # ---------------- probe pool, split into DIR and READ halves ------------------------
    # Same path, same k, same seed, same permutation as b1c_alllayer.py / b1d_subspace.py:
    # with the pool file from those runs present in --workdir the rank-1 directions are the
    # SAME vectors, which is what makes `emo_all` a bridge to reports 22 and 27 rather than
    # a new measurement.
    items = C.generate_pool(h, PROBE_K, pool_path, seed=a.seed, bs=a.gen_bs)
    feats, yp, yo = C.pool_features(h, items, hs_needed,
                                    cache=os.path.join(a.workdir, f"probefeat_{a.tag}.npz"))
    N = len(items)
    rs = np.random.default_rng(a.seed).permutation(N)
    DIR, READ = rs[:N // 2], rs[N // 2:]
    pool_sha = file_sha256(pool_path)
    print(f"[B1E] probe pool n={N}: DIR half {len(DIR)}, READ half {len(READ)}; "
          f"sha256 {pool_sha[:16]}...", flush=True)

    dir_dec = {L: C.fit_direction(feats[L][DIR], yp[DIR], a.estimator, seed=a.seed,
                                  pair_on=yo[DIR]) for L in hs_needed}
    read_dec = {L: C.fit_direction(feats[L][READ], yp[READ], a.estimator, seed=a.seed,
                                   pair_on=yo[READ]) for L in hs_needed}
    read_dec_other = C.fit_direction(feats[focus][READ], yo[READ], a.estimator, seed=a.seed)
    # pc1: the DIR half is what gets ABLATED; the READ half is computed ONLY to report the
    # descriptive DIR-vs-READ cosine below and is never installed as a hook.
    pc1_np = {L: pc1_direction(feats[L][DIR]) for L in abl_hs_all}
    pc1_read_np = {L: pc1_direction(feats[L][READ]) for L in abl_hs_all}
    emo_np = {e: {L: C.raw_direction(dir_dec[L], C.EMOTIONS.index(e)) for L in abl_hs_all}
              for e in EMOS}
    # <<POOL_END>>

    # ---- descriptive geometry (PREREG_B1e §3: reported, NOT gated) ----------------------
    pc1_vs_emo_cos = {str(L): {e: cosine(pc1_np[L], emo_np[e][L]) for e in EMOS}
                      for L in abl_hs_all}
    pc1_dir_read_cos = {str(L): cosine(pc1_np[L], pc1_read_np[L]) for L in abl_hs_all}
    _pe = [abs(v) for d in pc1_vs_emo_cos.values() for v in d.values()]
    _pr = [abs(v) for v in pc1_dir_read_cos.values()]
    pc1_geometry = {
        "note": "|cos| summaries of the two descriptive quantities of PREREG_B1e §3; "
                "pc1's sign is fixed by pc1_direction (largest-magnitude component "
                "positive) per half, which does NOT align the two halves, so read |cos| "
                "near 1 as the same axis. Neither number gates anything.",
        "pc1_vs_emo_abs_cos": {"min": float(np.min(_pe)), "median": float(np.median(_pe)),
                               "max": float(np.max(_pe))},
        "pc1_dir_read_abs_cos": {"min": float(np.min(_pr)),
                                 "median": float(np.median(_pr)),
                                 "max": float(np.max(_pr))}}
    print(f"[B1E] pc1 vs emotion directions |cos| min {np.min(_pe):.3f} median "
          f"{np.median(_pe):.3f} max {np.max(_pe):.3f}; pc1 DIR-vs-READ |cos| min "
          f"{np.min(_pr):.3f} median {np.median(_pr):.3f} max {np.max(_pr):.3f} "
          f"(descriptive, not gated)", flush=True)

    # <<GATE_START>>
    # ---- GATE (PREREG_B1e §3): rank-1 split-half at EVERY ablated layer, 13..n_layers-1 -
    # "split-half gate >= 0.80 at every ablated layer", inherited from B1c/B1d. A failure
    # here is FATAL, not a warning: the gate results are written and the process exits
    # before a single generation is spent. There is deliberately NO gate on `permdir_all`
    # or `pc1_all` — a permuted-label direction is supposed to be unstable, and gating it
    # would silently select the permutations that happen to look like signal.
    stab_by_layer = {}
    for L in hs_needed:
        stab_by_layer[L] = C.split_half(feats[L], yp, a.estimator,
                                        n_per_half=min(N // 2, 600), seeds=range(10),
                                        pair_on=yo)
        print(f"[B1E gate] hs {L:>3} split-half({a.estimator}) "
              f"{stab_by_layer[L]['mean']:.3f} CI {stab_by_layer[L]['ci']} "
              f"per-class {[round(x, 3) for x in stab_by_layer[L]['per_class']]}",
              flush=True)
    abl_means = [stab_by_layer[L]["mean"] for L in abl_hs_all]
    layers_below_gate = [L for L in abl_hs_all
                         if not (stab_by_layer[L]["mean"] >= STABILITY_GATE)]
    stab = stab_by_layer[focus]
    stab_logreg = C.split_half(feats[focus], yp, "logreg", n_per_half=min(N // 2, 600),
                               seeds=range(10))
    # <<GATE_END>>

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
    print(f"[B1E GATE] rank-1 hs {abl_hs_all[0]}..{abl_hs_all[-1]}: min "
          f"{np.min(abl_means):.3f} median {np.median(abl_means):.3f} max "
          f"{np.max(abl_means):.3f}; {len(layers_below_gate)} layer(s) below the "
          f"{STABILITY_GATE} gate: {layers_below_gate}", flush=True)
    if not gate_pass:
        C.write_result(os.path.join(a.outdir, f"b1e_footprint_{a.tag}.json"),
                       {"model": a.model, "tag": a.tag, "aborted": "gate",
                        "focus": focus, "n_layers": h.n_layers,
                        "abl_hs_all": abl_hs_all,
                        "probe_n": N, "probe_pool_sha256": pool_sha,
                        "direction_stability": gate_block,
                        "pc1_vs_emo_cos": pc1_vs_emo_cos,
                        "pc1_dir_read_cos": pc1_dir_read_cos,
                        "pc1_geometry": pc1_geometry,
                        "verdict": "gate_failed", "rows": []}, prov)
        print(f"[B1E GATE] *** ABORTING: {len(layers_below_gate)} ablated layer(s) below "
              f"the {STABILITY_GATE} rank-1 split-half gate ({layers_below_gate}). "
              f"PREREG_B1e §3 says the run proceeds ONLY if every ablated layer clears it, "
              f"so no generation was performed. The per-layer gate results are in "
              f"{a.outdir}/b1e_footprint_{a.tag}.json ***", flush=True)
        print("B1E_GATE_FAILED", flush=True)
        sys.exit(2)

    # <<DIRS_START>>
    # ---------------- the three rank-1 directions, as model tensors ----------------------
    def tt(v):
        return torch.tensor(v, dtype=h.model.dtype, device=h.model.device)

    d_model = int(feats[focus].shape[1])
    emo_dirs_all, steer_dir = {}, {}
    for e in EMOS:
        ei = C.EMOTIONS.index(e)
        emo_dirs_all[e] = {L: tt(emo_np[e][L]) for L in abl_hs_all}
        steer_dir[e] = tt(C.raw_direction(dir_dec[focus], ei))
    # the SAME pc1 direction for every emotion (PREREG_B1e §3), materialised once
    pc1_dirs = {L: tt(pc1_np[L]) for L in abl_hs_all}

    _perm_cache = {}

    def permdir_dirs_for(e, rep):
        """`permdir_all`: ONE permutation of the DIR half's labels per (emotion, rep), the
        `dom` direction of that emotion's class index refitted on it, used at EVERY ablated
        layer.

        One permutation and not one per layer, because the point of this control is a
        coherent mislabelling of the same pool: the estimator sees the same rows, the same
        class counts and the same noise it saw for `emo_all`, and the only thing it has
        lost is which row belongs to which emotion. A fresh permutation per layer would
        instead average the control over 51 independent mislabellings and understate its
        variance. Cached per (emotion, rep) so a resumed run rebuilds the same vectors.
        """
        key = (e, rep)
        if key not in _perm_cache:
            s = perm_seed(a.seed, e, rep)
            y_perm = np.random.default_rng(s).permutation(yp[DIR])
            _perm_cache[key] = ({L: tt(permuted_direction(feats[L][DIR], y_perm, e,
                                                          seed=a.seed))
                                 for L in abl_hs_all}, s)
        return _perm_cache[key]
    # <<DIRS_END>>

    steer = C.Steer(h, steer_layer)
    # ONE Ablate over 13..n_layers-1; the per-arm `dirs` dict decides what each hook
    # projects. Every arm here passes 1-D tensors, so Ablate._mk takes the rank-1 path
    # (proj = (h . d) d) for all three — the arms are mechanically identical apart from the
    # direction itself, which is exactly what PREREG_B1e §3 asks for.
    ab = C.Ablate(h, abl_dec_all)
    rms = float(np.linalg.norm(feats[focus], axis=1).mean())
    print(f"[B1E] rms {rms:.1f}; d_model {d_model}; one Ablate over {len(abl_dec_all)} "
          f"decoder blocks", flush=True)

    # ---------------- the sweep ----------------------------------------------------------
    S = C.SCENARIOS
    rlog = C.ResponseLog(os.path.join(a.workdir, f"responses_b1e_{a.tag}.jsonl"))
    ck = C.Checkpoint(os.path.join(a.workdir, f"b1e_cells_{a.tag}.json"),
                      {"doses": DOSES, "arms": ARMS, "emos": EMOS, "reps": a.reps,
                       "est": a.estimator, "n": N, "focus": focus, "probe_k": PROBE_K,
                       "abl_hs_all": abl_hs_all,
                       "driver": "b1e"})

    def gen_A(e, alpha, rep):
        """A's steered message. Unchanged from b1d_subspace.py / b1c_alllayer.py, seed
        included, so A's half of the stimulus stays comparable across the runs."""
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

        'none' installs no hook at all. 'emo_all' passes the rank-1 raw emotion direction
        per layer (B1c's object exactly); 'permdir_all' this cell's permuted-label
        direction; 'pc1_all' the one pc1 direction per layer, the same for every emotion.
        All three are 1-D unit tensors over the SAME hs 13..n_layers-1, so nothing but the
        provenance of the removed direction differs between them.

        Returns (dirs, meta) where meta carries `k` and, for permdir_all, its seed.
        """
        if arm in NO_HOOK_ARMS:
            return None, {"k": 0}
        if arm == "emo_all":
            return emo_dirs_all[e], {"k": 1}
        if arm == "pc1_all":
            return pc1_dirs, {"k": 1}
        if arm == "permdir_all":
            d, s = permdir_dirs_for(e, rep)
            return d, {"k": 1, "perm_seed": s}
        raise ValueError(f"unknown arm {arm!r}")

    def gen_B(ctx, arm, e, alpha, rep, bs=30):
        """B regenerates while the arm's direction is projected out of A's TOKEN SPAN
        during B's PREFILL. Seeded per batch; part 'B' shares one stream across arms.

        Returns removed_norm AND mask_n_positions, because removed_norm is a mean over the
        masked positions and over the ablated LAYERS. In B1e every hooked arm is rank 1
        over the same layers, so the three numbers ARE directly comparable — that
        comparison is the footprint check of §3 — but `mask_n_positions` and
        `n_layers_ablated` are still recorded next to it so the normalisation is on file.
        mask_hit_rate is None for `none` rather than a 1.0 that would read as "the span was
        found" when nothing was ever looked for.
        """
        dirs, _meta = dirs_for(arm, e, rep)
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

        who='A' is the §4.2 check and runs with the arm's ablation LIVE on A's span, so
        `emo_all` / `permdir_all` / `pc1_all` are comparable — and its DOSE-0 value is what
        §2's instrument_failed trigger is computed from. who='B' is the §4.1 SECONDARY
        OUTCOME: B's generated reply is appended to the conversation, the question asks
        about B, and NO hooks are installed — the outcome is read off a clean forward pass,
        exactly as the primary one is.

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

    def score_B(ctx, breps, bs=8):
        """PRIMARY OUTCOME: B's present-e and other-e on its own reply, under the HELD-OUT
        read probe, in a separate forward pass with ALL HOOKS OFF — which is what makes it
        valid under an all-layer ablation (PREREG §4.1)."""
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
                    breps, removed, hit, gseed, npos = gen_B(ctx, arm, e, alpha, rep)
                    pres, oth = score_B(ctx, breps)
                    # MC-footprint: is the affect still readable from the intervened
                    # context, and does the arm move the UNSTEERED (dose-0) readout?
                    ab_read = readout(ctx, arm, e, alpha, rep, who="A")
                    # SECONDARY outcome: the same question about B, on B's reply, hooks off
                    b_read = readout(ctx, arm, e, alpha, rep, who="B", replies=breps)
                    deg = [C.degeneracy(b) for b in breps]
                    ppl = C.perplexity(h, breps)
                    _d, meta = dirs_for(arm, e, rep)
                    cell.append({
                        "emotion": e, "alpha": alpha, "rep": rep, "arm": arm,
                        "scenario": [c["scenario"] for c in ctx],
                        "B_present_e": pres[:, ei].tolist(),
                        "B_other_e": oth[:, ei].tolist(),
                        "A_readout_e": a_read[:, ei].tolist(),
                        "ctx_readout_e": ab_read[:, ei].tolist(),
                        "ctx_readout_full": ab_read.mean(0).tolist(),
                        "B_readout_e": b_read[:, ei].tolist(),
                        "B_readout_full": b_read.mean(0).tolist(),
                        "k": meta["k"],
                        "removed_norm": removed, "mask_hit_rate": hit,
                        "mask_n_positions": npos,
                        "n_layers_ablated": (0 if arm in NO_HOOK_ARMS
                                             else len(_d) if _d else 0),
                        "degenerate_frac": float(np.mean([C.is_degenerate(d) for d in deg])),
                        "refusal_frac": float(np.mean([d["refusal"] for d in deg])),
                        "distinct2": float(np.mean([d["distinct2"] for d in deg])),
                        "n_words": float(np.mean([d["n_words"] for d in deg])),
                        "perplexity": float(np.median(ppl)),
                        "gen_seed": gseed,
                        "perm_seed": meta.get("perm_seed"),
                    })
                    r = cell[-1]
                    print(f"[{e:>9} a{alpha} r{rep} {arm:>11}] present "
                          f"{np.mean(r['B_present_e']):+.2f} B_readout "
                          f"{np.mean(r['B_readout_e']):.3f} ctx_readout "
                          f"{np.mean(r['ctx_readout_e']):.3f} removed {removed:.2f} "
                          f"(k={r['k']}) over "
                          f"{'--' if npos is None else f'{npos:.0f}'} pos x "
                          f"{r['n_layers_ablated']} layers deg "
                          f"{r['degenerate_frac']:.2f} ppl {r['perplexity']:.1f} "
                          f"seed {gseed}", flush=True)
                ck.put(cid, cell)
                rows.extend(cell)
                print(f"[B1E] {len(rows)} arm-rows, {time.time()-t_start:.0f}s elapsed",
                      flush=True)

    # ---------------- analysis -----------------------------------------------------------
    res = analyze(rows, DOSES, EMOS)
    print_summary(res)

    out = {
        "model": a.model, "tag": a.tag, "focus": focus, "steer_layer": steer_layer,
        "n_layers": h.n_layers,
        "abl_hs_all": abl_hs_all, "n_abl_blocks_all": len(abl_hs_all),
        "probe_n": N, "dir_half_n": len(DIR), "read_half_n": len(READ),
        "probe_pool_path": pool_path, "probe_pool_sha256": pool_sha,
        "estimator": a.estimator, "rms": rms, "doses": list(DOSES), "arms": list(ARMS),
        "prespec_emotions": list(PRESPEC_EMOTIONS),
        "prereg": prereg_stamp(),
        "predecessors": ["results/rev3/b1c_alllayer_qwen36-27b.json",
                         "results/rev3/b1d_subspace_qwen36-27b.json"],
        "direction_stability": gate_block,
        "pc1_vs_emo_cos": pc1_vs_emo_cos,
        "pc1_dir_read_cos": pc1_dir_read_cos,
        "pc1_geometry": pc1_geometry,
        "footprint": res["footprint"],
        "frozen_token_audit": C.frozen_token_audit({
            "generated": ["B's reply (regenerated under the intervention, arms "
                          "emo_all/permdir_all/pc1_all)",
                          "the readout answer token on A's span (produced with the "
                          "intervention live)",
                          "the readout answer token on B's reply (hooks OFF)"],
            "frozen": ["A's steered message (generated BEFORE any ablation and never "
                       "rewritten — B1e has no token-level arm)",
                       "the scenario setting and B's opening line (fixed stimuli)"]}),
        "verdict": res["verdict"], "decision": res["decision"], "counts": res["counts"],
        "quality": res["quality"], "summary": res["per_emotion"], "rows": rows,
    }
    C.write_result(os.path.join(a.outdir, f"b1e_footprint_{a.tag}.json"), out, prov)
    fp = res["footprint"]
    print(f"\n[B1E] footprint: mean removed_norm emo {_f(fp['mean_removed_norm'].get('emo_all'), 2)} "
          f"permdir {_f(fp['mean_removed_norm'].get('permdir_all'), 2)} pc1 "
          f"{_f(fp['mean_removed_norm'].get('pc1_all'), 2)}; permdir/emo "
          f"{_f(fp['ratio_permdir_over_emo'], 2)} pc1/emo "
          f"{_f(fp['ratio_pc1_over_emo'], 2)}; MC-steer pass "
          f"{res['counts']['n_mc_steer_pass']}/{len(EMOS)}; permdir dose-0 shifts too far "
          f"{res['counts']['n_permdir_dose0_bad']}", flush=True)
    print(f"VERDICT: {res['verdict']}", flush=True)
    print("B1E_DONE", flush=True)


if __name__ == "__main__":
    main()

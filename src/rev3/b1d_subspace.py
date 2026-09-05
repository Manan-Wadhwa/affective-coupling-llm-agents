#!/usr/bin/env python3
"""B1d — RANK-k SUBSPACE ablation: does a linear affect subspace along A's span carry the
agent-to-agent transfer? (docs/planning/PREREG_B1d.md, committed before this file existed.)

This driver implements that pre-registration and nothing else. Where the pre-registration
says a quantity is fixed in advance, it is written here as a constant, not chosen at run
time. It is a copy of `b1c_alllayer.py` with the rank-1 arm kept as a bridge, the token
(rewrite) arms deleted, and the rank-5 class-mean subspace plus its two footprint-matched
controls added.

HYPOTHESES, stated before the data (PREREG_B1d §2, verbatim)

  - **H2 (linear subspace carries it):** median blocked fraction of `sub_all` vs `perm_all`
    over testable emotions >= 0.60 with the lower CI bound >= 0.40.
  - **H2' (not even the class-mean subspace carries it):** median <= 0.35 with the upper CI
    bound <= 0.50, *while the subspace ablation does remove the readable affect on A's
    span* (§4.2).
  - **indeterminate** otherwise.
  - **instrument_failed:** the subspace ablation fails §4.2 for >= 3 of the 6 emotions.
    This is reported as its own finding — the readable affect on A's span is not confined
    to the rank-5 class-mean subspace — and the blocked fractions are then descriptive
    only.

  The verdict string is one of: "H2", "H2prime", "indeterminate", "instrument_failed".

DECISION RULE, fixed in advance (PREREG_B1d §5)

  - Per-arm dose-response slope: OLS over the four doses on per-scenario means; CI by
    **scenario-blocked** bootstrap (29 clusters).
  - Contrasts, scenario-blocked AND dose-paired
    (`acl_core.paired_slope_contrast(block=..., pair_doses=True)`, 5000 draws), in this
    order: `sub_all - perm_all` (decision), `sub_all - randsub_all`, `emo_all - perm_all`,
    `sub_all - emo_all`, `perm_all - none`, `randsub_all - none`, `emo_all - none`.
  - **Testable emotions:** those whose `none` slope CI excludes zero. Others are reported
    but do not enter the decision.
  - **Blocked fraction** = -(arm - control contrast) / `none` slope, sign-aware, with the
    CI-implied range; the denominator's own CI is reported next to it.
  - **Decision:** the median blocked fraction of `sub_all` vs `perm_all` over testable
    emotions, with a bootstrap CI over scenarios (numerator and denominator recomputed per
    draw). Thresholds in §2 above. Per-emotion counts (BH-FDR over the testable emotions)
    are descriptive and do not override the median rule.
  - The extra rank beyond 1 is read from `sub_all - emo_all` (descriptive, with its CI as
    an extra blocked-fraction bound).

ARMS (PREREG_B1d §3; five — no rewrite arms, "three designs in a row failed their own
checks; the token channel is out of scope here")

  none         no ablation                                                    baseline
  emo_all      B1c's rank-1 emotion direction out of A's span, hs 13..n-1     the bridge to
                                                                              report 22
  sub_all      the rank-5 class-mean SUBSPACE out of A's span, hs 13..n-1     the H2 vs H2'
                                                                              test
  perm_all     the SAME estimator on the SAME DIR half with the present-      footprint-
               emotion labels PERMUTED; rank 5, same layers; one fresh        matched
               permutation per (emotion, rep)                                 control
  randsub_all  a Gaussian random orthonormal 5-frame, fresh per (emotion,     naive control
               rep) AND per ablated layer; same layers

  Only `none` installs no hook. B's replies are seeded with a common seed across arms
  (b1_followup's rule), so shared sampling noise cancels inside each paired arm contrast.

SUBSPACE (§3), per layer L in hs 13..n_layers-1, on the DIR half in RAW space
  the six present-emotion class means minus their grand mean; orthonormal basis of their
  span from the SVD, top 5 right singular vectors (the null direction dropped). The SAME
  subspace is used for every steered emotion — it is the linear affect representation, not
  a per-emotion direction. `acl_core.Ablate` takes the [d_model, k] frame and applies
  proj = (h @ Q) @ Q.T; the rank-1 path is untouched, so `emo_all` is bit-identical to
  B1c's arm of the same name.

GATES (§3), both fatal before a single generation is spent
  rank-1   `acl_core.split_half` mean cosine >= 0.80 at every ablated hidden state
  rank-5   mean principal-angle cosine between the DIR-half and READ-half class-mean
           subspaces >= 0.70 at every ablated hidden state (NEW in B1d)

OUTCOMES (§4)
  primary    B_present_e   — the READ probe at the focus layer on B's GENERATED reply, in a
                             separate forward pass with all hooks off (`score_B`).
  secondary  B_readout_e   — the forced-choice readout applied to B's reply, asking about
                             B, hooks off.
  MC         ctx_readout_e — the forced-choice readout on A's span with the arm's ablation
                             LIVE, so all four hooked arms are comparable (§4.2).

Usage: python b1d_subspace.py --model Qwen/Qwen3.6-27B --tag qwen36-27b --reps 3
       python b1d_subspace.py --selftest    # analysis + frames only, no model, no GPU
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

# ---- declared grid (PREREG_B1d §3; fixed in advance, not chosen at run time) ------------
DOSES = (0.0, 0.33, 0.67, 1.0)          # four points: the slope is no longer an endpoint
ARMS = ("none", "emo_all", "sub_all", "perm_all", "randsub_all")
NO_HOOK_ARMS = ("none",)
EMOS = ("desperate", "afraid", "happy", "calm", "sad", "angry")
STABILITY_GATE = 0.80                   # rank-1 split-half gate, unchanged from B1c
SUB_K = 5                               # §3: six class means minus their grand mean
SUB_GATE = 0.70                         # §3: NEW DIR-vs-READ subspace stability gate
PROBE_K = 60
PREREG = "docs/planning/PREREG_B1d.md"

# the seven contrasts of §5, in the pre-registered order
CONTRASTS = (("sub_all", "perm_all"), ("sub_all", "randsub_all"),
             ("emo_all", "perm_all"), ("sub_all", "emo_all"),
             ("perm_all", "none"), ("randsub_all", "none"), ("emo_all", "none"))
DECISION_CONTRAST = "sub_all_vs_perm_all"

# §4.4 quality thresholds, pre-declared (as in B1c)
MAX_BAD_FRAC = 0.10
MAX_PPL_RATIO = 2.0
# §2 decision thresholds
H2_MEDIAN, H2_LOWER = 0.60, 0.40
H2P_MEDIAN, H2P_UPPER = 0.35, 0.50
# §4.2 subspace instrument check
SUB_REMOVED_MIN = 0.80                  # `sub_all` must remove >= 80% of readable affect
CONTROL_REMOVED_MAX = 0.20              # a control is inert if it removes <= 20%
MC_FAIL_LIMIT = 3                       # >= 3 emotions failing => instrument_failed


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
    """Path, content hash and (best effort) git commit of the pre-registration (§8).

    PREREG_B1d §8: "The driver resolves this document relative to its own directory (B1c's
    relative lookup failed) and the document is staged next to the driver on the box." So
    the search starts NEXT TO THE DRIVER and only then falls back to the repo root; the
    path actually used is recorded, so a stamp with `found` false is never mistaken for a
    stamp of the wrong file.
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
    stream in all five arms — common random numbers, so sampling noise shared between two
    arms cancels inside each paired contrast instead of widening its CI. "readout" and
    "readout_B" keep `arm`, because those calls differ between arms by construction.
    """
    key = f"{seed}|{emotion}|{alpha:g}|{rep}|{'' if part == 'B' else arm}|{part}"
    return int(zlib.crc32(key.encode()) & 0x7FFFFFFF)


def rand_dir_seed(seed: int, emotion: str, rep: int) -> int:
    """Seed for the fresh random-subspace control of one (emotion, rep) cell.

    Derived from (--seed, the CANONICAL emotion index in acl_core.EMOTIONS, rep) so it is
    stable under a resumed run and independent of the order EMOS happens to be written in.
    One generator per cell draws a fresh orthonormal 5-frame at EVERY ablated layer.
    """
    ei = C.EMOTIONS.index(emotion)
    return int(zlib.crc32(f"rand|{seed}|{ei}|{rep}".encode()) & 0x7FFFFFFF)


def perm_seed(seed: int, emotion: str, rep: int) -> int:
    """Seed for the label permutation of one (emotion, rep) cell (§3 `perm_all`).

    Same shape as `rand_dir_seed` and for the same reason. ONE permutation of the DIR
    half's labels is drawn per cell and re-used at EVERY ablated layer, so `perm_all` is
    one coherent mislabelling of the pool rather than 51 independent ones — the footprint
    the B1c critique asked for: same data, same estimator noise, no emotion information.
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
# the subspaces (PREREG_B1d §3) — pure numpy, raw space, importable with no torch
# ---------------------------------------------------------------------------------------

def class_mean_frame(X: np.ndarray, y: np.ndarray, k: int = SUB_K) -> np.ndarray:
    """[d_model, k] ORTHONORMAL frame spanning the present-emotion class means, centred.

    §3: "the six present-emotion class means minus their grand mean; orthonormal basis of
    their span from the SVD, keeping the top 5 right singular vectors (rank 5, the null
    direction dropped)".

    The grand mean subtracted is the mean OF THE CLASS MEANS, not of the samples — with
    unequal class counts those differ, and the class-count-weighted version leaves a
    residual along the majority class. Centring six vectors leaves rank <= 5, which is why
    k = 5 drops exactly the null direction and nothing that carries class information.

    Raw space, on purpose: this is what `acl_core.Ablate` projects with, and it is the same
    space `acl_core.raw_direction` returns the rank-1 direction in.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    labs = sorted({int(v) for v in np.asarray(y).ravel().tolist()})
    M = np.stack([X[y == c].mean(0) for c in labs])          # [n_cls, d]
    M = M - M.mean(0)                                        # grand mean of the CLASS means
    Vt = np.linalg.svd(M, full_matrices=False)[2]            # right singular vectors
    kk = int(min(k, Vt.shape[0]))
    return np.ascontiguousarray(Vt[:kk].T, dtype=np.float64)  # [d, kk]


def perm_frame(X: np.ndarray, y: np.ndarray, rng: np.random.Generator,
               k: int = SUB_K) -> np.ndarray:
    """§3 `perm_all`: the SAME estimator on the SAME rows with the labels permuted.

    Identical data, identical class counts, identical estimator noise, no emotion
    information — so a `sub_all` effect that survives this contrast cannot be the footprint
    of "project out five directions fitted on this pool".
    """
    return class_mean_frame(X, rng.permutation(np.asarray(y)), k)


def random_frame(d: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """§3 `randsub_all`: a Gaussian random orthonormal k-frame, [d, k], by QR.

    The sign convention (Q re-signed by the diagonal of R) makes the frame a function of
    the draw alone and not of the LAPACK build; it changes nothing downstream, because
    `Ablate` projects onto the SPAN and a column's sign cancels in Q Qᵀ.
    """
    Q, R = np.linalg.qr(rng.standard_normal((int(d), int(k))))
    return np.ascontiguousarray(Q[:, :k] * np.sign(np.diag(R) + 1e-12), dtype=np.float64)


def subspace_stability(Qa: np.ndarray, Qb: np.ndarray) -> float:
    """Mean cosine of the principal angles between span(Qa) and span(Qb) (§3 gate).

    For orthonormal frames the singular values of Qaᵀ Qb ARE those cosines, so this is the
    rank-k generalisation of the |cos| that `acl_core.split_half` reports for rank 1: 1.0
    for the same subspace, ~sqrt(k/d) for two independent random ones.
    """
    s = np.linalg.svd(np.asarray(Qa, float).T @ np.asarray(Qb, float), compute_uv=False)
    return float(np.mean(np.clip(s, 0.0, 1.0)))


# ---------------------------------------------------------------------------------------
# analysis — importable with no torch and no model, so it is testable (`--selftest`)
# ---------------------------------------------------------------------------------------

METRICS = ("B_present_e", "B_other_e", "A_readout_e", "ctx_readout_e", "B_readout_e")


def _rowkey(r) -> tuple:
    return (r["emotion"], r["arm"], float(r["alpha"]), int(r["rep"]))


def _index(rows, e, arm, d) -> dict:
    """{(rep, scenario): (row, position)} for one (emotion, arm, dose) cell.

    Everything downstream works off these keys rather than off array positions. B1d has no
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


def analyze(rows, doses, emos, arms=ARMS, n_boot: int = 5000, seed: int = 0,
            apply_quality_exclusion: bool = True) -> dict:
    """PREREG_B1d §4-§5 in one importable, CPU-only function.

    Returns {'per_emotion', 'counts', 'decision', 'verdict', 'quality', ...}. Every slope
    is scenario-blocked; every contrast is scenario-blocked AND dose-paired through
    `acl_core.paired_slope_contrast(block=..., pair_doses=True)`; the verdict is the median
    rule of §2/§5, with §4.2's instrument-failure override.
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
        # the decision fraction, its naive-control twin, the rank-1 bridge and the
        # extra-rank readout of §5's last bullet
        s["blocked_fraction"] = {n: blocked_fraction(s.get(n), ns) for n in
                                 ("sub_all_vs_perm_all", "sub_all_vs_randsub_all",
                                  "emo_all_vs_perm_all", "sub_all_vs_emo_all")}
        s["detectable_effect"] = {f"{x}_vs_{y}": detectable_effect(s.get(f"{x}_vs_{y}"))
                                  for x, y in CONTRASTS}

        # ---- §4 manipulation checks ---------------------------------------------------
        s["mc_steer"] = _mc_steer(good, e, doses, n_boot, seed)
        s["mc_subspace"] = _mc_subspace(good, e, doses, arms)

        # ---- §5 decision inputs: numerator and denominator on ONE scenario panel -------
        Pd = panel(good, e, ("none", "sub_all", "perm_all"), doses,
                   metrics=("B_present_e",))
        if Pd is not None:
            d0 = Pd["data"]["B_present_e"]
            sc, sl_s = block_slopes(doses, d0["sub_all"], Pd["blocks"])
            _, sl_p = block_slopes(doses, d0["perm_all"], Pd["blocks"])
            _, sl_n = block_slopes(doses, d0["none"], Pd["blocks"])
            dec_in[e] = {"scen": sc, "dd": sl_s - sl_p, "nn": sl_n}
            s["decision_input"] = {"n_scenarios": int(len(sc)),
                                   "mean_slope_diff": float(sl_s.mean() - sl_p.mean()),
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

    # §4.2: `sub_all` failing to remove the readable affect is the instrument failing; a
    # control that is not inert is REPORTED and does not override the verdict.
    n_sub_fail = sum(1 for e in emos
                     if per_emotion[e].get("mc_subspace", {}).get("sub_pass") is False)
    n_ctrl_bad = sum(1 for e in emos
                     if per_emotion[e].get("mc_subspace", {}).get("control_not_inert")
                     is True)

    decision = _decide(dec_in, testable, n_boot, seed)
    if n_sub_fail >= MC_FAIL_LIMIT:
        verdict = "instrument_failed"
    elif decision["median"] is None:
        verdict = "indeterminate"
    elif decision["median"] >= H2_MEDIAN and decision["ci"][0] >= H2_LOWER:
        verdict = "H2"
    elif decision["median"] <= H2P_MEDIAN and decision["ci"][1] <= H2P_UPPER:
        verdict = "H2prime"
    else:
        verdict = "indeterminate"

    counts = {"n_testable": len(testable), "n_untestable": len(emos) - len(testable),
              "n_block": len(block_e), "n_antiblock": len(anti_e), "n_null": len(null_e),
              "testable_emotions": testable, "blocking_emotions": block_e,
              "antiblocking_emotions": anti_e, "null_emotions": null_e,
              "n_sub_fail": n_sub_fail,
              "n_control_not_inert": n_ctrl_bad,
              "n_mc_steer_pass": sum(1 for e in emos if (per_emotion[e].get("mc_steer")
                                                         or {}).get("sig")),
              "bh_fdr_q": fdr, "p_values": pvals,
              "note": "counts are DESCRIPTIVE; the decision is the median rule (§5). "
                      "control_not_inert is reported and does NOT override the verdict "
                      "(§4.2)"}
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


def _mc_steer(rows, e, doses, n_boot, seed) -> dict | None:
    """§4.3 — did steering actually put the emotion into A's TEXT? Slope CI excludes 0."""
    P = panel(rows, e, ("none",), doses, metrics=("A_readout_e",))
    if P is None:
        return None
    out = C.slope_ci(doses, P["data"]["A_readout_e"]["none"], n_boot=n_boot, seed=seed,
                     block=P["blocks"])
    out["passes"] = bool(out["sig"] and out["slope"] > 0)
    return out


def _mc_subspace(rows, e, doses, arms) -> dict:
    """§4.2 SUBSPACE INSTRUMENT CHECK, per emotion, at the TOP dose, on A's span with the
    intervention LIVE (`ctx_readout_e`).

        share_removed[arm] = (none_top - arm_top) / (none_top - none_at_dose_0)

    The denominator is the FLOOR CORRECTION: the readout on an unsteered A is not 0, it is
    whatever the model guesses, so an ablation can only ever remove the part that steering
    put there. Dividing by the raw `none` level would make every arm look weaker than it is
    and would make the 0.80 threshold un-meetable by construction.

    `sub_all` PASSES if its share >= 0.80. `perm_all` and `randsub_all` are INERT if their
    shares are <= 0.20 each; a control that is not inert is flagged (`control_not_inert`)
    and, per §4.2, does not override the verdict. Everything is read off the balanced panel
    that carries all five arms, so `none` at dose 0 comes from the same scenarios as the
    top-dose numbers rather than from a wider set.
    """
    top, zero = float(doses[-1]), float(doses[0])
    mc = {"dose": top, "floor_dose": zero, "threshold_sub": SUB_REMOVED_MIN,
          "threshold_control": CONTROL_REMOVED_MAX}
    P = panel(rows, e, tuple(arms), doses, metrics=("ctx_readout_e",))
    if P is None:
        mc["sub_pass"] = None
        mc["passes"] = None
        return mc
    D = P["data"]["ctx_readout_e"]
    for arm in D:
        v = D[arm][top]
        mc[arm] = {"mean": float(np.nanmean(v)), "ci": C.ci_of(v), "n": int(len(v))}
    if "none" not in mc:
        mc["sub_pass"] = None
        mc["passes"] = None
        return mc
    none_top = mc["none"]["mean"]
    none_zero = float(np.nanmean(D["none"][zero]))
    span = none_top - none_zero
    mc["none_at_floor"] = none_zero
    mc["steering_span"] = float(span)
    if not np.isfinite(span) or abs(span) < 1e-9:
        # nothing for the ablation to remove: the check is undefined, not passed
        mc["share_removed"] = {}
        mc["sub_pass"] = None
        mc["passes"] = None
        return mc
    hooked = [arm for arm in ("emo_all", "sub_all", "perm_all", "randsub_all")
              if arm in mc]
    share = {arm: float((none_top - mc[arm]["mean"]) / span) for arm in hooked}
    mc["share_removed"] = share
    mc["drop_by_arm"] = {arm: float(none_top - mc[arm]["mean"]) for arm in hooked}
    if "sub_all" in share:
        mc["sub_pass"] = bool(share["sub_all"] >= SUB_REMOVED_MIN)
    else:
        mc["sub_pass"] = None
    if "perm_all" in share:
        mc["perm_inert"] = bool(share["perm_all"] <= CONTROL_REMOVED_MAX)
    if "randsub_all" in share:
        mc["randsub_inert"] = bool(share["randsub_all"] <= CONTROL_REMOVED_MAX)
    if "perm_inert" in mc and "randsub_inert" in mc:
        mc["control_not_inert"] = bool(not (mc["perm_inert"] and mc["randsub_inert"]))
    mc["separated"] = bool("sub_all" in mc and "perm_all" in mc
                           and mc["sub_all"]["ci"][1] < mc["perm_all"]["ci"][0])
    mc["passes"] = mc["sub_pass"]           # only `sub_all` gates the verdict (§4.2)
    return mc


def _decide(dec_in, testable, n_boot, seed) -> dict:
    """§5 DECISION: the median blocked fraction of `sub_all` vs `perm_all` over testable
    emotions, with a bootstrap CI OVER SCENARIOS.

    One scenario draw is shared by all emotions (the same 29 scenarios are run for every
    emotion), and numerator and denominator are recomputed inside each draw, so the CI is
    the CI of the ratio-of-means and not of a ratio of independent estimates.
    """
    use = [e for e in testable if e in dec_in and len(dec_in[e]["scen"]) > 1]
    out = {"median": None, "ci": [float("nan"), float("nan")], "n_emotions": len(use),
           "emotions": use, "per_emotion_fraction": {}, "n_scenarios": 0,
           "rule": f"H2 if median >= {H2_MEDIAN} and lower >= {H2_LOWER}; "
                   f"H2prime if median <= {H2P_MEDIAN} and upper <= {H2P_UPPER}; "
                   f"else indeterminate. instrument_failed overrides when the §4.2 "
                   f"subspace check fails for >= {MC_FAIL_LIMIT} emotions."}
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
    print(f"\n{'emotion':>10} {'MCsub':>6} {'MCstr':>6} {'shSub':>7} {'shPerm':>7} "
          f"{'shRand':>7} {'shEmo':>7} {'sub_all-perm_all':>26} {'blockfrac':>10} "
          f"{'det':>7} {'ctlOK':>6}")
    for e in res["emotions"]:
        s = per.get(e, {})
        mc = s.get("mc_subspace", {})
        sh = mc.get("share_removed", {})
        c = s.get(DECISION_CONTRAST)
        bf = s.get("blocked_fraction", {}).get(DECISION_CONTRAST, {}).get("point")
        de = s.get("detectable_effect", {}).get(DECISION_CONTRAST, {}).get("half_width")
        txt = (f"{c['diff']:+.3f} CI[{c['ci'][0]:+.3f},{c['ci'][1]:+.3f}]"
               if c else "--")
        print(f"{e:>10} {str(mc.get('sub_pass')):>6} "
              f"{str((s.get('mc_steer') or {}).get('passes')):>6} "
              f"{_f(sh.get('sub_all')):>7} {_f(sh.get('perm_all')):>7} "
              f"{_f(sh.get('randsub_all')):>7} {_f(sh.get('emo_all')):>7} {txt:>26} "
              f"{(f'{bf:+.3f}' if bf is not None else '--'):>10} "
              f"{_f(de):>7} "
              f"{str(not mc.get('control_not_inert')):>6}")
    print(f"\ntestable {cnt['n_testable']}/{len(res['emotions'])} · blocking "
          f"{cnt['n_block']} · anti-blocking {cnt['n_antiblock']} · null {cnt['n_null']} · "
          f"sub_all MC failures {cnt['n_sub_fail']} · controls not inert "
          f"{cnt['n_control_not_inert']} · quality-excluded cells "
          f"{res['quality']['n_excluded_cells']}")
    m = dec["median"]
    print(f"DECISION: median blocked fraction (sub_all vs perm_all) over "
          f"{dec['n_emotions']} testable emotions = "
          f"{('%.3f' % m) if m is not None else '--'} "
          f"CI[{dec['ci'][0]:.3f},{dec['ci'][1]:.3f}] over {dec['n_scenarios']} scenarios")
    print(f"VERDICT: {res['verdict']}")


def _f(x, nd=3):
    return "--" if x is None or (isinstance(x, float) and not np.isfinite(x)) \
        else f"{x:+.{nd}f}"


# ---------------------------------------------------------------------------------------
# --selftest — synthetic rows with a planted answer, plus the frame algebra (no torch)
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


def _synth_features(d=256, n_per=500, seed=1, sig=6.0, nuis=6.0, n_nuis=20):
    """A pool with the geometry the real one has: a 5-dimensional class-mean signal inside
    a much larger nuisance subspace, plus isotropic noise.

    The nuisance block matters. Without it a permuted-label frame would pick the SAME five
    directions the true one does, because the top principal directions of six random
    subsample means are the top directions of the total covariance — which is the signal
    subspace when nothing else has any variance. Residual streams are not like that, and
    neither is this fixture.

    Columns are scaled to unit variance so that `acl_core.fit_direction`'s per-dimension
    `sd` is a CONSTANT: `raw_direction` is then exactly proportional to the class-mean
    contrast (C[i] / sd with sd constant), which is what makes the "raw_direction lies in
    the frame" assertion below an exact algebraic claim rather than an approximation.
    """
    rng = np.random.default_rng(seed)
    B = np.linalg.qr(rng.standard_normal((d, SUB_K + n_nuis)))[0]
    S, Nu = B[:, :SUB_K], B[:, SUB_K:]
    U = rng.standard_normal((6, SUB_K))
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    X, y = [], []
    for c in range(6):
        Z = rng.standard_normal((n_per, d))
        Z = Z + (rng.standard_normal((n_per, n_nuis)) * nuis) @ Nu.T
        X.append(Z + sig * (U[c] @ S.T))
        y += [c] * n_per
    X = np.concatenate(X)
    X = X / (X.std(0) + 1e-12)
    return X, np.asarray(y), S


def _frame_selftest() -> None:
    """PREREG_B1d §3: the subspace estimator, its two controls and the gate statistic."""
    X, y, S_true = _synth_features()
    d = X.shape[1]
    rng = np.random.default_rng(0)

    Q = class_mean_frame(X, y, SUB_K)
    Qp = perm_frame(X, y, rng, SUB_K)
    Qr = random_frame(d, SUB_K, rng)
    assert Q.shape == (d, SUB_K) and Qp.shape == (d, SUB_K) and Qr.shape == (d, SUB_K)
    for nm, F in (("class_mean", Q), ("perm", Qp), ("random", Qr)):
        off = float(np.abs(F.T @ F - np.eye(SUB_K)).max())
        assert off < 1e-8, (nm, "frame is not orthonormal", off)
    print(f"[selftest] all three frames are orthonormal to <1e-8 (d={d}, k={SUB_K})")

    # the rank-1 direction of every emotion must LIE IN the rank-5 class-mean frame:
    # mu_i - mean(others) = (K/(K-1)) (mu_i - grand mean of class means), which is a row of
    # the very matrix the frame was SVD'd from. If this ever failed, `sub_all` would not be
    # a superset of `emo_all` and §5's `sub_all - emo_all` would not read as "extra rank".
    dec = C.fit_direction(X, y, "dom")
    cos = []
    for i in range(6):
        v = C.raw_direction(dec, i)
        p = Q @ (Q.T @ v)
        cos.append(float(v @ p / (np.linalg.norm(v) * np.linalg.norm(p) + 1e-30)))
    assert min(cos) > 0.999, ("a rank-1 emotion direction is NOT inside the class-mean "
                              "frame", min(cos), cos)
    print(f"[selftest] every raw_direction lies in the class-mean frame "
          f"(min cosine with its projection {min(cos):.6f})")

    assert abs(subspace_stability(Q, Q) - 1.0) < 1e-9, subspace_stability(Q, Q)
    r1, r2 = random_frame(512, SUB_K, rng), random_frame(512, SUB_K, rng)
    rr = subspace_stability(r1, r2)
    assert rr < 0.2, ("two independent random 5-frames in d=512 should barely overlap", rr)
    print(f"[selftest] subspace_stability: self 1.000, two random d=512 frames {rr:.3f}")

    pc = subspace_stability(Qp, Q)
    assert pc < 0.5, ("the permuted-label frame must not recover the class-mean subspace",
                      pc)
    sig_rec = subspace_stability(Q, S_true)
    print(f"[selftest] permuted-label frame vs class-mean frame {pc:.3f} < 0.5 "
          f"(the class-mean frame recovers the planted signal at {sig_rec:.3f})")

    # the §3 gate statistic itself, on disjoint halves of the same pool
    idx = np.random.default_rng(3).permutation(len(y))
    DIR, READ = idx[:len(y) // 2], idx[len(y) // 2:]
    g = subspace_stability(class_mean_frame(X[DIR], y[DIR], SUB_K),
                           class_mean_frame(X[READ], y[READ], SUB_K))
    assert g >= SUB_GATE, ("the fixture must clear the gate it demonstrates", g)
    print(f"[selftest] DIR/READ subspace stability {g:.3f} >= gate {SUB_GATE}")


# planted §4.2 shares: none rises from the floor with dose; sub_all removes nearly all of
# that rise, emo_all half of it, and the two controls almost none.
READ_FLOOR = 0.20
READ_TOP = {"none": 0.70, "emo_all": 0.45, "sub_all": 0.22, "perm_all": 0.66,
            "randsub_all": 0.67}


def _synth_rows(doses, emos, arms, frac_by_emotion, n_scen=10, reps=2, seed=7,
                blocking_sd=0.0, read_top=None):
    """Rows with a KNOWN answer.

    `frac_by_emotion[e]` is the blocked fraction planted for `sub_all` vs `perm_all`: the
    none, perm_all, randsub_all and emo_all arms all get slope 1.0, `sub_all` gets 1 - f,
    so -(slope(sub_all) - slope(perm_all)) / slope(none) == f exactly.

    Every arm in one (emotion, dose, rep) cell shares its noise draw, so an arm with no
    planted effect is IDENTICAL to its control rather than merely similar and its contrast
    is deterministically null. A per-scenario slope offset (shared by all arms, and CENTRED
    so the none-arm slope is exactly 1.0) gives the scenario-blocked machinery real
    clustering to handle.

    `ctx_readout_e` is planted at READ_FLOOR at dose 0 for EVERY arm and at
    `read_top[arm]` above it, so §4.2's floor-corrected share is exact:
    (0.70 - 0.22) / (0.70 - 0.20) = 0.96 for `sub_all`, 0.08 / 0.06 for the controls.

    `blocking_sd` > 0 makes HOW MUCH is blocked itself vary by scenario (f -> f(1 + w_s),
    w centred), which is what a genuinely mixed result looks like: the median blocked
    fraction is still f, but its scenario bootstrap widens.
    """
    rng = np.random.default_rng(seed)
    read_top = dict(READ_TOP if read_top is None else read_top)
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
        slope["sub_all"] = 1.0 - f
        for d in doses:
            for rep in range(reps):
                nz = rng.normal(0, 0.15, n_scen)            # shared across arms
                nzo = rng.normal(0, 0.15, n_scen)
                nzr = rng.normal(0, 0.02, n_scen)
                a_read = 0.2 + 0.5 * d + rng.normal(0, 0.05, n_scen)
                for arm in arms:
                    keep = list(range(n_scen))
                    base = READ_FLOOR if d == 0 else read_top[arm]
                    sl = ((1.0 - blocked) if arm == "sub_all"
                          else np.full(n_scen, slope[arm]))
                    rows.append({
                        "emotion": e, "alpha": float(d), "rep": rep, "arm": arm,
                        "scenario": keep,
                        "B_present_e": ((sl + scen_off) * d + nz).tolist(),
                        "B_other_e": (0.05 * d + nzo).tolist(),
                        "A_readout_e": a_read.tolist(),
                        "ctx_readout_e": (base + nzr).tolist(),
                        "ctx_readout_full": [1.0 / 6] * 6,
                        "B_readout_e": (0.15 + 0.30 * slope[arm] * d + nzr).tolist(),
                        "k": (0 if arm in NO_HOOK_ARMS
                              else 1 if arm == "emo_all" else SUB_K),
                        "removed_norm": 0.0 if arm in NO_HOOK_ARMS else 3.2,
                        "mask_hit_rate": None if arm in NO_HOOK_ARMS else 1.0,
                        "mask_n_positions": None if arm in NO_HOOK_ARMS else 46.0,
                        "n_layers_ablated": 0 if arm in NO_HOOK_ARMS else 51,
                        "degenerate_frac": 0.02, "refusal_frac": 0.0,
                        "distinct2": 0.9, "n_words": 40.0, "perplexity": 12.0,
                        "gen_seed": gen_seed(0, e, d, rep, arm, "B"),
                        "perm_seed": (perm_seed(0, e, rep) if arm == "perm_all" else None),
                        "rand_dir_seed": (rand_dir_seed(0, e, rep)
                                          if arm == "randsub_all" else None),
                    })
    return rows


def selftest() -> int:
    t0 = time.time()
    _frame_selftest()
    _contrast_selftest()

    # --- H2: the class-mean subspace carries most of the transfer -----------------------
    plant = {e: 0.85 for e in EMOS}
    rows = _synth_rows(DOSES, EMOS, ARMS, plant)
    print(f"\n[selftest] {len(rows)} synthetic arm-rows; planted blocked fraction 0.85 "
          f"for sub_all vs perm_all", flush=True)
    # n_boot below the 5000 default only to keep this fast on a CPU box; the planted
    # contrasts are 0 or a fixed fraction with a hair-width CI, so no significance call
    # here depends on the bootstrap size.
    res = analyze(rows, DOSES, EMOS, n_boot=500)
    print_summary(res)
    per, cnt, dec = res["per_emotion"], res["counts"], res["decision"]

    assert cnt["n_untestable"] == 0, cnt
    assert cnt["n_sub_fail"] == 0, cnt
    assert cnt["n_control_not_inert"] == 0, cnt
    assert cnt["n_mc_steer_pass"] == len(EMOS), cnt
    for e in EMOS:
        s = per[e]
        assert s["_complete"] and all(s.get(a) for a in ARMS), (e, s["_complete"])
        assert s["sub_all"]["k"] == SUB_K and s["emo_all"]["k"] == 1, (e, s["sub_all"]["k"])
        assert s["sub_all"]["n_layers_ablated"] == 51, e
        # the NUMERATOR is exact: sub_all and perm_all share their noise draw in every
        # cell, so the planted slope difference survives the dose-paired contrast with no
        # residual at all. The FRACTION carries the `none` denominator's own sampling
        # error (a slope estimated from 10 scenarios x 2 reps), which is where the 0.12
        # slack below comes from -- and that error is itself reported as denominator_ci.
        assert abs(s[DECISION_CONTRAST]["diff"] + plant[e]) < 1e-9, (e,
                                                                     s[DECISION_CONTRAST])
        bf = s["blocked_fraction"][DECISION_CONTRAST]
        assert bf["defined"] and abs(bf["point"] - plant[e]) < 0.12, (e, bf, plant[e])
        assert bf["denominator_ci"] is not None and all(np.isfinite(bf["range"])), (e, bf)
        for name in ("sub_all_vs_randsub_all", "emo_all_vs_perm_all",
                     "sub_all_vs_emo_all"):
            assert name in s["blocked_fraction"], (e, name)
        de = s["detectable_effect"][DECISION_CONTRAST]["half_width"]
        assert de is not None and np.isfinite(de) and de > 0, (e, de)
        # perm_all and randsub_all were planted equal to none: neither may read as blocking
        assert s["perm_all_vs_none"]["sig"] is False, (e, s["perm_all_vs_none"])
        assert s["randsub_all_vs_none"]["sig"] is False, (e, s["randsub_all_vs_none"])
        mc = s["mc_subspace"]
        assert mc["sub_pass"] is True and mc["separated"] is True, (e, mc)
        assert mc["perm_inert"] and mc["randsub_inert"], (e, mc)
        assert abs(mc["share_removed"]["sub_all"] - 0.96) < 0.05, (e, mc["share_removed"])
        for name in [f"{x}_vs_{y}" for x, y in CONTRASTS]:
            assert s[name]["n_blocks"] > 1, (e, name, s[name])
    assert res["verdict"] == "H2", (res["verdict"], dec)
    print(f"[selftest] H2 reproduced: median {dec['median']:.3f} "
          f"CI[{dec['ci'][0]:.3f},{dec['ci'][1]:.3f}] -> {res['verdict']}")

    # --- the rule must MOVE with the data: all three verdicts on a re-planted grid -------
    small = dict(n_scen=8, reps=2)
    h2p = analyze(_synth_rows(DOSES, EMOS, ARMS, {e: 0.10 for e in EMOS}, **small),
                  DOSES, EMOS, n_boot=300)
    assert h2p["verdict"] == "H2prime", (h2p["verdict"], h2p["decision"])
    mid = analyze(_synth_rows(DOSES, EMOS, ARMS, {e: 0.45 for e in EMOS}, **small),
                  DOSES, EMOS, n_boot=300)
    assert mid["verdict"] == "indeterminate", (mid["verdict"], mid["decision"])
    print(f"[selftest] verdict moves H2 ({dec['median']:.2f}) / H2prime "
          f"({h2p['decision']['median']:.2f}) / indeterminate "
          f"({mid['decision']['median']:.2f}) with the planted fraction")

    # --- §4.2 instrument failure: sub_all leaves the readout HIGH -----------------------
    bad = _synth_rows(DOSES, EMOS, ARMS, {e: 0.85 for e in EMOS}, **small,
                      read_top={**READ_TOP, "sub_all": 0.62})   # share 0.16 < 0.80
    binst = analyze(bad, DOSES, EMOS, n_boot=200)
    assert binst["verdict"] == "instrument_failed", (binst["verdict"], binst["counts"])
    assert binst["counts"]["n_sub_fail"] >= MC_FAIL_LIMIT, binst["counts"]
    print(f"[selftest] sub_all removing only "
          f"{binst['per_emotion'][EMOS[0]]['mc_subspace']['share_removed']['sub_all']:.2f} "
          f"of the readable affect for {binst['counts']['n_sub_fail']} emotions overrides "
          f"a median of {binst['decision']['median']:.2f} -> instrument_failed")

    # --- §4.2 control_not_inert: reported, and it does NOT touch the verdict -------------
    noisy = _synth_rows(DOSES, EMOS, ARMS, {e: 0.85 for e in EMOS}, **small,
                        read_top={**READ_TOP, "perm_all": 0.40})  # share 0.60 > 0.20
    bctl = analyze(noisy, DOSES, EMOS, n_boot=200)
    assert bctl["counts"]["n_control_not_inert"] == len(EMOS), bctl["counts"]
    assert bctl["counts"]["n_sub_fail"] == 0, bctl["counts"]
    assert bctl["verdict"] == "H2", (bctl["verdict"], bctl["decision"])
    assert all(bctl["per_emotion"][e]["mc_subspace"]["perm_inert"] is False for e in EMOS)
    print(f"[selftest] a non-inert perm_all (share "
          f"{bctl['per_emotion'][EMOS[0]]['mc_subspace']['share_removed']['perm_all']:.2f})"
          f" is flagged for {bctl['counts']['n_control_not_inert']} emotions and leaves "
          f"the verdict at {bctl['verdict']}")

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
                    help="run analyze() and the frame algebra on synthetic data and exit; "
                         "loads no model")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if not a.model or not a.tag:
        ap.error("--model and --tag are required (or pass --selftest)")
    os.makedirs(a.outdir, exist_ok=True); os.makedirs(a.workdir, exist_ok=True)

    pool_path = os.path.join(a.workdir, f"probe_{a.tag}.jsonl")
    prov = C.Provenance(
        script="src/rev3/b1d_subspace.py",
        config={"doses": DOSES, "arms": ARMS, "emotions": EMOS, "reps": a.reps,
                "driver": "b1d", "estimator": a.estimator, "probe_k": PROBE_K,
                "stability_gate": STABILITY_GATE, "subspace_gate": SUB_GATE,
                "sub_k": SUB_K, "max_new_A": 110, "max_new_B": 110,
                "temp": 0.9, "top_p": 0.95,
                "n_scenarios": len(C.SCENARIOS),
                "contrasts": [f"{x}_vs_{y}" for x, y in CONTRASTS],
                "decision_contrast": DECISION_CONTRAST,
                "decision_thresholds": {"H2_median_min": H2_MEDIAN,
                                        "H2_lower_min": H2_LOWER,
                                        "H2prime_median_max": H2P_MEDIAN,
                                        "H2prime_upper_max": H2P_UPPER,
                                        "sub_removed_min": SUB_REMOVED_MIN,
                                        "control_removed_max": CONTROL_REMOVED_MAX,
                                        "mc_fail_limit": MC_FAIL_LIMIT},
                "quality": {"max_bad_frac": MAX_BAD_FRAC,
                            "max_perplexity_ratio": MAX_PPL_RATIO},
                "gen_seed_rule": "crc32('seed|emotion|alpha|rep|arm|part') & 0x7fffffff, "
                                 "+ batch start index; `arm` is EMPTY for part 'B' so all "
                                 "five arms sample B's reply with common random numbers, "
                                 "and kept for parts 'readout' and 'readout_B'",
                "perm_seed_rule": "crc32('perm|seed|EMOTIONS.index(e)|rep') & 0x7fffffff; "
                                  "ONE permutation of the DIR half's labels per "
                                  "(emotion, rep), re-used at EVERY ablated layer",
                "rand_dir_seed_rule": "crc32('rand|seed|EMOTIONS.index(e)|rep') & "
                                      "0x7fffffff; one generator per (emotion, rep) draws "
                                      "a FRESH orthonormal 5-frame at EVERY ablated layer",
                "prereg": prereg_stamp(),
                "probe_pool_path": pool_path,
                "predecessors": ["results/rev3/b1_e4rerun_qwen36-27b.json",
                                 "results/rev3/b1_followup_qwen36-27b.json",
                                 "results/rev3/b1c_alllayer_qwen36-27b.json"]},
        model_id=a.model, model_revision="", seeds={"pool": a.seed, "run": a.seed},
        code_sha=C.code_hash(os.path.abspath(__file__), os.path.abspath(C.__file__)),
        control_pointers={
            # Every span below is read out of THIS file's AST at import time by fn_lines /
            # block_lines — never typed, so a pointer cannot drift away from the code it
            # names; a name that stops existing renders as "NOT FOUND — pointer is stale".
            "per_layer_gate": f"b1d_subspace.py main ({block_lines('GATE')}): "
                              f"acl_core.split_half at EVERY ablated hidden_states index "
                              f"(13..n_layers-1) plus the focus layer, n_per_half="
                              f"min(N//2,600), 10 seeds; the run EXITS without generating "
                              f"anything if any ablated layer's class mean is below "
                              f"{STABILITY_GATE} (PREREG §3, identical to B1c)",
            "subspace_gate": f"b1d_subspace.py main ({block_lines('SUBGATE')}) via "
                             f"::subspace_stability ({fn_lines('subspace_stability')}): "
                             f"the mean cosine of the principal angles between the "
                             f"DIR-half and READ-half rank-{SUB_K} class-mean frames must "
                             f"be >= {SUB_GATE} at EVERY ablated hidden state; every value "
                             f"is reported in direction_stability.subspace and the run "
                             f"EXITS before any generation if one fails (PREREG_B1d §3, "
                             f"NEW in this run)",
            "subspace_estimator": f"b1d_subspace.py::class_mean_frame "
                                  f"({fn_lines('class_mean_frame')}): on the DIR half in "
                                  f"RAW space, the class means of the present emotions "
                                  f"minus the grand mean OF THE CLASS MEANS, SVD, top "
                                  f"{SUB_K} right singular vectors — one orthonormal "
                                  f"[d_model, {SUB_K}] frame per ablated layer, THE SAME "
                                  f"frame for every steered emotion (PREREG_B1d §3)",
            "subspace_ablation": f"b1d_subspace.py main ({block_lines('LAYERS')}) and "
                                 f"::dirs_for ({fn_lines('dirs_for')}): ONE acl_core."
                                 f"Ablate over decoder blocks 12..n_layers-2 (hs "
                                 f"13..n_layers-1); Ablate applies proj = (h @ Q) @ Q.T on "
                                 f"a 2-D frame and the unchanged rank-1 path on a 1-D "
                                 f"direction, so `emo_all` is bit-identical to B1c's arm "
                                 f"of that name and `sub_all` differs from it only in the "
                                 f"rank of the object removed; per-row `k`, "
                                 f"`removed_norm`, `mask_n_positions` and "
                                 f"`n_layers_ablated` record what was actually taken out",
            "permutation_control": f"b1d_subspace.py::perm_frame "
                                   f"({fn_lines('perm_frame')}) seeded by ::perm_seed "
                                   f"({fn_lines('perm_seed')}), assembled in main's "
                                   f"::perm_dirs_for ({fn_lines('perm_dirs_for')}): the "
                                   f"SAME estimator on the SAME DIR-half rows with the "
                                   f"present-emotion labels PERMUTED — one permutation per "
                                   f"(emotion, rep), re-used at every ablated layer, so "
                                   f"the footprint (same data, same estimator noise, same "
                                   f"rank, same layers) is matched and only the emotion "
                                   f"information is gone; the seed is in every perm_all "
                                   f"row as perm_seed",
            "random_subspace_control": f"b1d_subspace.py::random_frame "
                                       f"({fn_lines('random_frame')}) -> QR of a Gaussian "
                                       f"[d_model, {SUB_K}], assembled in main's "
                                       f"::randsub_dirs_for "
                                       f"({fn_lines('randsub_dirs_for')}), a FRESH frame "
                                       f"per (emotion, rep) AND per ablated layer, seeded "
                                       f"by ::rand_dir_seed ({fn_lines('rand_dir_seed')}); "
                                       f"the seed is in every randsub_all row as "
                                       f"rand_dir_seed",
            "blocked_paired_contrast": f"acl_core.paired_slope_contrast(block=..., "
                                       f"pair_doses=True) -> acl_core."
                                       f"_blocked_paired_slope_contrast, called from "
                                       f"b1d_subspace.py::analyze ({fn_lines('analyze')}) "
                                       f"on the balanced scenario panel built by ::panel "
                                       f"({fn_lines('panel')}); one-sample bootstrap over "
                                       f"scenarios of the per-scenario paired slope "
                                       f"difference, 5000 draws, seed 0",
            "decision_rule": f"b1d_subspace.py::_decide ({fn_lines('_decide')}), called "
                             f"from ::analyze ({fn_lines('analyze')}); the median blocked "
                             f"fraction of {DECISION_CONTRAST} over testable emotions with "
                             f"a bootstrap CI over scenarios, thresholds "
                             f"{H2_MEDIAN}/{H2_LOWER}/{H2P_MEDIAN}/{H2P_UPPER} fixed in "
                             f"this file's module docstring and in PREREG_B1d §2 before "
                             f"the run; ::blocked_fraction "
                             f"({fn_lines('blocked_fraction')}) is sign-aware",
            "secondary_outcome": f"b1d_subspace.py::readout ({fn_lines('readout')}) with "
                                 f"who='B' and the reply appended to the conversation, "
                                 f"hooks OFF — stored per row as B_readout_e",
            "manipulation_checks": f"b1d_subspace.py::_mc_steer ({fn_lines('_mc_steer')}) "
                                   f"and ::_mc_subspace ({fn_lines('_mc_subspace')}) — the "
                                   f"latter is PREREG_B1d §4.2's floor-corrected share "
                                   f"removed, (none_top - arm_top) / (none_top - none at "
                                   f"dose 0) on ctx_readout_e at the top dose: sub_all "
                                   f"passes at >= {SUB_REMOVED_MIN}, a control is inert at "
                                   f"<= {CONTROL_REMOVED_MAX} and a non-inert control is "
                                   f"reported without overriding the verdict; "
                                   f"::quality_flags ({fn_lines('quality_flags')}) is §4.4",
            "reproducible_sampling": f"b1d_subspace.py::gen_seed ({fn_lines('gen_seed')}), "
                                     f"called immediately before every h.model.generate in "
                                     f"::gen_B ({fn_lines('gen_B')}) and before the readout "
                                     f"forward in ::readout ({fn_lines('readout')})",
            "non_circular_measurement": f"b1d_subspace.py main ({block_lines('POOL')}): "
                                        f"probe fitted on the READ half, steering "
                                        f"directions and ALL FOUR ablation objects "
                                        f"(rank-1, class-mean frame, permuted frame, "
                                        f"random frame) built on the disjoint DIR half; "
                                        f"the READ-half frames are used ONLY by the "
                                        f"subspace stability gate",
            "frozen_token_audit": "arms emo_all/sub_all/perm_all/randsub_all regenerate B "
                                  "under the intervention while A's message predates it; "
                                  "no arm rewrites A's message (PREREG_B1d §3: no rewrite "
                                  "arms). See out['frozen_token_audit'].",
        })

    h = C.load(a.model)
    prov.model_revision = h.revision

    # <<LAYERS_START>>
    # ---- the layer set (PREREG_B1d §3: every hooked arm ablates hs 13..n_layers-1) ------
    # `lo` is built exactly as b1c_alllayer.py built it, so `emo_all` is B1c's arm and not
    # a near-miss. The set runs to n_layers-1: hs n_layers is the output of the LAST block,
    # which feeds only the final norm and the LM head, so projecting there cannot change
    # what B attends to — it would be a no-op arm dressed as an ablation.
    focus = h.focus(); steer_layer = focus - 1
    lo = round(0.2 * h.n_layers)
    abl_hs_all = list(range(lo, h.n_layers))                # 13..63 at n_layers=64
    abl_dec_all = [L - 1 for L in abl_hs_all]               # decoder blocks 12..62
    hs_needed = sorted(set(abl_hs_all) | {focus})
    # <<LAYERS_END>>
    print(f"[B1D] focus {focus} steer_layer {steer_layer} n_layers {h.n_layers}; "
          f"all hooked arms hs {abl_hs_all[0]}..{abl_hs_all[-1]} "
          f"({len(abl_hs_all)} blocks); subspace rank {SUB_K}", flush=True)

    # <<POOL_START>>
    # ---------------- probe pool, split into DIR and READ halves ------------------------
    # Same path, same k, same seed, same permutation as b1c_alllayer.py: with the pool file
    # from that run present in --workdir the rank-1 directions are the SAME vectors, which
    # is what makes `emo_all` a bridge to report 22 rather than a new measurement.
    items = C.generate_pool(h, PROBE_K, pool_path, seed=a.seed, bs=a.gen_bs)
    feats, yp, yo = C.pool_features(h, items, hs_needed,
                                    cache=os.path.join(a.workdir, f"probefeat_{a.tag}.npz"))
    N = len(items)
    rs = np.random.default_rng(a.seed).permutation(N)
    DIR, READ = rs[:N // 2], rs[N // 2:]
    pool_sha = file_sha256(pool_path)
    print(f"[B1D] probe pool n={N}: DIR half {len(DIR)}, READ half {len(READ)}; "
          f"sha256 {pool_sha[:16]}...", flush=True)

    dir_dec = {L: C.fit_direction(feats[L][DIR], yp[DIR], a.estimator, seed=a.seed,
                                  pair_on=yo[DIR]) for L in hs_needed}
    read_dec = {L: C.fit_direction(feats[L][READ], yp[READ], a.estimator, seed=a.seed,
                                   pair_on=yo[READ]) for L in hs_needed}
    read_dec_other = C.fit_direction(feats[focus][READ], yo[READ], a.estimator, seed=a.seed)
    # the rank-5 class-mean frames: DIR half is what gets ABLATED, READ half exists ONLY to
    # feed the stability gate below and is never installed as a hook.
    sub_frames = {L: class_mean_frame(feats[L][DIR], yp[DIR], SUB_K) for L in abl_hs_all}
    sub_frames_read = {L: class_mean_frame(feats[L][READ], yp[READ], SUB_K)
                       for L in abl_hs_all}
    # <<POOL_END>>

    # <<GATE_START>>
    # ---- GATE 1 (PREREG §3): rank-1 split-half at EVERY ablated layer, 13..n_layers-1 ---
    # "the run proceeds only if the class mean is >= 0.80 at every ablated layer, and every
    # per-class value is reported". So a failure here is FATAL, not a warning: the gate
    # results are written and the process exits before a single generation is spent.
    stab_by_layer = {}
    for L in hs_needed:
        stab_by_layer[L] = C.split_half(feats[L], yp, a.estimator,
                                        n_per_half=min(N // 2, 600), seeds=range(10),
                                        pair_on=yo)
        print(f"[B1D gate] hs {L:>3} split-half({a.estimator}) "
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

    # <<SUBGATE_START>>
    # ---- GATE 2 (PREREG_B1d §3, NEW): rank-5 subspace stability, DIR half vs READ half --
    # "for every ablated layer, the mean cosine of the principal angles between the
    # DIR-half and READ-half subspaces must be >= 0.70; every value is reported; the run
    # aborts before any generation if any layer fails (as B1c's gate did)". The two halves
    # are the same disjoint split the probe uses, so this asks the only question that
    # matters for an ablation: is the subspace we are about to project out a property of
    # the model, or of the half of the pool it was fitted on?
    sub_stab_by_layer = {L: subspace_stability(sub_frames[L], sub_frames_read[L])
                         for L in abl_hs_all}
    for L in abl_hs_all:
        print(f"[B1D subgate] hs {L:>3} rank-{SUB_K} DIR/READ principal cosine "
              f"{sub_stab_by_layer[L]:.3f}", flush=True)
    sub_means = [sub_stab_by_layer[L] for L in abl_hs_all]
    sub_below_gate = [L for L in abl_hs_all if not (sub_stab_by_layer[L] >= SUB_GATE)]
    # <<SUBGATE_END>>

    gate_pass = not layers_below_gate and not sub_below_gate
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
        "subspace": {
            "gate": SUB_GATE, "k": SUB_K, "gate_pass": not sub_below_gate,
            "statistic": "mean cosine of the principal angles between the DIR-half and "
                         "READ-half rank-5 class-mean frames",
            "by_layer": {str(L): float(sub_stab_by_layer[L]) for L in abl_hs_all},
            "layers_below_gate": sub_below_gate,
            "summary": {"min": float(np.min(sub_means)),
                        "median": float(np.median(sub_means)),
                        "max": float(np.max(sub_means))}},
    }
    print(f"[B1D GATE] rank-1 hs {abl_hs_all[0]}..{abl_hs_all[-1]}: min "
          f"{np.min(abl_means):.3f} median {np.median(abl_means):.3f} max "
          f"{np.max(abl_means):.3f}; {len(layers_below_gate)} layer(s) below the "
          f"{STABILITY_GATE} gate: {layers_below_gate}", flush=True)
    print(f"[B1D GATE] rank-{SUB_K} subspace: min {np.min(sub_means):.3f} median "
          f"{np.median(sub_means):.3f} max {np.max(sub_means):.3f}; "
          f"{len(sub_below_gate)} layer(s) below the {SUB_GATE} gate: {sub_below_gate}",
          flush=True)
    if not gate_pass:
        C.write_result(os.path.join(a.outdir, f"b1d_subspace_{a.tag}.json"),
                       {"model": a.model, "tag": a.tag, "aborted": "gate",
                        "focus": focus, "n_layers": h.n_layers,
                        "abl_hs_all": abl_hs_all, "sub_k": SUB_K,
                        "probe_n": N, "probe_pool_sha256": pool_sha,
                        "direction_stability": gate_block,
                        "verdict": "gate_failed", "rows": []}, prov)
        print(f"[B1D GATE] *** ABORTING: {len(layers_below_gate)} ablated layer(s) below "
              f"the {STABILITY_GATE} rank-1 split-half gate ({layers_below_gate}) and "
              f"{len(sub_below_gate)} below the {SUB_GATE} rank-{SUB_K} subspace gate "
              f"({sub_below_gate}). PREREG_B1d §3 says the run proceeds ONLY if every "
              f"ablated layer clears both, so no generation was performed. The per-layer "
              f"gate results are in {a.outdir}/b1d_subspace_{a.tag}.json ***", flush=True)
        print("B1D_GATE_FAILED", flush=True)
        sys.exit(2)

    # ---------------- directions and frames ----------------------------------------------
    def tt(v):
        return torch.tensor(v, dtype=h.model.dtype, device=h.model.device)

    d_model = int(feats[focus].shape[1])
    emo_dirs_all, steer_dir = {}, {}
    for e in EMOS:
        ei = C.EMOTIONS.index(e)
        emo_dirs_all[e] = {L: tt(C.raw_direction(dir_dec[L], ei)) for L in abl_hs_all}
        steer_dir[e] = tt(C.raw_direction(dir_dec[focus], ei))
    # the SAME rank-5 frame for every emotion (PREREG §3), materialised once
    sub_dirs = {L: tt(sub_frames[L]) for L in abl_hs_all}

    _perm_cache, _rand_cache = {}, {}

    def perm_dirs_for(e, rep):
        """`perm_all`: ONE permutation of the DIR half's labels per (emotion, rep), used at
        EVERY ablated layer.

        One permutation and not one per layer, because the point of this control is a
        coherent mislabelling of the same pool: the estimator sees the same rows, the same
        class counts and the same noise it saw for `sub_all`, and the only thing it has
        lost is which row belongs to which emotion. A fresh permutation per layer would
        instead average the control over 51 independent mislabellings and understate its
        variance.
        """
        key = (e, rep)
        if key not in _perm_cache:
            s = perm_seed(a.seed, e, rep)
            rr = np.random.default_rng(s)
            yperm = rr.permutation(yp[DIR])
            _perm_cache[key] = ({L: tt(class_mean_frame(feats[L][DIR], yperm, SUB_K))
                                 for L in abl_hs_all}, s)
        return _perm_cache[key]

    def randsub_dirs_for(e, rep):
        """`randsub_all`: a FRESH Gaussian orthonormal 5-frame per (emotion, rep) AND per
        ablated layer, from one generator per cell (the rand_dir_seed rule B1c used for its
        rank-1 random control, unchanged)."""
        key = (e, rep)
        if key not in _rand_cache:
            s = rand_dir_seed(a.seed, e, rep)
            rr = np.random.default_rng(s)
            _rand_cache[key] = ({L: tt(random_frame(d_model, SUB_K, rr))
                                 for L in abl_hs_all}, s)
        return _rand_cache[key]

    steer = C.Steer(h, steer_layer)
    # ONE Ablate over 13..n_layers-1; the per-arm `dirs` dict decides what each hook
    # projects. Ablate._mk dispatches on the ndim of the entry: 1-D is the unchanged
    # rank-1 path, 2-D is proj = (h @ Q) @ Q.T.
    ab = C.Ablate(h, abl_dec_all)
    rms = float(np.linalg.norm(feats[focus], axis=1).mean())
    print(f"[B1D] rms {rms:.1f}; d_model {d_model}; one Ablate over {len(abl_dec_all)} "
          f"decoder blocks", flush=True)

    # ---------------- the sweep ----------------------------------------------------------
    S = C.SCENARIOS
    rlog = C.ResponseLog(os.path.join(a.workdir, f"responses_b1d_{a.tag}.jsonl"))
    ck = C.Checkpoint(os.path.join(a.workdir, f"b1d_cells_{a.tag}.json"),
                      {"doses": DOSES, "arms": ARMS, "emos": EMOS, "reps": a.reps,
                       "est": a.estimator, "n": N, "focus": focus, "probe_k": PROBE_K,
                       "abl_hs_all": abl_hs_all, "sub_k": SUB_K,
                       "driver": "b1d"})

    def gen_A(e, alpha, rep):
        """A's steered message. Unchanged from b1c_alllayer.py / b1_followup.py, seed
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

        'none' installs no hook at all. 'emo_all' passes the rank-1 raw direction per
        layer (a 1-D tensor, B1c's object exactly); 'sub_all' the one class-mean frame;
        'perm_all' this cell's permuted-label frames; 'randsub_all' this cell's fresh
        random frames. Every hooked arm covers the SAME hs 13..n_layers-1, so nothing but
        the rank and the provenance of the removed object differs between them.

        Returns (dirs, meta) where meta carries `k` and whichever control seed applies.
        """
        if arm in NO_HOOK_ARMS:
            return None, {"k": 0}
        if arm == "emo_all":
            return emo_dirs_all[e], {"k": 1}
        if arm == "sub_all":
            return sub_dirs, {"k": SUB_K}
        if arm == "perm_all":
            d, s = perm_dirs_for(e, rep)
            return d, {"k": SUB_K, "perm_seed": s}
        if arm == "randsub_all":
            d, s = randsub_dirs_for(e, rep)
            return d, {"k": SUB_K, "rand_dir_seed": s}
        raise ValueError(f"unknown arm {arm!r}")

    def gen_B(ctx, arm, e, alpha, rep, bs=30):
        """B regenerates while the arm's direction or frame is projected out of A's TOKEN
        SPAN during B's PREFILL. Seeded per batch; part 'B' shares one stream across arms.

        Returns removed_norm AND mask_n_positions, because removed_norm is a mean over the
        masked positions and over the ablated LAYERS, and it is also a mean over a rank-k
        projection: a rank-5 arm removes more norm than a rank-1 one by construction, so
        the number is uninterpretable without `k`, `mask_n_positions` and
        `n_layers_ablated` next to it. mask_hit_rate is None for `none` rather than a 1.0
        that would read as "the span was found" when nothing was ever looked for.
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

        who='A' is the §4.2 manipulation check and runs with the arm's ablation LIVE on A's
        span, so `emo_all` / `sub_all` / `perm_all` / `randsub_all` are comparable. who='B'
        is the §4.1 SECONDARY OUTCOME: B's generated reply is appended to the conversation,
        the question asks about B, and NO hooks are installed — the outcome is read off a
        clean forward pass, exactly as the primary one is.

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
                    # MC-subspace: is the affect still readable from the intervened context?
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
                        "rand_dir_seed": meta.get("rand_dir_seed"),
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
                print(f"[B1D] {len(rows)} arm-rows, {time.time()-t_start:.0f}s elapsed",
                      flush=True)

    # ---------------- analysis -----------------------------------------------------------
    res = analyze(rows, DOSES, EMOS)
    print_summary(res)

    out = {
        "model": a.model, "tag": a.tag, "focus": focus, "steer_layer": steer_layer,
        "n_layers": h.n_layers,
        "abl_hs_all": abl_hs_all, "n_abl_blocks_all": len(abl_hs_all), "sub_k": SUB_K,
        "probe_n": N, "dir_half_n": len(DIR), "read_half_n": len(READ),
        "probe_pool_path": pool_path, "probe_pool_sha256": pool_sha,
        "estimator": a.estimator, "rms": rms, "doses": list(DOSES), "arms": list(ARMS),
        "prereg": prereg_stamp(),
        "predecessors": ["results/rev3/b1_e4rerun_qwen36-27b.json",
                         "results/rev3/b1_followup_qwen36-27b.json",
                         "results/rev3/b1c_alllayer_qwen36-27b.json"],
        "direction_stability": gate_block,
        "frozen_token_audit": C.frozen_token_audit({
            "generated": ["B's reply (regenerated under the intervention, arms "
                          "emo_all/sub_all/perm_all/randsub_all)",
                          "the readout answer token on A's span (produced with the "
                          "intervention live)",
                          "the readout answer token on B's reply (hooks OFF)"],
            "frozen": ["A's steered message (generated BEFORE any ablation and never "
                       "rewritten — B1d has no token-level arm)",
                       "the scenario setting and B's opening line (fixed stimuli)"]}),
        "verdict": res["verdict"], "decision": res["decision"], "counts": res["counts"],
        "quality": res["quality"], "summary": res["per_emotion"], "rows": rows,
    }
    C.write_result(os.path.join(a.outdir, f"b1d_subspace_{a.tag}.json"), out, prov)
    d = res["decision"]
    print(f"\n[B1D] median blocked fraction (sub_all vs perm_all) "
          f"{d['median'] if d['median'] is None else round(d['median'], 3)} "
          f"CI[{d['ci'][0]:.3f},{d['ci'][1]:.3f}] over {d['n_emotions']} testable "
          f"emotions; sub_all MC failures {res['counts']['n_sub_fail']}; controls not "
          f"inert {res['counts']['n_control_not_inert']}", flush=True)
    print(f"VERDICT: {res['verdict']}", flush=True)
    print("B1D_DONE", flush=True)


if __name__ == "__main__":
    main()

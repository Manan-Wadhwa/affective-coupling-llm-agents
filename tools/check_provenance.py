#!/usr/bin/env python3
"""The standing provenance mechanism (RESEARCH_PLAN rev 3, §0.3).

Every reported number carries a pointer to script, config and output file. The check is
mechanical: DOES THE POINTER RESOLVE, AND DOES THE CONFIG MATCH THE CLAIM.

This exists because re-deriving a claim does not catch the failure that produced all six
known errors -- *a claim asserted about what the code did, without tracing what the code
did*. Checking that its pointer resolves does.

Two error classes are caught here automatically:
  * unsourced numbers            -- fail at step one, with no file to point at
  * config mismatches            -- the alpha-grid error: the claim's config and the
                                    result file's own recorded config disagree

A third is NOT catchable by any value check, because such claims have resolving pointers
and matching configs: *the code does not do what its name says* (E3's frozen tokens, a
scramble that permutes within one level). For those, the registry carries `control_lines`
naming the LINES that implement the control, and this tool verifies the lines exist and
still contain the construct the claim depends on. A claim whose control_lines no longer
match its `control_expect` is reported as DRIFTED -- the code moved under the claim.

`value_ok` and `claim_status` are deliberately separate. E3's arithmetic resolves exactly;
the claim built on it is still retracted, because what the code measured is not what the
claim says it measured. A checker that conflated the two would re-certify the retraction.

Usage:
  python tools/check_provenance.py                 # check every claim
  python tools/check_provenance.py --status quoted # only claims still cleared for quoting
  python tools/check_provenance.py --json          # machine-readable
Exit code is non-zero if any claim marked `quotable` fails to resolve.
"""
import argparse, json, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGISTRY = os.path.join(ROOT, "docs", "review", "claims.json")

# provenance categories, RESEARCH_PLAN §0.2
CATEGORIES = {
    "a": "reproducible from committed files",
    "b": "run, but the artifact was lost (lease expiry) -- re-run and commit",
    "c": "no evidence of a run -- retract, and investigate how it entered",
    "d": "run in a prior phase, artifact external to this repo -- locate or re-run",
}


def dig(obj, path):
    """Resolve a dotted json path with [i] indices. Returns (ok, value)."""
    cur = obj
    for part in re.findall(r"[^.\[\]]+|\[\d+\]", path):
        if part.startswith("["):
            i = int(part[1:-1])
            if not isinstance(cur, list) or i >= len(cur):
                return False, None
            cur = cur[i]
        else:
            if not isinstance(cur, dict) or part not in cur:
                return False, None
            cur = cur[part]
    return True, cur


# ---- derived quantities: claims that are counts or ratios over a result file -----------

def _d_count_sig(blob, args):
    """How many emotions have a significant slope of the named kind."""
    key = args["key"]
    n = sum(1 for e, v in blob["summary"].items() if v.get(key, {}).get("sig"))
    return n


def _d_count_positive(blob, args):
    field = args["field"]
    return sum(1 for e, v in blob["results"].items() if v.get(field, 0) > 0)


def _d_present_gt_other(blob, args):
    return sum(1 for e, v in blob["summary"].items()
               if v["present"]["slope"] > v["other"]["slope"])


def _d_range(blob, args):
    """min/max of a field across the per-emotion slopes block."""
    vals = [v[args["field"]] for v in blob["slopes"].values() if args["field"] in v]
    return [round(min(vals), 4), round(max(vals), 4)]


def _d_count_true(blob, args):
    """How many per-emotion summary entries have a truthy value at a dotted path
    (e.g. mc_ablate.passes). Entries lacking the path count as false."""
    n = 0
    for e, v in blob["summary"].items():
        ok, got = dig(v, args["path"])
        n += int(bool(ok and got))
    return n


DERIVERS = {"count_sig": _d_count_sig, "count_positive": _d_count_positive,
            "present_gt_other": _d_present_gt_other, "range": _d_range,
            "count_true": _d_count_true}


def check_claim(c):
    r = {"id": c["id"], "category": c["category"], "claim_status": c["claim_status"],
         "text": c["text"], "problems": [], "value": None, "value_ok": None}
    p = c.get("pointer") or {}

    # 1. the script must exist
    script = p.get("script")
    if script:
        if not os.path.exists(os.path.join(ROOT, script)):
            r["problems"].append(f"script missing: {script}")

    # 2. the output file must exist
    f = p.get("file")
    if not f:
        r["problems"].append("NO OUTPUT FILE -- this number has no committed source")
        r["value_ok"] = False
        return r
    fp = os.path.join(ROOT, f)
    if not os.path.exists(fp):
        r["problems"].append(f"output file missing: {f}")
        r["value_ok"] = False
        return r
    blob = json.load(open(fp))

    # 3. the config recorded IN the file must match what the claim asserts
    for k, want in (p.get("config") or {}).items():
        ok, got = dig(blob, k)
        if not ok:
            r["problems"].append(f"config key absent from result file: {k}")
        elif got != want:
            r["problems"].append(f"CONFIG MISMATCH {k}: claim says {want!r}, file says {got!r}")

    # 4. the value itself
    if "derive" in c:
        got = DERIVERS[c["derive"]["fn"]](blob, c["derive"].get("args", {}))
    elif "json_path" in p:
        ok, got = dig(blob, p["json_path"])
        if not ok:
            r["problems"].append(f"json_path does not resolve: {p['json_path']}")
            r["value_ok"] = False
            return r
    else:
        got = None
    r["value"] = got
    if "expected" in c and got is not None:
        exp, tol = c["expected"], c.get("tol", 0.005)
        if isinstance(exp, list):
            ok = isinstance(got, list) and len(got) == len(exp) and \
                 all(abs(a - b) <= tol for a, b in zip(got, exp))
        elif isinstance(exp, (int, float)) and isinstance(got, (int, float)):
            ok = abs(got - exp) <= tol
        else:
            ok = got == exp
        r["value_ok"] = bool(ok)
        if not ok:
            r["problems"].append(f"VALUE MISMATCH: claim says {exp!r}, file gives {got!r}")
    else:
        r["value_ok"] = True

    # 5. control-dependent claims: the named lines must still implement the control
    cl = p.get("control_lines")
    if cl:
        m = re.match(r"^(.*?):(\d+)-(\d+)$", cl)
        if not m:
            r["problems"].append(f"malformed control_lines: {cl}")
        else:
            path, lo, hi = m.group(1), int(m.group(2)), int(m.group(3))
            cp = os.path.join(ROOT, path)
            if not os.path.exists(cp):
                r["problems"].append(f"control file missing: {path}")
            else:
                lines = open(cp).read().splitlines()
                if hi > len(lines):
                    r["problems"].append(f"control_lines past EOF: {cl} ({len(lines)} lines)")
                else:
                    seg = "\n".join(lines[lo - 1:hi])
                    for tok in c.get("control_expect", []):
                        if tok not in seg:
                            r["problems"].append(
                                f"DRIFTED: {cl} no longer contains {tok!r} -- the code moved "
                                f"under the claim")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", help="filter by claim_status")
    ap.add_argument("--category", help="filter by provenance category a/b/c/d")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    reg = json.load(open(REGISTRY))
    claims = reg["claims"]
    if a.status:
        claims = [c for c in claims if c["claim_status"] == a.status]
    if a.category:
        claims = [c for c in claims if c["category"] == a.category]

    results = [check_claim(c) for c in claims]
    if a.json:
        print(json.dumps(results, indent=2))
    else:
        by_cat = {}
        for c, r in zip(claims, results):
            by_cat.setdefault(c["category"], []).append(r)
        for cat in sorted(by_cat):
            print(f"\n=== category ({cat}) — {CATEGORIES[cat]} ===")
            for r in by_cat[cat]:
                mark = "ok  " if not r["problems"] else "FAIL"
                print(f"  [{mark}] {r['id']:<28} {r['claim_status']:<10} value={r['value']}")
                for p in r["problems"]:
                    print(f"          ! {p}")
        n_fail = sum(1 for r in results if r["problems"])
        n_quot = sum(1 for c, r in zip(claims, results)
                     if c["claim_status"] == "quotable" and r["problems"])
        print(f"\n{len(results)} claims checked · {n_fail} with problems · "
              f"{n_quot} QUOTABLE claims that do not resolve")
        counts = {}
        for c in claims:
            counts[c["category"]] = counts.get(c["category"], 0) + 1
        print("categories: " + ", ".join(f"({k}) {v}" for k, v in sorted(counts.items())))
        if counts.get("c"):
            print("\n*** category (c) is NON-EMPTY. Per §0.2, 'we ran it but lost the file' "
                  "stops being credible anywhere in the corpus until this is resolved. ***")
        return 1 if n_quot else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Pull result files and checkpoints off a marimo/molab sandbox before its lease expires.

RESEARCH_PLAN rev 3 catalogues the same failure repeatedly: "the JSONs died with the
leases", "artifact external to this repo", "Leases have been expiring mid-run". Two of the
numbers in `docs/review/claims.json` are category (b) for exactly this reason -- real runs
whose output was never pulled. The remedy is not care, it is a cron-able command.

Pulls, base64-framed over the marimo scratchpad so no extra service is needed:
  * every *.json in the sandbox out/ directory  (finished results)
  * every *_cells_*.json in work/              (per-cell checkpoints -- a partial sweep is
                                                still interpretable, and §8 requires that
                                                gates land where partial output is usable)
  * response logs and pool jsonl on --all

Usage:
  export MARIMO_TOKEN=...
  python tools/pull_results.py --url https://sb-xxxx.sb.molab.run/ --dest results/rev3
  python tools/pull_results.py --url ... --dest results/rev3 --all --watch 600
"""
import argparse, base64, functools, json, os, subprocess, sys, time
print = functools.partial(print, flush=True)

HERE = os.path.dirname(os.path.abspath(__file__))
EXEC = os.path.join(HERE, "..", ".claude", "skills", "marimo-pair", "scripts",
                    "execute-code.sh")

LIST_SRC = '''
import json, os, glob
pats = {pats}
out = []
for p in pats:
    for f in glob.glob(p):
        try:
            out.append({{"path": f, "size": os.path.getsize(f),
                         "mtime": os.path.getmtime(f)}})
        except OSError:
            pass
print("@@LIST@@" + json.dumps(out))
'''

# Files are pulled in CHUNKS. A single scratchpad print of a ~1 MB base64 line comes back
# truncated (the stream splits long lines across chunks), which is exactly how the first
# watcher of 2026-09-04 died mid-run on a 700 KB checkpoint. Each chunk is its own marker
# line and carries its offset so a short read is detected rather than silently accepted.
CHUNK = 200_000

GET_SRC = '''
import base64, json, os
p = {path!r}; off = {off}; n = {n}
with open(p, "rb") as f:
    f.seek(off); data = f.read(n)
print("@@FILE@@" + json.dumps({{"path": p, "off": off, "len": len(data),
                                 "size": os.path.getsize(p),
                                 "b64": base64.b64encode(data).decode()}}))
'''


def run(url, code, timeout=300):
    """Run one scratchpad call. The helper is started in its own process group and the whole
    group is killed on timeout: with plain subprocess.run a stalled curl grandchild keeps the
    stdout pipe open after the shell is killed and communicate() blocks forever -- which is
    how both watchers hung silently for two hours on 2026-09-04."""
    import signal
    p = subprocess.Popen(["bash", EXEC, "--url", url, "-"], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         start_new_session=True)
    try:
        out, _ = p.communicate(code, timeout=timeout)
        return out
    except subprocess.TimeoutExpired:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except Exception:
            pass
        try:
            out, _ = p.communicate(timeout=10)
        except Exception:
            out = ""
        print(f"[pull] call timed out after {timeout}s; killed process group", file=sys.stderr,
              flush=True)
        return out or ""


def marker(out, tag):
    for line in out.splitlines():
        i = line.find(tag)
        if i >= 0:
            try:
                return json.loads(line[i + len(tag):])
            except json.JSONDecodeError:
                return None          # truncated line: report as a failed read, do not die
    return None


def fetch_file(url, path, size):
    """Pull one sandbox file in CHUNK-sized pieces; return bytes or None on any short read."""
    buf = bytearray()
    off = 0
    while off < size:
        got = marker(run(url, GET_SRC.format(path=path, off=off, n=CHUNK)), "@@FILE@@")
        if got is None or got.get("off") != off:
            return None
        piece = base64.b64decode(got["b64"])
        if len(piece) != got.get("len") or (len(piece) == 0 and off < size):
            return None
        buf += piece
        off += len(piece)
        if got.get("size", size) != size:      # file grew under us; caller retries next tick
            return None
    return bytes(buf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--dest", default="results/rev3")
    ap.add_argument("--outdir", default="/marimo/out")
    ap.add_argument("--workdir", default="/marimo/work")
    ap.add_argument("--all", action="store_true", help="also pull response logs and pools")
    ap.add_argument("--watch", type=int, default=0, help="repeat every N seconds")
    ap.add_argument("--max-mb", type=float, default=64.0)
    a = ap.parse_args()
    os.makedirs(a.dest, exist_ok=True)

    pats = [f"{a.outdir}/*.json", f"{a.workdir}/*_cells_*.json"]
    if a.all:
        pats += [f"{a.workdir}/*.jsonl", f"{a.workdir}/*.npz"]

    while True:
        listing = marker(run(a.url, LIST_SRC.format(pats=json.dumps(pats))), "@@LIST@@")
        if listing is None:
            print("[pull] could not list the sandbox (expired lease?)", file=sys.stderr)
            if not a.watch:
                return 2
            time.sleep(a.watch); continue
        for rec in sorted(listing, key=lambda r: r["path"]):
            name = os.path.basename(rec["path"])
            local = os.path.join(a.dest, name)
            if rec["size"] > a.max_mb * 2 ** 20:
                print(f"[skip] {name} ({rec['size']/2**20:.1f} MB > --max-mb)")
                continue
            if os.path.exists(local) and os.path.getsize(local) == rec["size"]:
                print(f"[same] {name} ({rec['size']} B)")
                continue
            blob = fetch_file(a.url, rec["path"], rec["size"])
            if blob is None:
                print(f"[FAIL] {name} (short/truncated read; will retry)", file=sys.stderr)
                continue
            tmp = local + ".part"
            with open(tmp, "wb") as f:
                f.write(blob)
            os.replace(tmp, local)
            note = ""
            if name.endswith(".json"):
                try:
                    p = json.load(open(local)).get("_provenance")
                    if p:
                        note = (f"  [prov rev={p.get('model_revision','')[:8]} "
                                f"core={p.get('acl_core_sha','')} "
                                f"{p.get('elapsed_s','?')}s]")
                    else:
                        note = "  [no provenance stamp -- checkpoint, not a result]"
                except Exception:
                    note = "  [unparsable json]"
            print(f"[pull] {name} ({len(blob)} B){note}")
        if not a.watch:
            return 0
        time.sleep(a.watch)


if __name__ == "__main__":
    sys.exit(main())

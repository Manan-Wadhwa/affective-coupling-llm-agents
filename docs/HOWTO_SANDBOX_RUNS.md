# How to run a rev-3 experiment on a molab sandbox (operations, 2026-09-05)

What this cycle taught, in the order you need it.

1. **Connect and inspect.** `bash .claude/skills/marimo-pair/scripts/execute-code.sh --url <URL> --token <TOKEN> -c "..."` runs Python in the notebook scratchpad. Check `torch`, `transformers`, GPU, `/marimo` contents first. A renewed lease is a fresh image: only `/marimo/acl` (code) came back; `/marimo/work`, `/marimo/out` and the HF cache did not.
2. **Write everything under `/marimo/results`** (`--outdir /marimo/results --workdir /marimo/results`); stage pools there too. For already-running jobs, `mirror.sh` copies `out/` and `work/` JSON and logs there every two minutes.
3. **Stage files with `tools/stage_box.sh <URL> <TOKEN> local:remote ...`** — chunked base64 (≤ 90 KB per call), per-chunk retries, sha-verified, idempotent. Stage the pre-registration next to the driver (`/marimo/acl/docs/planning/`), and make the driver resolve it from its own directory (B1c's relative lookup missed it).
4. **Install `accelerate` with `uv pip install --no-deps accelerate==1.14.0`.** marimo's package manager pulled a second torch and shadowed the pinned one.
5. **Pool caches are keyed by tag and by (k, seed).** `a2_regen_feats.py --tag X` looks for `pool_X.jsonl`; the 8B pool was built with k = 220, the 27B with 160. Copy the pool under the new tag when regenerating a subset of layers.
6. **Launch with a runner script** via `subprocess.Popen([...], start_new_session=True)`; never quote-less parentheses in a `log` line (a bash syntax error killed one launch silently). Verify with `bash -n` and `ps`.
7. **Prefetch the model** (`snapshot_download` of the pinned revision into `/marimo/hf`) while you verify the driver; the 27B takes ~2 min, the 8B ~1 min.
8. **Pull every 10–15 minutes** with `tools/pull_results.py --url ... --outdir /marimo/results --workdir /marimo/results --dest results/rev3/inflight_boxN --watch 900` (chunked, process-group kill on timeout). Watch logs with a monitor script that keys seen-counters by box and file. A suspended laptop freezes both; two run tails were lost that way — use `systemd-inhibit` or pull from a machine that stays up.
9. **Before a GPU run:** pre-registration committed; driver committed; `--selftest` locally and on the box; an independent verifier with a real forward-pass smoke test; then launch. Record the launch commit and the prereg sha in the night log.
10. **After a run:** pull, promote to `results/rev3/`, register with exact-precision pointer values (the checker compares floats exactly; round only in the text), write the report, launch a blind critic, fold the critique in with a reconciliation, commit. The checker: `.venv/bin/python tools/check_provenance.py`.
11. **Speed reference (one RTX PRO 6000, 20 CPUs):** 27B ablation arm-row ≈ 19–21 s; 8B follow-up sweep (3 layers) 10 min; 27B layer sweep at n ≤ 2000 ≈ 50 min; margin diagnostic ten seeds ≈ 18 min per layer.

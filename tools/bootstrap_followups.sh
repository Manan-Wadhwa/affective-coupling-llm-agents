#!/bin/bash
# Bring a BLANK molab sandbox to "both rev-3 follow-ups running" in one command.
#   bash tools/bootstrap_followups.sh <URL> <TOKEN>
# Steps: stage code + both pools (chunked base64, sha-verified) -> accelerate --no-deps ->
# regenerate A2 features from the saved pool (GPU, ~10 min) -> launch runner:
#   GPU  b1_followup.py  (B1's own probe pool -> identical directions to the completed run)
#   CPU  a2_followup.py  (14 BLAS threads, leaves cores for the GPU driver)
# Everything it uploads is local to this repo; nothing is generated twice.
set -euo pipefail
URL="$1"; TOK="$2"
cd "$(dirname "$0")/.."
EX=.claude/skills/marimo-pair/scripts/execute-code.sh
TMP=$(mktemp -d)
run() { bash "$EX" --url "$URL" --token "$TOK" "$@" 2>&1 | grep -v "^Warning: connecting"; }
upload() {  # upload <local> <remote>  (idempotent; chunked into per-part remote files with retries; sha-verified)
  local src="$1" dst="$2" sha n i=0 have got tries
  sha=$(sha256sum "$src" | cut -c1-16)
  have=$(run -c "import hashlib,os; print('SHA ' + (hashlib.sha256(open('$dst','rb').read()).hexdigest()[:16] if os.path.exists('$dst') else 'none'))" | grep -o 'SHA [0-9a-z]*' | cut -d' ' -f2)
  if [ "$have" = "$sha" ]; then echo "  already present $dst (sha $sha)"; return 0; fi
  base64 -w0 "$src" > "$TMP/b64"; split -b 90000 -d -a 3 "$TMP/b64" "$TMP/part."
  run -c "import os,glob; os.makedirs(os.path.dirname('$dst'), exist_ok=True); [os.remove(f) for f in glob.glob('$dst.part.*')]; print('reset')" >/dev/null
  for f in "$TMP"/part.*; do i=$((i+1)); tries=0
    printf 'open("%s.part.%03d","w").write("%s")\nprint("CHUNK_OK %d")\n' "$dst" "$i" "$(cat "$f")" "$i" > "$TMP/up.py"
    until run "$TMP/up.py" | grep -q "CHUNK_OK $i"; do tries=$((tries+1)); [ $tries -ge 4 ] && { echo "upload of $src failed at chunk $i after $tries tries"; exit 1; }; sleep 3; done
  done
  rm -f "$TMP"/part.* "$TMP/b64"
  got=$(run -c "import base64,hashlib,os,glob; parts=sorted(glob.glob('$dst.part.*')); raw=base64.b64decode(''.join(open(f).read() for f in parts)); open('$dst','wb').write(raw); [os.remove(f) for f in parts]; print('SHA ' + hashlib.sha256(raw).hexdigest()[:16] + ' PARTS ' + str(len(parts)))" | grep -o 'SHA [0-9a-f]*' | cut -d' ' -f2)
  [ "$got" = "$sha" ] || { echo "sha mismatch for $dst: $got vs $sha"; exit 1; }
  echo "  uploaded $dst ($i chunks, sha $sha)"
}
echo "== staging code =="
for f in src/lib/acl_core.py src/rev3/b1_followup.py src/rev3/a2_followup.py src/rev3/a2_regen_feats.py src/rev3/b1_analyze.py src/rev3/a2_analyze.py; do upload "$f" "/marimo/acl/$(basename "$f")"; done
echo "== staging pools =="
upload results/rev3/inflight_box1/probe_qwen36-27b.jsonl /marimo/work/probe_qwen36-27b.jsonl
upload results/rev3/inflight_box2/pool_qwen36-27b.jsonl  /marimo/work/pool_qwen36-27b.jsonl
echo "== accelerate (no deps, keeps the pinned torch) =="
run -c "import subprocess,sys; print(subprocess.run(['uv','pip','install','--no-deps','-p',sys.executable,'accelerate==1.14.0'],capture_output=True,text=True).stdout[-120:]); import torch,transformers,accelerate; print('stack', torch.__version__, transformers.__version__, accelerate.__version__, 'cuda', torch.cuda.is_available())" | tail -2
echo "== runner =="
run - <<'PY'
import os, subprocess
os.makedirs("/marimo/work", exist_ok=True); os.makedirs("/marimo/out", exist_ok=True)
open("/marimo/acl/runner_followups.sh","w").write("""#!/bin/bash
cd /marimo/acl
export HF_HOME=/marimo/hf PYTHONUNBUFFERED=1
PY=/tmp/uv-venv/bin/python
echo "[runner] followups start $(date -u +%FT%TZ)" >> /marimo/work/runner.log
$PY a2_regen_feats.py --model Qwen/Qwen3.6-27B --tag qwen36-27b --k 160 --layers 16,43,54 > /marimo/work/a2_regen.log 2>&1
echo "[runner] a2_regen exit $? $(date -u +%FT%TZ)" >> /marimo/work/runner.log
if grep -q REGEN_DONE /marimo/work/a2_regen.log; then
  ( OMP_NUM_THREADS=14 OPENBLAS_NUM_THREADS=14 MKL_NUM_THREADS=14 $PY a2_followup.py --feats /marimo/work/feats_qwen36-27b.npz --tag qwen36-27b --layers 16,43,54 > /marimo/work/a2_followup_qwen36-27b.log 2>&1; echo "[runner] a2_followup exit $? $(date -u +%FT%TZ)" >> /marimo/work/runner.log ) &
fi
$PY b1_followup.py --model Qwen/Qwen3.6-27B --tag qwen36-27b --reps 3 > /marimo/work/b1_followup_qwen36-27b.log 2>&1
echo "[runner] b1_followup exit $? $(date -u +%FT%TZ)" >> /marimo/work/runner.log
wait
echo "[runner] followups end $(date -u +%FT%TZ)" >> /marimo/work/runner.log
""")
p = subprocess.Popen(["bash","/marimo/acl/runner_followups.sh"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True, cwd="/marimo/acl")
print("launched runner pid", p.pid)
PY
echo "== done; watch with: MARIMO_TOKEN=<tok> .venv/bin/python tools/pull_results.py --url $URL --dest results/rev3/inflight_box3 --watch 600 =="

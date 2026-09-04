#!/bin/bash
# Upload files to a molab sandbox, chunked + sha-verified + idempotent.
#   bash tools/stage_box.sh <URL> <TOKEN> local1:remote1 [local2:remote2 ...]
set -euo pipefail
URL="$1"; TOK="$2"; shift 2
cd "$(dirname "$0")/.."
EX=.claude/skills/marimo-pair/scripts/execute-code.sh
TMP=$(mktemp -d)
run() { bash "$EX" --url "$URL" --token "$TOK" "$@" 2>&1 | grep -v "^Warning: connecting"; }
upload() {
  local src="$1" dst="$2" sha i=0 have got tries
  sha=$(sha256sum "$src" | cut -c1-16)
  have=$(run -c "import hashlib,os; print('SHA ' + (hashlib.sha256(open('$dst','rb').read()).hexdigest()[:16] if os.path.exists('$dst') else 'none'))" | grep -o 'SHA [0-9a-z]*' | cut -d' ' -f2)
  if [ "$have" = "$sha" ]; then echo "  already present $dst ($sha)"; return 0; fi
  base64 -w0 "$src" > "$TMP/b64"; split -b 90000 -d -a 3 "$TMP/b64" "$TMP/part."
  run -c "import os,glob; os.makedirs(os.path.dirname('$dst'), exist_ok=True); [os.remove(f) for f in glob.glob('$dst.part.*')]; print('reset')" >/dev/null
  for f in "$TMP"/part.*; do i=$((i+1)); tries=0
    printf 'open("%s.part.%03d","w").write("%s")\nprint("CHUNK_OK %d")\n' "$dst" "$i" "$(cat "$f")" "$i" > "$TMP/up.py"
    until run "$TMP/up.py" | grep -q "CHUNK_OK $i"; do tries=$((tries+1)); [ $tries -ge 4 ] && { echo "upload of $src failed at chunk $i"; exit 1; }; sleep 3; done
  done
  rm -f "$TMP"/part.* "$TMP/b64"
  got=$(run -c "import base64,hashlib,os,glob; parts=sorted(glob.glob('$dst.part.*')); raw=base64.b64decode(''.join(open(f).read() for f in parts)); open('$dst','wb').write(raw); [os.remove(f) for f in parts]; print('SHA ' + hashlib.sha256(raw).hexdigest()[:16])" | grep -o 'SHA [0-9a-f]*' | cut -d' ' -f2)
  [ "$got" = "$sha" ] || { echo "sha mismatch for $dst: $got vs $sha"; exit 1; }
  echo "  uploaded $dst ($i chunks, $sha)"
}
for spec in "$@"; do upload "${spec%%:*}" "${spec#*:}"; done

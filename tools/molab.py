#!/usr/bin/env python3
"""Drive a molab / marimo sandbox over its HTTP + WebSocket API.

Everything is derived at runtime from --url and --token, so a new sandbox needs
nothing but those two values.

The handshake needs all four of these; missing any one fails in a way that looks
like something else:
  1. GET / with ?access_token=...  -> the server sets a SIGNED cookie
     `session_<port>=...`. The raw access token alone returns 403 on any POST.
  2. The served page embeds  <marimo-server-token data-token="...">  which must be
     sent as the `Marimo-Server-Token` header on every POST (skew protection).
  3. Open wss://HOST/ws?session_id=<uuid> to register a kernel session; that id
     then goes in the `Marimo-Session-Id` header.
  4. A browser User-Agent. molab's edge returns 403 to urllib's default UA where
     curl succeeds.
POST bodies are camelCase: {"cellIds": [...], "codes": [...]}.

A dead sandbox answers 410 on every path, including unauthenticated /health —
that is how to tell an expired lease from a busy kernel.

Note: if a notebook already has a client attached, marimo may attach you as a
read-only kiosk consumer (`consumer_capabilities: {edit: false}`) and /api/kernel/run
returns "This connection is read-only for this action." The marimo-pair skill's
execute-code.sh uses the session scratchpad instead, which is not subject to that.

Usage:
  molab.py --url URL --token TOK probe
  molab.py --url URL --token TOK exec [--cell ID] [--wait S] [--src FILE]
  molab.py --url URL --token TOK write --src local.py --dest /marimo/x.py
"""
import argparse, asyncio, http.cookiejar, json, re, sys, urllib.parse, urllib.request, uuid

try:
    import websockets
except ImportError:
    sys.exit("need `websockets`  (uv pip install websockets)")

UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/140.0.0.0 Safari/537.36")
BROWSER = {"User-Agent": UA, "Accept": "text/html,application/xhtml+xml,*/*"}


class Sandbox:
    def __init__(self, url, token):
        self.base = url.rstrip("/")
        self.host = urllib.parse.urlparse(self.base).netloc
        self.token = token
        self.cookie = self.server_token = None
        self.session_id = str(uuid.uuid4())

    def handshake(self):
        jar = http.cookiejar.CookieJar()
        op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
        req = urllib.request.Request(f"{self.base}/?access_token={self.token}", headers=BROWSER)
        with op.open(req, timeout=30) as r:
            html = r.read().decode("utf-8", "replace")
        parts = [f"{c.name}={c.value}" for c in jar]
        if not parts:
            raise RuntimeError("no cookie set — wrong token?")
        self.cookie = "; ".join(parts)
        m = (re.search(r'<marimo-server-token[^>]*data-token="([^"]+)"', html)
             or re.search(r'"serverToken"\s*:\s*"([^"]+)"', html))
        if not m:
            raise RuntimeError("no marimo-server-token in the served page")
        self.server_token = m.group(1)
        return self

    def status(self):
        req = urllib.request.Request(f"{self.base}/api/status?access_token={self.token}",
                                     headers=BROWSER)
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)

    def post(self, path, payload):
        req = urllib.request.Request(
            f"{self.base}{path}?access_token={self.token}", method="POST",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "Cookie": self.cookie,
                     "Marimo-Server-Token": self.server_token,
                     "Marimo-Session-Id": self.session_id, **BROWSER})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.status, r.read()[:400].decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            return e.code, e.read()[:400].decode("utf-8", "replace")

    async def _connect(self):
        return await websockets.connect(
            f"wss://{self.host}/ws?session_id={self.session_id}&access_token={self.token}",
            additional_headers={"Cookie": self.cookie, "User-Agent": UA}, max_size=None)

    async def probe(self):
        async with await self._connect() as ws:
            info = {"session_id": self.session_id}
            for _ in range(6):
                try:
                    m = json.loads(await asyncio.wait_for(ws.recv(), timeout=15))
                except asyncio.TimeoutError:
                    break
                if m.get("op") == "kernel-ready":
                    d = m.get("data", {})
                    info.update(cell_ids=d.get("cell_ids"), names=d.get("names"),
                                kiosk=d.get("kiosk"),
                                consumer_capabilities=d.get("consumer_capabilities"))
                    break
            return info

    async def exec(self, code, cell, wait):
        async with await self._connect() as ws:
            done = asyncio.Event()

            async def pump():
                while True:
                    m = json.loads(await ws.recv())
                    op, d = m.get("op"), (m.get("data") or {})
                    if op == "cell-op":
                        for c in (d.get("console") or []):
                            print(c.get("data", "") if isinstance(c, dict) else c,
                                  end="", flush=True)
                        o = d.get("output")
                        if o and o.get("data") is not None and str(o["data"]).strip():
                            print("\n[out] " + str(o["data"])[:6000], flush=True)
                        if d.get("status") == "idle":
                            done.set()
                    elif op in ("kernel-error", "interrupted"):
                        print(f"\n[{op}] " + json.dumps(d)[:2000], flush=True)
                        done.set()

            t = asyncio.create_task(pump())
            await asyncio.sleep(2.0)
            print("[run]", self.post("/api/kernel/run", {"cellIds": [cell], "codes": [code]}),
                  flush=True)
            try:
                await asyncio.wait_for(done.wait(), timeout=wait)
            except asyncio.TimeoutError:
                print(f"\n[still running after {wait:.0f}s]", flush=True)
            t.cancel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--token", required=True)
    ap.add_argument("action", choices=["probe", "exec", "write"])
    ap.add_argument("--cell", default="Hbol")
    ap.add_argument("--wait", type=float, default=300.0)
    ap.add_argument("--src")
    ap.add_argument("--dest", default="/marimo/notebook.py")
    a = ap.parse_args()

    sb = Sandbox(a.url, a.token).handshake()
    st = sb.status()
    print(f"[ok] marimo {st.get('version')} py {st.get('python_version')} "
          f"mode={st.get('mode')} files={st.get('filenames')} sessions={st.get('sessions')}")
    print(f"[ok] cookie + server token acquired; session {sb.session_id}")

    if a.action == "probe":
        print(json.dumps(asyncio.run(sb.probe()), indent=2))
    elif a.action == "exec":
        asyncio.run(sb.exec(open(a.src).read() if a.src else sys.stdin.read(), a.cell, a.wait))
    elif a.action == "write":
        body = open(a.src).read()
        code = (f"import pathlib\n"
                f"p = pathlib.Path({a.dest!r}); p.parent.mkdir(parents=True, exist_ok=True)\n"
                f"p.write_text({body!r})\n"
                f"print('wrote', p, p.stat().st_size, 'bytes')\n")
        asyncio.run(sb.exec(code, a.cell, a.wait))


if __name__ == "__main__":
    main()

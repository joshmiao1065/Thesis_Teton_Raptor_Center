"""Local web server for the annotation tool: the app, audio clips cut on demand, labels saved to disk.

    python -m annotation.server --dir <annotation dir> [--port 8765]

The directory holds ``manifest.json`` (written by ``annotation.build``) and receives ``labels.jsonl``
(append only, latest line per item wins) and a random access key in ``.key``. Open the printed URL,
for example through the JupyterHub proxy: ``<hub>/user/<name>/proxy/<port>/?k=<key>``. Clients only get
what an annotator may see; model scores and strata stay in ``manifest.json`` for the analysis.
"""
import argparse
import io
import json
import secrets
import threading
import time
from functools import lru_cache
from http import cookies
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import soundfile as sf

from annotation.sampling import audio_root

APP = Path(__file__).parent / "app" / "index.html"
PUBLIC = ("id", "kind", "clip", "focal", "truth", "title", "species", "tier")
LOCK = threading.Lock()


class State:
    def __init__(self, folder):
        self.dir = Path(folder)
        m = json.loads((self.dir / "manifest.json").read_text())
        self.items = {it["id"]: it for grp in ("queue", "reference", "practice") for it in m[grp]}
        self.groups = {g: [self.public(it) for it in m[g]] for g in ("queue", "reference", "practice")}
        key = self.dir / ".key"
        if not key.exists():
            key.write_text(secrets.token_urlsafe(12))
            key.chmod(0o600)
        self.key = key.read_text().strip()
        self.labels_path = self.dir / "labels.jsonl"

    @staticmethod
    def public(it):
        return {k: it[k] for k in PUBLIC if k in it}

    def wav(self, item_id):
        return cut_clip(self.dir, self.items[item_id])

    def read_labels(self, labeler=None):
        out = {}
        if self.labels_path.exists():
            for line in self.labels_path.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    if labeler is None or r.get("labeler") == labeler:
                        out[r["id"]] = r
        return out

    def add_label(self, rec):
        rec = dict(rec, ts=time.time())
        with LOCK, open(self.labels_path, "a") as f:
            f.write(json.dumps(rec) + "\n")


def cut_clip(folder, it):
    """16-bit mono wav of the item's clip, at the recording's own sample rate (first channel)."""
    if "wav" in it:
        return (Path(folder) / it["wav"]).read_bytes()
    path = audio_root(it["site"]) / it["rel"]
    return _cut(str(path), float(it["clip"][0]), float(it["clip"][1]))


@lru_cache(maxsize=256)
def _cut(path, start, dur):
    sr = sf.info(path).samplerate
    y, _ = sf.read(path, start=int(start * sr), frames=int(dur * sr), dtype="int16", always_2d=True)
    buf = io.BytesIO()
    sf.write(buf, y[:, 0], sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def make_handler(state):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json", headers=()):
            body = body if isinstance(body, bytes) else body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in headers:
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _auth(self):
            q = parse_qs(urlparse(self.path).query)
            jar = cookies.SimpleCookie(self.headers.get("Cookie", ""))
            given = (q.get("k") or [None])[0] or (jar["annkey"].value if "annkey" in jar else None)
            return given is not None and secrets.compare_digest(given, state.key), q

        def do_GET(self):
            ok, q = self._auth()
            if not ok:
                return self._send(403, "missing or wrong key: open the URL printed by the server", "text/plain")
            path = urlparse(self.path).path.rstrip("/")
            path = path[path.index("/api"):] if "/api/" in path else path
            if path in ("", "/index.html") or not path.startswith("/api"):
                cookie = f"annkey={state.key}; Path=/; HttpOnly; SameSite=Lax"
                return self._send(200, APP.read_bytes(), "text/html; charset=utf-8", [("Set-Cookie", cookie)])
            if path == "/api/manifest":
                return self._send(200, json.dumps(state.groups))
            if path == "/api/labels":
                return self._send(200, json.dumps(state.read_labels((q.get("labeler") or [None])[0])))
            if path.startswith("/api/clip/"):
                item_id = path.rsplit("/", 1)[1].removesuffix(".wav")
                if item_id not in state.items:
                    return self._send(404, "unknown item", "text/plain")
                return self._send(200, state.wav(item_id), "audio/wav")
            return self._send(404, "not found", "text/plain")

        def do_POST(self):
            ok, _ = self._auth()
            if not ok:
                return self._send(403, "forbidden", "text/plain")
            n = int(self.headers.get("Content-Length", 0))
            rec = json.loads(self.rfile.read(n) or b"{}")
            if rec.get("id") not in state.items:
                return self._send(400, json.dumps({"error": "unknown id"}))
            state.add_label(rec)
            return self._send(200, json.dumps({"ok": True}))

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    st = State(a.dir)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(st))
    print(f"serving {len(st.groups['queue'])} queue items on port {a.port}; key file: {st.dir / '.key'}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()

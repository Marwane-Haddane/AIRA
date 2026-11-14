#!/usr/bin/env python3
"""Local AI line-art service for the RoboGenius Dobot portrait studio.

Serves the static app and exposes:

    GET  /api/health
    POST /api/lineart?size=768&threshold=150&max_strokes=320   (body = image)

The heavy lifting lives in lineart_engine.py.  Everything runs on this PC;
no Hugging Face token or cloud API is used at runtime.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import traceback
import urllib.parse
import urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
MODEL_PATH = MODELS / "informative-drawings.onnx"
FACE_MODEL_PATH = MODELS / "face_detection_yunet_2023mar.onnx"
APP_FILE = "robogenius_dobot_magician_3d_digital_twin_studio.html"

MODEL_REPO = "rocca/informative-drawings-line-art-onnx"
MODEL_URL = f"https://huggingface.co/{MODEL_REPO}/resolve/main/informative-drawings.onnx"
FACE_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)

_engine = None
_engine_lock = threading.Lock()


# --------------------------------------------------------------------------- #
# Model bootstrapping
# --------------------------------------------------------------------------- #
def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {dest.name} ...")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as response:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        with open(tmp, "wb") as out:
            while True:
                chunk = response.read(1024 * 256)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                if total:
                    pct = done * 100 // total
                    print(f"\r  {pct:3d}%  {done / 1e6:6.1f} / {total / 1e6:.1f} MB", end="")
    if total:
        print()
    tmp.replace(dest)
    print(f"Saved {dest}")


def ensure_models() -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    if not MODEL_PATH.exists():
        download(MODEL_URL, MODEL_PATH)
    if not FACE_MODEL_PATH.exists():
        try:
            download(FACE_URL, FACE_MODEL_PATH)
        except Exception as exc:
            print(f"Warning: face detector download failed ({exc}); using geometric crop.")


def get_engine():
    """Create the engine once (rembg session + ONNX session)."""
    global _engine
    with _engine_lock:
        if _engine is not None:
            return _engine

        from lineart_engine import LineArtEngine

        bg_session = None
        try:
            from rembg import new_session

            print("Loading background-removal model (u2net_human_seg) ...")
            bg_session = new_session("u2net_human_seg")
        except Exception as exc:
            print(f"Warning: rembg unavailable ({exc}); using the raw photo.")

        print("Loading Informative Drawings ONNX ...")
        engine = LineArtEngine(
            MODEL_PATH,
            bg_session=bg_session,
            face_model_path=FACE_MODEL_PATH if FACE_MODEL_PATH.exists() else None,
        )
        engine.session  # force model load so the first request is fast
        _engine = engine
        print("AI engine ready.")
        return _engine


# --------------------------------------------------------------------------- #
# Request helpers
# --------------------------------------------------------------------------- #
def _num(query: dict, key: str, default, cast, lo=None, hi=None):
    try:
        value = cast(query.get(key, [default])[0])
    except (TypeError, ValueError):
        value = default
    if lo is not None:
        value = max(lo, value)
    if hi is not None:
        value = min(hi, value)
    return value


class Handler(SimpleHTTPRequestHandler):
    server_version = "RoboLineArt/1.0"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    # -- shared response writers ------------------------------------------- #
    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, message: str) -> None:
        self._json(status, {"ok": False, "error": message})

    def log_message(self, fmt, *args):
        # Keep API request noise low; errors are printed explicitly.
        message = fmt % args
        if "/api/" in message:
            sys.stderr.write("[api] " + message + "\n")

    # -- routing ----------------------------------------------------------- #
    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/health":
            self._json(
                200,
                {
                    "ok": True,
                    "service": "robogenius-lineart",
                    "engine": MODEL_REPO,
                    "model_url": MODEL_URL,
                    "model_ready": MODEL_PATH.exists(),
                    "face_model_ready": FACE_MODEL_PATH.exists(),
                    "bg_model": "u2net_human_seg",
                    "device": "CPU",
                    "app": APP_FILE,
                },
            )
            return
        if parsed.path in ("/", "/index.html"):
            self.send_response(302)
            self.send_header("Location", "/" + urllib.parse.quote(APP_FILE))
            self.end_headers()
            return
        if parsed.path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return
        super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/api/lineart":
            self._error(404, "Unknown endpoint. Use POST /api/lineart")
            return

        try:
            length = int(self.headers.get("Content-Length") or "0")
            if length <= 0:
                self._error(400, "Empty request body; send a JPEG or PNG image.")
                return
            if length > 30 * 1024 * 1024:
                self._error(413, "Image is larger than 30 MB.")
                return
            image_bytes = self.rfile.read(length)

            query = urllib.parse.parse_qs(parsed.query)
            from lineart_engine import StrokeParams

            params = StrokeParams(
                size=_num(query, "size", 768, int, 512, 1024),
                threshold=_num(query, "threshold", 150, int, 100, 230),
                texture_threshold=_num(query, "texture_threshold", 112, int, 80, 200),
                head_threshold=_num(query, "head_threshold", 210, int, 150, 235),
                line_gamma=_num(query, "line_gamma", 1.5, float, 0.8, 3.0),
                min_length=_num(query, "min_length", 18.0, float, 5.0, 80.0),
                head_min_length=_num(query, "head_min_length", 4.0, float, 2.0, 30.0),
                max_strokes=_num(query, "max_strokes", 320, int, 40, 900),
                head_max_fraction=_num(query, "head_max_fraction", 0.7, float, 0.1, 1.0),
                rdp_eps=_num(query, "rdp_eps", 1.0, float, 0.3, 3.0),
                smoothing=_num(query, "smoothing", 2, int, 0, 3),
                denoise=_num(query, "denoise", 1, int, 0, 2),
            )

            engine = get_engine()
            result = engine.generate(image_bytes, params)
            self._json(200, result)
        except Exception as exc:
            traceback.print_exc()
            self._error(500, f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description="RoboGenius local AI line-art server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--no-preload", action="store_true",
                        help="Do not load the models until the first request")
    args = parser.parse_args()

    ensure_models()
    if not args.no_preload:
        get_engine()

    try:
        server = ThreadingHTTPServer((args.host, args.port), Handler)
    except OSError as exc:
        print(f"Cannot bind {args.host}:{args.port}: {exc}")
        print("Another server may already be running. Stop it or use --port 8770.")
        return 1

    url = f"http://{args.host}:{args.port}/"
    print()
    print("RoboGenius AI Portrait Studio")
    print(f"  App:    {url}")
    print(f"  Health: {url}api/health")
    print(f"  Model:  https://huggingface.co/{MODEL_REPO}")
    print("  Press Ctrl+C to stop.")
    print()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

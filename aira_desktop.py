"""AIRA desktop app: the studio in its own window (no browser needed).

Starts the local robot/AI service on a free 127.0.0.1 port in a background thread and
opens it in a native window (Microsoft Edge WebView2 on Windows, via pywebview).

Run:  .venv\\Scripts\\python.exe aira_desktop.py
Build: powershell -ExecutionPolicy Bypass -File .\\Build-Desktop.ps1
"""

from __future__ import annotations

import os
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path


def _app_root() -> Path:
    # PyInstaller one-folder build: data files live next to the executable
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def main() -> None:
    root = _app_root()
    os.chdir(root)
    # ctranslate2 (Whisper) and onnxruntime/OpenCV each ship an OpenMP runtime; allow both in one process
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    if getattr(sys, "frozen", False):
        # windowed build has no console: keep errors and hard crashes in aira.log next to AIRA.exe
        import faulthandler
        log = open(root / "aira.log", "a", buffering=1, encoding="utf-8")
        sys.stdout = sys.stderr = log
        faulthandler.enable(log)
    os.environ.setdefault("AIRA_MODELS", str(root / "models"))
    sys.path.insert(0, str(root))

    import webview  # pywebview
    import dobot_service

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), dobot_service.RoboGeniusHandler)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, name="aira-service", daemon=True).start()

    window = webview.create_window(
        "AIRA · Dobot Drawing Studio",
        f"http://127.0.0.1:{port}/",
        width=1440, height=900, min_size=(1024, 680), background_color="#090d15",
    )

    def _shutdown() -> None:
        try:
            dobot_service.LINK.stop_job()
            dobot_service.LINK.disconnect()
        except Exception:
            pass
        httpd.shutdown()

    window.events.closed += _shutdown
    icon = root / "docs" / "aira.ico"
    webview.start(private_mode=False, storage_path=str(root / ".webview"),
                  icon=str(icon) if icon.is_file() else None)


if __name__ == "__main__":
    main()

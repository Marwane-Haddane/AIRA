"""Run archive: every AI Coloring / Clean Line run keeps its input, outputs and timing.

Layout: archive/<id>/{input.jpg, sketch.png, page.png (AI only), drawing.svg, meta.json}
The folder is git-ignored; it lives next to the studio so it survives restarts.
"""

from __future__ import annotations

import io
import json
import re
import shutil
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
ARCHIVE = ROOT / "archive"
FILES = {"input.jpg": "image/jpeg", "sketch.png": "image/png", "page.png": "image/png",
         "drawing.svg": "image/svg+xml", "meta.json": "application/json"}
_ID = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{6}$")


def _save_png(arr: np.ndarray, path: Path) -> None:
    Image.fromarray(arr).save(path, format="PNG", optimize=True)


def save(source: Optional[bytes], line_image: np.ndarray, page: Optional[np.ndarray], raw_strokes, size: int,
         info: Dict[str, Any], seconds: float) -> Optional[str]:
    """Store one run; never raises (the archive must not break drawing)."""
    try:
        from .vectorize import strokes_to_svg

        now = datetime.now()
        run_id = f"{now:%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
        folder = ARCHIVE / run_id
        folder.mkdir(parents=True, exist_ok=True)
        if source:
            img = Image.open(io.BytesIO(source)).convert("RGB")
            img.thumbnail((1024, 1024))
            img.save(folder / "input.jpg", format="JPEG", quality=88)
        _save_png(line_image, folder / "sketch.png")
        if page is not None:
            _save_png(page, folder / "page.png")
        (folder / "drawing.svg").write_text(strokes_to_svg([s for s in raw_strokes if len(s) >= 2], size, size, 2.0, 130.0),
                                            encoding="utf-8")
        meta = {
            "id": run_id,
            "created": now.isoformat(timespec="seconds"),
            "engine": info.get("engine", "clean"),
            "subject": info.get("subject"),
            "font": info.get("font_label"),
            "model": info.get("model"),
            "ai_error": info.get("ai_error"),
            "stroke_count": int(info.get("stroke_count") or len(raw_strokes)),
            "seconds": round(float(seconds), 2),
            "files": sorted(p.name for p in folder.iterdir()),
        }
        (folder / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return run_id
    except Exception as exc:  # pragma: no cover - best effort
        print(f"[archive] could not save run: {exc}")
        return None


def list_runs(limit: int = 60) -> List[Dict[str, Any]]:
    if not ARCHIVE.is_dir():
        return []
    out = []
    for folder in sorted((p for p in ARCHIVE.iterdir() if p.is_dir() and _ID.match(p.name)), reverse=True)[:limit]:
        try:
            out.append(json.loads((folder / "meta.json").read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return out


def file_path(run_id: str, name: str) -> Optional[Path]:
    if not _ID.match(run_id) or name not in FILES:
        return None
    path = ARCHIVE / run_id / name
    return path if path.is_file() else None


def delete(run_id: str) -> bool:
    if not _ID.match(run_id):
        raise ValueError("Invalid archive id")
    folder = ARCHIVE / run_id
    if not folder.is_dir():
        return False
    shutil.rmtree(folder)
    return True

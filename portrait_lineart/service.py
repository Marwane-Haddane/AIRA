"""HTTP-facing helpers for the Clean Line portrait engine (used by dobot_service.py)."""

from __future__ import annotations

import base64
import threading
from typing import Any, Dict

_LOCK = threading.Lock()
NORM_HALF = 0.80  # UI paper coordinates are in [-0.82, 0.82]; keep a small margin


def _decode_png_data_url(value: str, field: str) -> bytes:
    if not value:
        return b""
    if not value.startswith("data:image/") or ";base64," not in value:
        raise ValueError(f"Clean Line expects '{field}' as a base64 image data URL")
    raw = base64.b64decode(value.split(",", 1)[1], validate=True)
    if len(raw) > 8_000_000:
        raise ValueError("Portrait image is too large for local processing")
    return raw


def status() -> Dict[str, Any]:
    try:
        from .pipeline import find_model
    except Exception as exc:  # missing numpy/opencv/etc.
        return {"available": False, "error": f"Clean Line dependencies missing: {exc}"}
    parse = find_model("face_parsing_resnet34.onnx") or find_model("face_parsing_resnet18.onnx")
    from .cloud_ai import config, is_configured
    return {
        "ai_coloring": is_configured(),
        "ai_model": config()["model"],
        "available": True,
        "model": "AIRA Clean Line",
        "face_parsing": bool(parse),
        "lineart": bool(find_model("informative-drawings.onnx")),
        "face_detector": bool(find_model("face_detection_yunet_2023mar.onnx")),
    }


def run(body: Dict[str, Any]) -> Dict[str, Any]:
    from .pipeline import PipelineParams, generate

    photo = _decode_png_data_url(str(body.get("photo", "")), "photo")
    cutout = _decode_png_data_url(str(body.get("image", "")), "image")
    source = photo or cutout
    if not source:
        raise ValueError("Clean Line needs a photo")

    def unit(name: str, default: float) -> float:
        try:
            return min(1.0, max(0.0, float(body.get(name, default))))
        except (TypeError, ValueError):
            return default

    engine = str(body.get("engine", "clean")).lower()
    ai_error = None
    if engine == "ai":
        from .cloud_ai import CloudAIError, generate as ai_generate
        try:
            with _LOCK:
                out = ai_generate(source, fallback_cutout=cutout if photo else None)
            secs = out["info"]["timings"]["total"]
            resp = _respond(out["strokes"], out["size"], out["line_image"], out["info"], secs)
            resp["archive_id"] = _archive(source, out["line_image"], out["page"], out["strokes"], out["size"],
                                          out["info"], secs)
            return resp
        except CloudAIError as exc:
            ai_error = str(exc)  # fall back to the local engine below

    params = PipelineParams(
        detail=unit("detail", 0.5),
        hair_strands=unit("hair", 0.5),
        clothing=unit("clothing", 0.35),
    )
    with _LOCK:  # models are shared and onnxruntime sessions are not re-entrant here
        result = generate(source, params, fallback_cutout=cutout if photo else None)

    info = dict(result.info)
    info["engine"] = "clean"
    if ai_error:
        info["ai_error"] = ai_error
    secs = result.timings.get("total", 0.0)
    resp = _respond(result.strokes, result.size, result.line_image, info, secs)
    resp["archive_id"] = _archive(source, result.line_image, None, result.strokes, result.size, info, secs)
    return resp


def _archive(source, line_image, page, strokes, size, info, secs):
    from .archive import save
    return save(source, line_image, page, strokes, size, info, secs)


def _respond(raw_strokes, size: int, line_image, info: Dict[str, Any], seconds: float) -> Dict[str, Any]:
    """Fit the drawing (not the canvas) to the paper square, keeping proportions."""
    import cv2
    import numpy as np
    usable = [s for s in raw_strokes if len(s) >= 2]
    if usable:
        allp = np.vstack(usable)
        lo, hi = allp.min(0), allp.max(0)
        center = (lo + hi) / 2.0
        span = float(max(hi - lo)) or 1.0
    else:
        center, span = np.array([size / 2.0] * 2), float(size)
    k = 2 * NORM_HALF / span
    strokes = [
        [[round(float((x - center[0]) * k), 5), round(float((y - center[1]) * k), 5)] for x, y in s]
        for s in usable
    ]
    ok, png = cv2.imencode(".png", line_image)
    image = "data:image/png;base64," + base64.b64encode(png.tobytes()).decode("ascii") if ok else None
    return {
        "ok": True,
        "image": image,
        "strokes": strokes,
        "info": info,
        "seconds": round(float(seconds), 2),
    }

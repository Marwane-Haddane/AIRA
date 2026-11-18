"""AI Coloring: photo -> coloring-book page with FLUX.2 [klein] on Cloudflare Workers AI.

The free Workers AI allocation (10,000 neurons/day, reset 00:00 UTC) covers roughly
90 drawings a day at 1024x1024. The generated page is traced into plotter strokes
locally by the centerline vectorizer. Any failure (no key, quota used up, offline)
raises CloudAIError so the caller can fall back to the local Clean Line engine.

Configuration (environment or a `.env` file in the project root):
    CLOUDFLARE_ACCOUNT_ID=...
    CLOUDFLARE_API_TOKEN=...          # token with "Workers AI: Read" (+ Edit) permission
    CLOUDFLARE_AI_MODEL=@cf/black-forest-labs/flux-2-klein-4b   # optional
"""

from __future__ import annotations

import base64
import io
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "@cf/black-forest-labs/flux-2-klein-4b"
INPUT_MAX = 496          # Workers AI FLUX.2 inputs must be smaller than 512x512
OUTPUT_SIZE = 1024

PROMPT = (
    "Turn this photo into a clean coloring book page: black and white line art of the same person, "
    "head and shoulders, front view. Smooth uniform black outlines of medium thickness on a pure white "
    "background. Simple clean contours for the face shape, eyebrows, eyes with irises, nose, lips, ears, "
    "neck and clothing collar; hair drawn as flowing strand outlines; clothing seams and logos as simple "
    "outlines. No shading, no hatching, no gray tones, no solid black fills, no color, no background, "
    "no text, no border. Keep the person's likeness, hairstyle and clothing."
)

SCENE_PROMPT = (
    "Turn this photo into a clean coloring book page: black and white line art of exactly the same scene "
    "and composition. Smooth uniform black outlines of medium thickness on a pure white background; "
    "simple clean contours for every object, building, landscape and texture. No people unless they are "
    "in the photo. No shading, no hatching, no gray tones, no solid black fills, no color, no text, no border."
)


class CloudAIError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
def _read_env_file() -> Dict[str, str]:
    out: Dict[str, str] = {}
    path = ROOT / ".env"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return out
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip().strip('"').strip("'")
    return out


def config() -> Dict[str, str]:
    env = _read_env_file()

    def get(name: str, default: str = "") -> str:
        return os.environ.get(name) or env.get(name) or default

    return {
        "account": get("CLOUDFLARE_ACCOUNT_ID"),
        "token": get("CLOUDFLARE_API_TOKEN"),
        "model": get("CLOUDFLARE_AI_MODEL", DEFAULT_MODEL),
    }


def is_configured() -> bool:
    c = config()
    return bool(c["account"] and c["token"])


# --------------------------------------------------------------------------- #
# Input preparation: head-and-shoulders crop on white, like the target pages
# --------------------------------------------------------------------------- #
def prepare_input(image_bytes: bytes, fallback_cutout: Optional[bytes] = None) -> Tuple[np.ndarray, Dict]:
    from .pipeline import MODELS_CACHE, PipelineParams, _largest_subject, frame_portrait, load_rgba

    p = PipelineParams(frame_face_width=0.34, frame_eye_y=0.38)
    rgba = load_rgba(image_bytes)
    rgb, alpha = rgba[:, :, :3].copy(), rgba[:, :, 3].copy()
    has_alpha = alpha.min() < 250
    face = MODELS_CACHE.detect_face(rgb)
    if face is not None and face[2] < 0.07 * max(rgb.shape[:2]):
        face = None  # a tiny face in a landscape: treat the photo as a scene
    if face is None and not has_alpha:
        # not a portrait: keep the whole picture, letterboxed on white
        h, w = rgb.shape[:2]
        S = p.size
        k = S / max(h, w)
        small = cv2.resize(rgb, (max(1, int(w * k)), max(1, int(h * k))), interpolation=cv2.INTER_AREA)
        out = np.full((S, S, 3), 255, np.uint8)
        y0, x0 = (S - small.shape[0]) // 2, (S - small.shape[1]) // 2
        out[y0:y0 + small.shape[0], x0:x0 + small.shape[1]] = small
        return out, {"face_found": False, "matte": "none", "subject": "scene"}
    matte = "alpha" if has_alpha else "none"
    if not has_alpha:
        m = MODELS_CACHE.matte(rgb)
        if m is not None:
            alpha, matte = m, "rembg"
        elif fallback_cutout:
            cut = load_rgba(fallback_cutout)[:, :, 3]
            alpha = cv2.resize(cut, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
            matte = "browser"
        else:
            alpha = np.full(alpha.shape, 255, np.uint8)
    framed, fa, _valid, _M = frame_portrait(rgb, alpha, face, p)
    subject = _largest_subject(fa > 127)
    a = (cv2.GaussianBlur(subject.astype(np.float32), (0, 0), 1.2) * (fa.astype(np.float32) / 255.0))[..., None]
    a = np.clip(a * 1.15, 0, 1)
    white = np.full_like(framed, 255)
    out = (framed.astype(np.float32) * a + white.astype(np.float32) * (1 - a)).astype(np.uint8)
    return out, {"face_found": face is not None, "matte": matte, "subject": "portrait" if face is not None else "object"}


# --------------------------------------------------------------------------- #
# Cloudflare call
# --------------------------------------------------------------------------- #
def _png_bytes(rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _error_text(resp) -> str:
    try:
        data = resp.json()
        errs = data.get("errors") or []
        if errs:
            return "; ".join(str(e.get("message", e)) for e in errs)
        return json.dumps(data)[:300]
    except Exception:
        return (resp.text or "")[:300]


def call_cloudflare(rgb: np.ndarray, prompt: str = PROMPT, seed: Optional[int] = None,
                    timeout: float = 120.0) -> bytes:
    import requests

    c = config()
    if not (c["account"] and c["token"]):
        raise CloudAIError("AI Coloring is not set up: add CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN to .env")
    small = cv2.resize(rgb, (INPUT_MAX, INPUT_MAX), interpolation=cv2.INTER_AREA)
    url = f"https://api.cloudflare.com/client/v4/accounts/{c['account']}/ai/run/{c['model']}"
    data = {"prompt": prompt, "width": str(OUTPUT_SIZE), "height": str(OUTPUT_SIZE)}
    if seed is not None:
        data["seed"] = str(int(seed))
    files = {"input_image_0": ("portrait.png", _png_bytes(small), "image/png")}
    try:
        resp = requests.post(url, headers={"Authorization": f"Bearer {c['token']}"},
                             data=data, files=files, timeout=timeout)
    except requests.RequestException as exc:
        raise CloudAIError(f"Cloudflare is unreachable: {exc}") from exc
    if resp.status_code == 429:
        raise CloudAIError("Cloudflare free daily limit reached (resets 00:00 UTC): " + _error_text(resp))
    if resp.status_code in (401, 403):
        raise CloudAIError("Cloudflare rejected the API token: " + _error_text(resp))
    if resp.status_code != 200:
        raise CloudAIError(f"Cloudflare error {resp.status_code}: " + _error_text(resp))
    ctype = resp.headers.get("content-type", "")
    if ctype.startswith("image/"):
        return resp.content
    try:
        payload = resp.json()
    except ValueError as exc:
        raise CloudAIError("Cloudflare returned an unexpected response") from exc
    result = payload.get("result", payload) if isinstance(payload, dict) else {}
    image = result.get("image") if isinstance(result, dict) else None
    if not image:
        raise CloudAIError("Cloudflare returned no image: " + json.dumps(payload)[:300])
    if image.startswith("data:"):
        image = image.split(",", 1)[1]
    return base64.b64decode(image)


# --------------------------------------------------------------------------- #
# Generated page -> clean ink -> strokes
# --------------------------------------------------------------------------- #
def clean_page(page_rgb: np.ndarray, size: int = 1024) -> np.ndarray:
    """Binary ink mask (bool) from a generated coloring page."""
    gray = cv2.cvtColor(page_rgb, cv2.COLOR_RGB2GRAY)
    if gray.shape != (size, size):
        gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    # flatten soft paper texture / vignetting before thresholding
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    norm = np.clip(gray.astype(np.float32) / np.maximum(bg.astype(np.float32), 1) * 255, 0, 255).astype(np.uint8)
    t, _ = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t = float(np.clip(t, 110, 200))
    ink = norm < t
    # drop a frame / border the model may add
    edge = max(4, size // 128)
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    for i in range(1, n):
        x, y, w, h = st[i, :4]
        # a frame hugs all four sides; a body outline touches at most three
        frame = x <= edge and y <= edge and x + w >= size - edge and y + h >= size - edge
        if frame and st[i, cv2.CC_STAT_AREA] < 0.08 * w * h:
            ink[lab == i] = False
    return ink


def page_to_strokes(page_rgb: np.ndarray, size: int = 1024):
    from .vectorize import VectorizeParams, ink_to_strokes

    ink = clean_page(page_rgb, size)
    params = VectorizeParams(min_area=16, close_px=1, spur_px=8, gap_px=5, smooth_sigma=1.4,
                             rdp_eps=0.5, min_length_px=8)
    return ink_to_strokes(ink, params), ink


def generate(image_bytes: bytes, fallback_cutout: Optional[bytes] = None, seed: Optional[int] = None) -> Dict:
    t0 = time.time()
    framed, info = prepare_input(image_bytes, fallback_cutout)
    t_prep = time.time() - t0
    t1 = time.time()
    raw = call_cloudflare(framed, prompt=PROMPT if info["subject"] == "portrait" else SCENE_PROMPT, seed=seed)
    t_cloud = time.time() - t1
    page = np.array(Image.open(io.BytesIO(raw)).convert("RGB"))
    t2 = time.time()
    strokes, ink = page_to_strokes(page, OUTPUT_SIZE)
    line = np.where(ink, 0, 255).astype(np.uint8)
    info.update({
        "engine": "ai",
        "model": config()["model"],
        "stroke_count": len(strokes),
        "timings": {"prepare": round(t_prep, 2), "cloud": round(t_cloud, 2),
                    "trace": round(time.time() - t2, 2), "total": round(time.time() - t0, 2)},
    })
    return {"strokes": strokes, "size": OUTPUT_SIZE, "line_image": line, "page": page, "info": info}

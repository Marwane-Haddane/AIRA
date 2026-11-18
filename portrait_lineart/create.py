"""Create mode: draw from a description, write text in a plotter font, voice input, settings.

- describe(prompt): FLUX.2 [klein] text-to-image coloring page -> centerline strokes
- write_text(text, font): Hershey single-line fonts (the pen draws each letter once)
- transcribe(audio): local Whisper base (faster-whisper, CPU int8), offline
- settings: read/write the Cloudflare key in the local .env file
"""

from __future__ import annotations

import base64
import io
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"
WHISPER_DIR = Path(os.environ.get("AIRA_WHISPER_MODEL", str(ROOT / "models" / "whisper-base")))

DESCRIBE_TEMPLATE = (
    "A clean coloring book page of {subject}. Black and white line art, smooth uniform black outlines of "
    "medium thickness on a pure white background, the whole subject centered and fully visible, simple clean "
    "contours with a few inner details. No shading, no hatching, no gray tones, no solid black fills, no color, "
    "no text, no border, no background scenery unless asked."
)

# Hershey single-stroke fonts offered in the UI (id -> label)
FONTS = {
    "futural": "Simple",
    "futuram": "Simple bold",
    "scripts": "Script",
    "scriptc": "Script bold",
    "cursive": "Handwriting",
    "rowmans": "Roman",
    "timesr": "Serif",
    "timesi": "Serif italic",
    "timesrb": "Serif bold",
    "gothiceng": "Gothic",
    "gothicger": "Fraktur",
    "greek": "Greek",
    "cyrillic": "Cyrillic",
}


# --------------------------------------------------------------------------- #
# Settings (.env)
# --------------------------------------------------------------------------- #
def _mask(value: str) -> str:
    if not value:
        return ""
    return value[:4] + "…" + value[-4:] if len(value) > 10 else "…"


def settings_status() -> Dict[str, Any]:
    from .cloud_ai import config
    c = config()
    return {
        "cloudflare": {
            "configured": bool(c["account"] and c["token"]),
            "account_hint": _mask(c["account"]),
            "token_hint": _mask(c["token"]),
            "model": c["model"],
        },
        "whisper": {"available": (WHISPER_DIR / "model.bin").is_file(), "model": "Whisper base (offline)"},
        "fonts": FONTS,
    }


def save_settings(body: Dict[str, Any]) -> Dict[str, Any]:
    updates = {}
    for field, key in (("account_id", "CLOUDFLARE_ACCOUNT_ID"), ("api_token", "CLOUDFLARE_API_TOKEN")):
        value = str(body.get(field, "") or "").strip()
        if value:
            if any(ch in value for ch in "\r\n\"'= ") or len(value) > 200:
                raise ValueError(f"Invalid {field.replace('_', ' ')}")
            updates[key] = value
    if body.get("clear"):
        updates = {"CLOUDFLARE_ACCOUNT_ID": "", "CLOUDFLARE_API_TOKEN": ""}
    if updates:
        lines = ENV_FILE.read_text(encoding="utf-8").splitlines() if ENV_FILE.is_file() else []
        seen = set()
        out = []
        for line in lines:
            key = line.split("=", 1)[0].strip() if "=" in line and not line.lstrip().startswith("#") else None
            if key in updates:
                out.append(f"{key}={updates[key]}")
                seen.add(key)
            else:
                out.append(line)
        for key, value in updates.items():
            if key not in seen:
                out.append(f"{key}={value}")
        ENV_FILE.write_text("\n".join(out) + "\n", encoding="utf-8")
        for key, value in updates.items():  # the process env wins over .env, keep it in sync
            if value:
                os.environ[key] = value
            else:
                os.environ.pop(key, None)
    return settings_status()


# --------------------------------------------------------------------------- #
# Describe -> AI coloring page -> strokes
# --------------------------------------------------------------------------- #
def describe(prompt: str, seed: Optional[int] = None) -> Dict[str, Any]:
    from PIL import Image
    from .cloud_ai import OUTPUT_SIZE, call_cloudflare, config, page_to_strokes

    subject = " ".join(str(prompt or "").split())[:300]
    if not subject:
        raise ValueError("Describe what to draw, for example: a cat sitting on a chair")
    t0 = time.time()
    raw = call_cloudflare(None, prompt=DESCRIBE_TEMPLATE.format(subject=subject), seed=seed)
    t_cloud = time.time() - t0
    page = np.array(Image.open(io.BytesIO(raw)).convert("RGB"))
    strokes, ink = page_to_strokes(page, OUTPUT_SIZE)
    info = {"engine": "describe", "subject": subject, "model": config()["model"], "stroke_count": len(strokes),
            "timings": {"cloud": round(t_cloud, 2), "total": round(time.time() - t0, 2)}}
    return {"strokes": strokes, "size": OUTPUT_SIZE, "line_image": np.where(ink, 0, 255).astype(np.uint8),
            "page": page, "info": info}


# --------------------------------------------------------------------------- #
# Text in a Hershey single-line font
# --------------------------------------------------------------------------- #
def _font(name: str):
    from HersheyFonts import HersheyFonts
    f = HersheyFonts()
    f.load_default_font(name if name in FONTS else "futural")
    return f


SMOOTH_FONTS = {"scripts", "scriptc", "cursive", "timesi", "gothiceng", "gothicger"}


def _chaikin(pts: np.ndarray, iterations: int = 2) -> np.ndarray:
    """Corner-cutting: turns the coarse Hershey polylines into smooth curves, keeping the end points."""
    for _ in range(iterations):
        if len(pts) < 3:
            return pts
        q = 0.75 * pts[:-1] + 0.25 * pts[1:]
        r = 0.25 * pts[:-1] + 0.75 * pts[1:]
        mid = np.empty((2 * len(q), 2))
        mid[0::2], mid[1::2] = q, r
        pts = np.vstack([pts[:1], mid[1:-1], pts[-1:]])
    return pts


def _line_strokes(f, text: str, smooth: bool = False) -> List[np.ndarray]:
    out = [np.asarray(s, np.float64) for s in f.strokes_for_text(text) if len(s) >= 2]
    return [_chaikin(s) for s in out] if smooth else out


def _wrap(words: List[str], width: int) -> List[str]:
    lines, cur = [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines


def write_text(text: str, font: str = "futural", size: int = 1024) -> Dict[str, Any]:
    t0 = time.time()
    text = str(text or "").replace("\r", "")
    text = "\n".join(" ".join(line.split()) for line in text.split("\n")).strip()[:240]
    if not text:
        raise ValueError("Type or say the text to write")
    f = _font(font)
    # choose a wrap width that makes the block as close to square-ish (4:3) as possible
    paragraphs = text.split("\n")
    best = None
    longest = max(len(p) for p in paragraphs)
    for width in range(max(4, min(longest, 8)), max(longest, 8) + 1):
        lines = [ln for p in paragraphs for ln in (_wrap(p.split(), width) or [""])]
        aspect = (width * 0.9) / (len(lines) * 1.6)
        score = abs(np.log(aspect / 1.4))
        if best is None or score < best[0]:
            best = (score, lines)
    lines = best[1]
    line_h = 34.0  # Hershey em ~ 32 units (cap -12 .. descender +16)
    strokes: List[np.ndarray] = []
    widths = []
    rendered = []
    for ln in lines:
        ss = _line_strokes(f, ln, smooth=font in SMOOTH_FONTS)
        xs = [s[:, 0] for s in ss]
        w = (max(x.max() for x in xs) - min(x.min() for x in xs)) if xs else 0.0
        x0 = min(x.min() for x in xs) if xs else 0.0
        rendered.append((ss, x0, w))
        widths.append(w)
    block_w = max(widths) if widths else 1.0
    for i, (ss, x0, w) in enumerate(rendered):
        dx = -x0 + (block_w - w) / 2.0  # center each line
        dy = i * line_h
        for s in ss:
            strokes.append(s + np.array([dx, dy]))
    if not strokes:
        raise ValueError("This font has no letters for that text (Hershey fonts cover Latin, Greek and Cyrillic)")
    allp = np.vstack(strokes)
    lo, hi = allp.min(0), allp.max(0)
    k = (size * 0.9) / max(hi - lo)
    center = (lo + hi) / 2.0
    strokes = [(s - center) * k + size / 2.0 for s in strokes]
    from .vectorize import render_strokes
    img = render_strokes(strokes, size, size, 3)
    line_image = img if img.mean() > 128 else 255 - img
    info = {"engine": "text", "subject": text, "font": font, "font_label": FONTS.get(font, font),
            "stroke_count": len(strokes), "timings": {"total": round(time.time() - t0, 3)}}
    return {"strokes": strokes, "size": size, "line_image": line_image, "page": None, "info": info}


# --------------------------------------------------------------------------- #
# Voice: local Whisper base
# --------------------------------------------------------------------------- #
_WHISPER = None
_WHISPER_LOCK = threading.Lock()


def _whisper():
    global _WHISPER
    if _WHISPER is None:
        if not (WHISPER_DIR / "model.bin").is_file():
            raise ValueError("Whisper base is not installed: run Setup-Clean-Line.ps1 (it downloads models/whisper-base)")
        from faster_whisper import WhisperModel
        _WHISPER = WhisperModel(str(WHISPER_DIR), device="cpu", compute_type="int8",
                                cpu_threads=max(1, (os.cpu_count() or 4)))
    return _WHISPER


def _decode_audio(path: Path, rate: int = 16000) -> np.ndarray:
    """Any browser recording (webm/ogg/mp4/wav) -> mono float32 at 16 kHz (works with every PyAV version)."""
    import av
    chunks = []
    with av.open(str(path), mode="r") as container:
        stream = next(s for s in container.streams if s.type == "audio")
        resampler = av.AudioResampler(format="s16", layout="mono", rate=rate)
        for frame in container.decode(stream):
            for out in resampler.resample(frame) or []:
                chunks.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None) or []:
            chunks.append(out.to_ndarray().reshape(-1))
    if not chunks:
        return np.zeros(0, np.float32)
    return np.concatenate(chunks).astype(np.float32) / 32768.0


def transcribe(audio_data_url: str, task: str = "transcribe", language: Optional[str] = None) -> Dict[str, Any]:
    if not audio_data_url.startswith("data:audio/") or ";base64," not in audio_data_url:
        raise ValueError("Voice expects an audio data URL")
    raw = base64.b64decode(audio_data_url.split(",", 1)[1], validate=True)
    if len(raw) > 6_000_000:
        raise ValueError("Recording is too long (keep it under a minute)")
    suffix = ".ogg" if "ogg" in audio_data_url[:30] else ".mp4" if "mp4" in audio_data_url[:30] else ".webm"
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="aira-voice-") as folder:
        path = Path(folder) / f"voice{suffix}"
        path.write_bytes(raw)
        samples = _decode_audio(path)
        if samples.size < 1600:
            raise ValueError("Recording is too short")
        with _WHISPER_LOCK:
            model = _whisper()
            segments, info = model.transcribe(samples, task="translate" if task == "translate" else "transcribe",
                                              language=language or None, beam_size=5, vad_filter=True)
            text = " ".join(s.text.strip() for s in segments).strip()
    return {"ok": True, "text": text, "language": info.language,
            "language_probability": round(float(info.language_probability), 2),
            "seconds": round(time.time() - t0, 2)}

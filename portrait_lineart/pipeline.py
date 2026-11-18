"""Photo -> clean coloring-book line art -> plotter strokes.

Stages
  1. orient + find the face (YuNet)
  2. subject matte (alpha from the browser cutout, or rembg if available)
  3. head-and-shoulders framing at a fixed working size
  4. semantic face parsing (BiSeNet, optional but recommended)
  5. line sources, each restricted to the regions where it is reliable:
       - silhouette + region boundaries from the matte / parsing map
       - fine interior lines from a line-art network (eyes, nose, mouth, hair strands)
       - clothing reduced to strong, long edges (collar, seams, logos)
  6. compose into one uniform-weight raster
  7. centerline vectorization (see vectorize.py)
"""

from __future__ import annotations

import io
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")

import cv2
import numpy as np
from PIL import Image, ImageOps

from .vectorize import (VectorizeParams, ink_to_strokes, polyline_length,
                        render_strokes, strokes_to_svg)

ROOT = Path(__file__).resolve().parent.parent
MODELS = Path(os.environ.get("AIRA_MODELS", str(ROOT / "models")))


def _first_existing(*paths: Path) -> Optional[Path]:
    for p in paths:
        if p and p.is_file():
            return p
    return None


def find_model(name: str) -> Optional[Path]:
    return _first_existing(
        MODELS / name,
        ROOT / "svg gen" / "models" / name,
        ROOT / "version mohsin" / "portrait_dobot_v3" / "assets" / name,
        ROOT / "version mohsin" / "portrait_dobot_v3" / "modeles" / name,
    )


# CelebAMask-HQ label order used by BiSeNet face-parsing checkpoints.
PARSE_LABELS = [
    "background", "skin", "l_brow", "r_brow", "l_eye", "r_eye", "eye_g",
    "l_ear", "r_ear", "ear_r", "nose", "mouth", "u_lip", "l_lip", "neck",
    "neck_l", "cloth", "hair", "hat",
]
L = {n: i for i, n in enumerate(PARSE_LABELS)}


@dataclass
class PipelineParams:
    size: int = 1024                 # working canvas (square)
    detail: float = 0.5              # 0 = minimal outline, 1 = rich detail
    hair_strands: float = 0.5        # 0 = outline only, 1 = many strands
    clothing: float = 0.35           # 0 = silhouette only, 1 = all garment edges
    line_px: int = 3                 # uniform stroke width in the composed raster
    frame_face_width: float = 0.30   # face box width as fraction of canvas
    frame_eye_y: float = 0.40        # eye line height as fraction of canvas
    vector: VectorizeParams = field(default_factory=VectorizeParams)


@dataclass
class Result:
    strokes: List[np.ndarray]
    size: int
    line_image: np.ndarray           # composed coloring-page raster (uint8, white bg)
    framed_rgb: np.ndarray
    matte: np.ndarray
    parsing: Optional[np.ndarray]
    timings: Dict[str, float]
    info: Dict[str, object]

    def svg(self, width_mm: Optional[float] = 150.0, stroke_width: float = 2.0) -> str:
        return strokes_to_svg(self.strokes, self.size, self.size, stroke_width, width_mm)

    def preview(self, thickness: int = 2) -> np.ndarray:
        return render_strokes(self.strokes, self.size, self.size, thickness)


# --------------------------------------------------------------------------- #
# Model wrappers (lazy, cached)
# --------------------------------------------------------------------------- #
class _Models:
    def __init__(self) -> None:
        self._face = None
        self._face_size = None
        self._lineart = None
        self._parser = None
        self._rembg = None

    # -- face detection ---------------------------------------------------- #
    def detect_face(self, rgb: np.ndarray):
        path = find_model("face_detection_yunet_2023mar.onnx")
        if path is None:
            return None
        h, w = rgb.shape[:2]
        scale = min(1.0, 960.0 / max(h, w))
        small = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else rgb
        sh, sw = small.shape[:2]
        if self._face is None or self._face_size != (sw, sh):
            self._face = cv2.FaceDetectorYN_create(str(path), "", (sw, sh), 0.7, 0.3, 5000)
            self._face_size = (sw, sh)
        _, faces = self._face.detect(np.ascontiguousarray(small[:, :, ::-1]))
        if faces is None or len(faces) == 0:
            return None
        f = max(faces, key=lambda r: float(r[2]) * float(r[3]))
        return (f[:14] / scale).astype(np.float64)  # box + 5 landmarks

    # -- background removal ------------------------------------------------- #
    def matte(self, rgb: np.ndarray) -> Optional[np.ndarray]:
        try:
            if self._rembg is None:
                seg = find_model("u2net_human_seg.onnx")
                if seg is not None:
                    os.environ.setdefault("U2NET_HOME", str(seg.parent))
                from rembg import new_session
                self._rembg = new_session("u2net_human_seg")
            from rembg import remove
            out = remove(Image.fromarray(rgb), session=self._rembg, only_mask=True, post_process_mask=True)
            return np.asarray(out.convert("L"))
        except Exception:
            return None

    # -- line-art network (Informative Drawings, ONNX) ----------------------- #
    def lineart(self, rgb: np.ndarray) -> Optional[np.ndarray]:
        path = find_model("informative-drawings.onnx")
        if path is None:
            return None
        import onnxruntime as ort
        if self._lineart is None:
            so = ort.SessionOptions()
            so.intra_op_num_threads = max(1, (os.cpu_count() or 4) - 1)
            self._lineart = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        feed = rgb.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        name = self._lineart.get_inputs()[0].name
        out = self._lineart.run(None, {name: feed})[0][0, 0]
        return np.clip(out, 0, 1)  # 1 = paper, 0 = ink

    # -- face parsing (BiSeNet, ONNX) ---------------------------------------- #
    def parse(self, rgb: np.ndarray) -> Optional[np.ndarray]:
        path = find_model("face_parsing_resnet34.onnx") or find_model("face_parsing_resnet18.onnx")
        if path is None:
            return None
        import onnxruntime as ort
        if self._parser is None:
            self._parser = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        h, w = rgb.shape[:2]
        x = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
        x = x.transpose(2, 0, 1)[None]
        inp = self._parser.get_inputs()[0].name
        logits = self._parser.run(None, {inp: x})[0][0]  # (19, 512, 512)
        # upsample logits, then argmax: smoother boundaries than upsampling labels
        up = np.stack([cv2.resize(c, (w, h), interpolation=cv2.INTER_LINEAR) for c in logits], 0)
        return up.argmax(0).astype(np.uint8)


MODELS_CACHE = _Models()


# --------------------------------------------------------------------------- #
# Framing
# --------------------------------------------------------------------------- #
def load_rgba(data: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img).convert("RGBA")
    return np.asarray(img)


def frame_portrait(rgb: np.ndarray, alpha: np.ndarray, face, p: PipelineParams):
    """Similarity-transform the photo so the face sits at a fixed place and roll is 0."""
    h, w = rgb.shape[:2]
    S = p.size
    if face is not None:
        x, y, fw, fh = face[:4]
        re, le = face[4:6], face[6:8]  # YuNet: right eye, left eye (image coords)
        eye_c = (re + le) / 2.0
        angle = float(np.degrees(np.arctan2(le[1] - re[1], le[0] - re[0])))
        if abs(angle) > 25:
            angle = 0.0
        scale = (p.frame_face_width * S) / max(fw, 1.0)
        M = cv2.getRotationMatrix2D((float(eye_c[0]), float(eye_c[1])), angle, scale)
        M[0, 2] += S * 0.5 - eye_c[0]
        M[1, 2] += S * p.frame_eye_y - eye_c[1]
    else:
        # no face: fit the matte bbox
        ys, xs = np.nonzero(alpha > 32)
        if len(xs) == 0:
            ys, xs = np.array([0, h - 1]), np.array([0, w - 1])
        bw, bh = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
        scale = 0.92 * S / max(bw, bh)
        cx, cy = (xs.max() + xs.min()) / 2, ys.min() + bh / 2
        M = np.array([[scale, 0, S / 2 - cx * scale], [0, scale, S / 2 - cy * scale]], np.float64)
    framed = cv2.warpAffine(rgb, M, (S, S), flags=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    fa = cv2.warpAffine(alpha, M, (S, S), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    # mark which framed pixels came from inside the photo (the rest is padding)
    valid = cv2.warpAffine(np.full((h, w), 255, np.uint8), M, (S, S), flags=cv2.INTER_NEAREST, borderValue=0)
    return framed, fa, valid > 0, M


# --------------------------------------------------------------------------- #
# Line sources
# --------------------------------------------------------------------------- #
def _draw_contours(canvas: np.ndarray, mask: np.ndarray, thickness: int, min_len: float,
                   smooth_sigma: float = 2.0, closed_only: bool = False, clip: Optional[np.ndarray] = None,
                   external: bool = False) -> None:
    """Draw smoothed outlines of a binary mask onto canvas (255 = ink)."""
    m = mask.astype(np.uint8)
    if m.sum() == 0:
        return
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL if external else cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    layer = np.zeros_like(canvas)
    for c in contours:
        c = c[:, 0, :].astype(np.float64)
        if len(c) < 8 or cv2.arcLength(c.astype(np.float32), True) < min_len:
            continue
        if smooth_sigma > 0:
            from scipy.ndimage import gaussian_filter1d
            c = np.stack([gaussian_filter1d(c[:, 0], smooth_sigma, mode="wrap"),
                          gaussian_filter1d(c[:, 1], smooth_sigma, mode="wrap")], 1)
        pts = np.round(c * 4).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(layer, [pts], True, 255, thickness, cv2.LINE_AA, shift=2)
    if clip is not None:
        layer[~clip] = 0
    np.maximum(canvas, layer, out=canvas)


def _boundary_between(parse: np.ndarray, a_labels, b_labels, width: int = 2) -> np.ndarray:
    a = np.isin(parse, a_labels).astype(np.uint8)
    b = np.isin(parse, b_labels).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * width + 1,) * 2)
    return (cv2.dilate(a, k) & cv2.dilate(b, k)).astype(bool)


def _keep_long_components(ink: np.ndarray, min_len_px: float) -> np.ndarray:
    """Keep connected ink pieces whose skeleton is at least min_len_px long."""
    from skimage.morphology import skeletonize
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    if n <= 1:
        return ink
    sk = skeletonize(ink)
    lengths = np.bincount(labels[sk], minlength=n)
    keep = lengths >= min_len_px
    keep[0] = False
    return keep[labels]


def _edge_preserving(rgb: np.ndarray, strength: int) -> np.ndarray:
    out = rgb
    for _ in range(max(0, strength)):
        out = cv2.bilateralFilter(out, d=9, sigmaColor=40, sigmaSpace=9)
    return out


# --------------------------------------------------------------------------- #
# Feature layers
# --------------------------------------------------------------------------- #
def _largest_subject(mask: np.ndarray, min_frac: float = 0.04) -> np.ndarray:
    """Keep the main person blob(s) and fill holes in the matte."""
    m = mask.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    big = areas.max()
    keep = np.zeros(n, bool)
    keep[1:] = areas >= max(big * min_frac, 1)
    out = keep[labels].astype(np.uint8)
    # fill interior holes
    inv = 1 - out
    n2, lab2, st2, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    border = np.unique(np.concatenate([lab2[0], lab2[-1], lab2[:, 0], lab2[:, -1]]))
    holes = np.ones(n2, bool)
    holes[border] = False
    holes[0] = holes[0] and 0 not in border
    out[holes[lab2] & (inv > 0)] = 1
    k = max(7, int(min(mask.shape) * 0.03) | 1)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k // 2 | 1,) * 2))
    return out.astype(bool)


def _smooth_closed(c: np.ndarray, sigma: float) -> np.ndarray:
    from scipy.ndimage import gaussian_filter1d
    if sigma <= 0 or len(c) < 5:
        return c
    return np.stack([gaussian_filter1d(c[:, 0], sigma, mode="wrap"),
                     gaussian_filter1d(c[:, 1], sigma, mode="wrap")], 1)


def _poly(canvas: np.ndarray, pts: np.ndarray, closed: bool, thickness: int) -> None:
    if len(pts) < 2:
        return
    q = np.round(np.asarray(pts) * 4).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(canvas, [q], closed, 255, thickness, cv2.LINE_AA, shift=2)


def _main_contour(mask: np.ndarray) -> Optional[np.ndarray]:
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea)
    return c[:, 0, :].astype(np.float64)


def _iris_rel(eye: np.ndarray, gray: np.ndarray) -> Optional[float]:
    """Horizontal iris position inside an eye blob, 0 = left corner, 1 = right corner."""
    ys, xs = np.nonzero(eye)
    if len(xs) < 20:
        return None
    x0, x1 = xs.min(), xs.max()
    eh, ew = ys.max() - ys.min() + 1, x1 - x0 + 1
    g = gray.astype(np.float32)
    cols = np.full(ew, 255.0)
    for i, x in enumerate(range(x0, x1 + 1)):
        col = eye[:, x]
        if col.sum() >= 2:
            cols[i] = float(np.mean(g[col, x]))
    k = max(3, int(eh * 0.8) | 1)
    cols_s = np.convolve(np.pad(cols, k // 2, mode="edge"), np.ones(k) / k, mode="valid")
    margin = int(ew * 0.18)
    if ew - 2 * margin < 3:
        return None
    ix = margin + int(np.argmin(cols_s[margin:ew - margin]))
    return ix / float(ew - 1)


def _draw_eye(canvas: np.ndarray, eye: np.ndarray, gray: np.ndarray, lw: int, S: int,
              iris_rel: Optional[float] = None, avoid: Optional[np.ndarray] = None) -> None:
    """Almond outline from the parsing region + iris circle + pupil dot."""
    if eye.sum() < 40:
        return
    ys0 = np.nonzero(eye)[0]
    grow = max(2, int(round((ys0.max() - ys0.min() + 1) * 0.22)))
    grown = cv2.dilate(eye.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1)))
    c = _main_contour(grown)
    if c is None or len(c) < 10:
        return
    c = _smooth_closed(c, 2.0)
    _poly(canvas, c, True, lw)
    # bolder upper lid, like an inked coloring page
    upper = c[c[:, 1] < c[:, 1].mean()]
    if len(upper) > 4:
        _poly(canvas, upper[np.argsort(upper[:, 0])], False, lw + 1)
    # upper-lid crease: the top of the almond, lifted and slightly widened (coloring-book eyes)
    cy_e = c[:, 1].mean()
    cx_e = c[:, 0].mean()
    top = c[c[:, 1] < cy_e]
    if len(top) > 6:
        top = top[np.argsort(top[:, 0])]
        eh_c = c[:, 1].max() - c[:, 1].min()
        ew_c = c[:, 0].max() - c[:, 0].min()
        if ew_c > 1.8 * eh_c:  # skip for nearly closed / odd eye blobs
            lifted = top.copy()
            lifted[:, 0] = cx_e + (lifted[:, 0] - cx_e) * 1.05
            lifted[:, 1] -= max(3.0, 0.55 * eh_c)
            n = len(lifted)
            crease = np.zeros_like(canvas)
            _poly(crease, lifted[int(n * 0.18):int(n * 0.88)], False, max(2, lw - 1))
            if avoid is not None:
                crease[avoid] = 0
            np.maximum(canvas, crease, out=canvas)
    eye = grown.astype(bool)
    ys, xs = np.nonzero(eye)
    x0, x1 = xs.min(), xs.max()
    eh = ys.max() - ys.min() + 1
    ew = x1 - x0 + 1
    # iris: shared position from both eyes when available (a shadowed eye misleads the darkest-column search)
    rel = iris_rel if iris_rel is not None else _iris_rel(eye, gray)
    if rel is None:
        return
    ix = int(round(x0 + rel * (ew - 1)))
    col = np.nonzero(eye[:, ix])[0]
    if len(col) == 0:
        return
    iy = 0.5 * (col.min() + col.max())
    r = max(3.0, min(0.62 * eh, 0.24 * ew))
    iris = np.zeros_like(canvas)
    cv2.circle(iris, (int(ix * 4), int(iy * 4)), int(r * 4), 255, max(2, lw - 1), cv2.LINE_AA, shift=2)
    inside = cv2.erode(grown, np.ones((3, 3), np.uint8)).astype(bool)
    iris[~inside] = 0
    np.maximum(canvas, iris, out=canvas)
    cv2.circle(canvas, (int(round(ix)), int(round(iy))), max(2, int(round(r * 0.42))), 255, -1, cv2.LINE_AA)


def _column_profile(mask: np.ndarray, top: bool, min_count: int = 1):
    """For each column of a mask: the top-most (or bottom-most) pixel -> (x, y) polyline."""
    cols = np.nonzero(mask.sum(0) >= min_count)[0]
    if len(cols) < 3:
        return None
    pts = []
    for x in cols:
        ys = np.nonzero(mask[:, x])[0]
        pts.append((float(x), float(ys.min() if top else ys.max())))
    return np.asarray(pts)


def _smooth_open(pts: np.ndarray, sigma: float) -> np.ndarray:
    from scipy.ndimage import gaussian_filter1d
    if pts is None or len(pts) < 5 or sigma <= 0:
        return pts
    out = np.stack([gaussian_filter1d(pts[:, 0], sigma, mode="nearest"),
                    gaussian_filter1d(pts[:, 1], sigma, mode="nearest")], 1)
    out[0], out[-1] = pts[0], pts[-1]
    return out


def _middle(pts: np.ndarray, frac: float) -> np.ndarray:
    if pts is None or len(pts) < 4:
        return pts
    n = len(pts)
    cut = int(n * (1 - frac) / 2)
    return pts[cut:n - cut] if n - 2 * cut >= 2 else pts


def _draw_lips(canvas: np.ndarray, parse: np.ndarray, lw: int, S: int) -> None:
    """Coloring-book mouth: mouth line, cupid's-bow upper lip, short lower-lip curve."""
    u = parse == L["u_lip"]
    lo = parse == L["l_lip"]
    m = parse == L["mouth"]
    if u.sum() + lo.sum() < 60:
        return
    lips = u | lo | m
    open_mouth = m.sum() > 0.18 * max(1, lips.sum())
    # mouth line = bottom edge of the upper lip, spanning the whole mouth width
    mouth_line = _column_profile(u, top=False) if u.any() else None
    if mouth_line is not None:
        xs_all = np.nonzero(lips.sum(0))[0]
        # extend to the true corners of the lips
        left, right = xs_all.min(), xs_all.max()
        ml = mouth_line
        ext = []
        if ml[0, 0] > left + 2:
            ext.append(np.array([[left, ml[0, 1] + 1.0]]))
        ext.append(ml)
        if ml[-1, 0] < right - 2:
            ext.append(np.array([[right, ml[-1, 1] + 1.0]]))
        mouth_line = np.vstack(ext)
        _poly(canvas, _smooth_open(mouth_line, 3.0), False, lw)
    # upper lip top (cupid's bow), middle 85%
    top = _column_profile(u, top=True)
    if top is not None:
        _poly(canvas, _smooth_open(_middle(top, 0.85), 2.0), False, lw)
    # lower lip: only a short curve under its middle
    bottom = _column_profile(lo, top=False)
    if bottom is not None:
        _poly(canvas, _smooth_open(_middle(bottom, 0.45), 2.5), False, lw)
    if open_mouth:
        inner_low = _column_profile(m, top=False)
        if inner_low is not None:
            _poly(canvas, _smooth_open(_middle(inner_low, 0.8), 2.0), False, lw)


def _draw_nose_bridge(canvas: np.ndarray, nose: np.ndarray, lw: int, S: int) -> None:
    """Short side lines of the nose (coloring pages draw them from the brows down)."""
    if nose.sum() < 120:
        return
    ys, xs = np.nonzero(nose)
    top, bot = ys.min(), ys.max()
    h = bot - top + 1
    rows = range(int(top + 0.28 * h), int(top + 0.66 * h))
    left, right = [], []
    for y in rows:
        row = np.nonzero(nose[y])[0]
        if len(row) >= 2:
            left.append((float(row.min()), float(y)))
            right.append((float(row.max()), float(y)))
    if len(left) < 6:
        return
    cx = 0.5 * (xs.min() + xs.max())
    for side in (np.asarray(left), np.asarray(right)):
        side = side.copy()
        side[:, 0] = cx + (side[:, 0] - cx) * 0.85  # pull slightly inward
        _poly(canvas, _smooth_open(side, 4.0), False, max(2, lw - 1))


def _draw_collar_band(canvas: np.ndarray, cloth: np.ndarray, neck: np.ndarray, lw: int, S: int,
                      clip: np.ndarray) -> None:
    """Second line parallel to the neckline: the collar rib of a T-shirt / jersey."""
    if cloth.sum() < S * S * 0.01 or neck.sum() < S * S * 0.003:
        return
    k = max(6, int(S * 0.022))
    eroded = cv2.erode(cloth.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
    layer = np.zeros_like(canvas)
    _draw_contours(layer, eroded.astype(bool), lw, min_len=S * 0.05, smooth_sigma=3.0)
    dist = cv2.distanceTransform((~neck).astype(np.uint8), cv2.DIST_L2, 3)
    near = dist < 1.8 * k
    # only the part under the neck (not the shoulders' outer edge)
    ys, xs = np.nonzero(neck)
    x_lo, x_hi = xs.min() - k, xs.max() + k
    xx = np.arange(S)[None, :]
    near &= (xx > x_lo) & (xx < x_hi)
    layer[~(near & clip)] = 0
    np.maximum(canvas, layer, out=canvas)


def _draw_inner_ears(canvas: np.ndarray, ears: np.ndarray, face_cx: float, lw: int, S: int) -> None:
    """A 'C' curve inside each ear, on the side away from the face."""
    n, lab, st, cents = cv2.connectedComponentsWithStats(ears.astype(np.uint8), connectivity=8)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < S * S * 0.0008:
            continue
        ear = (lab == i).astype(np.uint8)
        w = st[i, cv2.CC_STAT_WIDTH]
        k = max(2, int(w * 0.28))
        core = cv2.erode(ear, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
        c = _main_contour(core)
        if c is None or len(c) < 12:
            continue
        c = _smooth_closed(c, 2.5)
        outer = c[:, 0] < cents[i][0] if cents[i][0] < face_cx else c[:, 0] > cents[i][0]
        # contiguous run on the outer side
        idx = np.nonzero(outer)[0]
        if len(idx) < 6:
            continue
        roll = int(np.argmin(outer)) if not outer.all() else 0
        cc = np.roll(c, -roll, axis=0)
        oo = np.roll(outer, -roll)
        seg = cc[oo]
        if len(seg) >= 6:
            _poly(canvas, seg, False, max(2, lw - 1))


def _draw_nose(canvas: np.ndarray, nose: np.ndarray, ink_strength: np.ndarray, lw: int, S: int) -> None:
    """Coloring-book nose: the lower arc of the nose region (tip + wings) plus strong nostril marks."""
    if nose.sum() < 80:
        return
    c = _main_contour(nose)
    if c is None or len(c) < 20:
        return
    c = _smooth_closed(c, 3.0)
    ys = c[:, 1]
    top, bot = ys.min(), ys.max()
    cut = top + 0.62 * (bot - top)
    keep = ys >= cut
    # draw contiguous runs of the lower part as open arcs
    idx = np.nonzero(keep)[0]
    if len(idx) == 0:
        return
    runs, start = [], idx[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if b != a + 1:
            runs.append((start, a))
            start = b
    runs.append((start, idx[-1]))
    if len(runs) > 1 and runs[0][0] == 0 and runs[-1][1] == len(c) - 1:  # wrap-around
        first = runs.pop(0)
        last = runs.pop()
        runs.append((last[0], first[1] + len(c)))
    cc = np.vstack([c, c])
    for a, b in runs:
        seg = cc[a:b + 1]
        if len(seg) > 6:
            _poly(canvas, seg, False, lw)
    # nostrils: strongest model ink in the bottom of the nose
    yy = np.arange(S)[:, None]
    zone = cv2.dilate(nose.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool) & (yy > top + 0.55 * (bot - top))
    nostr = zone & (ink_strength > 0.55)
    nostr = _keep_long_components(nostr, S * 0.012)
    np.maximum(canvas, cv2.dilate(nostr.astype(np.uint8) * 255, np.ones((lw, lw), np.uint8)), out=canvas)


def _beard_mask(on_white: np.ndarray, skin: np.ndarray, nose: np.ndarray, lips: np.ndarray, S: int,
                texture: Optional[np.ndarray] = None) -> np.ndarray:
    """Dark, textured facial hair in the lower face (smooth shadows are rejected)."""
    if skin.sum() < 500 or nose.sum() < 30:
        return np.zeros_like(skin)
    lab = cv2.cvtColor(on_white, cv2.COLOR_RGB2LAB)
    Lc = cv2.GaussianBlur(lab[:, :, 0].astype(np.float32), (0, 0), 2.0)
    ys = np.nonzero(nose)[0]
    n_top, n_bot = ys.min(), ys.max()
    nose_mid = n_top + 0.7 * (n_bot - n_top)
    yy = np.arange(S)[:, None]
    cheeks = skin & (yy > n_top) & (yy < n_bot)
    if cheeks.sum() < 100:
        return np.zeros_like(skin)
    # the brighter half of the cheeks is the bare-skin reference (one side may be in shadow)
    ref = float(np.percentile(Lc[cheeks], 70))
    lower = (skin | lips) & (yy > nose_mid)
    dark = lower & (Lc < ref * 0.70) & ~lips
    if texture is not None:
        dark &= texture > 0.30
    dark = cv2.morphologyEx(dark.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= S * S * 0.0012
    return keep[labels]


def _hair_flow(gray: np.ndarray, hair: np.ndarray, S: int, density: float, seed: int = 7) -> List[np.ndarray]:
    """Stylized hair strands: streamlines of the hair's local orientation field."""
    if hair.sum() < S * S * 0.004 or density <= 0.02:
        return []
    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.2)
    ix = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    iy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    sig = S * 0.016
    jxx = cv2.GaussianBlur(ix * ix, (0, 0), sig)
    jyy = cv2.GaussianBlur(iy * iy, (0, 0), sig)
    jxy = cv2.GaussianBlur(ix * iy, (0, 0), sig)
    theta = 0.5 * np.arctan2(2 * jxy, jxx - jyy) + np.pi / 2  # along the strands
    tr = jxx + jyy + 1e-6
    coh = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / tr
    fx, fy = np.cos(theta), np.sin(theta)
    inside = cv2.erode(hair.astype(np.uint8), cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (int(S * 0.018) | 1,) * 2)).astype(bool)
    if inside.sum() < 50:
        return []
    spacing = S * (0.062 - 0.036 * density)
    sep = spacing * 0.42
    step = 2.0
    max_len = S * (0.14 + 0.10 * density)
    min_len = S * 0.035
    occ = np.zeros((S, S), np.uint8)
    rng = np.random.RandomState(seed)
    gy, gx = np.mgrid[spacing / 2:S:spacing, spacing / 2:S:spacing]
    seeds = np.stack([gx.ravel(), gy.ravel()], 1) + rng.uniform(-spacing * 0.3, spacing * 0.3, (gx.size, 2))
    seeds = [s for s in seeds if 0 <= s[0] < S and 0 <= s[1] < S and inside[int(s[1]), int(s[0])]]
    seeds.sort(key=lambda s: -coh[int(s[1]), int(s[0])])
    lines: List[np.ndarray] = []

    def trace(p0, sign):
        pts = []
        x, y = p0
        dx, dy = fx[int(y), int(x)] * sign, fy[int(y), int(x)] * sign
        length = 0.0
        while length < max_len / 2:
            xi, yi = int(x), int(y)
            if not (0 <= xi < S and 0 <= yi < S) or not inside[yi, xi] or occ[yi, xi]:
                break
            if coh[yi, xi] < 0.08:
                break
            ndx, ndy = fx[yi, xi], fy[yi, xi]
            if ndx * dx + ndy * dy < 0:
                ndx, ndy = -ndx, -ndy
            if ndx * dx + ndy * dy < 0.85:  # > ~32 degree turn per step: stop
                break
            dx, dy = 0.6 * dx + 0.4 * ndx, 0.6 * dy + 0.4 * ndy
            nrm = (dx * dx + dy * dy) ** 0.5 or 1.0
            dx, dy = dx / nrm, dy / nrm
            x, y = x + step * dx, y + step * dy
            pts.append((x, y))
            length += step
        return pts

    for s in seeds:
        if occ[int(s[1]), int(s[0])] or coh[int(s[1]), int(s[0])] < 0.12:
            continue
        fwd = trace(s, 1.0)
        bwd = trace(s, -1.0)
        line = np.asarray(bwd[::-1] + [tuple(s)] + fwd, np.float64)
        if len(line) < 3:
            continue
        L_ = float(np.hypot(*np.diff(line, axis=0).T).sum())
        if L_ < min_len:
            continue
        lines.append(line)
        q = np.round(line).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(occ, [q], False, 1, max(1, int(2 * sep)))
    return lines


def _texture_map(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    hp = gray - cv2.GaussianBlur(gray, (0, 0), 2.2)
    energy = cv2.GaussianBlur(np.abs(hp), (0, 0), 9.0)
    return np.clip(energy / 14.0, 0.0, 1.0)


# --------------------------------------------------------------------------- #
def is_line_art(rgb: np.ndarray) -> bool:
    """True for black-on-white drawings (coloring pages, scans) that need no AI step."""
    small = cv2.resize(rgb, (256, int(256 * rgb.shape[0] / rgb.shape[1])), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    g = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    white = float(np.mean(g > 215))
    mid = float(np.mean((g > 90) & (g <= 215)))
    return white > 0.78 and mid < 0.10 and float(np.mean(hsv[:, :, 1])) < 30


def _line_art_result(rgb: np.ndarray, p: PipelineParams, t0: float) -> Result:
    S = p.size
    h, w = rgb.shape[:2]
    scale = (S * 0.96) / max(h, w)
    img = cv2.resize(rgb, (max(1, int(w * scale)), max(1, int(h * scale))),
                     interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    canvas = np.full((S, S, 3), 255, np.uint8)
    y0, x0 = (S - img.shape[0]) // 2, (S - img.shape[1]) // 2
    canvas[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = img
    gray = cv2.cvtColor(canvas, cv2.COLOR_RGB2GRAY)
    from .vectorize import binarize
    ink = binarize(gray, p.vector)
    strokes = ink_to_strokes(ink, p.vector)
    T = {"total": time.time() - t0}
    info = {"mode": "line-art input", "face_found": False, "parsing": False, "lineart_model": False,
            "stroke_count": len(strokes), "length_px": float(sum(polyline_length(s) for s in strokes))}
    return Result(strokes, S, gray, canvas, np.full((S, S), 255, np.uint8), None, T, info)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def generate(image_bytes: bytes, params: Optional[PipelineParams] = None,
             fallback_cutout: Optional[bytes] = None) -> Result:
    p = params or PipelineParams()
    T: Dict[str, float] = {}
    t0 = time.time()
    rgba = load_rgba(image_bytes)
    rgb, alpha = rgba[:, :, :3].copy(), rgba[:, :, 3].copy()
    has_alpha = alpha.min() < 250
    if not has_alpha and is_line_art(rgb):
        return _line_art_result(rgb, p, t0)

    face = MODELS_CACHE.detect_face(rgb)
    T["face"] = time.time() - t0

    t = time.time()
    matte_source = "alpha" if has_alpha else "none"
    if not has_alpha:
        m = MODELS_CACHE.matte(rgb)
        if m is not None:
            alpha, matte_source = m, "rembg"
        elif fallback_cutout:
            cut = load_rgba(fallback_cutout)[:, :, 3]
            alpha = cv2.resize(cut, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
            matte_source = "browser"
    T["matte"] = time.time() - t

    framed, fa, valid, M = frame_portrait(rgb, alpha, face, p)
    S = p.size
    subject = _largest_subject(fa > 127)
    if face is not None:
        # torso: fill each row between the outer body edges (matte holes in busy clothing)
        y_chin = int((p.frame_eye_y + p.frame_face_width * 0.9) * S)
        rows = subject[y_chin:]
        if rows.any():
            has = rows.any(1)
            left = np.argmax(rows, 1)
            right = S - 1 - np.argmax(rows[:, ::-1], 1)
            xx_ = np.arange(S)[None, :]
            span = (xx_ >= left[:, None]) & (xx_ <= right[:, None]) & has[:, None]
            subject[y_chin:] |= span
    fa = np.where(subject, np.maximum(fa, 128), np.minimum(fa, 127)).astype(np.uint8)
    a3 = (fa.astype(np.float32) / 255.0)[..., None]
    on_white = (framed.astype(np.float32) * a3 + 255.0 * (1 - a3)).astype(np.uint8)
    gray = cv2.cvtColor(on_white, cv2.COLOR_RGB2GRAY)

    t = time.time()
    parse = MODELS_CACHE.parse(on_white)
    T["parse"] = time.time() - t
    if parse is not None:
        parse[~subject] = L["background"]
        # stray skin/neck/hair patches on clothing create fake contours: keep the main blob only
        for name in ("skin", "neck", "hair"):
            m = (parse == L[name]).astype(np.uint8)
            n_c, lab_c, st_c, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
            if n_c > 2:
                big = 1 + int(np.argmax(st_c[1:, cv2.CC_STAT_AREA]))
                stray = (lab_c > 0) & (lab_c != big)
                if name == "hair":  # keep sizeable pieces touching the main hair (long hair splits around the neck)
                    near = cv2.dilate((lab_c == big).astype(np.uint8), cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE, (int(S * 0.04) | 1,) * 2)).astype(bool)
                    touching = np.zeros(n_c, bool)
                    touching[np.unique(lab_c[near])] = True
                    keep = (st_c[:, cv2.CC_STAT_AREA] >= S * S * 0.004) & touching
                    stray &= ~keep[lab_c]
                parse[stray] = L["cloth"]
        # glasses / hat labels on clothing (busy garments, helmets) are parser mistakes
        fy, fx = np.nonzero(np.isin(parse, [L[n] for n in ("skin", "nose", "l_eye", "r_eye", "u_lip", "l_lip")]))
        if len(fy):
            chin = fy.max()
            pad = int(0.08 * (fx.max() - fx.min()))
            yy_, xx_ = np.mgrid[0:S, 0:S]
            in_face = (yy_ >= fy.min() - pad) & (yy_ <= chin) & (xx_ >= fx.min() - pad) & (xx_ <= fx.max() + pad)
            parse[(parse == L["eye_g"]) & ~in_face] = L["cloth"]
            parse[(parse == L["hat"]) & (yy_ > chin)] = L["cloth"]
            # the parser's own "background" inside the matte, below the chin, is clothing
            parse[(parse == L["background"]) & subject & (yy_ > chin)] = L["cloth"]

    def region(*names):
        return np.isin(parse, [L[n] for n in names]) if parse is not None else np.zeros((S, S), bool)

    hair = region("hair", "hat")
    skin = region("skin")
    nose = region("nose")
    lips = region("u_lip", "l_lip", "mouth")
    face_skin = region("skin", "nose", "l_brow", "r_brow", "l_eye", "r_eye", "eye_g", "mouth", "u_lip", "l_lip")
    ears = region("l_ear", "r_ear", "ear_r")
    neck = region("neck", "neck_l")
    cloth = region("cloth")
    features = region("l_brow", "r_brow", "l_eye", "r_eye", "eye_g", "nose", "mouth", "u_lip", "l_lip")
    if parse is None:
        head_ell = np.zeros((S, S), np.uint8)
        cv2.ellipse(head_ell, (S // 2, int(S * (p.frame_eye_y + 0.02))),
                    (int(S * p.frame_face_width * 0.95), int(S * p.frame_face_width * 1.35)), 0, 0, 360, 1, -1)
        face_skin = head_ell.astype(bool) & subject
        cloth = subject & ~face_skin

    # ---- line-art network on an edge-preserving smoothed copy ----------- #
    t = time.time()
    smooth_body = _edge_preserving(on_white, 3)
    body_zone = cv2.GaussianBlur((cloth | (subject & ~face_skin & ~hair & ~ears & ~neck)).astype(np.float32), (0, 0), 6)[..., None]
    net_in = (on_white * (1 - body_zone) + smooth_body * body_zone).astype(np.uint8)
    net_size = 768
    small = cv2.resize(net_in, (net_size, net_size), interpolation=cv2.INTER_AREA)
    la = MODELS_CACHE.lineart(small)
    T["lineart"] = time.time() - t
    if la is not None:
        ink_strength = 1.0 - cv2.resize(la, (S, S), interpolation=cv2.INTER_LINEAR)
        # the model draws faint grey lines; normalise so thresholds mean the same on every photo
        ref = float(np.percentile(ink_strength[subject], 99.5)) if subject.any() else 1.0
        ink_strength = np.clip(ink_strength / max(ref, 0.15), 0.0, 1.0)
    else:
        ink_strength = cv2.Canny(cv2.cvtColor(net_in, cv2.COLOR_RGB2GRAY), 60, 140).astype(np.float32) / 255.0

    # ---- compose ------------------------------------------------------- #
    t = time.time()
    lw = p.line_px
    det = float(np.clip(p.detail, 0, 1))
    hs = float(np.clip(p.hair_strands, 0, 1))
    cl = float(np.clip(p.clothing, 0, 1))
    canvas = np.zeros((S, S), np.uint8)
    inner = valid & (cv2.erode(valid.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0)
    edge = max(4, S // 128)
    inner[:edge] = inner[-edge:] = False
    inner[:, :edge] = inner[:, -edge:] = False

    # 1) silhouette
    _draw_contours(canvas, subject, lw, min_len=S * 0.05, smooth_sigma=3.0, clip=inner, external=True)

    beard = np.zeros((S, S), bool)
    if parse is not None:
        def boundary_lines(a, b, sigma=2.5, min_len=S * 0.02):
            band = _boundary_between(parse, [L[x] for x in a], [L[x] for x in b], width=1)
            mask_a = np.isin(parse, [L[x] for x in a])
            layer = np.zeros_like(canvas)
            _draw_contours(layer, mask_a, lw, min_len=min_len, smooth_sigma=sigma)
            layer[~cv2.dilate(band.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)] = 0
            np.maximum(canvas, layer, out=canvas)

        skinlike = ["skin", "nose", "l_brow", "r_brow", "l_eye", "r_eye", "eye_g", "mouth", "u_lip", "l_lip"]
        boundary_lines(["hair", "hat"], skinlike + ["l_ear", "r_ear", "neck"])          # hairline
        boundary_lines(skinlike, ["neck", "neck_l", "cloth"], sigma=3.5)                  # jaw/chin
        boundary_lines(["l_ear", "r_ear"], skinlike + ["hair", "background"], sigma=2.0, min_len=S * 0.03)
        boundary_lines(["cloth"], ["neck", "neck_l", "skin"], sigma=3.0)                  # collar

        # brows and lips as closed outlines; mouth line between the lips
        for name in ("l_brow", "r_brow"):
            _draw_contours(canvas, parse == L[name], lw, min_len=S * 0.03, smooth_sigma=2.0)
        _draw_lips(canvas, parse, lw, S)

        # this parser may give both eyes the same label: split into blobs
        eyes = region("l_eye", "r_eye").astype(np.uint8)
        n_e, lab_e, st_e, _ = cv2.connectedComponentsWithStats(eyes, connectivity=8)
        blobs = [lab_e == i for i in range(1, n_e) if st_e[i, cv2.CC_STAT_AREA] >= 40]
        rels = [r for r in (_iris_rel(b, gray) for b in blobs) if r is not None]
        shared = None
        if len(rels) >= 2:
            # trust the eye whose iris sits nearer the middle (the other is usually shadowed)
            shared = min(rels, key=lambda r: abs(r - 0.5)) if abs(rels[0] - rels[1]) > 0.18 else float(np.mean(rels))
        brows_zone = cv2.dilate(region("l_brow", "r_brow").astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
        for b in blobs:
            _draw_eye(canvas, b, gray, max(2, lw - 1), S, iris_rel=shared, avoid=brows_zone)
        _draw_nose(canvas, nose, ink_strength, lw, S)
        if det >= 0.25:
            _draw_nose_bridge(canvas, nose, lw, S)
        _draw_collar_band(canvas, cloth, neck, lw, S, inner)
        face_cols = np.nonzero(face_skin.any(0))[0]
        if len(face_cols):
            _draw_inner_ears(canvas, ears, 0.5 * (face_cols.min() + face_cols.max()), lw, S)

        beard = _beard_mask(on_white, skin, nose, lips, S, _texture_map(on_white))
        if beard.any():
            _draw_contours(canvas, beard, lw, min_len=S * 0.04, smooth_sigma=3.0)

    # 2) hair strands from the orientation field
    hair_lines = _hair_flow(gray, hair, S, hs) if hair.any() else []
    for ln in hair_lines:
        _poly(canvas, ln, False, lw)

    # 3) fine lines from the network, gated by region
    texture = _texture_map(on_white)
    zone_thr = np.full((S, S), 2.0, np.float32)  # >1 = never
    feat_zone = cv2.dilate(region("nose").astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))).astype(bool)
    zone_thr[face_skin | ears] = 0.94 - 0.20 * det
    zone_thr[neck] = 0.70 - 0.25 * det
    zone_thr[feat_zone] = 0.62 - 0.22 * det
    zone_thr[hair] = 0.92 - 0.12 * hs if len(hair_lines) >= 4 else 0.80 - 0.25 * hs
    cloth_thr = (0.88 - 0.30 * cl) + 0.30 * texture
    zone_thr[cloth] = cloth_thr[cloth]
    zone_thr[beard | lips | nose] = 2.0       # drawn by their own layers
    zone_thr[~subject] = 2.0
    net_ink = (ink_strength > zone_thr) & inner
    kept = np.zeros_like(net_ink)
    hair_min = S * (0.10 - 0.03 * hs) if len(hair_lines) >= 4 else S * (0.06 - 0.03 * hs)
    for zone, ml in ((hair, hair_min),
                     (cloth, S * (0.16 - 0.09 * cl)),
                     (neck, S * 0.05),
                     (face_skin | ears, S * (0.05 - 0.02 * det))):
        part = net_ink & zone
        if part.any():
            kept |= _keep_long_components(part, ml)
    net_layer = cv2.dilate(kept.astype(np.uint8) * 255, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (lw, lw)))
    if parse is not None:
        inside_feat = cv2.dilate(region("u_lip", "l_lip", "mouth").astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
        inside_feat |= cv2.erode(region("l_brow", "r_brow").astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        inside_feat |= cv2.dilate(region("l_eye", "r_eye").astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
        net_layer[inside_feat] = 0
    np.maximum(canvas, net_layer, out=canvas)

    # drop isolated specks (stray dots) except inside the eyes, where pupils are intended
    eye_zone = cv2.dilate(region("l_eye", "r_eye").astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    n_c, lab_c, st_c, _ = cv2.connectedComponentsWithStats((canvas > 110).astype(np.uint8), connectivity=8)
    if n_c > 1:
        small_c = st_c[:, cv2.CC_STAT_AREA] < max(40, lw * lw * 12)
        small_c[0] = False
        in_eye = np.zeros(n_c, bool)
        in_eye[np.unique(lab_c[eye_zone])] = True
        drop = small_c & ~in_eye
        canvas[drop[lab_c]] = 0

    line_image = 255 - canvas
    T["compose"] = time.time() - t

    t = time.time()
    strokes = ink_to_strokes(canvas > 110, p.vector)
    T["vectorize"] = time.time() - t
    T["total"] = time.time() - t0
    info = {
        "face_found": face is not None,
        "parsing": parse is not None,
        "lineart_model": la is not None,
        "matte": matte_source,
        "hair_strands": len(hair_lines),
        "beard": bool(beard.any()),
        "stroke_count": len(strokes),
        "length_px": float(sum(polyline_length(s) for s in strokes)),
    }
    res = Result(strokes, S, line_image, on_white, fa, parse, T, info)
    res.debug = {"ink": ink_strength, "beard": beard, "texture": texture}
    return res

"""Local photo -> line-art -> plotter SVG engine.

Uses:
  * rembg (u2net_human_seg) for subject cut-out
  * rocca/informative-drawings-line-art-onnx for photo -> line drawing
  * skeleton tracing + RDP / Chaikin for plotter-friendly polylines

No API key and no network access is needed once the two model files exist.
"""

from __future__ import annotations

import base64
import io
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

# YuNet prints a DNN-backend warning on every detector creation.  Quiet it
# before OpenCV initialises its logger.
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")

import cv2
import numpy as np
from PIL import Image, ImageOps

Point = Tuple[float, float]

MODEL_REPO = "rocca/informative-drawings-line-art-onnx"
MODEL_URL = f"https://huggingface.co/{MODEL_REPO}/resolve/main/informative-drawings.onnx"

_NB8: Tuple[Tuple[int, int], ...] = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1),
)


# --------------------------------------------------------------------------- #
# Low-level vector helpers
# --------------------------------------------------------------------------- #
def rdp(points: Sequence[Point], eps: float) -> List[Point]:
    """Ramer-Douglas-Peucker polyline simplification (iterative)."""
    if len(points) < 3:
        return list(points)
    pts = np.asarray(points, dtype=np.float64)
    keep = np.zeros(len(pts), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        start, end = stack.pop()
        if end <= start + 1:
            continue
        p0, p1 = pts[start], pts[end]
        seg = p1 - p0
        seg_len = float(np.hypot(seg[0], seg[1]))
        inner = pts[start + 1:end]
        if seg_len < 1e-9:
            dist = np.hypot(inner[:, 0] - p0[0], inner[:, 1] - p0[1])
        else:
            dist = np.abs(
                seg[0] * (p0[1] - inner[:, 1]) - seg[1] * (p0[0] - inner[:, 0])
            ) / seg_len
        idx = int(np.argmax(dist))
        if dist[idx] > eps:
            mid = start + 1 + idx
            keep[mid] = True
            stack.append((start, mid))
            stack.append((mid, end))
    return [tuple(map(float, p)) for p in pts[keep]]


def chaikin(points: Sequence[Point], passes: int = 1) -> List[Point]:
    """Chaikin corner cutting; keeps endpoints, rounds the polyline."""
    out = [tuple(map(float, p)) for p in points]
    for _ in range(max(0, passes)):
        if len(out) < 3:
            break
        new: List[Point] = [out[0]]
        for a, b in zip(out[:-1], out[1:]):
            new.append((0.75 * a[0] + 0.25 * b[0], 0.75 * a[1] + 0.25 * b[1]))
            new.append((0.25 * a[0] + 0.75 * b[0], 0.25 * a[1] + 0.75 * b[1]))
        new.append(out[-1])
        out = new
    return out


def polyline_length(points: Sequence[Point]) -> float:
    return sum(
        math.hypot(b[0] - a[0], b[1] - a[1])
        for a, b in zip(points[:-1], points[1:])
    )


def _neighbours(y: int, x: int, h: int, w: int) -> Iterable[Tuple[int, int]]:
    for dy, dx in _NB8:
        yy, xx = y + dy, x + dx
        if 0 <= yy < h and 0 <= xx < w:
            yield yy, xx


def trace_skeleton(skel: np.ndarray) -> List[List[Point]]:
    """Trace a 1-pixel skeleton, walking straight through junctions.

    Edges are visited once.  At a junction the straightest unvisited
    continuation is chosen, which turns X crossings into two long strokes
    instead of four short ones.  Returns pixel-space polylines (x, y).
    """
    h, w = skel.shape
    ys, xs = np.nonzero(skel)
    if len(ys) == 0:
        return []

    pixel_set = {(int(y), int(x)) for y, x in zip(ys, xs)}
    degree = {}
    for y, x in pixel_set:
        degree[(y, x)] = sum(
            1 for ny, nx in _neighbours(y, x, h, w) if (ny, nx) in pixel_set
        )

    visited: set[frozenset[Tuple[int, int]]] = set()
    paths: List[List[Tuple[int, int]]] = []

    def edge(a, b):
        return frozenset((a, b))

    def walk(start, first):
        path = [start, first]
        visited.add(edge(start, first))
        prev, cur = start, first
        while True:
            vy, vx = cur[0] - prev[0], cur[1] - prev[1]
            vlen = math.hypot(vy, vx) or 1.0
            options = []
            for nb in _neighbours(cur[0], cur[1], h, w):
                if nb not in pixel_set or nb == prev:
                    continue
                if edge(cur, nb) in visited:
                    continue
                wy, wx = nb[0] - cur[0], nb[1] - cur[1]
                wlen = math.hypot(wy, wx) or 1.0
                dot = (vy * wy + vx * wx) / (vlen * wlen)
                options.append((dot, nb))
            if not options:
                break
            options.sort(key=lambda item: item[0], reverse=True)
            dot, nxt = options[0]
            if dot < -0.35 and degree.get(cur, 0) <= 2:
                break
            visited.add(edge(cur, nxt))
            path.append(nxt)
            prev, cur = cur, nxt
        return path

    order = sorted(
        pixel_set,
        key=lambda p: (0 if degree[p] == 1 else (1 if degree[p] > 2 else 2), p[0], p[1]),
    )
    for start in order:
        if degree[start] == 2:
            continue
        for nb in _neighbours(start[0], start[1], h, w):
            if nb in pixel_set and degree[nb] == 2 and edge(start, nb) not in visited:
                paths.append(walk(start, nb))
        # also direct node-to-node edges (tiny fragments at junctions)
        for nb in _neighbours(start[0], start[1], h, w):
            if nb in pixel_set and edge(start, nb) not in visited:
                paths.append(walk(start, nb))

    # Remaining loops have no node with degree != 2.
    for start in order:
        for nb in _neighbours(start[0], start[1], h, w):
            if nb in pixel_set and edge(start, nb) not in visited:
                paths.append(walk(start, nb))

    return [[(float(x), float(y)) for y, x in p] for p in paths]


def _tangent(points: Sequence[Point], at_end: bool, n: int = 6) -> Tuple[float, float]:
    if len(points) < 2:
        return (0.0, 0.0)
    if at_end:
        a, b = points[max(0, len(points) - n)], points[-1]
    else:
        a, b = points[0], points[min(len(points) - 1, n - 1)]
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dy) or 1.0
    return dx / length, dy / length


def stitch_strokes(
    strokes: List[List[Point]],
    max_gap: float = 5.0,
    max_angle_cos: float = 0.15,
    rounds: int = 3,
) -> List[List[Point]]:
    """Join stroke ends that are close and point in a compatible direction."""
    items = [list(s) for s in strokes if len(s) >= 2]
    for _ in range(rounds):
        used = [False] * len(items)
        merged: List[List[Point]] = []
        changed = False
        for i in range(len(items)):
            if used[i]:
                continue
            cur = items[i]
            used[i] = True
            while True:
                if len(cur) < 2:
                    break
                tail = cur[-1]
                tail_dir = _tangent(cur, True)
                best = None
                best_score = None
                for j in range(len(items)):
                    if used[j] or not items[j]:
                        continue
                    cand = items[j]
                    for reverse in (False, True):
                        seq = cand[::-1] if reverse else cand
                        head = seq[0]
                        gap = math.hypot(head[0] - tail[0], head[1] - tail[1])
                        if gap > max_gap:
                            continue
                        head_dir = _tangent(seq, False)
                        # We want tail_dir ~ head_dir (continuing forward).
                        align = tail_dir[0] * head_dir[0] + tail_dir[1] * head_dir[1]
                        if align < max_angle_cos:
                            continue
                        score = gap - 2.0 * align
                        if best_score is None or score < best_score:
                            best_score = score
                            best = (j, seq)
                if best is None:
                    break
                j, seq = best
                used[j] = True
                cur = cur + seq
                changed = True
            merged.append(cur)
        items = merged
        if not changed:
            break
    return items


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
@dataclass
class StrokeParams:
    size: int = 768
    threshold: int = 148          # clothing / body line threshold
    texture_threshold: int = 112  # knit / high-frequency cloth texture
    head_threshold: int = 210     # face, hair and beard threshold
    line_gamma: float = 1.5       # deepen faint model lines before thresholding
    min_length: float = 18.0      # body strokes shorter than this are texture
    head_min_length: float = 4.0  # keep fine facial detail
    max_strokes: int = 360
    head_max_fraction: float = 0.7  # cap on strokes reserved for face/hair
    rdp_eps: float = 1.0
    smoothing: int = 2
    crop_margin: float = 0.04
    denoise: int = 1


class LineArtEngine:
    def __init__(
        self,
        model_path: str | Path,
        bg_session=None,
        providers: Optional[Sequence[str]] = None,
        face_model_path: Optional[str | Path] = None,
    ):
        self.model_path = Path(model_path)
        self.bg_session = bg_session
        self.providers = list(providers or ["CPUExecutionProvider"])
        if face_model_path is None:
            candidate = self.model_path.parent / "face_detection_yunet_2023mar.onnx"
            face_model_path = candidate if candidate.exists() else None
        self.face_model_path = Path(face_model_path) if face_model_path else None
        self._session = None
        self._face_detector = None
        self._face_size: Optional[Tuple[int, int]] = None
        try:  # OpenCV 5 is noisy about YuNet's DNN backend.
            cv2.setLogLevel(0)
        except Exception:
            pass

    # -- face detection ------------------------------------------------------ #
    def _detect_face(self, img: Image.Image):
        if not self.face_model_path or not self.face_model_path.exists():
            return None
        try:
            bgr = np.asarray(img.convert("RGB"))[:, :, ::-1]
            h, w = bgr.shape[:2]
            # Reuse the detector: recreating YuNet per call costs ~0.3 s and
            # prints the backend warning four times per generation.
            if self._face_detector is None or self._face_size != (w, h):
                self._face_detector = cv2.FaceDetectorYN_create(
                    str(self.face_model_path), "", (w, h), 0.75, 0.3, 5000
                )
                self._face_size = (w, h)
            _, faces = self._face_detector.detect(bgr)
            if faces is None or len(faces) == 0:
                return None
            return max(faces, key=lambda f: float(f[2]) * float(f[3]))
        except Exception:
            return None

    def _portrait_crop(self, img: Image.Image) -> Image.Image:
        """Crop to head + upper chest so the face fills the plot area."""
        w, h = img.size
        face = self._detect_face(img)
        if face is not None:
            fx, fy, fw, fh = (float(v) for v in face[:4])
            side = int(min(fw * 3.1, w, h))
            top = int(fy - 0.42 * fh)
            cx = int(fx + fw * 0.5)
        else:
            side = int(min(w, h) * 0.90)
            top = 0
            cx = w // 2

        side = max(64, side)
        top = max(0, min(top, h - side))
        left = cx - side // 2
        canvas = Image.new("RGB", (side, side), (255, 255, 255))
        src = img.crop(
            (
                max(0, left),
                top,
                min(w, left + side),
                min(h, top + side),
            )
        )
        canvas.paste(src, (max(0, -left), 0))
        return canvas

    # -- models ------------------------------------------------------------- #
    @property
    def session(self):
        if self._session is None:
            import onnxruntime as ort

            self._session = ort.InferenceSession(
                str(self.model_path), providers=self.providers
            )
        return self._session

    def _remove_background(self, img: Image.Image) -> Image.Image:
        """Return RGB image of the subject on white, cropped to the subject."""
        rgba = img.convert("RGBA")
        cut = None
        if self.bg_session is not None:
            try:
                from rembg import remove

                cut = remove(rgba, session=self.bg_session, post_process_mask=True)
            except Exception:
                cut = None
        if cut is None:
            cut = rgba

        white = Image.new("RGBA", cut.size, (255, 255, 255, 255))
        white.alpha_composite(cut)
        rgb = white.convert("RGB")

        alpha = np.asarray(cut.getchannel("A"))
        if cut is not rgba and alpha.max() > 0:
            ys, xs = np.where(alpha > 16)
            if len(xs):
                x0, x1 = int(xs.min()), int(xs.max())
                y0, y1 = int(ys.min()), int(ys.max())
                mx = int((x1 - x0) * 0.06) + 8
                my = int((y1 - y0) * 0.06) + 8
                rgb = rgb.crop(
                    (
                        max(0, x0 - mx),
                        max(0, y0 - my),
                        min(rgb.width, x1 + mx),
                        min(rgb.height, y1 + my),
                    )
                )
        return self._portrait_crop(rgb)

    @staticmethod
    def _square_pad(img: Image.Image, margin: float, size: int) -> Image.Image:
        w, h = img.size
        side = int(max(w, h) * (1.0 + 2.0 * margin))
        side = max(64, side)
        canvas = Image.new("RGB", (side, side), (255, 255, 255))
        canvas.paste(img, ((side - w) // 2, (side - h) // 2))
        return canvas.resize((size, size), Image.Resampling.LANCZOS)

    @staticmethod
    def _fade_bottom(img: Image.Image, fraction: float = 0.075) -> Image.Image:
        """Soften the crop edge into white so the model doesn't draw a border."""
        arr = np.asarray(img).astype(np.float32)
        h = arr.shape[0]
        start = int(h * (1.0 - fraction))
        if start >= h - 1:
            return img
        ramp = np.linspace(1.0, 0.0, h - start, dtype=np.float32)[:, None, None]
        arr[start:] = arr[start:] * ramp + 255.0 * (1.0 - ramp)
        return Image.fromarray(arr.astype(np.uint8))

    def _lineart(self, img: Image.Image, params: StrokeParams) -> np.ndarray:
        arr = np.asarray(img)
        if params.denoise > 0:
            # Keep hair/eyes sharp, but flatten knitwear and skin pores in the
            # body.  The model faithfully draws whatever texture it is given,
            # so removing cloth texture here is what keeps the SVG clean.
            head = cv2.bilateralFilter(arr, d=7, sigmaColor=28, sigmaSpace=28)
            body = cv2.bilateralFilter(arr, d=25, sigmaColor=90, sigmaSpace=90)
            body = cv2.pyrMeanShiftFiltering(body, sp=14, sr=32, maxLevel=1)
            body = cv2.medianBlur(body, 7)

            mask = self._head_mask(img).astype(np.float32)
            mask = cv2.GaussianBlur(mask, (0, 0), 15.0)[..., None]
            arr = (head.astype(np.float32) * mask +
                   body.astype(np.float32) * (1.0 - mask)).astype(np.uint8)

        feed = arr.astype(np.float32) / 255.0
        feed = feed.transpose(2, 0, 1)[None, ...]
        out = self.session.run(None, {"input": feed})[0]
        line = np.clip(out[0, 0], 0.0, 1.0)
        # The model renders anti-aliased strokes in mid-greys.  Without a
        # contrast boost the binariser breaks them into short dashes; a mild
        # gamma keeps them continuous while leaving the white paper white.
        if params.line_gamma != 1.0:
            line = np.power(line, params.line_gamma)
        return line * 255.0

    @staticmethod
    def _binarize(line: np.ndarray, params: StrokeParams) -> np.ndarray:
        bw = (line < params.threshold).astype(np.uint8)
        if bw.sum() == 0:
            return bw.astype(bool)
        # Remove isolated specks (knit texture, JPEG noise).
        n, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
        clean = np.zeros_like(bw)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= 6:
                clean[labels == i] = 1
        return clean.astype(bool)

    def _head_mask(self, img: Image.Image) -> np.ndarray:
        """Rounded head shape that includes hair and beard, not the shoulders.

        The detailed threshold must not cover the chest: knitwear lives there.
        A superellipse hugs the head far better than a wide rectangle.
        """
        w, h = img.size
        yy, xx = np.mgrid[0:h, 0:w]

        def ellipse(cx, cy, rx, ry, power=2.4):
            nx = (xx - cx) / max(rx, 1.0)
            ny = (yy - cy) / max(ry, 1.0)
            return (np.abs(nx) ** power + np.abs(ny) ** power) <= 1.0

        face = self._detect_face(img)
        if face is not None:
            fx, fy, fw, fh = (float(v) for v in face[:4])
            # Face-detector box covers forehead -> chin; extend up for hair.
            return ellipse(
                fx + fw * 0.50,
                fy + fh * 0.27,
                fw * 0.98,
                fh * 0.72,
                power=2.4,
            )

        return ellipse(w * 0.50, h * 0.28, w * 0.26, h * 0.30, power=2.4)

    @staticmethod
    def _mask_border(
        line: np.ndarray, x_fraction: float = 0.050, y_fraction: float = 0.070
    ) -> np.ndarray:
        """Blank the crop edges so the model's fake frame is never plotted."""
        work = line.copy()
        h, w = work.shape
        bx = max(2, int(w * x_fraction))
        by = max(2, int(h * y_fraction))
        work[:, :bx] = 255.0
        work[:, w - bx:] = 255.0
        work[h - by:, :] = 255.0
        return work

    @staticmethod
    def _texture_map(img: Image.Image) -> np.ndarray:
        """Local high-frequency energy, 0 = smooth, 1 = knit/fabric texture."""
        gray = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY).astype(np.float32)
        highpass = gray - cv2.GaussianBlur(gray, (0, 0), 2.2)
        energy = cv2.GaussianBlur(np.abs(highpass), (0, 0), 9.0)
        peak = float(np.percentile(energy, 97)) or 1.0
        return np.clip(energy / peak, 0.0, 1.0)

    def binarize_detail(
        self, line: np.ndarray, framed: Image.Image, params: StrokeParams
    ) -> np.ndarray:
        """Face-aware, texture-aware binarisation.

        * Face / hair / beard: generous threshold keeps fine detail.
        * Smooth cloth: normal threshold keeps folds and silhouettes.
        * Knitted / noisy cloth: only very dark ink survives, which removes
          the hatch noise without deleting the shirt outline.
        """
        # The model traces the hard crop boundary as if it were a contour.
        # Blank a thin border so it never becomes fake plot lines.
        work = self._mask_border(line)

        head = self._head_mask(framed)
        texture = self._texture_map(framed)

        # Blend body threshold down in high-frequency areas.
        blend = np.clip((0.34 - texture) / 0.22, 0.0, 1.0)
        body_thr = params.texture_threshold + (
            params.threshold - params.texture_threshold
        ) * blend
        body = work < body_thr

        detail = work < params.head_threshold
        bw = body.astype(np.uint8)
        bw[head] = detail[head].astype(np.uint8)

        n, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
        clean = np.zeros_like(bw)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= 5:
                clean[labels == i] = 1
        return clean.astype(bool)

    def vectorize(self, line: np.ndarray, params: StrokeParams,
                  framed: Optional[Image.Image] = None) -> List[List[Point]]:
        head = None
        if framed is not None:
            bw = self.binarize_detail(line, framed, params)
            head = self._head_mask(framed)
        else:
            bw = self._binarize(line, params)
        return self.vectorize_binary(bw, params, head)

    def vectorize_binary(
        self,
        bw: np.ndarray,
        params: StrokeParams,
        head: Optional[np.ndarray] = None,
    ) -> List[List[Point]]:
        from skimage.morphology import skeletonize

        skel = skeletonize(bw)
        paths = trace_skeleton(skel)

        head_strokes: List[List[Point]] = []
        body_strokes: List[List[Point]] = []
        for path in paths:
            if len(path) < 2:
                continue
            length = polyline_length(path)
            cx = sum(p[0] for p in path) / len(path)
            cy = sum(p[1] for p in path) / len(path)
            in_head = bool(
                head is not None
                and 0 <= int(cy) < head.shape[0]
                and 0 <= int(cx) < head.shape[1]
                and head[int(cy), int(cx)]
            )
            needed = params.head_min_length if in_head else params.min_length
            if length < needed:
                continue
            simp = rdp(path, params.rdp_eps)
            if len(simp) < 2:
                continue
            simp = chaikin(simp, params.smoothing)
            simp = rdp(simp, max(0.3, params.rdp_eps * 0.55))
            if len(simp) < 2 or polyline_length(simp) < needed * 0.8:
                continue
            (head_strokes if in_head else body_strokes).append(simp)

        head_strokes.sort(key=polyline_length, reverse=True)
        body_strokes.sort(key=polyline_length, reverse=True)

        head_cap = int(params.max_strokes * params.head_max_fraction)
        strokes = head_strokes[:head_cap]
        strokes += body_strokes[: max(0, params.max_strokes - len(strokes))]

        if len(strokes) > 1:
            strokes = stitch_strokes(strokes, max_gap=params.size * 0.008)
            strokes.sort(key=polyline_length, reverse=True)
            strokes = strokes[: params.max_strokes]
        return strokes

    # -- public API --------------------------------------------------------- #
    def generate(
        self, image_bytes: bytes, params: Optional[StrokeParams] = None
    ) -> dict:
        params = params or StrokeParams()
        t0 = time.time()

        src = Image.open(io.BytesIO(image_bytes))
        src = ImageOps.exif_transpose(src).convert("RGB")
        subject = self._remove_background(src)
        framed = self._square_pad(subject, params.crop_margin, params.size)
        framed = self._fade_bottom(framed)

        line = self._lineart(framed, params)
        strokes = self.vectorize(line, params, framed)

        # Normalised coordinates used by the web app: x right, y down,
        # clipped to the plottable [-0.84, 0.84] square.
        norm_strokes = []
        for stroke in strokes:
            norm_strokes.append(
                [
                    [
                        round((x / params.size * 2.0 - 1.0) * 0.84, 5),
                        round((y / params.size * 2.0 - 1.0) * 0.84, 5),
                    ]
                    for x, y in stroke
                ]
            )

        # PNG preview of the AI line-art (black on white) for the UI.
        # Use the same border mask as the vectoriser so the preview does not
        # show a crop-edge line that will never be plotted.
        line_img = Image.fromarray(
            np.clip(self._mask_border(line), 0, 255).astype(np.uint8), mode="L"
        ).convert("RGB")
        buf = io.BytesIO()
        line_img.save(buf, format="PNG", optimize=True)
        preview_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        return {
            "ok": True,
            "engine": MODEL_REPO,
            "strokes": norm_strokes,
            "stroke_count": len(norm_strokes),
            "point_count": sum(len(s) for s in norm_strokes),
            "size": params.size,
            "preview_png": "data:image/png;base64," + preview_b64,
            "seconds": round(time.time() - t0, 2),
            "params": {
                "threshold": params.threshold,
                "texture_threshold": params.texture_threshold,
                "head_threshold": params.head_threshold,
                "line_gamma": params.line_gamma,
                "min_length": params.min_length,
                "max_strokes": params.max_strokes,
                "rdp_eps": params.rdp_eps,
            },
        }

    def render_preview(
        self, strokes: Sequence[Sequence[Point]], size: int, width: int = 3
    ) -> Image.Image:
        canvas = Image.new("RGB", (size, size), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        for stroke in strokes:
            if len(stroke) >= 2:
                draw.line([(x, y) for x, y in stroke], fill=(0, 0, 0), width=width, joint="curve")
        return canvas


# Drawing import is only needed by render_preview; keep it local-friendly.
from PIL import ImageDraw  # noqa: E402  (after dataclass definitions)

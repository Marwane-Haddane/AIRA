"""Raster line art -> smooth centerline strokes for a pen plotter.

The input is any black-on-white line image (an AI sketch, a coloring page,
a scanned drawing).  The output is a list of polylines that follow the
*center* of each inked line, so the pen draws every line once instead of
tracing both of its edges as potrace-style outline tracers do.

Steps:
  1. binarize + remove specks + close pin-hole gaps
  2. thin to a 1-pixel skeleton
  3. turn the skeleton into a graph (junction clusters + edges)
  4. prune short spurs that thinning grows on thick strokes
  5. stitch edges through junctions when their directions continue
  6. bridge small gaps between nearly touching stroke ends
  7. smooth (Gaussian on resampled arc length), simplify (RDP), filter
  8. order strokes to reduce pen-up travel
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.ndimage import convolve, gaussian_filter1d
from skimage.morphology import skeletonize

Point = Tuple[float, float]
Stroke = List[Point]

_NB8 = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


@dataclass
class VectorizeParams:
    threshold: Optional[int] = None    # None = Otsu
    min_area: int = 12                 # ink specks smaller than this are dropped (px at work size)
    close_px: int = 1                  # morphological closing radius before thinning
    spur_px: float = 10.0              # remove dangling branches shorter than this
    join_angle_deg: float = 50.0       # max bend allowed when stitching through a junction
    gap_px: float = 6.0                # bridge stroke ends closer than this
    gap_angle_deg: float = 60.0
    smooth_sigma: float = 1.6          # px, Gaussian smoothing along the stroke
    resample_px: float = 1.0
    rdp_eps: float = 0.45              # px
    min_length_px: float = 10.0        # strokes shorter than this are dropped
    keep_dots: bool = True             # tiny closed blobs (moles, pupils) become small loops
    max_strokes: int = 0               # 0 = unlimited; otherwise keep the longest N
    hollow_fills: bool = True          # draw solid black areas as outlines


@dataclass
class _Edge:
    a: int
    b: int
    pts: List[Tuple[int, int]]  # (y, x) pixels, from node a to node b
    alive: bool = True

    @property
    def length(self) -> float:
        p = np.asarray(self.pts, dtype=np.float64)
        if len(p) < 2:
            return 0.0
        return float(np.hypot(*np.diff(p, axis=0).T).sum())


# --------------------------------------------------------------------------- #
# Raster preparation
# --------------------------------------------------------------------------- #
def binarize(gray: np.ndarray, params: VectorizeParams) -> np.ndarray:
    """Dark lines -> True."""
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
    g = cv2.GaussianBlur(gray, (3, 3), 0)
    if params.threshold is None:
        thr, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr = min(max(thr, 90), 210)
    else:
        thr = params.threshold
    ink = (g < thr).astype(np.uint8)
    if params.close_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * params.close_px + 1,) * 2)
        ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, k)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= params.min_area
    return keep[labels]


def hollow_fills(ink: np.ndarray, factor: float = 2.4, min_extra_px: float = 2.0) -> np.ndarray:
    """Turn solid ink areas (pupils, dark patches) into outline rings.

    Thinning a filled disc produces a star of spurs; a pen should draw its
    outline instead.  Pixels deeper than `factor` x the typical line half
    width are treated as fill and carved out, leaving a ring one line wide.
    """
    if not ink.any():
        return ink
    dt = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 3)
    skel = skeletonize(ink)
    half = float(np.median(dt[skel])) if skel.any() else 1.0
    deep_thr = max(half * factor, half + min_extra_px)
    deep = dt > deep_thr
    if not deep.any():
        return ink
    # grow the deep core back to (almost) the blob edge, keep a ring of ~line width
    ring = max(1, int(round(half * 2)))
    core = cv2.dilate(deep.astype(np.uint8), cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * int(deep_thr - ring) + 1,) * 2)) if deep_thr - ring >= 1 else deep.astype(np.uint8)
    return ink & ~core.astype(bool)


def fill_small_holes(ink: np.ndarray, max_hole: int) -> np.ndarray:
    """Fill enclosed white holes smaller than max_hole px (pupils, thick-line pinholes)."""
    inv = (~ink).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    small = np.zeros(n, dtype=bool)
    small[1:] = stats[1:, cv2.CC_STAT_AREA] < max_hole
    # never fill the region touching the border (the paper)
    border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    small[border] = False
    return ink | small[labels]


# --------------------------------------------------------------------------- #
# Skeleton graph
# --------------------------------------------------------------------------- #
def _skeleton_graph(skel: np.ndarray):
    h, w = skel.shape
    sk = skel.astype(np.uint8)
    k = np.ones((3, 3), dtype=np.uint8)
    k[1, 1] = 0
    nb = convolve(sk, k, mode="constant") * sk
    node_mask = (sk > 0) & (nb != 2)
    n_nodes, node_lab = cv2.connectedComponents(node_mask.astype(np.uint8), connectivity=8)
    node_pos = np.zeros((n_nodes, 2), dtype=np.float64)
    if n_nodes > 1:
        ys, xs = np.nonzero(node_mask)
        labs = node_lab[ys, xs]
        cnt = np.bincount(labs, minlength=n_nodes).astype(np.float64)
        cnt[cnt == 0] = 1
        node_pos[:, 0] = np.bincount(labs, weights=ys, minlength=n_nodes) / cnt
        node_pos[:, 1] = np.bincount(labs, weights=xs, minlength=n_nodes) / cnt

    visited = np.zeros_like(sk, dtype=bool)
    edges: List[_Edge] = []

    def neighbours(y, x):
        for dy, dx in _NB8:
            yy, xx = y + dy, x + dx
            if 0 <= yy < h and 0 <= xx < w and sk[yy, xx]:
                yield yy, xx

    def walk(start_node: int, start_px, first_px):
        path = [start_px, first_px]
        if node_lab[first_px] > 0:  # two node clusters touching directly
            return path, int(node_lab[first_px])
        visited[first_px] = True
        prev, cur = start_px, first_px
        while True:
            nxt = None
            hit_node = None
            for q in neighbours(*cur):
                if q == prev or q in path[-3:]:
                    continue
                lab = node_lab[q]
                if lab > 0:
                    if not (lab == start_node and len(path) < 3):
                        hit_node = q
                    continue
                if not visited[q]:
                    if nxt is None:
                        nxt = q
                    else:
                        # prefer 4-connected continuation (staircase artifacts)
                        if abs(q[0] - cur[0]) + abs(q[1] - cur[1]) == 1:
                            nxt = q
            if hit_node is not None and nxt is None:
                path.append(hit_node)
                return path, int(node_lab[hit_node])
            if nxt is None:
                if hit_node is not None:
                    path.append(hit_node)
                    return path, int(node_lab[hit_node])
                return path, -1  # dead end without node (should be rare)
            visited[nxt] = True
            path.append(nxt)
            prev, cur = cur, nxt

    # edges starting at node clusters
    ys, xs = np.nonzero(node_mask)
    seen_pairs = set()
    for y, x in zip(ys.tolist(), xs.tolist()):
        a = int(node_lab[y, x])
        for q in neighbours(y, x):
            lab = node_lab[q]
            if lab == a:
                continue
            if lab > 0:
                key = (min((y, x), q), max((y, x), q))
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                edges.append(_Edge(a, int(lab), [(y, x), q]))
                continue
            if visited[q]:
                continue
            path, b = walk(a, (y, x), q)
            if b < 0:
                # create an end node at the dead end
                node_pos = np.vstack([node_pos, np.asarray(path[-1], dtype=np.float64)[None]])
                b = len(node_pos) - 1
            edges.append(_Edge(a, b, path))

    # isolated loops (no node pixels at all)
    rem = (sk > 0) & ~visited & ~node_mask
    if rem.any():
        n_loop, loop_lab = cv2.connectedComponents(rem.astype(np.uint8), connectivity=8)
        for i in range(1, n_loop):
            yy, xx = np.nonzero(loop_lab == i)
            start = (int(yy[0]), int(xx[0]))
            if visited[start]:
                continue
            visited[start] = True
            path = [start]
            prev, cur = None, start
            while True:
                nxt = None
                for q in neighbours(*cur):
                    if q != prev and not visited[q]:
                        nxt = q
                        break
                if nxt is None:
                    break
                visited[nxt] = True
                path.append(nxt)
                prev, cur = cur, nxt
            path.append(start)
            node_pos = np.vstack([node_pos, np.asarray(start, dtype=np.float64)[None]])
            nid = len(node_pos) - 1
            edges.append(_Edge(nid, nid, path))
    return node_pos, edges


def _degrees(n_nodes: int, edges: Sequence[_Edge]) -> np.ndarray:
    deg = np.zeros(n_nodes, dtype=np.int32)
    for e in edges:
        if e.alive:
            deg[e.a] += 1
            deg[e.b] += 1
    return deg


def _prune_spurs(node_pos, edges: List[_Edge], spur_px: float, rounds: int = 2) -> None:
    for _ in range(rounds):
        deg = _degrees(len(node_pos), edges)
        changed = False
        for e in edges:
            if not e.alive or e.a == e.b:
                continue
            da, db = deg[e.a], deg[e.b]
            if (da == 1 and db >= 3) or (db == 1 and da >= 3):
                if e.length < spur_px:
                    e.alive = False
                    deg[e.a] -= 1
                    deg[e.b] -= 1
                    changed = True
        if not changed:
            break


def _end_direction(pts: np.ndarray, from_start: bool, span: int = 10) -> np.ndarray:
    """Unit vector pointing *into* the edge from the given end."""
    if from_start:
        seg = pts[: min(len(pts), span + 1)]
    else:
        seg = pts[::-1][: min(len(pts), span + 1)]
    v = seg[-1] - seg[0]
    n = float(np.hypot(*v))
    return v / n if n > 1e-9 else np.zeros(2)


def _stitch(node_pos, edges: List[_Edge], join_angle_deg: float) -> List[np.ndarray]:
    """Combine edges through junctions where the line continues straight."""
    live = [e for e in edges if e.alive]
    arrs = [np.asarray(e.pts, dtype=np.float64) for e in live]
    # incident ends per node: (edge index, is_start)
    incident: Dict[int, List[Tuple[int, bool]]] = {}
    for i, e in enumerate(live):
        incident.setdefault(e.a, []).append((i, True))
        incident.setdefault(e.b, []).append((i, False))

    cos_lim = math.cos(math.radians(180.0 - join_angle_deg))  # dirs must be near-opposite
    partner: Dict[Tuple[int, bool], Tuple[int, bool]] = {}
    for node, ends in incident.items():
        if len(ends) < 2:
            continue
        dirs = [_end_direction(arrs[i], s) for i, s in ends]
        cands = []
        for p in range(len(ends)):
            for q in range(p + 1, len(ends)):
                if ends[p][0] == ends[q][0] and live[ends[p][0]].a == live[ends[p][0]].b:
                    continue  # both ends of a self loop
                c = float(np.dot(dirs[p], dirs[q]))
                if c <= cos_lim:
                    # weight by straightness and by length (long lines win)
                    lp = live[ends[p][0]].length
                    lq = live[ends[q][0]].length
                    cands.append((c - 0.002 * min(lp, lq, 60.0), p, q))
        cands.sort()
        used = set()
        for _, p, q in cands:
            if p in used or q in used:
                continue
            used.add(p)
            used.add(q)
            partner[ends[p]] = ends[q]
            partner[ends[q]] = ends[p]

    done = [False] * len(live)
    strokes: List[np.ndarray] = []

    def oriented(i: int, enter_at_start: bool) -> np.ndarray:
        return arrs[i] if enter_at_start else arrs[i][::-1]

    def chain_from(i: int, enter_at_start: bool) -> np.ndarray:
        pieces = []
        cur, at_start = i, enter_at_start
        while True:
            done[cur] = True
            pieces.append(oriented(cur, at_start))
            exit_end = (cur, not at_start)
            nxt = partner.get(exit_end)
            if nxt is None or done[nxt[0]]:
                break
            cur, at_start = nxt[0], nxt[1]
        out = pieces[0]
        for p in pieces[1:]:
            out = np.vstack([out, p[1:]])
        return out

    # start from free ends first so open chains come out whole
    for i in range(len(live)):
        if done[i]:
            continue
        if (i, True) not in partner:
            strokes.append(chain_from(i, True))
        elif (i, False) not in partner:
            strokes.append(chain_from(i, False))
    for i in range(len(live)):  # remaining = closed cycles
        if not done[i]:
            strokes.append(chain_from(i, True))
    return strokes


def _bridge_gaps(strokes: List[np.ndarray], gap_px: float, gap_angle_deg: float) -> List[np.ndarray]:
    if gap_px <= 0 or len(strokes) < 2:
        return strokes
    cos_lim = math.cos(math.radians(gap_angle_deg))
    strokes = [s for s in strokes if len(s) >= 2]
    changed = True
    while changed:
        changed = False
        ends = []
        for i, s in enumerate(strokes):
            if np.hypot(*(s[0] - s[-1])) < 1.5:
                continue  # closed
            ends.append((i, True, s[0], -_end_direction(s, True)))   # outward dir
            ends.append((i, False, s[-1], -_end_direction(s, False)))
        if len(ends) < 2:
            break
        P = np.asarray([e[2] for e in ends])
        best = None
        for p in range(len(ends)):
            d = np.hypot(*(P - P[p]).T)
            for q in np.nonzero((d < gap_px) & (d > 0))[0]:
                q = int(q)
                if q <= p or ends[p][0] == ends[q][0]:
                    continue
                gap_vec = P[q] - P[p]
                gl = float(np.hypot(*gap_vec)) or 1.0
                g = gap_vec / gl
                # outward direction of p should point to q, and of q back to p
                if np.dot(ends[p][3], g) < cos_lim or np.dot(ends[q][3], -g) < cos_lim:
                    continue
                score = d[q]
                if best is None or score < best[0]:
                    best = (score, p, q)
        if best is not None:
            _, p, q = best
            ip, sp_, _, _ = ends[p]
            iq, sq, _, _ = ends[q]
            a = strokes[ip] if not sp_ else strokes[ip][::-1]   # a ends at the joined end
            b = strokes[iq] if sq else strokes[iq][::-1]         # b starts at the joined end
            merged = np.vstack([a, b])
            strokes = [s for k, s in enumerate(strokes) if k not in (ip, iq)] + [merged]
            changed = True
    return strokes


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def polyline_length(p: np.ndarray) -> float:
    if len(p) < 2:
        return 0.0
    return float(np.hypot(*np.diff(p, axis=0).T).sum())


def resample(p: np.ndarray, step: float) -> np.ndarray:
    if len(p) < 2:
        return p
    seg = np.hypot(*np.diff(p, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    if total < step:
        return p[[0, -1]]
    n = max(2, int(round(total / step)) + 1)
    t = np.linspace(0.0, total, n)
    return np.stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])], axis=1)


def smooth(p: np.ndarray, sigma: float, closed: bool) -> np.ndarray:
    if sigma <= 0 or len(p) < 5:
        return p
    mode = "wrap" if closed else "nearest"
    if closed:
        q = p[:-1]
        out = np.stack([gaussian_filter1d(q[:, 0], sigma, mode=mode),
                        gaussian_filter1d(q[:, 1], sigma, mode=mode)], axis=1)
        return np.vstack([out, out[:1]])
    out = np.stack([gaussian_filter1d(p[:, 0], sigma, mode=mode),
                    gaussian_filter1d(p[:, 1], sigma, mode=mode)], axis=1)
    out[0], out[-1] = p[0], p[-1]
    return out


def rdp(p: np.ndarray, eps: float) -> np.ndarray:
    if len(p) < 3 or eps <= 0:
        return p
    keep = np.zeros(len(p), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(p) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = p[i], p[j]
        ab = b - a
        L = float(np.hypot(*ab))
        inner = p[i + 1:j]
        if L < 1e-9:
            d = np.hypot(*(inner - a).T)
        else:
            d = np.abs(ab[0] * (a[1] - inner[:, 1]) - ab[1] * (a[0] - inner[:, 0])) / L
        k = int(np.argmax(d))
        if d[k] > eps:
            m = i + 1 + k
            keep[m] = True
            stack.append((i, m))
            stack.append((m, j))
    return p[keep]


def order_strokes(strokes: List[np.ndarray]) -> List[np.ndarray]:
    """Greedy nearest-neighbour ordering with stroke reversal."""
    if len(strokes) < 2:
        return strokes
    remaining = list(range(len(strokes)))
    starts = np.asarray([s[0] for s in strokes])
    ends = np.asarray([s[-1] for s in strokes])
    out = []
    pos = np.array([0.0, 0.0])
    alive = np.ones(len(strokes), dtype=bool)
    for _ in range(len(strokes)):
        ds = np.hypot(*(starts - pos).T)
        de = np.hypot(*(ends - pos).T)
        ds[~alive] = np.inf
        de[~alive] = np.inf
        i_s, i_e = int(np.argmin(ds)), int(np.argmin(de))
        if ds[i_s] <= de[i_e]:
            s = strokes[i_s]
            alive[i_s] = False
        else:
            s = strokes[i_e][::-1]
            alive[i_e] = False
        out.append(s)
        pos = s[-1]
    return out


# --------------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------------- #
def _dot_strokes(ink: np.ndarray, max_diam: float, min_area: int) -> Tuple[List[np.ndarray], np.ndarray]:
    """Small compact blobs (moles, pupils drawn as dots) -> tiny circles."""
    n, labels, stats, cents = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    dots: List[np.ndarray] = []
    remove = np.zeros(n, dtype=bool)
    for i in range(1, n):
        w, h, area = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT], stats[i, cv2.CC_STAT_AREA]
        if area < min_area or max(w, h) > max_diam:
            continue
        fill_ratio = area / float(w * h)
        if fill_ratio < 0.55 or max(w, h) > 2.2 * min(w, h):
            continue
        r = max(1.2, 0.3 * (w + h) / 2.0)
        t = np.linspace(0, 2 * np.pi, 13)
        cx, cy = cents[i]
        dots.append(np.stack([cx + r * np.cos(t), cy + r * np.sin(t)], axis=1))
        remove[i] = True
    return dots, ink & ~remove[labels]


def ink_to_strokes(ink: np.ndarray, params: Optional[VectorizeParams] = None) -> List[np.ndarray]:
    """Binary ink mask (True = line) -> list of (N, 2) arrays in (x, y) pixel coords."""
    params = params or VectorizeParams()
    ink = fill_small_holes(ink, max_hole=6)
    dots: List[np.ndarray] = []
    if params.keep_dots:
        dots, ink = _dot_strokes(ink, max_diam=params.min_length_px * 1.6, min_area=params.min_area)
    if params.hollow_fills:
        ink = hollow_fills(ink)
    skel = skeletonize(ink)
    node_pos, edges = _skeleton_graph(skel)
    _prune_spurs(node_pos, edges, params.spur_px)
    raw = _stitch(node_pos, edges, params.join_angle_deg)
    raw = _bridge_gaps(raw, params.gap_px, params.gap_angle_deg)

    out: List[np.ndarray] = []
    for s in raw:
        xy = s[:, ::-1].astype(np.float64)  # (y, x) -> (x, y)
        closed = len(xy) > 3 and np.hypot(*(xy[0] - xy[-1])) < 2.0
        L = polyline_length(xy)
        if L < params.min_length_px:
            if not (params.keep_dots and closed and L >= 4):
                continue
        xy = resample(xy, params.resample_px)
        xy = smooth(xy, params.smooth_sigma, closed)
        xy = rdp(xy, params.rdp_eps)
        if len(xy) >= 2:
            out.append(xy)
    if params.max_strokes and len(out) > params.max_strokes:
        out.sort(key=polyline_length, reverse=True)
        out = out[: params.max_strokes]
    return order_strokes(out + dots)


def image_to_strokes(gray: np.ndarray, params: Optional[VectorizeParams] = None) -> List[np.ndarray]:
    params = params or VectorizeParams()
    return ink_to_strokes(binarize(gray, params), params)


def strokes_to_svg(strokes: Sequence[np.ndarray], width: float, height: float,
                   stroke_width: float = 2.0, width_mm: Optional[float] = None) -> str:
    if width_mm:
        hmm = width_mm * height / width
        size_attr = f'width="{width_mm:.2f}mm" height="{hmm:.2f}mm"'
    else:
        size_attr = f'width="{width:.0f}" height="{height:.0f}"'
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" {size_attr} viewBox="0 0 {width:.0f} {height:.0f}">',
        f'<g fill="none" stroke="#000" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round">',
    ]
    for s in strokes:
        if len(s) < 2:
            continue
        d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in s)
        parts.append(f'<path d="{d}"/>')
    parts.append("</g></svg>")
    return "\n".join(parts)


def render_strokes(strokes: Sequence[np.ndarray], width: int, height: int, thickness: int = 2) -> np.ndarray:
    canvas = np.full((height, width), 255, dtype=np.uint8)
    for s in strokes:
        if len(s) >= 2:
            pts = np.round(s * 4).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [pts], False, 0, thickness, cv2.LINE_AA, shift=2)
    return canvas

#!/usr/bin/env python3
"""Convert a portrait photo into a clean, bold-line SVG.

Pipeline:
  1. Remove the background locally with rembg (BirefNet portrait model).
  2. Use OpenCV to clean the foreground mask and isolate the person.
  3. Detect/simplify the subject silhouette and internal image edges.
  4. Export a scalable SVG plus PNG previews.

This script does NOT send robot movement commands. The SVG is an artwork file;
calibration and Dobot workspace mapping should be handled in a separate stage.
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)

# Higher thresholds and stronger simplification produce fewer lines.
DETAIL_PRESETS = {
    "low": {
        "canny_low": 85,
        "canny_high": 175,
        "blur": 7,
        "min_path_length": 30.0,
        "simplify_px": 2.4,
        "max_paths": 170,
    },
    "medium": {
        "canny_low": 60,
        "canny_high": 135,
        "blur": 5,
        "min_path_length": 18.0,
        "simplify_px": 1.6,
        "max_paths": 300,
    },
    "high": {
        "canny_low": 40,
        "canny_high": 100,
        "blur": 3,
        "min_path_length": 10.0,
        "simplify_px": 0.9,
        "max_paths": 550,
    },
}


def remove_background(input_path: Path, model_name: str) -> Image.Image:
    """Run local foreground segmentation using rembg; returns RGBA PIL image."""
    try:
        from rembg import new_session, remove
    except ImportError as exc:
        raise RuntimeError(
            "The 'rembg' package is not installed. Run:\n"
            '  python -m pip install "rembg[cpu]"'
        ) from exc

    # Correct phone-photo orientation before segmentation.
    with Image.open(input_path) as opened:
        source = ImageOps.exif_transpose(opened).convert("RGB")

    print(f"Loading local background-removal model: {model_name}")
    print("The model weights may download the first time you run this script.")
    session = new_session(model_name)
    cutout = remove(source, session=session)
    if not isinstance(cutout, Image.Image):
        cutout = Image.fromarray(np.asarray(cutout))
    return cutout.convert("RGBA")


def keep_largest_subject(alpha: np.ndarray) -> np.ndarray:
    """Convert the alpha channel into a clean mask for the main subject."""
    # Keep partially-transparent hair/edges, then close tiny gaps in the mask.
    mask = np.where(alpha > 28, 255, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        raise RuntimeError(
            "Background removal did not find a foreground subject. "
            "Try another photo or a different rembg model."
        )

    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    largest = np.where(labels == largest_label, 255, 0).astype(np.uint8)

    # Fill small holes in the person silhouette while retaining the outer contour.
    contours, _ = cv2.findContours(
        largest, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    filled = np.zeros_like(largest)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled


def crop_to_subject(
    rgba: np.ndarray, mask: np.ndarray, padding_ratio: float = 0.025
) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        raise RuntimeError("The foreground mask is empty.")

    height, width = mask.shape
    pad = max(8, int(max(width, height) * padding_ratio))
    x1 = max(0, int(xs.min()) - pad)
    x2 = min(width, int(xs.max()) + pad + 1)
    y1 = max(0, int(ys.min()) - pad)
    y2 = min(height, int(ys.max()) + pad + 1)
    return rgba[y1:y2, x1:x2].copy(), mask[y1:y2, x1:x2].copy()


def resize_pair(
    rgba: np.ndarray, mask: np.ndarray, max_side: int
) -> tuple[np.ndarray, np.ndarray]:
    h, w = mask.shape
    scale = min(1.0, float(max_side) / float(max(h, w)))
    if scale >= 1.0:
        return rgba, mask

    new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
    rgba_small = cv2.resize(rgba, new_size, interpolation=cv2.INTER_AREA)
    mask_small = cv2.resize(mask, new_size, interpolation=cv2.INTER_NEAREST)
    return rgba_small, mask_small


def contour_to_xy(contour: np.ndarray, epsilon: float, closed: bool) -> np.ndarray:
    approx = cv2.approxPolyDP(contour, epsilon, closed)
    return approx.reshape(-1, 2)


def skeletonize(binary: np.ndarray) -> np.ndarray:
    """Thin binary strokes to approximately one-pixel centerlines."""
    image = np.where(binary > 0, 255, 0).astype(np.uint8)
    skeleton = np.zeros_like(image)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    # Canny lines are usually already thin, so this loop normally ends quickly.
    while cv2.countNonZero(image) > 0:
        eroded = cv2.erode(image, element)
        opened = cv2.dilate(eroded, element)
        residue = cv2.subtract(image, opened)
        skeleton = cv2.bitwise_or(skeleton, residue)
        image = eroded
    return skeleton


def _edge_key(a: tuple[int, int], b: tuple[int, int]):
    return (a, b) if a <= b else (b, a)


def trace_skeleton_paths(skeleton: np.ndarray) -> list[np.ndarray]:
    """Trace an 8-connected one-pixel skeleton into ordered (x, y) paths."""
    ys, xs = np.where(skeleton > 0)
    pixels = {(int(y), int(x)) for y, x in zip(ys, xs)}
    if not pixels:
        return []

    offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]
    neighbours: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for y, x in pixels:
        near = []
        for dy, dx in offsets:
            candidate = (y + dy, x + dx)
            if candidate in pixels:
                near.append(candidate)
        neighbours[(y, x)] = near

    visited: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    paths: list[np.ndarray] = []

    def trace(start: tuple[int, int], first: tuple[int, int]):
        path = [start]
        previous, current = start, first
        visited.add(_edge_key(previous, current))
        path.append(current)

        # Continue through ordinary pixels; stop at endpoints or junctions.
        while len(neighbours[current]) == 2:
            candidates = [
                p for p in neighbours[current]
                if p != previous and _edge_key(current, p) not in visited
            ]
            if not candidates:
                break
            nxt = candidates[0]
            visited.add(_edge_key(current, nxt))
            previous, current = current, nxt
            path.append(current)

        arr = np.array([[x, y] for y, x in path], dtype=np.int32)
        if len(arr) >= 2:
            paths.append(arr)

    # Start paths at endpoints and junctions.
    for point in pixels:
        if len(neighbours[point]) != 2:
            for neighbour in neighbours[point]:
                if _edge_key(point, neighbour) not in visited:
                    trace(point, neighbour)

    # Any edges left belong to closed loops.
    for point in pixels:
        for neighbour in neighbours[point]:
            if _edge_key(point, neighbour) not in visited:
                trace(point, neighbour)

    return paths


def polyline_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    deltas = np.diff(points.astype(np.float32), axis=0)
    return float(np.sqrt((deltas * deltas).sum(axis=1)).sum())


def make_line_paths(
    rgba: np.ndarray,
    mask: np.ndarray,
    detail: str,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], np.ndarray]:
    """Build silhouette + sparse internal line paths using OpenCV."""
    preset = DETAIL_PRESETS[detail]
    bgr = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Bilateral filtering suppresses photo texture without destroying main edges.
    filtered = cv2.bilateralFilter(gray, d=7, sigmaColor=45, sigmaSpace=45)
    blur_size = int(preset["blur"])
    if blur_size > 1:
        if blur_size % 2 == 0:
            blur_size += 1
        filtered = cv2.GaussianBlur(filtered, (blur_size, blur_size), 0)

    # Keep edge details inside the person; eroding avoids a duplicate outer edge.
    inner_mask = cv2.erode(
        mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    edges = cv2.Canny(
        filtered,
        threshold1=int(preset["canny_low"]),
        threshold2=int(preset["canny_high"]),
    )
    edges = cv2.bitwise_and(edges, inner_mask)
    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )

    thin_edges = skeletonize(edges)
    raw_paths = trace_skeleton_paths(thin_edges)

    internal_paths: list[tuple[float, np.ndarray]] = []
    min_len = float(preset["min_path_length"])
    epsilon = float(preset["simplify_px"])
    for path in raw_paths:
        length = polyline_length(path)
        if length < min_len:
            continue
        approx = cv2.approxPolyDP(
            path.reshape(-1, 1, 2), epsilon=epsilon, closed=False
        ).reshape(-1, 2)
        if len(approx) >= 2:
            internal_paths.append((length, approx))

    # Keep the most visually important long strokes for a less cluttered result.
    internal_paths.sort(key=lambda item: item[0], reverse=True)
    internal_paths = internal_paths[: int(preset["max_paths"])]

    # Extract and simplify only the main outer contour of the person.
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        raise RuntimeError("Could not trace the person's outer silhouette.")
    outer = max(contours, key=cv2.contourArea)
    silhouette = contour_to_xy(outer, epsilon=max(1.5, epsilon), closed=True)

    final_internal = [p for _, p in internal_paths]
    return silhouette, edges, final_internal, thin_edges


def path_data(points: np.ndarray, close: bool = False) -> str:
    if len(points) < 2:
        return ""
    commands = [f"M {int(points[0, 0])} {int(points[0, 1])}"]
    commands.extend(f"L {int(x)} {int(y)}" for x, y in points[1:])
    if close:
        commands.append("Z")
    return " ".join(commands)


def save_svg(
    output_path: Path,
    width_px: int,
    height_px: int,
    silhouette: np.ndarray,
    internal_paths: list[np.ndarray],
    max_size_mm: float,
    stroke_mm: float,
) -> tuple[float, float]:
    longest_side = max(width_px, height_px)
    scale = max_size_mm / float(longest_side)
    width_mm = width_px * scale
    height_mm = height_px * scale

    root = ET.Element(
        f"{{{SVG_NS}}}svg",
        {
            "version": "1.1",
            "width": f"{width_mm:.2f}mm",
            "height": f"{height_mm:.2f}mm",
            "viewBox": f"0 0 {width_px} {height_px}",
        },
    )
    title = ET.SubElement(root, f"{{{SVG_NS}}}title")
    title.text = "Simplified portrait line drawing"
    desc = ET.SubElement(root, f"{{{SVG_NS}}}desc")
    desc.text = (
        "Foreground-only portrait outline with simplified internal strokes. "
        "Review and scale this artwork before sending it to a robot."
    )
    group = ET.SubElement(
        root,
        f"{{{SVG_NS}}}g",
        {
            "fill": "none",
            "stroke": "#000000",
            "stroke-width": f"{stroke_mm:.2f}mm",
            "stroke-linecap": "round",
            "stroke-linejoin": "round",
        },
    )

    outline = ET.SubElement(group, f"{{{SVG_NS}}}path")
    outline.set("d", path_data(silhouette, close=True))
    outline.set("id", "person-outline")

    for index, points in enumerate(internal_paths, start=1):
        element = ET.SubElement(group, f"{{{SVG_NS}}}path")
        element.set("d", path_data(points, close=False))
        element.set("id", f"detail-{index:03d}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(root)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return width_mm, height_mm


def save_preview(
    output_path: Path,
    shape: tuple[int, int],
    silhouette: np.ndarray,
    internal_paths: list[np.ndarray],
    max_size_mm: float,
    stroke_mm: float,
):
    height, width = shape
    scale = max_size_mm / float(max(width, height))
    pixels_per_mm = max(width, height) / max_size_mm
    thickness = max(2, int(round(stroke_mm * pixels_per_mm)))
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)

    cv2.polylines(
        canvas,
        [silhouette.reshape(-1, 1, 2)],
        isClosed=True,
        color=(0, 0, 0),
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )
    for points in internal_paths:
        cv2.polylines(
            canvas,
            [points.reshape(-1, 1, 2)],
            isClosed=False,
            color=(0, 0, 0),
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a simplified, bold-line SVG from a portrait photo."
    )
    parser.add_argument("--input", required=True, help="Input portrait image path")
    parser.add_argument(
        "--output", default="output/portrait.svg", help="Output SVG path"
    )
    parser.add_argument(
        "--model",
        default="birefnet-portrait",
        help="rembg model name (default: birefnet-portrait)",
    )
    parser.add_argument(
        "--detail",
        choices=DETAIL_PRESETS.keys(),
        default="low",
        help="low = clean/simple, medium = more facial detail, high = more edges",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=1100,
        help="Maximum pixel dimension for processing (default: 1100)",
    )
    parser.add_argument(
        "--max-size-mm",
        type=float,
        default=120.0,
        help="Scale longest side of SVG to this physical size (default: 120mm)",
    )
    parser.add_argument(
        "--stroke-mm",
        type=float,
        default=0.70,
        help="Visual SVG pen-line thickness (default: 0.70mm)",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not input_path.exists():
        print(f"ERROR: Input file does not exist: {input_path}", file=sys.stderr)
        return 2
    if args.max_side < 200:
        print("ERROR: --max-side should be at least 200 pixels.", file=sys.stderr)
        return 2
    if args.max_size_mm <= 0 or args.stroke_mm <= 0:
        print("ERROR: SVG size and stroke width must be greater than zero.", file=sys.stderr)
        return 2

    try:
        cutout = remove_background(input_path, args.model)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        stem = output_path.stem
        cutout_path = output_path.with_name(f"{stem}_cutout.png")
        preview_path = output_path.with_name(f"{stem}_preview.png")
        edges_path = output_path.with_name(f"{stem}_edges.png")
        cutout.save(cutout_path)

        rgba = np.array(cutout, dtype=np.uint8)
        mask = keep_largest_subject(rgba[:, :, 3])
        rgba, mask = crop_to_subject(rgba, mask)
        rgba, mask = resize_pair(rgba, mask, args.max_side)

        silhouette, edges, internal_paths, thin_edges = make_line_paths(
            rgba, mask, args.detail
        )
        height, width = mask.shape
        width_mm, height_mm = save_svg(
            output_path,
            width,
            height,
            silhouette,
            internal_paths,
            args.max_size_mm,
            args.stroke_mm,
        )
        save_preview(
            preview_path,
            (height, width),
            silhouette,
            internal_paths,
            args.max_size_mm,
            args.stroke_mm,
        )
        cv2.imwrite(str(edges_path), thin_edges)

        print("\nSUCCESS — portrait SVG created")
        print(f"SVG:       {output_path}")
        print(f"Preview:   {preview_path}")
        print(f"Cutout:    {cutout_path}")
        print(f"Edge map:  {edges_path}")
        print(f"SVG size:  {width_mm:.1f} mm × {height_mm:.1f} mm")
        print(f"Detail:    {args.detail}")
        print(f"Paths:     1 silhouette + {len(internal_paths)} internal strokes")
        print("\nTip: Open the PNG preview first. If it is too busy, rerun with --detail low.")
        print("Robot note: This SVG is not robot-calibrated. Confirm the Dobot drawing area,")
        print("paper origin, scale, and pen-up/pen-down heights before any physical run.")
        return 0

    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        print(
            "Check the install instructions and try a clear, well-lit portrait photo. ",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

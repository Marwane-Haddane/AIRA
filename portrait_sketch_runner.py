"""Run the local U²-Net portrait sketch model on one image."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from ai_edge_litert.interpreter import Interpreter


def main() -> int:
    if len(sys.argv) != 4:
        print("Usage: portrait_sketch_runner.py MODEL INPUT_IMAGE OUTPUT_IMAGE", file=sys.stderr)
        return 2

    model_path, input_path, output_path = map(Path, sys.argv[1:])
    with Image.open(input_path) as opened:
        rgba = opened.convert("RGBA")
        white = Image.new("RGBA", rgba.size, "white")
        white.alpha_composite(rgba)
        photo = ImageOps.contain(white.convert("RGB"), (512, 512), Image.Resampling.LANCZOS)
        image = Image.new("RGB", (512, 512), "white")
        image.paste(photo, ((512 - photo.width) // 2, (512 - photo.height) // 2))

    pixels = np.asarray(image, dtype=np.float32) / 255.0
    peak = float(pixels.max())
    if peak > 0:
        pixels /= peak
    mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
    std = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
    tensor = ((pixels - mean) / std).transpose(2, 0, 1)[None, ...].astype(np.float32)

    interpreter = Interpreter(model_path=str(model_path), num_threads=4)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    interpreter.set_tensor(input_detail["index"], tensor)
    interpreter.invoke()
    sketch = interpreter.get_tensor(interpreter.get_output_details()[0]["index"])[0, 0]

    low = float(sketch.min())
    high = float(sketch.max())
    if high - low < 1e-8:
        raise RuntimeError("Portrait Sketch returned a blank image")
    dark_strokes = 1.0 - (sketch - low) / (high - low)
    result = Image.fromarray(np.clip(dark_strokes * 255.0, 0, 255).astype(np.uint8), mode="L").convert("RGB")
    result.save(output_path, format="PNG")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

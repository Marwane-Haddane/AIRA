"""Run the original ArtLine FastAI-v1 portrait checkpoint on one image.

Use a separate legacy Python environment. See ARTLINE_SETUP.md.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch
from fastai.vision import defaults, image2np, load_learner, open_image
from PIL import Image
import numpy as np

class FeatureLoss:
    """Training-only object retained in some exported FastAI checkpoints."""
    pass


def main() -> int:
    if len(sys.argv) != 4:
        print("Usage: artline_runner.py CHECKPOINT INPUT_IMAGE OUTPUT_IMAGE", file=sys.stderr)
        return 2
    checkpoint, source, target = map(Path, sys.argv[1:])
    defaults.device = torch.device("cpu")
    learner = load_learner(checkpoint.parent, checkpoint.name)
    with tempfile.TemporaryDirectory(prefix="robogenius-artline-input-") as folder:
        rgb_path = Path(folder) / "portrait-rgb.png"
        with Image.open(source) as opened:
            photo = opened.convert("RGBA")
            white = Image.new("RGBA", photo.size, "white")
            white.alpha_composite(photo)
            white.convert("RGB").save(rgb_path)
        _, result, _ = learner.predict(open_image(rgb_path))
    pixels = np.asarray(np.clip(image2np(result), 0, 1) * 255, dtype=np.uint8)
    if pixels.ndim == 2:
        output = Image.fromarray(pixels, mode="L").convert("RGB")
    else:
        output = Image.fromarray(pixels[..., :3], mode="RGB")
    output.save(target, format="PNG")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

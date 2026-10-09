#!/usr/bin/env python3
"""Semantic segmentation demo (RudraEdge RE-T1, Day 10).

Assigns a class label to every pixel of an image using a free pretrained
torchvision DeepLabV3-MobileNetV3-Large model (20 Pascal VOC classes +
background) and saves:

  * mask.png    - colorized per-pixel class mask (same size as the input)
  * overlay.png - the mask alpha-blended over the original image

Usage:
    python semantic_segmentation.py --input samples/test.jpg --output-dir outputs

Exit codes:
    0  success
    2  input problem (missing / unreadable / unsupported / corrupt image)
    3  required dependency (torch / torchvision) not installed
    4  model weights could not be loaded (e.g. no network and no cache)
    5  output problem (cannot create / write output files)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

MODEL_NAME = "deeplabv3_mobilenet_v3_large"
WEIGHTS_NAME = "COCO_WITH_VOC_LABELS_V1"

# Pascal VOC label set used by the torchvision weights above.
VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
    "tvmonitor",
]

EXIT_OK, EXIT_INPUT, EXIT_DEPS, EXIT_WEIGHTS, EXIT_OUTPUT = 0, 2, 3, 4, 5


class InputError(Exception):
    """Raised for missing / unreadable / unsupported input images."""


class DependencyError(Exception):
    """Raised when torch / torchvision are not importable."""


class WeightsError(Exception):
    """Raised when pretrained weights cannot be loaded."""


def voc_palette(n: int = 256) -> np.ndarray:
    """Standard Pascal VOC colormap, shape (n, 3), dtype uint8."""
    palette = np.zeros((n, 3), dtype=np.uint8)
    for i in range(n):
        c, r, g, b = i, 0, 0, 0
        for j in range(8):
            r |= ((c >> 0) & 1) << (7 - j)
            g |= ((c >> 1) & 1) << (7 - j)
            b |= ((c >> 2) & 1) << (7 - j)
            c >>= 3
        palette[i] = (r, g, b)
    return palette


def load_image(path: str | Path) -> Image.Image:
    """Validate and open an input image as RGB. Raises InputError."""
    p = Path(path)
    if not p.exists():
        raise InputError(f"input file does not exist: {p}")
    if not p.is_file():
        raise InputError(f"input path is not a file: {p}")
    try:
        with Image.open(p) as im:
            im.load()  # force decode so truncated/corrupt files fail here
            return im.convert("RGB")
    except UnidentifiedImageError:
        raise InputError(f"unsupported or corrupt image (cannot identify): {p}")
    except PermissionError:
        raise InputError(f"no permission to read: {p}")
    except (OSError, ValueError, SyntaxError) as exc:  # truncated data etc.
        raise InputError(f"could not read image {p}: {exc}")


def colorize_mask(class_map: np.ndarray) -> Image.Image:
    """Map an (H, W) integer class map to an RGB image using the VOC palette."""
    if class_map.ndim != 2:
        raise ValueError(f"class_map must be 2-D (H, W), got shape {class_map.shape}")
    palette = voc_palette()
    return Image.fromarray(palette[class_map.astype(np.uint8)], mode="RGB")


def blend_overlay(image: Image.Image, mask: Image.Image, alpha: float = 0.5) -> Image.Image:
    """Alpha-blend mask over image. Both must have identical size."""
    if image.size != mask.size:
        raise ValueError(f"size mismatch: image {image.size} vs mask {mask.size}")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    return Image.blend(image.convert("RGB"), mask.convert("RGB"), alpha)


def save_outputs(image: Image.Image, class_map: np.ndarray, output_dir: str | Path,
                 alpha: float = 0.5) -> tuple[Path, Path]:
    """Write mask.png and overlay.png into output_dir (created if needed)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    mask = colorize_mask(class_map)
    overlay = blend_overlay(image, mask, alpha)
    mask_path, overlay_path = out / "mask.png", out / "overlay.png"
    mask.save(mask_path)
    overlay.save(overlay_path)
    return mask_path, overlay_path


def load_model():
    """Load the pretrained model lazily so I/O code works without torch."""
    try:
        import torch  # noqa: F401
        from torchvision.models.segmentation import (
            DeepLabV3_MobileNet_V3_Large_Weights,
            deeplabv3_mobilenet_v3_large,
        )
    except ImportError as exc:
        raise DependencyError(
            f"torch/torchvision not available ({exc}). "
            "Install with: pip install torch torchvision"
        )
    weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
    try:
        model = deeplabv3_mobilenet_v3_large(weights=weights)
    except Exception as exc:  # network / cache / checksum problems
        raise WeightsError(
            f"could not load pretrained weights ({type(exc).__name__}: {exc}). "
            "They are downloaded once to the torch hub cache "
            "(~/.cache/torch/hub/checkpoints); copy the .pth file there to run offline."
        )
    model.eval()
    return model, weights


def predict(image: Image.Image, model, weights) -> np.ndarray:
    """Run inference and return an (H, W) uint8 class map at the source size."""
    import torch
    import torch.nn.functional as F

    preprocess = weights.transforms()
    batch = preprocess(image).unsqueeze(0)
    with torch.inference_mode():
        logits = model(batch)["out"]
        # Resize logits back to the original image size so outputs align 1:1.
        logits = F.interpolate(logits, size=(image.height, image.width),
                               mode="bilinear", align_corners=False)
    return logits.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)


def classes_present(class_map: np.ndarray) -> list[str]:
    ids = np.unique(class_map)
    return [VOC_CLASSES[i] if i < len(VOC_CLASSES) else f"class_{i}" for i in ids]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Semantic segmentation demo (DeepLabV3-MobileNetV3).")
    ap.add_argument("--input", required=True, help="path to input image")
    ap.add_argument("--output-dir", default="outputs", help="folder for mask.png / overlay.png")
    ap.add_argument("--alpha", type=float, default=0.5, help="overlay mask opacity 0..1 (default 0.5)")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0.0 <= args.alpha <= 1.0:
        print("error: --alpha must be between 0 and 1", file=sys.stderr)
        return EXIT_INPUT
    try:
        image = load_image(args.input)
    except InputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INPUT
    try:
        model, weights = load_model()
    except DependencyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_DEPS
    except WeightsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_WEIGHTS

    start = time.perf_counter()
    class_map = predict(image, model, weights)
    elapsed = time.perf_counter() - start  # includes preprocessing; single cold run

    try:
        mask_path, overlay_path = save_outputs(image, class_map, args.output_dir, args.alpha)
    except OSError as exc:
        print(f"error: cannot write outputs to {args.output_dir}: {exc}", file=sys.stderr)
        return EXIT_OUTPUT

    print(f"input        : {args.input} ({image.width}x{image.height})")
    print(f"model        : {MODEL_NAME} ({WEIGHTS_NAME})")
    print(f"mask         : {mask_path}")
    print(f"overlay      : {overlay_path}")
    print(f"classes found: {', '.join(classes_present(class_map))}")
    print(f"inference    : {elapsed:.2f} s (single cold run, includes preprocessing)")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())

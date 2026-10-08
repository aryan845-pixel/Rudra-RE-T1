"""High-level Preprocessor: raw image -> model-ready tensor."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np

from . import ops
from .config import PipelineConfig

log = logging.getLogger("preprocessing")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class Result:
    tensor: np.ndarray
    stages: Dict[str, np.ndarray] = field(default_factory=dict)  # intermediate 8-bit images
    timings_ms: Dict[str, float] = field(default_factory=dict)


class Preprocessor:
    """Resize -> (BGR->RGB) -> spatial filter -> float32 -> normalize -> layout."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.cfg = config or PipelineConfig()

    @staticmethod
    def load(path: Union[str, Path]) -> np.ndarray:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not decode image (corrupt/unsupported): {path}")
        return img

    def __call__(self, img_bgr: np.ndarray, keep_stages: bool = False) -> Result:
        self._validate(img_bgr)
        c, stages, timings = self.cfg, {}, {}

        def step(name, fn, *args):
            t0 = time.perf_counter()
            out = fn(*args)
            timings[name] = (time.perf_counter() - t0) * 1000
            if keep_stages and out.dtype == np.uint8:
                stages[name] = out
            return out

        x = step("resize", ops.resize, img_bgr, c.width, c.height,
                 c.resize_mode, c.interpolation, c.pad_value)
        if c.to_rgb:
            x = step("rgb", ops.bgr_to_rgb, x)
        x = step("filter", ops.spatial_filter, x, c.filter, c.ksize, c.sigma,
                 c.bilateral_d, c.bilateral_sigma_color, c.bilateral_sigma_space)
        x = step("normalize", ops.normalize, x, c.mean, c.std)
        x = step("layout", ops.to_layout, x, c.channels_first, c.add_batch_dim)
        return Result(tensor=x, stages=stages, timings_ms=timings)

    def process_file(self, path, keep_stages: bool = False) -> Result:
        return self(self.load(path), keep_stages=keep_stages)

    def process_dir(self, folder: Union[str, Path]) -> List[Result]:
        files = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTS)
        results = []
        for p in files:
            try:
                results.append(self.process_file(p))
            except Exception as e:  # keep the batch alive on a bad file
                log.error("Skipping %s: %s", p.name, e)
        return results

    @staticmethod
    def _validate(img: np.ndarray) -> None:
        if not isinstance(img, np.ndarray) or img.dtype != np.uint8:
            raise TypeError("Expected a uint8 numpy image")
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 image, got shape {img.shape}")

    def to_displayable(self, tensor: np.ndarray) -> np.ndarray:
        """Invert layout/normalization back to an 8-bit BGR image for saving."""
        t = tensor[0] if self.cfg.add_batch_dim else tensor
        if self.cfg.channels_first:
            t = np.transpose(t, (1, 2, 0))
        if self.cfg.mean is not None:
            t = t * np.asarray(self.cfg.std, np.float32) + np.asarray(self.cfg.mean, np.float32)
        u8 = (t * 255.0).round().clip(0, 255).astype(np.uint8)
        return cv2.cvtColor(u8, cv2.COLOR_RGB2BGR) if self.cfg.to_rgb else u8

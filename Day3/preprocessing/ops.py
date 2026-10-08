"""Pure, individually testable image operations (no I/O)."""
from __future__ import annotations

from typing import Optional, Tuple
import cv2
import numpy as np

_INTERP = {
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "area": cv2.INTER_AREA,
    "lanczos": cv2.INTER_LANCZOS4,
}


def _pick_interp(src_hw: Tuple[int, int], dst_hw: Tuple[int, int], name: Optional[str]) -> int:
    if name:
        return _INTERP[name]
    # INTER_AREA is best for shrinking (anti-aliasing), LINEAR for enlarging.
    shrinking = dst_hw[0] * dst_hw[1] < src_hw[0] * src_hw[1]
    return cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR


def resize(img: np.ndarray, width: int, height: int, mode: str = "stretch",
           interpolation: Optional[str] = None, pad_value: int = 114) -> np.ndarray:
    h, w = img.shape[:2]
    interp = _pick_interp((h, w), (height, width), interpolation)
    if mode == "stretch":
        return cv2.resize(img, (width, height), interpolation=interp)

    # letterbox: preserve aspect ratio, centre-pad the remainder
    scale = min(width / w, height / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    scaled = cv2.resize(img, (nw, nh), interpolation=interp)
    canvas = np.full((height, width, *img.shape[2:]), pad_value, dtype=img.dtype)
    top, left = (height - nh) // 2, (width - nw) // 2
    canvas[top:top + nh, left:left + nw] = scaled
    return canvas


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def spatial_filter(img: np.ndarray, kind: str, ksize: int = 5, sigma: float = 0.0,
                   d: int = 9, sigma_color: float = 75.0, sigma_space: float = 75.0) -> np.ndarray:
    if kind == "none":
        return img
    if kind == "gaussian":
        return cv2.GaussianBlur(img, (ksize, ksize), sigma)
    if kind == "median":
        return cv2.medianBlur(img, ksize)
    if kind == "bilateral":
        return cv2.bilateralFilter(img, d, sigma_color, sigma_space)
    raise ValueError(f"Unknown filter: {kind}")


def normalize(img_u8: np.ndarray, mean=None, std=None) -> np.ndarray:
    """uint8 [0,255] -> float32 [0,1], optionally standardised per channel."""
    out = img_u8.astype(np.float32) / 255.0
    if mean is not None:
        out = (out - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return out


def to_layout(t: np.ndarray, channels_first: bool, add_batch_dim: bool) -> np.ndarray:
    if channels_first:
        t = np.transpose(t, (2, 0, 1))
    if add_batch_dim:
        t = t[np.newaxis, ...]
    return np.ascontiguousarray(t)

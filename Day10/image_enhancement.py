"""Day 10 - Image Denoising & Enhancement (classical OpenCV/NumPy only).

Pipeline:
    Input -> Validation -> Denoising -> Contrast Stretch / Low-Light -> Comparison -> Metrics -> Outputs

NOTE: Everything here is classical image processing (filters, percentile
stretching, gamma, CLAHE). No AI model is trained or used.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

DENOISE_METHODS = ("none", "gaussian", "median", "bilateral")
ENHANCE_MODES = ("none", "stretch", "lowlight")

DEFAULT_DENOISE_PARAMS = {
    "gaussian": {"ksize": 5, "sigma": 0.0},
    "median": {"ksize": 5},
    "bilateral": {"d": 9, "sigma_color": 50.0, "sigma_space": 50.0},
}
DEFAULT_ENHANCE_PARAMS = {
    "stretch": {"low_percentile": 1.0, "high_percentile": 99.0},
    "lowlight": {
        "low_percentile": 1.0, "high_percentile": 99.0,
        "gamma": 0.8, "use_clahe": True, "clip_limit": 1.5, "tile_grid": 8,
    },
}


# ----------------------------------------------------------------- validation
def validate_image(img) -> np.ndarray:
    """Return img if it is a non-empty uint8 grayscale (H,W) or BGR (H,W,3) array."""
    if not isinstance(img, np.ndarray):
        raise TypeError("Image must be a numpy.ndarray")
    if img.size == 0:
        raise ValueError("Image is empty")
    if img.dtype != np.uint8:
        raise ValueError(f"Image dtype must be uint8, got {img.dtype}")
    if img.ndim == 2:
        return img
    if img.ndim == 3 and img.shape[2] == 3:
        return img
    raise ValueError(f"Unsupported image shape {img.shape}; expected (H,W) or (H,W,3)")


def load_image(path) -> np.ndarray:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Image file not found: {p}")
    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Image file is unreadable or not a valid image: {p}")
    if img.ndim == 3 and img.shape[2] == 4:  # drop alpha
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if img.dtype == np.uint16:
        img = (img / 257.0).round().astype(np.uint8)
    return validate_image(img)


def _check_odd(name, value, minimum):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if value < minimum or value % 2 == 0:
        raise ValueError(f"{name} must be an odd integer >= {minimum}, got {value}")


def _check_positive(name, value, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be a number, got {value!r}")
    if value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{name} must be {'>= 0' if allow_zero else '> 0'}, got {value}")


# ------------------------------------------------------------------ denoising
def denoise(img, method="gaussian", **params) -> np.ndarray:
    """Denoise with 'gaussian', 'median', 'bilateral' or 'none'."""
    validate_image(img)
    if method not in DENOISE_METHODS:
        raise ValueError(f"Unknown denoise method {method!r}; choose from {DENOISE_METHODS}")
    if method == "none":
        return img.copy()
    p = {**DEFAULT_DENOISE_PARAMS[method], **params}
    extra = set(params) - set(DEFAULT_DENOISE_PARAMS[method])
    if extra:
        raise ValueError(f"Invalid parameter(s) for {method}: {sorted(extra)}")
    if method == "gaussian":
        _check_odd("ksize", p["ksize"], 1)
        _check_positive("sigma", p["sigma"], allow_zero=True)
        return cv2.GaussianBlur(img, (int(p["ksize"]),) * 2, float(p["sigma"]))
    if method == "median":
        _check_odd("ksize", p["ksize"], 3)
        return cv2.medianBlur(img, int(p["ksize"]))
    # bilateral
    if isinstance(p["d"], bool) or not isinstance(p["d"], (int, np.integer)) or p["d"] < 1:
        raise ValueError(f"d must be an integer >= 1, got {p['d']!r}")
    _check_positive("sigma_color", p["sigma_color"])
    _check_positive("sigma_space", p["sigma_space"])
    return cv2.bilateralFilter(img, int(p["d"]), float(p["sigma_color"]), float(p["sigma_space"]))


# ----------------------------------------------------------- contrast stretch
def _stretch_channel(ch: np.ndarray, lo_p: float, hi_p: float) -> np.ndarray:
    lo, hi = np.percentile(ch, [lo_p, hi_p])
    if hi - lo < 1e-6:  # constant (or near-constant) -> nothing to stretch
        return ch.copy()
    out = (ch.astype(np.float32) - lo) / (hi - lo) * 255.0
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def _check_percentiles(lo_p, hi_p):
    if not (0 <= lo_p < hi_p <= 100):
        raise ValueError(f"Need 0 <= low_percentile < high_percentile <= 100, got {lo_p}, {hi_p}")


def contrast_stretch(img, low_percentile=1.0, high_percentile=99.0) -> np.ndarray:
    """Percentile-based linear contrast stretching.

    Grayscale: stretched directly. Colour: only the luminance (Y of YCrCb) is
    stretched so hue is preserved. Constant images are returned unchanged.
    """
    validate_image(img)
    _check_percentiles(low_percentile, high_percentile)
    if img.ndim == 2:
        return _stretch_channel(img, low_percentile, high_percentile)
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycc[..., 0] = _stretch_channel(ycc[..., 0], low_percentile, high_percentile)
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)


# ------------------------------------------------------------- low-light mode
def low_light_enhance(img, low_percentile=1.0, high_percentile=99.0, gamma=0.6,
                      use_clahe=True, clip_limit=2.0, tile_grid=8) -> np.ndarray:
    """Classical low-light enhancement: stretch -> gamma brighten -> optional CLAHE.

    Run denoising BEFORE this step (the pipeline does) so noise is not amplified.
    """
    validate_image(img)
    _check_percentiles(low_percentile, high_percentile)
    _check_positive("gamma", gamma)
    _check_positive("clip_limit", clip_limit)
    if isinstance(tile_grid, bool) or not isinstance(tile_grid, (int, np.integer)) or tile_grid < 1:
        raise ValueError(f"tile_grid must be an integer >= 1, got {tile_grid!r}")

    out = contrast_stretch(img, low_percentile, high_percentile)
    lut = (255.0 * (np.arange(256) / 255.0) ** float(gamma)).round().clip(0, 255).astype(np.uint8)

    def _lum(ch):
        ch = cv2.LUT(ch, lut)
        if use_clahe:
            ch = cv2.createCLAHE(clipLimit=float(clip_limit),
                                 tileGridSize=(int(tile_grid),) * 2).apply(ch)
        return ch

    if out.ndim == 2:
        return _lum(out)
    ycc = cv2.cvtColor(out, cv2.COLOR_BGR2YCrCb)
    ycc[..., 0] = _lum(ycc[..., 0])
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)


def enhance(img, mode="lowlight", **params) -> np.ndarray:
    validate_image(img)
    if mode not in ENHANCE_MODES:
        raise ValueError(f"Unknown enhance mode {mode!r}; choose from {ENHANCE_MODES}")
    if mode == "none":
        return img.copy()
    allowed = set(DEFAULT_ENHANCE_PARAMS[mode])
    extra = set(params) - allowed
    if extra:
        raise ValueError(f"Invalid parameter(s) for {mode}: {sorted(extra)}")
    p = {**DEFAULT_ENHANCE_PARAMS[mode], **params}
    return contrast_stretch(img, **p) if mode == "stretch" else low_light_enhance(img, **p)


# -------------------------------------------------------------------- metrics
def psnr(a: np.ndarray, b: np.ndarray):
    if a.shape != b.shape:
        raise ValueError("PSNR needs equal shapes")
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return None if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))  # None = identical


def ssim(a: np.ndarray, b: np.ndarray):
    """SSIM via scikit-image if installed, else None (never invented)."""
    try:
        from skimage.metrics import structural_similarity
    except ImportError:
        return None
    kw = {"channel_axis": 2} if a.ndim == 3 else {}
    return float(structural_similarity(a, b, data_range=255, **kw))


def image_stats(img: np.ndarray) -> dict:
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    prob = hist[hist > 0] / hist.sum()
    return {
        "mean_intensity": float(gray.mean()),
        "std_intensity": float(gray.std()),
        "min": int(gray.min()), "max": int(gray.max()),
        "entropy_bits": float(-(prob * np.log2(prob)).sum()),
        "laplacian_variance": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
    }


def compute_metrics(original, denoised, enhanced, reference=None) -> dict:
    m = {"original": image_stats(original), "denoised": image_stats(denoised),
         "enhanced": image_stats(enhanced),
         "notes": ["Higher contrast/entropy does not automatically mean better quality.",
                   "laplacian_variance rises with both sharpness AND noise; interpret with care."]}
    if reference is not None:
        validate_image(reference)
        m["reference_comparison"] = {
            "psnr_original_vs_reference_db": psnr(original, reference),
            "psnr_denoised_vs_reference_db": psnr(denoised, reference),
            "psnr_enhanced_vs_reference_db": psnr(enhanced, reference),
            "ssim_original_vs_reference": ssim(original, reference),
            "ssim_denoised_vs_reference": ssim(denoised, reference),
            "ssim_enhanced_vs_reference": ssim(enhanced, reference),
            "note": "PSNR/SSIM are meaningful only when the reference is the intended clean image.",
        }
    else:
        m["reference_comparison"] = None
    return m


# ----------------------------------------------------------------- comparison
def make_comparison(original, enhanced) -> np.ndarray:
    def bgr(i): return cv2.cvtColor(i, cv2.COLOR_GRAY2BGR) if i.ndim == 2 else i
    a, b = bgr(original), bgr(enhanced)
    bar = 28
    def label(img, text):
        canvas = np.full((img.shape[0] + bar, img.shape[1], 3), 30, np.uint8)
        canvas[bar:] = img
        cv2.putText(canvas, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        return canvas
    return np.hstack([label(a, "Original"), label(b, "Enhanced")])


# ------------------------------------------------------------------- pipeline
def run_pipeline(input_path, output_dir, denoise_method="gaussian", denoise_params=None,
                 enhance_mode="lowlight", enhance_params=None, reference_path=None) -> dict:
    """Full workflow. Writes 6 output files into output_dir; never modifies input."""
    t0 = time.perf_counter()
    denoise_params = dict(denoise_params or {})
    enhance_params = dict(enhance_params or {})
    inp = Path(input_path).resolve()
    out_dir = Path(output_dir).resolve()

    # 1. validation (before touching disk)
    img = load_image(inp)
    if denoise_method not in DENOISE_METHODS:
        raise ValueError(f"Unknown denoise method {denoise_method!r}; choose from {DENOISE_METHODS}")
    if enhance_mode not in ENHANCE_MODES:
        raise ValueError(f"Unknown enhance mode {enhance_mode!r}; choose from {ENHANCE_MODES}")
    reference = load_image(reference_path) if reference_path else None
    if reference is not None and reference.shape != img.shape:
        raise ValueError("Reference image must have the same shape as the input image")

    # 2-3. denoise, enhance (raise on bad params before any output is written)
    den = denoise(img, denoise_method, **denoise_params)
    enh = enhance(den, enhance_mode, **enhance_params)

    # protect the input: never write an output on top of it
    out_dir.mkdir(parents=True, exist_ok=True)
    names = ["original.png", "denoised.png", "enhanced.png", "comparison.png",
             "run_summary.json", "quality_metrics.json"]
    if inp in {(out_dir / n).resolve() for n in names}:
        raise ValueError("Output directory would overwrite the input image; choose another directory")

    # 4-6. comparison, metrics, save
    paths = {n: str(out_dir / n) for n in names}
    cv2.imwrite(paths["original.png"], img)
    cv2.imwrite(paths["denoised.png"], den)
    cv2.imwrite(paths["enhanced.png"], enh)
    cv2.imwrite(paths["comparison.png"], make_comparison(img, enh))
    metrics = compute_metrics(img, den, enh, reference)
    with open(paths["quality_metrics.json"], "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    summary = {
        "input_path": str(inp),
        "reference_path": str(Path(reference_path).resolve()) if reference_path else None,
        "image_height": int(img.shape[0]), "image_width": int(img.shape[1]),
        "channels": 1 if img.ndim == 2 else int(img.shape[2]),
        "denoise_method": denoise_method,
        "denoise_params": ({**DEFAULT_DENOISE_PARAMS[denoise_method], **denoise_params}
                           if denoise_method != "none" else {}),
        "enhance_mode": enhance_mode,
        "enhance_params": ({**DEFAULT_ENHANCE_PARAMS[enhance_mode], **enhance_params}
                           if enhance_mode != "none" else {}),
        "output_paths": paths,
    }
    summary["processing_time_seconds"] = round(time.perf_counter() - t0, 4)
    with open(paths["run_summary.json"], "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary

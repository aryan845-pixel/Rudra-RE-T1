#!/usr/bin/env python3
"""Rudra Edge Aerospace - Day 11: Image Super-Resolution Workbench.

Pipeline:
    Input image -> validation -> low-resolution sample -> bilinear / bicubic /
    Lanczos upscaling -> optional AI super-resolution (OpenCV dnn_superres) ->
    PSNR / SSIM evaluation -> interactive OpenCV GUI -> saved outputs + reports.

Only OpenCV and NumPy are required. Nothing is uploaded and no cloud service is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}
INTERPOLATIONS = {
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
    "lanczos": cv2.INTER_LANCZOS4,
}
MIN_SCALE, MAX_SCALE = 2, 8
MIN_LR_SIDE = 8  # smallest allowed low-resolution side, in pixels
WINDOW_NAME = "Rudra Edge - Super-Resolution Workbench"
MODEL_RE = re.compile(r"^(edsr|espcn|fsrcnn|lapsrn)_x(\d+)\.pb$", re.IGNORECASE)
AI_PREFERENCE = ["edsr", "lapsrn", "fsrcnn", "espcn"]
SCRIPT_DIR = Path(__file__).resolve().parent


class SuperResError(Exception):
    """Base class for expected, user-reportable errors."""


class ImageLoadError(SuperResError):
    """Image is missing, unsupported, or unreadable."""


class InvalidScaleError(SuperResError, ValueError):
    """Scale factor is not usable."""


class GuiUnavailableError(SuperResError):
    """A GUI window cannot be created in this environment."""


# --------------------------------------------------------------------------- #
# Image I/O and validation
# --------------------------------------------------------------------------- #
def validate_scale(scale) -> int:
    if isinstance(scale, bool) or not isinstance(scale, (int, float, np.integer)):
        raise InvalidScaleError(f"Scale must be an integer, got {scale!r}")
    if float(scale) != int(scale):
        raise InvalidScaleError(f"Scale must be a whole number, got {scale!r}")
    scale = int(scale)
    if not MIN_SCALE <= scale <= MAX_SCALE:
        raise InvalidScaleError(f"Scale must be between {MIN_SCALE} and {MAX_SCALE}, got {scale}")
    return scale


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def to_bgr_uint8(img: np.ndarray) -> tuple[np.ndarray, dict]:
    """Normalise any decoded image to 3-channel uint8 BGR; report what changed."""
    info = {"decoded_shape": list(img.shape), "decoded_dtype": str(img.dtype), "was_grayscale": False, "alpha_dropped": False}
    if img.dtype == np.uint16:
        img = (img // 257).astype(np.uint8)
    elif img.dtype != np.uint8:
        raise ImageLoadError(f"Unsupported pixel type {img.dtype}")
    if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
        info["was_grayscale"] = True
        img = cv2.cvtColor(img.reshape(img.shape[:2]), cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        info["alpha_dropped"] = True
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif img.shape[2] != 3:
        raise ImageLoadError(f"Unsupported channel count {img.shape[2]}")
    return np.ascontiguousarray(img), info


def load_image(path) -> tuple[np.ndarray, dict]:
    p = Path(path)
    if not p.exists():
        raise ImageLoadError(f"Image file not found: {p}")
    if not p.is_file():
        raise ImageLoadError(f"Not a file: {p}")
    if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ImageLoadError(f"Unsupported format '{p.suffix}'. Use one of: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
    try:
        data = np.fromfile(str(p), dtype=np.uint8)  # also works for non-ASCII Windows paths
    except OSError as exc:
        raise ImageLoadError(f"Cannot read {p}: {exc}") from exc
    if data.size == 0:
        raise ImageLoadError(f"Image file is empty: {p}")
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None or img.size == 0:
        raise ImageLoadError(f"Could not decode {p} (corrupt or not a valid image)")
    return to_bgr_uint8(img)


def save_png(path: Path, img: np.ndarray) -> int:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise SuperResError(f"PNG encoding failed for {path}")
    buf.tofile(str(path))
    return path.stat().st_size


def make_synthetic_image(width: int = 640, height: int = 480) -> np.ndarray:
    """Deterministic test image with edges, text, curves and fine texture."""
    y, x = np.mgrid[0:height, 0:width]
    img = np.zeros((height, width, 3), np.uint8)
    img[..., 0] = (40 + 120 * x / width).astype(np.uint8)
    img[..., 1] = (60 + 100 * y / height).astype(np.uint8)
    img[..., 2] = (200 - 120 * x / width).astype(np.uint8)
    cx, cy = width // 2, height // 2
    for r in range(20, min(width, height) // 2, 18):
        cv2.circle(img, (cx, cy), r, (255, 255, 255) if (r // 18) % 2 else (20, 20, 20), 2, cv2.LINE_AA)
    for i in range(0, width, 40):
        cv2.line(img, (i, 0), (width - 1 - i // 2, height - 1), (0, 255, 255), 1, cv2.LINE_AA)
    tile = 8
    ty0, tx0 = height - 100, 20
    for r in range(8):
        for c in range(8):
            if (r + c) % 2 == 0:
                cv2.rectangle(img, (tx0 + c * tile, ty0 + r * tile), (tx0 + (c + 1) * tile - 1, ty0 + (r + 1) * tile - 1), (255, 255, 255), -1)
    sw, sh = min(96, width // 3), min(48, height // 3)  # stripe patch shrinks with the image
    if sw >= 6 and sh >= 6:
        stripes = (127 + 127 * np.sin(2 * np.pi * np.arange(sw) / 6.0)).astype(np.uint8)
        img[20:20 + sh, width - 20 - sw:width - 20] = stripes[None, :, None]
    cv2.putText(img, "RUDRA EDGE SR TEST 0123456789", (24, height // 2 - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, "fine detail: the quick brown fox", (24, height // 2 + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return img


# --------------------------------------------------------------------------- #
# Super-resolution
# --------------------------------------------------------------------------- #
def prepare_reference(img: np.ndarray, scale: int) -> tuple[np.ndarray, np.ndarray]:
    """Crop to a multiple of `scale`, then build the LR image with INTER_AREA.

    INTER_AREA is the standard anti-aliased downscale, which approximates the
    way a real low-resolution sensor would have sampled the scene.
    """
    scale = validate_scale(scale)
    h, w = img.shape[:2]
    h2, w2 = h - h % scale, w - w % scale
    if h2 // scale < MIN_LR_SIDE or w2 // scale < MIN_LR_SIDE:
        raise SuperResError(f"Image {w}x{h} is too small for {scale}x (low-res side would be under {MIN_LR_SIDE}px)")
    reference = np.ascontiguousarray(img[:h2, :w2])
    lr = cv2.resize(reference, (w2 // scale, h2 // scale), interpolation=cv2.INTER_AREA)
    return reference, lr


def upscale(lr: np.ndarray, scale: int, method: str, repeats: int = 5) -> tuple[np.ndarray, dict]:
    scale = validate_scale(scale)
    if method not in INTERPOLATIONS:
        raise ValueError(f"Unknown method {method!r}")
    h, w = lr.shape[:2]
    dsize = (w * scale, h * scale)
    flag = INTERPOLATIONS[method]
    out = cv2.resize(lr, dsize, interpolation=flag)  # warm-up, also the returned result
    times = []
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        out = cv2.resize(lr, dsize, interpolation=flag)
        times.append((time.perf_counter() - t0) * 1000.0)
    return out, {"time_ms_median": round(statistics.median(times), 4), "time_ms_min": round(min(times), 4), "repeats": len(times)}


def find_ai_models(model_dirs, scale: int) -> list[tuple[str, Path]]:
    found = {}
    for d in model_dirs:
        d = Path(d)
        if not d.is_dir():
            continue
        for f in sorted(d.iterdir()):
            m = MODEL_RE.match(f.name)
            if m and int(m.group(2)) == scale and f.is_file():
                found.setdefault(m.group(1).lower(), f)
    return [(n, found[n]) for n in AI_PREFERENCE if n in found]


def run_ai_superres(lr: np.ndarray, scale: int, model_dirs) -> dict:
    """Try real dnn_superres inference. Never fabricates a result."""
    result = {"available": False, "image": None, "model": None, "model_file": None, "scale": scale, "time_ms": None, "reason": None}
    mod = getattr(cv2, "dnn_superres", None)
    factory = getattr(mod, "DnnSuperResImpl_create", None) or getattr(mod, "DnnSuperResImpl", None) if mod else None
    if factory is None:
        result["reason"] = "cv2.dnn_superres has no DnnSuperResImpl in this OpenCV build (needs a working opencv-contrib-python)"
        return result
    candidates = find_ai_models(model_dirs, scale)
    if not candidates:
        result["reason"] = f"No compatible model file (EDSR/LapSRN/FSRCNN/ESPCN _x{scale}.pb) found in: " + ", ".join(str(d) for d in model_dirs)
        return result
    errors = []
    for name, path in candidates:
        try:
            sr = factory()
            sr.readModel(str(path))
            sr.setModel(name, scale)
            t0 = time.perf_counter()
            out = sr.upsample(lr)
            ms = (time.perf_counter() - t0) * 1000.0
            expected = (lr.shape[0] * scale, lr.shape[1] * scale)
            if out is None or out.shape[:2] != expected:
                raise RuntimeError(f"unexpected output shape {None if out is None else out.shape}, expected {expected}")
            result.update(available=True, image=out, model=name.upper(), model_file=str(path), time_ms=round(ms, 3), reason=None)
            return result
        except Exception as exc:  # cv2.error, RuntimeError, ... - any failure means no AI result
            errors.append(f"{path.name}: {type(exc).__name__}: {str(exc).strip().splitlines()[0] if str(exc).strip() else ''}")
    result["reason"] = "Model file(s) found but inference failed - " + "; ".join(errors)
    return result


# --------------------------------------------------------------------------- #
# Quality metrics (reference-based)
# --------------------------------------------------------------------------- #
def psnr(ref: np.ndarray, test: np.ndarray):
    if ref.shape != test.shape:
        raise ValueError(f"Shape mismatch {ref.shape} vs {test.shape}")
    mse = float(np.mean((ref.astype(np.float64) - test.astype(np.float64)) ** 2))
    return None if mse == 0 else 10.0 * math.log10(255.0 ** 2 / mse)  # None = identical images


def ssim(ref: np.ndarray, test: np.ndarray) -> float:
    """Mean SSIM, 11x11 Gaussian window (sigma 1.5), averaged over channels."""
    if ref.shape != test.shape:
        raise ValueError(f"Shape mismatch {ref.shape} vs {test.shape}")
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    a, b = ref.astype(np.float64), test.astype(np.float64)

    def blur(x):
        return cv2.GaussianBlur(x, (11, 11), 1.5)

    scores = []
    for ch in range(a.shape[2]):
        x, y = a[..., ch], b[..., ch]
        mx, my = blur(x), blur(y)
        sxx, syy, sxy = blur(x * x) - mx * mx, blur(y * y) - my * my, blur(x * y) - mx * my
        smap = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (sxx + syy + c2))
        scores.append(float(smap[5:-5, 5:-5].mean()) if min(smap.shape) > 10 else float(smap.mean()))
    return float(np.mean(scores))


# --------------------------------------------------------------------------- #
# Comparison canvas (display + comparison.png)
# --------------------------------------------------------------------------- #
FONT = cv2.FONT_HERSHEY_SIMPLEX
BG, CELL_BG, FG, ACCENT, DIM, WARN = (32, 28, 26), (46, 40, 38), (235, 235, 235), (255, 200, 90), (150, 150, 150), (80, 160, 255)


def put_fit(canvas, text, org, max_w, scale=0.5, color=FG, thick=1):
    while scale > 0.3 and cv2.getTextSize(text, FONT, scale, thick)[0][0] > max_w:
        scale -= 0.02
    cv2.putText(canvas, text, org, FONT, scale, color, thick, cv2.LINE_AA)


def fit_into(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """Letterbox `img` into w x h keeping aspect ratio (display only)."""
    ih, iw = img.shape[:2]
    s = min(w / iw, h / ih)
    nw, nh = max(1, int(round(iw * s))), max(1, int(round(ih * s)))
    shown = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_NEAREST)
    cell = np.full((h, w, 3), CELL_BG, np.uint8)
    y0, x0 = (h - nh) // 2, (w - nw) // 2
    cell[y0:y0 + nh, x0:x0 + nw] = shown
    return cell


def fmt_ms(v):
    return "-" if v is None else f"{v:.3f}"


def compose_canvas(title, panels, width=1620, cell_h=340, footer=None):
    """panels: list of 6 dicts {title, image|None, lines, note}. Returns BGR canvas."""
    cols, rows = 3, 2
    cell_w = width // cols
    head, cap_h, title_h = 56, 46, 28
    table_rows = [p["row"] for p in panels]
    table_h = 34 + 26 * len(table_rows) + 16
    foot_h = 30 if footer else 0
    height = head + rows * cell_h + table_h + foot_h
    canvas = np.full((height, cell_w * cols, 3), BG, np.uint8)
    put_fit(canvas, title, (20, 38), width - 40, 0.9, ACCENT, 2)
    for i, p in enumerate(panels):
        r, c = divmod(i, cols)
        x0, y0 = c * cell_w + 4, head + r * cell_h + 4
        w, h = cell_w - 8, cell_h - 8
        canvas[y0:y0 + h, x0:x0 + w] = CELL_BG
        put_fit(canvas, p["title"], (x0 + 10, y0 + 20), w - 20, 0.6, ACCENT, 1)
        area_h = h - title_h - cap_h
        if p["image"] is not None:
            canvas[y0 + title_h:y0 + title_h + area_h, x0:x0 + w] = fit_into(p["image"], w, area_h)
        else:
            put_fit(canvas, "UNAVAILABLE", (x0 + 20, y0 + title_h + area_h // 2 - 10), w - 40, 0.9, WARN, 2)
            words, line, ly = p["note"].split(), "", y0 + title_h + area_h // 2 + 20
            for word in words:
                if cv2.getTextSize(line + " " + word, FONT, 0.42, 1)[0][0] > w - 40 and line:
                    put_fit(canvas, line, (x0 + 20, ly), w - 40, 0.42, DIM)
                    line, ly = word, ly + 18
                else:
                    line = (line + " " + word).strip()
            put_fit(canvas, line, (x0 + 20, ly), w - 40, 0.42, DIM)
        for k, text in enumerate(p["lines"][:2]):
            put_fit(canvas, text, (x0 + 10, y0 + h - cap_h + 18 + 18 * k), w - 20, 0.47, FG)
    ty = head + rows * cell_h + 6
    cols_x = [20, 330, 560, 760, 940, 1120]
    headers = ["Method", "Output size", "Time (ms)", "PSNR (dB)", "SSIM", "File size"]
    cv2.line(canvas, (10, ty), (canvas.shape[1] - 10, ty), DIM, 1)
    for x, t in zip(cols_x, headers):
        put_fit(canvas, t, (x, ty + 22), 200, 0.55, ACCENT, 1)
    for k, row in enumerate(table_rows):
        for x, t in zip(cols_x, row):
            put_fit(canvas, t, (x, ty + 24 + 26 * (k + 1)), 300 if x == cols_x[0] else 190, 0.5, FG)
    if footer:
        put_fit(canvas, footer, (20, height - 10), canvas.shape[1] - 40, 0.5, DIM)
    return canvas


def build_panels(res: dict, detail=None):
    """Build the six panels (+ table rows). `detail` = (x0, y0, rw, rh) in reference pixels."""
    s, ref, lr = res["scale"], res["reference"], res["lr"]

    def crop(img, factor=1):
        if detail is None:
            return img
        x0, y0, rw, rh = detail
        return img[y0 // factor:(y0 + rh) // factor, x0 // factor:(x0 + rw) // factor]

    def q(m):
        return m["psnr_db"], m["ssim"]

    def qtxt(m):
        p = "inf" if m["psnr_db"] is None else f"{m['psnr_db']:.2f}"
        return p, f"{m['ssim']:.4f}"

    panels = []
    rh_, rw_ = ref.shape[:2]
    lh_, lw_ = lr.shape[:2]
    panels.append(dict(title="Original (reference)", image=crop(ref), lines=[f"{rw_}x{rh_} px", "ground truth for PSNR/SSIM"], note="",
                       row=["Original (reference)", f"{rw_}x{rh_}", "-", "-", "-", fmt_bytes(res["files"]["original"])]))
    panels.append(dict(title=f"Low resolution ({s}x down, INTER_AREA)", image=crop(lr, s), lines=[f"{lw_}x{lh_} px", "shown with nearest-neighbour zoom"], note="",
                       row=["Low resolution (input)", f"{lw_}x{lh_}", "-", "-", "-", fmt_bytes(res["files"]["low_resolution"])]))
    for name in INTERPOLATIONS:
        m = res["methods"][name]
        p, ss = qtxt(m)
        w_, h_ = m["output_size"]
        panels.append(dict(title=f"{name.capitalize()} {s}x", image=crop(m["image"]), lines=[f"{w_}x{h_} px | {m['time_ms_median']:.3f} ms", f"PSNR {p} dB | SSIM {ss}"], note="",
                           row=[f"{name.capitalize()} {s}x", f"{w_}x{h_}", fmt_ms(m["time_ms_median"]), p, ss, fmt_bytes(m["file_bytes"])]))
    ai = res["ai"]
    if ai["available"]:
        m = res["methods"]["ai"]
        p, ss = qtxt(m)
        w_, h_ = m["output_size"]
        panels.append(dict(title=f"AI: {ai['model']} {s}x", image=crop(m["image"]), lines=[f"{w_}x{h_} px | {ai['time_ms']:.1f} ms (1 run)", f"PSNR {p} dB | SSIM {ss}"], note="",
                           row=[f"AI {ai['model']} {s}x", f"{w_}x{h_}", fmt_ms(ai["time_ms"]), p, ss, fmt_bytes(m["file_bytes"])]))
    else:
        panels.append(dict(title="AI super-resolution", image=None, lines=["no AI output - not fabricated", "interpolation is NOT shown here"], note=ai["reason"],
                           row=["AI super-resolution", "unavailable", "-", "-", "-", "-"]))
    return panels


def fmt_bytes(n):
    return "-" if n is None else (f"{n / 1024:.1f} KiB" if n < 1 << 20 else f"{n / (1 << 20):.2f} MiB")


def default_detail_region(ref: np.ndarray, scale: int):
    """Pick the most detailed window (max Laplacian energy) as the default zoom region."""
    h, w = ref.shape[:2]
    rw = max(scale * 4, min(160, w // 4) // scale * scale)
    rh = max(scale * 4, min(160, h // 4) // scale * scale)
    rw, rh = min(rw, w // scale * scale), min(rh, h // scale * scale)
    lap = np.abs(cv2.Laplacian(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), cv2.CV_32F))
    resp = cv2.boxFilter(lap, -1, (rw, rh), normalize=True)
    y, x = np.unravel_index(int(np.argmax(resp)), resp.shape)
    return clamp_region(x - rw // 2, y - rh // 2, rw, rh, w, h, scale)


def clamp_region(x0, y0, rw, rh, w, h, scale):
    x0 = int(min(max(0, x0), w - rw)) // scale * scale
    y0 = int(min(max(0, y0), h - rh)) // scale * scale
    return (x0, y0, rw, rh)


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def default_model_dirs():
    dirs = [Path.cwd() / "models", SCRIPT_DIR / "models"]
    if os.environ.get("SR_MODEL_DIR"):
        dirs.insert(0, Path(os.environ["SR_MODEL_DIR"]))
    seen, out = set(), []
    for d in dirs:
        if str(d) not in seen:
            seen.add(str(d))
            out.append(d)
    return out


def run_pipeline(image_path, scale, out_dir, model_dirs=None, repeats=5) -> dict:
    scale = validate_scale(scale)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dirs = list(model_dirs) if model_dirs else default_model_dirs()
    names = {"original.png", "low_resolution.png", "comparison.png", "run_summary.json", "quality_metrics.json", "ai_super_resolution.png", "sample_input.png"}
    names |= {f"{m}_x{scale}.png" for m in INTERPOLATIONS}

    input_info = {"source": "synthetic", "path": None, "sha256_before": None}
    if image_path is None:
        img = make_synthetic_image()
        decode_info = {"was_grayscale": False, "alpha_dropped": False}
        save_png(out_dir / "sample_input.png", img)
    else:
        p = Path(image_path)
        if p.exists() and p.resolve().parent == out_dir.resolve() and p.name in names:
            raise SuperResError(f"Input {p.name} lives in the output folder and would be overwritten; choose another --out directory")
        img, decode_info = load_image(p)
        input_info = {"source": "file", "path": str(p.resolve()), "sha256_before": sha256_file(p)}

    reference, lr = prepare_reference(img, scale)
    files, methods = {}, {}
    files["original"] = save_png(out_dir / "original.png", reference)
    files["low_resolution"] = save_png(out_dir / "low_resolution.png", lr)

    for name in INTERPOLATIONS:
        out, timing = upscale(lr, scale, name, repeats)
        fname = f"{name}_x{scale}.png"
        methods[name] = {"image": out, "output_size": [out.shape[1], out.shape[0]], "file": fname, "file_bytes": save_png(out_dir / fname, out),
                         "psnr_db": psnr(reference, out), "ssim": ssim(reference, out), **timing}

    ai = run_ai_superres(lr, scale, model_dirs)
    if ai["available"]:
        out = ai["image"]
        methods["ai"] = {"image": out, "output_size": [out.shape[1], out.shape[0]], "file": "ai_super_resolution.png",
                         "file_bytes": save_png(out_dir / "ai_super_resolution.png", out), "psnr_db": psnr(reference, out), "ssim": ssim(reference, out)}
    else:
        stale = out_dir / "ai_super_resolution.png"
        if stale.exists():
            stale.unlink()  # remove our own output from an earlier run so it cannot be mistaken for this run's result

    res = {"scale": scale, "reference": reference, "lr": lr, "methods": methods, "ai": ai, "files": files, "out_dir": out_dir,
           "input": input_info, "decode_info": decode_info, "repeats": repeats, "model_dirs": [str(d) for d in model_dirs]}
    canvas = compose_canvas("RUDRA EDGE - SUPER-RESOLUTION WORKBENCH", build_panels(res))
    files["comparison"] = save_png(out_dir / "comparison.png", canvas)
    res["canvas"] = canvas

    if input_info["path"]:
        after = sha256_file(Path(input_info["path"]))
        input_info.update(sha256_after=after, unchanged=after == input_info["sha256_before"])
    write_reports(res)
    return res


def write_reports(res: dict, gui_status: str = "pending", gui_detail: str = "") -> None:
    ref, lr, s = res["reference"], res["lr"], res["scale"]
    ai = res["ai"]
    summary = {
        "task": "Day 11 - Super-Resolution Study", "date": datetime.now().astimezone().isoformat(timespec="seconds"),
        "environment": {"python": platform.python_version(), "opencv": cv2.__version__, "numpy": np.__version__, "platform": platform.platform()},
        "input": {**res["input"], **res["decode_info"]}, "scale": s,
        "reference_size_wh": [ref.shape[1], ref.shape[0]], "low_res_size_wh": [lr.shape[1], lr.shape[0]],
        "low_res_generation": "reference cropped to a multiple of scale, then cv2.resize(..., INTER_AREA)",
        "timing_note": f"interpolation time = median of {res['repeats']} runs after 1 warm-up; AI time = single run",
        "methods": {n: {k: v for k, v in m.items() if k in ("output_size", "file", "time_ms_median", "time_ms_min", "repeats")} for n, m in res["methods"].items() if n != "ai"},
        "ai": {"available": ai["available"], "model": ai["model"], "model_file": ai["model_file"], "scale": s, "inference_time_ms": ai["time_ms"], "reason_unavailable": ai["reason"],
               "model_search_dirs": res["model_dirs"]},
        "gui": {"status": gui_status, "detail": gui_detail},
        "output_dir": str(res["out_dir"]), "files": sorted(p.name for p in Path(res["out_dir"]).iterdir() if p.is_file()),
    }
    metrics = {
        "note": "Reference-based: each result is compared with the cropped original at identical dimensions. PSNR in dB (higher = closer); SSIM 0..1. "
                "Larger output size alone says nothing about quality. Synthetic/downscaled-reference results do not predict performance on real low-res photos.",
        "scale": s, "psnr_infinite_is_null": True,
        "methods": {n: {"output_size_wh": m["output_size"], "psnr_db": None if m["psnr_db"] is None else round(m["psnr_db"], 4), "ssim": round(m["ssim"], 6), "file_bytes": m["file_bytes"]}
                    for n, m in res["methods"].items()},
        "ai_available": ai["available"],
    }
    for name, data in (("run_summary.json", summary), ("quality_metrics.json", metrics)):
        (Path(res["out_dir"]) / name).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


# --------------------------------------------------------------------------- #
# GUI
# --------------------------------------------------------------------------- #
def gui_available() -> tuple[bool, str]:
    if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return False, "No DISPLAY / WAYLAND_DISPLAY is set (headless environment)."
    return True, ""


def screen_size() -> tuple[int, int]:
    try:
        import tkinter
        root = tkinter.Tk()
        root.withdraw()
        size = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        return size
    except Exception:
        return 1366, 768


def pick_image_dialog():
    try:
        import tkinter
        from tkinter import filedialog
        root = tkinter.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(title="Select an image", filetypes=[("Images", "*.png *.jpg *.jpeg")])
        root.destroy()
        return path or None, ""
    except Exception as exc:
        return None, f"File dialog unavailable ({type(exc).__name__}); restart with --image PATH"


class Workbench:
    HELP = "[q / Esc] quit   [o] open image   [2][3][4] scale   [v] detail zoom   [w a s d] move zoom region"

    def __init__(self, image_path, scale, out_dir, model_dirs, repeats):
        self.image_path, self.scale, self.out_dir, self.model_dirs, self.repeats = image_path, scale, out_dir, model_dirs, repeats
        self.res = None
        self.detail_mode = False
        self.region = None
        self.status = ""

    def process(self):
        self.res = run_pipeline(self.image_path, self.scale, self.out_dir, self.model_dirs, self.repeats)
        self.region = default_detail_region(self.res["reference"], self.scale)

    def frame(self):
        footer = (self.status + "   |   " if self.status else "") + self.HELP
        if self.detail_mode:
            x0, y0, rw, rh = self.region
            title = f"DETAIL VIEW - reference region x={x0} y={y0} {rw}x{rh}px (nearest-neighbour zoom)"
            return compose_canvas(title, build_panels(self.res, self.region), footer=footer)
        return compose_canvas("RUDRA EDGE - SUPER-RESOLUTION WORKBENCH", build_panels(self.res), footer=footer)

    def run(self, auto_close_s=None) -> str:
        ok, msg = gui_available()
        if not ok:
            raise GuiUnavailableError(msg)
        try:
            frame = self.frame()
            sw, sh = screen_size()
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            ww = min(frame.shape[1], int(sw * 0.95))
            cv2.resizeWindow(WINDOW_NAME, ww, min(int(ww * frame.shape[0] / frame.shape[1]), int(sh * 0.92)))
            cv2.imshow(WINDOW_NAME, frame)
        except cv2.error as exc:
            raise GuiUnavailableError(f"OpenCV could not create a window: {exc}") from exc
        start, reason = time.time(), "closed"
        try:
            while True:
                key = cv2.waitKey(30) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    reason = "closed with keyboard (q/Esc)"
                    break
                try:
                    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                        reason = "closed with window close button"
                        break
                except cv2.error:
                    reason = "window closed"
                    break
                if auto_close_s is not None and time.time() - start > auto_close_s:
                    reason = f"auto-closed after {auto_close_s}s (--selftest-close-after)"
                    break
                dirty = self.handle_key(key)
                if dirty:
                    cv2.imshow(WINDOW_NAME, self.frame())
        finally:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        return reason

    def handle_key(self, key) -> bool:
        if key == 255:
            return False
        ch = chr(key).lower() if 32 <= key < 127 else ""
        if ch == "v":
            self.detail_mode = not self.detail_mode
            return True
        if ch in "wasd" and ch and self.detail_mode:
            x0, y0, rw, rh = self.region
            step = max(self.scale, rw // 4) // self.scale * self.scale
            dx, dy = {"a": (-step, 0), "d": (step, 0), "w": (0, -step), "s": (0, step)}[ch]
            ref = self.res["reference"]
            self.region = clamp_region(x0 + dx, y0 + dy, rw, rh, ref.shape[1], ref.shape[0], self.scale)
            return True
        if ch in ("2", "3", "4"):
            return self.reprocess(scale=int(ch))
        if ch == "o":
            path, err = pick_image_dialog()
            if err:
                self.status = err
                return True
            return self.reprocess(path=path) if path else False
        return False

    def reprocess(self, scale=None, path=None) -> bool:
        old = (self.scale, self.image_path)
        self.scale = scale or self.scale
        self.image_path = path or self.image_path
        self.status = "Processing..."
        cv2.imshow(WINDOW_NAME, self.frame())
        cv2.waitKey(1)
        try:
            self.process()
            self.status = f"Processed at {self.scale}x"
        except SuperResError as exc:
            self.scale, self.image_path = old
            self.status = f"Error: {exc}"
        return True


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser():
    ap = argparse.ArgumentParser(description="Rudra Edge Day 11 - image super-resolution workbench")
    ap.add_argument("image", nargs="?", help="input image (PNG/JPG/JPEG); a synthetic sample is used if omitted")
    ap.add_argument("--image", dest="image_opt", help="same as the positional image argument")
    ap.add_argument("--scale", type=int, default=4, help=f"upscale factor {MIN_SCALE}-{MAX_SCALE} (default 4)")
    ap.add_argument("--out", default=str(Path("outputs") / "day11"), help="output directory (default outputs/day11)")
    ap.add_argument("--model-dir", action="append", help="folder holding EDSR_x4.pb / FSRCNN_x2.pb ... (repeatable)")
    ap.add_argument("--repeats", type=int, default=5, help="timing repeats for interpolation methods")
    ap.add_argument("--require-ai", action="store_true", help="exit with code 3 (before opening the GUI) if real AI super-resolution is unavailable")
    ap.add_argument("--no-gui", action="store_true", help="explicitly skip the window (for automated tests only)")
    ap.add_argument("--selftest-close-after", type=float, default=None, metavar="SEC", help="auto-close the window after SEC seconds (smoke tests)")
    return ap


def print_summary(res):
    print(f"\nScale {res['scale']}x | reference {res['reference'].shape[1]}x{res['reference'].shape[0]} -> low-res {res['lr'].shape[1]}x{res['lr'].shape[0]}")
    print(f"{'Method':<12}{'Output':>12}{'Time ms':>11}{'PSNR dB':>10}{'SSIM':>9}")
    for n, m in res["methods"].items():
        p = "inf" if m["psnr_db"] is None else f"{m['psnr_db']:.2f}"
        t = m.get("time_ms_median", res["ai"]["time_ms"])
        print(f"{n:<12}{m['output_size'][0]}x{m['output_size'][1]:<6}{t:>11.3f}{p:>10}{m['ssim']:>9.4f}")
    print("AI super-resolution:", "OK - " + res["ai"]["model"] if res["ai"]["available"] else "UNAVAILABLE - " + res["ai"]["reason"])
    print(f"Outputs saved in: {res['out_dir']}")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    image = args.image_opt or args.image
    try:
        wb = Workbench(image, validate_scale(args.scale), args.out, args.model_dir, args.repeats)
        wb.process()
    except SuperResError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print_summary(wb.res)
    if args.require_ai and not wb.res["ai"]["available"]:
        print(f"ERROR: --require-ai was given but AI super-resolution is unavailable: {wb.res['ai']['reason']}", file=sys.stderr)
        write_reports(wb.res, "not_opened", "--require-ai failed")
        return 3
    if args.no_gui:
        print("GUI skipped: --no-gui was given. No window was displayed.")
        write_reports(wb.res, "skipped", "--no-gui flag")
        return 0
    try:
        reason = wb.run(args.selftest_close_after)
    except GuiUnavailableError as exc:
        print(f"\nGUI ERROR: {exc}", file=sys.stderr)
        print("Outputs were saved, but NO window was displayed. Run this on a desktop session with a display "
              "(e.g. Windows PowerShell on a normal desktop) and a GUI-enabled OpenCV (pip install opencv-python).", file=sys.stderr)
        write_reports(wb.res, "failed", str(exc))
        return 2
    print(f"GUI window {reason}.")
    write_reports(wb.res, "opened", f"{reason} [DISPLAY={os.environ.get('DISPLAY') or 'native desktop'}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Day 13 - Visual Change Detection (Rudra Edge Aerospace, RE-T1 / Spectra)

Offline, classical pipeline: compare a reference image with a changed image,
produce raw difference, delta heatmap, binary change mask and a final Change Map.

Libraries: Python, OpenCV (cv2), NumPy, Matplotlib (optional, only for the figure).
No network, no paid service, no pretrained model.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "input" if (HERE / "input" / "reference.png").exists() else HERE / "samples"
DEFAULT_REFERENCE = DEFAULT_INPUT / "reference.png"
DEFAULT_CHANGED = DEFAULT_INPUT / "changed.png"
DEFAULT_OUTPUT = HERE / "output"
SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}



class ChangeDetectionError(Exception):
    """Clean, user-facing error (bad input, incompatible images, ...)."""


# --------------------------------------------------------------------------- #
# Configuration / result containers
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    threshold: int = 30            # 0-255, applied on the absolute difference
    blur_kernel: int = 5           # Gaussian kernel (odd). 0 or 1 disables blur
    morph_kernel: int = 3          # structuring element size. 0 disables morphology
    min_area: int = 50             # contour areas (px) below this are discarded
    resize_tolerance: float = 0.02 # max aspect-ratio mismatch allowed for auto-resize
    force_resize: bool = False     # resize even if aspect ratios differ
    use_grayscale: bool = True     # intensity-based comparison


@dataclass
class DetectionResult:
    difference: np.ndarray         # uint8 gray (raw absdiff, before blur)
    heatmap: np.ndarray            # uint8 BGR
    mask: np.ndarray               # uint8 0/255
    change_map: np.ndarray         # uint8 BGR
    regions: list                  # [{"x","y","w","h","area"}, ...]
    stats: dict
    reference: np.ndarray          # BGR (original, untouched)
    changed: np.ndarray            # BGR (aligned copy used for comparison)
    resized: bool


# --------------------------------------------------------------------------- #
# Loading and validation
# --------------------------------------------------------------------------- #
def load_image(path) -> np.ndarray:
    """Load an image as 3-channel BGR. Raises ChangeDetectionError on any problem."""
    p = Path(path)
    if not p.exists():
        raise ChangeDetectionError(f"File not found: {p}")
    if not p.is_file():
        raise ChangeDetectionError(f"Not a file: {p}")
    if p.suffix.lower() not in SUPPORTED_EXT:
        raise ChangeDetectionError(
            f"Unsupported image format '{p.suffix}' for {p}. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXT))}")
    try:
        # imdecode(fromfile) also handles non-ASCII Windows paths, unlike imread
        data = np.fromfile(str(p), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED) if data.size else None
    except OSError as exc:
        raise ChangeDetectionError(f"Cannot read file {p}: {exc}") from exc
    if img is None:
        raise ChangeDetectionError(f"File could not be decoded as an image (corrupt/empty?): {p}")
    return _to_bgr(img)


def _to_bgr(img: np.ndarray) -> np.ndarray:
    if img.dtype != np.uint8:  # 16-bit etc. -> 8-bit
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:  # drop alpha
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def align_images(reference: np.ndarray, changed: np.ndarray, cfg: Config):
    """
    Make 'changed' the same size as 'reference'. Originals are never modified.
    Policy: resize only when aspect ratios match within cfg.resize_tolerance
    (or --force-resize is given). Resizing uses INTER_AREA when shrinking and
    INTER_LINEAR when enlarging. Returns (changed_aligned, was_resized).
    """
    if reference.shape == changed.shape:
        return changed.copy(), False
    rh, rw = reference.shape[:2]
    ch, cw = changed.shape[:2]
    ar_ref, ar_chg = rw / rh, cw / ch
    mismatch = abs(ar_ref - ar_chg) / ar_ref
    if mismatch > cfg.resize_tolerance and not cfg.force_resize:
        raise ChangeDetectionError(
            f"Incompatible dimensions: reference {rw}x{rh} vs changed {cw}x{ch} "
            f"(aspect-ratio mismatch {mismatch:.1%}). Resizing would distort the scene; "
            f"use --force-resize to override.")
    interp = cv2.INTER_AREA if (cw * ch) > (rw * rh) else cv2.INTER_LINEAR
    return cv2.resize(changed, (rw, rh), interpolation=interp), True


# --------------------------------------------------------------------------- #
# Core pipeline
# --------------------------------------------------------------------------- #
def _odd(k: int) -> int:
    return k if k % 2 == 1 else k + 1


def detect_changes(reference: np.ndarray, changed: np.ndarray, cfg: Config) -> DetectionResult:
    if not 0 <= cfg.threshold <= 255:
        raise ChangeDetectionError(f"threshold must be in 0..255, got {cfg.threshold}")
    ref = reference.copy()
    chg, resized = align_images(ref, changed, cfg)

    if cfg.use_grayscale:
        a = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
        b = cv2.cvtColor(chg, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(a, b)
    else:  # colour: largest per-channel difference
        diff = cv2.absdiff(ref, chg).max(axis=2)

    # Raw difference (saved as-is); blur is applied only to the copy used for thresholding.
    work = diff
    if cfg.blur_kernel and cfg.blur_kernel > 1:
        k = _odd(cfg.blur_kernel)
        work = cv2.GaussianBlur(diff, (k, k), 0)

    # '>' threshold semantics: pixels with value > T become 255.
    _, mask = cv2.threshold(work, cfg.threshold, 255, cv2.THRESH_BINARY)

    if cfg.morph_kernel and cfg.morph_kernel > 0:
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(cfg.morph_kernel),) * 2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se)    # remove specks
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, se)   # fill small holes

    # Remove regions smaller than min_area, collect bounding boxes.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    kept, regions = [], []
    for c in contours:
        area = cv2.contourArea(c)
        if area >= cfg.min_area:
            kept.append(c)
    clean = np.zeros_like(mask)
    cv2.drawContours(clean, kept, -1, 255, thickness=cv2.FILLED)
    mask = clean
    for c in sorted(kept, key=lambda c: (cv2.boundingRect(c)[1], cv2.boundingRect(c)[0])):
        x, y, w, h = cv2.boundingRect(c)
        regions.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h),
                        "area": int(cv2.contourArea(c))})

    heatmap = make_heatmap(work)
    change_map = make_change_map(ref, mask, kept, regions)

    total = int(mask.size)
    changed_px = int(np.count_nonzero(mask))
    stats = {
        "image_width": int(mask.shape[1]),
        "image_height": int(mask.shape[0]),
        "total_pixels": total,
        "changed_pixels": changed_px,
        "changed_area_percent": round(100.0 * changed_px / total, 4),
        "threshold": cfg.threshold,
        "blur_kernel": cfg.blur_kernel,
        "morph_kernel": cfg.morph_kernel,
        "min_area": cfg.min_area,
        "grayscale": cfg.use_grayscale,
        "change_regions": len(regions),
        "changed_image_resized": resized,
        "max_difference": int(diff.max()),
        "mean_difference": round(float(diff.mean()), 4),
    }
    return DetectionResult(diff, heatmap, mask, change_map, regions, stats, ref, chg, resized)


def make_heatmap(diff: np.ndarray) -> np.ndarray:
    """Delta heatmap: difference stretched to 0-255 (fixed scale: 0 = no change) -> JET."""
    return cv2.applyColorMap(diff, cv2.COLORMAP_JET)


def make_change_map(reference, mask, contours, regions) -> np.ndarray:
    """Reference image, dimmed, with changed pixels tinted red, outlined, and boxed/numbered."""
    base = (reference.astype(np.float32) * 0.6).astype(np.uint8)
    red = np.zeros_like(base)
    red[:, :] = (0, 0, 255)
    tinted = cv2.addWeighted(base, 0.55, red, 0.45, 0)
    out = np.where(mask[..., None] > 0, tinted, base)
    cv2.drawContours(out, contours, -1, (0, 255, 255), 2)
    for i, r in enumerate(regions, 1):
        x, y, w, h = r["x"], r["y"], r["w"], r["h"]
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.putText(out, str(i), (x, max(y - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return out


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def _write_png(path: Path, img: np.ndarray):
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ChangeDetectionError(f"Could not encode image for {path}")
    buf.tofile(str(path))  # works with non-ASCII Windows paths


def save_outputs(res: DetectionResult, out_dir, ref_path, chg_path, extra=None) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "reference_preview": out / "reference_preview.png",
        "changed_preview": out / "changed_preview.png",
        "difference": out / "difference.png",
        "delta_heatmap": out / "delta_heatmap.png",
        "change_mask": out / "change_mask.png",
        "change_map": out / "change_map.png",
        "run_summary": out / "run_summary.json",
    }
    _write_png(paths["reference_preview"], res.reference)
    _write_png(paths["changed_preview"], res.changed)
    _write_png(paths["difference"], res.difference)
    _write_png(paths["delta_heatmap"], res.heatmap)
    _write_png(paths["change_mask"], res.mask)
    _write_png(paths["change_map"], res.change_map)
    summary = {
        "task": "Day 13 - Visual Change Detection",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "reference_image": str(ref_path),
        "changed_image": str(chg_path),
        "statistics": res.stats,
        "regions": res.regions,
        "outputs": {k: str(v) for k, v in paths.items()},
    }
    if extra:
        summary.update(extra)
    paths["run_summary"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {k: str(v) for k, v in paths.items()}


def _gui_available() -> bool:
    if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return False
    try:  # Windows OpenCV builds without GUI (opencv-python-headless) raise cv2.error here
        cv2.namedWindow("_probe_", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("_probe_")
        cv2.waitKey(1)
        return True
    except cv2.error:
        return False


def build_comparison_figure(res: DetectionResult, out_dir):
    """6-panel comparison saved as comparison_figure.png. Uses Matplotlib if present, else OpenCV tiles."""
    out_path = Path(out_dir) / "comparison_figure.png"
    try:
        import matplotlib
        matplotlib.use("Agg")  # always render off-screen; no GUI needed
        import matplotlib.pyplot as plt
        rgb = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        panels = [("Reference", rgb(res.reference), None), ("Changed", rgb(res.changed), None),
                  ("Difference", res.difference, "gray"), ("Delta heatmap", rgb(res.heatmap), None),
                  ("Change mask", res.mask, "gray"), ("Change Map", rgb(res.change_map), None)]
        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        for ax, (title, im, cmap) in zip(axes.ravel(), panels):
            ax.imshow(im, cmap=cmap, vmin=0 if cmap else None, vmax=255 if cmap else None)
            ax.set_title(title); ax.axis("off")
        s = res.stats
        fig.suptitle(f"Visual Change Detection | threshold={s['threshold']} | "
                     f"changed={s['changed_area_percent']}% | regions={s['change_regions']}")
        fig.tight_layout()
        fig.savefig(out_path, dpi=100)
        plt.close(fig)
    except ImportError:
        tiles = [res.reference, res.changed, cv2.cvtColor(res.difference, cv2.COLOR_GRAY2BGR),
                 res.heatmap, cv2.cvtColor(res.mask, cv2.COLOR_GRAY2BGR), res.change_map]
        labels = ["Reference", "Changed", "Difference", "Heatmap", "Mask", "Change Map"]
        tiles = [t.copy() for t in tiles]
        for t, l in zip(tiles, labels):
            cv2.putText(t, l, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        grid = np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])])
        _write_png(out_path, grid)
    return str(out_path)


def show_window(fig_path: str) -> bool:
    """Show the comparison figure in an OpenCV window. Returns False if no GUI."""
    if not _gui_available():
        return False
    img = cv2.imdecode(np.fromfile(fig_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    cv2.imshow("Visual Change Detection (press any key to close)", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return True


# --------------------------------------------------------------------------- #
# Synthetic samples (fixed seed -> reproducible)
# --------------------------------------------------------------------------- #
def make_base_scene(width=640, height=480, seed=13) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, width, dtype=np.float32)
    y = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    img = np.zeros((height, width, 3), np.uint8)
    img[..., 0] = (60 + 60 * x[None, :]).astype(np.uint8)
    img[..., 1] = (70 + 60 * y).astype(np.uint8)
    img[..., 2] = (90 + 30 * (x[None, :] + y) / 2).astype(np.uint8)
    cv2.rectangle(img, (40, 40), (180, 140), (200, 200, 200), -1)
    cv2.circle(img, (480, 120), 50, (80, 160, 220), -1)
    cv2.line(img, (0, 400), (width, 330), (30, 30, 30), 6)
    cv2.putText(img, "RE-T1", (250, 260), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (240, 240, 240), 3, cv2.LINE_AA)
    # light static texture so the scene is not perfectly flat (still deterministic)
    img = np.clip(img.astype(np.int16) + rng.integers(-2, 3, img.shape), 0, 255).astype(np.uint8)
    return img


def make_changed_scene(base: np.ndarray, scenario="single") -> np.ndarray:
    """single: one rectangle | multi: rectangle + circle + triangle | identical | noisy"""
    out = base.copy()
    if scenario == "identical":
        return out
    if scenario == "single":
        cv2.rectangle(out, (300, 300), (400, 380), (0, 0, 255), -1)
    elif scenario == "multi":
        cv2.rectangle(out, (300, 300), (400, 380), (0, 0, 255), -1)
        cv2.circle(out, (100, 380), 35, (0, 255, 0), -1)
        cv2.fillPoly(out, [np.array([[540, 300], [600, 300], [570, 250]], np.int32)], (255, 0, 0))
    elif scenario == "noisy":
        cv2.rectangle(out, (300, 300), (400, 380), (0, 0, 255), -1)
        rng = np.random.default_rng(42)
        noise = rng.normal(0, 12, out.shape)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    else:
        raise ValueError(f"unknown scenario {scenario}")
    return out


def generate_samples(samples_dir=None, scenario="single", seed=13):
    d = Path(samples_dir) if samples_dir else (HERE / "input" if (HERE / "input").exists() else HERE / "samples")
    d.mkdir(parents=True, exist_ok=True)
    base = make_base_scene(seed=seed)
    changed = make_changed_scene(base, scenario)
    ref_p, chg_p = d / "reference.png", d / "changed.png"
    _write_png(ref_p, base)
    _write_png(chg_p, changed)
    if not samples_dir and (HERE / "samples").exists() and d != (HERE / "samples"):
        _write_png(HERE / "samples" / "reference.png", base)
        _write_png(HERE / "samples" / "changed.png", changed)
    return str(ref_p), str(chg_p)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Day 13 - Visual Change Detection (offline, classical OpenCV)")
    p.add_argument("--input-dir", default=None, help="directory containing reference.png and changed.png")
    p.add_argument("--reference", default=str(DEFAULT_REFERENCE), help="reference image path")
    p.add_argument("--changed", default=str(DEFAULT_CHANGED), help="changed image path")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT), help="output directory (created if missing)")
    p.add_argument("--threshold", type=int, default=30, help="binary threshold 0-255 (default 30)")
    p.add_argument("--blur", type=int, default=5, help="Gaussian blur kernel, odd; 0/1 = off (default 5)")
    p.add_argument("--morph", type=int, default=3, help="morphology kernel; 0 = off (default 3)")
    p.add_argument("--min-area", type=int, default=50, help="min region area in px (default 50)")
    p.add_argument("--color", action="store_true", help="compare colour channels instead of grayscale")
    p.add_argument("--force-resize", action="store_true", help="resize changed image even if aspect ratio differs")
    p.add_argument("--generate-samples", choices=["single", "multi", "identical", "noisy"],
                   help="(re)generate synthetic samples into the samples folder, then exit unless --run is given")
    p.add_argument("--samples-dir", default=None, help="folder for --generate-samples (default: Day13/samples)")
    p.add_argument("--run", action="store_true", help="with --generate-samples: also run detection afterwards")
    p.add_argument("--headless", action="store_true", help="never open a window (comparison_figure.png is still saved)")
    p.add_argument("--no-show", action="store_true", help="alias of --headless")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        ref_path, chg_path = args.reference, args.changed
        if args.input_dir:
            ref_path = str(Path(args.input_dir) / "reference.png")
            chg_path = str(Path(args.input_dir) / "changed.png")
        if args.generate_samples:
            ref_path, chg_path = generate_samples(args.samples_dir, args.generate_samples)
            print(f"[samples] scenario='{args.generate_samples}' written:\n  {ref_path}\n  {chg_path}")
            if not args.run:
                return 0

        cfg = Config(threshold=args.threshold, blur_kernel=args.blur, morph_kernel=args.morph,
                     min_area=args.min_area, force_resize=args.force_resize, use_grayscale=not args.color)
        reference = load_image(ref_path)
        changed = load_image(chg_path)
        res = detect_changes(reference, changed, cfg)
        paths = save_outputs(res, args.output_dir, ref_path, chg_path)
        fig_path = build_comparison_figure(res, args.output_dir)
        paths["comparison_figure"] = fig_path

        s = res.stats
        print("=" * 60)
        print(" VISUAL CHANGE DETECTION - EXECUTION SUMMARY")
        print("=" * 60)
        print(f" Reference         : {ref_path}")
        print(f" Changed           : {chg_path}  (resized: {s['changed_image_resized']})")
        print(f" Image size        : {s['image_width']} x {s['image_height']}")
        print(f" Threshold         : {s['threshold']}  blur={s['blur_kernel']}  morph={s['morph_kernel']}  min_area={s['min_area']}")
        print(f" Total pixels      : {s['total_pixels']}")
        print(f" Changed pixels    : {s['changed_pixels']}")
        print(f" Changed area      : {s['changed_area_percent']} %")
        print(f" Change regions    : {s['change_regions']}")
        for i, r in enumerate(res.regions, 1):
            print(f"   #{i}: x={r['x']} y={r['y']} w={r['w']} h={r['h']} area={r['area']}")
        print(" Generated files:")
        for k, v in paths.items():
            print(f"   {k:<18}: {v}")

        if args.headless or args.no_show:
            print(" Display           : headless mode (no window)")
        elif show_window(fig_path):
            print(" Display           : window shown and closed")
        else:
            print(" Display           : no GUI available -> skipped (figure saved to file)")
        return 0
    except ChangeDetectionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

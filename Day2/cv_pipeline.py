#!/usr/bin/env python3
"""
CV Starter Pipeline  --  RudraEdge RE-T1 (Spectra)
Author : Aryan
Task   : OpenCV Pipeline Fundamentals (6 Oct 2026)

Pipeline
    Input (image / video / webcam)
      -> Frame acquisition   (cv2.imread / cv2.VideoCapture)
      -> Resize              (cv2.resize)
      -> Grayscale           (cv2.cvtColor)
      -> Gaussian blur       (cv2.GaussianBlur)
      -> Canny edges         (cv2.Canny)
      -> Display / Save      (cv2.imshow / cv2.imwrite / cv2.VideoWriter)

Usage
    python cv_pipeline.py                                   # image mode, input/test.jpg
    python cv_pipeline.py --mode image  --input input/test.jpg
    python cv_pipeline.py --mode video  --input input/test.mp4
    python cv_pipeline.py --mode webcam
    python cv_pipeline.py --make-samples                    # create input/test.jpg + test.mp4

Local only: no cloud, no paid API, no extra hardware.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_IMAGE = BASE_DIR / "input" / "test.jpg"
DEFAULT_INPUT_VIDEO = BASE_DIR / "input" / "test.mp4"
DEFAULT_OUTPUT_DIR = BASE_DIR / "output"

WINDOW_NAME = "CV Starter Pipeline  (press q / ESC to quit)"


# --------------------------------------------------------------------------- #
# Core processing steps (each step is a small, reusable function)
# --------------------------------------------------------------------------- #
def resize_frame(frame, width=640):
    """Resize to a fixed width, keeping the aspect ratio. width<=0 keeps size."""
    if width is None or width <= 0:
        return frame
    h, w = frame.shape[:2]
    if w == width:
        return frame
    scale = width / float(w)
    new_size = (width, max(1, int(round(h * scale))))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(frame, new_size, interpolation=interp)


def to_grayscale(frame):
    """BGR -> single-channel grayscale."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def apply_blur(gray, ksize=5):
    """Gaussian blur. Kernel size must be odd."""
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(gray, (ksize, ksize), 0)


def detect_edges(blurred, low=50, high=150):
    """Canny edge detection on the blurred grayscale image."""
    return cv2.Canny(blurred, low, high)


def process_frame(frame, width=640, ksize=5, low=50, high=150):
    """Run the full preprocessing chain and return every intermediate result."""
    resized = resize_frame(frame, width)
    gray = to_grayscale(resized)
    blurred = apply_blur(gray, ksize)
    edges = detect_edges(blurred, low, high)
    return {"original": resized, "grayscale": gray, "blurred": blurred, "edges": edges}


# --------------------------------------------------------------------------- #
# Output helpers
# --------------------------------------------------------------------------- #
def save_results(results, out_dir):
    """Save original / grayscale / blurred / edges as JPG files."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for name in ("original", "grayscale", "blurred", "edges"):
        path = out_dir / f"{name}.jpg"
        if not cv2.imwrite(str(path), results[name]):
            raise IOError(f"Could not write {path}")
        saved.append(path)
    return saved


def show_window(image, delay_ms=0):
    """Show an image. Returns key pressed, or -1 if no GUI is available (headless)."""
    try:
        cv2.imshow(WINDOW_NAME, image)
        return cv2.waitKey(delay_ms) & 0xFF
    except cv2.error:
        return -1


def make_panel(results):
    """Stack original | grayscale | blurred | edges side by side with labels."""
    tiles = []
    for name in ("original", "grayscale", "blurred", "edges"):
        img = results[name]
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = img.copy()
        cv2.putText(img, name, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255), 2, cv2.LINE_AA)
        tiles.append(img)
    top = np.hstack(tiles[:2])
    bottom = np.hstack(tiles[2:])
    return np.vstack([top, bottom])


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #
def run_image_mode(args):
    path = Path(args.input) if args.input else DEFAULT_INPUT_IMAGE
    if not path.is_file():
        sys.exit(f"[ERROR] Image not found: {path}\n"
                 f"        Put an image in input/ or run:  python cv_pipeline.py --make-samples")

    image = cv2.imread(str(path))              # returns None if unreadable
    if image is None:
        sys.exit(f"[ERROR] Could not decode image (corrupt / unsupported): {path}")
    print(f"[INFO] Loaded {path.name}  shape={image.shape}")

    results = process_frame(image, args.width, args.blur, args.canny_low, args.canny_high)
    saved = save_results(results, args.output_dir)
    for p in saved:
        print(f"[OK]   saved {p}")

    if not args.no_display:
        key = show_window(make_panel(results), 0)
        if key == -1:
            print("[INFO] No display available, results were saved to disk only.")
        cv2.destroyAllWindows()


def run_stream_mode(args, source, label):
    """Frame-by-frame processing for a video file or webcam."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Could not open {label}: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = None
    first_saved = False
    frame_count = 0
    display_ok = not args.no_display

    print(f"[INFO] Reading {label} ... (q / ESC to quit)")
    while True:
        ret, frame = cap.read()                # ret=False -> end of stream / error
        if not ret:
            break
        frame_count += 1

        results = process_frame(frame, args.width, args.blur, args.canny_low, args.canny_high)

        if not first_saved:                    # keep a snapshot of the first frame
            save_results(results, out_dir)
            first_saved = True

        if not args.no_save_video and label != "webcam":
            if writer is None:
                h, w = results["edges"].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(out_dir / "processed_video.mp4"),
                                         fourcc, fps, (w, h), isColor=False)
            writer.write(results["edges"])

        if display_ok:
            key = show_window(make_panel(results), 1)
            if key == -1:
                display_ok = False
                print("[INFO] No display available, running headless.")
            elif key in (ord("q"), 27):
                break

        if args.max_frames and frame_count >= args.max_frames:
            break

    cap.release()
    if writer is not None:
        writer.release()
        print(f"[OK]   saved {out_dir / 'processed_video.mp4'}")
    cv2.destroyAllWindows()
    print(f"[DONE] Processed {frame_count} frame(s) from {label}.")


# --------------------------------------------------------------------------- #
# Sample data generator (so the pipeline is runnable with zero downloads)
# --------------------------------------------------------------------------- #
def _draw_scene(canvas, t=0.0):
    """Draw simple shapes + text. t moves the shapes so video has motion."""
    h, w = canvas.shape[:2]
    cv2.rectangle(canvas, (40 + int(60 * t), 60), (220 + int(60 * t), 200), (70, 210, 90), -1)
    cv2.circle(canvas, (w - 160 - int(50 * t), 150), 70, (60, 120, 245), -1)
    cv2.line(canvas, (0, h - 120), (w, h - 40 - int(40 * t)), (240, 240, 240), 4)
    pts = np.array([[300, 330], [420, 240 + int(30 * t)], [540, 330]], np.int32)
    cv2.fillPoly(canvas, [pts], (240, 200, 60))
    cv2.putText(canvas, "RudraEdge RE-T1", (40, h - 30), cv2.FONT_HERSHEY_SIMPLEX,
                1.1, (255, 255, 255), 2, cv2.LINE_AA)


def make_samples(input_dir=None):
    input_dir = Path(input_dir) if input_dir else BASE_DIR / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    w, h = 640, 480

    gradient = np.tile(np.linspace(15, 70, w, dtype=np.uint8), (h, 1))
    img = cv2.merge([gradient + 20, gradient // 2, gradient // 2])
    _draw_scene(img, 0.0)
    rng = np.random.default_rng(0)
    img = cv2.add(img, rng.integers(0, 12, img.shape, dtype=np.uint8))
    cv2.imwrite(str(input_dir / "test.jpg"), img)

    writer = cv2.VideoWriter(str(input_dir / "test.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (w, h))
    for i in range(60):
        frame = cv2.merge([gradient + 20, gradient // 2, gradient // 2])
        _draw_scene(frame, i / 59.0)
        writer.write(frame)
    writer.release()
    print(f"[OK] Sample files created in {input_dir}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser():
    p = argparse.ArgumentParser(description="OpenCV CV Starter Pipeline (image / video / webcam)")
    p.add_argument("--mode", choices=["image", "video", "webcam"], default="image",
                   help="input type (default: image)")
    p.add_argument("--input", help="path to image/video (default: input/test.jpg or input/test.mp4)")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="where results are saved")
    p.add_argument("--width", type=int, default=640, help="resize width, 0 = keep original")
    p.add_argument("--blur", type=int, default=5, help="Gaussian kernel size (odd number)")
    p.add_argument("--canny-low", type=int, default=50, help="Canny lower threshold")
    p.add_argument("--canny-high", type=int, default=150, help="Canny upper threshold")
    p.add_argument("--no-display", action="store_true", help="do not open any window")
    p.add_argument("--no-save-video", action="store_true",
                   help="do not write output/processed_video.mp4 (video mode)")
    p.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = all)")
    p.add_argument("--make-samples", action="store_true", help="create input/test.jpg and test.mp4")
    return p


def main():
    args = build_parser().parse_args()

    if args.make_samples:
        make_samples()
        return

    if args.mode == "image":
        run_image_mode(args)
    elif args.mode == "video":
        src = Path(args.input) if args.input else DEFAULT_INPUT_VIDEO
        if not src.is_file():
            sys.exit(f"[ERROR] Video not found: {src}")
        run_stream_mode(args, str(src), f"video ({src.name})")
    else:
        run_stream_mode(args, 0, "webcam")


if __name__ == "__main__":
    main()

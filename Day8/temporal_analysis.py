#!/usr/bin/env python3
"""Day 12 - Temporal Image Analysis (frame differencing / change detection).

Compares consecutive frames from a webcam or video file using classical
computer vision only: grayscale -> Gaussian blur -> cv2.absdiff -> cv2.threshold
-> morphology -> contours. The output is *change / potential motion detection*;
it does NOT identify what kind of object moved.
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

WINDOW_FRAME = "Day8 - Detected Changes"
WINDOW_MASK = "Day8 - Change Mask"
WINDOW_DIFF = "Day8 - Difference"
WINDOW_GRID = "Day8 - Motion Analysis Grid"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Temporal image analysis using frame differencing.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--webcam", type=int, nargs="?", const=0, metavar="INDEX",
                     help="Use webcam (default index 0).")
    src.add_argument("--video", type=str, metavar="PATH", help="Path to a local video file.")
    p.add_argument("--threshold", type=int, default=25, help="Pixel-difference threshold 0-255 (default 25).")
    p.add_argument("--min-area", type=int, default=500, help="Minimum changed-region area in px (default 500).")
    p.add_argument("--blur", type=int, default=5, help="Gaussian blur kernel size, odd number; 0 disables (default 5).")
    p.add_argument("--output-dir", type=str, default="output", help="Base output directory (default ./output).")
    p.add_argument("--save-video", action="store_true", help="Save processed video (detected_output.mp4).")
    p.add_argument("--max-frames", type=int, default=0, help="Stop after N frames; 0 = no limit "
                   "(webcam with --save-video defaults to 600 frames).")
    p.add_argument("--no-gui", action="store_true", help="Headless mode: no windows (for servers/tests).")
    p.add_argument("--separate-windows", action="store_true",
                   help="Show 3 separate floating windows instead of single unified 2x2 grid.")
    p.add_argument("--motion-hold", type=int, default=8,
                   help="Frames to hold 'MOTION DETECTED' status to prevent flickering (default 8).")
    return p.parse_args(argv)


def open_source(args):
    """Open webcam or video file; raise RuntimeError with a clear message on failure."""
    if args.video is not None:
        path = Path(args.video)
        if not path.is_file():
            raise RuntimeError(f"Video file not found: {path}")
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video (unsupported or corrupt file?): {path}")
        return cap, f"video:{path}"
    cap = cv2.VideoCapture(args.webcam)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open webcam index {args.webcam}. Check that it is connected, "
                           "not used by another app, and that camera permission is granted.")
    return cap, f"webcam:{args.webcam}"


def preprocess(frame, blur):
    """BGR frame -> blurred grayscale."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if blur and blur > 1:
        k = blur if blur % 2 == 1 else blur + 1
        gray = cv2.GaussianBlur(gray, (k, k), 0)
    return gray


def detect_change(prev_gray, curr_gray, threshold, min_area):
    """Return (diff, mask, boxes, changed_pixels) for two grayscale frames."""
    diff = cv2.absdiff(prev_gray, curr_gray)
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)    # remove speckle noise
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.dilate(mask, dilate_kernel, iterations=2)     # merge nearby fragments
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= min_area]
    changed = int(cv2.countNonZero(mask))
    if not boxes and changed >= min_area:
        sub_boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= max(80, min_area * 0.35)]
        if sub_boxes:
            boxes = sub_boxes
    return diff, mask, boxes, changed


def annotate(frame, boxes, frame_idx, fps, moving=None):
    out = frame.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
    if moving is None:
        moving = len(boxes) > 0
    label = "MOTION DETECTED" if moving else "NO MOTION"
    color = (0, 0, 255) if moving else (0, 200, 0)
    cv2.putText(out, label, (12, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
    cv2.putText(out, f"Frame {frame_idx}  FPS {fps:.1f}  Regions {len(boxes)}", (12, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out, moving


def make_tile(img, title, tile_size):
    tw, th = tile_size
    if len(img.shape) == 2:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_bgr = img
    resized = cv2.resize(img_bgr, (tw, th))
    header_h = 28
    cv2.rectangle(resized, (0, 0), (tw, header_h), (25, 25, 25), -1)
    cv2.putText(resized, title, (10, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (225, 225, 225), 1, cv2.LINE_AA)
    return resized


def make_dashboard_tile(tile_size, moving, changed_pixels, num_boxes, fps, idx, motion_frames, threshold, min_area):
    tw, th = tile_size
    dash = np.zeros((th, tw, 3), dtype=np.uint8)
    dash[:] = (26, 26, 30)

    header_h = 28
    cv2.rectangle(dash, (0, 0), (tw, header_h), (38, 40, 46), -1)
    cv2.putText(dash, "4. Real-time Telemetry & Status", (10, 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (225, 225, 225), 1, cv2.LINE_AA)

    # Status Badge
    badge_top = 40
    badge_bottom = 76
    badge_color = (0, 0, 210) if moving else (0, 160, 40)
    status_text = "STATUS: MOTION DETECTED" if moving else "STATUS: NO MOTION"
    cv2.rectangle(dash, (12, badge_top), (tw - 12, badge_bottom), badge_color, -1)
    cv2.putText(dash, status_text, (22, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    stats = [
        f"Active Regions : {num_boxes}",
        f"Changed Pixels : {changed_pixels:,}",
        f"Live FPS       : {fps:.1f}",
        f"Frame Progress : {idx} (Motion: {motion_frames})",
        f"Config         : Threshold={threshold}, MinArea={min_area}px",
    ]

    curr_y = 104
    line_step = 23
    for s in stats:
        if curr_y + 10 < th - 20:
            cv2.putText(dash, s, (16, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (205, 215, 225), 1, cv2.LINE_AA)
            curr_y += line_step

    cv2.putText(dash, "Press 'q' or 'ESC' to exit", (16, th - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 130, 140), 1, cv2.LINE_AA)
    return dash


def build_grid_display(shown, diff, mask, idx, fps, changed_pixels, boxes, args, motion_frames, moving=None):
    h, w = shown.shape[:2]
    tile_w = 480
    tile_h = max(240, int(tile_w * h / w))
    tile_size = (tile_w, tile_h)

    tile1 = make_tile(shown, "1. Live Camera / Annotated Feed", tile_size)
    tile2 = make_tile(diff, "2. Absolute Difference (absdiff)", tile_size)
    tile3 = make_tile(mask, "3. Cleaned Change Mask (Threshold+Morph)", tile_size)
    if moving is None:
        moving = len(boxes) > 0
    tile4 = make_dashboard_tile(tile_size, moving, changed_pixels, len(boxes),
                                fps, idx, motion_frames, args.threshold, args.min_area)

    row1 = np.hstack([tile1, tile2])
    row2 = np.hstack([tile3, tile4])
    return np.vstack([row1, row2])


def make_run_dir(base):
    run_dir = Path(base) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    suffix = 1
    while run_dir.exists():          # never overwrite a previous run
        run_dir = Path(base) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True)
    return run_dir


def run(args):
    if not 0 <= args.threshold <= 255:
        raise RuntimeError("--threshold must be between 0 and 255.")
    if args.min_area < 0:
        raise RuntimeError("--min-area must be >= 0.")

    cap, source = open_source(args)
    run_dir = make_run_dir(args.output_dir)
    max_frames = args.max_frames
    if args.webcam is not None and args.save_video and max_frames == 0:
        max_frames = 600             # don't record a webcam indefinitely by default

    writer = None
    summary = {"source": source,
               "config": {"threshold": args.threshold, "min_area": args.min_area, "blur": args.blur,
                          "save_video": args.save_video, "gui": not args.no_gui, "max_frames": max_frames},
               "started": datetime.now().isoformat(timespec="seconds"),
               "frames_read": 0, "pairs_compared": 0, "motion_frames": 0,
               "status": "running", "video_saved": False}
    best = None                      # representative pair: the one with most changed pixels
    last = None
    prev_gray = None
    t_prev = time.time()
    fps = 0.0
    t_start = t_prev
    idx = 0
    motion_hold = max(1, getattr(args, "motion_hold", 8))
    hold_counter = 0

    if not args.no_gui and not args.separate_windows:
        cv2.namedWindow(WINDOW_GRID, cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                summary["status"] = "completed" if idx > 0 else "failed: no frames could be read"
                break
            idx += 1
            gray = preprocess(frame, args.blur)
            if prev_gray is not None and prev_gray.shape == gray.shape:
                diff, mask, boxes, changed = detect_change(prev_gray, gray, args.threshold, args.min_area)
                now = time.time()
                fps = 1.0 / max(now - t_prev, 1e-6)
                t_prev = now

                # Motion smoothing hold buffer: keeps MOTION DETECTED stable across frame gaps
                if len(boxes) > 0:
                    hold_counter = motion_hold
                elif hold_counter > 0:
                    hold_counter -= 1
                moving = (hold_counter > 0)

                shown, _ = annotate(frame, boxes, idx, fps, moving)
                summary["pairs_compared"] += 1
                summary["motion_frames"] += int(moving)
                last = (diff, mask, changed)
                if best is None or changed > best[2]:
                    best = (diff, mask, changed)

                if args.save_video:
                    if writer is None:
                        h, w = frame.shape[:2]
                        src_fps = cap.get(cv2.CAP_PROP_FPS)
                        src_fps = src_fps if src_fps and src_fps > 1 else 20.0
                        writer = cv2.VideoWriter(str(run_dir / "detected_output.mp4"),
                                                 cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (w, h))
                        if not writer.isOpened():
                            print("Warning: could not create video writer; continuing without recording.")
                            writer, args.save_video = None, False
                    if writer is not None:
                        writer.write(shown)

                if not args.no_gui:
                    if args.separate_windows:
                        cv2.imshow(WINDOW_FRAME, shown)
                        cv2.imshow(WINDOW_MASK, mask)
                        cv2.imshow(WINDOW_DIFF, diff)
                    else:
                        grid = build_grid_display(shown, diff, mask, idx, fps, changed, boxes, args, summary["motion_frames"], moving)
                        cv2.imshow(WINDOW_GRID, grid)
            elif not args.no_gui:
                if args.separate_windows:
                    cv2.imshow(WINDOW_FRAME, frame)
                else:
                    blank = np.zeros(frame.shape[:2], dtype=np.uint8)
                    grid = build_grid_display(frame, blank, blank, idx, 0.0, 0, [], args, 0, False)
                    cv2.imshow(WINDOW_GRID, grid)

            prev_gray = gray
            if not args.no_gui:
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    summary["status"] = "stopped by user (q)"
                    break
            if max_frames and idx >= max_frames:
                summary["status"] = "completed (max-frames reached)"
                break
    except KeyboardInterrupt:
        summary["status"] = "stopped by user (Ctrl+C)"
    finally:
        cap.release()
        if writer is not None:
            writer.release()
            summary["video_saved"] = True
        if not args.no_gui:
            cv2.destroyAllWindows()

    summary["frames_read"] = idx
    summary["elapsed_seconds"] = round(time.time() - t_start, 3)
    rep = best if (best and best[2] > 0) else last
    if rep is not None:
        cv2.imwrite(str(run_dir / "difference.png"), rep[0])
        cv2.imwrite(str(run_dir / "change_mask.png"), rep[1])
        summary["representative_changed_pixels"] = rep[2]
    summary["output_dir"] = str(run_dir)
    summary["finished"] = datetime.now().isoformat(timespec="seconds")
    with open(run_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Done. status={summary['status']} frames={idx} motion_frames={summary['motion_frames']}")
    print(f"Results saved in: {run_dir}")
    return summary


def main(argv=None):
    args = parse_args(argv)
    try:
        run(args)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

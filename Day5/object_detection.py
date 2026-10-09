#!/usr/bin/env python3
"""
Day09 - Object Detection Pipeline (RudraEdge RE-T1 / Spectra)

Route chosen: OpenCV DNN backend with YOLOv8n ONNX model (no PyTorch needed).

Workflow:
    validate image -> load ONNX model via cv2.dnn -> run inference -> NMS ->
    draw boxes -> save image + JSON report

Usage:
    python object_detection.py --image images/test.jpg --confidence 0.25

Exit codes:
    0 = success (including "zero detections")
    2 = input / argument problem (missing, unreadable, corrupt image, bad threshold)
    3 = model problem (ONNX file missing or could not be loaded)
    4 = unexpected error during inference or saving
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import urllib.request
from dataclasses import dataclass, asdict
from datetime import datetime
from importlib import metadata
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGE = BASE_DIR / "images" / "test.jpg"
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs"
DEFAULT_MODEL = str(BASE_DIR / "yolov8n.onnx")
SAMPLE_URL = "https://ultralytics.com/images/bus.jpg"  # public sample image used by Ultralytics docs
ONNX_MODEL_URL = "https://huggingface.co/Xuban/yolo_weights_database/resolve/main/yolov8n.onnx"

EXIT_OK, EXIT_INPUT, EXIT_MODEL, EXIT_RUNTIME = 0, 2, 3, 4

# COCO class names (80 classes) — YOLOv8n is pretrained on COCO
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]

INPUT_SIZE = 640  # YOLOv8n default input size


# ----------------------------------------------------------------------------
# Errors (each maps to an exit code)
# ----------------------------------------------------------------------------
class InputError(Exception):
    """Problem with the user's input (path, image file, arguments)."""


class ModelError(Exception):
    """Problem importing, downloading or loading the detector."""


# ----------------------------------------------------------------------------
# Data structure for one detection
# ----------------------------------------------------------------------------
@dataclass
class Detection:
    class_id: int
    label: str
    confidence: float
    x1: float  # bounding box: top-left corner (pixels)
    y1: float
    x2: float  # bounding box: bottom-right corner (pixels)
    y2: float


# ----------------------------------------------------------------------------
# Step 1: input validation
# ----------------------------------------------------------------------------
def validate_confidence(value: float) -> float:
    if not (0.0 <= value <= 1.0):
        raise InputError(f"Confidence threshold must be between 0 and 1, got {value}.")
    return value


def load_image(path: str | Path) -> np.ndarray:
    """Check that the file exists and decodes into an image; return BGR array."""
    p = Path(path)
    if not p.exists():
        raise InputError(f"Image not found: '{p}'. Check the path or put a photo at images/test.jpg.")
    if not p.is_file():
        raise InputError(f"Path is not a file: '{p}'.")
    try:
        data = p.read_bytes()
    except OSError as exc:
        raise InputError(f"Cannot read '{p}': {exc}") from exc
    if not data:
        raise InputError(f"Image file is empty: '{p}'.")
    # imdecode (instead of imread) also works with non-ASCII paths on Windows
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise InputError(f"Could not decode '{p}' as an image (corrupt or unsupported format).")
    return image


# ----------------------------------------------------------------------------
# Step 2: model loading + inference (OpenCV DNN)
# ----------------------------------------------------------------------------
def load_model(model_path: str):
    """Load a YOLOv8 ONNX model via OpenCV DNN."""
    p = Path(model_path)
    if not p.exists():
        raise ModelError(
            f"ONNX model not found at '{p}'.\n"
            f"  Fix: Download the model first:\n"
            f"  python -c \"import urllib.request; urllib.request.urlretrieve('{ONNX_MODEL_URL}', '{model_path}')\""
        )
    try:
        net = cv2.dnn.readNetFromONNX(str(p))
    except Exception as exc:
        raise ModelError(f"Could not load ONNX model '{p}': {exc}") from exc
    return net


def preprocess(image: np.ndarray, input_size: int = INPUT_SIZE):
    """Prepare image for YOLOv8: resize with letterbox, normalize, NCHW blob."""
    h, w = image.shape[:2]
    scale = min(input_size / h, input_size / w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Letterbox padding (center the image)
    pad_w = (input_size - new_w) // 2
    pad_h = (input_size - new_h) // 2
    padded = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

    # HWC BGR -> NCHW RGB, float32, 0-1
    blob = cv2.dnn.blobFromImage(padded, scalefactor=1.0 / 255.0, swapRB=True)
    return blob, scale, pad_w, pad_h


def postprocess(output: np.ndarray, confidence: float, orig_h: int, orig_w: int,
                scale: float, pad_w: int, pad_h: int) -> list[Detection]:
    """Parse YOLOv8 raw output [1, 84, 8400] -> list[Detection] after NMS."""
    # output shape: (1, 84, 8400) -> transpose to (8400, 84)
    predictions = output[0].T  # (8400, 84)

    # Extract boxes (cx, cy, w, h) and class scores
    boxes_cxcywh = predictions[:, :4]
    class_scores = predictions[:, 4:]  # (8400, 80)

    # Get best class per detection
    class_ids = np.argmax(class_scores, axis=1)
    confidences = class_scores[np.arange(len(class_ids)), class_ids]

    # Filter by confidence
    mask = confidences >= confidence
    boxes_cxcywh = boxes_cxcywh[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]

    if len(boxes_cxcywh) == 0:
        return []

    # Convert cx,cy,w,h -> x1,y1,w,h (OpenCV NMS expects this)
    boxes_xywh = np.zeros_like(boxes_cxcywh)
    boxes_xywh[:, 0] = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2  # x1
    boxes_xywh[:, 1] = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2  # y1
    boxes_xywh[:, 2] = boxes_cxcywh[:, 2]  # w
    boxes_xywh[:, 3] = boxes_cxcywh[:, 3]  # h

    # NMS
    indices = cv2.dnn.NMSBoxes(
        boxes_xywh.tolist(), confidences.tolist(),
        score_threshold=confidence, nms_threshold=0.45
    )

    detections: list[Detection] = []
    if len(indices) > 0:
        for i in indices.flatten():
            x1 = (boxes_xywh[i, 0] - pad_w) / scale
            y1 = (boxes_xywh[i, 1] - pad_h) / scale
            x2 = (boxes_xywh[i, 0] + boxes_xywh[i, 2] - pad_w) / scale
            y2 = (boxes_xywh[i, 1] + boxes_xywh[i, 3] - pad_h) / scale

            # Clip to image bounds
            x1 = max(0, min(x1, orig_w))
            y1 = max(0, min(y1, orig_h))
            x2 = max(0, min(x2, orig_w))
            y2 = max(0, min(y2, orig_h))

            cls_id = int(class_ids[i])
            label = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else f"class_{cls_id}"
            detections.append(Detection(cls_id, label, float(confidences[i]),
                                        float(x1), float(y1), float(x2), float(y2)))

    return detections


def run_inference(net, image: np.ndarray, confidence: float, device: str):
    """Run the detector via OpenCV DNN. Returns (list[Detection], inference_time_ms)."""
    h, w = image.shape[:2]
    blob, scale, pad_w, pad_h = preprocess(image)

    start = time.perf_counter()
    net.setInput(blob)
    output = net.forward()
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    detections = postprocess(output, confidence, h, w, scale, pad_w, pad_h)
    return detections, elapsed_ms


# ----------------------------------------------------------------------------
# Step 3: drawing + saving
# ----------------------------------------------------------------------------
def draw_detections(image: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Return a copy of the image with boxes and 'label confidence' text drawn."""
    out = image.copy()
    thickness = max(2, round(min(out.shape[:2]) / 300))
    font_scale = max(0.5, min(out.shape[:2]) / 900)
    for d in detections:
        # deterministic colour per class so the same class always looks the same
        color = tuple(int(c) for c in np.random.default_rng(d.class_id).integers(60, 255, 3))
        p1, p2 = (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2))
        cv2.rectangle(out, p1, p2, color, thickness)
        text = f"{d.label} {d.confidence:.2f}"
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        top = max(p1[1] - th - base - 4, 0)
        cv2.rectangle(out, (p1[0], top), (p1[0] + tw + 6, top + th + base + 4), color, -1)
        cv2.putText(out, text, (p1[0] + 3, top + th + 1), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def save_image(image: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Failed to write image to '{path}'.")


# ----------------------------------------------------------------------------
# Step 4: run report (JSON)
# ----------------------------------------------------------------------------
def package_versions() -> dict:
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for pkg in ("opencv-python", "numpy"):
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = "not installed"
    versions["backend"] = "OpenCV DNN (ONNX)"
    return versions


def build_report(input_path, model_name, confidence, device, detections,
                 inference_ms, output_path) -> dict:
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "input_path": str(input_path),
        "model": model_name,
        "device": device,
        "confidence_threshold": confidence,
        "inference_time_ms": round(inference_ms, 2) if inference_ms is not None else None,
        "num_detections": len(detections),
        "class_labels": [d.label for d in detections],
        "confidence_scores": [round(d.confidence, 4) for d in detections],
        "detections": [asdict(d) for d in detections],
        "output_image": str(output_path),
        "versions": package_versions(),
    }


# ----------------------------------------------------------------------------
# Video processing
# ----------------------------------------------------------------------------
def process_video(video_path: str, net, confidence: float, device: str, out_dir: Path) -> int:
    """Process a video file: detect objects frame-by-frame, save annotated output video."""
    vp = Path(video_path)
    if not vp.exists():
        raise InputError(f"Video not found: '{vp}'. Check the path.")
    if not vp.is_file():
        raise InputError(f"Path is not a file: '{vp}'.")

    cap = cv2.VideoCapture(str(vp))
    if not cap.isOpened():
        raise InputError(f"Could not open video: '{vp}' (corrupt or unsupported format).")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[1/3] Video loaded: {vp.name} ({width}x{height}, {fps:.1f} FPS, {total_frames} frames)")

    # Output video path
    out_dir.mkdir(parents=True, exist_ok=True)
    out_video = out_dir / f"detected_{vp.stem}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video writer for '{out_video}'.")

    print(f"[2/3] Running detection on each frame...")
    frame_idx = 0
    total_detections = 0
    total_inference_ms = 0.0
    all_labels = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        detections, ms = run_inference(net, frame, confidence, device)
        total_detections += len(detections)
        total_inference_ms += ms
        for d in detections:
            all_labels.add(d.label)

        annotated = draw_detections(frame, detections)
        writer.write(annotated)

        # Progress update every 30 frames
        if frame_idx % 30 == 0 or frame_idx == total_frames:
            avg_fps = 1000.0 / (total_inference_ms / frame_idx) if total_inference_ms > 0 else 0
            print(f"      Frame {frame_idx}/{total_frames} | "
                  f"{len(detections)} det | "
                  f"Avg: {total_inference_ms / frame_idx:.1f} ms/frame ({avg_fps:.1f} FPS)")

    cap.release()
    writer.release()

    avg_ms = total_inference_ms / frame_idx if frame_idx > 0 else 0
    avg_fps = 1000.0 / avg_ms if avg_ms > 0 else 0
    print(f"\n[3/3] Done! Summary:")
    print(f"      Frames processed: {frame_idx}")
    print(f"      Total detections: {total_detections}")
    print(f"      Unique classes:   {sorted(all_labels) if all_labels else 'none'}")
    print(f"      Avg inference:    {avg_ms:.1f} ms/frame ({avg_fps:.1f} FPS)")
    print(f"      Output saved:     {out_video}")

    # Save video report
    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "input_video": str(vp),
        "model": str(net),
        "device": device,
        "confidence_threshold": confidence,
        "video_info": {"width": width, "height": height, "fps": fps, "total_frames": total_frames},
        "frames_processed": frame_idx,
        "total_detections": total_detections,
        "unique_classes": sorted(all_labels),
        "avg_inference_ms": round(avg_ms, 2),
        "avg_fps": round(avg_fps, 2),
        "output_video": str(out_video),
        "versions": package_versions(),
    }
    report_path = out_dir / f"video_report_{vp.stem}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"      Report saved:     {report_path}")
    return EXIT_OK


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def fetch_sample(dest: Path) -> None:
    """Download a public sample image (free). Only used when --fetch-sample is passed."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(SAMPLE_URL, dest)
    except Exception as exc:
        raise InputError(f"Could not download sample image ({exc}). Copy any photo to '{dest}' manually.") from exc
    print(f"Sample image saved to {dest}")


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Local object detection demo (YOLOv8n ONNX, CPU, OpenCV DNN).")
    ap.add_argument("--image", default=str(DEFAULT_IMAGE), help="path to input image (default: images/test.jpg)")
    ap.add_argument("--video", default=None, help="path to input video (use this instead of --image for video)")
    ap.add_argument("--confidence", type=float, default=0.25, help="confidence threshold 0-1 (default 0.25)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="path to ONNX model (default yolov8n.onnx)")
    ap.add_argument("--device", default="cpu", help="inference device (default cpu)")
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="where to save outputs")
    ap.add_argument("--fetch-sample", action="store_true",
                    help="download a public sample photo to the --image path, then run")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.output_dir)
    try:
        validate_confidence(args.confidence)
        net = load_model(args.model)

        # ---- VIDEO MODE ----
        if args.video:
            print(f"=== VIDEO MODE ===")
            print(f"Model loaded: {args.model} (OpenCV DNN, {args.device})")
            return process_video(args.video, net, args.confidence, args.device, out_dir)

        # ---- IMAGE MODE ----
        if args.fetch_sample:
            fetch_sample(Path(args.image))
        image = load_image(args.image)                                   # AC-01
        print(f"[1/4] Image loaded: {args.image} ({image.shape[1]}x{image.shape[0]})")
        print(f"[2/4] Model loaded: {args.model} (OpenCV DNN, {args.device})")

        detections, ms = run_inference(net, image, args.confidence, args.device)   # AC-02, AC-05
        print(f"[3/4] Inference done in {ms:.1f} ms, {len(detections)} detection(s)")
        if not detections:
            print("      No objects found above the threshold. Saving the image without boxes.")
        for d in detections:
            print(f"      - {d.label}: {d.confidence:.2f}  box=({d.x1:.0f},{d.y1:.0f},{d.x2:.0f},{d.y2:.0f})")

        out_img = out_dir / "detected.jpg"
        save_image(draw_detections(image, detections), out_img)          # AC-03
        report = build_report(args.image, args.model, args.confidence, args.device, detections, ms, out_img)
        (out_dir / "run_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[4/4] Saved: {out_img} and {out_dir / 'run_report.json'}")
        return EXIT_OK
    except InputError as exc:
        print(f"INPUT ERROR: {exc}", file=sys.stderr)
        return EXIT_INPUT
    except ModelError as exc:
        print(f"MODEL ERROR: {exc}", file=sys.stderr)
        return EXIT_MODEL
    except Exception as exc:
        print(f"RUNTIME ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_RUNTIME


if __name__ == "__main__":
    sys.exit(main())

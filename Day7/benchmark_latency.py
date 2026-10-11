#!/usr/bin/env python3
"""Inference latency benchmark for RudraEdge RE-T1 / Spectra (Day 11).

Reports three things separately:
  1. model load time
  2. model-only inference latency (tensor in -> raw output out)
  3. end-to-end latency (image preprocessing + inference + postprocessing)

Backends: onnx (.onnx via onnxruntime), torchscript (.pt/.pth via torch.jit),
dummy (NO real model; only for checking the pipeline itself - never real performance).
"""
import argparse, csv, json, math, os, platform, sys, time
from datetime import datetime, timezone

import numpy as np

TIMING_SCOPES = {
    "model_only": "forward pass only: preprocessed tensor -> raw model output",
    "end_to_end": "resize+normalize -> forward pass -> argmax mask (image already decoded in memory)",
}


# ---------------------------------------------------------------- metrics
def percentile(values, q):
    """Linear-interpolation percentile (same as numpy default), q in [0, 100]."""
    if not values:
        raise ValueError("no values")
    s = sorted(values)
    pos = (len(s) - 1) * q / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def compute_summary(latencies_ms):
    """Aggregate stats (ms). Throughput = 1000/mean for batch size 1."""
    if not latencies_ms:
        raise ValueError("latencies_ms is empty")
    if any((not isinstance(v, (int, float))) or not math.isfinite(v) or v < 0 for v in latencies_ms):
        raise ValueError("latencies must be finite, non-negative numbers")
    n = len(latencies_ms)
    mean = sum(latencies_ms) / n
    return {
        "count": n,
        "mean_ms": mean,
        "median_ms": percentile(latencies_ms, 50),
        "min_ms": min(latencies_ms),
        "max_ms": max(latencies_ms),
        "p95_ms": percentile(latencies_ms, 95),
        "throughput_img_per_s": (1000.0 / mean) if mean > 0 else None,
    }


# ---------------------------------------------------------------- backends
class DummyBackend:
    name = "dummy_no_model"
    device = "cpu"

    def __init__(self, size, classes=21):
        self.size, self.classes = size, classes

    def run(self, x):  # cheap fake convolution-ish work; NOT a real model
        w = np.ones((x.shape[1], self.classes), dtype=np.float32)
        return np.einsum("bchw,ck->bkhw", x, w)

    def sync(self):
        pass


class OnnxBackend:
    def __init__(self, path, device):
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise RuntimeError(f"onnxruntime not installed: {e}")
        avail = ort.get_available_providers()
        want = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        providers = [p for p in want if p in avail]
        if device == "cuda" and "CUDAExecutionProvider" not in avail:
            print("WARNING: CUDA requested but unavailable; using CPU", file=sys.stderr)
        self.sess = ort.InferenceSession(path, providers=providers)
        self.inp = self.sess.get_inputs()[0]
        self.name = os.path.basename(path)
        self.device = "cuda" if providers[0] == "CUDAExecutionProvider" else "cpu"

    def run(self, x):
        return self.sess.run(None, {self.inp.name: x})[0]

    def sync(self):
        pass  # onnxruntime run() is synchronous


class TorchScriptBackend:
    def __init__(self, path, device):
        try:
            import torch
        except ImportError as e:
            raise RuntimeError(f"torch not installed: {e}")
        self.torch = torch
        if device == "cuda" and not torch.cuda.is_available():
            print("WARNING: CUDA requested but unavailable; using CPU", file=sys.stderr)
            device = "cpu"
        self.device = device
        self.model = torch.jit.load(path, map_location=device).eval()
        self.name = os.path.basename(path)

    def run(self, x):
        with self.torch.inference_mode():
            out = self.model(self.torch.from_numpy(x).to(self.device))
        if isinstance(out, dict):
            out = out.get("out", next(iter(out.values())))
        return out

    def sync(self):
        if self.device == "cuda":
            self.torch.cuda.synchronize()


def build_backend(args):
    if args.backend == "auto":
        if not args.model:
            raise ValueError("--model is required (or use --backend dummy for a pipeline self-check)")
        ext = os.path.splitext(args.model)[1].lower()
        args.backend = {".onnx": "onnx", ".pt": "torchscript", ".pth": "torchscript"}.get(ext)
        if args.backend is None:
            raise ValueError(f"cannot infer backend from extension '{ext}'; pass --backend")
    if args.backend == "dummy":
        return DummyBackend(args.input_size)
    if not args.model or not os.path.isfile(args.model):
        raise FileNotFoundError(f"model weights not found: {args.model}")
    return (OnnxBackend if args.backend == "onnx" else TorchScriptBackend)(args.model, args.device)


# ---------------------------------------------------------------- pipeline
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".m4v", ".wmv"}


def load_input(path):
    import cv2
    if not os.path.isfile(path):
        raise FileNotFoundError(f"input file not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext in VIDEO_EXTENSIONS:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError(f"could not open video file: {path}")
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            raise ValueError(f"could not read frame from video file: {path}")
        return frame, "video"

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is not None:
        return img, "image"

    cap = cv2.VideoCapture(path)
    if cap.isOpened():
        ret, frame = cap.read()
        cap.release()
        if ret and frame is not None:
            return frame, "video"

    raise ValueError(f"could not decode image or video: {path}")


def load_image(path):
    img, _ = load_input(path)
    return img


def preprocess(img_bgr, size):
    import cv2
    img = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_LINEAR)
    x = img[:, :, ::-1].astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    return np.ascontiguousarray((x - MEAN) / STD, dtype=np.float32)


def postprocess(out):
    out = np.asarray(out) if not hasattr(out, "cpu") else out.cpu().numpy()
    return out.argmax(axis=1).astype(np.uint8)


def time_runs(fn, sync, warmup, runs):
    for _ in range(warmup):
        fn()
    sync()
    lat = []
    for _ in range(runs):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        lat.append((time.perf_counter() - t0) * 1000.0)
    return lat


# ---------------------------------------------------------------- main
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="input image or video file path")
    p.add_argument("--model", help="model weights (.onnx / .pt / .pth)")
    p.add_argument("--backend", default="auto", choices=["auto", "onnx", "torchscript", "dummy"])
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--input-size", type=int, default=512, help="square model input size (default 512)")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--runs", type=int, default=30)
    p.add_argument("--output-dir", default="results")
    a = p.parse_args(argv)
    if a.runs < 1:
        p.error("--runs must be >= 1")
    if a.warmup < 0:
        p.error("--warmup must be >= 0")
    if a.input_size < 1:
        p.error("--input-size must be >= 1")
    return a


def env_info(backend):
    info = {"python": sys.version.split()[0], "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(), "cpu_count": os.cpu_count(),
            "numpy": np.__version__}
    for mod in ("cv2", "onnxruntime", "torch"):
        try:
            m = __import__(mod)
            info[mod] = m.__version__
        except Exception:
            pass
    return info


def main(argv=None):
    args = parse_args(argv)
    try:
        img, input_type = load_input(args.input)
        t0 = time.perf_counter()
        be = build_backend(args)
        load_ms = (time.perf_counter() - t0) * 1000.0
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    x = preprocess(img, args.input_size)
    mo = time_runs(lambda: be.run(x), be.sync, args.warmup, args.runs)
    ee = time_runs(lambda: postprocess(be.run(preprocess(img, args.input_size))), be.sync, args.warmup, args.runs)

    os.makedirs(args.output_dir, exist_ok=True)
    meta = {
        "model": be.name,
        "device": be.device,
        "input_type": input_type,
        "input_dims": f"{img.shape[1]}x{img.shape[0]}->{args.input_size}x{args.input_size}"
    }
    csv_path = os.path.join(args.output_dir, "latency_runs.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_index", "timing_scope", "latency_ms", "model", "device", "input_dims"])
        for scope, lat in (("model_only", mo), ("end_to_end", ee)):
            for i, v in enumerate(lat):
                w.writerow([i, scope, f"{v:.4f}", meta["model"], meta["device"], meta["input_dims"]])

    limitations = ["Batch size 1 only.", "Decode time excluded from end-to-end scope.",
                   "Single process; background load on the machine affects results."]
    if be.name == "dummy_no_model":
        limitations.insert(0, "DUMMY BACKEND: no real model was run. These numbers are NOT model performance.")
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "warmup": args.warmup, "runs": args.runs, "batch_size": 1, "backend": args.backend,
            "input_size": args.input_size, "input_type": input_type, "input_file": os.path.basename(args.input),
            "input_image": os.path.basename(args.input), **meta
        },
        "timing_scope": TIMING_SCOPES,
        "model_load_time_ms": load_ms,
        "model_only": compute_summary(mo),
        "end_to_end": compute_summary(ee),
        "environment": env_info(be),
        "limitations": limitations,
    }
    json_path = os.path.join(args.output_dir, "latency_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    mo_s = summary["model_only"]
    ee_s = summary["end_to_end"]

    print("\n" + "=" * 70)
    print("                    INFERENCE LATENCY BENCHMARK RESULTS")
    print("=" * 70)
    print(f" Input File   : {os.path.basename(args.input)} ({input_type})")
    print(f" Input Dims   : {meta['input_dims']}")
    print(f" Model / Dev  : {be.name} ({be.device})")
    print(f" Load Time    : {load_ms:.2f} ms")
    print("-" * 70)
    print(f" {'Metric':<20} | {'Model-Only':<20} | {'End-to-End':<20}")
    print("-" * 70)
    print(f" {'Mean Latency':<20} | {mo_s['mean_ms']:>16.2f} ms | {ee_s['mean_ms']:>16.2f} ms")
    print(f" {'Median Latency':<20} | {mo_s['median_ms']:>16.2f} ms | {ee_s['median_ms']:>16.2f} ms")
    print(f" {'Min Latency':<20} | {mo_s['min_ms']:>16.2f} ms | {ee_s['min_ms']:>16.2f} ms")
    print(f" {'Max Latency':<20} | {mo_s['max_ms']:>16.2f} ms | {ee_s['max_ms']:>16.2f} ms")
    print(f" {'P95 Latency':<20} | {mo_s['p95_ms']:>16.2f} ms | {ee_s['p95_ms']:>16.2f} ms")
    print(f" {'Throughput':<20} | {mo_s['throughput_img_per_s']:>13.2f} img/s | {ee_s['throughput_img_per_s']:>13.2f} img/s")
    print("-" * 70)
    print(f" Saved CSV  : {csv_path}")
    print(f" Saved JSON : {json_path}")
    print("=" * 70 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Task 4 - Video Stream Processing (RE-T1 Spectra / Rudra-Tactical)

Pipeline:
Video Input -> Frame Capture -> Frame Buffer -> OpenCV Processing ->
FPS + Latency Measurement -> Display/Save Output -> Performance Logs
"""
import argparse
import csv
import json
import os
import sys
import time
from collections import deque

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W, H = 640, 480


def process_frame(frame, size=(W, H)):
    """Basic local OpenCV processing: resize early to keep FPS high."""
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


def run(source, output_path, log_dir, show=True, buffer_size=120, max_frames=None):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {output_path}")

    buffer = deque(maxlen=buffer_size)   # bounded frame buffer
    latencies = []
    rows = []
    drops = 0
    count = 0
    start = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()

        if len(buffer) == buffer.maxlen:
            drops += 1                    # oldest frame is dropped by deque
        buffer.append(frame)

        frame = process_frame(frame)

        latency_ms = (time.perf_counter() - t0) * 1000
        latencies.append(latency_ms)
        count += 1
        elapsed = time.perf_counter() - start
        fps = count / elapsed if elapsed > 0 else 0.0
        rows.append((count, round(elapsed, 4), round(fps, 2), round(latency_ms, 3), len(buffer)))

        cv2.putText(frame, f"FPS: {fps:.1f}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Latency: {latency_ms:.1f} ms", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Buffer: {len(buffer)}", (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        writer.write(frame)

        if show:
            cv2.imshow("Video Stream Processing", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        if max_frames and count >= max_frames:
            break

    total_time = time.perf_counter() - start
    cap.release()
    writer.release()
    if show:
        cv2.destroyAllWindows()

    report = {
        "total_frames": count,
        "elapsed_seconds": round(total_time, 3),
        "average_fps": round(count / total_time, 2) if total_time > 0 else 0,
        "avg_latency_ms": round(sum(latencies) / len(latencies), 3) if latencies else 0,
        "min_latency_ms": round(min(latencies), 3) if latencies else 0,
        "max_latency_ms": round(max(latencies), 3) if latencies else 0,
        "buffer_max_size": buffer_size,
        "buffer_final_size": len(buffer),
        "buffer_drops": drops,
        "output_video": output_path,
    }

    with open(os.path.join(log_dir, "performance_log.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "elapsed_s", "fps", "latency_ms", "buffer_len"])
        w.writerows(rows)
    with open(os.path.join(log_dir, "performance_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    _plot(rows, os.path.join(log_dir, "performance_graphs.png"))
    return report


def _plot(rows, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return  # graphs are optional
    t = [r[1] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
    ax[0].plot(t, [r[2] for r in rows], color="tab:blue")
    ax[0].set(title="FPS over time", xlabel="Time (s)", ylabel="FPS")
    ax[1].plot(t, [r[3] for r in rows], color="tab:orange")
    ax[1].set(title="Latency over time", xlabel="Time (s)", ylabel="Latency (ms)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Video stream processing demo")
    ap.add_argument("--source", default=os.path.join(ROOT, "input", "sample_video.mp4"),
                    help="video file path, or 0 for webcam")
    ap.add_argument("--output", default=os.path.join(ROOT, "output", "processed_video.mp4"))
    ap.add_argument("--logs", default=os.path.join(ROOT, "output"))
    ap.add_argument("--no-display", action="store_true", help="run headless (no window)")
    ap.add_argument("--buffer", type=int, default=200)
    a = ap.parse_args()

    source = int(a.source) if str(a.source).isdigit() else a.source
    if isinstance(source, str) and not os.path.exists(source):
        print(f"Input not found: {source}\nGenerating synthetic sample video...")
        from generate_sample_video import generate
        generate(source)

    r = run(source, a.output, a.logs, show=not a.no_display, buffer_size=a.buffer)
    print("Processed frames:", r["total_frames"])
    print(f"Average FPS: {r['average_fps']}")
    print(f"Latency avg/min/max: {r['avg_latency_ms']} / {r['min_latency_ms']} / {r['max_latency_ms']} ms")
    print("Buffer drops:", r["buffer_drops"])
    print("Output:", r["output_video"])


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()

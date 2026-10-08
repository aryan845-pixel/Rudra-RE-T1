"""Generate a synthetic road-style sample video (zero-cost: no external data needed)."""
import argparse
import os
import cv2
import numpy as np


def generate(path, seconds=10, fps=30, size=(640, 480)):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    w, h = size
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError("Could not create sample video writer")
    rng = np.random.default_rng(42)
    total = seconds * fps
    for i in range(total):
        frame = np.zeros((h, w, 3), np.uint8)
        frame[: h // 2] = (235, 206, 135)          # sky
        frame[h // 2:] = (60, 60, 60)              # road
        # lane markings scrolling toward the camera
        for k in range(-1, 8):
            y = int(h // 2 + ((k * 40 + i * 6) % 280))
            half = 4 + (y - h // 2) // 25
            cv2.rectangle(frame, (w // 2 - half, y), (w // 2 + half, y + 18), (255, 255, 255), -1)
        # moving "vehicle" rectangle
        x = int((i * 5) % (w + 120)) - 100
        cv2.rectangle(frame, (x, h // 2 + 40), (x + 90, h // 2 + 100), (0, 0, 200), -1)
        cv2.rectangle(frame, (x + 15, h // 2 + 50), (x + 75, h // 2 + 75), (200, 220, 255), -1)
        # light sensor noise
        noise = rng.integers(0, 12, frame.shape, dtype=np.uint8)
        frame = cv2.add(frame, noise)
        writer.write(frame)
    writer.release()
    return total


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="input/sample_video.mp4")
    ap.add_argument("--seconds", type=int, default=10)
    a = ap.parse_args()
    n = generate(a.out, a.seconds)
    print(f"Generated {n} frames -> {a.out}")

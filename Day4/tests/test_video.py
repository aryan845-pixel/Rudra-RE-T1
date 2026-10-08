"""Testing checklist from Task 4 (run: python -m pytest tests  OR  python tests/test_video.py)"""
import os
import sys
import tempfile

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "video_processing"))
from generate_sample_video import generate  # noqa: E402
from video_processor import run, W, H  # noqa: E402


def _setup(tmp, seconds=2):
    src = os.path.join(tmp, "in.mp4")
    generate(src, seconds=seconds)
    return src


def test_video_opens_and_frames_processed():
    with tempfile.TemporaryDirectory() as tmp:
        src = _setup(tmp)
        r = run(src, os.path.join(tmp, "out.mp4"), tmp, show=False)
        assert r["total_frames"] == 60


def test_buffer_remains_bounded():
    with tempfile.TemporaryDirectory() as tmp:
        src = _setup(tmp)
        r = run(src, os.path.join(tmp, "out.mp4"), tmp, show=False, buffer_size=20)
        assert r["buffer_final_size"] <= 20
        assert r["buffer_drops"] == 60 - 20


def test_fps_and_latency_recorded():
    with tempfile.TemporaryDirectory() as tmp:
        src = _setup(tmp)
        r = run(src, os.path.join(tmp, "out.mp4"), tmp, show=False)
        assert r["average_fps"] > 0
        assert r["min_latency_ms"] <= r["avg_latency_ms"] <= r["max_latency_ms"]
        assert os.path.exists(os.path.join(tmp, "performance_log.csv"))


def test_processed_video_saved_with_correct_size():
    with tempfile.TemporaryDirectory() as tmp:
        src = _setup(tmp)
        out = os.path.join(tmp, "out.mp4")
        run(src, out, tmp, show=False)
        cap = cv2.VideoCapture(out)
        assert cap.isOpened()
        assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == W
        assert int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == H
        cap.release()


def test_bad_path_raises():
    try:
        run("does_not_exist.mp4", "x.mp4", ".", show=False)
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)

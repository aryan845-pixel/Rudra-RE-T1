"""Automated tests for Day 12 (headless). Run: python test_temporal.py"""
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

import temporal_analysis as ta
from make_sample_video import make


class TestDetect(unittest.TestCase):
    def test_identical_images_no_change(self):
        a = np.full((100, 100), 80, np.uint8)
        _, mask, boxes, changed = ta.detect_change(a, a.copy(), 25, 100)
        self.assertEqual(changed, 0)
        self.assertEqual(boxes, [])

    def test_changed_object_detected(self):
        a = np.full((100, 100), 80, np.uint8)
        b = a.copy()
        cv2.rectangle(b, (30, 30), (70, 70), 255, -1)
        diff, mask, boxes, changed = ta.detect_change(a, b, 25, 100)
        self.assertGreater(changed, 0)
        self.assertEqual(len(boxes), 1)


class TestRuns(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        make(self.tmp / "static.mp4", False)
        make(self.tmp / "moving.mp4", True)

    def _run(self, video, extra=()):
        argv = ["--video", str(video), "--no-gui", "--output-dir", str(self.tmp / "out"), *extra]
        return ta.run(ta.parse_args(argv))

    def test_static_video(self):
        s = self._run(self.tmp / "static.mp4")
        self.assertEqual(s["motion_frames"], 0)
        self.assertTrue((Path(s["output_dir"]) / "run_summary.json").exists())

    def test_moving_video_outputs(self):
        s = self._run(self.tmp / "moving.mp4", ["--save-video"])
        d = Path(s["output_dir"])
        self.assertGreater(s["motion_frames"], 10)
        for name in ("difference.png", "change_mask.png", "detected_output.mp4", "run_summary.json"):
            self.assertTrue((d / name).exists() and (d / name).stat().st_size > 0, name)
        self.assertIsNotNone(cv2.imread(str(d / "change_mask.png"), 0))
        self.assertGreater(cv2.countNonZero(cv2.imread(str(d / "change_mask.png"), 0)), 0)
        self.assertEqual(json.loads((d / "run_summary.json").read_text())["frames_read"], 60)

    def test_invalid_path(self):
        with self.assertRaises(RuntimeError):
            self._run(self.tmp / "nope.mp4")

    def test_unsupported_file(self):
        bad = self.tmp / "bad.mp4"
        bad.write_text("not a video")
        with self.assertRaises(RuntimeError):
            self._run(bad)

    def test_bad_threshold(self):
        with self.assertRaises(RuntimeError):
            self._run(self.tmp / "static.mp4", ["--threshold", "999"])

    def test_unique_run_dirs(self):
        a = self._run(self.tmp / "static.mp4")["output_dir"]
        b = self._run(self.tmp / "static.mp4")["output_dir"]
        self.assertNotEqual(a, b)


if __name__ == "__main__":
    unittest.main(verbosity=2)

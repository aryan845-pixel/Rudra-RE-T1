"""
Input-validation and pipeline-logic tests. Do NOT need ultralytics or internet.

Run from Day09/:   python -m unittest -v tests.test_input_validation
(or: pytest -v)
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import object_detection as od  # noqa: E402


class TestInputValidation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_missing_file(self):                                   # T-02
        with self.assertRaises(od.InputError) as cm:
            od.load_image(self.dir / "nope.jpg")
        self.assertIn("not found", str(cm.exception))

    def test_directory_instead_of_file(self):
        with self.assertRaises(od.InputError):
            od.load_image(self.dir)

    def test_empty_file(self):
        f = self.dir / "empty.jpg"
        f.write_bytes(b"")
        with self.assertRaises(od.InputError):
            od.load_image(f)

    def test_corrupt_image(self):                                  # T-03
        f = self.dir / "corrupt.jpg"
        f.write_bytes(b"this is not an image at all")
        with self.assertRaises(od.InputError) as cm:
            od.load_image(f)
        self.assertIn("decode", str(cm.exception))

    def test_valid_image_loads(self):
        f = self.dir / "ok.png"
        cv2.imwrite(str(f), np.full((40, 60, 3), 128, np.uint8))
        img = od.load_image(f)
        self.assertEqual(img.shape, (40, 60, 3))

    def test_confidence_range(self):
        self.assertEqual(od.validate_confidence(0.25), 0.25)
        for bad in (-0.1, 1.5):
            with self.assertRaises(od.InputError):
                od.validate_confidence(bad)

    def test_cli_missing_file_nonzero_exit(self):                  # AC-04
        proc = subprocess.run(
            [sys.executable, str(ROOT / "object_detection.py"), "--image", str(self.dir / "nope.jpg"),
             "--output-dir", str(self.dir / "out")],
            capture_output=True, text=True)
        self.assertEqual(proc.returncode, od.EXIT_INPUT)
        self.assertIn("INPUT ERROR", proc.stderr)
        self.assertNotIn("Saved", proc.stdout)                     # no misleading success message
        self.assertFalse((self.dir / "out" / "detected.jpg").exists())

    def test_cli_corrupt_file_nonzero_exit(self):
        f = self.dir / "bad.jpg"
        f.write_bytes(b"garbage")
        proc = subprocess.run([sys.executable, str(ROOT / "object_detection.py"), "--image", str(f),
                               "--output-dir", str(self.dir / "out")],
                              capture_output=True, text=True)
        self.assertEqual(proc.returncode, od.EXIT_INPUT)


class TestDrawingAndReport(unittest.TestCase):
    """Logic tests with hand-made Detection objects. These are NOT real model detections."""

    def test_draw_changes_pixels(self):                            # supports T-06 logic only
        img = np.full((200, 300, 3), 255, np.uint8)
        det = od.Detection(0, "person", 0.9, 50, 50, 150, 180)
        out = od.draw_detections(img, [det])
        self.assertFalse(np.array_equal(img, out))
        self.assertTrue(np.array_equal(img, np.full((200, 300, 3), 255, np.uint8)))  # input untouched

    def test_draw_zero_detections_ok(self):                        # supports T-04 logic only
        img = np.full((50, 50, 3), 10, np.uint8)
        self.assertTrue(np.array_equal(od.draw_detections(img, []), img))

    def test_report_fields(self):
        det = od.Detection(2, "car", 0.5, 1, 2, 3, 4)
        rep = od.build_report("a.jpg", "yolov8n.pt", 0.25, "cpu", [det], 12.345, Path("outputs/detected.jpg"))
        for key in ("input_path", "model", "inference_time_ms", "num_detections",
                    "class_labels", "confidence_scores", "output_image", "versions"):
            self.assertIn(key, rep)
        self.assertEqual(rep["num_detections"], 1)
        json.dumps(rep)  # must be JSON serialisable

    def test_report_zero_detections(self):
        rep = od.build_report("a.jpg", "yolov8n.pt", 0.9, "cpu", [], 5.0, Path("x.jpg"))
        self.assertEqual(rep["num_detections"], 0)
        self.assertEqual(rep["class_labels"], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)

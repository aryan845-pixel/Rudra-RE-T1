"""I/O and error-handling tests. They do NOT need torch or model weights.

Run (no extra installs needed):  python -m unittest discover -s tests -v
Also works with pytest if installed:  pytest tests -v
"""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import semantic_segmentation as ss  # noqa: E402


def make_image(path: Path, size=(64, 48)) -> Path:
    arr = np.random.RandomState(0).randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)
    return path


class TestPalette(unittest.TestCase):
    def test_voc_palette_known_values(self):
        p = ss.voc_palette()
        self.assertEqual(tuple(p[0]), (0, 0, 0))        # background
        self.assertEqual(tuple(p[1]), (128, 0, 0))      # aeroplane
        self.assertEqual(tuple(p[15]), (192, 128, 128))  # person


class TestOutputs(unittest.TestCase):
    def test_mask_and_overlay_saved_with_source_dimensions(self):
        with tempfile.TemporaryDirectory() as d:
            img = ss.load_image(make_image(Path(d) / "in.png", (64, 48)))
            class_map = np.zeros((48, 64), dtype=np.uint8)
            class_map[10:30, 10:40] = 15  # fake "person" region (synthetic, not model output)
            mask_p, over_p = ss.save_outputs(img, class_map, Path(d) / "new" / "out")
            for p in (mask_p, over_p):
                self.assertTrue(p.exists())
                with Image.open(p) as im:
                    self.assertEqual(im.size, (64, 48))
            with Image.open(mask_p) as m:
                self.assertEqual(m.getpixel((20, 20)), (192, 128, 128))
                self.assertEqual(m.getpixel((0, 0)), (0, 0, 0))

    def test_overlay_size_mismatch_raises(self):
        with self.assertRaises(ValueError):
            ss.blend_overlay(Image.new("RGB", (4, 4)), Image.new("RGB", (5, 5)))


class TestInputErrors(unittest.TestCase):
    def test_missing_file(self):
        with self.assertRaises(ss.InputError):
            ss.load_image("does_not_exist.jpg")

    def test_directory_is_not_a_file(self):
        with tempfile.TemporaryDirectory() as d, self.assertRaises(ss.InputError):
            ss.load_image(d)

    def test_corrupt_file(self):
        with tempfile.TemporaryDirectory() as d:
            bad = Path(d) / "bad.jpg"
            bad.write_bytes(b"this is not an image")
            with self.assertRaises(ss.InputError):
                ss.load_image(bad)

    def test_cli_missing_input_nonzero_exit_and_message(self):
        r = subprocess.run(
            [sys.executable, str(ROOT / "semantic_segmentation.py"),
             "--input", "nope.jpg", "--output-dir", "x"],
            capture_output=True, text=True,
        )
        self.assertEqual(r.returncode, ss.EXIT_INPUT)
        self.assertIn("does not exist", r.stderr)


if __name__ == "__main__":
    unittest.main()

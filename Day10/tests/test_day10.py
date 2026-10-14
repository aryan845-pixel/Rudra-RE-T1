"""Day 10 tests (unittest; also collected by pytest)."""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import generate_samples as gs
import image_enhancement as ie


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmp.name)
        cls.samples = {k: Path(v) for k, v in gs.generate(cls.tmp / "samples").items()}
        cls.clean = cv2.imread(str(cls.samples["clean_reference"]))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def load(self, name): return cv2.imread(str(self.samples[name]))


class TestDay10(Base):
    # 1
    def test_valid_gray_and_color(self):
        color = self.load("gaussian_noise")
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        gp = self.tmp / "gray.png"
        cv2.imwrite(str(gp), gray)
        for name, path in (("color", self.samples["gaussian_noise"]), ("gray", gp)):
            s = ie.run_pipeline(path, self.tmp / f"o1_{name}")
            out = cv2.imread(s["output_paths"]["enhanced.png"], cv2.IMREAD_UNCHANGED)
            self.assertEqual(out.ndim, 3 if name == "color" else 2)

    # 2
    def test_missing_and_unreadable(self):
        with self.assertRaises(FileNotFoundError):
            ie.load_image(self.tmp / "nope.png")
        bad = self.tmp / "bad.png"
        bad.write_text("not an image")
        with self.assertRaises(ValueError):
            ie.load_image(bad)
        with self.assertRaises(ValueError):
            ie.validate_image(np.zeros((0, 0), np.uint8))
        with self.assertRaises(TypeError):
            ie.validate_image([1, 2, 3])
        with self.assertRaises(ValueError):
            ie.validate_image(np.zeros((8, 8), np.float32))

    # 3
    def test_very_dark(self):
        dark = np.full((64, 64, 3), 5, np.uint8)
        dark[16:48, 16:48] = 12
        out = ie.enhance(dark, "lowlight")
        self.assertGreater(out.mean(), dark.mean() * 3)

    # 4
    def test_low_contrast(self):
        img = self.load("low_contrast")
        out = ie.contrast_stretch(img)
        g = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).std()
        self.assertGreater(g(out), g(img) * 2)

    # 5
    def test_gaussian_noise_improves_psnr(self):
        noisy = self.load("gaussian_noise")
        base = ie.psnr(noisy, self.clean)
        for m in ("gaussian", "bilateral"):
            self.assertGreater(ie.psnr(ie.denoise(noisy, m), self.clean), base, m)

    # 6
    def test_salt_pepper_median_best(self):
        noisy = self.load("salt_pepper_noise")
        med = ie.psnr(ie.denoise(noisy, "median"), self.clean)
        gau = ie.psnr(ie.denoise(noisy, "gaussian"), self.clean)
        self.assertGreater(med, ie.psnr(noisy, self.clean))
        self.assertGreater(med, gau)

    # 7
    def test_constant_images(self):
        for shape in ((32, 32), (32, 32, 3)):
            img = np.full(shape, 77, np.uint8)
            out = ie.contrast_stretch(img)
            np.testing.assert_array_equal(out, img)
            ll = ie.enhance(img, "lowlight", use_clahe=False)
            self.assertEqual(ll.shape, img.shape)
            self.assertFalse(np.isnan(ll.astype(float)).any())
        p = self.tmp / "const.png"
        cv2.imwrite(str(p), np.full((32, 32, 3), 77, np.uint8))
        ie.run_pipeline(p, self.tmp / "o7")  # must not raise

    # 8
    def test_invalid_names_and_params(self):
        img = self.load("gaussian_noise")
        bad = [
            lambda: ie.denoise(img, "sharpen"),
            lambda: ie.denoise(img, "gaussian", ksize=4),
            lambda: ie.denoise(img, "gaussian", ksize=-3),
            lambda: ie.denoise(img, "gaussian", sigma=-1),
            lambda: ie.denoise(img, "median", ksize=1),
            lambda: ie.denoise(img, "median", ksize=2.5),
            lambda: ie.denoise(img, "bilateral", d=0),
            lambda: ie.denoise(img, "bilateral", sigma_color=0),
            lambda: ie.denoise(img, "gaussian", bogus=1),
            lambda: ie.contrast_stretch(img, 60, 40),
            lambda: ie.contrast_stretch(img, -1, 99),
            lambda: ie.enhance(img, "lowlight", gamma=0),
            lambda: ie.enhance(img, "magic"),
            lambda: ie.enhance(img, "stretch", gamma=0.5),
            lambda: ie.run_pipeline(self.samples["low_light"], self.tmp / "o8", denoise_method="x"),
        ]
        for i, f in enumerate(bad):
            with self.assertRaises(ValueError, msg=f"case {i}"):
                f()
        self.assertFalse((self.tmp / "o8").exists())  # nothing written on failure

    # 9
    def test_dimensions_and_dtype(self):
        img = self.load("low_light")
        for m in ie.DENOISE_METHODS:
            d = ie.denoise(img, m)
            self.assertEqual((d.shape, d.dtype), (img.shape, np.uint8))
            for mode in ie.ENHANCE_MODES:
                e = ie.enhance(d, mode)
                self.assertEqual((e.shape, e.dtype), (img.shape, np.uint8))
                self.assertTrue(0 <= e.min() and e.max() <= 255)

    # 10
    def test_output_files(self):
        s = ie.run_pipeline(self.samples["low_light"], self.tmp / "o10", "bilateral",
                            enhance_mode="lowlight", reference_path=self.samples["clean_reference"])
        for name, p in s["output_paths"].items():
            self.assertTrue(Path(p).is_file(), name)
            if name.endswith(".png"):
                self.assertIsNotNone(cv2.imread(p), name)
            else:
                json.loads(Path(p).read_text())
        for k in ("denoise_method", "denoise_params", "image_width", "processing_time_seconds", "output_paths"):
            self.assertIn(k, s)
        q = json.loads(Path(s["output_paths"]["quality_metrics.json"]).read_text())
        self.assertIsNotNone(q["reference_comparison"]["psnr_enhanced_vs_reference_db"])

    # 11
    def test_original_preserved(self):
        src = self.samples["salt_pepper_noise"]
        before = sha(src)
        ie.run_pipeline(src, self.tmp / "o11")
        self.assertEqual(before, sha(src))
        # output dir == input dir with input named original.png -> refused, file intact
        d = self.tmp / "o11b"
        d.mkdir()
        orig = d / "original.png"
        cv2.imwrite(str(orig), self.load("low_light"))
        h = sha(orig)
        with self.assertRaises(ValueError):
            ie.run_pipeline(orig, d)
        self.assertEqual(h, sha(orig))

    # 12
    def test_repeat_runs(self):
        src = self.samples["gaussian_noise"]
        out = self.tmp / "o12"
        out.mkdir()
        unrelated = out / "keep_me.txt"
        unrelated.write_text("unrelated")
        h_in = sha(src)
        ie.run_pipeline(src, out)
        first = {n: sha(out / n) for n in ("denoised.png", "enhanced.png", "comparison.png")}
        ie.run_pipeline(src, out)
        second = {n: sha(out / n) for n in first}
        self.assertEqual(first, second)
        self.assertEqual(unrelated.read_text(), "unrelated")
        self.assertEqual(h_in, sha(src))


if __name__ == "__main__":
    unittest.main(verbosity=2)

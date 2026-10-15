"""Unit tests for the Day 11 super-resolution workbench (stdlib unittest, no extra deps).

Run from the project folder:  python -m unittest discover -s tests -v
"""
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sr_workbench as sr  # noqa: E402


def write_png(path: Path, img: np.ndarray):
    ok, buf = cv2.imencode(".png", img)
    assert ok
    buf.tofile(str(path))


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.out = self.tmp / "out"
        self.no_models = [self.tmp / "no_models_here"]

    def tearDown(self):
        self._tmp.cleanup()

    def colour_file(self, name="colour.png", w=96, h=72):
        p = self.tmp / name
        write_png(p, sr.make_synthetic_image(w, h))
        return p


class TestInput(Base):
    def test_01_valid_image(self):
        img, info = sr.load_image(self.colour_file())
        self.assertEqual(img.shape, (72, 96, 3))
        self.assertEqual(img.dtype, np.uint8)
        self.assertFalse(info["was_grayscale"])

    def test_02_missing_file(self):
        with self.assertRaises(sr.ImageLoadError):
            sr.load_image(self.tmp / "nope.png")

    def test_03_corrupted_file(self):
        bad = self.tmp / "bad.png"
        bad.write_bytes(b"this is definitely not a png")
        with self.assertRaises(sr.ImageLoadError):
            sr.load_image(bad)
        empty = self.tmp / "empty.jpg"
        empty.write_bytes(b"")
        with self.assertRaises(sr.ImageLoadError):
            sr.load_image(empty)

    def test_03b_unsupported_extension(self):
        p = self.tmp / "x.bmp"
        p.write_bytes(b"BM")
        with self.assertRaises(sr.ImageLoadError):
            sr.load_image(p)

    def test_04_grayscale(self):
        p = self.tmp / "gray.png"
        write_png(p, cv2.cvtColor(sr.make_synthetic_image(96, 72), cv2.COLOR_BGR2GRAY))
        img, info = sr.load_image(p)
        self.assertTrue(info["was_grayscale"])
        self.assertEqual(img.shape, (72, 96, 3))
        res = sr.run_pipeline(p, 2, self.out, self.no_models, repeats=1)
        self.assertEqual(res["methods"]["bicubic"]["image"].shape, (72, 96, 3))

    def test_05_colour(self):
        res = sr.run_pipeline(self.colour_file(), 2, self.out, self.no_models, repeats=1)
        self.assertEqual(res["reference"].shape[2], 3)

    def test_05b_alpha_and_16bit(self):
        rgba = cv2.cvtColor(sr.make_synthetic_image(64, 64), cv2.COLOR_BGR2BGRA)
        p = self.tmp / "alpha.png"
        write_png(p, rgba)
        img, info = sr.load_image(p)
        self.assertEqual(img.shape[2], 3)
        self.assertTrue(info["alpha_dropped"])
        p16 = self.tmp / "deep.png"
        write_png(p16, sr.make_synthetic_image(64, 64).astype(np.uint16) * 257)
        img16, _ = sr.load_image(p16)
        self.assertEqual(img16.dtype, np.uint8)

    def test_05c_non_ascii_path(self):
        d = self.tmp / "फोल्डर"
        d.mkdir()
        p = d / "चित्र.png"
        write_png(p, sr.make_synthetic_image(64, 64))
        img, _ = sr.load_image(p)
        self.assertEqual(img.shape, (64, 64, 3))


class TestScaling(Base):
    def test_06_x2_dimensions(self):
        lr = np.zeros((30, 40, 3), np.uint8)
        for m in sr.INTERPOLATIONS:
            out, _ = sr.upscale(lr, 2, m, repeats=1)
            self.assertEqual(out.shape, (60, 80, 3), m)

    def test_07_x4_dimensions(self):
        lr = np.zeros((30, 40, 3), np.uint8)
        for m in sr.INTERPOLATIONS:
            out, _ = sr.upscale(lr, 4, m, repeats=1)
            self.assertEqual(out.shape, (120, 160, 3), m)

    def test_07b_odd_size_cropped_and_aspect_preserved(self):
        img = sr.make_synthetic_image(203, 151)
        ref, lr = sr.prepare_reference(img, 4)
        self.assertEqual(ref.shape[:2], (148, 200))
        self.assertEqual(lr.shape[:2], (37, 50))
        out, _ = sr.upscale(lr, 4, "bicubic", 1)
        self.assertEqual(out.shape, ref.shape)

    def test_08_invalid_scales(self):
        for bad in (0, 1, -2, 9, 2.5, "4", None, True):
            with self.assertRaises(sr.InvalidScaleError, msg=repr(bad)):
                sr.validate_scale(bad)
        self.assertEqual(sr.validate_scale(4), 4)
        self.assertEqual(sr.validate_scale(2.0), 2)

    def test_08b_tiny_image_rejected(self):
        with self.assertRaises(sr.SuperResError):
            sr.prepare_reference(np.zeros((10, 10, 3), np.uint8), 4)


class TestOutputs(Base):
    def test_09_save_and_reopen(self):
        res = sr.run_pipeline(self.colour_file(), 4, self.out, self.no_models, repeats=1)
        for name in ("original.png", "low_resolution.png", "bilinear_x4.png", "bicubic_x4.png", "lanczos_x4.png", "comparison.png"):
            img, _ = sr.load_image(self.out / name)
            self.assertGreater(img.size, 0, name)
        self.assertEqual(sr.load_image(self.out / "bicubic_x4.png")[0].shape, res["methods"]["bicubic"]["image"].shape)
        self.assertTrue(np.array_equal(sr.load_image(self.out / "bicubic_x4.png")[0], res["methods"]["bicubic"]["image"]))

    def test_10_original_preserved(self):
        p = self.colour_file()
        before = sr.sha256_file(p)
        res = sr.run_pipeline(p, 4, self.out, self.no_models, repeats=1)
        self.assertEqual(sr.sha256_file(p), before)
        self.assertTrue(res["input"]["unchanged"])

    def test_10b_refuses_to_overwrite_input_in_output_dir(self):
        self.out.mkdir()
        p = self.out / "original.png"
        write_png(p, sr.make_synthetic_image(96, 72))
        before = sr.sha256_file(p)
        with self.assertRaises(sr.SuperResError):
            sr.run_pipeline(p, 2, self.out, self.no_models, repeats=1)
        self.assertEqual(sr.sha256_file(p), before)

    def test_11_summary_and_metrics_reports(self):
        sr.run_pipeline(self.colour_file(), 4, self.out, self.no_models, repeats=3)
        summary = json.loads((self.out / "run_summary.json").read_text(encoding="utf-8"))
        metrics = json.loads((self.out / "quality_metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["scale"], 4)
        for m in ("bilinear", "bicubic", "lanczos"):
            self.assertGreater(summary["methods"][m]["time_ms_median"], 0)
            self.assertEqual(summary["methods"][m]["repeats"], 3)
            self.assertIn("psnr_db", metrics["methods"][m])
            self.assertGreater(metrics["methods"][m]["ssim"], 0)
            self.assertGreater(metrics["methods"][m]["file_bytes"], 0)
        self.assertFalse(summary["ai"]["available"])
        self.assertTrue(summary["input"]["unchanged"])
        self.assertEqual(summary["gui"]["status"], "pending")

    def test_11b_synthetic_default_input(self):
        res = sr.run_pipeline(None, 2, self.out, self.no_models, repeats=1)
        self.assertEqual(res["input"]["source"], "synthetic")
        self.assertTrue((self.out / "sample_input.png").exists())


class TestMetrics(unittest.TestCase):
    def test_identical_images(self):
        img = sr.make_synthetic_image(64, 64)
        self.assertIsNone(sr.psnr(img, img))
        self.assertAlmostEqual(sr.ssim(img, img), 1.0, places=6)

    def test_degraded_is_worse(self):
        img = sr.make_synthetic_image(96, 96)
        blur = cv2.GaussianBlur(img, (9, 9), 3)
        self.assertLess(sr.ssim(img, blur), 0.99)
        self.assertGreater(sr.psnr(img, blur), 0)
        with self.assertRaises(ValueError):
            sr.psnr(img, img[:-1])

    def test_psnr_known_value(self):
        a = np.zeros((16, 16, 3), np.uint8)
        b = np.full((16, 16, 3), 10, np.uint8)
        self.assertAlmostEqual(sr.psnr(a, b), 10 * np.log10(255 ** 2 / 100), places=6)


class TestAI(Base):
    def test_12_no_model_files(self):
        res = sr.run_pipeline(self.colour_file(), 2, self.out, self.no_models, repeats=1)
        self.assertFalse(res["ai"]["available"])
        self.assertIsNone(res["ai"]["image"])
        self.assertNotIn("ai", res["methods"])
        self.assertFalse((self.out / "ai_super_resolution.png").exists())
        self.assertTrue(res["ai"]["reason"])

    def _fake_cv2_module(self, factory):
        return types.SimpleNamespace(DnnSuperResImpl_create=factory)

    def test_12b_corrupt_model_file_handled(self):
        mdir = self.tmp / "models"
        mdir.mkdir()
        (mdir / "EDSR_x2.pb").write_bytes(b"garbage, not a protobuf")

        class Broken:
            def readModel(self, path):
                raise cv2.error("cannot parse model")

        with mock.patch.object(cv2, "dnn_superres", self._fake_cv2_module(Broken), create=True):
            ai = sr.run_ai_superres(np.zeros((20, 20, 3), np.uint8), 2, [mdir])
        self.assertFalse(ai["available"])
        self.assertIn("inference failed", ai["reason"])

    def test_12c_wrong_scale_model_ignored(self):
        mdir = self.tmp / "models"
        mdir.mkdir()
        (mdir / "EDSR_x4.pb").write_bytes(b"x")
        self.assertEqual(sr.find_ai_models([mdir], 2), [])
        self.assertEqual(len(sr.find_ai_models([mdir], 4)), 1)

    def test_12d_wrong_output_shape_rejected(self):
        mdir = self.tmp / "models"
        mdir.mkdir()
        (mdir / "FSRCNN_x2.pb").write_bytes(b"x")

        class WrongShape:
            def readModel(self, path): pass
            def setModel(self, name, scale): pass
            def upsample(self, img): return img  # not upscaled

        with mock.patch.object(cv2, "dnn_superres", self._fake_cv2_module(WrongShape), create=True):
            ai = sr.run_ai_superres(np.zeros((20, 20, 3), np.uint8), 2, [mdir])
        self.assertFalse(ai["available"])

    def test_12e_mocked_success_plumbing(self):
        """Uses a MOCK engine - verifies saving/labelling plumbing only, not real AI quality."""
        mdir = self.tmp / "models"
        mdir.mkdir()
        (mdir / "ESPCN_x2.pb").write_bytes(b"x")

        class Fake:
            def readModel(self, path): self.path = path
            def setModel(self, name, scale): self.scale = scale
            def upsample(self, img):
                return cv2.resize(img, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_CUBIC)

        with mock.patch.object(cv2, "dnn_superres", self._fake_cv2_module(Fake), create=True):
            res = sr.run_pipeline(self.colour_file(), 2, self.out, [mdir], repeats=1)
        self.assertTrue(res["ai"]["available"])
        self.assertEqual(res["ai"]["model"], "ESPCN")
        self.assertTrue((self.out / "ai_super_resolution.png").exists())

    @staticmethod
    def _real_factory():
        mod = getattr(cv2, "dnn_superres", None)
        return getattr(mod, "DnnSuperResImpl_create", None) or getattr(mod, "DnnSuperResImpl", None) if mod else None

    def test_12g_REAL_library_rejects_corrupt_model(self):
        """Uses the real cv2.dnn_superres (no mock). Blocked if this OpenCV build lacks it."""
        if self._real_factory() is None:
            self.skipTest("BLOCKED - this OpenCV build has no DnnSuperResImpl (install opencv-contrib-python only)")
        mdir = self.tmp / "models"
        mdir.mkdir()
        (mdir / "EDSR_x2.pb").write_bytes(b"garbage, not a protobuf")
        ai = sr.run_ai_superres(np.zeros((20, 20, 3), np.uint8), 2, [mdir])
        self.assertFalse(ai["available"])
        self.assertIn("inference failed", ai["reason"])

    def test_12h_REAL_inference_with_local_models(self):
        """Real inference with genuine pretrained weights from .\\models. Blocked if none are present."""
        if self._real_factory() is None:
            self.skipTest("BLOCKED - this OpenCV build has no DnnSuperResImpl")
        mdirs = sr.default_model_dirs()
        scale = next((s for s in (2, 3, 4) if sr.find_ai_models(mdirs, s)), None)
        if scale is None:
            self.skipTest("BLOCKED - no pretrained model files (EDSR/LapSRN/FSRCNN/ESPCN _x<scale>.pb) in models/")
        res = sr.run_pipeline(self.colour_file(w=96, h=72), scale, self.out, mdirs, repeats=1)
        self.assertTrue(res["ai"]["available"], res["ai"]["reason"])
        self.assertEqual(res["methods"]["ai"]["image"].shape, res["reference"].shape)
        self.assertGreater(res["ai"]["time_ms"], 0)
        self.assertTrue((self.out / "ai_super_resolution.png").exists())

    def test_12f_stale_ai_output_removed_when_unavailable(self):
        self.out.mkdir()
        (self.out / "ai_super_resolution.png").write_bytes(b"stale")
        sr.run_pipeline(self.colour_file(), 2, self.out, self.no_models, repeats=1)
        self.assertFalse((self.out / "ai_super_resolution.png").exists())


class TestGui(Base):
    def test_13a_headless_error_is_reported(self):
        if not sys.platform.startswith("linux"):
            self.skipTest("DISPLAY check applies to Linux only")
        env = {k: v for k, v in os.environ.items() if k not in ("DISPLAY", "WAYLAND_DISPLAY")}
        with mock.patch.dict(os.environ, env, clear=True):
            wb = sr.Workbench(None, 2, self.out, self.no_models, 1)
            wb.process()
            with self.assertRaises(sr.GuiUnavailableError) as ctx:
                wb.run(auto_close_s=0.1)
        self.assertIn("headless", str(ctx.exception).lower())

    def test_13b_canvas_layout_and_aspect(self):
        res = sr.run_pipeline(self.colour_file(w=200, h=60), 4, self.out, self.no_models, repeats=1)
        canvas = res["canvas"]
        self.assertEqual(canvas.shape[1], 1620)
        cell = sr.fit_into(np.full((60, 200, 3), 255, np.uint8), 400, 300)
        white = np.where((cell == 255).all(axis=2))
        w, h = white[1].max() - white[1].min() + 1, white[0].max() - white[0].min() + 1
        self.assertAlmostEqual(w / h, 200 / 60, delta=0.1)  # aspect ratio preserved, not stretched

    def test_13c_detail_region_in_bounds_and_aligned(self):
        res = sr.run_pipeline(self.colour_file(w=300, h=200), 4, self.out, self.no_models, repeats=1)
        x0, y0, rw, rh = sr.default_detail_region(res["reference"], 4)
        self.assertEqual((x0 % 4, y0 % 4, rw % 4, rh % 4), (0, 0, 0, 0))
        self.assertLessEqual(x0 + rw, 300)
        self.assertLessEqual(y0 + rh, 200)
        panels = sr.build_panels(res, (x0, y0, rw, rh))
        self.assertEqual(panels[1]["image"].shape[:2], (rh // 4, rw // 4))
        self.assertEqual(panels[2]["image"].shape[:2], (rh, rw))

    def test_13c2_detail_keys_move_region_within_bounds(self):
        wb = sr.Workbench(None, 4, self.out, self.no_models, 1)
        wb.process()
        self.assertFalse(wb.handle_key(255))          # no key pressed
        self.assertTrue(wb.handle_key(ord("v")))      # toggle detail view
        self.assertTrue(wb.detail_mode)
        h, w = wb.res["reference"].shape[:2]
        for key in "aaaaaaaawwwwwwwwwwdddddddddddddddddssssssssssssssss":
            wb.handle_key(ord(key))
            x0, y0, rw, rh = wb.region
            self.assertTrue(0 <= x0 <= w - rw and 0 <= y0 <= h - rh)
            self.assertEqual((x0 % 4, y0 % 4), (0, 0))
        self.assertEqual(wb.frame().shape[1], 1620)

    def test_13d_real_window_opens_and_closes(self):
        ok, why = sr.gui_available()
        if not ok:
            self.skipTest(f"BLOCKED - cannot open a window here: {why}")
        wb = sr.Workbench(None, 2, self.out, self.no_models, 1)
        wb.process()
        reason = wb.run(auto_close_s=1.0)
        self.assertIn("auto-closed", reason)

    def test_13e_cli_no_display_exit_code(self):
        if not sys.platform.startswith("linux"):
            self.skipTest("Linux only")
        env = {k: v for k, v in os.environ.items() if k not in ("DISPLAY", "WAYLAND_DISPLAY")}
        import subprocess
        proc = subprocess.run([sys.executable, "-I", str(Path(sr.__file__)), "--out", str(self.out), "--repeats", "1"],
                              capture_output=True, text=True, env=env, cwd=self.tmp)
        self.assertEqual(proc.returncode, 2)
        self.assertIn("GUI ERROR", proc.stderr)
        self.assertIn("NO window was displayed", proc.stderr)
        summary = json.loads((self.out / "run_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["gui"]["status"], "failed")

    def test_cli_require_ai_exit_code(self):
        rc = sr.main(["--scale", "2", "--no-gui", "--require-ai", "--model-dir", str(self.no_models[0]), "--out", str(self.out)])
        self.assertEqual(rc, 3)
        summary = json.loads((self.out / "run_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["gui"]["status"], "not_opened")

    def test_cli_error_exit_codes(self):
        self.assertEqual(sr.main([str(self.tmp / "missing.png"), "--no-gui", "--out", str(self.out)]), 1)
        self.assertEqual(sr.main(["--scale", "1", "--no-gui", "--out", str(self.out)]), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)

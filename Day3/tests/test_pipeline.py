import os
import tempfile
import unittest

import cv2
import numpy as np

from preprocessing import PipelineConfig, Preprocessor, IMAGENET_MEAN, IMAGENET_STD
from preprocessing import ops


def fake(h=1080, w=1920, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


class TestOps(unittest.TestCase):
    def test_resize_stretch_shape(self):
        self.assertEqual(ops.resize(fake(), 640, 480).shape, (480, 640, 3))

    def test_resize_letterbox_keeps_aspect(self):
        out = ops.resize(fake(1000, 500), 640, 480, mode="letterbox", pad_value=7)
        self.assertEqual(out.shape, (480, 640, 3))
        self.assertTrue((out[:, 0] == 7).all())          # left padding present

    def test_bgr_to_rgb_swaps(self):
        px = np.zeros((2, 2, 3), np.uint8); px[..., 0] = 255   # pure blue in BGR
        self.assertEqual(tuple(ops.bgr_to_rgb(px)[0, 0]), (0, 0, 255))

    def test_filters_reduce_variance(self):
        img = fake(200, 200)
        for kind in ("gaussian", "median", "bilateral"):
            self.assertLess(ops.spatial_filter(img, kind).std(), img.std(), kind)

    def test_filter_none_identity(self):
        img = fake(50, 50)
        self.assertTrue(np.array_equal(ops.spatial_filter(img, "none"), img))

    def test_unknown_filter(self):
        with self.assertRaises(ValueError):
            ops.spatial_filter(fake(10, 10), "bogus")

    def test_normalize_range_and_value(self):
        n = ops.normalize(np.full((4, 4, 3), 128, np.uint8))
        self.assertEqual(n.dtype, np.float32)
        self.assertAlmostEqual(float(n[0, 0, 0]), 128 / 255, places=6)   # ~0.502

    def test_normalize_imagenet(self):
        n = ops.normalize(np.zeros((2, 2, 3), np.uint8), IMAGENET_MEAN, IMAGENET_STD)
        self.assertAlmostEqual(float(n[0, 0, 0]), -0.485 / 0.229, places=5)


class TestConfig(unittest.TestCase):
    def test_even_kernel_rejected(self):
        with self.assertRaises(ValueError):
            PipelineConfig(ksize=4)

    def test_mean_without_std_rejected(self):
        with self.assertRaises(ValueError):
            PipelineConfig(mean=(0.5, 0.5, 0.5))

    def test_json_roundtrip(self):
        cfg = PipelineConfig(mean=IMAGENET_MEAN, std=IMAGENET_STD, filter="median")
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            f.write(cfg.to_json())
        try:
            self.assertEqual(PipelineConfig.from_json(f.name), cfg)
        finally:
            os.unlink(f.name)


class TestPipeline(unittest.TestCase):
    def test_default_pipeline_matches_task_spec(self):
        r = Preprocessor()(fake())
        self.assertEqual(r.tensor.shape, (480, 640, 3))
        self.assertEqual(r.tensor.dtype, np.float32)
        self.assertGreaterEqual(float(r.tensor.min()), 0.0)
        self.assertLessEqual(float(r.tensor.max()), 1.0)

    def test_layouts(self):
        r = Preprocessor(PipelineConfig(channels_first=True, add_batch_dim=True))(fake())
        self.assertEqual(r.tensor.shape, (1, 3, 480, 640))

    def test_roundtrip_displayable(self):
        pre = Preprocessor(PipelineConfig(filter="none", mean=IMAGENET_MEAN, std=IMAGENET_STD,
                                          channels_first=True))
        img = fake(480, 640)
        back = pre.to_displayable(pre(img).tensor)
        self.assertLessEqual(int(np.abs(back.astype(int) - img.astype(int)).max()), 1)

    def test_keep_stages_and_timings(self):
        r = Preprocessor()(fake(), keep_stages=True)
        self.assertEqual(set(r.stages), {"resize", "rgb", "filter"})
        self.assertIn("normalize", r.timings_ms)

    def test_bad_inputs(self):
        pre = Preprocessor()
        with self.assertRaises(TypeError):
            pre(np.zeros((10, 10, 3), np.float32))
        with self.assertRaises(ValueError):
            pre(np.zeros((10, 10), np.uint8))
        with self.assertRaises(FileNotFoundError):
            pre.process_file("nope.jpg")

    def test_corrupt_file(self):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"not an image")
        try:
            with self.assertRaises(ValueError):
                Preprocessor().process_file(f.name)
        finally:
            os.unlink(f.name)

    def test_deterministic(self):
        img = fake()
        a, b = Preprocessor()(img).tensor, Preprocessor()(img).tensor
        self.assertTrue(np.array_equal(a, b))


if __name__ == "__main__":
    unittest.main()

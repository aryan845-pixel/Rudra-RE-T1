"""Tests for Day 13 Visual Change Detection. Run: python -m unittest discover -s tests -v
All tests write only to temporary directories - project samples/output are never touched."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import visual_change_detection as vcd  # noqa: E402

BOX = (300, 300, 400, 380)  # x1,y1,x2,y2 of the inserted rectangle in scenario 'single'


def iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    iw = max(0, min(ax2, bx2) - max(ax1, bx1)); ih = max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    return inter / ((ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter)


class TestDetection(unittest.TestCase):
    def setUp(self):
        self.base = vcd.make_base_scene()
        self.cfg = vcd.Config()

    def test_A_identical_images_zero_changes(self):
        res = vcd.detect_changes(self.base, vcd.make_changed_scene(self.base, "identical"), self.cfg)
        self.assertEqual(res.stats["changed_pixels"], 0)
        self.assertEqual(res.stats["change_regions"], 0)
        self.assertEqual(res.stats["changed_area_percent"], 0.0)

    def test_B_known_region_detected_at_expected_location(self):
        res = vcd.detect_changes(self.base, vcd.make_changed_scene(self.base, "single"), self.cfg)
        self.assertEqual(res.stats["change_regions"], 1)
        r = res.regions[0]
        found = (r["x"], r["y"], r["x"] + r["w"], r["y"] + r["h"])
        self.assertGreater(iou(found, BOX), 0.85, f"IoU too low: found={found}")

    def test_B2_area_percentage_is_correct(self):
        # Colour mode, no blur/morph: mask must equal the inserted rectangle exactly.
        res = vcd.detect_changes(self.base, vcd.make_changed_scene(self.base, "single"),
                                 vcd.Config(use_grayscale=False, blur_kernel=0, morph_kernel=0))
        s = res.stats
        expected = (BOX[2]-BOX[0]+1) * (BOX[3]-BOX[1]+1)  # cv2.rectangle includes end pixel
        self.assertEqual(s["total_pixels"], 640 * 480)
        self.assertEqual(s["changed_pixels"], expected)
        self.assertAlmostEqual(s["changed_area_percent"], round(100.0 * expected / (640 * 480), 4), places=4)
        self.assertEqual(s["changed_pixels"], int(np.count_nonzero(res.mask)))

    def test_B3_default_colour_mode_close_to_truth(self):
        # With default blur+morph the mask grows slightly (blur spreads edges): allow 10 %.
        res = vcd.detect_changes(self.base, vcd.make_changed_scene(self.base, "single"), vcd.Config(use_grayscale=False))
        expected = (BOX[2]-BOX[0]+1) * (BOX[3]-BOX[1]+1)
        self.assertLess(abs(res.stats["changed_pixels"] - expected) / expected, 0.10)

    def test_C_multiple_regions(self):
        res = vcd.detect_changes(self.base, vcd.make_changed_scene(self.base, "multi"), self.cfg)
        self.assertEqual(res.stats["change_regions"], 3)

    def test_D_noise_filtering_reduces_false_detections(self):
        noisy = vcd.make_changed_scene(self.base, "noisy")
        keep = np.zeros((480, 640), bool); keep[290:391, 290:411] = True  # true box + 10 px margin
        unf = vcd.detect_changes(self.base, noisy, vcd.Config(threshold=15, blur_kernel=0, morph_kernel=0, min_area=0))
        fil = vcd.detect_changes(self.base, noisy, vcd.Config(threshold=30, blur_kernel=5, morph_kernel=3, min_area=50))
        fp_unf = int(np.count_nonzero((unf.mask > 0) & ~keep))
        fp_fil = int(np.count_nonzero((fil.mask > 0) & ~keep))
        print(f"\n   noisy: false px outside box unfiltered={fp_unf} (regions={unf.stats['change_regions']}) | "
              f"filtered={fp_fil} (regions={fil.stats['change_regions']})")
        self.assertGreater(fp_unf, 1000)
        self.assertEqual(fp_fil, 0)
        self.assertEqual(fil.stats["change_regions"], 1)  # only the real rectangle survives

    def test_grayscale_mode_limitation_partial_detection(self):
        # Documented limitation: red box has luminance ~76, similar to parts of the background,
        # so grayscale mode finds the region but covers less area than colour mode.
        chg = vcd.make_changed_scene(self.base, "single")
        gray = vcd.detect_changes(self.base, chg, vcd.Config())
        col = vcd.detect_changes(self.base, chg, vcd.Config(use_grayscale=False))
        self.assertGreater(gray.stats["changed_pixels"], 0)
        self.assertLess(gray.stats["changed_pixels"], col.stats["changed_pixels"])

    def test_resize_same_aspect_ratio_is_allowed_and_original_preserved(self):
        small = cv2.resize(vcd.make_changed_scene(self.base, "single"), (320, 240), interpolation=cv2.INTER_AREA)
        before = small.copy()
        res = vcd.detect_changes(self.base, small, self.cfg)
        self.assertTrue(res.resized)
        self.assertEqual(res.changed.shape, self.base.shape)
        np.testing.assert_array_equal(small, before)  # caller's array untouched

    def test_incompatible_aspect_ratio_rejected(self):
        wide = np.zeros((100, 400, 3), np.uint8)
        with self.assertRaises(vcd.ChangeDetectionError):
            vcd.detect_changes(self.base, wide, self.cfg)
        res = vcd.detect_changes(self.base, wide, vcd.Config(force_resize=True))  # explicit override works
        self.assertTrue(res.resized)

    def test_grayscale_input_channels_handled(self):
        gray = cv2.cvtColor(vcd.make_changed_scene(self.base, "single"), cv2.COLOR_BGR2GRAY)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "g.png"; cv2.imwrite(str(p), gray)
            self.assertEqual(vcd.load_image(p).shape[2], 3)

    def test_invalid_threshold(self):
        with self.assertRaises(vcd.ChangeDetectionError):
            vcd.detect_changes(self.base, self.base, vcd.Config(threshold=300))


class TestOutputsAndErrors(unittest.TestCase):
    def test_outputs_valid_images_and_change_map_highlights(self):
        base = vcd.make_base_scene(); chg = vcd.make_changed_scene(base, "single")
        with tempfile.TemporaryDirectory() as d:
            res = vcd.detect_changes(base, chg, vcd.Config())
            paths = vcd.save_outputs(res, Path(d) / "new" / "out", "ref", "chg")  # dir auto-created
            vcd.build_comparison_figure(res, Path(d) / "new" / "out")
            for k in ("reference_preview", "changed_preview", "difference", "delta_heatmap", "change_mask", "change_map"):
                img = cv2.imread(paths[k], cv2.IMREAD_UNCHANGED)
                self.assertIsNotNone(img, k)
                self.assertEqual(img.shape[:2], (480, 640), k)
            mask = cv2.imread(paths["change_mask"], cv2.IMREAD_UNCHANGED)
            self.assertEqual(set(np.unique(mask)), {0, 255})
            cmap = cv2.imread(paths["change_map"])
            inside = cmap[310:370, 310:390].astype(int)   # inside inserted box
            outside = cmap[20:60, 500:600].astype(int)    # untouched area
            # red channel dominance inside the change, not outside
            self.assertGreater(inside[..., 2].mean() - inside[..., 1].mean(), 80)
            self.assertLess(outside[..., 2].mean() - outside[..., 1].mean(), 40)
            self.assertTrue(Path(d, "new", "out", "comparison_figure.png").exists())
            data = json.loads(Path(paths["run_summary"]).read_text(encoding="utf-8"))
            self.assertIn("changed_area_percent", data["statistics"])

    def test_E_missing_file(self):
        with self.assertRaises(vcd.ChangeDetectionError) as cm:
            vcd.load_image("does_not_exist.png")
        self.assertIn("not found", str(cm.exception).lower())

    def test_E_unreadable_corrupt_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "bad.png"; p.write_bytes(b"this is not an image")
            with self.assertRaises(vcd.ChangeDetectionError):
                vcd.load_image(p)
            e = Path(d) / "empty.png"; e.write_bytes(b"")
            with self.assertRaises(vcd.ChangeDetectionError):
                vcd.load_image(e)

    def test_E_unsupported_extension(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.txt"; p.write_text("hi")
            with self.assertRaises(vcd.ChangeDetectionError) as cm:
                vcd.load_image(p)
            self.assertIn("unsupported", str(cm.exception).lower())

    def test_E_cli_clean_error_no_traceback(self):
        r = subprocess.run([sys.executable, str(ROOT / "visual_change_detection.py"),
                            "--reference", "nope.png", "--changed", "nope2.png", "--headless"],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 2)
        self.assertIn("ERROR:", r.stderr)
        self.assertNotIn("Traceback", r.stderr)

    def test_cli_headless_end_to_end_and_no_collateral_writes(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            vcd.generate_samples(d / "samples", "single")
            marker = d / "unrelated.txt"; marker.write_text("keep me")
            args = [sys.executable, str(ROOT / "visual_change_detection.py"),
                    "--reference", str(d / "samples" / "reference.png"),
                    "--changed", str(d / "samples" / "changed.png"),
                    "--output-dir", str(d / "out"), "--headless"]
            for _ in range(2):  # repeated runs
                r = subprocess.run(args, capture_output=True, text=True)
                self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(marker.read_text(), "keep me")
            produced = {p.name for p in (d / "out").iterdir()}
            self.assertEqual(produced, {"reference_preview.png", "changed_preview.png", "difference.png",
                                        "delta_heatmap.png", "change_mask.png", "change_map.png",
                                        "comparison_figure.png", "run_summary.json"})

    def test_samples_are_reproducible(self):
        with tempfile.TemporaryDirectory() as d:
            a = vcd.generate_samples(Path(d) / "a", "multi"); b = vcd.generate_samples(Path(d) / "b", "multi")
            for x, y in zip(a, b):
                self.assertEqual(Path(x).read_bytes(), Path(y).read_bytes())


if __name__ == "__main__":
    unittest.main(verbosity=2)

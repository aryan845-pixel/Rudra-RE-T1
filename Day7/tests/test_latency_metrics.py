import csv, json, math, os, sys, tempfile, unittest
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import benchmark_latency as bl


class TestMetrics(unittest.TestCase):
    def test_known_values(self):
        s = bl.compute_summary([10.0, 20.0, 30.0, 40.0, 50.0])
        self.assertAlmostEqual(s["mean_ms"], 30.0)
        self.assertAlmostEqual(s["median_ms"], 30.0)
        self.assertEqual((s["min_ms"], s["max_ms"]), (10.0, 50.0))
        self.assertAlmostEqual(s["p95_ms"], 48.0)  # 40 + 0.8*10
        self.assertAlmostEqual(s["throughput_img_per_s"], 1000 / 30)

    def test_p95_matches_numpy(self):
        v = list(np.random.default_rng(0).uniform(5, 50, 30))
        self.assertAlmostEqual(bl.percentile(v, 95), float(np.percentile(v, 95)))

    def test_rejects_bad_values(self):
        for bad in ([], [1.0, float("nan")], [1.0, float("inf")], [-1.0]):
            with self.assertRaises(ValueError):
                bl.compute_summary(bad)


class TestCLI(unittest.TestCase):
    def setUp(self):
        import cv2
        self.d = tempfile.mkdtemp()
        self.img = os.path.join(self.d, "t.png")
        cv2.imwrite(self.img, np.random.default_rng(1).integers(0, 255, (120, 160, 3), dtype=np.uint8))

    def test_missing_image(self):
        self.assertEqual(bl.main(["--input", "nope.jpg", "--backend", "dummy"]), 2)

    def test_missing_weights(self):
        self.assertEqual(bl.main(["--input", self.img, "--model", "nope.onnx"]), 2)

    def test_invalid_counts(self):
        for args in (["--runs", "0"], ["--runs", "-3"], ["--warmup", "-1"]):
            with self.assertRaises(SystemExit):
                bl.parse_args(["--input", self.img] + args)

    def test_outputs(self):
        out = os.path.join(self.d, "o")
        rc = bl.main(["--input", self.img, "--backend", "dummy", "--input-size", "64",
                      "--warmup", "1", "--runs", "4", "--output-dir", out])
        self.assertEqual(rc, 0)
        rows = list(csv.DictReader(open(os.path.join(out, "latency_runs.csv"))))
        self.assertEqual(len(rows), 8)  # 4 model_only + 4 end_to_end
        self.assertTrue(all(math.isfinite(float(r["latency_ms"])) and float(r["latency_ms"]) >= 0 for r in rows))
        js = json.load(open(os.path.join(out, "latency_summary.json")))
        for k in ("model_load_time_ms", "model_only", "end_to_end", "environment", "limitations", "config"):
            self.assertIn(k, js)
        for k in ("mean_ms", "median_ms", "min_ms", "max_ms", "p95_ms"):
            self.assertIn(k, js["model_only"])

    def test_video_input(self):
        import cv2
        vid_path = os.path.join(self.d, "test_vid.avi")
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        out_vid = cv2.VideoWriter(vid_path, fourcc, 10.0, (160, 120))
        frame = np.random.default_rng(2).integers(0, 255, (120, 160, 3), dtype=np.uint8)
        for _ in range(5):
            out_vid.write(frame)
        out_vid.release()

        out_dir = os.path.join(self.d, "o_vid")
        rc = bl.main(["--input", vid_path, "--backend", "dummy", "--input-size", "64",
                      "--warmup", "1", "--runs", "2", "--output-dir", out_dir])
        self.assertEqual(rc, 0)
        js = json.load(open(os.path.join(out_dir, "latency_summary.json")))
        self.assertEqual(js["config"]["input_type"], "video")


if __name__ == "__main__":
    unittest.main()

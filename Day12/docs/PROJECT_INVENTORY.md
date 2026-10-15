# Project Inventory — Rudra Edge Aerospace | RE-T1 (Spectra)

**Prepared:** 16 October 2026  
**Task:** Day 12 — Final Documentation & Package  
**Author:** Aryan  
**Scope:** Entire repository, Days 1–11

---

## 1. Project Directory Structure

```
rudra edge/                        (repository root)
├── .git/                          (version control — excluded from display)
├── .gitignore
├── README.md                      (root — minimal, 3 lines)
│
├── Day1/
│   └── CVnotes.md                 (21 KB — Computer Vision study notes)
│
├── Day2/
│   ├── cv_pipeline.py             (10.9 KB — OpenCV pipeline: image/video/webcam)
│   ├── requirements.txt
│   ├── README.md
│   ├── input/
│   │   ├── test.jpg               (92.8 KB — synthetic input image)
│   │   └── test.mp4               (195.7 KB — synthetic input video)
│   └── output/
│       ├── original.jpg           (99.9 KB)
│       ├── grayscale.jpg          (84.2 KB)
│       ├── blurred.jpg            (53.8 KB)
│       ├── edges.jpg              (100.2 KB)
│       └── processed_video.mp4    (424.7 KB)
│
├── Day3/
│   ├── preprocess.py              (4.0 KB — CLI entry point)
│   ├── preprocessing/             (package: pipeline.py, __init__.py, etc.)
│   ├── requirements.txt
│   ├── README.md
│   ├── scripts/
│   ├── tests/
│   │   └── test_pipeline.py       (4.5 KB)
│   ├── input/
│   │   └── test.jpg               (1.4 MB — real input image)
│   └── output/
│       ├── test_preprocessed.jpg  (49.6 KB)
│       ├── test_montage.jpg       (79.7 KB)
│       └── test_tensor.npy        (3.7 MB — CHW float32 tensor)
│
├── Day4/
│   ├── video_processing/
│   │   ├── __init__.py
│   │   ├── video_processor.py     (5.6 KB — frame-by-frame video processor)
│   │   └── generate_sample_video.py (1.7 KB)
│   ├── requirements.txt
│   ├── README.md
│   ├── tests/
│   │   └── test_video.py          (2.1 KB)
│   ├── input/
│   │   └── sample_video.mp4       (627.4 KB)
│   └── output/
│       ├── performance_graphs.png (59.9 KB)
│       ├── performance_log.csv    (12.3 KB)
│       ├── performance_report.json(347 B)
│       └── processed_video.mp4    (3.0 MB)
│
├── Day5/
│   ├── object_detection.py        (18.9 KB — YOLOv8n ONNX object detector)
│   ├── yolov8n.onnx               (12.8 MB — pretrained YOLOv8 nano weights)
│   ├── requirements.txt
│   ├── README.md  NOTES.md
│   ├── images/
│   │   └── test.jpg               (actual COCO bus photo used for detection)
│   ├── tests/
│   │   └── test_input_validation.py (4.5 KB)
│   └── outputs/
│       ├── detected.jpg           (353.9 KB — annotated output)
│       └── run_report.json        (1.97 KB — real inference results)
│
├── Day6/
│   ├── semantic_segmentation.py   (8.0 KB — DeepLabV3 segmentation)
│   ├── requirements.txt
│   ├── README.md  NOTES.md
│   ├── samples/
│   │   ├── segmentation_model.pt  (44.7 MB — DeepLabV3-MobileNetV3 TorchScript)
│   │   └── synthetic_test.png     (207.8 KB)
│   ├── tests/
│   └── outputs/
│       ├── mask.png               (7.96 KB — segmentation class mask)
│       └── overlay.png            (1.62 MB — colour overlay)
│
├── Day7/
│   ├── benchmark_latency.py       (12.1 KB — inference latency benchmark)
│   ├── requirements.txt
│   ├── README.md
│   ├── DAILY_LOG_11Oct2026.md     (3.6 KB — engineering log for Day 7)
│   ├── samples/
│   │   ├── segmentation_model.pt  (44.7 MB — DeepLabV3 TorchScript model)
│   │   └── synthetic_test.png     (207.8 KB)
│   ├── tests/
│   │   └── test_latency_metrics.py(3.4 KB)
│   ├── results/
│   │   ├── latency_runs.csv       (4.1 KB — 60 individual run timings)
│   │   └── latency_summary.json   (1.66 KB — aggregated benchmark metrics)
│   └── results_controlled/        (3 timestamped run folders)
│
├── Day8/
│   ├── temporal_analysis.py       (13.9 KB — frame differencing / motion detection)
│   ├── make_sample_video.py       (806 B — synthetic video generator)
│   ├── test_temporal.py           (2.8 KB — 8 unit tests)
│   ├── requirements.txt
│   ├── README.md
│   ├── samples/
│   │   ├── moving.mp4             (12.4 KB — synthetic moving-object video)
│   │   └── static.mp4             (7.8 KB — synthetic static video)
│   └── output/
│       └── run_YYYYMMDD_HHMMSS/   (3 timestamped run folders)
│
├── Day9/
│   ├── visual_change_detection.py (20.7 KB — pixel-wise change detection)
│   ├── README.md
│   ├── tests/
│   ├── input/
│   │   ├── reference.png          (364.9 KB)
│   │   └── changed.png            (351.3 KB)
│   ├── samples/
│   └── output/
│       ├── reference_preview.png  (364.9 KB)
│       ├── changed_preview.png    (351.3 KB)
│       ├── difference.png         (7.3 KB)
│       ├── delta_heatmap.png      (10.9 KB)
│       ├── change_mask.png        (2.1 KB)
│       ├── change_map.png         (298.9 KB)
│       ├── comparison_figure.png  (467.4 KB)
│       └── run_summary.json       (1.74 KB)
│
├── Day10/
│   ├── image_enhancement.py       (13.4 KB — denoising + enhancement pipeline)
│   ├── generate_samples.py        (1.9 KB)
│   ├── run_day10.py               (2.2 KB — convenience runner)
│   ├── requirements.txt
│   ├── README.md
│   ├── Day10_Engineering_Log.md   (1.6 KB)
│   ├── samples/
│   │   ├── clean_reference.png
│   │   ├── gaussian_noise.png
│   │   ├── salt_pepper_noise.png
│   │   ├── low_contrast.png
│   │   └── low_light.png
│   ├── tests/
│   │   └── test_day10.py          (7.3 KB — 12 unit tests)
│   └── outputs/
│       ├── gaussian_noise/
│       ├── low_contrast/
│       ├── low_light/
│       └── salt_pepper_noise/
│
├── Day11/
│   ├── sr_workbench.py            (33.3 KB — super-resolution workbench)
│   ├── get_models.ps1             (2.65 KB — model downloader, untested)
│   ├── README.md
│   ├── logs/
│   │   └── day11_engineering_log.md (3.98 KB)  ← UNTRACKED in git
│   ├── tests/
│   │   └── test_sr_workbench.py   (17.0 KB — 37 unit tests)
│   └── outputs/
│       └── day11/
│           ├── original.png        (250.2 KB)
│           ├── low_resolution.png  (26.6 KB)
│           ├── bilinear_x2.png    (447.9 KB)
│           ├── bilinear_x4.png    (247.3 KB)
│           ├── bicubic_x2.png     (503.8 KB)
│           ├── bicubic_x4.png     (276.2 KB)
│           ├── lanczos_x2.png     (544.9 KB)
│           ├── lanczos_x4.png     (317.2 KB)
│           ├── comparison.png     (532.4 KB)
│           ├── sample_input.png   (341.9 KB)
│           ├── quality_metrics.json (897 B)
│           └── run_summary.json   (2.53 KB)
│
└── Day12/                         ← Created during this task (16 Oct 2026)
    ├── docs/
    │   ├── PROJECT_INVENTORY.md   (this file)
    │   ├── ARCHITECTURE.md
    │   ├── TEST_REPORT.md
    │   └── OUTPUT_CATALOG.md
    ├── ENGINEERING_LOG_16Oct2026.md
    ├── README.md
    └── requirements.txt
```

---

## 2. Important Source Files and Their Actual Purpose

| File | Day | Purpose |
|------|-----|---------|
| `Day2/cv_pipeline.py` | 2 | OpenCV fundamentals pipeline — resize, grayscale, blur, Canny edge detection, display & save. Supports image / video / webcam modes. |
| `Day3/preprocess.py` | 3 | CLI driver for `preprocessing/` package — letterbox/stretch resize, filter (Gaussian/median/bilateral), ImageNet normalisation, CHW tensor export, montage output. |
| `Day3/preprocessing/` | 3 | Package: `PipelineConfig`, `Preprocessor`, `IMAGENET_MEAN/STD`. |
| `Day4/video_processing/video_processor.py` | 4 | Frame-by-frame video processing with performance logging (CSV, JSON, PNG graphs). |
| `Day4/video_processing/generate_sample_video.py` | 4 | Synthetic video generator for testing. |
| `Day5/object_detection.py` | 5 | YOLOv8n object detection via OpenCV DNN (ONNX), bounding-box drawing, JSON run report. |
| `Day5/yolov8n.onnx` | 5 | 12.8 MB pretrained YOLOv8 nano ONNX model (80 COCO classes). Present locally. |
| `Day6/semantic_segmentation.py` | 6 | DeepLabV3-MobileNetV3 semantic segmentation via PyTorch TorchScript, 21-class PASCAL VOC palette. |
| `Day6/samples/segmentation_model.pt` | 6 | 44.7 MB TorchScript segmentation model. |
| `Day7/benchmark_latency.py` | 7 | Inference latency benchmark — model-only and end-to-end timing, CSV + JSON output. |
| `Day7/samples/segmentation_model.pt` | 7 | Copy of same TorchScript model used for benchmarking. |
| `Day8/temporal_analysis.py` | 8 | Frame differencing + motion detection from video/webcam. Saves diff, mask, video, JSON. |
| `Day8/make_sample_video.py` | 8 | Generates `samples/moving.mp4` and `samples/static.mp4`. |
| `Day8/test_temporal.py` | 8 | 8 unit tests (in root of Day8, not a tests/ sub-package). |
| `Day9/visual_change_detection.py` | 9 | Pixel-wise two-image change detection — absdiff, heatmap, binary mask, change map, stats. |
| `Day10/image_enhancement.py` | 10 | Denoising (Gaussian/median/bilateral) + enhancement (contrast stretch, gamma, CLAHE) with PSNR/SSIM. |
| `Day10/generate_samples.py` | 10 | Generates synthetic noisy/low-contrast/low-light sample images. |
| `Day10/run_day10.py` | 10 | Convenience runner for all four sample types. |
| `Day11/sr_workbench.py` | 11 | Super-resolution workbench — bilinear/bicubic/Lanczos interpolation, optional AI (`dnn_superres`), PSNR/SSIM, OpenCV GUI. |
| `Day11/get_models.ps1` | 11 | PowerShell script to download FSRCNN/ESPCN/EDSR weights. **Untested** (proxy blocked GitHub). |

---

## 3. Available Modules and Features

| Day | Module | Feature | Status |
|-----|---------|---------|--------|
| 1 | Study notes | CV theory and OpenCV fundamentals | ✅ Documented |
| 2 | `cv_pipeline.py` | Image/video/webcam pipeline + Canny edges | ✅ Implemented & outputs exist |
| 3 | `preprocess.py` + `preprocessing/` | Resize, filter, ImageNet normalisation, tensor export | ✅ Implemented & outputs exist |
| 4 | `video_processor.py` | Video stream processing + performance logging | ✅ Implemented & outputs exist |
| 5 | `object_detection.py` | YOLOv8n ONNX detection (OpenCV DNN) | ✅ Implemented & real run outputs exist |
| 6 | `semantic_segmentation.py` | DeepLabV3 segmentation (TorchScript) | ✅ Implemented & mask/overlay outputs exist |
| 7 | `benchmark_latency.py` | Inference latency measurement (model-only + end-to-end) | ✅ Implemented & real benchmark results exist |
| 8 | `temporal_analysis.py` | Frame differencing / motion detection | ✅ Implemented & test outputs exist |
| 9 | `visual_change_detection.py` | Two-image change detection, heatmap, change map | ✅ Implemented & real run output exists |
| 10 | `image_enhancement.py` | Denoising + enhancement (Gaussian/median/bilateral, CLAHE, gamma) | ✅ Implemented & outputs exist |
| 11 | `sr_workbench.py` (classical) | Bilinear / Bicubic / Lanczos super-resolution + PSNR/SSIM + GUI | ✅ Implemented & outputs exist |
| 11 | `sr_workbench.py` (AI) | AI super-resolution via `dnn_superres` (EDSR/LapSRN/FSRCNN/ESPCN) | ⚠️ BLOCKED — no model weights available; code path exists but was never run end-to-end |

---

## 4. Existing Tests and Their Locations

| Day | Test File | Count | Notes |
|-----|-----------|-------|-------|
| 3 | `Day3/tests/test_pipeline.py` | — | Present; not run in this task |
| 4 | `Day4/tests/test_video.py` | — | Present |
| 5 | `Day5/tests/test_input_validation.py` | 12 | 12 PASS per Day5 README |
| 6 | `Day6/tests/` | — | Folder present; contents not inspected |
| 7 | `Day7/tests/test_latency_metrics.py` | 8 | 8/8 PASS per Day7 log |
| 8 | `Day8/test_temporal.py` | 8 | 8/8 PASS per Day8 README |
| 9 | `Day9/tests/` | 18 | 18/18 PASS per Day9 README |
| 10 | `Day10/tests/test_day10.py` | 12 | 12/12 PASS per Day10 log |
| 11 | `Day11/tests/test_sr_workbench.py` | 37 | 34–36/37 PASS (1–3 skipped, see Day11 log) |

---

## 5. Existing Output Files and Sample Results

| Day | Output File | Description |
|-----|-------------|-------------|
| 2 | `output/original.jpg`, `grayscale.jpg`, `blurred.jpg`, `edges.jpg` | CV pipeline first-frame snapshots |
| 2 | `output/processed_video.mp4` | Edge-detected video |
| 3 | `output/test_preprocessed.jpg`, `test_montage.jpg`, `test_tensor.npy` | Preprocessed tensor + stage montage |
| 4 | `output/processed_video.mp4`, `performance_log.csv`, `performance_report.json`, `performance_graphs.png` | Video processing with metrics |
| 5 | `outputs/detected.jpg` | YOLOv8n annotated detection (5 detections: 4×person, 1×bus) |
| 5 | `outputs/run_report.json` | Inference: 236.23 ms on CPU |
| 6 | `outputs/mask.png`, `outputs/overlay.png` | Segmentation mask + colour overlay |
| 7 | `results/latency_runs.csv`, `results/latency_summary.json` | 30-run benchmark: model-only mean 1307 ms, end-to-end mean 2299 ms |
| 8 | `output/run_*/difference.png`, `change_mask.png`, `run_summary.json` | Motion detection run outputs |
| 9 | `output/change_map.png`, `delta_heatmap.png`, `comparison_figure.png`, `run_summary.json` | 3 change regions, 3.88% area changed |
| 10 | `outputs/{sample_type}/` | Denoised + enhanced images (4 noise categories) |
| 11 | `outputs/day11/*.png`, `quality_metrics.json`, `run_summary.json` | SR: Lanczos 4× PSNR=23.53 dB / SSIM=0.852 |

---

## 6. Dependencies and Their Purpose

| Package | Used In | Purpose |
|---------|---------|---------|
| `opencv-python` | Day2, Day3, Day4, Day8, Day9 | Core CV operations, image/video I/O |
| `opencv-python` or `opencv-contrib-python` | Day11 | SR workbench; contrib version needed for `dnn_superres` AI path |
| `numpy` | All | Array operations |
| `torch` + `torchvision` | Day6, Day7 | DeepLabV3 TorchScript model inference |
| `onnxruntime` | Day7 (optional) | ONNX model backend (Day5 uses OpenCV DNN instead) |
| `scikit-image` | Day10 | SSIM calculation (optional; falls back if missing) |
| `matplotlib` | Day9 (optional) | Comparison figure (falls back to OpenCV if missing) |

---

## 7. Documentation Already Present

| File | Location | Notes |
|------|----------|-------|
| `CVnotes.md` | Day1/ | Theory study notes |
| `README.md` | Day2–Day11 | Per-day READMEs |
| `NOTES.md` | Day5/, Day6/ | Supplementary implementation notes |
| `DAILY_LOG_11Oct2026.md` | Day7/ | Detailed engineering log (Day 7) |
| `Day10_Engineering_Log.md` | Day10/ | Engineering log (Day 10) |
| `Day11/logs/day11_engineering_log.md` | Day11/logs/ | Engineering log (Day 11) — **untracked in git** |

---

## 8. Missing Components and Known Limitations

| Component | Status | Notes |
|-----------|--------|-------|
| AI super-resolution weights (`.pb`) | ❌ Not available | `get_models.ps1` exists but download untested; internet blocked in build environment |
| Real AI SR inference end-to-end | ❌ Not executed | Code path exists; needs weights + `opencv-contrib-python` |
| Root `README.md` | ⚠️ Minimal (updated this task) | Original was only 3 lines |
| Root `requirements.txt` | ❌ Missing | Each day has its own; Day12/ provides a consolidated reference |
| Day1 has no source code | ℹ️ By design | Day1 is study notes only |
| GUI tests on Windows | ⚠️ Not verified | All GUI tests done on Linux with Xvfb |
| Webcam tests | ⚠️ Not verified | No physical camera in build environment |
| `Day5/outputs/video_report_*.json` | ⚠️ Deleted | Listed as `deleted` in `git status` — not restored by this task |

---

## 9. Files Created or Modified During This Task (Day 12)

| File | Action |
|------|--------|
| `Day12/docs/PROJECT_INVENTORY.md` | ✅ Created (this file) |
| `Day12/docs/ARCHITECTURE.md` | ✅ Created |
| `Day12/docs/TEST_REPORT.md` | ✅ Created |
| `Day12/docs/OUTPUT_CATALOG.md` | ✅ Created |
| `Day12/ENGINEERING_LOG_16Oct2026.md` | ✅ Created |
| `Day12/README.md` | ✅ Created |
| `Day12/requirements.txt` | ✅ Created |
| Root `README.md` | ✅ Updated (original 3-line content preserved and expanded) |

> No existing project files were modified, deleted, or overwritten.  
> No git commits, pushes, or branches were created.

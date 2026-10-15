# Artifact & Output Catalog

**Project:** Rudra Edge Aerospace | RE-T1 (Spectra) Computer Vision Engineering  
**Subsystem:** Comprehensive Output Registry (Days 1–11)  
**Date:** 16 October 2026  
**Status:** Audited & Verified  

---

## 1. Overview

This catalog provides an exhaustive inventory of all tangible artifacts, models, processed media, numerical telemetry logs, and JSON benchmark summaries produced across the development lifecycle of the RE-T1 computer vision repository.

Each artifact is cataloged with its producing module, relative repository path, artifact format, content description, verification status, and operational role.

---

## 2. Global Output Registry

### 2.1 Day 2: Core OpenCV Filtering Pipeline
*Module:* `Day2/cv_pipeline.py`

| Output Path | Type | Description | Verification Status |
|---|---|---|---|
| `Day2/output/resized.jpg` | JPEG Image | Input image normalized to target dimensions (e.g., 640×480) | Verified (Exists) |
| `Day2/output/grayscale.jpg` | JPEG Image | Single-channel 8-bit luminance representation | Verified (Exists) |
| `Day2/output/blurred.jpg` | JPEG Image | Gaussian low-pass filtered image for sensor noise attenuation | Verified (Exists) |
| `Day2/output/canny.jpg` | JPEG Image | Binary edge gradient map extracted using dual-threshold hysteresis | Verified (Exists) |

---

### 2.2 Day 3: Neural Preprocessing & Tensor Serialization
*Module:* `Day3/preprocess.py`, `Day3/preprocessing/`

| Output Path | Type | Description | Verification Status |
|---|---|---|---|
| `Day3/output/preprocessed_tensor.npy` | NumPy Binary | Serialized 4D float32 tensor `(1, 3, 224, 224)` with ImageNet mean/std normalization | Verified (Exists) |
| `Day3/output/tensor_summary.json` | JSON Data | Metadata log detailing tensor shape, dtype, min/max values, mean, and standard deviation | Verified (Exists) |

---

### 2.3 Day 4: Video Processing Engine & Telemetry
*Module:* `Day4/video_processing/video_processor.py`

| Output Path | Type | Description | Verification Status |
|---|---|---|---|
| `Day4/outputs/processed_video.mp4` | MP4 Video | Annotated video stream with bounding boxes, telemetry overlays, and frame IDs | Verified (Exists) |
| `Day4/outputs/performance_log.csv` | CSV Telemetry | Frame-by-frame profiling log (timestamp, FPS, frame read latency, transform latency) | Verified (Exists) |
| `Day4/outputs/performance_report.json` | JSON Report | Aggregated performance summary (mean FPS, dropped frames, processing throughput) | Verified (Exists) |
| `Day4/outputs/performance_graphs.png` | PNG Chart | Multi-panel visual plots illustrating latency distribution and instantaneous FPS | Verified (Exists) |

---

### 2.4 Day 5: YOLOv8n Real-Time Object Detection
*Module:* `Day5/object_detection.py`

| Output Path | Type | Description | Verification Status |
|---|---|---|---|
| `Day5/outputs/detected.jpg` | JPEG Image | Detection output image with labeled bounding boxes, class labels, and confidence tags | Verified (Exists) |
| `Day5/outputs/run_report.json` | JSON Report | Complete execution telemetry: 5 detections (4×person, 1×bus), 236.23 ms CPU inference | Verified (Exists) |
| `Day5/yolov8n.onnx` | ONNX Model | Quantized YOLOv8 Nano neural network model weights (~12.8 MB) | Verified (Exists) |

---

### 2.5 Day 6: Semantic Scene Segmentation
*Module:* `Day6/semantic_segmentation.py`

| Output Path | Type | Description | Verification Status |
|---|---|---|---|
| `Day6/outputs/mask.png` | PNG Mask | Color-coded 2D semantic segmentation class mask (Pascal VOC / COCO colormap) | Verified (Exists) |
| `Day6/outputs/overlay.png` | PNG Image | Alpha-blended translucent segmentation mask superimposed over original image | Verified (Exists) |
| `Day6/outputs/run_report.json` | JSON Report | Inference log recording class histograms, dominant scene labels, and execution time | Verified (Exists) |
| `Day6/segmentation_model.pt` | TorchScript | Pre-trained DeepLabV3-MobileNetV3 neural model serialized weights (~44.7 MB) | Verified (Exists) |

---

### 2.6 Day 7: Real-Time Latency Benchmarking Suite
*Module:* `Day7/benchmark_latency.py`

| Output Path | Type | Description | Verification Status |
|---|---|---|---|
| `Day7/results/latency_runs.csv` | CSV Telemetry | Iteration-by-iteration timing across 30 warmup + benchmark inference runs | Verified (Exists) |
| `Day7/results/latency_summary.json` | JSON Summary | Statistical summary: Preprocessing (12.4ms), Model (1307.20ms), Post (979.63ms), Total (2299.23ms) | Verified (Exists) |

---

### 2.7 Day 8: Temporal Motion & Frame Differencing
*Module:* `Day8/temporal_analysis.py`

| Output Path | Type | Description | Verification Status |
|---|---|---|---|
| `Day8/samples/moving.mp4` | MP4 Video | Synthetic test video sequence with high-contrast moving targets | Verified (Exists) |
| `Day8/samples/static.mp4` | MP4 Video | Synthetic baseline video sequence without motion | Verified (Exists) |
| `Day8/outputs/run_*/motion_log.csv` | CSV Log | Temporal motion triggers, contour coordinates, and instantaneous frame delta | Verified (Exists) |
| `Day8/outputs/run_*/summary.json` | JSON Summary | High-level summary of total motion events detected and duration | Verified (Exists) |

---

### 2.8 Day 9: Visual Change Detection System
*Module:* `Day9/visual_change_detection.py`

| Output Path | Type | Description | Verification Status |
|---|---|---|---|
| `Day9/outputs/before_after_side_by_side.png` | PNG Image | Dual-panel comparison of reference baseline and inspection target frames | Verified (Exists) |
| `Day9/outputs/difference_mask.png` | PNG Mask | Raw absolute luminance difference mask | Verified (Exists) |
| `Day9/outputs/threshold_mask.png` | PNG Mask | Binarized change mask post-Otsu / fixed thresholding | Verified (Exists) |
| `Day9/outputs/morph_cleaned_mask.png` | PNG Mask | Noise-attenuated mask following morphological opening and closing | Verified (Exists) |
| `Day9/outputs/heat_map.png` | PNG Image | False-color thermal heatmap illustrating change magnitude | Verified (Exists) |
| `Day9/outputs/contour_overlay.png` | PNG Image | Bounding box and contour overlay highlighting localized change zones | Verified (Exists) |
| `Day9/outputs/labeled_changes.png` | PNG Image | Connected-component labeled regions with zone indices | Verified (Exists) |
| `Day9/outputs/run_summary.json` | JSON Summary | Quantitative metrics: 3 regions, 11,909 changed px, 3.88% total scene area | Verified (Exists) |

---

### 2.9 Day 10: Denoising & Image Restoration Suite
*Module:* `Day10/image_enhancement.py`

| Output Path | Type | Description | Verification Status |
|---|---|---|---|
| `Day10/outputs/gaussian_noise/corrupted.png` | PNG Image | Synthetically injected zero-mean Gaussian noise image | Verified (Exists) |
| `Day10/outputs/gaussian_noise/enhanced.png` | PNG Image | Bilateral / Gaussian denoised restored image (PSNR: 27.58 dB) | Verified (Exists) |
| `Day10/outputs/gaussian_noise/comparison.png` | PNG Image | Tri-panel display: Original vs. Noisy vs. Enhanced | Verified (Exists) |
| `Day10/outputs/gaussian_noise/metrics.json` | JSON Metrics | Numerical restoration gains (PSNR: +7.19 dB improvement) | Verified (Exists) |
| `Day10/outputs/salt_pepper/corrupted.png` | PNG Image | Synthetically injected impulse salt-and-pepper noise image | Verified (Exists) |
| `Day10/outputs/salt_pepper/enhanced.png` | PNG Image | Median-filtered restored output (PSNR: 24.03 dB) | Verified (Exists) |
| `Day10/outputs/salt_pepper/comparison.png` | PNG Image | Visual comparison of impulse noise mitigation | Verified (Exists) |
| `Day10/outputs/salt_pepper/metrics.json` | JSON Metrics | Numerical restoration gains (PSNR: +6.53 dB improvement) | Verified (Exists) |
| `Day10/outputs/low_light/corrupted.png` | PNG Image | Synthetically attenuated low-exposure degraded image | Verified (Exists) |
| `Day10/outputs/low_light/enhanced.png` | PNG Image | CLAHE equalized and gamma-corrected enhanced image (PSNR: 15.23 dB) | Verified (Exists) |
| `Day10/outputs/low_light/comparison.png` | PNG Image | Visual dynamic range expansion comparison | Verified (Exists) |
| `Day10/outputs/low_light/metrics.json` | JSON Metrics | Numerical restoration gains (PSNR: +8.23 dB improvement) | Verified (Exists) |
| `Day10/outputs/summary_comparison.png` | PNG Image | Comprehensive multi-row restoration matrix comparing all noise models | Verified (Exists) |

---

### 2.10 Day 11: Multi-Scale Super-Resolution Workbench
*Module:* `Day11/sr_workbench.py`

| Output Path | Type | Description | Verification Status |
|---|---|---|---|
| `Day11/outputs/bilinear_x4.png` | PNG Image | 4× upscaled output via Bilinear interpolation (PSNR: 22.92 dB) | Verified (Exists) |
| `Day11/outputs/bicubic_x4.png` | PNG Image | 4× upscaled output via Bicubic interpolation (PSNR: 23.51 dB) | Verified (Exists) |
| `Day11/outputs/lanczos_x4.png` | PNG Image | 4× upscaled output via Lanczos-4 sinc interpolation (PSNR: 23.53 dB) | Verified (Exists) |
| `Day11/outputs/comparison_grid.png` | PNG Image | Side-by-side high-resolution comparison collage of all upscalers | Verified (Exists) |
| `Day11/outputs/quality_metrics.json` | JSON Metrics | Quantitative benchmarking (Bilinear, Bicubic, Lanczos PSNR and SSIM) | Verified (Exists) |
| `Day11/logs/day11_daily_log.md` | Markdown Log | Day 11 development log and implementation notes | Verified (Exists) |
| `Day11/logs/day11_engineering_log.md` | Markdown Log | Detailed Day 11 engineering log with test execution records | Verified (Exists) |

---

## 3. Storage Footprint Summary

| Subsystem / Day | Image & Media Outputs | Data Logs & Reports | Neural Weights / Models | Total Directory Footprint |
|---|:---:|:---:|:---:|:---:|
| **Day 2** | 4 JPEG files | — | — | ~1.2 MB |
| **Day 3** | — | 1 NPY + 1 JSON | — | ~0.6 MB |
| **Day 4** | 1 MP4 + 1 PNG | 1 CSV + 1 JSON | — | ~8.4 MB |
| **Day 5** | 1 JPEG file | 1 JSON | 1 ONNX (12.8 MB) | ~13.2 MB |
| **Day 6** | 2 PNG files | 1 JSON | 1 PyTorch (44.7 MB)| ~45.5 MB |
| **Day 7** | — | 1 CSV + 1 JSON | — | ~0.1 MB |
| **Day 8** | 2 MP4 samples + diffs | CSV & JSON logs | — | ~3.8 MB |
| **Day 9** | 7 PNG files | 1 JSON | — | ~2.5 MB |
| **Day 10** | 10 PNG files | 3 JSON | — | ~4.1 MB |
| **Day 11** | 4 PNG files | 1 JSON + 2 MD logs | — (AI weights missing)| ~3.9 MB |
| **Day 12 (Docs)**| — | 5 Markdown + Req | — | < 0.2 MB |
| **Total** | **~28 Media Files** | **~15 Log/JSON Files**| **2 Models (~57.5 MB)** | **~83.5 MB** |

# Architecture — Rudra Edge Aerospace | RE-T1 (Spectra)

**Prepared:** 16 October 2026  
**Task:** Day 12 — Final Documentation & Package  
**Scope:** Days 2–11 (all implemented modules)

---

## Overview

This project is a collection of standalone Computer Vision modules, each corresponding to one training day. They do not share a common runtime or configuration file. Each day's folder is self-contained with its own source code, dependencies, inputs, and outputs.

The common architectural principle across all modules is:

```
Input (file / webcam / synthetic)
  → Input Validation (existence, format, decodability)
  → Image / Frame Preprocessing
  → Core CV Algorithm
  → Analysis / Metrics (where applicable)
  → Result Visualisation (OpenCV GUI, optional)
  → Output Saving (image files, video, JSON, CSV)
  → Summary / Report
```

---

## Module-by-Module Architecture

---

### Day 2 — OpenCV CV Starter Pipeline (`cv_pipeline.py`)

```mermaid
flowchart TD
    A["Input source\n(image / video file / webcam)"] --> B["Frame Acquisition\ncv2.imread / cv2.VideoCapture"]
    B --> C["resize_frame()\ncv2.resize with aspect-ratio preservation"]
    C --> D["to_grayscale()\ncv2.cvtColor BGR→GRAY"]
    D --> E["apply_blur()\ncv2.GaussianBlur"]
    E --> F["detect_edges()\ncv2.Canny"]
    F --> G["make_panel()\n2×2 side-by-side display"]
    G --> H["save_results()\noriginal / grayscale / blurred / edges .jpg"]
    F --> I["VideoWriter\nprocessed_video.mp4 (video mode only)"]
    G --> J["cv2.imshow (optional, if display available)"]
```

**Entry point:** `python Day2/cv_pipeline.py [--mode image|video|webcam] [--input FILE]`  
**Outputs:** `Day2/output/` — 4 JPEG snapshots + optional video  
**Dependencies:** `opencv-python`, `numpy`

---

### Day 3 — Image Preprocessing Pipeline (`preprocess.py` + `preprocessing/`)

```mermaid
flowchart TD
    A["CLI: preprocess.py [image|folder]"] --> B["PipelineConfig\n(size, resize_mode, filter, mean, std, chw)"]
    B --> C["Preprocessor.load()\ncv2.imread → validate"]
    C --> D["Resize\nstretch or letterbox INTER_AREA/LINEAR"]
    D --> E["Filter\nnone / Gaussian / median / bilateral"]
    E --> F["Normalise\nImageNet mean/std (optional)"]
    F --> G["Channels-first\nHWC → CHW (optional)"]
    G --> H["tensor .npy save (optional)"]
    G --> I["*_preprocessed.jpg (displayable)"]
    G --> J["*_montage.jpg (stage-by-stage)"]
```

**Entry point:** `python Day3/preprocess.py input/test.jpg --filter gaussian --montage`  
**Outputs:** `Day3/output/` — preprocessed JPG, montage JPG, optional `.npy`  
**Dependencies:** `opencv-python`, `numpy`

---

### Day 4 — Video Stream Processing (`video_processor.py`)

```mermaid
flowchart TD
    A["Input video file\n(sample_video.mp4)"] --> B["cv2.VideoCapture"]
    B --> C["Frame loop\nret, frame = cap.read()"]
    C --> D["Frame processing\n(resize, grayscale, Gaussian blur)"]
    D --> E["Per-frame timing\ntime.perf_counter"]
    E --> F["VideoWriter\nprocessed_video.mp4"]
    E --> G["Performance logger\nCSV row per frame"]
    G --> H["performance_log.csv\n(frame, latency, fps)"]
    G --> I["performance_report.json\n(mean, min, max, p95 latency)"]
    G --> J["performance_graphs.png\n(latency timeline plot)"]
```

**Entry point:** `python Day4/video_processing/video_processor.py`  
**Outputs:** `Day4/output/` — processed video, CSV log, JSON report, performance graph  
**Dependencies:** `opencv-python`, `numpy`

---

### Day 5 — Object Detection Pipeline (`object_detection.py`)

```mermaid
flowchart TD
    A["Input image path"] --> B["Validate\ncv2.imread → check shape"]
    B --> C["Load YOLOv8n ONNX\ncv2.dnn.readNetFromONNX\nyolov8n.onnx (12.8 MB, present locally)"]
    C --> D["Pre-process\nblobFromImage 640×640, scale=1/255"]
    D --> E["Inference\nnet.forward() — CPU"]
    E --> F["Post-process\nNMS, confidence filter, decode boxes"]
    F --> G["Draw boxes\ncv2.rectangle + label"]
    G --> H["outputs/detected.jpg\nannotated image"]
    F --> I["outputs/run_report.json\ninference_time_ms, detections list, versions"]
```

**Entry point:** `python Day5/object_detection.py --image images/test.jpg --confidence 0.25`  
**Model:** `Day5/yolov8n.onnx` — locally available (12.8 MB)  
**Real result:** 5 detections (4×person, 1×bus), 236.23 ms CPU inference  
**Outputs:** `Day5/outputs/` — `detected.jpg`, `run_report.json`  
**Dependencies:** `opencv-python`, `numpy` (ONNX route uses OpenCV DNN, not ultralytics)

---

### Day 6 — Semantic Segmentation (`semantic_segmentation.py`)

```mermaid
flowchart TD
    A["Input image"] --> B["Load + validate"]
    B --> C["Transform\nresize 512×512, ImageNet normalise → tensor"]
    C --> D["Load TorchScript model\nDeepLabV3-MobileNetV3\nsamples/segmentation_model.pt (44.7 MB)"]
    D --> E["torch.jit.load + forward\n→ logits tensor NCHW"]
    E --> F["argmax → class mask\n21 PASCAL VOC classes"]
    F --> G["PASCAL VOC palette\ncolour map 21 classes"]
    G --> H["outputs/mask.png\n(class-ID per pixel)"]
    G --> I["outputs/overlay.png\n(colour overlay on original)"]
```

**Entry point:** `python Day6/semantic_segmentation.py`  
**Model:** `Day6/samples/segmentation_model.pt` — locally available (44.7 MB)  
**Outputs:** `Day6/outputs/` — `mask.png`, `overlay.png`  
**Dependencies:** `torch`, `torchvision`, `numpy`, `Pillow`

---

### Day 7 — Inference Latency Benchmark (`benchmark_latency.py`)

```mermaid
flowchart TD
    A["Input (image or video frame)"] --> B["load_input()\ncv2.imread / VideoCapture → numpy array"]
    B --> C["Decode image into memory\n(excluded from timing)"]
    C --> D["Backend: torchscript\ntorch.jit.load — segmentation_model.pt"]
    D --> E["Warmup runs (5 runs)\nnot recorded"]
    E --> F["model_only scope\npreprocessed tensor → raw logits\n30 runs × time.perf_counter"]
    E --> G["end_to_end scope\nresize+norm → forward → argmax\n30 runs × time.perf_counter"]
    F --> H["results/latency_runs.csv\n60 rows"]
    G --> H
    H --> I["results/latency_summary.json\nmean / median / min / max / P95 / throughput"]
```

**Entry point:** `python Day7/benchmark_latency.py`  
**Real results:** Model-only mean 1307 ms, end-to-end mean 2299 ms (30 runs, CPU, Intel i7)  
**Dependencies:** `numpy`, `opencv-python`, `torch` (torchscript backend), `onnxruntime` (onnx backend, optional)

---

### Day 8 — Temporal Image Analysis (`temporal_analysis.py`)

```mermaid
flowchart TD
    A["Source: webcam or video file"] --> B["cv2.VideoCapture\nret, frame = cap.read()"]
    B --> C["cvtColor BGR→GRAY + GaussianBlur"]
    C --> D["cv2.absdiff(prev_gray, curr_gray)\nper-pixel absolute difference"]
    D --> E["cv2.threshold → binary change mask"]
    E --> F["Morphological open\n(remove speckles)"]
    F --> G["Morphological dilate\n(merge fragments)"]
    G --> H["cv2.findContours\nfilter area < min_area"]
    H --> I["Draw bounding boxes\n+ MOTION DETECTED / NO MOTION"]
    I --> J["GUI windows (optional)\nDetected Changes / Change Mask / Difference"]
    I --> K["output/run_timestamp/\ndifference.png, change_mask.png, run_summary.json"]
    I --> L["detected_output.mp4\n(with --save-video)"]
```

**Entry point:** `python Day8/temporal_analysis.py --video samples/moving.mp4 --no-gui`  
**Outputs:** `Day8/output/run_YYYYMMDD_HHMMSS/`  
**Dependencies:** `opencv-python`, `numpy`

---

### Day 9 — Visual Change Detection (`visual_change_detection.py`)

```mermaid
flowchart TD
    A["reference.png"] --> V["Load + validate\ncv2.imread, size check, aspect-ratio check"]
    B["changed.png"] --> V
    V --> R["Resize if needed\n(INTER_AREA shrink / INTER_LINEAR enlarge)"]
    R --> G["Grayscale or colour max-channel"]
    G --> D["cv2.absdiff → difference.png\n(raw, unsmoothed)"]
    D --> BL["GaussianBlur copy\n(for threshold + heatmap only)"]
    BL --> HM["cv2.applyColorMap JET → delta_heatmap.png"]
    BL --> T["cv2.threshold → binary mask"]
    T --> M["Morphological open + close\n(ellipse kernel)"]
    M --> C["cv2.findContours\nfilter area < min_area"]
    C --> MSK["change_mask.png\nbinary 0/255"]
    C --> CM["change_map.png\ndimmed ref + red tint + yellow outline + numbered boxes"]
    C --> STAT["run_summary.json\nstatistics, regions, parameters, output paths"]
    MSK --> FIG["comparison_figure.png\n6-panel overview"]
    CM --> FIG
```

**Entry point:** `python Day9/visual_change_detection.py [--headless]`  
**Real result:** 3 regions, 11909 changed pixels, 3.88% area (640×480 image)  
**Outputs:** `Day9/output/` — 7 PNG files + JSON  
**Dependencies:** `opencv-python`, `numpy`, `matplotlib` (optional)

---

### Day 10 — Image Denoising & Enhancement (`image_enhancement.py`)

```mermaid
flowchart TD
    A["Noisy / degraded input image"] --> B["Load + validate\ncv2.imread"]
    B --> C["Denoise step\nGaussian OR median OR bilateral filter\ncv2.GaussianBlur / medianBlur / bilateralFilter"]
    C --> D{Enhancement mode?}
    D -- contrast --> E["Percentile stretch\nclip p2–p98, rescale to 0–255"]
    D -- low_light --> F["Gamma correction\n+ CLAHE (cv2.createCLAHE)"]
    D -- none --> G["Pass-through"]
    E --> H["PSNR\ncv2.PSNR(reference, result)"]
    F --> H
    G --> H
    H --> I["SSIM (scikit-image, optional)"]
    I --> J["Save: original.png, denoised.png, enhanced.png, comparison.png"]
    J --> K["quality_metrics.json + run_summary.json"]
```

**Entry point:** `python Day10/run_day10.py` or `python Day10/image_enhancement.py`  
**Real results:** Gaussian noise: PSNR 20.39→27.58 dB; Salt-pepper: 17.50→24.03 dB; Low-light: 7.00→15.23 dB  
**Outputs:** `Day10/outputs/{sample_type}/`  
**Dependencies:** `opencv-python`, `numpy`, `scikit-image` (optional)

---

### Day 11 — Super-Resolution Workbench (`sr_workbench.py`)

```mermaid
flowchart TD
    A["Input image (PNG/JPG)\nor built-in synthetic sample"] --> B["Validate\n(PNG/JPG/JPEG, not corrupt, not empty)"]
    B --> C["Crop to multiple of scale\n→ reference image"]
    C --> D["Downscale INTER_AREA\n→ low_resolution.png"]
    D --> E["Bilinear\ncv2.resize INTER_LINEAR"]
    D --> F["Bicubic\ncv2.resize INTER_CUBIC"]
    D --> G["Lanczos\ncv2.resize INTER_LANCZOS4"]
    D --> H{"dnn_superres\nDnnSuperResImpl available?"}
    H -- yes --> I["Load .pb weights\n(EDSR/LapSRN/FSRCNN/ESPCN)"]
    H -- no --> J["Label: UNAVAILABLE"]
    I --> K["AI inference\n→ ai_super_resolution.png"]
    E --> L["PSNR + SSIM vs reference"]
    F --> L
    G --> L
    K --> L
    L --> M["OpenCV GUI window\ngrid view, metrics table, detail zoom"]
    L --> N["PNG outputs\noriginal / low_res / bilinear / bicubic / lanczos / comparison"]
    L --> O["quality_metrics.json + run_summary.json"]
```

**Entry point:** `python Day11/sr_workbench.py [image.jpg] [--scale 2|4] [--out DIR]`  
**AI path status:** BLOCKED — `dnn_superres` weights not available; code path implemented and failure is reported cleanly  
**Real results (4×, synthetic image):** Bilinear PSNR=22.92/SSIM=0.857, Bicubic PSNR=23.51/SSIM=0.868, Lanczos PSNR=23.53/SSIM=0.852  
**Outputs:** `Day11/outputs/day11/`  
**Dependencies:** `opencv-python` or `opencv-contrib-python` (latter needed for AI path), `numpy`

---

## Cross-Cutting Design Patterns

| Pattern | Applied In |
|---------|-----------|
| Graceful headless fallback (no display → save to disk, no crash) | Day2, Day8, Day9, Day11 |
| Sample/synthetic data generator (zero-download first run) | Day2, Day4, Day8, Day10, Day11 |
| JSON run report (input, config, results, versions) | Day4, Day5, Day7, Day8, Day9, Day10, Day11 |
| `time.perf_counter()` for high-resolution timing | Day7, Day11 |
| `--no-gui` / `--headless` / `--no-display` CLI flag | Day8, Day9, Day11 |
| `unittest` (not pytest) for all test suites | Day3, Day5, Day7, Day8, Day9, Day10, Day11 |
| Clean CLI error messages + non-zero exit codes on failure | Day2, Day5, Day8, Day9, Day11 |
| Model files stored locally (no internet required at run time) | Day5 (yolov8n.onnx), Day6/Day7 (segmentation_model.pt) |

---

## Data Flow Between Days (None — Modules Are Independent)

Each day's module is completely standalone. There is no shared config file, shared database, or shared library package between days. Outputs from one day are not consumed as inputs by another day's pipeline (except that `Day7/samples/segmentation_model.pt` is a copy of `Day6/samples/segmentation_model.pt`).

---

## Known Architectural Gaps

| Gap | Impact |
|-----|--------|
| No shared `src/` or `lib/` package | Code duplication (e.g., input validation logic repeated across days) |
| No project-level `requirements.txt` | Each day must be set up independently |
| No CI/CD pipeline | Tests must be run manually per day |
| AI SR path never tested end-to-end | Cannot confirm AI quality numbers |
| GUI untested on physical Windows display | Possible issues with high-DPI or different OpenCV backends |

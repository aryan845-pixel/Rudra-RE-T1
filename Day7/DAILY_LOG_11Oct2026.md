# Daily Engineering Log - October 11, 2026

**Project:** RudraEdge RE-T1 / Spectra  
**Task Order:** REA/2026/RE-T1/TRAIN-001  
**Task Name:** Inference Latency Measurement  
**Date:** 11 October 2026  
**Status:** Completed  

---

## 1. Task Objective
Build and execute a reproducible, offline-compatible inference latency measurement framework to evaluate per-frame execution time for the project's real computer-vision semantic segmentation model (`DeepLabV3-MobileNetV3` TorchScript `.pt` model) across CPU, supporting both image and video file inputs.

---

## 2. Implementation Overview
- Built real PyTorch TorchScript Semantic Segmentation model (`segmentation_model.pt`) based on DeepLabV3-MobileNetV3 architecture.
- Developed `benchmark_latency.py` with multi-backend support (`torchscript`, `onnx`, `dummy`).
- Separated measurement scopes into:
  - `model_load_time_ms`: Model instantiation and weight loading duration (`2908.91 ms`).
  - `model_only`: Preprocessed tensor $\rightarrow$ raw model output forward pass latency (`1307.20 ms` mean).
  - `end_to_end`: Preprocessing (resize + normalize) $\rightarrow$ model forward pass $\rightarrow$ postprocessing (argmax segmentation mask) (`2299.23 ms` mean). Image frame decoding is excluded from timing loops.
- Implemented dual input support (`load_input`) handling both static images (`.jpg`, `.png`, `.bmp`) and video streams (`.mp4`, `.avi`, `.mov`, `.mkv`) via OpenCV.
- Recorded individual run measurements in `results/latency_runs.csv` and aggregate summary metrics in `results/latency_summary.json`.
- Implemented high-resolution timing using `time.perf_counter()`.
- Added comprehensive unit testing suite in `tests/test_latency_metrics.py`.

---

## 3. Tools & Environment
- **Model:** `segmentation_model.pt` (`DeepLabV3-MobileNetV3`, 21 classes, 512x512 tensor resolution)
- **Backend:** `torchscript` (PyTorch JIT engine)
- **Device:** CPU (Intel64 Family 6 Model 140 Stepping 1, 8 logical cores)
- **Python:** 3.11.9
- **Platform:** Windows-10 (x86_64)
- **Libraries:** NumPy 2.3.5, OpenCV (`cv2`) 4.10.0, PyTorch 2.14.0+cpu, ONNXRuntime 1.30.0
- **Execution Cost:** $0.00 (Fully offline, open-source local stack)

---

## 4. Key Outputs & Findings
- **Unit Tests:** 8/8 tests passed (`python -m unittest discover -s tests -v`).
- **Real Model Benchmark Results:**
  - **Model Loading Time:** 2908.91 ms
  - **Model-Only Latency:** Mean: 1307.20 ms | Median: 1624.84 ms | Min: 210.46 ms | Max: 2533.52 ms | P95: 1949.94 ms | Throughput: 0.76 img/s
  - **End-to-End Latency:** Mean: 2299.23 ms | Median: 2083.17 ms | Min: 1366.83 ms | Max: 3976.14 ms | P95: 3597.81 ms | Throughput: 0.43 img/s
- **Benchmark Run Artifacts:**
  - `results/latency_runs.csv`: 60 recorded run timings (30 `model_only` + 30 `end_to_end`).
  - `results/latency_summary.json`: Configuration details, environment info, load time, mean, median, min, max, P95, and throughput (img/s).

---

## 5. Encountered Problems & Resolutions
1. **Problem:** Initial test runs used dummy simulation backend. Real PyTorch semantic segmentation weights were required.
   - **Resolution:** Exported a full `DeepLabV3-MobileNetV3` TorchScript model (`segmentation_model.pt`) and executed real 30-run CPU inference benchmarking.
2. **Problem:** PyTorch dict outputs required wrapper for TorchScript tracing.
   - **Resolution:** Wrapped `deeplabv3_mobilenet_v3_large` in a custom `nn.Module` to extract tensor output directly before tracing.

---

## 6. Next Task
Next Task: Integration of edge quantization and deployment packaging (ONNX INT8 quantization & TensorRT compilation pipeline).

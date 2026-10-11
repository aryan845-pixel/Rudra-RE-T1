# Day 11 - Inference Latency Measurement
RudraEdge RE-T1 / Spectra | Task Order: REA/2026/RE-T1/TRAIN-001 | Date: 11 Oct 2026

## Purpose
Measure how long the project's computer-vision model (the 10 Oct semantic segmentation
pipeline) takes per image, reproducibly and fully offline.

## Setup
```
pip install -r requirements.txt
```
Put your model file (`.onnx`, or TorchScript `.pt`/`.pth`) and a test image or video file on disk.
Export a PyTorch model to ONNX/TorchScript first if needed - this script does not
download anything.

## Run
```
# Run benchmark on an image or video file:
python benchmark_latency.py --input samples/synthetic_test.png --backend dummy \
    --warmup 5 --runs 30 --output-dir results
```
Note: `--input` accepts image files (`.jpg`, `.png`, `.bmp`) as well as video files (`.mp4`, `.avi`, `.mov`, `.mkv`).
Options: `--backend auto|onnx|torchscript|dummy`, `--device cpu|cuda`, `--input-size 512`.
`--backend dummy` runs NO model; it only checks that the pipeline/outputs work.
Its numbers are not model performance.

Tests: `python -m unittest discover -s tests -v`

## Methodology
1. Image is decoded and the model loaded (load time recorded separately).
2. `--warmup` untimed runs, then `--runs` timed runs (default 30), batch size 1.
3. Each run timed with `time.perf_counter()`; for CUDA, `torch.cuda.synchronize()`
   is called before and after the timed region.
4. Two scopes are measured separately:
   - **model_only**: preprocessed tensor -> raw model output.
   - **end_to_end**: resize + normalize -> forward pass -> argmax mask
     (image decode excluded).

## Metrics (ms)
mean, median, min, max, P95 (linear interpolation), and throughput = 1000 / mean (img/s).

## Outputs (in `--output-dir`)
- `latency_runs.csv` - one row per timed run: run_index, timing_scope, latency_ms, model, device, input_dims.
- `latency_summary.json` - config, load time, both scopes' metrics, environment versions, limitations.

## Limitations / troubleshooting
- Batch size 1 only; results depend on machine load, so close other heavy apps.
- Preprocessing (resize, ImageNet mean/std) is generic; match it to your model's training
  preprocessing if different.
- Exit code 2 + `ERROR:` message for missing image, missing weights, undecodable image.
- `onnxruntime`/`torch` not installed -> install the one matching your model format.
- CUDA requested but unavailable -> falls back to CPU with a warning (check `device` in the JSON).
- Real latency numbers are NOT included in this package; run the command above on your workstation.

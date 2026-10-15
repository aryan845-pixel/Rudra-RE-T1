# Engineering Log - Day 11

- **Date:** 15 October 2026
- **Assigned Task:** Super-Resolution Study

## What Was Studied
- Difference between fixed-kernel interpolation (bilinear, bicubic, Lanczos) and learned super-resolution.
- OpenCV `dnn_superres` requirements (contrib module plus pretrained `.pb` model with matching scale).
- Reference-based quality measurement (PSNR, SSIM) using a clean image and a synthetically downscaled copy.

## What Was Implemented
- `sr_workbench.py`: input validation (PNG/JPG/JPEG, missing/corrupt/unsupported handling), synthetic sample generator,
  low-res generation (INTER_AREA), the three OpenCV interpolation methods with median timing, PSNR and SSIM (OpenCV/NumPy only),
  optional `dnn_superres` AI path that reports unavailability instead of fabricating output, JSON reports, and an OpenCV GUI
  (grid comparison, metrics table, detail zoom view, scale switching, image picker, q/Esc/close handling).
- `tests/test_sr_workbench.py`: 34 unittest tests. `README.md` and this log.

## Tools Used
Python 3.13.16, OpenCV 5.0.0, NumPy 2.5.3, unittest, Xvfb (virtual display for GUI testing).

## Output Produced
`outputs/day11/`: original, low_resolution, bilinear_x4, bicubic_x4, lanczos_x4, comparison (PNG), run_summary.json, quality_metrics.json, sample_input.png.
4x results (synthetic image): bilinear 18.48 dB / 0.6560, bicubic 18.97 dB / 0.6767, Lanczos 18.94 dB / 0.6588.

## Problems Encountered
1. The user's repository was not present in the workspace (empty directory, not a git repo), so the existing-repo review could not be done.
2. No pretrained super-resolution model files existed on the machine.
3. `cv2.dnn_superres` existed but was an empty namespace (three OpenCV packages installed together), so a plain `hasattr` check was misleading.
4. First test run: the synthetic image generator crashed on small images (fixed-position stripe patch).
5. Comparison canvas width was 1599 px instead of the intended value due to integer division.
6. No physical display, and `pytest` was not installed.

## How Problems Were Resolved
1. Built the project as a self-contained folder (`rudra_day11`) that can be copied into the repo; the review remains to be done there.
2. Implemented model discovery and clear "unavailable" labelling; AI inference was not run.
3. Detection now checks for the actual `DnnSuperResImpl` class and records the reason.
4. Stripe patch now scales with image size; regression covered by the small-image tests.
5. Canvas width changed to 1620 (divisible by 3); test updated.
6. Used stdlib `unittest`; tested the real window on Xvfb and reported the no-display case as blocked.

## Actual Test Results
- No display: 34 run, 33 passed, 1 skipped as blocked (real window test).
- Xvfb virtual display: 34 run, 34 passed (window opened, rendered, auto-closed, exit code 0).
- Not tested: a physical Windows desktop, real AI inference, Windows PowerShell commands.

## Follow-up Work (same day)
- Tried to obtain a working `dnn_superres` and real weights. A venv install of `opencv-contrib-python` failed: the sandbox
  proxy answered 403 for pypi.org (policy denial, not retried or bypassed). GitHub and WebFetch of GitHub pages were also denied.
- Found a complete `opencv-contrib-python` 5.0.0.93 build in the local uv cache; with it, `DnnSuperResImpl` exists and
  rejects a corrupt `.pb` with a real `cv2.error`, which the app reports correctly.
- Added `--require-ai` (exit code 3), `get_models.ps1` (untested downloader), real-library tests (corrupt model; inference
  with local weights) and a `--require-ai` test. README updated.
- Test results after follow-up: system OpenCV 37 run / 34 passed / 3 blocked; contrib build + Xvfb 37 run / 36 passed / 1 blocked
  (real inference: no weights available).
- Still not done: real AI inference with pretrained weights. This needs the weights on a machine with internet access.

## Next Task
Final Documentation & Package

## Personal Expenditure
₹0

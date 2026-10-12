# Day 12 — Temporal Image Analysis (Frame Differencing)

## Objective
Compare consecutive video frames with classical computer vision to detect **changes / potential motion**, show it live in OpenCV windows, and save results to disk. No deep learning, no internet, no paid services.

## What is temporal image analysis?
Analysing how an image changes **over time** (across frames) instead of looking at one frame alone. Here: if a pixel's brightness differs a lot between frame *t-1* and frame *t*, something in the scene probably changed there.
Note: this detects *change*, not *what* changed. It cannot tell a person from a car.

## Method
1. Read frame → convert to grayscale → Gaussian blur (reduces sensor noise)
2. `cv2.absdiff(prev, curr)` → per-pixel absolute difference
3. `cv2.threshold(diff, T, 255, THRESH_BINARY)` → binary change mask
4. Morphological open (remove speckles) + dilate (merge fragments)
5. `findContours` → keep regions with area ≥ `--min-area` → bounding boxes
6. Any box present → `MOTION DETECTED`, otherwise `NO MOTION`

## Workflow
`source (webcam/video) → grayscale+blur → absdiff → threshold → morphology → contours → overlay + GUI + saved outputs`

## Installation
```
pip install -r requirements.txt     # opencv-python, numpy
```

## Run
```
# Webcam (index 0), live windows
python temporal_analysis.py --webcam 0

# Webcam, save processed recording (auto-stops at 600 frames unless --max-frames given)
python temporal_analysis.py --webcam 0 --save-video --max-frames 300

# Video file
python temporal_analysis.py --video path/to/video.mp4 --save-video

# Headless (no windows)
python temporal_analysis.py --video samples/moving.mp4 --no-gui

# Generate sample videos, run tests
python make_sample_video.py
python test_temporal.py
```

## Parameters
| Option | Default | Purpose |
|---|---|---|
| `--webcam [INDEX]` / `--video PATH` | required (one of) | input source |
| `--threshold` | 25 | min brightness difference (0-255) counted as change. Lower = more sensitive/noisier |
| `--min-area` | 500 | ignore changed regions smaller than this many pixels |
| `--blur` | 5 | Gaussian kernel (odd); 0 disables |
| `--output-dir` | `output` | base folder; each run gets its own `run_<timestamp>` folder |
| `--save-video` | off | write `detected_output.mp4` |
| `--max-frames` | 0 | stop after N frames |
| `--no-gui` | off | no windows |

Defaults are starting points. No single threshold works in all lighting; tune per scene.

## GUI controls
Windows: *Detected Changes* (frame + boxes + status + FPS), *Change Mask*, *Difference*. Press **`q`** (with a window focused) or Ctrl+C in the terminal to quit. Camera, writer and windows are released on exit.

## Output structure
```
output/run_YYYYMMDD_HHMMSS/
  difference.png      # diff of the frame pair with most changed pixels (last pair if no motion)
  change_mask.png     # binary mask of that pair
  detected_output.mp4 # only with --save-video
  run_summary.json    # source, config, frame counts, status
```

## Tests actually executed (sandbox: Linux, headless, no camera, OpenCV 4.13.0)
| # | Test | Result |
|---|---|---|
| 1 | Script imports/starts, `parse_args` works | PASS |
| 2 | absdiff on two synthetic images (identical) → 0 changed px | PASS |
| 3 | absdiff on synthetic images with a white rectangle → 1 region found | PASS |
| 4 | Static synthetic video → 0 motion frames | PASS (0/59) |
| 5 | Moving-square synthetic video → motion detected | PASS (45/59 pairs) |
| 6 | `difference.png`, `change_mask.png`, `detected_output.mp4`, `run_summary.json` written, non-empty, mask has non-zero pixels | PASS |
| 7 | Invalid path / corrupt file / bad threshold → clear error | PASS |
| 8 | Two runs create different run folders (no overwrite) | PASS |
| 9 | Video-file mode via CLI (`--no-gui --save-video`) | PASS |
| 10 | Webcam mode | NOT TESTED OK — no camera in sandbox; script exited with clean error message (exit 1), which verifies the failure path only |
| 11 | Live GUI windows | NOT TESTED — sandbox has no display (Qt xcb abort). **Must be verified on your PC** |

Automated tests (items 1-3, 4-8) live in `test_temporal.py` (8 tests, all OK). Sample videos are synthetic, not real footage.

### Manual verification on your machine
1. `python make_sample_video.py`
2. `python temporal_analysis.py --video samples/moving.mp4` → 3 windows appear, status flips to MOTION DETECTED around frame 16, `q` closes everything.
3. `python temporal_analysis.py --webcam 0 --save-video --max-frames 200` → wave a hand; check `output/run_*/`.

## Limitations
- Lighting changes (auto-exposure, flicker, clouds) cause false detections across the whole frame.
- Shadows and reflections count as change.
- Camera shake/moving camera makes the whole scene "move".
- A still object that was moving is no longer detected; very slow motion may fall below threshold.
- Only compares adjacent frames, so the interior of large uniform objects can be missed (outline only).
- Tells *that* something changed, not *what*.

## Troubleshooting
- *Could not open webcam*: try index 1, close other apps using the camera, grant camera permission.
- *No windows / Qt xcb error*: you have no display; use `--no-gui`.
- *Too many false boxes*: raise `--threshold`/`--min-area` or `--blur 7`.
- *Misses small movement*: lower them.
- *mp4 won't play*: codec `mp4v`; try VLC.

## Next steps (Day 13)
Check your task plan. Likely extensions: background subtraction (MOG2), running-average background, motion heatmaps, or tracking regions across frames.

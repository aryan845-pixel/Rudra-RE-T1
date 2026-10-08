# OpenCV CV Starter Pipeline

**RudraEdge RE-T1 (Spectra)** | Task: OpenCV Pipeline Fundamentals | Author: Aryan | 6 Oct 2026

## Objective
Build a basic Computer Vision pipeline that handles image, video-file and webcam input,
processes every frame with OpenCV and produces visual output. Runs fully local, with free and
open-source tools only (no cloud, no paid API, no extra hardware).

## Pipeline
```
Input (image / video / webcam)
  -> Frame acquisition   cv2.imread() / cv2.VideoCapture()
  -> Resize              cv2.resize()
  -> Grayscale           cv2.cvtColor()
  -> Gaussian blur       cv2.GaussianBlur()
  -> Canny edges         cv2.Canny()
  -> Display / Save      cv2.imshow() / cv2.imwrite() / cv2.VideoWriter()
```

## Project structure
```
opencv_cv_pipeline/
├── cv_pipeline.py
├── requirements.txt
├── README.md
├── input/
│   ├── test.jpg
│   └── test.mp4
└── output/
    ├── original.jpg
    ├── grayscale.jpg
    ├── blurred.jpg
    ├── edges.jpg
    └── processed_video.mp4   (video mode)
```

## Setup
Requires Python 3.10+.
```bash
pip install opencv-python numpy
```
Or: `pip install -r requirements.txt`

## Run
| Purpose | Command |
|---|---|
| Default (image mode) | `python cv_pipeline.py` |
| Image mode | `python cv_pipeline.py --mode image --input input/test.jpg` |
| Video mode | `python cv_pipeline.py --mode video --input input/test.mp4` |
| Webcam (local) | `python cv_pipeline.py --mode webcam` |
| Create sample input files | `python cv_pipeline.py --make-samples` |

Press `q` or `ESC` in the window to quit video/webcam mode.

### Useful options
| Option | Default | Meaning |
|---|---|---|
| `--width` | 640 | Resize width (aspect ratio kept), `0` keeps original size |
| `--blur` | 5 | Gaussian kernel size (odd number) |
| `--canny-low` / `--canny-high` | 50 / 150 | Canny thresholds |
| `--output-dir` | `output/` | Where results are saved |
| `--no-display` | off | Do not open a window (headless / save only) |
| `--no-save-video` | off | Skip writing `processed_video.mp4` |
| `--max-frames` | 0 | Stop after N frames (0 = all) |

## Output
- Image mode: `original.jpg`, `grayscale.jpg`, `blurred.jpg`, `edges.jpg` in `output/`
- Video mode: first-frame snapshots plus `processed_video.mp4` (edge video)
- With a display available, a 2x2 panel (original / grayscale / blurred / edges) is shown live.

## Code overview
- `resize_frame`, `to_grayscale`, `apply_blur`, `detect_edges`: one small function per step.
- `process_frame`: runs the full chain and returns every intermediate result.
- `run_image_mode` / `run_stream_mode`: input handling for image, video file and webcam
  (`ret, frame = cap.read()` loop).
- Errors handled: missing file, unreadable image, video/webcam that cannot be opened.

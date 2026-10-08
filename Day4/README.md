# Task 4 — Video Stream Processing
**RudraEdge Aerospace | RE-T1 (Spectra) / Rudra-Tactical | Aryan — Computer Vision Engineer | 08 Oct 2026**

Local OpenCV pipeline: reads a video/webcam stream frame-by-frame, keeps a bounded frame buffer,
processes each frame, and measures FPS and latency.

```
Video Input -> Frame Capture -> Frame Buffer -> OpenCV Processing -> FPS + Latency -> Display/Save -> Logs
```

## Structure
```
Rudra-RE-T1/
├── input/sample_video.mp4          (auto-generated synthetic video if missing)
├── output/                         processed_video.mp4, performance_log.csv,
│                                   performance_report.json, performance_graphs.png
├── video_processing/
│   ├── video_processor.py          main pipeline
│   └── generate_sample_video.py    synthetic sample data (zero-cost)
├── tests/test_video.py             testing checklist
├── requirements.txt
└── README.md
```

## Install & Run
```bash
pip install -r requirements.txt
python video_processing/video_processor.py                  # sample video, with display window (q = quit)
python video_processing/video_processor.py --no-display     # headless
python video_processing/video_processor.py --source 0       # webcam
python video_processing/video_processor.py --source my.mp4 --buffer 30
python tests/test_video.py                                  # run checklist tests
```
(Optional: `pip install matplotlib` for FPS/latency graphs.)

## Key concepts
- **Frame buffer:** `deque(maxlen=20)` — bounded, oldest frame dropped automatically, memory never grows.
- **FPS:** processed frames / elapsed time.
- **Latency:** per-frame processing time in ms (`time.perf_counter`). 35 ms/frame ≈ 28.6 FPS capacity.

## Testing checklist
- [x] Video opens successfully
- [x] Frames read sequentially
- [x] Buffer remains bounded
- [x] FPS calculated and displayed
- [x] Per-frame and average latency recorded
- [x] Processed video saved
- [x] No repeated frame-read/runtime errors

## Problems & fixes
| Problem | Fix |
|---|---|
| Video not opening | Check path/format; `RuntimeError` is raised with the path |
| Buffer growth | `deque(maxlen=20)` |
| Low FPS | Resize early, remove unnecessary processing |
| Latency spikes | Per-frame timings in `performance_log.csv` show the expensive frame |

## Performance report
Auto-saved to `output/performance_report.json` after each run (total frames, elapsed time, average FPS,
avg/min/max latency, buffer drops).

## Zero-cost compliance
Existing laptop/PC, open-source software (Python, OpenCV), local/offline execution, synthetic sample data. Expenditure: ₹0.

# Day09 – Object Detection Pipeline (RudraEdge RE-T1 / Spectra)

**Objective:** a local, free, CPU-only object-detection demo. Load an image, detect objects with a small pretrained model, draw boxes + labels, save the annotated image and a JSON run report.

## Workflow
```
image path -> validate/decode (OpenCV) -> load YOLOv8n -> inference (CPU)
           -> draw boxes/labels -> outputs/detected.jpg + outputs/run_report.json
```

## Model / route chosen
- **Route:** PyTorch through the Ultralytics package (one runtime only; ONNX not required).
- **Model:** `yolov8n.pt` (YOLOv8 nano, pretrained on COCO, 80 classes).
- **Weights:** downloaded automatically on first run (free; needs internet once), then cached.
- **License note:** Ultralytics YOLOv8 is released under AGPL-3.0 (verify at ultralytics.com/license). Fine for a learning demo; check terms before any commercial use.

## Setup
Python 3.9+ (the sandbox used for authoring had 3.12.3).
```bash
cd Day09
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
python object_detection.py --image images/test.jpg --confidence 0.25
```
Options: `--model yolov8n.pt` · `--device cpu` · `--output-dir outputs` · `--fetch-sample` (downloads a public sample photo to the `--image` path first).

**Sample image:** `images/test.jpg` shipped here is a **synthetic placeholder** (shapes only; a detector will probably find nothing). Replace it with any photo you are allowed to use, or run once with `--fetch-sample` to get a public sample (Ultralytics' `bus.jpg`).

## Folder structure
```
Day09/
  object_detection.py          main script (CLI)
  requirements.txt
  README.md  NOTES.md  DAILY_ENGINEERING_LOG.md
  images/test.jpg              input
  outputs/detected.jpg         annotated output   (created by a real run)
  outputs/run_report.json      JSON report        (created by a real run)
  tests/test_input_validation.py
```

## Output meaning
- **class / label:** what the model thinks the object is (e.g. `person`).
- **confidence:** 0–1 score; detections below `--confidence` are dropped.
- **bounding box:** `x1,y1,x2,y2` pixel corners (top-left, bottom-right).
- **run_report.json:** input path, model, device, threshold, inference time (ms), count, labels, scores, boxes, output path, package versions.

Exit codes: `0` ok · `2` bad input/argument · `3` model/ultralytics problem · `4` other runtime error.

## Tests (status at time of authoring)
Run: `python -m unittest -v tests.test_input_validation`

| ID | Test | Status | Note |
|----|------|--------|------|
| T-01 | Valid image -> annotated image + JSON | **NOT RUN** | authoring sandbox had no internet / no ultralytics; run on your PC and update |
| T-02 | Nonexistent path | **PASS** | exit 2, "INPUT ERROR", no success message |
| T-03 | Corrupt/unsupported image | **PASS** | exit 2, readable decode error |
| T-04 | High threshold / zero detections | **NOT RUN** (real model) | drawing + report code path for 0 detections passes in unit tests only |
| T-05 | Same command twice | **NOT RUN** | needs real run |
| T-06 | Inspect annotated image | **NOT RUN** | unit test only checks pixels change when a hand-made box is drawn |

Automated: 12 unit tests, all PASS. Also verified: with ultralytics missing, the script prints a clear MODEL ERROR and exits 3 (no fake detections).

## Known limitations
- CPU inference is slower than GPU; first run also pays the weight download.
- COCO's 80 classes only; small, blurry, dark or occluded objects are often missed.
- Speed/accuracy numbers are not claimed here: take them from your own `run_report.json`.

## After your first real run
Fill T-01/T-04/T-05/T-06 above, copy the real detections + `inference_time_ms` into `DAILY_ENGINEERING_LOG.md`, and run `pip freeze` to pin versions.

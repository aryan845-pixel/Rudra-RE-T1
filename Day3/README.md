# Image Preprocessing Pipeline
**Task 3 — RE-T1 (Spectra) / Rudra-Tactical · RudraEdge Aerospace**

Production-style pipeline that turns a raw camera frame into a clean, model-ready tensor:

```
Raw image → Resize → BGR→RGB → Spatial filter → float32 → Normalize → Layout (HWC/CHW) → Model
```

Runs 100% locally with Python + OpenCV + NumPy (zero cost).

## Features
- **Resize**: `stretch` or aspect-preserving `letterbox`; auto interpolation (INTER_AREA down / LINEAR up)
- **Spatial filters**: Gaussian, Median, Bilateral, or none
- **Normalization**: `/255` → [0,1], optional per-channel mean/std (ImageNet preset)
- **Layouts**: HWC (OpenCV/TF) or CHW (+ batch dim) for PyTorch/ONNX
- **Validated config** (dataclass + JSON), typed errors, batch-safe folder processing
- **Debug tools**: per-stage images, per-stage timings, stage-by-stage montage, `.npy` export
- **18 unit tests** (stdlib `unittest`, also pytest-compatible)

## Structure
```
image-preprocessing/
├── preprocessing/
│   ├── config.py      # PipelineConfig (validated, JSON-serialisable)
│   ├── ops.py         # pure ops: resize, rgb, filter, normalize, layout
│   └── pipeline.py    # Preprocessor class (single / folder / inverse for saving)
├── tests/test_pipeline.py
├── scripts/make_sample.py   # synthetic 1920x1080 test image
├── preprocess.py      # CLI
├── input/test.jpg
├── output/
├── requirements.txt
└── README.md
```

## Setup & Run
```bash
pip install -r requirements.txt
python scripts/make_sample.py                         # optional: creates input/test.jpg
python preprocess.py input/test.jpg                   # spec default: 640x480, gaussian 5x5, [0,1]
python preprocess.py input/ --montage --save-npy      # whole folder + debug montage + tensors
python preprocess.py input/test.jpg --imagenet --chw --filter bilateral
python preprocess.py input/test.jpg --size 416 416 --resize-mode letterbox
python -m unittest discover -s tests -v               # run tests
```

## Python API
```python
from preprocessing import Preprocessor, PipelineConfig, IMAGENET_MEAN, IMAGENET_STD

pre = Preprocessor(PipelineConfig(width=640, height=480, filter="gaussian",
                                  mean=IMAGENET_MEAN, std=IMAGENET_STD,
                                  channels_first=True, add_batch_dim=True))
res = pre.process_file("input/test.jpg", keep_stages=True)
res.tensor.shape      # (1, 3, 480, 640) float32  -> feed to model
res.timings_ms        # per-stage latency
```

## Verification (Task checklist)
| Test | Result |
|---|---|
| Image loads | ✅ typed errors for missing/corrupt file |
| Resize → 480×640×3 | ✅ |
| float32 tensor | ✅ |
| min ≈ 0, max ≤ 1 | ✅ |
| Filtering smooths noise | ✅ (variance drops for all 3 filters) |
| Output file created | ✅ `output/<name>_preprocessed.jpg` |

## Design notes
- Filtering is done on the 8-bit image *before* float conversion (fast, matches the task report).
- Filter choice is an implementation decision (task order doesn't mandate one); Gaussian is the default.
- Mean/std standardisation is optional and off by default, so default output is exactly the [0,1] spec.

# Day 10 - Semantic Segmentation Demo (RudraEdge RE-T1)

Task Order: REA/2026/RE-T1/TRAIN-001 | Date: 10 October 2026

## 1. Purpose

Label **every pixel** of an image with a class (person, car, dog, ...) and save a colorized mask
plus an overlay on the original image.

**Semantic segmentation vs object detection**

| | Semantic segmentation | Object detection |
|---|---|---|
| Output | One class per pixel (dense map) | Boxes + class + score per object |
| Shape detail | Exact object boundaries | Rectangles only |
| Separate instances | No (two people = same label) | Yes (two boxes) |
| This project | Yes (DeepLabV3) | No |

## 2. Setup

Python 3.9+ (code was written on Python 3.13.16).

```bash
cd Day10
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
# CPU-only torch (smaller download):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install Pillow numpy
```

or simply `pip install -r requirements.txt`.

## 3. Run

```bash
python semantic_segmentation.py --input samples/test.jpg --output-dir outputs
```

Options: `--alpha 0.5` (overlay opacity, 0 to 1).

Outputs in `outputs/`:
- `mask.png` - colorized class mask, same width/height as the input
- `overlay.png` - mask blended over the original image

Terminal summary prints input path/size, model, output paths, classes found and inference time
(single cold run including preprocessing - not a benchmark; proper latency is the 11 Oct task).

## 4. Tests

```bash
python -m unittest discover -s tests -v      # no extra packages needed
pytest tests -v                              # optional
```

The tests cover the palette, mask/overlay saving and size alignment, and input error handling
(missing, directory, corrupt file, CLI exit code). They **do not** run the neural network.

## 5. Model

- Model: torchvision `deeplabv3_mobilenet_v3_large`, weights `DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT`
  (COCO subset trained with the 21 Pascal VOC labels).
- Why: true semantic-segmentation model, lightweight (about 11 M parameters), runs on CPU.
- Source: torchvision model zoo (https://pytorch.org/vision/stable/models.html). Code is BSD-3-Clause;
  check the weights' training-data terms on that page before any commercial use.
- Download behaviour: weights are fetched **once** on first run and cached in
  `~/.cache/torch/hub/checkpoints`. After that the program works offline.
- Hardware: CPU is enough (runs on CPU by default); no GPU code path is used.

## 6. Colors / classes (Pascal VOC palette)

Index: class (RGB) - `0` background (0,0,0), `1` aeroplane (128,0,0), `2` bicycle (0,128,0),
`3` bird (128,128,0), `4` boat (0,0,128), `5` bottle (128,0,128), `6` bus (0,128,128),
`7` car (128,128,128), `8` cat (64,0,0), `9` chair (192,0,0), `10` cow (64,128,0),
`11` diningtable (192,128,0), `12` dog (64,0,128), `13` horse (192,0,128), `14` motorbike (64,128,128),
`15` person (192,128,128), `16` pottedplant (0,64,0), `17` sheep (128,64,0), `18` sofa (0,192,0),
`19` train (128,192,0), `20` tvmonitor (0,64,128).

Overlay = `alpha * mask + (1 - alpha) * original`, so black background pixels darken the photo.

## 7. Known limits

- Only the 21 classes above; anything else becomes background or the nearest known class.
- Input is resized internally by the model transform; logits are resized back so outputs match the
  source size exactly.
- Accuracy is that of a small mobile model: boundaries are soft, small objects can be missed.
- Speed on CPU depends on the machine; not yet measured here.
- Offline use works only after the weights are cached.

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| `error: input file does not exist` (exit 2) | Check the path; run from `Day10/` or pass an absolute path |
| `unsupported or corrupt image` (exit 2) | Re-export as JPG/PNG |
| `torch/torchvision not available` (exit 3) | Install step in section 2 |
| `could not load pretrained weights` (exit 4) | No network: download `deeplabv3_mobilenet_v3_large-fc3c493d.pth` on another machine and place it in `~/.cache/torch/hub/checkpoints/` |
| Dependency conflicts | Use a fresh virtual environment |
| `cannot write outputs` (exit 5) | Check folder permissions |

## 9. Verification status (honest)

| Item | Status |
|---|---|
| Unit tests (7) | PASSED (observed) |
| CLI missing-input error, exit code 2 | PASSED (observed) |
| CLI without torch gives clear error, exit code 3 | PASSED (observed) |
| **Real DeepLabV3 inference** | **BLOCKED - NOT PASSED** |

Real inference was not run: in the build session `pip install torch` (PyPI and download.pytorch.org)
and the weights host returned HTTP 403 from the network egress policy, so torch/torchvision and weights
were unavailable. The model code path (`load_model`, `predict`) is written but **untested**. Run the
command in section 3 on a machine with internet to complete the check.

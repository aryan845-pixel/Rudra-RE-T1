# Detailed Notes - Semantic Segmentation (Day 10)

## 1. Concepts

**Image classification** gives one label per image. **Object detection** gives boxes. **Semantic
segmentation** gives one label per pixel. **Instance segmentation** also separates each object;
**panoptic** combines both.

## 2. How the pipeline works

1. `load_image` - check the file exists, is a file, decodes fully (`im.load()` catches truncated files); convert to RGB.
2. `load_model` - import torch lazily, build DeepLabV3-MobileNetV3-Large with pretrained weights, `eval()` mode.
3. `predict`
   - `weights.transforms()` resizes and normalizes the image (ImageNet mean/std).
   - The network outputs logits of shape `[1, 21, h, w]` (one channel per class).
   - Logits are bilinearly resized to the original `(H, W)`, then `argmax` over channels gives the class map.
4. `colorize_mask` - look up each class id in the VOC palette.
5. `blend_overlay` - `Image.blend(image, mask, alpha)`.
6. `save_outputs` - create the folder, write `mask.png` and `overlay.png`.

Resizing logits (not the final class map) gives smoother boundaries than nearest-neighbour upscaling.

## 3. Why DeepLabV3 + MobileNetV3

- **DeepLab** uses atrous (dilated) convolutions and ASPP to see context at several scales without losing resolution.
- **MobileNetV3-Large** backbone is light enough for CPU / edge, which matches the edge-AI direction of RE-T1.
- The ResNet-50 variant is more accurate but slower and bigger.

## 4. Design decisions

- Torch is imported inside `load_model`, so image I/O, palette and error handling are testable without torch.
- Distinct exit codes (2 input, 3 deps, 4 weights, 5 output) make failures scriptable.
- Tests use stdlib `unittest` (pytest also runs them), so nothing extra is needed to test.
- Generated outputs are git-ignored; only `outputs/.gitkeep` is tracked.
- No machine-specific paths anywhere.

## 5. What was actually observed on 9 Oct 2026 (build session)

- Workspace: empty Linux home, not a Git repository, Python 3.13.16, 2 CPU cores, about 7 GB RAM.
- Already installed: Pillow 12.3.0, numpy 2.5.3. Not installed: torch, torchvision, pytest.
- `pip install` for pytest and torch failed; `pypi.org` returned 403; `download.pytorch.org` returned 403.
  Per the session's egress policy these were reported, not worked around.
- Created an empty virtualenv, then removed it because it could not install anything.
- Result: 7 unit tests passed with `python -I -m unittest discover -s tests -v`.
- CLI with a missing file printed the error and exited with code 2; with a valid image but no torch it printed the dependency error and exited with code 3.
- Not done: real inference, latency measurement, visual check of a real mask.

## 6. Next steps

1. On a networked machine: install deps, put a photo at `samples/test.jpg`, run the command, open `mask.png` / `overlay.png`.
2. Record the real inference time; then do the 11 Oct task (Inference Latency Measurement: warm-up runs, repeated timing, mean/std).
3. Optional: try the ResNet-50 variant or GPU for comparison.

## 7. Revision questions

- Why does semantic segmentation merge two adjacent people into one region?
- Why resize logits instead of the argmax map?
- What does `torch.inference_mode()` save compared with normal mode?
- Why is the first inference slower than later ones? (relevant for tomorrow's latency task)

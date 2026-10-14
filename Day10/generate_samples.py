"""Create deterministic synthetic samples (+ clean references) for Day 10."""
from pathlib import Path
import cv2
import numpy as np

SIZE = 256


def clean_scene() -> np.ndarray:
    """Synthetic BGR scene with gradient, shapes, edges and fine lines."""
    y, x = np.mgrid[0:SIZE, 0:SIZE]
    img = np.zeros((SIZE, SIZE, 3), np.uint8)
    img[..., 0] = (60 + 120 * x / SIZE).astype(np.uint8)
    img[..., 1] = (80 + 100 * y / SIZE).astype(np.uint8)
    img[..., 2] = 110
    cv2.rectangle(img, (30, 30), (110, 110), (40, 200, 240), -1)
    cv2.circle(img, (180, 80), 40, (230, 60, 60), -1)
    cv2.rectangle(img, (140, 150), (230, 230), (30, 30, 30), -1)
    for i in range(0, 80, 8):  # fine stripes (detail)
        cv2.line(img, (20 + i, 150), (20 + i, 230), (250, 250, 250), 1)
    cv2.putText(img, "RE-T1", (30, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def generate(out_dir) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    clean = clean_scene()
    f = clean.astype(np.float32)

    gauss = np.clip(f + rng.normal(0, 25, f.shape), 0, 255).astype(np.uint8)

    sp = clean.copy()
    mask = rng.random(clean.shape[:2])
    sp[mask < 0.03] = 0
    sp[mask > 0.97] = 255

    low_contrast = np.clip(100 + (f - 128) * 0.25, 0, 255).astype(np.uint8)

    dark = f * 0.12
    low_light = np.clip(dark + rng.normal(0, 4, f.shape), 0, 255).astype(np.uint8)

    files = {"clean_reference": clean, "gaussian_noise": gauss, "salt_pepper_noise": sp,
             "low_contrast": low_contrast, "low_light": low_light}
    paths = {}
    for name, im in files.items():
        p = out / f"{name}.png"
        cv2.imwrite(str(p), im)
        paths[name] = str(p)
    return paths


if __name__ == "__main__":
    for k, v in generate(Path(__file__).parent / "samples").items():
        print(k, "->", v)

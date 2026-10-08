"""Generate a synthetic 1920x1080 'road' test image with noise (no downloads needed)."""
import cv2
import numpy as np

rng = np.random.default_rng(42)
H, W = 1080, 1920
img = np.zeros((H, W, 3), np.uint8)
img[:H // 2] = (235, 206, 135)                       # sky (BGR)
img[H // 2:] = (60, 60, 60)                          # road
pts = np.array([[W // 2 - 80, H // 2], [W // 2 + 80, H // 2], [W - 200, H], [200, H]])
cv2.fillPoly(img, [pts], (80, 80, 80))
for i in range(8):                                    # lane dashes
    y0 = H // 2 + 40 + i * 70
    cv2.line(img, (W // 2, y0), (W // 2, y0 + 30 + i * 6), (255, 255, 255), 3 + i)
cv2.rectangle(img, (300, 380), (520, 520), (40, 40, 160), -1)   # "vehicle"
noise = rng.normal(0, 18, img.shape)
img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
cv2.imwrite("input/test.jpg", img)
print("saved input/test.jpg", img.shape)

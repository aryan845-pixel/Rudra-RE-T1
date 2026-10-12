"""Generate synthetic test videos (static scene and moving square) into samples/."""
from pathlib import Path
import cv2
import numpy as np


def make(path, moving, frames=60, size=(320, 240)):
    w, h = size
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 20, (w, h))
    base = np.full((h, w, 3), 90, np.uint8)
    for i in range(frames):
        f = base.copy()
        if moving and i >= 15:
            x = 20 + (i - 15) * 5
            cv2.rectangle(f, (x, 90), (x + 50, 140), (255, 255, 255), -1)
        wr.write(f)
    wr.release()


if __name__ == '__main__':
    d = Path(__file__).parent / 'samples'
    d.mkdir(parents=True, exist_ok=True)
    make(d / 'static.mp4', False)
    make(d / 'moving.mp4', True)
    print('Created', d / 'static.mp4', 'and', d / 'moving.mp4')

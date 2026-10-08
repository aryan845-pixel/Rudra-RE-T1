#!/usr/bin/env python3
"""CLI: python preprocess.py input/test.jpg --out output --filter gaussian --montage"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

from preprocessing import PipelineConfig, Preprocessor, IMAGENET_MEAN, IMAGENET_STD
from preprocessing.pipeline import IMAGE_EXTS


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Image preprocessing pipeline (resize + normalize + filter)")
    p.add_argument("input", help="image file or folder of images")
    p.add_argument("--out", default="output", help="output folder (default: output)")
    p.add_argument("--config", help="JSON config file (overrides other flags)")
    p.add_argument("--size", nargs=2, type=int, metavar=("W", "H"), default=[640, 480])
    p.add_argument("--resize-mode", choices=["stretch", "letterbox"], default="stretch")
    p.add_argument("--filter", choices=["none", "gaussian", "median", "bilateral"], default="gaussian")
    p.add_argument("--ksize", type=int, default=5)
    p.add_argument("--imagenet", action="store_true", help="apply ImageNet mean/std standardisation")
    p.add_argument("--chw", action="store_true", help="output channels-first (PyTorch layout)")
    p.add_argument("--save-npy", action="store_true", help="save tensor as .npy")
    p.add_argument("--montage", action="store_true", help="save stage-by-stage comparison image")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def make_montage(original: np.ndarray, stages: dict, size=(320, 240)) -> np.ndarray:
    tiles = [("Original", original)]
    tiles += [(k.capitalize(), cv2.cvtColor(v, cv2.COLOR_RGB2BGR) if k in ("rgb", "filter") else v)
              for k, v in stages.items()]
    out = []
    for name, im in tiles:
        t = cv2.resize(im, size, interpolation=cv2.INTER_AREA)
        cv2.rectangle(t, (0, 0), (size[0], 22), (30, 30, 30), -1)
        cv2.putText(t, f"{name} {im.shape[1]}x{im.shape[0]}", (6, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        out.append(t)
    return np.hstack(out)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(message)s")
    log = logging.getLogger("preprocess")

    if args.config:
        cfg = PipelineConfig.from_json(args.config)
    else:
        cfg = PipelineConfig(
            width=args.size[0], height=args.size[1], resize_mode=args.resize_mode,
            filter=args.filter, ksize=args.ksize, channels_first=args.chw,
            mean=IMAGENET_MEAN if args.imagenet else None,
            std=IMAGENET_STD if args.imagenet else None,
        )
    pre = Preprocessor(cfg)

    src = Path(args.input)
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTS) if src.is_dir() else [src]
    if not files:
        log.error("No images found in %s", src)
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ok = 0
    for f in files:
        try:
            raw = pre.load(f)
            res = pre(raw, keep_stages=True)
        except Exception as e:
            log.error("%s: %s", f.name, e)
            continue
        t = res.tensor
        log.info("%s | in=%s -> tensor shape=%s dtype=%s min=%.3f max=%.3f | %.1f ms",
                 f.name, raw.shape, t.shape, t.dtype, t.min(), t.max(), sum(res.timings_ms.values()))
        cv2.imwrite(str(out_dir / f"{f.stem}_preprocessed.jpg"), pre.to_displayable(t))
        if args.save_npy:
            np.save(out_dir / f"{f.stem}_tensor.npy", t)
        if args.montage:
            cv2.imwrite(str(out_dir / f"{f.stem}_montage.jpg"), make_montage(raw, res.stages))
        ok += 1
    log.info("Done: %d/%d processed -> %s", ok, len(files), out_dir)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

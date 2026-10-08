"""Typed, validated configuration for the preprocessing pipeline."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Literal, Optional, Tuple

FilterName = Literal["none", "gaussian", "median", "bilateral"]
InterpName = Literal["nearest", "linear", "cubic", "area", "lanczos"]
ResizeMode = Literal["stretch", "letterbox"]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class PipelineConfig:
    # --- resize ---
    width: int = 640
    height: int = 480
    resize_mode: ResizeMode = "stretch"          # "letterbox" keeps aspect ratio
    interpolation: Optional[InterpName] = None   # None -> auto (area down / linear up)
    pad_value: int = 114                         # letterbox padding colour

    # --- colour ---
    to_rgb: bool = True

    # --- spatial filter ---
    filter: FilterName = "gaussian"
    ksize: int = 5                               # odd kernel size (gaussian/median)
    sigma: float = 0.0                           # gaussian sigma (0 -> derived from ksize)
    bilateral_d: int = 9
    bilateral_sigma_color: float = 75.0
    bilateral_sigma_space: float = 75.0

    # --- normalization ---
    mean: Optional[Tuple[float, float, float]] = None   # per-channel, 0-1 scale
    std: Optional[Tuple[float, float, float]] = None
    channels_first: bool = False                 # True -> CHW (PyTorch), False -> HWC
    add_batch_dim: bool = False                  # True -> NCHW / NHWC

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")
        if self.filter in ("gaussian", "median") and (self.ksize < 1 or self.ksize % 2 == 0):
            raise ValueError("ksize must be a positive odd integer")
        if (self.mean is None) != (self.std is None):
            raise ValueError("mean and std must be provided together")
        if self.std is not None and any(s <= 0 for s in self.std):
            raise ValueError("std values must be > 0")

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, path: str) -> "PipelineConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for key in ("mean", "std"):
            if data.get(key) is not None:
                data[key] = tuple(data[key])
        return cls(**data)

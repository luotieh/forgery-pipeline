"""图像/掩码的磁盘读写。"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import Image as PILImage


def save_image(img: np.ndarray, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    PILImage.fromarray(img).save(path)


def save_mask(mask: np.ndarray, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    PILImage.fromarray(mask).save(path)


def load_image(path) -> np.ndarray:
    return np.asarray(PILImage.open(path).convert("RGB"), dtype=np.uint8)


def load_mask(path) -> np.ndarray:
    return np.asarray(PILImage.open(path).convert("L"), dtype=np.uint8)

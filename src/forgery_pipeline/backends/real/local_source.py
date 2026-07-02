"""真实底图源：读本地图目录，中心裁剪 + resize 到工作分辨率。"""
from __future__ import annotations
from pathlib import Path
from typing import Iterator
import cv2
import numpy as np
from forgery_pipeline import image_io
from forgery_pipeline.backends import base

_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _center_crop_resize(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    return cv2.resize(img[y0:y0 + s, x0:x0 + s], (size, size))


class LocalImageSource(base.ImageSource):
    def __init__(self, root, size: int = 512, seed: int = 0):
        self.root = Path(root); self.size = size; self.seed = seed

    def iter_images(self, n: int) -> Iterator[tuple[np.ndarray, dict]]:
        files = sorted(p for p in self.root.rglob("*") if p.suffix.lower() in _EXTS)
        count = 0
        for p in files:
            if count >= n:
                break
            try:
                img = image_io.load_image(p)
            except Exception:
                continue
            yield _center_crop_resize(img, self.size), {
                "source_dataset": "local", "camera_model": None,
                "resolution": [self.size, self.size], "license": "unknown"}
            count += 1

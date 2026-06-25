"""基于感知哈希（pHash）的近重复检测（报告 §4.3）。"""
from __future__ import annotations
import imagehash
import numpy as np
from PIL import Image as PILImage


class PHashDeduper:
    def __init__(self, hamming_threshold: int = 5):
        self.threshold = hamming_threshold
        self._hashes: list = []

    def _hash(self, img: np.ndarray):
        return imagehash.phash(PILImage.fromarray(img))

    def is_duplicate(self, img: np.ndarray) -> bool:
        h = self._hash(img)
        return any((h - prev) <= self.threshold for prev in self._hashes)

    def add(self, img: np.ndarray) -> bool:
        h = self._hash(img)
        if any((h - prev) <= self.threshold for prev in self._hashes):
            return False
        self._hashes.append(h)
        return True

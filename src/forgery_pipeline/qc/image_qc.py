"""图像质量过滤（报告 §11.1）。"""
from __future__ import annotations
import cv2
import numpy as np


def check_image(img: np.ndarray, min_short_side: int = 256,
                max_aspect: float = 4.0) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if img is None or img.ndim != 3 or img.shape[2] != 3:
        return False, ["解码失败或非 RGB 三通道"]
    h, w = img.shape[:2]
    if min(h, w) < min_short_side:
        reasons.append(f"短边过小 (<{min_short_side})")
    if max(h, w) / max(min(h, w), 1) > max_aspect:
        reasons.append(f"极端长宽比 (>{max_aspect})")
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if float(gray.std()) < 3.0:
        reasons.append("大面积纯色/空白")
    return (len(reasons) == 0, reasons)

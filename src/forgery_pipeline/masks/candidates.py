"""候选 mask 的面积比与尺度分桶（报告 §6.4）。"""
from __future__ import annotations
from typing import Optional
import numpy as np

MIN_RATIO = 0.01
MAX_RATIO = 0.50


def area_ratio(mask: np.ndarray) -> float:
    return float((mask > 127).sum()) / float(mask.size)


def bucket_for_ratio(r: float) -> Optional[str]:
    if 0.01 <= r < 0.05:
        return "small"
    if 0.05 <= r < 0.20:
        return "mid"
    if 0.20 <= r <= 0.50:
        return "large"
    return None


def filter_and_sample(masks: list[np.ndarray]) -> list[tuple[np.ndarray, float, str]]:
    out: list[tuple[np.ndarray, float, str]] = []
    for m in masks:
        r = area_ratio(m)
        b = bucket_for_ratio(r)
        if b is not None and MIN_RATIO <= r <= MAX_RATIO:
            out.append((m, r, b))
    return out

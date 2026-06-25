"""mask 质量过滤（报告 §11.2）。"""
from __future__ import annotations
import cv2
import numpy as np
from forgery_pipeline.masks.candidates import area_ratio


def num_components(mask: np.ndarray) -> int:
    n, _ = cv2.connectedComponents((mask > 127).astype(np.uint8), 8)
    return max(0, n - 1)  # 去掉背景


def check_mask(mask: np.ndarray, max_components: int = 15) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    r = area_ratio(mask)
    if not (0.01 <= r <= 0.50):
        reasons.append(f"面积比越界 ({r:.3f} ∉ [0.01, 0.50])")
    if r > 0.99:
        reasons.append("mask 覆盖整图")
    if num_components(mask) > max_components:
        reasons.append("mask 过度碎片化")
    return (len(reasons) == 0, reasons)

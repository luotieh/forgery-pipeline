"""真实 real-fake pair 的差分伪标注（报告 §7.3/§7.4，MIML 思路工程近似）。"""
from __future__ import annotations
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


def align(real: np.ndarray, fake: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if fake.shape[:2] != real.shape[:2]:
        fake = cv2.resize(fake, (real.shape[1], real.shape[0]),
                          interpolation=cv2.INTER_AREA)
    return real, fake


def diff_map(real: np.ndarray, fake: np.ndarray) -> tuple[np.ndarray, float]:
    """RGB L1 + (1-SSIM) 融合差异图，范围 [0,1]。"""
    r = real.astype(np.float32)
    f = fake.astype(np.float32)
    rgb = np.abs(r - f).mean(axis=2) / 255.0
    rg = cv2.cvtColor(real, cv2.COLOR_RGB2GRAY)
    fg = cv2.cvtColor(fake, cv2.COLOR_RGB2GRAY)
    score, smap = ssim(rg, fg, full=True, data_range=255)
    ssim_diff = np.clip(1.0 - smap, 0.0, 1.0)
    diff = np.clip(0.5 * rgb + 0.5 * ssim_diff, 0.0, 1.0).astype(np.float32)
    return diff, float(score)


def coarse_mask(diff: np.ndarray, thresh: float = 0.15) -> np.ndarray:
    return ((diff >= thresh).astype(np.uint8)) * 255


def refine(mask: np.ndarray) -> np.ndarray:
    k = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx((mask > 127).astype(np.uint8), cv2.MORPH_CLOSE, k)
    return (closed * 255).astype(np.uint8)


def connected_component_filter(mask: np.ndarray, min_frac: float = 0.005) -> np.ndarray:
    binm = (mask > 127).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binm, 8)
    out = np.zeros_like(mask)
    total = mask.size
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_frac * total:
            out[labels == i] = 255
    return out


def _boundary_sharpness(diff: np.ndarray, mask: np.ndarray) -> float:
    """边界处差异梯度均值，作为边界清晰度代理，归一化到 [0,1]。"""
    edges = cv2.Canny((mask > 127).astype(np.uint8) * 255, 50, 150)
    if edges.sum() == 0:
        return 0.0
    grad = cv2.magnitude(cv2.Sobel(diff, cv2.CV_32F, 1, 0),
                         cv2.Sobel(diff, cv2.CV_32F, 0, 1))
    return float(np.clip(grad[edges > 0].mean(), 0.0, 1.0))


def pseudo_mask(real: np.ndarray, fake: np.ndarray,
                thresh: float = 0.15) -> tuple[np.ndarray, dict]:
    real, fake = align(real, fake)
    diff, ssim_score = diff_map(real, fake)
    coarse = coarse_mask(diff, thresh)
    refined = refine(coarse)
    final = connected_component_filter(refined)
    fg = final > 127
    confidence = float(diff[fg].mean()) if fg.any() else 0.0
    metrics = {
        "confidence": confidence,
        "ssim_score": ssim_score,
        "boundary_sharpness": _boundary_sharpness(diff, final),
        "area_ratio": float(fg.sum()) / float(final.size),
    }
    return final, metrics

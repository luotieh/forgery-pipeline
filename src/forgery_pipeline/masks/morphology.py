"""mask 形态学扰动（报告 §6.4：dilation/erosion/boundary blur/irregular）。"""
from __future__ import annotations
import cv2
import numpy as np


def _binarize(mask: np.ndarray) -> np.ndarray:
    return ((mask > 127).astype(np.uint8)) * 255


def _kernel(ksize: int) -> np.ndarray:
    ksize = max(1, ksize | 1)  # 取奇数
    return np.ones((ksize, ksize), np.uint8)


def dilate(mask: np.ndarray, ksize: int = 5) -> np.ndarray:
    return _binarize(cv2.dilate(_binarize(mask), _kernel(ksize)))


def erode(mask: np.ndarray, ksize: int = 5) -> np.ndarray:
    return _binarize(cv2.erode(_binarize(mask), _kernel(ksize)))


def boundary_blur(mask: np.ndarray, ksize: int = 5) -> np.ndarray:
    """高斯模糊后再二值化，得到轻微抖动的边界。"""
    k = max(3, ksize | 1)
    blurred = cv2.GaussianBlur(_binarize(mask), (k, k), 0)
    return _binarize(blurred)


def make_irregular(mask: np.ndarray, seed: int = 0, ksize: int = 5) -> np.ndarray:
    """在膨胀与腐蚀之间按平滑随机场切换，制造不规则但连贯的边界。

    随机场经高斯平滑以避免椒盐式碎片化，使边界扰动呈连贯波状。
    """
    rng = np.random.default_rng(seed)
    d = cv2.dilate(_binarize(mask), _kernel(ksize))
    e = cv2.erode(_binarize(mask), _kernel(ksize))
    field = rng.random(mask.shape).astype(np.float32)
    field = cv2.GaussianBlur(field, (0, 0), sigmaX=4.0)
    out = np.where(field < 0.5, d, e).astype(np.uint8)
    return _binarize(out)

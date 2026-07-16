"""paste-back 显式化（PATCH 7.3）：none=整图直出 / paste=硬回贴 / paste_feather=羽化混合。"""
from __future__ import annotations
import cv2
import numpy as np

def composite(orig_rgb_u8, gen_rgb_u8, mask01, mode: str = "none",
              feather_px: int = 8) -> np.ndarray:
    if mode == "none":
        return gen_rgb_u8
    assert orig_rgb_u8.shape == gen_rgb_u8.shape, "orig/gen 分辨率必须一致（先对齐再混合）"
    assert orig_rgb_u8.shape[:2] == np.asarray(mask01).shape, "mask 与图像 HxW 不一致"
    m = np.asarray(mask01, np.float32)
    if mode == "paste_feather":
        m = np.clip(cv2.GaussianBlur(m, (0, 0), float(feather_px)), 0.0, 1.0)
    elif mode != "paste":
        raise ValueError(f"未知 compositing: {mode!r}")
    m = m[..., None]
    out = orig_rgb_u8.astype(np.float32) * (1 - m) + gen_rgb_u8.astype(np.float32) * m
    return np.clip(np.round(out), 0, 255).astype(np.uint8)

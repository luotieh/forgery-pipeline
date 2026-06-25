"""后处理退化增强（报告 §10）。保持图像尺寸不变以兼容 mask；参数写入 Postprocess。"""
from __future__ import annotations
import cv2
import numpy as np
from forgery_pipeline.schema import Postprocess

JPEG_QUALITIES = [50, 60, 70, 80, 90, 95]
RESIZE_SCALES = [0.5, 0.67, 0.75, 1.5]
BLUR_KERNELS = [3, 5]
NOISE_SIGMAS = [3, 5, 10]


def apply_jpeg(img: np.ndarray, q: int) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                           [int(cv2.IMWRITE_JPEG_QUALITY), int(q)])
    dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)


def apply_resize(img: np.ndarray, scale: float) -> np.ndarray:
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                       interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def apply_blur(img: np.ndarray, k: int) -> np.ndarray:
    k = max(3, k | 1)
    return cv2.GaussianBlur(img, (k, k), 0)


def apply_noise(img: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def apply_social(img: np.ndarray) -> np.ndarray:
    return apply_jpeg(apply_resize(img, 0.75), 60)


def sample_and_apply(img: np.ndarray, rng: np.random.Generator
                     ) -> tuple[np.ndarray, Postprocess]:
    kind = rng.choice(["jpeg", "resize", "blur", "noise"])
    pp = Postprocess()
    if kind == "jpeg":
        q = int(rng.choice(JPEG_QUALITIES))
        img, pp.jpeg_quality = apply_jpeg(img, q), q
    elif kind == "resize":
        s = float(rng.choice(RESIZE_SCALES))
        img, pp.resize = apply_resize(img, s), str(s)
    elif kind == "blur":
        k = int(rng.choice(BLUR_KERNELS))
        img, pp.blur = apply_blur(img, k), f"k{k}"
    else:
        sigma = int(rng.choice(NOISE_SIGMAS))
        seed = int(rng.integers(0, 2**31 - 1))
        img, pp.noise = apply_noise(img, sigma, seed), f"sigma{sigma}"
    return img, pp

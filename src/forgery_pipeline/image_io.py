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


def chain(*nodes: str) -> str:
    """io_chain 组装：节点用 '>' 连接（PATCH 7.1）。"""
    return ">".join(nodes)


def load_and_resize(path, size: int | None = None) -> np.ndarray:
    """统一载入：同解码器；size 给定时中心裁剪为方形后 LANCZOS 缩放（真/假共享）。"""
    img = load_image(path)
    if size is not None:
        h, w = img.shape[:2]; side = min(h, w)
        y0, x0 = (h - side) // 2, (w - side) // 2
        img = img[y0:y0 + side, x0:x0 + side]
        img = np.asarray(PILImage.fromarray(img).resize((size, size), PILImage.LANCZOS))
    return img


def save_canonical(img: np.ndarray, path) -> None:
    """统一存储出口：主库一律 PNG（无损，不再引入有损层，PATCH 7.1）。"""
    assert str(path).endswith(".png"), f"canonical 存储必须 PNG: {path}"
    save_image(img, path)

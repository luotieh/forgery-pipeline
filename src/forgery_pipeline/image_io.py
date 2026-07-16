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


def resize_square(img: np.ndarray, size: int) -> np.ndarray:
    """中心裁剪为方形后 LANCZOS 缩放到 size×size（真/假共享；PATCH 7.1 起 load_and_resize
    的裁剪+缩放核，PATCH 9 Wave2 9.2c 抽出为独立函数，供 D0 多分辨率组摄取直接对内存中的
    ndarray 复用，无需先落盘再经 load_and_resize 重新解码）。"""
    h, w = img.shape[:2]; side = min(h, w)
    y0, x0 = (h - side) // 2, (w - side) // 2
    img = img[y0:y0 + side, x0:x0 + side]
    return np.asarray(PILImage.fromarray(img).resize((size, size), PILImage.LANCZOS))


def load_and_resize(path, size: int | None = None) -> np.ndarray:
    """统一载入：同解码器；size 给定时经 resize_square 中心裁剪为方形后 LANCZOS 缩放。"""
    img = load_image(path)
    if size is not None:
        img = resize_square(img, size)
    return img


def chain_resolution(io_chain) -> int | None:
    """从 io_chain 中取分辨率节点 rs{N} 的 N（PATCH 9 Wave2 9.2c：多分辨率组路由查表用，
    如 D0 行按基准组分辨率过滤、grid 按分辨率组反查底图同源行）。查无/字段缺失返回 None；
    "legacy" 整体视为不可拆分的谱系标记（同 validate.nongen_chain 的口径），同样返回 None。"""
    if not io_chain or io_chain == "legacy":
        return None
    for node in io_chain.split(">"):
        if node.startswith("rs") and node[2:].isdigit():
            return int(node[2:])
    return None


def save_canonical(img: np.ndarray, path) -> None:
    """统一存储出口：主库一律 PNG（无损，不再引入有损层，PATCH 7.1）。"""
    assert str(path).endswith(".png"), f"canonical 存储必须 PNG: {path}"
    save_image(img, path)

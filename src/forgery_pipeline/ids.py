"""稳定、确定性的样本 ID 生成。"""
from __future__ import annotations
import hashlib
import numpy as np


def content_hash(img: np.ndarray) -> bytes:
    """对图像原始字节求 sha1 摘要。"""
    return hashlib.sha1(np.ascontiguousarray(img).tobytes()).digest()


def make_image_id(prefix: str, payload: bytes | str) -> str:
    """生成 `<prefix>_<sha1[:12]>` 形式的稳定 ID。"""
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    digest = hashlib.sha1(payload).hexdigest()[:12]
    return f"{prefix}_{digest}"

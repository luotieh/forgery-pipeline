"""生成质量过滤（报告 §11.3）。mock 用低层统计量近似；真实可接入美学/一致性模型。"""
from __future__ import annotations
import cv2
import numpy as np


def check_generation(img: np.ndarray, prompt: str | None = None
                     ) -> tuple[bool, list[str], str]:
    reasons: list[str] = []
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    std = float(gray.std())
    if std < 5.0:
        reasons.append("近乎纯色，疑似生成失败")
    bucket = "high" if std > 40 else "mid" if std > 15 else "low"
    # 真实后端可在此加入 prompt-图像一致性（如 CLIPScore）；mock 跳过。
    return (len(reasons) == 0, reasons, bucket)

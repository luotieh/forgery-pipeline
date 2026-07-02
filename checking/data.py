"""读取本管线 manifest 并批量提取特征。"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from forgery_pipeline import manifest, image_io


def load(manifest_path) -> list:
    return manifest.read_jsonl(manifest_path)


def image_of(root, s) -> np.ndarray:
    return image_io.load_image(Path(root) / s.image_path)


def mask_of(root, s):
    if not s.mask_path:
        return None
    return image_io.load_mask(Path(root) / s.mask_path)


def profiles(extractor, root, samples):
    """返回 (X:(N,D) float, kept:list[Sample])，跳过读失败的样本。"""
    X, kept = [], []
    for s in samples:
        try:
            img = image_of(root, s)
        except Exception:
            continue
        X.append(extractor.profile(img)); kept.append(s)
    return (np.array(X, float) if X else np.zeros((0, 1))), kept

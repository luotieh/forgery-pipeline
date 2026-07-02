"""闸门信号提取器：CPU 多尺度残差代理 + 真实 SD2 骨架。"""
from __future__ import annotations
from abc import ABC, abstractmethod
import cv2
import numpy as np


def _region_descriptors(agg: np.ndarray) -> np.ndarray:
    """高残差区（agg≥p90）的几何：归一化重心 cy/cx、扩散、边界带/中心区残差比。"""
    H, W = agg.shape
    thr = np.quantile(agg, 0.9)
    ys, xs = np.nonzero(agg >= thr)
    if len(ys) == 0:
        cy = cx = 0.5; spread = 0.0
    else:
        cy = float(ys.mean()) / H; cx = float(xs.mean()) / W
        spread = float(np.sqrt(((ys / H - cy) ** 2 + (xs / W - cx) ** 2).mean()))
    b = max(1, min(H, W) // 8)
    border = np.concatenate([agg[:b].ravel(), agg[-b:].ravel(),
                             agg[:, :b].ravel(), agg[:, -b:].ravel()])
    center = agg[b:H - b, b:W - b]
    bc = float(border.mean() / (center.mean() + 1e-8)) if center.size else 1.0
    return np.array([cy, cx, spread, bc], np.float32)


class ResidualExtractor(ABC):
    sigmas: list

    @abstractmethod
    def residual_stack(self, image: np.ndarray) -> np.ndarray:
        """返回 (K,H,W) float[0,1] 每尺度残差图。"""

    def profile(self, image: np.ndarray) -> np.ndarray:
        rs = self.residual_stack(image)
        per_scale = np.concatenate([rs.mean(axis=(1, 2)), rs.std(axis=(1, 2))])
        agg = rs.mean(axis=0)
        q = np.quantile(agg, [0.1, 0.5, 0.9])
        return np.concatenate([per_scale, q, _region_descriptors(agg)]).astype(np.float32)

    def residual_map(self, image: np.ndarray) -> np.ndarray:
        return self.residual_stack(image).mean(axis=0).astype(np.float32)

    def detection_score(self, image: np.ndarray) -> float:
        """残差图 top-decile 均值：对局部编辑比全局均值更灵敏。"""
        rm = self.residual_map(image)
        thr = float(np.quantile(rm, 0.90))
        top = rm[rm >= thr]
        return float(top.mean()) if top.size else float(rm.mean())


class MultiSigmaResidual(ResidualExtractor):
    """多尺度「高斯重建残差」CPU 代理（快、确定）。"""
    def __init__(self, sigmas=(3, 5, 9, 17, 33)):
        self.sigmas = list(sigmas)

    def residual_stack(self, image: np.ndarray) -> np.ndarray:
        g = image.astype(np.float32)
        maps = []
        for k in self.sigmas:
            recon = cv2.GaussianBlur(g, (0, 0), sigmaX=float(k))
            maps.append(np.abs(g - recon).mean(axis=2) / 255.0)
        return np.stack(maps).astype(np.float32)


class DiffusersSD2Residual(ResidualExtractor):
    """真实多 σ Tweedie 残差骨架（冻结 SD2）。需 `pip install .[real]` + GPU。"""
    def __init__(self, model_id: str = "stabilityai/stable-diffusion-2-base",
                 device: str = "cuda", sigmas=(0.1, 0.2, 0.4, 0.6, 0.8)):
        self.sigmas = list(sigmas)
        try:
            import torch  # noqa: F401
            import diffusers  # noqa: F401
        except ImportError as e:
            raise NotImplementedError(
                "真实 SD2 提取器未启用：请 `pip install .[real]`（torch/diffusers）并提供 GPU。") from e
        raise NotImplementedError(
            "参考骨架：VAE 编码 z0 → 多 t 加噪 z_t → UNet ε̂ → r_ε(t)=‖ε−ε̂‖²、"
            "一步反演 ẑ0 → r_x(t)=‖z0−ẑ0‖²，堆叠成 residual_stack。")

    def residual_stack(self, image):
        raise NotImplementedError


def get_extractor(name: str = "multisigma") -> ResidualExtractor:
    if name == "multisigma":
        return MultiSigmaResidual()
    if name == "real":
        return DiffusersSD2Residual()
    raise ValueError(f"未知 extractor: {name!r}（可选 multisigma / real）")

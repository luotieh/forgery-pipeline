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
    """真实多 σ Tweedie 残差（冻结 SD2）。懒加载；需 diffusers + GPU。"""
    def __init__(self, model_id: str = "stabilityai/stable-diffusion-2-base",
                 device: str = "cuda", timesteps=(50, 150, 300, 500, 700)):
        self.model_id, self.device = model_id, device
        self.sigmas = list(timesteps)          # 多 σ = 多 t
        self._unet = None

    def _ensure(self):
        if self._unet is not None:
            return
        import torch
        from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
        from transformers import CLIPTextModel, CLIPTokenizer
        m = self.model_id
        self._vae = AutoencoderKL.from_pretrained(
            m, subfolder="vae", torch_dtype=torch.float32).to(self.device).eval()
        self._unet = UNet2DConditionModel.from_pretrained(
            m, subfolder="unet", torch_dtype=torch.float16).to(self.device).eval()
        self._abar = DDPMScheduler.from_pretrained(m, subfolder="scheduler").alphas_cumprod
        tok = CLIPTokenizer.from_pretrained(m, subfolder="tokenizer")
        te = CLIPTextModel.from_pretrained(
            m, subfolder="text_encoder", torch_dtype=torch.float16).to(self.device).eval()
        ids = tok("", padding="max_length", max_length=tok.model_max_length,
                  return_tensors="pt").input_ids.to(self.device)
        with torch.no_grad():
            self._null = te(ids)[0]

    def residual_stack(self, image: np.ndarray) -> np.ndarray:
        import hashlib
        import torch
        self._ensure()
        H, W = image.shape[:2]
        img512 = cv2.resize(image, (512, 512))
        x = (torch.from_numpy(img512).float().permute(2, 0, 1)[None] / 127.5 - 1.0
             ).to(self.device, torch.float32)
        seed_base = int.from_bytes(
            hashlib.sha1(np.ascontiguousarray(img512).tobytes()).digest()[:4], "big")
        maps = []
        with torch.no_grad():
            z0 = self._vae.encode(x).latent_dist.mean * self._vae.config.scaling_factor
            for t in self.sigmas:
                g = torch.Generator(self.device).manual_seed((seed_base + int(t)) & 0x7FFFFFFF)
                eps = torch.randn(z0.shape, generator=g, device=self.device, dtype=torch.float32)
                abar = self._abar[int(t)].to(self.device).float()
                zt = abar.sqrt() * z0 + (1 - abar).sqrt() * eps
                eps_hat = self._unet(zt.half(), int(t),
                                     encoder_hidden_states=self._null).sample.float()
                r_eps = ((eps - eps_hat) ** 2).mean(dim=1)[0]
                z0_hat = (zt - (1 - abar).sqrt() * eps_hat) / abar.sqrt()
                r_x = ((z0 - z0_hat) ** 2).mean(dim=1)[0]
                mp = (r_eps + r_x).cpu().numpy()
                p99 = float(np.percentile(mp, 99)) + 1e-8
                mp = np.clip(mp / p99, 0.0, 1.0)
                maps.append(cv2.resize(mp.astype(np.float32), (W, H)))
        return np.stack(maps).astype(np.float32)


def get_extractor(name: str = "multisigma") -> ResidualExtractor:
    if name == "multisigma":
        return MultiSigmaResidual()
    if name == "real":
        return DiffusersSD2Residual()
    raise ValueError(f"未知 extractor: {name!r}（可选 multisigma / real）")

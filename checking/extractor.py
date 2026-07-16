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


def _direction_descriptors(eps_stack: np.ndarray, x_stack: np.ndarray) -> np.ndarray:
    """逐 t 的方向/相位特征：把被 r_eps+r_x 坍缩掉的信息拿回来（Phase A①）。

    输入 (K,H,W) 的 ε 误差图栈与 x0 误差图栈；输出长度 4K−1：
      - 分离双通道 per-t 均值 [ε_mean(t), x_mean(t)]           (2K)
      - 相邻 t 的 ε 误差图方向余弦（去噪轨迹形状而非幅值）     (K−1)
      - 每 t 的 ε/x 比值（轨迹在流形上的相位签名）             (K)
    """
    eps_stack = np.asarray(eps_stack, np.float32)
    x_stack = np.asarray(x_stack, np.float32)
    K = eps_stack.shape[0]
    eps_mean = eps_stack.mean(axis=(1, 2))
    x_mean = x_stack.mean(axis=(1, 2))
    cos = []
    flat = eps_stack.reshape(K, -1)
    for i in range(K - 1):
        a, b = flat[i], flat[i + 1]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        cos.append(float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0)
    ratio = eps_mean / (x_mean + 1e-8)
    return np.concatenate([eps_mean, x_mean, np.array(cos, np.float32), ratio]).astype(np.float32)


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


class DiffusersResidual(ResidualExtractor):
    """真实多 σ Tweedie 残差（冻结扩散先验）。懒加载；需 diffusers + GPU。

    默认用 SD1.5（512, ε-prediction）——stabilityai SD2 repo 已从 HF 下架；
    方法与具体先验无关（PAPER §2.1 的多 σ 去噪残差）。
    """
    def __init__(self, model_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
                 device: str = "cuda", timesteps=(50, 150, 300, 500, 700),
                 direction_features: bool = True):
        self.model_id, self.device = model_id, device
        self.sigmas = list(timesteps)          # 多 σ = 多 t
        self.direction_features = direction_features
        self._unet = None
        self._eps_stack = None                 # residual_stack 计算时缓存的分离分量
        self._x_stack = None

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
        maps, eps_maps, x_maps = [], [], []
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
                # 缓存分离分量供方向特征（Phase A①）——原始幅值，方向余弦对尺度不敏感
                eps_maps.append(cv2.resize(r_eps.cpu().numpy().astype(np.float32), (W, H)))
                x_maps.append(cv2.resize(r_x.cpu().numpy().astype(np.float32), (W, H)))
        self._eps_stack = np.stack(eps_maps).astype(np.float32)
        self._x_stack = np.stack(x_maps).astype(np.float32)
        return np.stack(maps).astype(np.float32)

    def profile(self, image: np.ndarray) -> np.ndarray:
        """幅值剖面（基类，第一维仍是单尺度幅值→gate1 单σ基线口径不破）
        + 可选逐 t 方向/相位特征（Phase A①，把坍缩掉的方向拿回来）。"""
        base = super().profile(image)          # 内部调用 residual_stack → 刷新 _eps/_x_stack
        if not self.direction_features or self._eps_stack is None:
            return base
        extra = _direction_descriptors(self._eps_stack, self._x_stack)
        return np.concatenate([base, extra]).astype(np.float32)


def get_extractor(name: str = "multisigma") -> ResidualExtractor:
    if name == "multisigma":
        return MultiSigmaResidual()
    if name == "real":
        # 消融开关：CHECKING_DIRECTION_FEATURES=0 → 仅幅值 baseline，隔离方向特征贡献
        import os
        dir_feat = os.environ.get("CHECKING_DIRECTION_FEATURES", "1") != "0"
        return DiffusersResidual(direction_features=dir_feat)
    raise ValueError(f"未知 extractor: {name!r}（可选 multisigma / real）")

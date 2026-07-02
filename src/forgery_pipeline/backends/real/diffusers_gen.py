"""真实 diffusers 生成后端。懒加载：__init__ 不 import diffusers/不占显存。

注：stabilityai 的 SD2 repo 已从 HF 下架，改用可获取的 SD1.5（512 分辨率、
ε-prediction，与残差提取器口径一致）。方法与具体先验模型无关。
"""
from __future__ import annotations
import cv2
import numpy as np
from PIL import Image
from forgery_pipeline.backends import base

_SD = "stable-diffusion-v1-5/stable-diffusion-v1-5"
_SD_INPAINT = "stable-diffusion-v1-5/stable-diffusion-inpainting"


def _to_uint8_like(pil_img, ref: np.ndarray) -> np.ndarray:
    arr = np.asarray(pil_img.convert("RGB"), np.uint8)
    if arr.shape[:2] != ref.shape[:2]:
        arr = cv2.resize(arr, (ref.shape[1], ref.shape[0]))
    return arr


class DiffusersImg2Img(base.Img2ImgGenerator):
    def __init__(self, model_id: str = _SD, device: str = "cuda", dtype: str = "fp16"):
        self.model_id, self.device, self.dtype = model_id, device, dtype
        self.name, self.family = "stable-diffusion-1-5-img2img", "diffusion"
        self._pipe = None

    def _ensure(self):
        if self._pipe is not None:
            return
        import torch
        from diffusers import AutoPipelineForImage2Image
        self._pipe = AutoPipelineForImage2Image.from_pretrained(
            self.model_id, torch_dtype=torch.float16,
            safety_checker=None, requires_safety_checker=False).to(self.device)
        self._pipe.set_progress_bar_config(disable=True)
        self._pipe.enable_attention_slicing()

    def img2img(self, image, prompt, strength, params):
        import torch
        self._ensure()
        seed = int(params.get("seed", 0))
        g = torch.Generator(self.device).manual_seed(seed)
        out = self._pipe(prompt=prompt or "a realistic high quality photo",
                         image=Image.fromarray(image), strength=float(strength),
                         num_inference_steps=int(params.get("steps", 30)),
                         guidance_scale=float(params.get("cfg_scale", 7.5)), generator=g)
        meta = {"generator_name": self.name, "generator_family": self.family, "seed": seed,
                "strength": float(strength), "steps": int(params.get("steps", 30)),
                "cfg_scale": float(params.get("cfg_scale", 7.5)), "sampler": "default"}
        return _to_uint8_like(out.images[0], image), meta


class DiffusersInpainter(base.Inpainter):
    def __init__(self, model_id: str = _SD_INPAINT, device: str = "cuda", dtype: str = "fp16"):
        self.model_id, self.device, self.dtype = model_id, device, dtype
        self.name, self.family = "stable-diffusion-1-5-inpaint", "diffusion"
        self._pipe = None

    def _ensure(self):
        if self._pipe is not None:
            return
        import torch
        from diffusers import AutoPipelineForInpainting
        self._pipe = AutoPipelineForInpainting.from_pretrained(
            self.model_id, torch_dtype=torch.float16,
            safety_checker=None, requires_safety_checker=False).to(self.device)
        self._pipe.set_progress_bar_config(disable=True)
        self._pipe.enable_attention_slicing()

    def inpaint(self, image, mask, prompt, params):
        import torch
        self._ensure()
        seed = int(params.get("seed", 0))
        g = torch.Generator(self.device).manual_seed(seed)
        out = self._pipe(prompt=prompt or "a realistic object, high quality",
                         image=Image.fromarray(image), mask_image=Image.fromarray(mask),
                         num_inference_steps=int(params.get("steps", 30)),
                         guidance_scale=float(params.get("cfg_scale", 7.5)), generator=g)
        meta = {"generator_name": self.name, "generator_family": self.family, "seed": seed}
        return _to_uint8_like(out.images[0], image), meta

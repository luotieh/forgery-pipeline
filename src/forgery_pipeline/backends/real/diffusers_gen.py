"""真实 diffusers 生成后端（SD2）。懒加载：__init__ 不 import diffusers/不占显存。"""
from __future__ import annotations
import cv2
import numpy as np
from PIL import Image
from forgery_pipeline.backends import base

_SD2 = "stabilityai/stable-diffusion-2-base"
_SD2_INPAINT = "stabilityai/stable-diffusion-2-inpainting"


def _to_uint8_like(pil_img, ref: np.ndarray) -> np.ndarray:
    arr = np.asarray(pil_img.convert("RGB"), np.uint8)
    if arr.shape[:2] != ref.shape[:2]:
        arr = cv2.resize(arr, (ref.shape[1], ref.shape[0]))
    return arr


class DiffusersImg2Img(base.Img2ImgGenerator):
    def __init__(self, model_id: str = _SD2, device: str = "cuda", dtype: str = "fp16"):
        self.model_id, self.device, self.dtype = model_id, device, dtype
        self.name, self.family = "stable-diffusion-2-img2img", "diffusion"
        self._pipe = None

    def _ensure(self):
        if self._pipe is not None:
            return
        import torch
        from diffusers import AutoPipelineForImage2Image
        self._pipe = AutoPipelineForImage2Image.from_pretrained(
            self.model_id, torch_dtype=torch.float16).to(self.device)
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
    def __init__(self, model_id: str = _SD2_INPAINT, device: str = "cuda", dtype: str = "fp16"):
        self.model_id, self.device, self.dtype = model_id, device, dtype
        self.name, self.family = "stable-diffusion-2-inpaint", "diffusion"
        self._pipe = None

    def _ensure(self):
        if self._pipe is not None:
            return
        import torch
        from diffusers import AutoPipelineForInpainting
        self._pipe = AutoPipelineForInpainting.from_pretrained(
            self.model_id, torch_dtype=torch.float16).to(self.device)
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

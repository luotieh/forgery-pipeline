"""真实 diffusers 生成后端。懒加载：__init__ 不 import diffusers/不占显存。

注：stabilityai 的 SD2 repo 已从 HF 下架，改用可获取的 SD1.5（512 分辨率、
ε-prediction，与残差提取器口径一致）。方法与具体先验模型无关。

多生成器：按 spec name 映射到不同底层模型（SD1.5 / Kandinsky 2.2 异族先验），
供 gate2 cross_model 与 gate3 跨生成器判定。多模型合计超 8GB 显存，
统一 enable_model_cpu_offload（峰值≈最大单模块，安全换速度）。
"""
from __future__ import annotations
import cv2
import numpy as np
from PIL import Image
from forgery_pipeline.backends import base

_SD = "stable-diffusion-v1-5/stable-diffusion-v1-5"
_SD_INPAINT = "stable-diffusion-v1-5/stable-diffusion-inpainting"
_KANDINSKY_INPAINT = "kandinsky-community/kandinsky-2-2-decoder-inpaint"
_SDXL = "stabilityai/stable-diffusion-xl-base-1.0"
_SDXL_INPAINT = "stabilityai/stable-diffusion-xl-1.0-inpainting-0.1"

# spec name -> (model_id, family)；未列出的 name 落到 SD1.5（族仍如实取 spec）
# sdxl-img2img/sdxl-inpaint（PATCH 9 Wave2 T3）：SDXL 1024 分辨率先验，供 grid_ops 的
# resolution_groups 高分辨率组使用；懒加载路径复用 _load_pipeline 不变，fp16 变体已在
# 数据盘预缓存；GPU 冒烟随 Wave2 T4（分辨率组接线）另行验证，此处仅代码级映射。
IMG2IMG_MODELS = {
    "stable-diffusion-img2img": (_SD, "diffusion"),
    "sdxl-img2img": (_SDXL, "diffusion-sdxl"),
}
INPAINT_MODELS = {
    "stable-diffusion-inpaint": (_SD_INPAINT, "diffusion"),
    "kandinsky-inpaint": (_KANDINSKY_INPAINT, "kandinsky"),
    "sdxl-inpaint": (_SDXL_INPAINT, "diffusion-sdxl"),
}


def _to_uint8_like(pil_img, ref: np.ndarray) -> np.ndarray:
    arr = np.asarray(pil_img.convert("RGB"), np.uint8)
    if arr.shape[:2] != ref.shape[:2]:
        arr = cv2.resize(arr, (ref.shape[1], ref.shape[0]))
    return arr


def _load_pipeline(auto_cls, model_id: str, device: str):
    import torch
    kwargs = {"torch_dtype": torch.float16}
    if "stable-diffusion" in model_id:
        kwargs.update(safety_checker=None, requires_safety_checker=False)
    pipe = auto_cls.from_pretrained(model_id, **kwargs)
    pipe.enable_model_cpu_offload(device=device)
    pipe.set_progress_bar_config(disable=True)
    pipe.enable_attention_slicing()
    return pipe


class DiffusersImg2Img(base.Img2ImgGenerator):
    def __init__(self, model_id: str = _SD, device: str = "cuda", dtype: str = "fp16",
                 name: str = "stable-diffusion-img2img", family: str = "diffusion"):
        self.model_id, self.device, self.dtype = model_id, device, dtype
        self.name, self.family = name, family
        self._pipe = None

    def _ensure(self):
        if self._pipe is not None:
            return
        from diffusers import AutoPipelineForImage2Image
        self._pipe = _load_pipeline(AutoPipelineForImage2Image, self.model_id, self.device)

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
    def __init__(self, model_id: str = _SD_INPAINT, device: str = "cuda", dtype: str = "fp16",
                 name: str = "stable-diffusion-inpaint", family: str = "diffusion"):
        self.model_id, self.device, self.dtype = model_id, device, dtype
        self.name, self.family = name, family
        self._pipe = None

    def _ensure(self):
        if self._pipe is not None:
            return
        from diffusers import AutoPipelineForInpainting
        self._pipe = _load_pipeline(AutoPipelineForInpainting, self.model_id, self.device)

    def inpaint(self, image, mask, prompt, params):
        import torch
        self._ensure()
        seed = int(params.get("seed", 0))
        g = torch.Generator(self.device).manual_seed(seed)
        # 审计守卫（PATCH 7.3）：diffusers 默认管线不做隐式 paste-back/overlay，
        # padding_mask_crop 会让管线自行回贴掩码外区域——回贴必须走显式的 compositing()。
        assert "padding_mask_crop" not in params, "禁止隐式 paste-back（PATCH 7.3 审计）：compositing 必须显式"
        out = self._pipe(prompt=prompt or "a realistic object, high quality",
                         image=Image.fromarray(image), mask_image=Image.fromarray(mask),
                         num_inference_steps=int(params.get("steps", 30)),
                         guidance_scale=float(params.get("cfg_scale", 7.5)), generator=g)
        meta = {"generator_name": self.name, "generator_family": self.family, "seed": seed}
        return _to_uint8_like(out.images[0], image), meta


class SDVaeRoundtrip:
    """SD1.5 VAE encode→decode（fp32 防 NaN，懒加载；PATCH 7.2）。名称 sd15 记入 io_chain。"""
    name = "sd15"

    def __init__(self, model_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
                 device: str = "cuda"):
        self.model_id, self.device, self._vae = model_id, device, None

    def roundtrip(self, img):
        import torch
        from diffusers import AutoencoderKL
        if self._vae is None:
            self._vae = AutoencoderKL.from_pretrained(
                self.model_id, subfolder="vae", torch_dtype=torch.float32
            ).to(self.device).eval()
        import numpy as np
        x = (torch.from_numpy(img).float().permute(2, 0, 1)[None] / 127.5 - 1.0).to(self.device)
        with torch.no_grad():
            y = self._vae.decode(self._vae.encode(x).latent_dist.mean).sample
        y = ((y.clamp(-1, 1) + 1) * 127.5).round().byte()[0].permute(1, 2, 0).cpu().numpy()
        return y

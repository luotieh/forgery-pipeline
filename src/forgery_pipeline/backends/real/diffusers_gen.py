"""真实整图/重绘生成器适配器骨架（diffusers）。需 `pip install .[real]` 与 GPU/权重。"""
from __future__ import annotations
from forgery_pipeline.backends import base


class DiffusersWholeGenerator(base.WholeImageGenerator):
    def __init__(self, model_id: str, device: str = "cuda"):
        try:
            from diffusers import AutoPipelineForText2Image  # noqa: F401
        except ImportError as e:
            raise NotImplementedError(
                "未安装 diffusers：`pip install .[real]`。") from e
        raise NotImplementedError("参考骨架：在此加载 pipeline 并实现 generate()。")

    def generate(self, prompt, params):
        raise NotImplementedError


class DiffusersInpainter(base.Inpainter):
    def __init__(self, model_id: str, device: str = "cuda"):
        try:
            from diffusers import AutoPipelineForInpainting  # noqa: F401
        except ImportError as e:
            raise NotImplementedError(
                "未安装 diffusers：`pip install .[real]`。") from e
        raise NotImplementedError("参考骨架：在此加载 inpaint pipeline 并实现 inpaint()。")

    def inpaint(self, image, mask, prompt, params):
        raise NotImplementedError

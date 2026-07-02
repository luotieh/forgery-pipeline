"""按名称解析 backend。默认 mock；真实后端给出清晰的启用提示。"""
from __future__ import annotations
from forgery_pipeline.backends import base, mock

_HINTS = {
    "real:diffusers": "[real]",
    "real:sam": "[sam]",
    "real:mllm": "[mllm]",
}


def _unsupported(backend: str):
    extra = _HINTS.get(backend, "[real]")
    raise NotImplementedError(
        f"backend {backend!r} 未启用：请 `pip install .{extra}` 安装依赖、提供模型权重/API key，"
        f"并在 forgery_pipeline/backends/real/ 中完成适配器实现。当前可用：'mock'。")


def get_image_source(backend: str, seed: int = 0) -> base.ImageSource:
    if backend == "mock":
        return mock.MockImageSource(seed=seed)
    if backend == "real":
        import os
        from forgery_pipeline.backends.real.local_source import LocalImageSource
        return LocalImageSource(os.environ.get("FORGERY_REAL_IMAGE_DIR", "data/real_base"),
                                seed=seed)
    _unsupported(backend)


def get_whole_generator(backend: str, name: str, family: str) -> base.WholeImageGenerator:
    if backend == "mock":
        return mock.MockWholeImageGenerator(name=name, family=family)
    _unsupported(backend)


def get_inpainter(backend: str, name: str, family: str) -> base.Inpainter:
    if backend == "mock":
        return mock.MockInpainter(name=name, family=family)
    if backend == "real":
        from forgery_pipeline.backends.real.diffusers_gen import DiffusersInpainter
        return DiffusersInpainter()
    _unsupported(backend)


def get_segmenter(backend: str, seed: int = 0) -> base.Segmenter:
    if backend == "mock":
        return mock.MockSegmenter(seed=seed)
    if backend == "real":
        return mock.MockSegmenter(seed=seed)  # probe 用几何掩码，占位无害
    _unsupported(backend)


def get_explainer(backend: str) -> base.Explainer:
    if backend == "mock":
        return mock.MockExplainer()
    _unsupported(backend)


def get_img2img(backend: str, name: str, family: str) -> base.Img2ImgGenerator:
    if backend == "mock":
        return mock.MockImg2Img(name=name, family=family)
    if backend == "real":
        from forgery_pipeline.backends.real.diffusers_gen import DiffusersImg2Img
        return DiffusersImg2Img()
    _unsupported(backend)

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


_REAL_CACHE: dict[str, object] = {}


def _real_cached(key: str, factory):
    # probe 循环内每个样本都取生成器；缓存实例避免重复加载扩散管线
    if key not in _REAL_CACHE:
        _REAL_CACHE[key] = factory()
    return _REAL_CACHE[key]


def get_inpainter(backend: str, name: str, family: str) -> base.Inpainter:
    if backend == "mock":
        return mock.MockInpainter(name=name, family=family)
    if backend == "real":
        from forgery_pipeline.backends.real import diffusers_gen as dg
        model_id, fam = dg.INPAINT_MODELS.get(name, (None, family))
        model_id = model_id or dg.DiffusersInpainter().model_id
        return _real_cached(f"inpaint:{name}", lambda: dg.DiffusersInpainter(
            model_id=model_id, name=name, family=fam))
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
        from forgery_pipeline.backends.real import diffusers_gen as dg
        model_id, fam = dg.IMG2IMG_MODELS.get(name, (None, family))
        model_id = model_id or dg.DiffusersImg2Img().model_id
        return _real_cached(f"img2img:{name}", lambda: dg.DiffusersImg2Img(
            model_id=model_id, name=name, family=fam))
    _unsupported(backend)


def get_vae_rt(backend: str) -> base.VaeRoundtrip:
    if backend == "mock":
        return mock.MockVaeRoundtrip()
    if backend == "real":
        from forgery_pipeline.backends.real import diffusers_gen as dg
        return dg.SDVaeRoundtrip()
    _unsupported(backend)

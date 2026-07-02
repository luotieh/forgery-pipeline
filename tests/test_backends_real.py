import numpy as np
from forgery_pipeline import image_io
from forgery_pipeline.backends.real.local_source import LocalImageSource
from forgery_pipeline.backends import registry


def test_local_image_source_reads_and_crops(tmp_path):
    for i in range(2):
        img = np.random.default_rng(i).integers(0, 256, (300, 400, 3), dtype=np.uint8)
        image_io.save_image(img, tmp_path / f"p{i}.jpg")
    got = list(LocalImageSource(tmp_path, size=128).iter_images(2))
    assert len(got) == 2
    im, meta = got[0]
    assert im.shape == (128, 128, 3) and im.dtype == np.uint8
    assert meta["source_dataset"] == "local"


def test_registry_real_image_source(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGERY_REAL_IMAGE_DIR", str(tmp_path))
    assert isinstance(registry.get_image_source("real"), LocalImageSource)


def test_registry_real_generators_lazy():
    from forgery_pipeline.backends.real.diffusers_gen import DiffusersImg2Img, DiffusersInpainter
    from forgery_pipeline.backends import mock
    assert isinstance(registry.get_img2img("real", "x", "y"), DiffusersImg2Img)
    assert isinstance(registry.get_inpainter("real", "x", "y"), DiffusersInpainter)
    # probe 用几何掩码，real segmenter 占位为 mock
    assert isinstance(registry.get_segmenter("real"), mock.MockSegmenter)
    # 构造不触发模型加载（无 GPU/无 diffusers 也能构造）
    assert DiffusersImg2Img()._pipe is None


def test_registry_real_generators_cached():
    # probe 循环内反复取生成器；real 后端必须复用实例，否则每个样本重载管线
    assert registry.get_img2img("real", "x", "y") is registry.get_img2img("real", "z", "w")
    assert registry.get_inpainter("real", "x", "y") is registry.get_inpainter("real", "z", "w")

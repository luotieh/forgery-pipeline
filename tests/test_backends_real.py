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
    # probe 循环内反复取生成器；real 后端必须按 name 复用实例，否则每个样本重载管线
    a = registry.get_img2img("real", "stable-diffusion-img2img", "diffusion")
    assert a is registry.get_img2img("real", "stable-diffusion-img2img", "diffusion")
    b = registry.get_inpainter("real", "stable-diffusion-inpaint", "diffusion")
    assert b is registry.get_inpainter("real", "stable-diffusion-inpaint", "diffusion")


def test_real_generators_config_names_all_mapped():
    # real 清单里的 name 必须有真实模型映射，否则会静默落到 SD1.5、标签撒谎
    from forgery_pipeline.config import load_generators
    from forgery_pipeline.backends.real import diffusers_gen as dg
    _, inps, imgs = load_generators("configs/generators.real.yaml")
    assert inps and imgs
    for s in inps:
        assert s.name in dg.INPAINT_MODELS, s.name
        assert dg.INPAINT_MODELS[s.name][1] == s.family
    for s in imgs:
        assert s.name in dg.IMG2IMG_MODELS, s.name
        assert dg.IMG2IMG_MODELS[s.name][1] == s.family


def test_registry_real_multi_generator_mapping():
    # 不同 name → 不同底层模型实例；meta 标签如实（跨生成器闸门的前提）
    sd = registry.get_inpainter("real", "stable-diffusion-inpaint", "diffusion")
    kd = registry.get_inpainter("real", "kandinsky-inpaint", "kandinsky")
    assert sd is not kd
    assert "kandinsky" in kd.model_id
    assert kd.name == "kandinsky-inpaint" and kd.family == "kandinsky"
    assert sd.name == "stable-diffusion-inpaint" and sd.family == "diffusion"
    assert kd._pipe is None  # 懒加载不变

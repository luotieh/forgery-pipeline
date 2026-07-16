import numpy as np
from forgery_pipeline.backends import registry


def test_mock_vae_rt_global_deterministic():
    rt = registry.get_vae_rt("mock")
    img = np.random.default_rng(0).integers(0, 256, (32, 32, 3), np.uint8)
    a, b = rt.roundtrip(img), rt.roundtrip(img)
    assert np.array_equal(a, b)                       # 确定性
    assert a.shape == img.shape and a.dtype == np.uint8
    assert (a != img).mean() > 0.5                    # 全局印记：过半像素被触碰


def test_real_vae_rt_lazy():
    rt = registry.get_vae_rt("real")
    assert rt._vae is None                            # 构造不加载模型


def test_pipeline_inserts_vae_rt_rows(tmp_path):
    from forgery_pipeline.config import PipelineConfig, StageScales, GeneratorSpec
    from forgery_pipeline.pipeline import run_pipeline
    cfg = PipelineConfig(
        out_dir=str(tmp_path / "run"), seed=0, backend="mock",
        stages={"d0": True, "d1": False, "d2": True, "d3": True, "d4": False,
                "postprocess": False, "split": True},
        scales=StageScales(d0=12, d2=6, d3=4),
        inpainters=[GeneratorSpec("i1", "diffusion", "inpaint")],
        vae_rt_frac=0.5)
    run_pipeline(cfg)
    from forgery_pipeline import manifest
    rows = manifest.read_jsonl(tmp_path / "run" / "manifest.jsonl")
    rt = [r for r in rows if r.sample_kind == "real_vae_rt"]
    assert rt and all(r.is_fake == 0 and "vae_rt:mock" in r.io_chain for r in rt)
    assert all(r.split == next(x.split for x in rows if x.image_path == r.real_image_path)
               for r in rt)                            # 与源同 split（同 origin-group 防泄漏）

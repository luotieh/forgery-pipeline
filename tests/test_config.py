from forgery_pipeline.config import load_config, PipelineConfig


def test_load_example_config():
    cfg = load_config("configs/pipeline.example.yaml")
    assert isinstance(cfg, PipelineConfig)
    assert cfg.seed == 1234
    assert cfg.backend == "mock"
    assert cfg.stages["d0"] is True
    assert cfg.scales.d1_per_generator >= 1
    assert len(cfg.generators) >= 3
    fams = {g.family for g in cfg.generators}
    assert "diffusion" in fams

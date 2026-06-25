import dataclasses
from pathlib import Path
from forgery_pipeline.config import load_config, StageScales
from forgery_pipeline.pipeline import run_pipeline
from forgery_pipeline import manifest
from forgery_pipeline.split.leakage import check_leakage


def test_run_pipeline_end_to_end(tmp_path):
    cfg = load_config("configs/pipeline.example.yaml")
    cfg = dataclasses.replace(
        cfg, out_dir=str(tmp_path / "run"),
        scales=StageScales(d0=16, d1_per_generator=1, d2=8, d3=4, d4=3))
    st = run_pipeline(cfg)
    assert st["total"] > 0
    mani = Path(cfg.out_dir) / "manifest.jsonl"
    assert mani.exists()
    samples = manifest.read_jsonl(mani)            # 全部行通过 schema 校验
    assert len(samples) == st["total"]
    assert check_leakage(samples) == []            # 无泄漏
    # 局部篡改样本必须有 mask
    assert all(s.mask_path for s in samples
               if s.manipulation_level1 == "partial_manipulated")
    assert st["by_split"].get("train", 0) > 0

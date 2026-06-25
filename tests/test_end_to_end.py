import dataclasses
from pathlib import Path
from forgery_pipeline.config import load_config, StageScales
from forgery_pipeline.pipeline import run_pipeline
from forgery_pipeline import manifest
from forgery_pipeline.split.leakage import check_leakage
from forgery_pipeline.split.splitter import SPLITS


def test_full_mock_pipeline(tmp_path):
    cfg = load_config("configs/pipeline.example.yaml")
    cfg = dataclasses.replace(
        cfg, out_dir=str(tmp_path / "run"),
        scales=StageScales(d0=40, d1_per_generator=3, d2=24, d3=12, d4=8))
    st = run_pipeline(cfg)

    samples = manifest.read_jsonl(Path(cfg.out_dir) / "manifest.jsonl")
    # 1) 全部样本 schema 合法（read_jsonl 已校验）
    assert len(samples) == st["total"] > 0
    # 2) 无数据泄漏
    assert check_leakage(samples) == []
    # 3) 局部篡改样本必有 mask 且文件存在
    for s in samples:
        if s.manipulation_level1 == "partial_manipulated":
            assert s.mask_path and (Path(cfg.out_dir) / s.mask_path).exists()
    # 4) 训练/验证非空，且划分覆盖足够多类别
    present = set(st["by_split"])
    assert present <= set(SPLITS)
    assert st["by_split"].get("train", 0) > 0
    assert st["by_split"].get("val", 0) > 0
    assert len(present) >= 5
    # 5) 五个子库各有产出（real + 四类 fake 任务）
    assert st["real"] > 0 and st["fake"] > 0
    assert st["by_task_type"].get("explainable", 0) > 0

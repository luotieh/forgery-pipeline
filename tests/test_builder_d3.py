from pathlib import Path
from forgery_pipeline.builders.d0_real import build_d0
from forgery_pipeline.builders.d3_web import build_d3


def test_build_d3_pseudo_mask_localization(tmp_path):
    base = build_d0(tmp_path, n=8, seed=0)
    samples = build_d3(tmp_path, base, n=4, seed=0)
    assert len(samples) >= 1  # 部分可能被 QES 过滤
    for s in samples:
        assert s.manipulation_level1 == "partial_manipulated"
        assert s.mask_source == "diff"
        assert s.mask_path and (Path(tmp_path) / s.mask_path).exists()
        assert s.quality_score is not None and s.quality_score >= 0.60
        assert s.task_type.value == "localization"

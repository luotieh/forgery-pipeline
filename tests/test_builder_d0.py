from pathlib import Path
from forgery_pipeline.builders.d0_real import build_d0


def test_build_d0_produces_real_samples(tmp_path):
    samples = build_d0(tmp_path, n=5, seed=0)
    assert len(samples) == 5
    for s in samples:
        assert s.is_fake == 0
        assert s.task_type.value == "real_pristine"
        assert (Path(tmp_path) / s.image_path).exists()
        assert s.manipulation_level1 is None

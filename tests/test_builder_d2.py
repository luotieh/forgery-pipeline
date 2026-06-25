from pathlib import Path
from forgery_pipeline.config import GeneratorSpec
from forgery_pipeline.builders.d0_real import build_d0
from forgery_pipeline.builders.d2_local import build_d2, MANIP_TYPES


def test_build_d2_localization(tmp_path):
    base = build_d0(tmp_path, n=6, seed=0)
    inps = [GeneratorSpec("stable-diffusion-inpaint", "diffusion", "inpaint")]
    samples = build_d2(tmp_path, base, n=4, inpainters=inps, seed=0)
    assert len(samples) == 4
    for s in samples:
        assert s.is_fake == 1
        assert s.task_type.value == "localization"
        assert s.manipulation_level1 == "partial_manipulated"
        assert s.mask_path and (Path(tmp_path) / s.mask_path).exists()
        assert 0.01 <= s.mask_area_ratio <= 0.50
        assert s.real_image_path is not None


def test_manip_types_cover_seven():
    assert len(MANIP_TYPES) == 7

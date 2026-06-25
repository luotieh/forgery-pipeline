import pytest
from pydantic import ValidationError
from forgery_pipeline.labels import validate_labels, LEVEL1, LEVEL2, LOSS_TERMS
from forgery_pipeline.schema import Sample, TaskType


def test_real_must_have_no_manip_labels():
    errs = validate_labels(0, "real_pristine", None, "whole_generated", None, None)
    assert errs  # 真实图不应带 level1


def test_partial_requires_mask():
    errs = validate_labels(1, "localization", None, "partial_manipulated", "AIGC-editing", None)
    assert any("mask" in e for e in errs)


def test_whole_generated_ok_without_mask():
    errs = validate_labels(1, "whole_image_detection", None, "whole_generated", "diffusion", None)
    assert errs == []


def test_level2_must_be_known():
    errs = validate_labels(1, "whole_image_detection", None, "whole_generated", "bogus", None)
    assert any("level2" in e for e in errs)


def test_constants_present():
    assert "whole_generated" in LEVEL1 and "partial_manipulated" in LEVEL1
    assert "diffusion" in LEVEL2
    assert "detection_loss" in LOSS_TERMS


def test_sample_model_rejects_inconsistent_labels():
    with pytest.raises(ValidationError):
        Sample(image_id="x", image_path="x.jpg", is_fake=1,
               task_type=TaskType.localization,
               manipulation_level1="partial_manipulated")  # 缺 mask_path

import pytest
from pydantic import ValidationError
from forgery_pipeline.schema import Sample, Postprocess, Explanation, TaskType


def test_minimal_real_sample_ok():
    s = Sample(image_id="real_0001", image_path="D0/real_0001.jpg",
              is_fake=0, task_type=TaskType.real_pristine)
    assert s.postprocess.jpeg_quality == "none"
    assert s.mask_path is None


def test_is_fake_must_be_binary():
    with pytest.raises(ValidationError):
        Sample(image_id="x", image_path="x.jpg", is_fake=2,
               task_type=TaskType.real_pristine)


def test_mask_area_ratio_range():
    with pytest.raises(ValidationError):
        Sample(image_id="x", image_path="x.jpg", is_fake=1,
               task_type=TaskType.localization, mask_area_ratio=1.5)


def test_roundtrip_json():
    s = Sample(image_id="g1", image_path="D1/g1.png", is_fake=1,
               task_type=TaskType.whole_image_detection,
               manipulation_level1="whole_generated",
               manipulation_level2="diffusion",
               postprocess=Postprocess(jpeg_quality=70))
    data = s.model_dump()
    s2 = Sample(**data)
    assert s2.postprocess.jpeg_quality == 70

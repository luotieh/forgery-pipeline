from forgery_pipeline.schema import Sample, TaskType, Postprocess
from forgery_pipeline.split.grouping import origin_key, is_degraded


def test_origin_links_base_and_edit():
    base = Sample(image_id="real_abc", image_path="D0_real_pristine/real_abc.jpg",
                  is_fake=0, task_type=TaskType.real_pristine)
    edit = Sample(image_id="local_x", image_path="D2_local_aigc_edit/local_x.jpg",
                  real_image_path="D0_real_pristine/real_abc.jpg",
                  mask_path="D2_local_aigc_edit/masks/local_x.png", is_fake=1,
                  task_type=TaskType.localization,
                  manipulation_level1="partial_manipulated",
                  manipulation_level2="AIGC-editing",
                  manipulation_level3="object_replacement")
    assert origin_key(base) == origin_key(edit) == "real_abc"


def test_is_degraded():
    assert is_degraded(Postprocess(jpeg_quality=70)) is True
    assert is_degraded(Postprocess(noise="sigma5")) is True
    assert is_degraded(Postprocess()) is False

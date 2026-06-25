import numpy as np
from forgery_pipeline.qc.gen_qc import check_generation


def test_normal_image_ok_with_bucket():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(128, 128, 3), dtype=np.uint8)
    ok, reasons, bucket = check_generation(img, "a dog")
    assert ok and reasons == []
    assert bucket in {"high", "mid", "low"}


def test_solid_image_flagged_failure():
    ok, reasons, bucket = check_generation(np.full((128, 128, 3), 10, np.uint8))
    assert not ok and bucket == "low"

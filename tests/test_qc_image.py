import numpy as np
from forgery_pipeline.qc.image_qc import check_image


def _good():
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(300, 300, 3), dtype=np.uint8)


def test_good_image_passes():
    ok, reasons = check_image(_good())
    assert ok and reasons == []


def test_short_side_rejected():
    ok, reasons = check_image(np.zeros((100, 400, 3), np.uint8))
    assert not ok and any("短边" in r for r in reasons)


def test_solid_image_rejected():
    ok, reasons = check_image(np.full((300, 300, 3), 128, np.uint8))
    assert not ok and any("纯色" in r for r in reasons)


def test_extreme_aspect_rejected():
    img = np.random.default_rng(1).integers(0, 256, (300, 2000, 3), dtype=np.uint8)
    ok, reasons = check_image(img)
    assert not ok and any("长宽比" in r for r in reasons)

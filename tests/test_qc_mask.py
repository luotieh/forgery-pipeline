import numpy as np
from forgery_pipeline.qc.mask_qc import check_mask


def test_valid_mask_passes():
    m = np.zeros((100, 100), np.uint8)
    m[30:60, 30:60] = 255  # 9% 面积，单连通
    ok, reasons = check_mask(m)
    assert ok and reasons == []


def test_too_small_rejected():
    m = np.zeros((100, 100), np.uint8)
    m[:2, :2] = 255  # 0.04%
    ok, reasons = check_mask(m)
    assert not ok and any("面积" in r for r in reasons)


def test_full_image_rejected():
    ok, reasons = check_mask(np.full((100, 100), 255, np.uint8))
    assert not ok


def test_fragmented_rejected():
    rng = np.random.default_rng(0)
    m = ((rng.random((100, 100)) < 0.06) * 255).astype(np.uint8)  # 大量散点
    ok, reasons = check_mask(m)
    assert not ok and any("碎片" in r for r in reasons)

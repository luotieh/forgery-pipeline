import numpy as np
from forgery_pipeline.masks import candidates as ca


def _mask_with_frac(frac, h=100, w=100):
    m = np.zeros((h, w), np.uint8)
    n = int(frac * h * w)
    m.flat[:n] = 255
    return m


def test_area_ratio():
    assert abs(ca.area_ratio(_mask_with_frac(0.1)) - 0.1) < 1e-6


def test_buckets():
    assert ca.bucket_for_ratio(0.02) == "small"
    assert ca.bucket_for_ratio(0.10) == "mid"
    assert ca.bucket_for_ratio(0.30) == "large"
    assert ca.bucket_for_ratio(0.005) is None  # 太小
    assert ca.bucket_for_ratio(0.7) is None     # 太大


def test_filter_and_sample_drops_invalid():
    masks = [_mask_with_frac(f) for f in (0.005, 0.03, 0.10, 0.30, 0.70)]
    kept = ca.filter_and_sample(masks)
    buckets = {b for _, _, b in kept}
    assert buckets == {"small", "mid", "large"}
    assert len(kept) == 3

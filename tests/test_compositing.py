import numpy as np, pytest
from forgery_pipeline.compositing import composite

def _pair(h=32, w=32):
    rng = np.random.default_rng(0)
    orig = rng.integers(0, 256, (h, w, 3), np.uint8)
    gen = rng.integers(0, 256, (h, w, 3), np.uint8)
    m = np.zeros((h, w), np.float32); m[8:20, 8:20] = 1.0
    return orig, gen, m

def test_none_returns_gen():
    o, g, m = _pair(); assert composite(o, g, m, "none") is g

def test_paste_exact_outside_inside():
    o, g, m = _pair(); out = composite(o, g, m, "paste")
    assert np.array_equal(out[m == 0], o[m == 0])     # 掩码外逐像素==orig
    assert np.array_equal(out[m == 1], g[m == 1])     # 掩码内==gen

def test_paste_feather_blends_band_exact_far_outside():
    o, g, m = _pair(); out = composite(o, g, m, "paste_feather", feather_px=2)
    assert np.array_equal(out[:2], o[:2])             # 远离羽化带 == orig
    band = out[7, 8:20]                               # 边界带为混合值
    assert not np.array_equal(band, o[7, 8:20]) and not np.array_equal(band, g[7, 8:20])

def test_shape_mismatch_raises():
    o, g, m = _pair()
    with pytest.raises(AssertionError):
        composite(o[:16], g, m, "paste")

import numpy as np
from forgery_pipeline.masks import pseudo_mask as pm
from forgery_pipeline.masks.candidates import area_ratio


def _scene(seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(128, 128, 3), dtype=np.uint8)


def test_diff_zero_when_identical():
    img = _scene()
    diff, score = pm.diff_map(img, img)
    assert diff.max() < 1e-3
    assert score > 0.99


def test_pseudo_mask_recovers_edited_region():
    real = _scene(1)
    fake = real.copy()
    fake[40:80, 50:90] = 0  # 已知篡改矩形
    mask, metrics = pm.pseudo_mask(real, fake, thresh=0.1)
    assert set(np.unique(mask)).issubset({0, 255})
    # 召回：篡改矩形内大部分被标记
    region = mask[40:80, 50:90]
    assert (region > 127).mean() > 0.7
    # 精确：整体面积不离谱
    assert area_ratio(mask) < 0.3
    assert metrics["confidence"] > 0

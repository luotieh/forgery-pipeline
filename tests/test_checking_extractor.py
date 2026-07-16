import numpy as np
from checking.extractor import (MultiSigmaResidual, DiffusersResidual, get_extractor,
                                _direction_descriptors)


def test_multisigma_profile_and_map_shapes():
    ext = MultiSigmaResidual(sigmas=(3, 5, 9))
    img = np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    assert ext.profile(img).shape == (13,)   # 2*3 + 3 分位 + 4 区域（幅值提取器不含方向）
    rm = ext.residual_map(img)
    assert rm.shape == (64, 64) and rm.dtype == np.float32
    assert ext.residual_stack(img).shape == (3, 64, 64)
    ds = ext.detection_score(img)
    assert isinstance(ds, float) and ds == ds  # 有限


def test_get_extractor_and_real_lazy():
    assert isinstance(get_extractor("multisigma"), MultiSigmaResidual)
    ext = get_extractor("real")               # 惰性：构造不加载模型
    assert isinstance(ext, DiffusersResidual)
    assert ext._unet is None
    assert ext.direction_features is True      # 默认开


def test_get_extractor_real_direction_toggle(monkeypatch):
    monkeypatch.setenv("CHECKING_DIRECTION_FEATURES", "0")
    assert get_extractor("real").direction_features is False
    monkeypatch.setenv("CHECKING_DIRECTION_FEATURES", "1")
    assert get_extractor("real").direction_features is True


def test_direction_descriptors_shape_and_finite():
    rng = np.random.default_rng(0)
    K, H, W = 5, 16, 16
    eps = rng.random((K, H, W), np.float32)
    x = rng.random((K, H, W), np.float32)
    d = _direction_descriptors(eps, x)
    assert d.shape == (4 * K - 1,)             # 2K 双通道均值 + (K-1) 方向余弦 + K 比值
    assert np.isfinite(d).all()
    assert d.dtype == np.float32


def test_direction_descriptors_deterministic_and_cosine_bounds():
    rng = np.random.default_rng(1)
    eps = rng.random((4, 8, 8), np.float32)
    x = rng.random((4, 8, 8), np.float32)
    d1 = _direction_descriptors(eps, x)
    d2 = _direction_descriptors(eps.copy(), x.copy())
    assert np.array_equal(d1, d2)              # 确定性
    cos = d1[8:11]                             # 2K=8 后的 K-1=3 个余弦
    assert (cos >= -1.0 - 1e-5).all() and (cos <= 1.0 + 1e-5).all()


def test_direction_descriptors_identical_maps_cosine_one():
    m = np.random.default_rng(2).random((3, 8, 8), np.float32)
    eps = np.stack([m[0], m[0], m[0]])         # 相邻 t 相同 → 余弦=1
    d = _direction_descriptors(eps, np.ones_like(eps))
    cos = d[6:8]                               # 2K=6 后的 K-1=2 个余弦
    assert np.allclose(cos, 1.0, atol=1e-5)


def test_direction_descriptors_zero_norm_safe():
    eps = np.zeros((3, 4, 4), np.float32)      # 全零 → 余弦定义为 0，不 NaN
    d = _direction_descriptors(eps, np.zeros_like(eps))
    assert np.isfinite(d).all()

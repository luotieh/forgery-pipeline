import numpy as np
from checking.extractor import MultiSigmaResidual, DiffusersSD2Residual, get_extractor


def test_multisigma_profile_and_map_shapes():
    ext = MultiSigmaResidual(sigmas=(3, 5, 9))
    img = np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    assert ext.profile(img).shape == (13,)   # 2*3 + 3 分位 + 4 区域
    rm = ext.residual_map(img)
    assert rm.shape == (64, 64) and rm.dtype == np.float32
    assert ext.residual_stack(img).shape == (3, 64, 64)
    ds = ext.detection_score(img)
    assert isinstance(ds, float) and ds == ds  # 有限


def test_get_extractor_and_real_lazy():
    assert isinstance(get_extractor("multisigma"), MultiSigmaResidual)
    ext = get_extractor("real")               # 惰性：构造不加载模型
    assert isinstance(ext, DiffusersSD2Residual)
    assert ext._unet is None

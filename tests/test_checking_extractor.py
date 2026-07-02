import numpy as np
import pytest
from checking.extractor import MultiSigmaResidual, get_extractor


def test_multisigma_profile_and_map_shapes():
    ext = MultiSigmaResidual(sigmas=(3, 5, 9))
    img = np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    assert ext.profile(img).shape == (6,)          # 3 尺度 × [mean,std]
    rm = ext.residual_map(img)
    assert rm.shape == (64, 64) and rm.dtype == np.float32
    assert ext.residual_stack(img).shape == (3, 64, 64)


def test_get_extractor_and_real_stub():
    assert isinstance(get_extractor("multisigma"), MultiSigmaResidual)
    with pytest.raises(NotImplementedError):
        get_extractor("real")

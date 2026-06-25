import numpy as np
from forgery_pipeline.postprocess import degradations as dg
from forgery_pipeline.schema import Postprocess


def _img():
    return np.random.default_rng(0).integers(0, 256, (128, 128, 3), dtype=np.uint8)


def test_jpeg_changes_pixels_keeps_shape():
    img = _img()
    out = dg.apply_jpeg(img, 50)
    assert out.shape == img.shape and out.dtype == np.uint8
    assert not np.array_equal(out, img)


def test_resize_preserves_shape():
    img = _img()
    assert dg.apply_resize(img, 0.5).shape == img.shape


def test_noise_deterministic():
    img = _img()
    a = dg.apply_noise(img, 10, seed=3)
    b = dg.apply_noise(img, 10, seed=3)
    assert np.array_equal(a, b)


def test_sample_and_apply_records_one_param():
    img = _img()
    out, pp = dg.sample_and_apply(img, np.random.default_rng(1))
    assert out.shape == img.shape
    assert isinstance(pp, Postprocess)
    changed = [pp.jpeg_quality != "none", pp.resize != "none",
               pp.blur != "none", pp.noise != "none"]
    assert sum(changed) >= 1

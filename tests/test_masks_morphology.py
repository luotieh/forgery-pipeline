import numpy as np
from forgery_pipeline.masks import morphology as mo


def _square():
    m = np.zeros((100, 100), np.uint8)
    m[30:70, 30:70] = 255
    return m


def test_dilate_grows_erode_shrinks():
    m = _square()
    base = int((m > 127).sum())
    assert int((mo.dilate(m, 5) > 127).sum()) > base
    assert int((mo.erode(m, 5) > 127).sum()) < base


def test_outputs_binary():
    for fn in (mo.dilate, mo.erode, mo.boundary_blur):
        out = fn(_square(), 5)
        assert set(np.unique(out)).issubset({0, 255})


def test_make_irregular_deterministic_and_binary():
    a = mo.make_irregular(_square(), seed=1)
    b = mo.make_irregular(_square(), seed=1)
    assert np.array_equal(a, b)
    assert set(np.unique(a)).issubset({0, 255})

import numpy as np
from forgery_pipeline import image_io


def test_image_roundtrip(tmp_path):
    img = np.random.default_rng(0).integers(0, 256, (32, 48, 3), dtype=np.uint8)
    p = tmp_path / "a.png"
    image_io.save_image(img, p)
    got = image_io.load_image(p)
    assert got.shape == img.shape and np.array_equal(got, img)


def test_mask_roundtrip(tmp_path):
    m = np.zeros((20, 20), np.uint8)
    m[5:15, 5:15] = 255
    p = tmp_path / "m.png"
    image_io.save_mask(m, p)
    got = image_io.load_mask(p)
    assert got.shape == (20, 20) and set(np.unique(got)).issubset({0, 255})

import numpy as np
from forgery_pipeline.ids import make_image_id, content_hash


def test_make_image_id_deterministic():
    a = make_image_id("real", "hello")
    b = make_image_id("real", "hello")
    assert a == b
    assert a.startswith("real_") and len(a) == len("real_") + 12


def test_make_image_id_varies_with_payload():
    assert make_image_id("real", "a") != make_image_id("real", "b")


def test_content_hash_stable():
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    assert content_hash(img) == content_hash(img.copy())

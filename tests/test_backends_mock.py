import numpy as np
from forgery_pipeline.backends import mock
from forgery_pipeline.schema import Explanation


def test_image_source_deterministic_and_shape():
    a = list(mock.MockImageSource(seed=7).iter_images(3))
    b = list(mock.MockImageSource(seed=7).iter_images(3))
    assert len(a) == 3
    img, meta = a[0]
    assert img.shape == (256, 256, 3) and img.dtype == np.uint8
    assert np.array_equal(a[0][0], b[0][0])  # 确定性
    assert meta["source_dataset"]


def test_whole_generator_deterministic():
    g = mock.MockWholeImageGenerator("stable-diffusion-xl", "diffusion")
    i1, m1 = g.generate("a dog", {"seed": 5})
    i2, _ = g.generate("a dog", {"seed": 5})
    i3, _ = g.generate("a cat", {"seed": 5})
    assert np.array_equal(i1, i2)
    assert not np.array_equal(i1, i3)  # prompt 改变结果
    assert m1["generator_name"] == "stable-diffusion-xl"


def test_inpainter_changes_only_masked_region():
    img = np.full((64, 64, 3), 100, np.uint8)
    mask = np.zeros((64, 64), np.uint8)
    mask[10:30, 10:30] = 255
    out, meta = mock.MockInpainter().inpaint(img, mask, "replace", {"seed": 1})
    assert out.shape == img.shape
    assert not np.array_equal(out[10:30, 10:30], img[10:30, 10:30])  # 区域被改
    assert np.array_equal(out[40:60, 40:60], img[40:60, 40:60])      # 区域外不变


def test_segmenter_masks_binary_and_count():
    img = np.zeros((128, 128, 3), np.uint8)
    masks = mock.MockSegmenter(seed=3).propose_masks(img, 5)
    assert len(masks) == 5
    for m in masks:
        assert m.shape == (128, 128) and m.dtype == np.uint8
        assert set(np.unique(m)).issubset({0, 255})


def test_explainer_returns_explanation():
    e = mock.MockExplainer().explain(np.zeros((8, 8, 3), np.uint8), None,
                                     {"manipulation_level3": "object_replacement"})
    assert isinstance(e, Explanation)
    assert "object_replacement" in e.forensic_conclusion

import numpy as np
from forgery_pipeline.dedup import PHashDeduper


def test_dedup_detects_repeat():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    d = PHashDeduper()
    assert d.add(img) is True          # 首次为新
    assert d.add(img.copy()) is False  # 重复
    assert d.is_duplicate(img) is True


def test_dedup_accepts_distinct():
    # 用两张结构不同的随机图（纯色图的 pHash 会塌缩为同一值，不适合做该断言）
    d = PHashDeduper()
    a = np.random.default_rng(1).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    b = np.random.default_rng(2).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    assert d.add(a) is True
    assert d.add(b) is True

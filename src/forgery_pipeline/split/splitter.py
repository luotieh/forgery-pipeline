"""8-way 数据划分（报告 §12）。确定性、按原图分组、防泄漏。"""
from __future__ import annotations
import hashlib
from collections import defaultdict
from forgery_pipeline.schema import Sample
from forgery_pipeline.split.grouping import origin_key, is_degraded

SPLITS = ["train", "val", "test_a", "test_b", "test_c", "test_d", "test_e", "test_f"]


def _hash01(key: str, salt: str) -> float:
    digest = hashlib.sha1(f"{salt}|{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") / 2 ** 32


def assign_splits(samples: list[Sample], holdout_generators, holdout_manipulation,
                  holdout_domains=("Places",), seed: int = 0) -> list[Sample]:
    hg, hm, hd = set(holdout_generators), set(holdout_manipulation), set(holdout_domains)
    groups: dict[str, list[Sample]] = defaultdict(list)
    for s in samples:
        groups[origin_key(s)].append(s)

    for okey, members in groups.items():
        gens = {m.generator_name for m in members if m.generator_name}
        manips = {m.manipulation_level3 for m in members if m.manipulation_level3}
        domains = {m.source_dataset for m in members if m.source_dataset}
        real_only = all(m.is_fake == 0 for m in members)
        if gens & hg:
            split = "test_b"
        elif manips & hm:
            split = "test_c"
        elif domains & hd:
            split = "test_d"
        elif real_only:
            r = _hash01(okey, f"real-{seed}")
            split = "train" if r < 0.60 else "val" if r < 0.75 else "test_f"
        else:
            r = _hash01(okey, f"fake-{seed}")
            split = "train" if r < 0.70 else "val" if r < 0.80 else "test_a"
        for m in members:
            m.split = split

    # 从 test_a 的退化 fake 中切出 degradation 测试集 test_e（仍属非 train，不破坏 train 隔离）
    for s in samples:
        if s.split == "test_a" and s.is_fake == 1 and is_degraded(s.postprocess):
            s.split = "test_e"
    return samples

"""数据泄漏检查（报告 §11.4）。"""
from __future__ import annotations
from collections import defaultdict
from forgery_pipeline.schema import Sample
from forgery_pipeline.split.grouping import origin_key

# 公开取证 benchmark，不得混入训练集
BENCHMARK_SOURCES = {"CASIA", "Columbia", "Coverage", "NIST16", "DSO-1", "IMD2020"}


def check_leakage(samples: list[Sample]) -> list[str]:
    errs: list[str] = []

    # 规则 1&2：train 的原图不得出现在任何非 train split（同源/压缩版本泄漏）
    train_o = {origin_key(s) for s in samples if s.split == "train"}
    other_o = {origin_key(s) for s in samples if s.split and s.split != "train"}
    shared = train_o & other_o
    if shared:
        errs.append(f"原图跨越 train 与非 train: {sorted(shared)[:3]}")

    # 规则 3：同一 prompt+seed 不得跨 train/非 train
    ps = defaultdict(set)
    for s in samples:
        if s.split and s.prompt is not None and s.seed is not None:
            ps[(s.prompt, s.seed)].add("train" if s.split == "train" else "other")
    if any("train" in v and "other" in v for v in ps.values()):
        errs.append("存在 prompt+seed 跨 train/非 train")

    # 规则 4：cross-generator 测试集（test_b）的生成器不得出现在 train
    train_gen = {s.generator_name for s in samples
                 if s.split == "train" and s.generator_name}
    tb_gen = {s.generator_name for s in samples
              if s.split == "test_b" and s.generator_name}
    if train_gen & tb_gen:
        errs.append(f"cross-generator 生成器出现在 train: {sorted(train_gen & tb_gen)}")

    # 规则 5：公开 benchmark 不得混入 train
    bench = {s.source_dataset for s in samples
             if s.split == "train" and s.source_dataset in BENCHMARK_SOURCES}
    if bench:
        errs.append(f"benchmark 数据混入 train: {sorted(bench)}")

    return errs

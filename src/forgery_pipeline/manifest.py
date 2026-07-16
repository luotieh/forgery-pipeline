"""统一 manifest 的 JSONL 读写、合并与统计。"""
from __future__ import annotations
from collections import Counter
from pathlib import Path
from typing import Iterable
from forgery_pipeline.schema import Sample


def write_jsonl(path, samples: Iterable[Sample], mode: str = "w") -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, mode, encoding="utf-8") as f:
        for s in samples:
            f.write(s.model_dump_json() + "\n")
            n += 1
    return n


def append_jsonl(path, samples: Iterable[Sample]) -> int:
    return write_jsonl(path, samples, mode="a")


def read_jsonl(path) -> list[Sample]:
    out: list[Sample] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(Sample.model_validate_json(line))
    return out


def merge(paths, out_path) -> int:
    all_samples: list[Sample] = []
    for p in paths:
        if Path(p).exists():
            all_samples.extend(read_jsonl(p))
    return write_jsonl(out_path, all_samples)


def stats(samples: list[Sample]) -> dict:
    # 延迟到函数内 import：manifest 依赖 validate（用其 nongen_chain），但 validate 不得
    # 反向 import manifest，否则成环（见 validate.py 模块 docstring）。
    from forgery_pipeline.validate import nongen_chain

    io_chain_by_fake_split: dict = {}
    for s in samples:
        if not s.split:
            continue
        bucket = (io_chain_by_fake_split.setdefault(s.split, {})
                 .setdefault(nongen_chain(s.io_chain), {"real": 0, "fake": 0}))
        bucket["fake" if s.is_fake else "real"] += 1

    return {
        "total": len(samples),
        "real": sum(1 for s in samples if s.is_fake == 0),
        "fake": sum(1 for s in samples if s.is_fake == 1),
        "with_mask": sum(1 for s in samples if s.mask_path),
        "by_task_type": dict(Counter(s.task_type.value for s in samples)),
        "by_generator_family": dict(
            Counter(s.generator_family for s in samples if s.generator_family)),
        "by_generator_name": dict(
            Counter(s.generator_name for s in samples if s.generator_name)),
        "by_operator": dict(Counter(s.operator for s in samples if s.operator)),
        "by_split": dict(Counter(s.split for s in samples if s.split)),
        "by_sample_kind": dict(Counter(s.sample_kind for s in samples if s.sample_kind)),
        "by_compositing": dict(Counter(s.compositing for s in samples if s.compositing)),
        "io_chain_by_fake_split": io_chain_by_fake_split,
    }

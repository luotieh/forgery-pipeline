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
    return {
        "total": len(samples),
        "real": sum(1 for s in samples if s.is_fake == 0),
        "fake": sum(1 for s in samples if s.is_fake == 1),
        "with_mask": sum(1 for s in samples if s.mask_path),
        "by_task_type": dict(Counter(s.task_type.value for s in samples)),
        "by_generator_family": dict(
            Counter(s.generator_family for s in samples if s.generator_family)),
        "by_split": dict(Counter(s.split for s in samples if s.split)),
    }

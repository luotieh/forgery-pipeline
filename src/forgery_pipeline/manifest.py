"""统一 manifest 的 JSONL 读写、合并与统计。"""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path
from typing import Iterable
import numpy as np
from forgery_pipeline.schema import Sample

# stats() 无 config/policies 上下文可读（不像 validate.check_v12 能接收 area_buckets
# 参数），面积分桶固定用 PATCH 9 Wave2 的默认桶界（与 config.PipelineConfig.area_buckets
# 默认值、validate.check_v12 的默认参数同源同值；三处如需联动调整需一并修改）。
_STATS_AREA_BUCKETS = (0.05, 0.15, 0.35, 0.7)


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

    # by_nuisance_cell（PATCH 9 Wave2 Task5）：op_params 可解析为 JSON object 且同时含
    # cfg_scale/steps 键的行才计数（同 validate.check_v11 的合规判据，但 stats 不区分
    # is_fake/io_chain 判定域——纯描述性计数，不是校验，行不需要"在 V11 判定域内"才被计入）。
    # 值非数值（如手工回填误存成字符串）时 :g 格式化会抛 TypeError/ValueError——stats()
    # 是尽力而为的描述性统计，静默跳过该行而不是让 pipeline.run_pipeline() 末尾崩溃
    # （已实测复现，见 report；与上面 json.loads 的异常吞咽同一处理哲学）。cfg_scale 与
    # steps 两键同约 :g（审查修复对称化：steps 裸 {} 会让字符串 "30" 与合法 st30 单元格
    # 文本合并、[1,2] 拼出垃圾单元格——与 validate.check_v11 同一守卫口径）。
    by_nuisance_cell: Counter = Counter()
    for s in samples:
        if not s.op_params:
            continue
        try:
            params = json.loads(s.op_params)
        except (TypeError, ValueError):
            continue
        if isinstance(params, dict) and "cfg_scale" in params and "steps" in params:
            try:
                by_nuisance_cell[f"cfg{params['cfg_scale']:g}/st{params['steps']:g}"] += 1
            except (TypeError, ValueError):
                continue

    # by_area_bucket（PATCH 9 Wave2 Task5）：全部 mask_area_ratio 非 None 的行按默认桶界
    # 归桶计数（不像 validate.check_v12 那样限定 operator ∈ masked 集合——纯描述性计数）；
    # 溢出桶（np.digitize 返回 len(_STATS_AREA_BUCKETS)）保留展示，不像 V12 的下限检查那样丢弃。
    by_area_bucket: Counter = Counter()
    for s in samples:
        if s.mask_area_ratio is None:
            continue
        bi = int(np.digitize(s.mask_area_ratio, _STATS_AREA_BUCKETS))
        by_area_bucket[f"b{bi}"] += 1

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
        "by_nuisance_cell": dict(by_nuisance_cell),
        "by_area_bucket": dict(by_area_bucket),
    }

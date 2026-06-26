"""划分分组键（报告 §12.1：按原图 ID 分组，避免同源泄漏）。"""
from __future__ import annotations
from pathlib import Path
from forgery_pipeline.schema import Sample, Postprocess


def origin_key(s: Sample) -> str:
    ref = s.real_image_path or s.image_path
    stem = Path(ref).stem
    # 退化样本（image_path 带 __deg 后缀且无 real_image_path，如 whole_generated）
    # 归并到原图同组，避免「原图/退化版」跨 split 的隐性泄漏（补全 PATCH 5 的无泄漏保证）
    if not s.real_image_path and stem.endswith("__deg"):
        stem = stem[:-len("__deg")]
    return stem


def is_degraded(pp: Postprocess) -> bool:
    return (pp.jpeg_quality != "none" or pp.resize != "none"
            or pp.blur != "none" or pp.noise != "none")

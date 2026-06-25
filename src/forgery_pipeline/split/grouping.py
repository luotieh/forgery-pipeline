"""划分分组键（报告 §12.1：按原图 ID 分组，避免同源泄漏）。"""
from __future__ import annotations
from pathlib import Path
from forgery_pipeline.schema import Sample, Postprocess


def origin_key(s: Sample) -> str:
    ref = s.real_image_path or s.image_path
    return Path(ref).stem


def is_degraded(pp: Postprocess) -> bool:
    return (pp.jpeg_quality != "none" or pp.resize != "none"
            or pp.blur != "none" or pp.noise != "none")

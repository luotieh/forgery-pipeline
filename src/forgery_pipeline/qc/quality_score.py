"""QES-like 质量评分（报告 §7.5）。"""
from __future__ import annotations

WEIGHTS = {
    "confidence": 0.3,
    "boundary_sharpness": 0.2,
    "mask_consistency": 0.2,
    "semantic_consistency": 0.2,
    "area_validity": 0.1,
}


def area_validity(area_ratio: float) -> float:
    return 1.0 if 0.01 <= area_ratio <= 0.50 else 0.0


def qes_score(confidence: float, boundary_sharpness: float,
              mask_consistency: float, semantic_consistency: float,
              area_validity: float) -> float:
    return float(
        WEIGHTS["confidence"] * confidence
        + WEIGHTS["boundary_sharpness"] * boundary_sharpness
        + WEIGHTS["mask_consistency"] * mask_consistency
        + WEIGHTS["semantic_consistency"] * semantic_consistency
        + WEIGHTS["area_validity"] * area_validity
    )


def route_from_score(score: float) -> str:
    if score >= 0.75:
        return "accept"
    if score >= 0.60:
        return "review"
    return "reject"


def bucket_from_score(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.60:
        return "mid"
    return "low"

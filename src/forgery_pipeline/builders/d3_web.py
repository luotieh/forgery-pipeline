"""D3 网页人工篡改：real-fake pair → 差分伪 mask → QES（报告 §7，借鉴 MIML）。"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from forgery_pipeline import image_io, ids
from forgery_pipeline.masks.pseudo_mask import pseudo_mask
from forgery_pipeline.qc.mask_qc import check_mask
from forgery_pipeline.qc.quality_score import (
    qes_score, route_from_score, bucket_from_score, area_validity)
from forgery_pipeline.schema import Sample, TaskType

_L2 = ["splicing", "copy-move", "removal"]
_L3 = {"splicing": "image_guided_editing", "copy-move": "image_guided_editing",
       "removal": "object_removal"}
# 模拟“人工拼贴”用的高对比色块（占位真实网页编辑，保证差分可检出）
_SPLICE_COLORS = [(220, 30, 30), (30, 200, 30), (30, 30, 220),
                  (230, 200, 20), (200, 30, 200)]


def _synthesize_web_fake(real: np.ndarray, seed: int) -> np.ndarray:
    """在真实图上贴入一块高对比区域，模拟明显的人工拼贴篡改。"""
    rng = np.random.default_rng(seed)
    h, w = real.shape[:2]
    bh, bw = max(8, h // 4), max(8, w // 4)
    ty = int(rng.integers(0, max(1, h - bh)))
    tx = int(rng.integers(0, max(1, w - bw)))
    fake = real.copy()
    color = np.array(_SPLICE_COLORS[int(seed) % len(_SPLICE_COLORS)], np.uint8)
    fake[ty:ty + bh, tx:tx + bw] = color
    return fake


def build_d3(out_dir, base_samples: list[Sample], n: int,
             backend: str = "mock", seed: int = 0) -> list[Sample]:
    out_dir = Path(out_dir)
    samples: list[Sample] = []
    attempts = 0
    max_attempts = max(n * 8, 8)
    while len(samples) < n and base_samples and attempts < max_attempts:
        base = base_samples[attempts % len(base_samples)]
        attempts += 1
        real = image_io.load_image(out_dir / base.image_path)
        fake = _synthesize_web_fake(real, seed + attempts)
        mask, metrics = pseudo_mask(real, fake, thresh=0.1)
        ok, _ = check_mask(mask)
        if not ok:
            continue
        ratio = metrics["area_ratio"]
        score = qes_score(
            confidence=min(metrics["confidence"], 1.0),
            boundary_sharpness=metrics["boundary_sharpness"],
            mask_consistency=1.0 if 0.01 <= ratio <= 0.50 else 0.5,
            semantic_consistency=0.8,
            area_validity=area_validity(ratio),
        )
        if route_from_score(score) == "reject":
            continue
        l2 = _L2[len(samples) % len(_L2)]
        iid = ids.make_image_id("web_forgery", f"{base.image_id}-{attempts}")
        img_rel = f"D3_web_human_forgery/{iid}.jpg"
        mask_rel = f"D3_web_human_forgery/masks/{iid}.png"
        image_io.save_image(fake, out_dir / img_rel)
        image_io.save_mask(mask, out_dir / mask_rel)
        samples.append(Sample(
            image_id=iid, image_path=img_rel,
            real_image_path=base.image_path, mask_path=mask_rel, is_fake=1,
            task_type=TaskType.localization,
            manipulation_level1="partial_manipulated",
            manipulation_level2=l2, manipulation_level3=_L3[l2],
            generator_name="manual-web-edit", generator_family="editing",
            mask_source="diff", mask_area_ratio=ratio,
            quality_score=round(score, 4), quality_bucket=bucket_from_score(score),
            source_dataset=base.source_dataset,
        ))
    return samples

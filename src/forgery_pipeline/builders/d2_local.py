"""D2 局部 AIGC 篡改：mask → prompt → inpaint（报告 §6，借鉴 GIM）。"""
from __future__ import annotations
from pathlib import Path
from forgery_pipeline import image_io, ids
from forgery_pipeline.backends import registry
from forgery_pipeline.backends.mock import stable_hash
from forgery_pipeline.config import GeneratorSpec
from forgery_pipeline.masks.candidates import filter_and_sample, area_ratio
from forgery_pipeline.masks import morphology
from forgery_pipeline.qc.mask_qc import check_mask
from forgery_pipeline.qc.quality_score import (
    qes_score, route_from_score, bucket_from_score, area_validity)
from forgery_pipeline.schema import Sample, TaskType

# (篡改类型, level3, 编辑 prompt 模板)；level3 取自 LEVEL3 合法值
MANIP_TYPES = [
    ("object_insertion", "mask_guided_inpainting",
     "Insert a new realistic object into the masked region."),
    ("object_replacement", "object_replacement",
     "Replace the object in the masked region with a different realistic object."),
    ("object_removal", "object_removal",
     "Remove the object in the masked region and fill the background naturally."),
    ("attribute_editing", "text_guided_editing",
     "Change the color or attribute of the object in the masked region."),
    ("background_editing", "image_guided_editing",
     "Repaint the background within the masked region."),
    ("text_editing", "text_editing",
     "Modify the text content within the masked region."),
    ("face_editing", "face_swap",
     "Edit the face in the masked region (expression/glasses/hair)."),
]


def build_d2(out_dir, base_samples: list[Sample], n: int,
             inpainters: list[GeneratorSpec], backend: str = "mock",
             seed: int = 0, holdout_inpainters=()) -> list[Sample]:
    out_dir = Path(out_dir)
    seg = registry.get_segmenter(backend, seed=seed)
    # 按底图把 inpainter 划入互斥池：保证每个 origin-group 只用一类生成器，
    # 避免 splitter 整组判定 test_b 时把非 holdout 生成器混入 test_b（PATCH 6）。
    hold = {i.name for i in inpainters if i.name in set(holdout_inpainters)}
    pool_hold = [i for i in inpainters if i.name in hold]
    pool_train = [i for i in inpainters if i.name not in hold] or inpainters
    samples: list[Sample] = []
    attempts = 0
    max_attempts = max(n * 8, 8)
    while len(samples) < n and base_samples and attempts < max_attempts:
        base = base_samples[attempts % len(base_samples)]
        attempts += 1
        img = image_io.load_image(out_dir / base.image_path)
        valid = filter_and_sample(seg.propose_masks(img, 6))
        if not valid:
            continue
        mask = morphology.make_irregular(valid[len(samples) % len(valid)][0],
                                         seed=seed + attempts)
        ok, _ = check_mask(mask)
        if not ok:
            continue
        ratio = area_ratio(mask)
        # QES 质量评分（PATCH 7）；mock 用占位置信度，真实后端换实测信号
        score = qes_score(
            confidence=0.9, boundary_sharpness=0.8,
            mask_consistency=1.0 if 0.01 <= ratio <= 0.50 else 0.5,
            semantic_consistency=0.8, area_validity=area_validity(ratio),
        )
        if route_from_score(score) == "reject":
            continue
        mtype, level3, tmpl = MANIP_TYPES[len(samples) % len(MANIP_TYPES)]
        # 按底图选池（~20% 底图走 holdout 池），同一 origin-group 生成器同池
        okey = base.real_image_path or base.image_path
        use_hold = bool(pool_hold) and (stable_hash(okey) % 5 == 0)
        pool = pool_hold if use_hold else pool_train
        inp = pool[len(samples) % len(pool)]
        painter = registry.get_inpainter(backend, inp.name, inp.family)
        s = seed + attempts
        fake, _ = painter.inpaint(img, mask, tmpl, {"seed": s})
        iid = ids.make_image_id("local_edit", f"{base.image_id}-{mtype}-{s}")
        img_rel = f"D2_local_aigc_edit/{iid}.jpg"
        mask_rel = f"D2_local_aigc_edit/masks/{iid}.png"
        image_io.save_image(fake, out_dir / img_rel)
        image_io.save_mask(mask, out_dir / mask_rel)
        samples.append(Sample(
            image_id=iid, image_path=img_rel,
            real_image_path=base.image_path, mask_path=mask_rel, is_fake=1,
            task_type=TaskType.localization,
            manipulation_level1="partial_manipulated",
            manipulation_level2="AIGC-editing",
            manipulation_level3=level3, manipulation_level4=inp.name,
            generator_name=inp.name, generator_family=inp.family,
            mask_source="SAM", mask_area_ratio=ratio, prompt=tmpl, seed=s,
            quality_score=round(score, 4), quality_bucket=bucket_from_score(score),
            source_dataset=base.source_dataset,
        ))
    return samples

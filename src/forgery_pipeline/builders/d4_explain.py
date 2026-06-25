"""D4 可解释取证子集：image+mask → MLLM 文本解释（报告 §8，借鉴 FakeShield）。"""
from __future__ import annotations
from pathlib import Path
from forgery_pipeline import image_io, ids
from forgery_pipeline.backends import registry
from forgery_pipeline.schema import Sample, TaskType


def build_d4(out_dir, source_samples: list[Sample], n: int,
             backend: str = "mock") -> list[Sample]:
    out_dir = Path(out_dir)
    explainer = registry.get_explainer(backend)
    cands = [s for s in source_samples if s.mask_path][:n]
    samples: list[Sample] = []
    for s in cands:
        img = image_io.load_image(out_dir / s.image_path)
        mask = image_io.load_mask(out_dir / s.mask_path)
        expl = explainer.explain(
            img, mask,
            {"manipulation_level3": s.manipulation_level3 or "local AIGC inpainting"})
        iid = ids.make_image_id("explain", s.image_id)
        samples.append(Sample(
            image_id=iid, image_path=s.image_path,
            real_image_path=s.real_image_path, mask_path=s.mask_path, is_fake=1,
            task_type=TaskType.explainable,
            manipulation_level1=s.manipulation_level1,
            manipulation_level2=s.manipulation_level2,
            manipulation_level3=s.manipulation_level3,
            manipulation_level4=s.manipulation_level4,
            generator_name=s.generator_name, generator_family=s.generator_family,
            mask_source=s.mask_source, mask_area_ratio=s.mask_area_ratio,
            explanation=expl,
        ))
    return samples

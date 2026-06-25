"""D2 局部 AIGC 篡改：mask → prompt → inpaint（报告 §6，借鉴 GIM）。"""
from __future__ import annotations
from pathlib import Path
from forgery_pipeline import image_io, ids
from forgery_pipeline.backends import registry
from forgery_pipeline.config import GeneratorSpec
from forgery_pipeline.masks.candidates import filter_and_sample, area_ratio
from forgery_pipeline.masks import morphology
from forgery_pipeline.qc.mask_qc import check_mask
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
             seed: int = 0) -> list[Sample]:
    out_dir = Path(out_dir)
    seg = registry.get_segmenter(backend, seed=seed)
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
        mtype, level3, tmpl = MANIP_TYPES[len(samples) % len(MANIP_TYPES)]
        inp = inpainters[len(samples) % len(inpainters)]
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
            source_dataset=base.source_dataset,
        ))
    return samples

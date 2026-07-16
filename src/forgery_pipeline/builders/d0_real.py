"""D0 真实图像池：摄取 → 清洗 → 去重（报告 §4）。"""
from __future__ import annotations
from pathlib import Path
from forgery_pipeline import image_io, ids
from forgery_pipeline.backends import registry
from forgery_pipeline.dedup import PHashDeduper
from forgery_pipeline.qc.image_qc import check_image
from forgery_pipeline.schema import Sample, TaskType


def build_d0(out_dir, n: int, backend: str = "mock", seed: int = 0) -> list[Sample]:
    out_dir = Path(out_dir)
    src = registry.get_image_source(backend, seed=seed)
    dedup = PHashDeduper()
    samples: list[Sample] = []
    for img, meta in src.iter_images(n * 3):  # 过采样以容忍 QC 丢弃
        if len(samples) >= n:
            break
        ok, _ = check_image(img)
        if not ok or not dedup.add(img):
            continue
        iid = ids.make_image_id("real", ids.content_hash(img))
        rel = f"D0_real_pristine/{iid}.png"
        image_io.save_canonical(img, out_dir / rel)
        samples.append(Sample(
            image_id=iid, image_path=rel, is_fake=0,
            task_type=TaskType.real_pristine,
            source_dataset=meta.get("source_dataset"),
            license=meta.get("license"),
            sample_kind="real",
            io_chain=image_io.chain("decode", f"rs{img.shape[0]}", "png"),
        ))
    return samples

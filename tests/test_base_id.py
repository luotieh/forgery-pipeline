# tests/test_base_id.py
from forgery_pipeline.config import GeneratorSpec, PipelineConfig, StageScales
from forgery_pipeline.pipeline import run_pipeline
from forgery_pipeline import manifest
from scripts.backfill_manifest_v7 import backfill

def test_pipeline_rows_all_carry_base_id_and_groups_are_consistent(tmp_path):
    cfg = PipelineConfig(out_dir=str(tmp_path / "run"), seed=0, backend="mock",
                         stages={"d0": True, "d1": True, "d2": True, "d3": True,
                                 "d4": True, "postprocess": True, "split": True},
                         scales=StageScales(d0=12, d1_per_generator=1, d2=6, d3=4, d4=2),
                         generators=[GeneratorSpec("g1", "gan", "whole")],
                         inpainters=[GeneratorSpec("i1", "diffusion", "inpaint")],
                         vae_rt_frac=0.3)
    run_pipeline(cfg)
    rows = manifest.read_jsonl(tmp_path / "run" / "manifest.jsonl")
    assert all(r.base_id for r in rows)                      # 全行必填
    real_ids = {r.image_id for r in rows if r.sample_kind == "real"}
    for r in rows:
        if r.sample_kind == "real":
            assert r.base_id == r.image_id                   # D0 自指
        if r.sample_kind == "real_vae_rt":
            assert r.base_id in real_ids                     # 继承源
    # 同 base_id 组内 split 一致（V8 的数据前提）
    by_base = {}
    for r in rows:
        by_base.setdefault(r.base_id, set()).add(r.split)
    assert all(len(s) == 1 for s in by_base.values())

def test_backfill_fills_base_id(tmp_path):
    from forgery_pipeline.schema import Sample, TaskType
    rows = [Sample(image_id="r0", image_path="D0/r0.png", is_fake=0,
                   task_type=TaskType.real_pristine),
            Sample(image_id="f0", image_path="a.png", real_image_path="D0/r0.png",
                   is_fake=1, mask_path="m.png", manipulation_level1="partial_manipulated",
                   task_type=TaskType.localization)]
    p = tmp_path / "old.jsonl"; manifest.write_jsonl(p, rows)
    out = tmp_path / "new.jsonl"; backfill(p, out)
    r = manifest.read_jsonl(out)
    assert r[0].base_id == "r0" and r[1].base_id == "r0"     # 衍生行映射到 real 行 image_id

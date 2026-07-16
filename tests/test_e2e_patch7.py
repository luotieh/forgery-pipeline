"""PATCH 7 冒烟验收 e2e（任务 7.5）：mock 全链 → V1–V7 全过 + 成对 compositing 断言过。

controller 修正 B（覆盖 task-7-brief 的 cfg 取值）：
- scales d0=40（d1_per_generator=2、d2=10、d3=6 不变，d4 关闭）、vae_rt_frac=0.3——
  经验证 train split 稳定得到 real≥10（本次运行 real=14，不被 V4 的 min_real 守卫豁免，
  配比断言真正生效）；test_a/test_f 的 real 行数天然 <10，由 min_real 守卫豁免（符合预期，
  不是 bug）。
- 额外断言 manifest 中存在 ≥1 条 real_vae_rt 行（存在性断言）：证明分层插入确实跑了，而不是
  因为被 min_real 守卫豁免而“看起来通过”。
"""
from __future__ import annotations


def test_mock_smoke_passes_v1_v7_and_pair_assertions(tmp_path):
    from forgery_pipeline.config import PipelineConfig, StageScales, GeneratorSpec
    from forgery_pipeline.pipeline import run_pipeline
    from forgery_pipeline import manifest
    from forgery_pipeline.validate import check_all
    cfg = PipelineConfig(out_dir=str(tmp_path / "run"), seed=0, backend="mock",
                         stages={"d0": True, "d1": True, "d2": True, "d3": True,
                                 "d4": False, "postprocess": True, "split": True},
                         scales=StageScales(d0=40, d1_per_generator=2, d2=10, d3=6),
                         generators=[GeneratorSpec("g1", "gan", "whole")],
                         inpainters=[GeneratorSpec("i1", "diffusion", "inpaint")],
                         vae_rt_frac=0.3)
    st = run_pipeline(cfg)
    rows = manifest.read_jsonl(tmp_path / "run" / "manifest.jsonl")
    assert check_all(rows, profile="run") == []
    # W1T2 接线冒烟：V9/V10 参数透传不破坏主链 manifest（本 run 无 kandinsky 行、
    # builder 不设 operator 字段 → 空真绿，但证明参数通路接通且 V8 裁决A豁免生效）
    assert check_all(rows, profile="run", holdout_generators={"kandinsky-inpaint"},
                     testc_holdout="object_replacement") == []
    assert "by_sample_kind" in st and "io_chain_by_fake_split" in st
    # amendment B：存在性断言——证明 vae_rt 分层插入确实跑了（而非被 min_real 守卫豁免所掩盖）
    assert any(r.sample_kind == "real_vae_rt" for r in rows)

    # 成对 probe（PATCH 7.3 gate0 定位复查用）
    from forgery_pipeline.builders.probe import run_probe
    run_probe(tmp_path / "p", n_base=4, strengths=[0.5], operators=["inpaint"],
              img2img_specs=[GeneratorSpec("g", "diffusion", "img2img")],
              inpainter_specs=[GeneratorSpec("i", "diffusion", "inpaint")],
              seed=0, compositing_pairs=4)
    from scripts.assert_compositing_pairs import check
    assert check(tmp_path / "p") == []

"""PATCH 7 冒烟验收 e2e（任务 7.5）：mock 全链 → V1–V7 全过 + 成对 compositing 断言过。

controller 修正 B（覆盖 task-7-brief 的 cfg 取值）：
- scales d0=40（d1_per_generator=2、d2=10、d3=6 不变，d4 关闭）、vae_rt_frac=0.3——
  经验证 train split 稳定得到 real≥10（当时运行 real=14；PATCH 9 Wave2 裁决② 把
  object_replacement 组也路由 test_c 后实测为 13，仍 ≥10，V4 不被 min_real 守卫豁免，
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
    # W1T2 接线冒烟：V9 参数透传不破坏主链 manifest（本 run 无 kandinsky 行 → 空真绿，
    # 但证明参数通路接通且 V8 裁决A豁免生效）。
    # testc_holdout="object_replacement" 自 Wave2 裁决② 起是**真断言**（非 vacuous）：
    # D2 政策接线让 builder 真的产出 operator=object_replacement 的行，且该类行的 level3
    # 恰为 "object_replacement"——configs/split.yaml 已把它加入 holdout_manipulation
    # （level3 键），splitter 按 origin-group 整组路由 test_c，故 V10 在真实路由下全绿。
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


def test_object_replacement_routes_to_test_c_and_v10_green(tmp_path):
    """PATCH 9 Wave2 裁决②：object_replacement 的 V10 路由闭环（零新代码）。

    D2 政策接线产出的 operator=object_replacement 行，其 manipulation_level3 恰为同名
    "object_replacement"；configs/split.yaml 把它加入 holdout_manipulation（level3 键）后，
    splitter 按 origin-group 整组路由 test_c——路由把行送进 test_c，V10
    （testc_holdout=object_replacement，operator 键）断言没有漏网进 train/val，两机制协同。
    cfg 规模说明：d0=10 时两个 Places 底图组必进 test_d，train 的 real 行数 <10，V4 被
    min_real 守卫豁免（vae_rt 关闭）；d2=8 使 MANIP_TYPES 轮转必产出 object_replacement
    行（index 1），断言非空真。且 d2 底图数=5（与 7 类轮转互质）使 object_replacement 行
    与 text_editing 行落在**不同**底图组——若同组，text_editing（已有 holdout_manipulation
    路由）会把 object_replacement 行捎带进 test_c，本测试就成了区分不了新路由的空转断言
    （d0=8 即如此：5≡1 (mod 4) 两类行结构性同组，TDD 红阶段实测发现）。
    """
    from forgery_pipeline.config import PipelineConfig, StageScales, GeneratorSpec
    from forgery_pipeline.pipeline import run_pipeline
    from forgery_pipeline import manifest
    from forgery_pipeline.validate import check_all
    cfg = PipelineConfig(out_dir=str(tmp_path / "run"), seed=0, backend="mock",
                         stages={"d0": True, "d1": False, "d2": True, "d3": False,
                                 "d4": False, "postprocess": False, "split": True},
                         scales=StageScales(d0=10, d2=8),
                         inpainters=[GeneratorSpec("i1", "diffusion", "inpaint")],
                         vae_rt_frac=0.0)
    run_pipeline(cfg)
    rows = manifest.read_jsonl(tmp_path / "run" / "manifest.jsonl")
    orep = [r for r in rows if r.operator == "object_replacement"]
    assert orep  # 非空真前提：确有 object_replacement 行参与下面的路由断言
    assert all(r.split == "test_c" for r in orep)
    assert check_all(rows, profile="run", testc_holdout="object_replacement") == []

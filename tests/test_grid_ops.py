"""grid_ops：主 run img2img/outpaint 算子轴（PATCH 9 Wave2 Task3）。

D0-D4 主链缺 img2img/outpaint 两类操纵行——B1 矩阵要求它们作为训练可见轴。断言风格
沿用 test_builder_d2.py 的密集单测（不逐字段拆分成海量小测试）；theoretical border
面积在测试里独立按公式重算（不 import grid_ops 的 `_border_mask`），避免自证式测试。
"""
import dataclasses
import json
from pathlib import Path

from forgery_pipeline import image_io, manifest, prompts
from forgery_pipeline.builders.d0_real import build_d0
from forgery_pipeline.builders.grid_ops import build_grid
from forgery_pipeline.config import GeneratorSpec, PipelineConfig, StageScales

_IMG2IMG = [GeneratorSpec("stable-diffusion-img2img", "diffusion", "img2img"),
            GeneratorSpec("sdxl-img2img", "diffusion-sdxl", "img2img")]
_INPS = [GeneratorSpec("stable-diffusion-inpaint", "diffusion", "inpaint")]


def _cfg(tmp_path, **overrides) -> PipelineConfig:
    """满足 build_grid 政策接线所需字段的最小 PipelineConfig（同 test_builder_d2.py 的
    _min_config 惯例），其余字段用 spec 默认值。"""
    kw = dict(out_dir=str(tmp_path / "run"), seed=0, backend="mock",
              stages={}, scales=StageScales())
    kw.update(overrides)
    return PipelineConfig(**kw)


def _theoretical_border_ratios(h, w, fracs) -> set[float]:
    """边带掩码理论面积占比：与 grid_ops._border_mask 的像素级实现独立重算（公式而非
    调用被测代码），避免测试沦为对实现自身的重复断言。"""
    out = set()
    for bf in fracs:
        b = int(min(h, w) * bf)
        inner_h, inner_w = max(h - 2 * b, 0), max(w - 2 * b, 0)
        out.add(1.0 - (inner_h * inner_w) / (h * w))
    return out


# ---------------------------------------------------------------------------
# 行数、operator/task_type/level 值域、base_id/sample_kind/io_chain/real_image_path 惯例
# ---------------------------------------------------------------------------

def test_build_grid_row_counts_and_field_conventions(tmp_path):
    bases = build_d0(tmp_path, n=6, seed=0)
    cfg = _cfg(tmp_path)
    samples = build_grid(tmp_path, bases, _IMG2IMG, _INPS, cfg, seed=0)

    assert len(samples) == 6 * (len(_IMG2IMG) + 1) == 18
    i2i = [s for s in samples if s.operator == "img2img"]
    outp = [s for s in samples if s.operator == "outpaint"]
    assert len(i2i) == 12 and len(outp) == 6
    assert {s.operator for s in samples} == {"img2img", "outpaint"}

    base_ids = {b.image_id for b in bases}
    base_paths = {b.image_path for b in bases}
    for s in samples:
        assert s.base_id in base_ids
        assert s.sample_kind == "edited"
        assert s.io_chain and s.io_chain.startswith("decode>") and s.io_chain.endswith(">png")
        assert ">edit:" in s.io_chain
        # V8 前提：real_image_path 必须回指某个 base 的 image_path，使 origin_key 与
        # base_id 组一致（否则 assign_splits 不会把 grid 行并入底图的 origin-group）。
        assert s.real_image_path in base_paths
        assert s.source_dataset is not None

    for s in i2i:
        assert s.compositing == "none"
        assert s.task_type.value == "whole_image_detection"
        assert s.manipulation_level1 == "whole_generated"
        assert s.manipulation_level2 == "diffusion"
        assert s.mask_path is None
        assert 0.1 <= s.strength <= 0.95

    for s in outp:
        assert s.compositing in {"none", "paste_feather"}
        assert s.task_type.value == "localization"
        assert s.manipulation_level1 == "partial_manipulated"
        assert s.manipulation_level2 == "AIGC-editing"
        assert s.manipulation_level3 == "image_guided_editing"
        assert s.mask_path and (Path(tmp_path) / s.mask_path).exists()
        assert (Path(tmp_path) / s.image_path).exists()


# ---------------------------------------------------------------------------
# img2img 强度：连续性 + 确定性 + init_timestep 公式
# ---------------------------------------------------------------------------

def test_build_grid_img2img_strength_continuous_and_init_timestep(tmp_path):
    bases = build_d0(tmp_path, n=12, seed=0)
    cfg = _cfg(tmp_path)
    samples = build_grid(tmp_path, bases, _IMG2IMG, _INPS, cfg, seed=0)
    i2i = [s for s in samples if s.operator == "img2img"]
    assert len(i2i) == 24  # >=20 行门槛

    distinct = {s.strength for s in i2i}
    assert len(distinct) > 10, "strength 应连续取值（非离散网格），唯一值应远超 10 个"
    for s in i2i:
        assert cfg.strength_range[0] <= s.strength <= cfg.strength_range[1]
        assert s.init_timestep == round(s.strength * 999)


# ---------------------------------------------------------------------------
# outpaint 掩码面积：命中边带宽度网格理论面积（±0.03）
# ---------------------------------------------------------------------------

def test_build_grid_outpaint_mask_area_matches_border_theory(tmp_path):
    bases = build_d0(tmp_path, n=6, seed=0)
    cfg = _cfg(tmp_path)
    samples = build_grid(tmp_path, bases, _IMG2IMG, _INPS, cfg, seed=0)
    outp = [s for s in samples if s.operator == "outpaint"]
    assert outp

    by_id = {b.image_id: b for b in bases}
    for s in outp:
        base = by_id[s.base_id]
        img = image_io.load_image(Path(tmp_path) / base.image_path)
        h, w = img.shape[:2]
        theoretical = _theoretical_border_ratios(h, w, cfg.outpaint_border_fracs)
        assert any(abs(s.mask_area_ratio - t) <= 0.03 for t in theoretical), (
            f"{s.mask_area_ratio} 未命中任一理论边带面积 {theoretical}")


# ---------------------------------------------------------------------------
# op_params 四键 + prompt 来自对应 bank 节
# ---------------------------------------------------------------------------

def test_build_grid_op_params_four_keys_and_prompt_bank_kind(tmp_path):
    bases = build_d0(tmp_path, n=6, seed=0)
    cfg = _cfg(tmp_path)
    bank = prompts.load_bank(cfg.prompt_bank)
    bver = prompts.bank_version(cfg.prompt_bank)
    samples = build_grid(tmp_path, bases, _IMG2IMG, _INPS, cfg, seed=0)

    i2i_bank, bg_bank = set(bank["img2img"]), set(bank["background"])
    for s in samples:
        params = json.loads(s.op_params)
        assert set(params) == {"cfg_scale", "steps", "prompt", "prompt_bank_version"}
        assert params["cfg_scale"] in cfg.nuisance_cfg_grid
        assert params["steps"] in cfg.nuisance_steps_grid
        assert params["prompt_bank_version"] == bver
        assert s.prompt == params["prompt"]
        if s.operator == "img2img":
            assert s.prompt in i2i_bank
        else:
            assert s.prompt in bg_bank


# ---------------------------------------------------------------------------
# 确定性：同参数重跑逐行相等
# ---------------------------------------------------------------------------

def test_build_grid_is_deterministic_across_reruns(tmp_path):
    out1, out2 = tmp_path / "r1", tmp_path / "r2"
    b1 = build_d0(out1, n=6, seed=0)
    b2 = build_d0(out2, n=6, seed=0)
    cfg = _cfg(tmp_path)
    s1 = build_grid(out1, b1, _IMG2IMG, _INPS, cfg, seed=0)
    s2 = build_grid(out2, b2, _IMG2IMG, _INPS, cfg, seed=0)
    assert len(s1) == len(s2) == 18
    for a, b in zip(s1, s2):
        assert a.image_id == b.image_id
        assert a.operator == b.operator
        assert a.op_params == b.op_params
        assert a.strength == b.strength
        assert a.mask_area_ratio == b.mask_area_ratio
        assert a.prompt == b.prompt


# ---------------------------------------------------------------------------
# 空输入优雅返回
# ---------------------------------------------------------------------------

def test_build_grid_empty_inputs_return_empty(tmp_path):
    cfg = _cfg(tmp_path)
    assert build_grid(tmp_path, [], _IMG2IMG, _INPS, cfg, seed=0) == []
    bases = build_d0(tmp_path, n=2, seed=0)
    assert build_grid(tmp_path, bases, [], [], cfg, seed=0) == []


# ---------------------------------------------------------------------------
# pipeline e2e：grid_per_op>0 时行入主 manifest 且 V1-V10 全绿
# ---------------------------------------------------------------------------

def test_pipeline_grid_stage_e2e_manifest_and_check_all_green(tmp_path):
    from forgery_pipeline.config import load_config
    from forgery_pipeline.pipeline import run_pipeline
    from forgery_pipeline.validate import check_all

    cfg = load_config("configs/pipeline.example.yaml")
    stages = dict(cfg.stages)
    stages["grid"] = True
    cfg = dataclasses.replace(
        cfg, out_dir=str(tmp_path / "run"), stages=stages, grid_per_op=4,
        scales=StageScales(d0=16, d1_per_generator=1, d2=8, d3=4, d4=3))

    st = run_pipeline(cfg)
    assert st["total"] > 0
    rows = manifest.read_jsonl(Path(cfg.out_dir) / "manifest.jsonl")
    assert any(r.operator == "img2img" for r in rows)
    assert any(r.operator == "outpaint" for r in rows)

    errs = check_all(rows, profile="run", testc_holdout="object_replacement")
    assert errs == [], f"check_all 非空: {errs[:5]}"

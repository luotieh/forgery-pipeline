import json
from pathlib import Path

import numpy as np

from forgery_pipeline import prompts
from forgery_pipeline.config import GeneratorSpec, PipelineConfig, StageScales
from forgery_pipeline.builders.d0_real import build_d0
from forgery_pipeline.builders.d2_local import build_d2, MANIP_TYPES

# level3→operator / level3→prompt-kind 的期望映射直接抄自 PATCH 9 Wave2 Task2 binding
# spec 的文字描述（预裁决①「operator 是粗轴」+ prompt kind 规则），不导入 d2_local 内部
# 的 _OP_MAP/_PROMPT_KIND_MAP，避免测试变成对实现的自证。level3 与 MANIP_TYPES 里的
# mtype 一一对应，可从行上的 manipulation_level3 无损反推 mtype。
_EXPECT_OPERATOR_BY_LEVEL3 = {"object_replacement": "object_replacement",
                              "image_guided_editing": "background_editing"}
_EXPECT_PROMPT_KIND_BY_LEVEL3 = {"object_replacement": "object", "mask_guided_inpainting": "object",
                                 "image_guided_editing": "background"}


def _min_config(tmp_path, **overrides) -> PipelineConfig:
    """满足 build_d2 政策接线所需字段的最小 PipelineConfig（其余字段用 spec 默认值）。"""
    kw = dict(out_dir=str(tmp_path / "run"), seed=0, backend="mock",
              stages={}, scales=StageScales())
    kw.update(overrides)
    return PipelineConfig(**kw)


def test_build_d2_localization(tmp_path):
    base = build_d0(tmp_path, n=6, seed=0)
    inps = [GeneratorSpec("stable-diffusion-inpaint", "diffusion", "inpaint")]
    samples = build_d2(tmp_path, base, n=4, inpainters=inps, seed=0)
    assert len(samples) == 4
    for s in samples:
        assert s.is_fake == 1
        assert s.task_type.value == "localization"
        assert s.manipulation_level1 == "partial_manipulated"
        assert s.mask_path and (Path(tmp_path) / s.mask_path).exists()
        assert 0.01 <= s.mask_area_ratio <= 0.50
        assert s.real_image_path is not None


def test_manip_types_cover_seven():
    assert len(MANIP_TYPES) == 7


# ---------------------------------------------------------------------------
# PATCH 9 Wave 2 Task 2：D2 政策接线（operator 映射 + nuisance/prompt/面积分层）
# ---------------------------------------------------------------------------

def test_build_d2_policies_none_leaves_new_fields_unset(tmp_path):
    """回归锚（裁决①修订）：policies 缺省 None 时不设 operator/op_params，prompt 仍是
    MANIP_TYPES 里的英文模板——config 驱动政策未接入前的口径，供既存调用点（d3/d4/probe
    等未传 policies 的 build_d2 调用）保持不变。mask_area_ratio 不受 policies 门控：该字段
    自 D2 初版（commit 5a9320c）起就无条件落行，政策接线只新增 operator/op_params 与
    prompt 政策，不得移除既有记录。"""
    base = build_d0(tmp_path, n=6, seed=0)
    inps = [GeneratorSpec("stable-diffusion-inpaint", "diffusion", "inpaint")]
    samples = build_d2(tmp_path, base, n=4, inpainters=inps, seed=0)
    templates = {tmpl for _, _, tmpl in MANIP_TYPES}
    assert len(samples) == 4
    for s in samples:
        assert s.operator is None
        assert s.op_params is None
        assert s.mask_area_ratio is not None and 0.01 <= s.mask_area_ratio <= 0.50
        assert s.prompt in templates


def test_build_d2_with_policies_sets_operator_nuisance_prompt_area(tmp_path):
    base = build_d0(tmp_path, n=12, seed=0)
    inps = [GeneratorSpec("stable-diffusion-inpaint", "diffusion", "inpaint")]
    cfg = _min_config(tmp_path)
    bank = prompts.load_bank(cfg.prompt_bank)
    bver = prompts.bank_version(cfg.prompt_bank)
    samples = build_d2(tmp_path, base, n=24, inpainters=inps, seed=0, policies=cfg)
    assert len(samples) == 24
    seen_operators = set()
    for s in samples:
        expected_op = _EXPECT_OPERATOR_BY_LEVEL3.get(s.manipulation_level3, "inpaint")
        assert s.operator == expected_op
        assert s.operator in {"inpaint", "object_replacement", "background_editing"}
        seen_operators.add(s.operator)

        params = json.loads(s.op_params)
        assert set(params) == {"cfg_scale", "steps", "prompt", "prompt_bank_version"}
        assert params["cfg_scale"] in cfg.nuisance_cfg_grid
        assert params["steps"] in cfg.nuisance_steps_grid
        assert params["prompt_bank_version"] == bver

        assert s.mask_area_ratio is not None and 0.0 < s.mask_area_ratio < 1.0

        expected_kind = _EXPECT_PROMPT_KIND_BY_LEVEL3.get(s.manipulation_level3, "inpaint")
        expected_prompt = prompts.pick_prompt(bank, expected_kind, s.image_id)
        assert s.prompt and s.prompt == expected_prompt
        assert params["prompt"] == s.prompt
    # n=24 覆盖 7 种操纵类型轮转 3+ 圈，三个 operator 值都应出现
    assert seen_operators == {"inpaint", "object_replacement", "background_editing"}


def test_build_d2_with_policies_is_deterministic_across_reruns(tmp_path):
    inps = [GeneratorSpec("stable-diffusion-inpaint", "diffusion", "inpaint")]
    cfg = _min_config(tmp_path)
    out1, out2 = tmp_path / "r1", tmp_path / "r2"
    base1 = build_d0(out1, n=12, seed=0)
    base2 = build_d0(out2, n=12, seed=0)
    s1 = build_d2(out1, base1, n=24, inpainters=inps, seed=0, policies=cfg)
    s2 = build_d2(out2, base2, n=24, inpainters=inps, seed=0, policies=cfg)
    assert len(s1) == len(s2) == 24
    for a, b in zip(s1, s2):
        assert a.image_id == b.image_id
        assert a.operator == b.operator
        assert a.op_params == b.op_params
        assert a.mask_area_ratio == b.mask_area_ratio
        assert a.prompt == b.prompt


def test_build_d2_with_policies_area_buckets_all_covered(tmp_path):
    """面积桶分层（9.2b）：n 足够时默认 4 桶（area_buckets=[0.05,0.15,0.35,0.7]）都被采到。"""
    base = build_d0(tmp_path, n=12, seed=0)
    inps = [GeneratorSpec("stable-diffusion-inpaint", "diffusion", "inpaint")]
    cfg = _min_config(tmp_path)
    samples = build_d2(tmp_path, base, n=40, inpainters=inps, seed=0, policies=cfg)
    buckets = {int(np.digitize(s.mask_area_ratio, cfg.area_buckets)) for s in samples}
    assert buckets == set(range(len(cfg.area_buckets)))

# tests/test_prompts.py —— PATCH 9 Wave 2 Task 1：prompt bank + config 采样政策字段（TDD 先红）
from pathlib import Path

import pytest
import yaml

from forgery_pipeline.config import PipelineConfig, StageScales, load_config
from forgery_pipeline.prompts import bank_version, load_bank, pick_prompt

BANK_PATH = "configs/prompt_bank.yaml"


def test_load_bank_has_four_nonempty_sections():
    bank = load_bank(BANK_PATH)
    for kind in ("img2img", "inpaint", "object", "background"):
        assert isinstance(bank[kind], list)
        assert len(bank[kind]) > 0
        assert all(isinstance(p, str) and p for p in bank[kind])


def test_load_bank_missing_section_raises(tmp_path):
    # 缺 background 节的残缺 bank：load_bank 必须拒绝并在错误信息里点名缺失节
    partial = tmp_path / "partial_bank.yaml"
    partial.write_text(yaml.dump({
        "img2img": ["a"], "inpaint": ["b"], "object": ["c"],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="background"):
        load_bank(str(partial))


def test_pick_prompt_unknown_kind_raises():
    bank = load_bank(BANK_PATH)
    with pytest.raises(ValueError):
        pick_prompt(bank, "outpaint", "any-key")


def test_pick_prompt_same_key_is_deterministic():
    bank = load_bank(BANK_PATH)
    p1 = pick_prompt(bank, "img2img", "sample-042")
    p2 = pick_prompt(bank, "img2img", "sample-042")
    assert p1 == p2


def test_pick_prompt_different_keys_give_different_prompts():
    # "img2img-key-a"/"img2img-key-b" 已用 stable_hash 对当前 prompt_bank.yaml 的
    # img2img 节（len=22）验证：分别落在 idx 4 与 idx 7——不同槽位，故此断言非偶然。
    # 若未来编辑 img2img 节改变了长度/顺序，需重新核验这两个 key 仍落在不同槽位。
    bank = load_bank(BANK_PATH)
    p1 = pick_prompt(bank, "img2img", "img2img-key-a")
    p2 = pick_prompt(bank, "img2img", "img2img-key-b")
    assert p1 != p2


def test_bank_version_is_12_char_hex_and_changes_on_one_byte_edit(tmp_path):
    original = Path(BANK_PATH).read_bytes()
    copy_path = tmp_path / "bank_copy.yaml"
    copy_path.write_bytes(original)

    v1 = bank_version(str(copy_path))
    assert len(v1) == 12
    int(v1, 16)  # 必须是合法十六进制，否则 int() 抛异常

    copy_path.write_bytes(original + b"\n# one extra byte-ish comment\n")
    v2 = bank_version(str(copy_path))
    assert v1 != v2


def test_pipeline_config_new_policy_fields_have_spec_defaults():
    # 不传任何 Wave2 新字段即可构造，且新字段取 brief 指定的默认值（向后兼容锚之一）
    cfg = PipelineConfig(out_dir="d", seed=0, backend="mock",
                         stages={}, scales=StageScales())
    assert cfg.nuisance_cfg_grid == [5.0, 7.5, 10.0]
    assert cfg.nuisance_steps_grid == [30, 50]
    assert cfg.strength_range == (0.1, 0.95)
    assert cfg.area_buckets == [0.05, 0.15, 0.35, 0.7]
    assert cfg.outpaint_border_fracs == [0.125, 0.25]
    assert cfg.resolution_groups == {}
    assert cfg.prompt_bank == "configs/prompt_bank.yaml"
    assert cfg.grid_per_op == 0


def test_load_config_backward_compatible_with_legacy_yaml_without_new_keys(tmp_path):
    # 旧式 yaml（不含任何 Wave2 新键）：load_config 不炸，新字段落到 spec 默认值——向后兼容锚
    legacy = {
        "out_dir": str(tmp_path / "run"),
        "seed": 1234,
        "backend": "mock",
        "stages": {"d0": True, "d1": True, "d2": True, "d3": True, "d4": True,
                   "qc": True, "postprocess": True, "split": True},
        "scales": {"d0": 4, "d1_per_generator": 1, "d2": 2, "d3": 1, "d4": 1},
        "generators_config": "configs/generators.yaml",
        "split_config": "configs/split.yaml",
    }
    p = tmp_path / "legacy_pipeline.yaml"
    p.write_text(yaml.dump(legacy), encoding="utf-8")

    cfg = load_config(str(p))
    assert cfg.nuisance_cfg_grid == [5.0, 7.5, 10.0]
    assert cfg.nuisance_steps_grid == [30, 50]
    assert cfg.strength_range == (0.1, 0.95)
    assert cfg.area_buckets == [0.05, 0.15, 0.35, 0.7]
    assert cfg.outpaint_border_fracs == [0.125, 0.25]
    assert cfg.resolution_groups == {}
    assert cfg.prompt_bank == "configs/prompt_bank.yaml"
    assert cfg.grid_per_op == 0

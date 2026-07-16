"""V8–V10 split 防泄漏校验测试（PATCH 9 Wave 1 Task 2，含控制器裁决A/B）。

_row/_has helper 原样比照 tests/test_validate_v7.py 的构造惯例复制（该文件本身也是独立自带
helper、不依赖共享模块，此处保持同一约定，避免跨测试文件耦合）。

裁决A：V8 组一致性排除 postprocess 退化行；退化行单独断言与母行同 split，唯一豁免
test_a→test_e（splitter 的 degradation carve-out，eval→eval 无训练泄漏）。
裁决B：V8/V10 仅 profile=="run" 执行（probe 产物是受控仪器，故意同底图跨 holdout
生成器、算子网格进 train，validator 不罚仪器设计）。
"""
from forgery_pipeline.validate import check_all
from forgery_pipeline.schema import Sample, TaskType


def _row(i, f, **kw):
    """构造满足 labels 校验器的最小合法 Sample（同 tests/test_validate_v7.py）：
    is_fake=1 无 mask_path 时按 whole_generated 处理，有 mask_path 时按 partial_manipulated
    处理；is_fake=0 固定 real_pristine。其余字段按 kw 覆盖/追加。
    """
    has_mask = bool(kw.get("mask_path"))
    kw.setdefault("image_path", f"{i}.png")
    kw.setdefault("split", "train")
    if f:
        kw.setdefault("task_type", TaskType.localization if has_mask
                      else TaskType.whole_image_detection)
        kw.setdefault("manipulation_level1", "partial_manipulated" if has_mask
                      else "whole_generated")
        kw.setdefault("sample_kind", "edited")
    else:
        kw.setdefault("task_type", TaskType.real_pristine)
        kw.setdefault("sample_kind", "real")
    return Sample(image_id=i, is_fake=f, **kw)


def _has(errs, prefix):
    return any(e.startswith(prefix) for e in errs)


# ---------------------------------------------------------------------------
# V8：base_id 组 split 互斥 + postprocess 派生行与母行同 split（仅 run profile，裁决B）
# ---------------------------------------------------------------------------

def test_v8_same_base_id_across_splits_fails():
    rows = [
        _row("r0", 0, split="train", base_id="shared"),
        _row("r1", 0, split="val", base_id="shared"),
    ]
    errs = check_all(rows, profile="run")
    assert _has(errs, "V8: ")


def test_v8_postprocess_row_split_mismatch_fails():
    rows = [
        _row("f0", 1, split="train"),                          # 母行
        _row("f0__deg", 1, split="val", postprocess_of="f0"),   # 派生行 split 与母行不一致
    ]
    errs = check_all(rows, profile="run")
    assert _has(errs, "V8: ")


def test_v8_green_when_groups_consistent():
    rows = [
        _row("r0", 0, split="train", base_id="r0"),
        _row("f0", 1, split="train", base_id="r0"),
        _row("f0__deg", 1, split="train", base_id="r0", postprocess_of="f0"),
    ]
    assert not _has(check_all(rows, profile="run"), "V8: ")


def test_v8_postprocess_test_a_to_test_e_carveout_allowed():
    """裁决A豁免正例：test_a→test_e 退化 carve-out 属既定设计（eval→eval 无训练泄漏）。
    两行共享 base_id：同时覆盖组一致性对 postprocess 行的排除（否则组侧就会先误报）。"""
    rows = [
        _row("f0", 1, split="test_a", base_id="b0"),
        _row("f0__deg", 1, split="test_e", base_id="b0", postprocess_of="f0"),
    ]
    assert not _has(check_all(rows, profile="run"), "V8: ")


def test_v8_postprocess_train_to_test_e_fails():
    """裁决A豁免负例：train→test_e 不在豁免内（母行在训练侧，退化行必须同 split）。"""
    rows = [
        _row("f0", 1, split="train", base_id="b0"),
        _row("f0__deg", 1, split="test_e", base_id="b0", postprocess_of="f0"),
    ]
    assert _has(check_all(rows, profile="run"), "V8: ")


# ---------------------------------------------------------------------------
# V9：cross-generator holdout（holdout_generators）不得泄入 train/val（参数门控，不限 profile）
# ---------------------------------------------------------------------------

def test_v9_holdout_generator_in_train_fails():
    rows = [_row("f0", 1, split="train", generator_name="kandinsky-inpaint")]
    errs = check_all(rows, holdout_generators={"kandinsky-inpaint"})
    assert _has(errs, "V9: ")


def test_v9_holdout_family_in_val_fails():
    rows = [_row("f0", 1, split="val", generator_family="kandinsky")]
    errs = check_all(rows, holdout_generators={"kandinsky"})
    assert _has(errs, "V9: ")


def test_v9_skipped_when_param_none():
    rows = [_row("f0", 1, split="train", generator_name="kandinsky-inpaint")]
    errs = check_all(rows)   # holdout_generators=None 默认 → 跳过
    assert not _has(errs, "V9: ")


# ---------------------------------------------------------------------------
# V10：Test-C holdout 算子（testc_holdout）不得泄入 train/val（仅 run profile，裁决B）
# ---------------------------------------------------------------------------

def test_v10_testc_holdout_operator_in_train_fails():
    rows = [_row("f0", 1, split="train", operator="object_replacement", compositing="none")]
    errs = check_all(rows, profile="run", testc_holdout="object_replacement")
    assert _has(errs, "V10: ")


def test_v10_allowed_in_test_c():
    rows = [_row("f0", 1, split="test_c", operator="object_replacement", compositing="none")]
    errs = check_all(rows, profile="run", testc_holdout="object_replacement")
    assert not _has(errs, "V10: ")


def test_v10_skipped_when_param_none():
    rows = [_row("f0", 1, split="train", operator="object_replacement", compositing="none")]
    errs = check_all(rows, profile="run")   # profile 已是 run，仅 testc_holdout=None → 跳过
    assert not _has(errs, "V10: ")


# ---------------------------------------------------------------------------
# 裁决B：profile=="auto" 时 V8/V10 整体跳过（probe 等受控仪器产物不受罚）
# ---------------------------------------------------------------------------

def test_v8_v10_skipped_when_profile_auto():
    rows = [
        _row("r0", 0, split="train", base_id="shared"),
        _row("r1", 0, split="val", base_id="shared"),      # V8 违例行（若 run profile 会报）
        _row("f0", 1, split="train", operator="object_replacement", compositing="none"),
    ]
    errs = check_all(rows, testc_holdout="object_replacement")   # profile 默认 "auto"
    assert not _has(errs, "V8: ") and not _has(errs, "V10: ")

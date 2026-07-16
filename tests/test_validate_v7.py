"""V1–V7 validator 断言集测试（PATCH 7 收尾）。"""
import json
from forgery_pipeline.validate import check_all, nongen_chain
from forgery_pipeline.schema import Sample, TaskType, Postprocess
from forgery_pipeline import manifest
from scripts.backfill_manifest_v7 import backfill


def _row(i, f, **kw):
    """构造满足 labels 校验器的最小合法 Sample（镜像 tests/test_backfill_v7.py 的构造模式）；
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
# nongen_chain 单元测试
# ---------------------------------------------------------------------------

def test_nongen_chain_normalizes_various_chains():
    assert nongen_chain("decode>rs256>edit:sd15_inpaint>png") == "rs256>png"
    assert nongen_chain("gen:stable-diffusion-1-5>rs256>png") == "rs256>png"
    assert nongen_chain("decode>rs256>vae_rt:mock>png") == "rs256>png"
    assert nongen_chain("decode>rs256>png") == "rs256>png"
    assert nongen_chain("legacy") == "legacy"
    assert nongen_chain(None) == ""


# ---------------------------------------------------------------------------
# V1：存储格式与分辨率唯一
# ---------------------------------------------------------------------------

def test_v1_mixed_suffix_within_split_fails_but_postprocess_rows_exempt():
    bad = [
        _row("r0", 0, io_chain="decode>rs256>png"),
        _row("r1", 0, image_path="r1.jpg", io_chain="decode>rs256>png"),
    ]
    assert _has(check_all(bad), "V1: ")

    exempt = [
        _row("r0", 0, io_chain="decode>rs256>png"),
        _row("r1", 0, image_path="r1.jpg", io_chain="decode>rs256>png",
             postprocess=Postprocess(jpeg_quality=80)),
    ]
    assert not _has(check_all(exempt), "V1: ")


# ---------------------------------------------------------------------------
# V2：real/fake 非生成链一致
# ---------------------------------------------------------------------------

def test_v2_mismatched_nongen_chain_between_real_and_fake_fails_with_symdiff():
    rows = [
        _row("r0", 0, io_chain="decode>rs256>png"),
        _row("f0", 1, io_chain="decode>rs512>edit:sd15>png"),
    ]
    errs = check_all(rows)
    v2 = [e for e in errs if e.startswith("V2: ")]
    assert v2
    assert "rs256>png" in v2[0] and "rs512>png" in v2[0]


def test_v2_legacy_rows_exempt_from_comparison():
    rows = [
        _row("r0", 0, io_chain="decode>rs256>png"),
        _row("f0", 1, io_chain="decode>rs256>edit:m>png"),
        _row("f1", 1, io_chain="legacy"),   # legacy fake 行，若不豁免会拖累比较（多出 "legacy"）
    ]
    assert not _has(check_all(rows), "V2: ")


# ---------------------------------------------------------------------------
# V3：masked 算子 compositing 完备性 + 值域
# ---------------------------------------------------------------------------

def test_v3_operator_scope_missing_compositing_fails():
    rows = [_row("f0", 1, mask_path="m.png", operator="inpaint")]
    assert _has(check_all(rows), "V3: ")


def test_v3_bad_compositing_enum_value_fails():
    rows = [_row("f0", 1, mask_path="m.png", operator="inpaint", compositing="overlay")]
    assert _has(check_all(rows), "V3: ")


def test_v3_paste_feather_without_feather_px_fails():
    rows = [_row("f0", 1, mask_path="m.png", operator="inpaint", compositing="paste_feather")]
    assert _has(check_all(rows), "V3: ")


def test_v3_bad_sample_kind_value_fails():
    """T1 遗留值域校验折入 V3：sample_kind 一旦设置须落在枚举内（与 operator 无关）。"""
    rows = [_row("f0", 1, sample_kind="bogus")]
    assert _has(check_all(rows), "V3: ")


# ---------------------------------------------------------------------------
# V4：real_vae_rt 占比（profile 语义）
# ---------------------------------------------------------------------------

def test_v4_profile_run_forces_check_and_fails_without_any_vae_rt_row():
    # 规模改为 range(10)（原 3）：min_real 守卫默认阈值=10，real 行数须达标才不被跳过
    rows = [_row(f"r{i}", 0, split="train") for i in range(10)]
    errs = check_all(rows, profile="run")
    v4 = [e for e in errs if e.startswith("V4: ")]
    assert v4 and "train" in v4[0]


def test_v4_ratio_out_of_range_fails():
    # 规模改为 range(10)+range(10)（原 2+2，比值同为 1.0）：同上，避免被 min_real 守卫跳过
    rows = [_row(f"r{i}", 0, split="train") for i in range(10)]
    rows += [_row(f"v{i}", 0, split="train", sample_kind="real_vae_rt",
                  real_image_path=f"r{i}.png") for i in range(10)]
    errs = check_all(rows)   # profile="auto" 默认；含 real_vae_rt 行即触发
    v4 = [e for e in errs if e.startswith("V4: ")]
    assert v4 and "train" in v4[0]


def test_v4_profile_auto_skips_when_no_vae_rt_rows_present():
    rows = [_row(f"r{i}", 0, split="train") for i in range(2)]
    errs = check_all(rows)   # 无 real_vae_rt 行、profile="auto" → 不触发 V4（min_real 守卫不涉及此分支）
    assert not _has(errs, "V4: ")


# ---------------------------------------------------------------------------
# V4 min_real 守卫（小 n 时比值离散取值结构性落不进 band，见 check_v4 docstring）
# ---------------------------------------------------------------------------

def test_v4_min_real_guard_skips_split_below_threshold():
    """real 行数=5 < min_real(默认 10)：即使比值 0/5=0.0 越界也跳过，不产生 V4 消息。"""
    rows = [_row(f"r{i}", 0, split="train") for i in range(5)]
    errs = check_all(rows, profile="run")
    assert not _has(errs, "V4: ")


def test_v4_min_real_guard_still_fails_at_or_above_threshold():
    """real 行数=12 ≥ min_real(默认 10)：守卫不豁免，比值 0/12=0.0 越界须 FAIL。"""
    rows = [_row(f"r{i}", 0, split="train") for i in range(12)]
    errs = check_all(rows, profile="run")
    v4 = [e for e in errs if e.startswith("V4: ")]
    assert v4 and "train" in v4[0]


def test_v4_min_real_param_threads_through_check_all():
    """min_real 须由 check_all 原样传给 check_v4（非硬编码默认值）。"""
    rows = [_row(f"r{i}", 0, split="train") for i in range(5)]
    assert not _has(check_all(rows, profile="run"), "V4: ")             # 默认 min_real=10 → 跳过
    errs = check_all(rows, profile="run", min_real=3)                   # 显式调小阈值 → 5≥3 生效
    v4 = [e for e in errs if e.startswith("V4: ")]
    assert v4 and "train" in v4[0]


# ---------------------------------------------------------------------------
# V5：向后兼容——backfill 后的旧 manifest 应整体通过 check_all
# ---------------------------------------------------------------------------

def test_v5_backfilled_legacy_manifest_passes_check_all(tmp_path):
    rows = [
        Sample(image_id="r0", image_path="a.jpg", is_fake=0,
               task_type=TaskType.real_pristine),
        Sample(image_id="f0", image_path="b.jpg", is_fake=1, operator="inpaint",
               mask_path="m.png", task_type=TaskType.localization,
               manipulation_level1="partial_manipulated"),
    ]
    p = tmp_path / "old.jsonl"
    manifest.write_jsonl(p, rows)
    out = tmp_path / "new.jsonl"
    backfill(p, out)
    backfilled = manifest.read_jsonl(out)
    assert check_all(backfilled, profile="auto") == []


# ---------------------------------------------------------------------------
# V6：instruct_edit 行 op_params 完备性
# ---------------------------------------------------------------------------

def test_v6_instruct_edit_invalid_json_fails():
    rows = [_row("f0", 1, operator="instruct_edit", op_params="{not valid json")]
    assert _has(check_all(rows), "V6: ")


def test_v6_instruct_edit_missing_image_guidance_scale_fails():
    rows = [_row("f0", 1, operator="instruct_edit", op_params=json.dumps({"steps": 20}))]
    assert _has(check_all(rows), "V6: ")


# ---------------------------------------------------------------------------
# V7：成对 probe 样本一致性
# ---------------------------------------------------------------------------

def test_v7_pair_id_count_not_two_fails():
    rows = [
        _row("p0", 1, mask_path="m.png", probe_group="compositing_pair", pair_id="cp0",
             seed=1, real_image_path="base.png", compositing="none", generator_name="g1"),
    ]
    assert _has(check_all(rows), "V7: ")


def test_v7_pair_seed_mismatch_fails():
    rows = [
        _row("p0", 1, mask_path="m.png", probe_group="compositing_pair", pair_id="cp0",
             seed=1, real_image_path="base.png",
             compositing="none", generator_name="g1"),
        _row("p1", 1, mask_path="m.png", probe_group="compositing_pair", pair_id="cp0",
             seed=2, real_image_path="base.png",
             compositing="paste_feather", feather_px=8, generator_name="g1"),
    ]
    assert _has(check_all(rows), "V7: ")


def test_v7_compositing_pair_generator_name_mismatch_fails():
    rows = [
        _row("p0", 1, mask_path="m.png", probe_group="compositing_pair", pair_id="cp0",
             seed=1, real_image_path="base.png",
             compositing="none", generator_name="g1"),
        _row("p1", 1, mask_path="m.png", probe_group="compositing_pair", pair_id="cp0",
             seed=1, real_image_path="base.png",
             compositing="paste_feather", feather_px=8, generator_name="g2"),
    ]
    assert _has(check_all(rows), "V7: ")


# ---------------------------------------------------------------------------
# Green path：结构完整的 manifest 应整体通过 check_all
# ---------------------------------------------------------------------------

def test_check_all_green_path_passes_on_well_formed_manifest():
    rows = [
        _row("r0", 0, split="train", io_chain="decode>rs256>png"),
        _row("r1", 0, split="train", io_chain="decode>rs256>png"),
        _row("r2", 0, split="train", io_chain="decode>rs256>png"),
        _row("v0", 0, split="train", sample_kind="real_vae_rt",
             real_image_path="r0.png", io_chain="decode>rs256>vae_rt:mock>png"),
        _row("f0", 1, split="train", io_chain="gen:sd15>rs256>png"),
        _row("f1", 1, split="train", mask_path="m.png", operator="inpaint",
             compositing="paste_feather", feather_px=8,
             io_chain="decode>rs256>edit:sd15_inpaint>png"),
        _row("f2", 1, split="train", operator="instruct_edit",
             op_params=json.dumps({"image_guidance_scale": 1.5, "guidance_scale": 7.5}),
             io_chain="decode>rs256>edit:ip2p>png"),
        _row("cp0", 1, split="train", mask_path="m.png", probe_group="compositing_pair",
             pair_id="cp0", seed=7, real_image_path="base.png", operator="inpaint",
             compositing="none", generator_name="g1",
             io_chain="decode>rs256>edit:g1>png"),
        _row("cp1", 1, split="train", mask_path="m.png", probe_group="compositing_pair",
             pair_id="cp0", seed=7, real_image_path="base.png", operator="inpaint",
             compositing="paste_feather", feather_px=8, generator_name="g1",
             io_chain="decode>rs256>edit:g1>png"),
        _row("nd0", 1, split="train", mask_path="m.png", probe_group="nd_pair",
             pair_id="nd0", seed=9, real_image_path="base2.png",
             compositing="none", generator_name="gA",
             io_chain="decode>rs256>edit:gA>png"),
        _row("nd1", 1, split="train", mask_path="m.png", probe_group="nd_pair",
             pair_id="nd0", seed=9, real_image_path="base2.png",
             compositing="none", generator_name="gB",
             io_chain="decode>rs256>edit:gB>png"),
    ]
    assert check_all(rows, profile="run") == []

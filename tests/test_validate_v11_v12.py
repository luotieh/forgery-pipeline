"""V11/V12 校验测试（PATCH 9 Wave 2 Task 5：扩散编辑行 nuisance 记录完备性 / masked 算子行
面积分桶下限 + manifest.stats() 的 by_nuisance_cell/by_area_bucket 扩展）。

_row/_has helper 原样比照 tests/test_validate_v8_v10.py 的既定惯例复制（各校验测试文件
自带 helper、不跨文件耦合，同一约定）。

控制器裁决要点（详见 validate.py 模块 docstring 与 `_is_diffusion_edit_row`）：
- V11 判定域用 io_chain 含 "edit:" 节点（而非 op_params 自指——否则缺记录的行会因为域
  定义本身依赖 op_params 而结构性逃出判定域，检查空转）。
- D3 manual-web-edit 行的豁免键实测改用 `generator_name=="manual-web-edit"`：读
  `d3_web.py` 源码核实其实际写入 `generator_family="editing"`，并非 brief 猜测的
  "manual"/"non_diffusion"。
- 全部单测断言走 `_has`/`not _has` 前缀判定（V11:/V12: 开头），不用全局 `== []`——
  `check_all` 同时跑 V1–V12，构造出的最小合法行可能顺带触发无关检查的噪声消息，机制作用域
  断言只关心被测检查本身（引用 PATCH 9 Wave2 T4 裁决4 教训）。绿路径 e2e 同理。
"""
import dataclasses
import json

from forgery_pipeline.validate import check_all
from forgery_pipeline.schema import Sample, TaskType


def _row(i, f, **kw):
    """构造满足 labels 校验器的最小合法 Sample（同 tests/test_validate_v8_v10.py）：
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
# V11：扩散编辑行 nuisance（cfg_scale/steps）记录完备性（仅 run profile，裁决B）
# ---------------------------------------------------------------------------

def test_v11_missing_cfg_key_fails():
    rows = [_row("f0", 1, io_chain="decode>rs256>edit:sd15>png",
                 generator_family="diffusion", generator_name="sd15",
                 op_params=json.dumps({"steps": 30}))]
    errs = check_all(rows, profile="run")
    assert _has(errs, "V11: ")


def test_v11_missing_steps_key_fails():
    rows = [_row("f0", 1, io_chain="decode>rs256>edit:sd15>png",
                 generator_family="diffusion", generator_name="sd15",
                 op_params=json.dumps({"cfg_scale": 7.5}))]
    errs = check_all(rows, profile="run")
    assert _has(errs, "V11: ")


def test_v11_empty_op_params_in_domain_fails():
    """B3 全部扩散编辑行必录：op_params=None 的 edit 行也 FAIL——判定域不因"没有可解析的
    op_params"而把该行本身排除在外（判定域用 io_chain，不用 op_params 自指）。"""
    rows = [_row("f0", 1, io_chain="decode>rs256>edit:sd15>png",
                 generator_family="diffusion", generator_name="sd15", op_params=None)]
    errs = check_all(rows, profile="run")
    assert _has(errs, "V11: ")


def test_v11_malformed_json_op_params_fails():
    rows = [_row("f0", 1, io_chain="decode>rs256>edit:sd15>png",
                 generator_family="diffusion", generator_name="sd15",
                 op_params="{not valid json")]
    errs = check_all(rows, profile="run")
    assert _has(errs, "V11: ")


def test_v11_non_dict_json_op_params_fails():
    """op_params 是合法 JSON 但不是 object（如数组）——同样判为缺记录。"""
    rows = [_row("f0", 1, io_chain="decode>rs256>edit:sd15>png",
                 generator_family="diffusion", generator_name="sd15",
                 op_params=json.dumps([1, 2, 3]))]
    errs = check_all(rows, profile="run")
    assert _has(errs, "V11: ")


def test_v11_complete_op_params_passes():
    rows = [_row("f0", 1, io_chain="decode>rs256>edit:sd15>png",
                 generator_family="diffusion", generator_name="sd15",
                 op_params=json.dumps({"cfg_scale": 7.5, "steps": 30}))]
    errs = check_all(rows, profile="run")
    assert not _has(errs, "V11: ")


def test_v11_non_numeric_cfg_scale_value_fails_without_crashing():
    """回归锚：op_params 键齐全但值非数值（如手工回填 manifest 误把 cfg_scale 存成字符串
    "7.5"）——:g 格式化会对 str/list 等类型抛 TypeError/ValueError，此前会让该异常穿透
    check_v11 直接崩溃 validate-manifest CLI（已实测复现）；现在应归为"记录不合格"同缺键
    处理，产生正常的 V11 消息而不是抛异常。"""
    rows = [_row("f0", 1, io_chain="decode>rs256>edit:sd15>png",
                 generator_family="diffusion", generator_name="sd15",
                 op_params=json.dumps({"cfg_scale": "7.5", "steps": 30}))]
    errs = check_all(rows, profile="run")   # 不应抛异常
    assert _has(errs, "V11: ")


def test_v11_non_numeric_steps_string_fails_and_stats_excludes():
    """审查修复回归锚（steps 侧守卫对称化）：steps 值为字符串 "30" 时，修复前 cell 拼装的
    steps 位是裸 {}（str() 转换）——非数值静默通过且文本与合法 st30 单元格合并，实测可污染
    nuisance_cell_floor 计数；现与 cfg_scale 同约（:g 渲染失败 → 记录不合格），check_v11
    须发 V11 消息、manifest.stats() 的 by_nuisance_cell 须排除该行。"""
    from forgery_pipeline import manifest
    rows = [_row("f0", 1, io_chain="decode>rs256>edit:sd15>png",
                 generator_family="diffusion", generator_name="sd15",
                 op_params=json.dumps({"cfg_scale": 7.5, "steps": "30"}))]
    errs = check_all(rows, profile="run")
    assert _has(errs, "V11: ")
    assert manifest.stats(rows)["by_nuisance_cell"] == {}


def test_v11_non_numeric_steps_list_fails_and_stats_excludes():
    """同上，steps=[1,2]（列表）：修复前裸 {} 格式化不抛异常，拼出 "st[1, 2]" 垃圾单元格
    计入合规；现应同判记录不合格（:g 对 list 抛 TypeError，走 offenders 路径）。"""
    from forgery_pipeline import manifest
    rows = [_row("f0", 1, io_chain="decode>rs256>edit:sd15>png",
                 generator_family="diffusion", generator_name="sd15",
                 op_params=json.dumps({"cfg_scale": 7.5, "steps": [1, 2]}))]
    errs = check_all(rows, profile="run")
    assert _has(errs, "V11: ")
    assert manifest.stats(rows)["by_nuisance_cell"] == {}


def test_v11_non_diffusion_family_exempt():
    """豁免①：generator_family=="non_diffusion"（预留未来 LaMa 等非扩散 inpainter，无
    cfg/steps 概念）。"""
    rows = [_row("f0", 1, io_chain="decode>rs256>edit:lama>png",
                 generator_family="non_diffusion", generator_name="lama", op_params=None)]
    errs = check_all(rows, profile="run")
    assert not _has(errs, "V11: ")


def test_v11_legacy_io_chain_exempt():
    """豁免②：io_chain=="legacy"（旧谱系整体标记）。"""
    rows = [_row("f0", 1, io_chain="legacy",
                 generator_family="diffusion", generator_name="sd15", op_params=None)]
    errs = check_all(rows, profile="run")
    assert not _has(errs, "V11: ")


def test_v11_manual_web_edit_exempt():
    """豁免③（控制器裁决，实测修正）：generator_name=="manual-web-edit"（D3 网页人工篡改行；
    实际 generator_family=="editing"，不是 brief 猜测的 "manual"/"non_diffusion"，见
    d3_web.py 源码）。"""
    rows = [_row("f0", 1, io_chain="decode>rs256>edit:manual-web-edit>png",
                 mask_path="m.png",
                 generator_family="editing", generator_name="manual-web-edit", op_params=None)]
    errs = check_all(rows, profile="run")
    assert not _has(errs, "V11: ")


def test_v11_gen_node_not_edit_out_of_scope():
    """D1 全生成行：io_chain 是 gen: 节点、不含 edit:，结构性不在判定域内。"""
    rows = [_row("f0", 1, io_chain="gen:g1>rs256>png",
                 generator_family="gan", generator_name="g1", op_params=None)]
    errs = check_all(rows, profile="run")
    assert not _has(errs, "V11: ")


def test_v11_real_row_out_of_scope():
    """is_fake==0 结构性不在判定域内（即便 io_chain 字面恰好含 edit: 也不豁免于该前提，
    仅作防御性覆盖——真实 pipeline 不会给 real 行填 edit: 节点）。"""
    rows = [_row("r0", 0, io_chain="decode>rs256>edit:sd15>png")]
    errs = check_all(rows, profile="run")
    assert not _has(errs, "V11: ")


def test_v11_cell_floor_not_met_fails():
    """nuisance_cell_floor=2，唯一单元格只有 1 行 → FAIL。"""
    rows = [_row("f0", 1, split="train", io_chain="decode>rs256>edit:sd15>png",
                 generator_family="diffusion", generator_name="sd15",
                 op_params=json.dumps({"cfg_scale": 7.5, "steps": 30}))]
    errs = check_all(rows, profile="run", nuisance_cell_floor=2)
    assert _has(errs, "V11: ")


def test_v11_cell_floor_met_passes():
    """nuisance_cell_floor=2，同一单元格恰有 2 行 → 通过。"""
    rows = [_row(f"f{k}", 1, split="train", io_chain="decode>rs256>edit:sd15>png",
                 generator_family="diffusion", generator_name="sd15",
                 op_params=json.dumps({"cfg_scale": 7.5, "steps": 30}))
            for k in range(2)]
    errs = check_all(rows, profile="run", nuisance_cell_floor=2)
    assert not _has(errs, "V11: ")


def test_v11_cell_floor_only_checks_cells_present_in_split():
    """floor 只检查该 split 内**实际出现过**的 (cfg,steps) 单元格——(cfg,steps) 是笛卡尔积，
    某组合在某 split 从未出现是结构性没覆盖到，不该报"计数不足"。这里 train 只有一个单元格
    （cfg7.5/st30，计数=1 < floor=5），预期恰好 1 条 V11 cell 消息，而非对其余理论组合
    也各报一条。"""
    rows = [_row("f0", 1, split="train", io_chain="decode>rs256>edit:sd15>png",
                 generator_family="diffusion", generator_name="sd15",
                 op_params=json.dumps({"cfg_scale": 7.5, "steps": 30}))]
    errs = check_all(rows, profile="run", nuisance_cell_floor=5)
    cell_errs = [e for e in errs if e.startswith("V11: split=")]
    assert len(cell_errs) == 1
    assert "cfg7.5/st30" in cell_errs[0]


# ---------------------------------------------------------------------------
# V12：masked 算子行 mask_area_ratio 完备性 + 面积分桶下限（仅 run profile，裁决B）
# ---------------------------------------------------------------------------

def test_v12_masked_row_missing_ratio_fails():
    rows = [_row("f0", 1, mask_path="m.png", operator="inpaint", mask_area_ratio=None)]
    errs = check_all(rows, profile="run")
    assert _has(errs, "V12: ")


def test_v12_masked_row_with_ratio_passes():
    rows = [_row("f0", 1, mask_path="m.png", operator="inpaint", mask_area_ratio=0.1)]
    errs = check_all(rows, profile="run")
    assert not _has(errs, "V12: ")


def test_v12_non_masked_operator_out_of_scope():
    """operator=="img2img" 不在 masked 集合内，即使 mask_area_ratio=None 也不触发 V12
    （复用 V3 同一 _MASKED_OPS 判定域）。"""
    rows = [_row("f0", 1, operator="img2img", mask_area_ratio=None)]
    errs = check_all(rows, profile="run")
    assert not _has(errs, "V12: ")


def test_v12_bucket_floor_not_met_fails():
    """area_buckets=(0.3, 0.6) 下两个非溢出桶各恰 1 行，floor=2 → 两个桶各报一条。"""
    rows = [
        _row("f0", 1, mask_path="m0.png", operator="inpaint", mask_area_ratio=0.1),   # 桶0
        _row("f1", 1, mask_path="m1.png", operator="outpaint", mask_area_ratio=0.4),  # 桶1
    ]
    errs = check_all(rows, profile="run", area_bucket_floor=2, area_buckets=(0.3, 0.6))
    bucket_errs = [e for e in errs if e.startswith("V12: 面积桶")]
    assert len(bucket_errs) == 2


def test_v12_bucket_floor_met_passes():
    """area_buckets=(0.3, 0.6) 有两个非溢出桶（0/1）——floor 检查逐桶都要满足，故两桶
    各需 ≥2 行（只填一个桶会让另一个桶"命中 0 次"触发 test_v12_bucket_floor_flags_zero_hit_
    bucket verified 的那条规则，不是本测试想覆盖的场景）。"""
    rows = [
        _row(f"f{k}", 1, mask_path=f"m{k}.png", operator="inpaint", mask_area_ratio=0.1)
        for k in range(2)
    ] + [
        _row(f"g{k}", 1, mask_path=f"n{k}.png", operator="outpaint", mask_area_ratio=0.4)
        for k in range(2)
    ]
    errs = check_all(rows, profile="run", area_bucket_floor=2, area_buckets=(0.3, 0.6))
    assert not _has(errs, "V12: 面积桶")


def test_v12_bucket_floor_flags_zero_hit_bucket():
    """全部行落桶0，桶1（[0.3,0.6)）完全无命中——area_bucket_floor>0 时"命中 0 次的桶"
    同样应报告：面积分层覆盖率下限的本意正是抓住"某一档面积完全没有样本"这种最严重的
    覆盖缺口，而非只在"该桶曾出现过"时才检查。"""
    rows = [
        _row(f"f{k}", 1, mask_path=f"m{k}.png", operator="inpaint", mask_area_ratio=0.1)
        for k in range(3)
    ]
    errs = check_all(rows, profile="run", area_bucket_floor=1, area_buckets=(0.3, 0.6))
    assert _has(errs, "V12: 面积桶 1 计数 0 < 1")


def test_v12_overflow_bucket_excluded_from_floor():
    """mask_area_ratio 落在最高边界之上（np.digitize 返回 len(area_buckets)，此处
    area_buckets=(0.3,) 单边界下 ratio=0.9 落溢出桶 index 1）的行不参与下限检查——与
    d2_local.py `_bucketed_pick` 的分层丢弃口径一致。只有 0..len(area_buckets)-1 = {0}
    参与 floor 检查，故预期唯一消息是"桶 0 计数 0 < 1"，不会额外冒出"桶 1"。"""
    rows = [_row("f0", 1, mask_path="m.png", operator="inpaint", mask_area_ratio=0.9)]
    errs = check_all(rows, profile="run", area_bucket_floor=1, area_buckets=(0.3,))
    bucket_errs = [e for e in errs if e.startswith("V12: 面积桶")]
    assert bucket_errs == ["V12: 面积桶 0 计数 0 < 1"]


# ---------------------------------------------------------------------------
# 裁决B：profile=="auto" 时 V11/V12 整体跳过（probe 等受控仪器产物不受罚）
# ---------------------------------------------------------------------------

def test_v11_v12_skipped_when_profile_auto():
    rows = [
        _row("f0", 1, io_chain="decode>rs256>edit:sd15>png",         # V11 违例行（run 下会报）
             generator_family="diffusion", generator_name="sd15", op_params=None),
        _row("f1", 1, mask_path="m.png", operator="inpaint", mask_area_ratio=None),  # V12 违例行
    ]
    # floor 参数一并传入且必不满足，验证 profile=="auto" 时连 floor 检查也整体不执行
    errs = check_all(rows, nuisance_cell_floor=99, area_bucket_floor=99)   # profile 默认 "auto"
    assert not _has(errs, "V11: ") and not _has(errs, "V12: ")


# ---------------------------------------------------------------------------
# manifest.stats()：by_nuisance_cell / by_area_bucket
# ---------------------------------------------------------------------------

def test_stats_by_nuisance_cell_and_by_area_bucket_reasonable_counts():
    from forgery_pipeline import manifest

    rows = [
        _row("f0", 1, op_params=json.dumps({"cfg_scale": 7.5, "steps": 30})),
        _row("f1", 1, op_params=json.dumps({"cfg_scale": 7.5, "steps": 30})),
        _row("f2", 1, op_params=json.dumps({"cfg_scale": 5.0, "steps": 50})),
        _row("f3", 1, op_params=None),                       # 不计入 by_nuisance_cell
        _row("f4", 1, op_params="{not valid json"),          # 不计入 by_nuisance_cell
        _row("f4b", 1, op_params=json.dumps({"cfg_scale": "oops", "steps": 30})),
            # 键齐全但值非数值——:g 格式化会抛异常，回归锚：不应崩溃 stats()，静默不计入
        _row("f5", 1, mask_path="m5.png", operator="inpaint", mask_area_ratio=0.1),   # 默认桶界 → b1
        _row("f6", 1, mask_path="m6.png", operator="inpaint", mask_area_ratio=0.9),   # 默认桶界 → b4
        _row("f7", 1, mask_path="m7.png", operator="inpaint", mask_area_ratio=None),  # 不计入 by_area_bucket
    ]
    st = manifest.stats(rows)
    # 键格式用 :g（general format）——5.0 这种"整数值 float"会被规整为不带小数点的 "5"
    # （而 7.5 保留小数），与 validate.check_v11 的单元格 key 同一格式化口径。
    assert st["by_nuisance_cell"] == {"cfg7.5/st30": 2, "cfg5/st50": 1}
    assert st["by_area_bucket"] == {"b1": 1, "b4": 1}


# ---------------------------------------------------------------------------
# 绿路径：真实 pipeline run（D2 政策接线 + grid + D4）在 run-profile + nuisance_cell_floor=1
# 下 V11/V12 无消息（机制作用域断言——filter 输出取 V11/V12 前缀判空，不用全局 == []，
# 引用裁决4教训：全局 ==[] 会把无关 split 的 V2/V4 小 n 组合噪声一并背上）。
# ---------------------------------------------------------------------------

def test_pipeline_grid_d2_d4_policies_e2e_v11_v12_green(tmp_path):
    from forgery_pipeline.config import load_config
    from forgery_pipeline.pipeline import run_pipeline
    from forgery_pipeline import manifest

    cfg = load_config("configs/pipeline.example.yaml")
    stages = dict(cfg.stages)
    stages["grid"] = True
    cfg = dataclasses.replace(
        cfg, out_dir=str(tmp_path / "run"), stages=stages, grid_per_op=4,
        scales=dataclasses.replace(cfg.scales, d0=16, d1_per_generator=1, d2=8, d3=4, d4=3))

    st = run_pipeline(cfg)
    rows = manifest.read_jsonl(tmp_path / "run" / "manifest.jsonl")
    # 非空真前提：确有扩散编辑行/masked 行/D4 行参与下面的判定域断言，不是空转。D4 行的
    # 存在性尤其重要——它是 PATCH 9 Wave2 Task5 派生修复（d4_explain.py 回填 operator/
    # op_params）的回归锚：D4 与源 D2 行共用同一张图、复制 io_chain（含 edit: 节点）但此前
    # 不回填 op_params，会被 V11 误判"扩散编辑行缺 nuisance 记录"。
    assert any(r.operator == "img2img" for r in rows)
    assert any(r.operator == "outpaint" for r in rows)
    assert any(r.task_type.value == "explainable" for r in rows)

    errs = check_all(rows, profile="run", testc_holdout="object_replacement",
                     nuisance_cell_floor=1)
    v11_v12 = [e for e in errs if e.startswith(("V11:", "V12:"))]
    assert not v11_v12, v11_v12

    # stats 两键存在且计数合理（真实数据下应非空——D2/grid 产出扩散编辑行与 masked 行）
    assert st["by_nuisance_cell"], "by_nuisance_cell 不应为空（D2/grid 均政策接线产 op_params）"
    assert st["by_area_bucket"], "by_area_bucket 不应为空（D2/grid outpaint 均产 mask_area_ratio）"

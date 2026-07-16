"""V1–V12 manifest 级别断言集（V1–V7：PATCH 7 收尾，spec
`docs/PATCHES_addendum_06_07_2026-07-15.md`；V8–V10：PATCH 9 Wave 1 split 防泄漏；
V11–V12：PATCH 9 Wave 2 nuisance 记录/面积分桶下限）。

范围说明（裁决B）：V8/V10/V11/V12 仅在 profile=="run" 时执行——probe 产物是受控仪器（故意
让同一底图横跨 holdout 生成器、让算子网格进 train，这正是 Gate 1/2 的设计），validator 不罚
仪器设计（且 probe 从未与主 run manifest 合并后过 check_all(profile="run")，故 V11/V12
在现有调用点上只约束 D1–D4/grid 的主链行，与 probe 无实际交互）；V9 由 holdout_generators
参数门控即可（probe 的 holdout 行本就标 test_b）。

设计约束：
- 本模块**不**反向 import `forgery_pipeline.manifest`（`manifest.stats()` 要 import 本模块的
  `nongen_chain`，若本模块也 import manifest 会成环）。因此只通过 duck-typing 读取传入对象的
  字段属性（`.io_chain`/`.is_fake`/`.split`/... ），不 import `forgery_pipeline.schema`。
- **V2 谱系适配**（相对 spec 字面的已记录偏差，写死在这里）：`nongen_chain` 归一时忽略首节点
  `decode` 并过滤 `edit:*`/`vae_rt:*`/`gen:*` 节点——D1 全生成行无源可解码（链头是
  `gen:{name}`），字面比较链头会结构性 FAIL；V2 的本意是「管线附加处理（分辨率/编码）不可
  预测 is_fake」，不是要求生成器链头字面相同，故做此归一。
- **V11 豁免键的实证修正**（相对 brief 字面的已记录偏差）：`d3_web.py` 的 manual-web-edit
  行实际写入 `generator_family="editing"`（非 brief 猜测的 "non_diffusion"/"manual"），
  故 V11 对 D3 的豁免改用 `generator_name=="manual-web-edit"` 命中，见 `_is_diffusion_edit_row`。
"""
from __future__ import annotations
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

_SAMPLE_KIND_ENUM = {"real", "real_vae_rt", "edited"}
_COMPOSITING_ENUM = {"none", "paste", "paste_feather"}
# masked（掩码引导）算子集合——V3（compositing 完备性）与 V12（mask_area_ratio 完备性/
# 面积分桶下限）共享同一判定域，故提取为模块级常量（PATCH 9 Wave2 Task5）。
_MASKED_OPS = {"inpaint", "outpaint", "object_replacement", "background_editing"}
_V4_SPLITS = ("train", "test_a", "test_f")
_V7_PAIR_GROUPS = {"compositing_pair", "nd_pair"}


def nongen_chain(io_chain) -> str:
    """io_chain 去生成段归一化：拆 '>' → 过滤 edit:/vae_rt:/gen: 节点 → 丢弃首个 decode 节点 → 重组。

    legacy 行整体视为不可拆分的谱系标记，原样返回 "legacy"；None（字段未填）返回 ""。
    """
    if io_chain is None:
        return ""
    if io_chain == "legacy":
        return "legacy"
    nodes = [n for n in io_chain.split(">")
             if not (n.startswith("edit:") or n.startswith("vae_rt:") or n.startswith("gen:"))]
    if nodes and nodes[0] == "decode":
        nodes = nodes[1:]
    return ">".join(nodes)


def _by_split(samples):
    groups = defaultdict(list)
    for s in samples:
        groups[s.split].append(s)
    return groups


def _postprocess_is_set(s) -> bool:
    """postprocess 非默认（即发生过退化）——V1 的退化行豁免判据。"""
    pp = s.postprocess
    if pp is None:
        return False
    return bool(pp.jpeg_quality != "none" or pp.resize != "none"
               or pp.blur != "none" or pp.noise != "none")


def check_v1(samples) -> list[str]:
    """V1 存储格式与分辨率唯一：split 内非豁免行的 image_path 后缀集合须恰为 {'.png'}。"""
    errs: list[str] = []
    for split, rows in _by_split(samples).items():
        by_suffix: dict[str, list[str]] = defaultdict(list)
        for s in rows:
            if s.io_chain == "legacy" or _postprocess_is_set(s):
                continue
            by_suffix[Path(s.image_path).suffix].append(s.image_id)
        present = set(by_suffix)
        if present and present != {".png"}:
            offenders = {suf: ids for suf, ids in by_suffix.items() if suf != ".png"}
            errs.append(f"V1: split={split!r} 存储格式不唯一（非 .png 后缀={offenders}）")
    return errs


def check_v2(samples) -> list[str]:
    """V2 real/fake 非生成链一致：split 内 real 与 fake 的 nongen_chain 集合须相等。

    legacy 行豁免（不计入任一集合）；某一类样本缺失的 split 跳过比较。
    """
    errs: list[str] = []
    for split, rows in _by_split(samples).items():
        real_chains, fake_chains = set(), set()
        for s in rows:
            if s.io_chain == "legacy":
                continue
            target = real_chains if s.is_fake == 0 else fake_chains
            target.add(nongen_chain(s.io_chain))
        if not real_chains or not fake_chains:
            continue
        if real_chains != fake_chains:
            sym = real_chains ^ fake_chains
            errs.append(f"V2: split={split!r} real/fake 非生成链不一致，差集={sym}"
                       f"（real={real_chains}, fake={fake_chains}）")
    return errs


def check_v3(samples) -> list[str]:
    """V3 masked 算子 compositing 完备性 + sample_kind/compositing 值域（T1 遗留校验折入）。"""
    errs: list[str] = []
    for s in samples:
        if s.sample_kind is not None and s.sample_kind not in _SAMPLE_KIND_ENUM:
            errs.append(f"V3: image_id={s.image_id} sample_kind 非法: {s.sample_kind!r}")
        if s.compositing is not None and s.compositing not in _COMPOSITING_ENUM:
            errs.append(f"V3: image_id={s.image_id} compositing 非法: {s.compositing!r}")
        if s.operator in _MASKED_OPS:
            if s.compositing is None:
                errs.append(f"V3: image_id={s.image_id} operator={s.operator!r} "
                           f"缺 compositing（须 ∈ {_COMPOSITING_ENUM}）")
            elif s.compositing not in _COMPOSITING_ENUM:
                pass  # 值域非法已由上面的通用值域检查报告，此处不重复
            elif s.compositing == "paste_feather" and s.feather_px is None:
                errs.append(f"V3: image_id={s.image_id} compositing=paste_feather 缺 feather_px")
    return errs


def check_v4(samples, profile: str = "auto", vae_rt_range=(0.05, 0.35),
             min_real: int = 10) -> list[str]:
    """V4 real_vae_rt 占比：train/test_a/test_f 中每个含 real 行的 split 须落在 vae_rt_range。

    触发条件：profile=="run"，或 profile=="auto" 且 manifest 中存在任一 real_vae_rt 行；
    否则跳过（不产生任何消息）。

    min_real 守卫：split 内 real 行数 < min_real 时跳过该 split 的配比断言（不产生消息）——
    小 n 时比值离散取值结构性落不进 band，配比断言只对达到统计规模的 split 有意义。
    """
    has_vae_rt = any(s.sample_kind == "real_vae_rt" for s in samples)
    if not (profile == "run" or (profile == "auto" and has_vae_rt)):
        return []
    lo, hi = vae_rt_range
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"real": 0, "real_vae_rt": 0})
    for s in samples:
        if s.split in _V4_SPLITS and s.sample_kind in ("real", "real_vae_rt"):
            counts[s.split][s.sample_kind] += 1
    errs: list[str] = []
    for split in _V4_SPLITS:
        c = counts.get(split)
        if not c or c["real"] == 0 or c["real"] < min_real:
            continue
        ratio = c["real_vae_rt"] / c["real"]
        if not (lo <= ratio <= hi):
            errs.append(f"V4: split={split!r} real_vae_rt 占比越界: {ratio:.4f} "
                       f"(real_vae_rt={c['real_vae_rt']}, real={c['real']}, "
                       f"期望区间=[{lo}, {hi}])")
    return errs


def check_v5(samples, profile: str = "auto") -> list[str]:
    """V5 run-profile legacy 禁令（spec 7.4："主 run 中不得出现 legacy"）：profile=="run" 时，
    任何 io_chain 缺失（None）或仍为 "legacy" 的行判为 FAIL。

    V5 的另一半语义（旧 manifest 经回填脚本后须过 V1–V4）不是运行时检查，由
    `test_v5_backfilled_legacy_manifest_passes_check_all` 这条测试本身保证——那条路径用的是
    profile="auto"，因此不会被这里的禁令拦下，向后兼容不受影响。

    多条违规行合并成一条消息（只列前 5 个 image_id + 总数），避免刷屏。
    """
    if profile != "run":
        return []
    offenders = [s.image_id for s in samples if s.io_chain in (None, "legacy")]
    if not offenders:
        return []
    head = ", ".join(offenders[:5])
    suffix = "..." if len(offenders) > 5 else ""
    return [f"V5: run profile 禁止 legacy/缺失 io_chain 行: {head}{suffix}"
            f"（共 {len(offenders)} 条）"]


def check_v6(samples) -> list[str]:
    """V6 instruct_edit 行 op_params 完备性：须为合法 JSON 且含 image_guidance_scale 键。"""
    errs: list[str] = []
    for s in samples:
        if s.operator != "instruct_edit":
            continue
        try:
            params = json.loads(s.op_params)
        except (TypeError, ValueError):
            errs.append(f"V6: image_id={s.image_id} op_params 非法 JSON: {s.op_params!r}")
            continue
        if not isinstance(params, dict):
            errs.append(f"V6: image_id={s.image_id} op_params 不是 JSON object: {s.op_params!r}")
        elif "image_guidance_scale" not in params:
            errs.append(f"V6: image_id={s.image_id} op_params 缺 image_guidance_scale 键: "
                       f"{s.op_params!r}")
    return errs


def check_v7(samples) -> list[str]:
    """V7 成对 probe 样本一致性：compositing_pair/nd_pair 按 pair_id 恰好成对且组内字段一致。"""
    errs: list[str] = []
    groups: dict = defaultdict(list)
    for s in samples:
        if s.probe_group in _V7_PAIR_GROUPS:
            groups[s.pair_id].append(s)
    for pid, rows in groups.items():
        if len(rows) != 2:
            errs.append(f"V7: pair_id={pid!r} 行数≠2（实得 {len(rows)}）")
            continue
        a, b = rows
        if a.seed != b.seed:
            errs.append(f"V7: pair_id={pid!r} 组内 seed 不一致: {a.seed} vs {b.seed}")
        if a.real_image_path != b.real_image_path:
            errs.append(f"V7: pair_id={pid!r} 组内 real_image_path 不一致: "
                       f"{a.real_image_path!r} vs {b.real_image_path!r}")
        if a.mask_path != b.mask_path:
            errs.append(f"V7: pair_id={pid!r} 组内 mask_path 不一致: "
                       f"{a.mask_path!r} vs {b.mask_path!r}")
        if a.probe_group != b.probe_group:
            errs.append(f"V7: pair_id={pid!r} 组内 probe_group 不一致: "
                       f"{a.probe_group!r} vs {b.probe_group!r}")
            continue
        if a.probe_group == "compositing_pair":
            if {a.compositing, b.compositing} != {"none", "paste_feather"}:
                errs.append(f"V7: pair_id={pid!r} compositing_pair 组内 compositing 应为"
                           f"{{'none','paste_feather'}} 各一，实得 "
                           f"{{{a.compositing!r}, {b.compositing!r}}}")
            if a.generator_name != b.generator_name:
                errs.append(f"V7: pair_id={pid!r} compositing_pair 组内 generator_name "
                           f"应相同: {a.generator_name!r} vs {b.generator_name!r}")
        else:  # nd_pair
            if a.generator_name == b.generator_name:
                errs.append(f"V7: pair_id={pid!r} nd_pair 组内 generator_name 应不同，"
                           f"均为 {a.generator_name!r}")
            if a.compositing != b.compositing:
                errs.append(f"V7: pair_id={pid!r} nd_pair 组内 compositing 应相同: "
                           f"{a.compositing!r} vs {b.compositing!r}")
    return errs


def check_v8(samples, profile: str = "auto") -> list[str]:
    """V8 底图组防泄漏（PATCH 9.3；裁决B：仅 profile=="run" 执行，见模块 docstring）：

    - 同 base_id 的行须落在同一 split。base_id 未设置的行不参与；postprocess_of 非空的
      退化行也排除在组一致性之外（裁决A）——其归属由下面的母行断言单独管辖。
    - postprocess 退化行须与母行同 split，**唯一豁免**：mother.split=="test_a" 且
      child.split=="test_e"。方法学依据（裁决A）：splitter 从 test_a 的退化 fake 中切出
      degradation 测试集 test_e 属既定设计——这是 eval→eval 的移动，母行与退化行都不在
      训练侧，不构成训练泄漏；母行在 train/val 时退化行必须同 split（豁免不适用）。
      按 image_id 建母行索引，母行缺失时跳过该行（不误报）。
    """
    if profile != "run":
        return []
    errs: list[str] = []
    groups: dict[str, set] = defaultdict(set)
    for s in samples:
        if s.base_id and not s.postprocess_of:
            groups[s.base_id].add(s.split)
    for base_id, splits in groups.items():
        if len(splits) > 1:
            errs.append(f"V8: base_id 组跨 split: {base_id} → {sorted(str(x) for x in splits)}")

    by_id = {s.image_id: s for s in samples}
    for s in samples:
        if not s.postprocess_of:
            continue
        mother = by_id.get(s.postprocess_of)
        if mother is None:
            continue
        if s.split == mother.split:
            continue
        if mother.split == "test_a" and s.split == "test_e":
            continue    # 裁决A豁免：test_e degradation carve-out（eval→eval，无训练泄漏）
        errs.append(f"V8: postprocess 行 split 与母行不一致: {s.image_id}")
    return errs


def check_v9(samples, holdout_generators=None) -> list[str]:
    """V9 cross-generator holdout 防泄漏：holdout_generators（generator_name 或
    generator_family 命中即算）不得出现在 train/val（须仅见于 test_b 等留出 split）。

    holdout_generators 为 None 或空时跳过（未配置留出生成器，向后兼容默认关闭）。
    """
    if not holdout_generators:
        return []
    hold = set(holdout_generators)
    errs: list[str] = []
    for s in samples:
        if s.split in ("train", "val") and (
            (s.generator_name in hold) or (s.generator_family in hold)
        ):
            errs.append(f"V9: holdout 生成器泄入 {s.split}: {s.image_id} "
                       f"({s.generator_name}/{s.generator_family})")
    return errs


def check_v10(samples, testc_holdout=None, profile: str = "auto") -> list[str]:
    """V10 Test-C holdout 算子防泄漏：operator == testc_holdout 的行结构性留给 test_c
    （PATCH 8.3 几何探针裁定，唯一 config 源见 configs/split.yaml），不得出现在 train/val。

    testc_holdout 为 None 时跳过（未配置，向后兼容默认关闭）；
    裁决B：仅 profile=="run" 执行——probe 的算子×生成器网格故意进 train，属仪器设计。
    """
    if profile != "run" or not testc_holdout:
        return []
    errs: list[str] = []
    for s in samples:
        if s.operator == testc_holdout and s.split in ("train", "val"):
            errs.append(f"V10: Test-C holdout 算子泄入 {s.split}: {s.image_id}")
    return errs


def _is_diffusion_edit_row(s) -> bool:
    """V11 扩散编辑行判定域（PATCH 9 Wave2 Task5 控制器裁决）：`is_fake==1` 且 `io_chain`
    含 `"edit:"` 节点。判定域刻意用 io_chain 而非 op_params 自指——否则缺记录的行会因为
    域定义本身依赖 op_params 而结构性逃出判定域，检查空转（见 wave2 计划"风险自查"）。

    豁免（不落入判定域）：
    - `generator_family == "non_diffusion"`：预留给未来 LaMa 等非扩散 inpainter（无
      cfg_scale/steps 概念）。
    - `io_chain == "legacy"`：旧谱系整体标记；结构上本就不含 `"edit:"` 子串，此处显式复述
      spec 原文豁免口径作为双重保险（不依赖"legacy 字面不含 edit:"这一事实性质）。
    - `generator_name == "manual-web-edit"`：D3 网页人工篡改行的豁免键。brief 原文猜测的
      豁免键是 `generator_family in {"non_diffusion", "manual"}`，但读 `d3_web.py` 源码
      核实其实际写入的是 `generator_family="editing"`（不是 "manual"），故改用
      `generator_name=="manual-web-edit"` 命中——D3 是人工合成的高对比色块贴片，不存在
      cfg/steps 概念，语义上仍应豁免。
    """
    if s.is_fake != 1:
        return False
    chain = s.io_chain
    if not chain or chain == "legacy" or "edit:" not in chain:
        return False
    if s.generator_family == "non_diffusion":
        return False
    if s.generator_name == "manual-web-edit":
        return False
    return True


def check_v11(samples, profile: str = "auto", nuisance_cell_floor: int = 0) -> list[str]:
    """V11 扩散编辑行 nuisance（cfg_scale/steps）记录完备性（run-profile only，裁决B同 V8/V10：
    probe 是受控仪器，不受罚；但 probe 从不与主 run manifest 合并过 check_all(profile="run")，
    故本检查在现有调用点上只约束 D1–D4/grid 的主链行）。

    判定域见 `_is_diffusion_edit_row`。域内每行的 op_params 须是可解析 JSON object 且同时
    含 "cfg_scale"/"steps" 键、且两者的值须能以 `:g` 数值格式渲染（非数值类型，如手工
    回填 manifest 里误存成字符串的 "7.5"，视同未合规记录，同样计入失败——而不是让格式化
    本身抛出未捕获异常砸穿 validate-manifest CLI）；None/空串/非法 JSON/非 dict/缺键，
    一律计入同一条聚合失败消息（只列前 5 个 image_id + 总数，沿用 V5 风格——不像 V6 按
    失败原因拆分多条消息，因为 B3 只关心"这行有没有全须全尾地记录 nuisance"，原因不重要）。

    nuisance_cell_floor > 0 时另检查覆盖率下限：按 split 分组，域内合规行按
    f"cfg{cfg_scale:g}/st{steps}" 归入单元格，**该 split 内实际出现过**的单元格计数
    < floor → FAIL。只检查出现过的单元格——(cfg, steps) 是笛卡尔积，某单元格在某 split
    从未出现，是该 split 结构性没有覆盖到该组合（例如 holdout 生成器只进 test_b），不是
    "记录不够"，不应报告为下限不足。
    """
    if profile != "run":
        return []
    offenders: list[str] = []
    cell_counts: dict = defaultdict(Counter)
    for s in samples:
        if not _is_diffusion_edit_row(s):
            continue
        params = None
        if s.op_params:
            try:
                parsed = json.loads(s.op_params)
            except (TypeError, ValueError):
                parsed = None
            if isinstance(parsed, dict):
                params = parsed
        if not params or "cfg_scale" not in params or "steps" not in params:
            offenders.append(s.image_id)
            continue
        try:
            cell = f"cfg{params['cfg_scale']:g}/st{params['steps']}"
        except (TypeError, ValueError):
            # 键存在但值非数值（如手工回填误存成字符串）——记录不合格，同缺键处理，
            # 不让格式化异常穿透（这里不是防御性冗余：已实测复现，见 report）。
            offenders.append(s.image_id)
            continue
        cell_counts[s.split][cell] += 1

    errs: list[str] = []
    if offenders:
        head = ", ".join(offenders[:5])
        suffix = "..." if len(offenders) > 5 else ""
        errs.append(f"V11: 扩散编辑行缺 nuisance 记录: {head}{suffix}（共 {len(offenders)} 条）")
    if nuisance_cell_floor > 0:
        for split, counts in cell_counts.items():
            for cell, n in counts.items():
                if n < nuisance_cell_floor:
                    errs.append(f"V11: split={split} cell={cell} 计数 {n} < {nuisance_cell_floor}")
    return errs


def check_v12(samples, profile: str = "auto", area_bucket_floor: int = 0,
              area_buckets=(0.05, 0.15, 0.35, 0.7)) -> list[str]:
    """V12 masked 算子行 mask_area_ratio 完备性 + 面积分桶下限（run-profile only，裁决B同
    V8/V10/V11）。

    判定域：`operator in _MASKED_OPS`（复用 V3 的同一集合常量）。域内每行 mask_area_ratio
    须非 None，否则计入同一条聚合失败消息（同 V11/V5 风格：前 5 个 image_id + 总数）。

    area_bucket_floor > 0 时另检查覆盖率下限：域内且 mask_area_ratio 非 None 的行按
    area_buckets 用 np.digitize 归桶（桶号 0..len(area_buckets)-1；np.digitize 对
    >= 最高边界的值返回 len(area_buckets)，该溢出桶不参与下限检查——与
    d2_local.py `_bucketed_pick` 的分层丢弃口径一致）。逐桶计数 < floor → FAIL，
    全 manifest 聚合（不分 split——D2 的面积分层本身是全局设计，非按 split 分层，故此处
    与 V11 的逐 split 检查口径不同，均按 spec 字面）。桶完全无命中（计数 0）同样 < floor，
    与"有命中但不够"同等报告——面积分层覆盖率下限的本意正是要能抓住"某一档面积完全没有
    样本"这种最严重的覆盖缺口，而非只在"该桶曾出现过"时才检查。
    """
    if profile != "run":
        return []
    offenders: list[str] = []
    n_buckets = len(area_buckets)
    bucket_counts: Counter = Counter()
    for s in samples:
        if s.operator not in _MASKED_OPS:
            continue
        if s.mask_area_ratio is None:
            offenders.append(s.image_id)
            continue
        bi = int(np.digitize(s.mask_area_ratio, area_buckets))
        if bi < n_buckets:
            bucket_counts[bi] += 1

    errs: list[str] = []
    if offenders:
        head = ", ".join(offenders[:5])
        suffix = "..." if len(offenders) > 5 else ""
        errs.append(f"V12: masked 算子行缺 mask_area_ratio: {head}{suffix}（共 {len(offenders)} 条）")
    if area_bucket_floor > 0:
        for bi in range(n_buckets):
            n = bucket_counts.get(bi, 0)
            if n < area_bucket_floor:
                errs.append(f"V12: 面积桶 {bi} 计数 {n} < {area_bucket_floor}")
    return errs


def check_all(samples, profile: str = "auto", vae_rt_range=(0.05, 0.35),
              min_real: int = 10, holdout_generators=None, testc_holdout=None,
              nuisance_cell_floor: int = 0, area_bucket_floor: int = 0,
              area_buckets=(0.05, 0.15, 0.35, 0.7)) -> list[str]:
    """跑全部 V1–V12（V5 只在 profile=="run" 时额外执行 run-profile legacy 禁令；
    向后兼容语义主要由「backfill 后过 check_all」这条测试本身保证，见 check_v5 docstring）。

    V8/V10/V11/V12 仅 profile=="run" 执行（裁决B，见模块 docstring）；V9/V10 另需
    holdout_generators/testc_holdout，V11/V12 另需 nuisance_cell_floor/area_bucket_floor
    （默认皆 0/None=跳过覆盖率下限检查，向后兼容所有既存调用点不受影响——新增参数纯尾附，
    不改变任何既有位置参数的顺序/语义）。

    `samples`：任意可迭代对象，元素只需 duck-typing 具备 Sample 的相应字段属性。
    返回空列表 = 全部通过；非空 = 每条以 "V{n}: " 为前缀的失败消息。
    """
    samples = list(samples)  # 下面逐检查各自完整遍历一遍；具体化以支持一次性迭代器
    errs: list[str] = []
    errs += check_v1(samples)
    errs += check_v2(samples)
    errs += check_v3(samples)
    errs += check_v4(samples, profile=profile, vae_rt_range=vae_rt_range, min_real=min_real)
    errs += check_v5(samples, profile=profile)
    errs += check_v6(samples)
    errs += check_v7(samples)
    errs += check_v8(samples, profile=profile)
    errs += check_v9(samples, holdout_generators=holdout_generators)
    errs += check_v10(samples, testc_holdout=testc_holdout, profile=profile)
    errs += check_v11(samples, profile=profile, nuisance_cell_floor=nuisance_cell_floor)
    errs += check_v12(samples, profile=profile, area_bucket_floor=area_bucket_floor,
                      area_buckets=area_buckets)
    return errs

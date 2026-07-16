"""V1–V10 manifest 级别断言集（V1–V7：PATCH 7 收尾，spec
`docs/PATCHES_addendum_06_07_2026-07-15.md`；V8–V10：PATCH 9 Wave 1 split 防泄漏）。

范围说明（裁决B）：V8/V10 仅在 profile=="run" 时执行——probe 产物是受控仪器（故意让同一
底图横跨 holdout 生成器、让算子网格进 train，这正是 Gate 1/2 的设计），validator 不罚
仪器设计；V9 由 holdout_generators 参数门控即可（probe 的 holdout 行本就标 test_b）。

设计约束：
- 本模块**不**反向 import `forgery_pipeline.manifest`（`manifest.stats()` 要 import 本模块的
  `nongen_chain`，若本模块也 import manifest 会成环）。因此只通过 duck-typing 读取传入对象的
  字段属性（`.io_chain`/`.is_fake`/`.split`/... ），不 import `forgery_pipeline.schema`。
- **V2 谱系适配**（相对 spec 字面的已记录偏差，写死在这里）：`nongen_chain` 归一时忽略首节点
  `decode` 并过滤 `edit:*`/`vae_rt:*`/`gen:*` 节点——D1 全生成行无源可解码（链头是
  `gen:{name}`），字面比较链头会结构性 FAIL；V2 的本意是「管线附加处理（分辨率/编码）不可
  预测 is_fake」，不是要求生成器链头字面相同，故做此归一。
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path

_SAMPLE_KIND_ENUM = {"real", "real_vae_rt", "edited"}
_COMPOSITING_ENUM = {"none", "paste", "paste_feather"}
_V3_OPERATOR_SCOPE = {"inpaint", "outpaint", "object_replacement", "background_editing"}
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
        if s.operator in _V3_OPERATOR_SCOPE:
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


def check_all(samples, profile: str = "auto", vae_rt_range=(0.05, 0.35),
              min_real: int = 10, holdout_generators=None, testc_holdout=None) -> list[str]:
    """跑全部 V1–V10（V5 只在 profile=="run" 时额外执行 run-profile legacy 禁令；
    向后兼容语义主要由「backfill 后过 check_all」这条测试本身保证，见 check_v5 docstring）。

    V8/V10 仅 profile=="run" 执行（裁决B，见模块 docstring）；V9/V10 另需
    holdout_generators/testc_holdout（默认 None=跳过，向后兼容所有既存调用点不受影响）。

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
    return errs

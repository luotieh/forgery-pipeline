"""V1–V7 manifest 级别断言集（PATCH 7 收尾，spec `docs/PATCHES_addendum_06_07_2026-07-15.md`）。

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


def check_v4(samples, profile: str = "auto", vae_rt_range=(0.05, 0.35)) -> list[str]:
    """V4 real_vae_rt 占比：train/test_a/test_f 中每个含 real 行的 split 须落在 vae_rt_range。

    触发条件：profile=="run"，或 profile=="auto" 且 manifest 中存在任一 real_vae_rt 行；
    否则跳过（不产生任何消息）。
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
        if not c or c["real"] == 0:
            continue
        ratio = c["real_vae_rt"] / c["real"]
        if not (lo <= ratio <= hi):
            errs.append(f"V4: split={split!r} real_vae_rt 占比越界: {ratio:.4f} "
                       f"(real_vae_rt={c['real_vae_rt']}, real={c['real']}, "
                       f"期望区间=[{lo}, {hi}])")
    return errs


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


def check_all(samples, profile: str = "auto", vae_rt_range=(0.05, 0.35)) -> list[str]:
    """跑全部 V1–V7（V5 不是运行时检查，是「backfill 后过 check_all」这条测试本身）。

    `samples`：任意可迭代对象，元素只需 duck-typing 具备 Sample 的相应字段属性。
    返回空列表 = 全部通过；非空 = 每条以 "V{n}: " 为前缀的失败消息。
    """
    samples = list(samples)  # 下面逐检查各自完整遍历一遍；具体化以支持一次性迭代器
    errs: list[str] = []
    errs += check_v1(samples)
    errs += check_v2(samples)
    errs += check_v3(samples)
    errs += check_v4(samples, profile=profile, vae_rt_range=vae_rt_range)
    errs += check_v6(samples)
    errs += check_v7(samples)
    return errs

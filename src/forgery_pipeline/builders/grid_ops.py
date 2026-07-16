"""D_grid 主 run 算子轴：img2img（连续强度）+ outpaint（边带宽度网格）
（PATCH 9 Wave2 Task3）。

D0-D4 主链没有 img2img/outpaint 两类操纵行——B1 矩阵要求它们作为训练可见轴。本模块
为每个底图产【本池】img2img spec 数行 img2img（全图重绘、无掩码，operator="img2img"）+
1 行 outpaint（边带掩码，operator="outpaint"，本池 inpainter 侧非空时），nuisance/
prompt/op_params 记录方式逐字复刻 d2_local.py 的 policies 分支（cfg/steps 网格、prompt
bank 确定性抽取、op_params 四键 JSON）；两轴的 hold/train 池分离机制同样镜像 d2_local
（PATCH 9 Wave2 T4 裁决3，行数公式见 build_grid docstring）。policies 是必填
PipelineConfig——grid 是全新算子轴，没有"政策接入前"的旧行为需要向后兼容，故不像
build_d2 那样支持 policies=None。
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from forgery_pipeline import image_io, ids, prompts
from forgery_pipeline.backends import registry
from forgery_pipeline.backends.mock import stable_hash
from forgery_pipeline.compositing import composite
from forgery_pipeline.config import GeneratorSpec, PipelineConfig
from forgery_pipeline.schema import Sample, TaskType

# GATE_DATA.md 约定：init_timestep = round(strength × 1000)（SDEdit 起始 timestep），
# 与 probe.py 的 _NUM_TRAIN_TIMESTEPS 同源同值。
_NUM_TRAIN_TIMESTEPS = 1000

_GRID_SEED_BASE = 10_000_000  # 与 D2 的 seed+attempts 密集区间不相交：防 leakage 规则3 的 (prompt,seed) 跨 split 假阳（grid 与 D2 共享 background prompt bank 节；B3 量级实测期望碰撞 0.1–0.2/run）


def _split_for(name, holdout) -> str:
    """同 probe.py 的同名函数：本地复制而非跨 builder 导入（避免模块间私有耦合）。pipeline 语境下 assign_splits 会整体重赋 split，此预设仅 standalone 使用时生效（与 probe.py 同约）。"""
    return "test_b" if name in set(holdout) else "train"


def _border_mask(h: int, w: int, b_frac: float) -> np.ndarray:
    """边带掩码：255=边带（outpaint 待重绘区）、0=中心保留区。像素级实现与语义均照抄
    probe.py 的模块私有函数 `_border`——该函数不对外导出，本地复制一份而非跨模块 import
    私有实现（Context 裁决）。"""
    m = np.full((h, w), 255, np.uint8)
    b = int(min(h, w) * b_frac)
    m[b:h - b, b:w - b] = 0
    return m


def build_grid(out_dir, bases: list[Sample], img2img_specs: list[GeneratorSpec],
               inpainter_specs: list[GeneratorSpec], policies: PipelineConfig,
               backend: str = "mock", seed: int = 0, holdout_generators=(),
               feather_px: int = 8, resolution_pool: list[Sample] | None = None) -> list[Sample]:
    """每 base × 每【本池】img2img spec 产一行 img2img + 每 base（本池 inpainter 池非空
    时）产一行 outpaint。

    行数公式（PATCH 9 Wave2 T4 裁决3 池分离后）：
        |train 池底图| × |train i2i 池| + |train 池底图| × [train inp 池非空]
      + |hold 池底图| × |hold i2i 池|  + |hold 池底图| × [hold inp 池非空]
    holdout_generators=() 时 hold 池恒空、全部底图落 train 池，退化为
    len(bases) × (len(img2img_specs) + 1)——与池分离前行为逐行一致（回归锚：
    test_grid_ops.py::test_build_grid_row_counts_and_field_conventions 的逐底图断言）。

    空 bases，或 img2img_specs 与 inpainter_specs 皆空时优雅返回 []（同 probe.py 的
    build_compositing_pairs 惯例：空输入静默返回空列表而非报错）。两条算子轴各自独立
    降级——本池 i2i 侧空时该 base 只产 outpaint 行，本池 inpainter 侧空时只产
    img2img 行（不回退混池，见下方池分离注释）。

    resolution_pool（PATCH 9 Wave2 9.2c 多分辨率组路由，可选，默认 None=不路由/现状）：
    D0 产出的**全部**分辨率行（不止 bases 所在的基准组）。img2img 行按 spec.name 在
    `policies.resolution_groups` 中反查所属分辨率组：命中且该 base 在该组存在同源行
    （按 base_id 关联）时，本行的生成输入换成该分辨率的真实图像（而非用基准组底图
    模拟）——real 侧该分辨率的 nongen_chain 已由 D0 产出，fake 侧也要有同分辨率的行，
    V2（split 内 real/fake 非生成链集合相等）才能在两个分辨率组之间同时成立（否则
    非基准分辨率组只有 real 行，恰恰成了"非生成链可预测 is_fake"的反例）。查无该分辨率
    的同源行、resolution_pool 未提供、或 policies.resolution_groups 未命中该 spec 名，
    一律退回基准组底图（现状，向后兼容）。

    **不变式（务必保持）**：无论生成输入用了哪个分辨率的图像，输出行的
    real_image_path/base_id 始终锚定 `base`（bases 列表里的基准组行），而非用于生成的
    那张分辨率兄弟行——这是维持 `split/grouping.origin_key`（real_image_path 或
    image_path 的 stem，只做一跳解析）与 base_id 组一致的关键：若这里改指向分辨率兄弟
    行的路径，会让该行与其兄弟组各自形成独立 origin-group，被 `assign_splits` 各自独立
    哈希，从而打破 V8（同 base_id 须同 split）。`test_build_grid_row_counts_and_field_
    conventions` 里 `real_image_path in base_paths` 正是这条不变式的既有回归锚，路由
    实现不得违反。outpaint 行本 wave 不做分辨率路由（只有 img2img 覆盖两个分辨率组即
    足以让 V2 成立，见上）；真实 SDXL 映射复用同一路由机制，GPU 侧另行接入。
    """
    out = Path(out_dir)
    samples: list[Sample] = []
    if not bases or (not img2img_specs and not inpainter_specs):
        return samples
    bank = prompts.load_bank(policies.prompt_bank)
    bver = prompts.bank_version(policies.prompt_bank)
    lo, hi = policies.strength_range

    # spec 名 → 分辨率组反查表（resolution_groups 为空时恒查无，下面路由分支全程 no-op）。
    name_to_resolution: dict[str, int] = {
        name: res for res, names in (policies.resolution_groups or {}).items() for name in names
    }
    # base_id → {分辨率: 该分辨率的 D0 行}：仅当调用方提供 resolution_pool 时建立。
    siblings_by_base: dict[str, dict[int, Sample]] = {}
    if resolution_pool:
        for row in resolution_pool:
            res = image_io.chain_resolution(row.io_chain)
            if res is not None and row.base_id:
                siblings_by_base.setdefault(row.base_id, {})[res] = row
    # 池分离（PATCH 9 Wave2 T4 裁决3，恢复 PATCH 6 不变式「每个 origin-group 只含一类
    # （holdout/非 holdout）生成器」；机制镜像 d2_local.py 的 pool_hold/pool_train）：
    # img2img 与 inpainter 两轴各自按 holdout_generators 名二分为 hold/train 池；每个底图
    # 按 okey 哈希整体选边（~20% 走 holdout 池，与 d2_local 同公式同盐——同一 origin-group
    # 两轴同池），选边后只从本侧池取生成器：holdout 池底图只产 holdout 行（整组随 splitter
    # 进 test_b），train 池底图只产非 holdout 行。任一侧某轴池空 → 该底图该轴的行优雅跳过，
    # **绝不回退混池**：混池正是泄漏源——holdout img2img 行把底图组拖进 test_b 时，同组的
    # 非 holdout outpaint 行会让同名生成器同时出现在 train（D2 侧同名池）与 test_b，
    # check_leakage 规则4 必炸（W2T3 递延的 img2img 守卫缺口，已实测复现，裁决3 到期修复；
    # W2T3 时代的「outpaint 排除 holdout + 全 holdout 回退全量池」旧方案随本裁决废除）。
    hold_names = set(holdout_generators)
    pool_hold_i2i = [s for s in img2img_specs if s.name in hold_names]
    pool_train_i2i = [s for s in img2img_specs if s.name not in hold_names]
    pool_hold_inp = [i for i in inpainter_specs if i.name in hold_names]
    pool_train_inp = [i for i in inpainter_specs if i.name not in hold_names]

    for bi, base in enumerate(bases):
        img = image_io.load_image(out / base.image_path)
        h, w = img.shape[:2]
        # 同 d2_local.py：~20% 底图走 holdout 池（任一轴存在 holdout spec 才启用），
        # 同一 origin-group 的生成器全部同池。
        okey = base.real_image_path or base.image_path
        use_hold = bool(pool_hold_i2i or pool_hold_inp) and (stable_hash(okey) % 5 == 0)
        i2i_pool = pool_hold_i2i if use_hold else pool_train_i2i
        inp_pool = pool_hold_inp if use_hold else pool_train_inp

        for spec in i2i_pool:
            # 分辨率组路由（见上方 docstring）：spec 名命中某非基准分辨率组、且该 base
            # 在该组存在同源行时，生成输入换成该行图像；否则退回基准组 img/h（现状）。
            gen_img, gen_h = img, h
            target_res = name_to_resolution.get(spec.name)
            if target_res is not None and target_res != h:
                sib = siblings_by_base.get(base.base_id or base.image_id, {}).get(target_res)
                if sib is not None:
                    gen_img = image_io.load_image(out / sib.image_path)
                    gen_h = target_res

            iid = ids.make_image_id("grid_i2i", f"{base.image_id}-{spec.name}")
            s = round(lo + (hi - lo) * ((stable_hash(iid + "s") % 10000) / 10000.0), 4)
            cfg_v = policies.nuisance_cfg_grid[
                stable_hash(iid + "cfg") % len(policies.nuisance_cfg_grid)]
            st_v = policies.nuisance_steps_grid[
                stable_hash(iid + "st") % len(policies.nuisance_steps_grid)]
            prompt_text = prompts.pick_prompt(bank, "img2img", iid)
            sd = seed + bi * 1000 + stable_hash(spec.name) % 500 + _GRID_SEED_BASE
            gen = registry.get_img2img(backend, spec.name, spec.family)
            fake, meta = gen.img2img(gen_img, prompt_text, s,
                                     {"seed": sd, "cfg_scale": cfg_v, "steps": st_v})
            st_final = float(meta.get("strength", s))
            img_rel = f"Grid_ops/{iid}.png"
            image_io.save_canonical(fake, out / img_rel)
            samples.append(Sample(
                image_id=iid, image_path=img_rel, real_image_path=base.image_path,
                is_fake=1, task_type=TaskType.whole_image_detection,
                manipulation_level1="whole_generated", manipulation_level2="diffusion",
                manipulation_level4=spec.name,
                generator_name=spec.name, generator_family=spec.family,
                operator="img2img", strength=st_final,
                init_timestep=int(round(st_final * _NUM_TRAIN_TIMESTEPS)),
                seed=sd, split=_split_for(spec.name, holdout_generators),
                source_dataset=base.source_dataset,
                compositing="none", sample_kind="edited", base_id=base.image_id,
                prompt=prompt_text,
                op_params=json.dumps(
                    {"cfg_scale": cfg_v, "steps": st_v, "prompt": prompt_text,
                     "prompt_bank_version": bver}, sort_keys=True, ensure_ascii=False),
                io_chain=image_io.chain("decode", f"rs{gen_h}", f"edit:{spec.name}", "png"),
            ))

        if inp_pool:   # 本池 inpainter 侧空 → 该底图不产 outpaint 行（不回退混池，见上）
            inp = inp_pool[bi % len(inp_pool)]
            fracs = policies.outpaint_border_fracs
            b_frac = fracs[stable_hash(base.image_id + "bf") % len(fracs)]
            mask = _border_mask(h, w, b_frac)
            iid = ids.make_image_id("grid_outpaint", f"{base.image_id}-{inp.name}")
            cfg_v = policies.nuisance_cfg_grid[
                stable_hash(iid + "cfg") % len(policies.nuisance_cfg_grid)]
            st_v = policies.nuisance_steps_grid[
                stable_hash(iid + "st") % len(policies.nuisance_steps_grid)]
            prompt_text = prompts.pick_prompt(bank, "background", iid)
            # 与本 base 的 img2img seed 命名空间区分（同 probe.py build_probe_operator 的
            # f"{op}-{name}" 加盐惯例），避免同名生成器跨算子池撞种子。
            sd2 = seed + bi * 1000 + stable_hash(f"outpaint-{inp.name}") % 500 + _GRID_SEED_BASE
            painter = registry.get_inpainter(backend, inp.name, inp.family)
            fake, _ = painter.inpaint(img, mask, prompt_text,
                                      {"seed": sd2, "cfg_scale": cfg_v, "steps": st_v})
            mode = "paste_feather" if stable_hash(iid + "comp") % 2 else "none"
            fake = composite(img, fake, (mask > 127).astype(np.float32), mode,
                             feather_px=feather_px)
            img_rel = f"Grid_ops/{iid}.png"
            mask_rel = f"Grid_ops/masks/{iid}.png"
            image_io.save_canonical(fake, out / img_rel)
            image_io.save_mask(mask, out / mask_rel)
            samples.append(Sample(
                image_id=iid, image_path=img_rel, real_image_path=base.image_path,
                mask_path=mask_rel, is_fake=1, task_type=TaskType.localization,
                manipulation_level1="partial_manipulated", manipulation_level2="AIGC-editing",
                manipulation_level3="image_guided_editing", manipulation_level4=inp.name,
                generator_name=inp.name, generator_family=inp.family,
                operator="outpaint", mask_source="border",
                mask_area_ratio=round(float((mask > 127).mean()), 4),
                seed=sd2, split=_split_for(inp.name, holdout_generators),
                source_dataset=base.source_dataset,
                compositing=mode, feather_px=(feather_px if mode == "paste_feather" else None),
                sample_kind="edited", base_id=base.image_id, prompt=prompt_text,
                op_params=json.dumps(
                    {"cfg_scale": cfg_v, "steps": st_v, "prompt": prompt_text,
                     "prompt_bank_version": bver}, sort_keys=True, ensure_ascii=False),
                io_chain=image_io.chain("decode", f"rs{h}", f"edit:{inp.name}", "png"),
            ))
    return samples

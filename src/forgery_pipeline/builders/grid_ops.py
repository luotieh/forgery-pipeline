"""D_grid 主 run 算子轴：img2img（连续强度）+ outpaint（边带宽度网格）
（PATCH 9 Wave2 Task3）。

D0-D4 主链没有 img2img/outpaint 两类操纵行——B1 矩阵要求它们作为训练可见轴。本模块
为每个底图产 len(img2img_specs) 行 img2img（全图重绘、无掩码，operator="img2img"）+
1 行 outpaint（边带掩码，operator="outpaint"），nuisance/prompt/op_params 记录方式
逐字复刻 d2_local.py 的 policies 分支（cfg/steps 网格、prompt bank 确定性抽取、
op_params 四键 JSON）。policies 是必填 PipelineConfig——grid 是全新算子轴，没有"政策
接入前"的旧行为需要向后兼容，故不像 build_d2 那样支持 policies=None。
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


def _split_for(name, holdout) -> str:
    """同 probe.py 的同名函数：本地复制而非跨 builder 导入（避免模块间私有耦合）。"""
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
               feather_px: int = 8) -> list[Sample]:
    """每 base × 每 img2img spec 产一行 img2img + 每 base 产一行 outpaint。

    空 bases，或 img2img_specs 与 inpainter_specs 皆空时优雅返回 []（同 probe.py 的
    build_compositing_pairs 惯例：空输入静默返回空列表而非报错）。两条算子轴各自独立
    降级——img2img_specs 空时该 base 只产 outpaint 行，inpainter_specs 空时只产
    img2img 行。
    """
    out = Path(out_dir)
    samples: list[Sample] = []
    if not bases or (not img2img_specs and not inpainter_specs):
        return samples
    bank = prompts.load_bank(policies.prompt_bank)
    bver = prompts.bank_version(policies.prompt_bank)
    lo, hi = policies.strength_range
    # outpaint 生成器池排除 holdout（PATCH 6 不变式「每个 origin-group 只含一类
    # holdout/非 holdout 生成器」）：img2img 行对每个 base 无条件套用全部 img2img_specs
    # （非 holdout，见下），若 outpaint 按裸 `bi % len` 轮转到 holdout 生成器，该 base 的
    # origin-group 会被 splitter 整组划入 test_b，拖着同 base 的非 holdout img2img
    # 生成器名一起进 test_b——一旦别的 base 上同一生成器名落在 train，
    # check_leakage 规则4（cross-generator 生成器不得同时出现在 train 与 test_b）即炸
    # （已用 configs/pipeline.example.yaml + configs/split.yaml 实测复现）。全部落 holdout
    # 时无可选生成器，退回原始全量池（同 d2_local.py 的 pool_train fallback 惯例）。
    hold = {i.name for i in inpainter_specs if i.name in set(holdout_generators)}
    outpaint_pool = [i for i in inpainter_specs if i.name not in hold] or inpainter_specs

    for bi, base in enumerate(bases):
        img = image_io.load_image(out / base.image_path)
        h, w = img.shape[:2]

        for spec in img2img_specs:
            iid = ids.make_image_id("grid_i2i", f"{base.image_id}-{spec.name}")
            s = round(lo + (hi - lo) * ((stable_hash(iid + "s") % 10000) / 10000.0), 4)
            cfg_v = policies.nuisance_cfg_grid[
                stable_hash(iid + "cfg") % len(policies.nuisance_cfg_grid)]
            st_v = policies.nuisance_steps_grid[
                stable_hash(iid + "st") % len(policies.nuisance_steps_grid)]
            prompt_text = prompts.pick_prompt(bank, "img2img", iid)
            sd = seed + bi * 1000 + stable_hash(spec.name) % 500
            gen = registry.get_img2img(backend, spec.name, spec.family)
            fake, meta = gen.img2img(img, prompt_text, s,
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
                io_chain=image_io.chain("decode", f"rs{h}", f"edit:{spec.name}", "png"),
            ))

        if inpainter_specs:
            inp = outpaint_pool[bi % len(outpaint_pool)]
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
            sd2 = seed + bi * 1000 + stable_hash(f"outpaint-{inp.name}") % 500
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

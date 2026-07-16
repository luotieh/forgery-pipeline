"""D2 局部 AIGC 篡改：mask → prompt → inpaint（报告 §6，借鉴 GIM）。"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from forgery_pipeline import image_io, ids, prompts
from forgery_pipeline.backends import registry
from forgery_pipeline.backends.mock import stable_hash
from forgery_pipeline.compositing import composite
from forgery_pipeline.config import GeneratorSpec, PipelineConfig
from forgery_pipeline.masks.candidates import filter_and_sample, area_ratio
from forgery_pipeline.masks import morphology
from forgery_pipeline.qc.mask_qc import check_mask
from forgery_pipeline.qc.quality_score import (
    qes_score, route_from_score, bucket_from_score, area_validity)
from forgery_pipeline.schema import Sample, TaskType

# (篡改类型, level3, 编辑 prompt 模板)；level3 取自 LEVEL3 合法值
MANIP_TYPES = [
    ("object_insertion", "mask_guided_inpainting",
     "Insert a new realistic object into the masked region."),
    ("object_replacement", "object_replacement",
     "Replace the object in the masked region with a different realistic object."),
    ("object_removal", "object_removal",
     "Remove the object in the masked region and fill the background naturally."),
    ("attribute_editing", "text_guided_editing",
     "Change the color or attribute of the object in the masked region."),
    ("background_editing", "image_guided_editing",
     "Repaint the background within the masked region."),
    ("text_editing", "text_editing",
     "Modify the text content within the masked region."),
    ("face_editing", "face_swap",
     "Edit the face in the masked region (expression/glasses/hair)."),
]

# operator 映射（PATCH 9 Wave2 预裁决①）：operator 是粗轴口径——仅 object_replacement/
# background_editing 两类操纵与 operator 同名；其余五类（object_insertion/object_removal/
# attribute_editing/text_editing/face_editing）统一归并为 "inpaint"。细粒度语义不因归并
# 丢失，仍完整保留在 manipulation_level3/level4 上（outpaint 不在 D2，归 grid builder）。
_OP_MAP = {"object_replacement": "object_replacement", "background_editing": "background_editing"}

# prompt bank 的 kind 轴与 _OP_MAP 不是同一分类：object_insertion 的 operator 虽归并进
# "inpaint"，但语义仍是"放物体"，prompt 要从 object 节选，而非通用 inpaint 节。
_PROMPT_KIND_MAP = {"object_replacement": "object", "object_insertion": "object",
                     "background_editing": "background"}


def _bucketed_pick(valid: list[tuple], area_buckets: list[float], salt: str) -> np.ndarray:
    """面积桶分层选 mask 候选（PATCH 9 Wave2 9.2b，仅 policies 非 None 时调用）。

    按 area_buckets 边界用 np.digitize 给候选归桶（沿用 filter_and_sample 已算好的
    ratio，不重复计算）：桶数 = len(area_buckets)，比值落在最高边界及以上视为越界，
    丢弃不参与分层（维持既有 QC 口径，不因分层放宽面积上限）。目标桶由
    stable_hash(salt+"bkt") 确定性选出；目标桶为空则桶号 +1 顺延（模桶数），保证 n
    足够时各桶都被采到。桶内取第一个候选（保持与旧版"取第一个"同侧的确定性）。
    """
    n_buckets = len(area_buckets)
    buckets: dict[int, list[tuple]] = {}
    for cand in valid:
        bi = int(np.digitize(cand[1], area_buckets))
        if bi < n_buckets:
            buckets.setdefault(bi, []).append(cand)
    target = stable_hash(salt + "bkt") % n_buckets
    for _ in range(n_buckets):
        bucket = buckets.get(target)
        if bucket:
            return bucket[0][0]
        target = (target + 1) % n_buckets
    # 兜底：因 valid 的 ratio 上限恒为 0.50（filter_and_sample），默认最高桶界 0.7 下
    # 不会触发；仅在自定义 area_buckets 令全部候选越界丢弃时退回旧策略，保证恒有可用 mask。
    return valid[0][0]


def build_d2(out_dir, base_samples: list[Sample], n: int,
             inpainters: list[GeneratorSpec], backend: str = "mock",
             seed: int = 0, holdout_inpainters=(), feather_px: int = 8,
             policies: PipelineConfig | None = None) -> list[Sample]:
    out_dir = Path(out_dir)
    seg = registry.get_segmenter(backend, seed=seed)
    # 按底图把 inpainter 划入互斥池：保证每个 origin-group 只用一类生成器，
    # 避免 splitter 整组判定 test_b 时把非 holdout 生成器混入 test_b（PATCH 6）。
    hold = {i.name for i in inpainters if i.name in set(holdout_inpainters)}
    pool_hold = [i for i in inpainters if i.name in hold]
    pool_train = [i for i in inpainters if i.name not in hold] or inpainters
    # policies=None 时 bank/bver 保持 None 且下面各分支全程不触碰——policies 缺省的调用
    # （d3/d4/probe 等既存调用点）行为不变（PATCH 9 Wave2 Task2 回归护栏）。
    bank = bver = None
    if policies is not None:
        bank = prompts.load_bank(policies.prompt_bank)
        bver = prompts.bank_version(policies.prompt_bank)
    samples: list[Sample] = []
    attempts = 0
    max_attempts = max(n * 8, 8)
    while len(samples) < n and base_samples and attempts < max_attempts:
        base = base_samples[attempts % len(base_samples)]
        attempts += 1
        img = image_io.load_image(out_dir / base.image_path)
        valid = filter_and_sample(seg.propose_masks(img, 6))
        if not valid:
            continue
        if policies is not None:
            # 面积桶分层（9.2b）：此刻 mtype/iid 还没定下来（要等下面选完生成器池才知道
            # seed 拼出的 iid），用 base.image_id+attempts 这个此刻已确定性可得的 salt
            # 代替——同样是"每次尝试独立、跨 seed 稳定"的确定性分层键。
            cand0 = _bucketed_pick(valid, policies.area_buckets, f"{base.image_id}-{attempts}")
        else:
            cand0 = valid[len(samples) % len(valid)][0]
        mask = morphology.make_irregular(cand0, seed=seed + attempts)
        ok, _ = check_mask(mask)
        if not ok:
            continue
        ratio = area_ratio(mask)
        # QES 质量评分（PATCH 7）；mock 用占位置信度，真实后端换实测信号
        score = qes_score(
            confidence=0.9, boundary_sharpness=0.8,
            mask_consistency=1.0 if 0.01 <= ratio <= 0.50 else 0.5,
            semantic_consistency=0.8, area_validity=area_validity(ratio),
        )
        if route_from_score(score) == "reject":
            continue
        mtype, level3, tmpl = MANIP_TYPES[len(samples) % len(MANIP_TYPES)]
        # 按底图选池（~20% 底图走 holdout 池），同一 origin-group 生成器同池
        okey = base.real_image_path or base.image_path
        use_hold = bool(pool_hold) and (stable_hash(okey) % 5 == 0)
        pool = pool_hold if use_hold else pool_train
        inp = pool[len(samples) % len(pool)]
        painter = registry.get_inpainter(backend, inp.name, inp.family)
        s = seed + attempts
        iid = ids.make_image_id("local_edit", f"{base.image_id}-{mtype}-{s}")
        if policies is not None:
            kind = _PROMPT_KIND_MAP.get(mtype, "inpaint")
            cfg_v = policies.nuisance_cfg_grid[stable_hash(iid + "cfg") % len(policies.nuisance_cfg_grid)]
            st_v = policies.nuisance_steps_grid[stable_hash(iid + "st") % len(policies.nuisance_steps_grid)]
            prompt_text = prompts.pick_prompt(bank, kind, iid)
            fake, _ = painter.inpaint(img, mask, prompt_text,
                                      {"seed": s, "cfg_scale": cfg_v, "steps": st_v})
        else:
            fake, _ = painter.inpaint(img, mask, tmpl, {"seed": s})
        # PATCH 7.3：回贴显式化——50/50 决定是否羽化回贴到原图，而非隐式整图直出
        mode = "paste_feather" if stable_hash(iid + "comp") % 2 else "none"
        fake = composite(img, fake, (mask > 127).astype(np.float32), mode, feather_px=feather_px)
        img_rel = f"D2_local_aigc_edit/{iid}.png"
        mask_rel = f"D2_local_aigc_edit/masks/{iid}.png"
        image_io.save_canonical(fake, out_dir / img_rel)
        image_io.save_mask(mask, out_dir / mask_rel)
        samples.append(Sample(
            image_id=iid, image_path=img_rel,
            real_image_path=base.image_path, mask_path=mask_rel, is_fake=1,
            task_type=TaskType.localization,
            manipulation_level1="partial_manipulated",
            manipulation_level2="AIGC-editing",
            manipulation_level3=level3, manipulation_level4=inp.name,
            generator_name=inp.name, generator_family=inp.family,
            mask_source="SAM",
            mask_area_ratio=(round(ratio, 4) if policies is not None else None),
            prompt=(prompt_text if policies is not None else tmpl), seed=s,
            quality_score=round(score, 4), quality_bucket=bucket_from_score(score),
            source_dataset=base.source_dataset,
            compositing=mode, feather_px=(feather_px if mode == "paste_feather" else None),
            sample_kind="edited", base_id=base.image_id,
            io_chain=image_io.chain("decode", f"rs{img.shape[0]}", f"edit:{inp.name}", "png"),
            operator=(_OP_MAP.get(mtype, "inpaint") if policies is not None else None),
            op_params=(json.dumps(
                {"cfg_scale": cfg_v, "steps": st_v, "prompt": prompt_text,
                 "prompt_bank_version": bver}, sort_keys=True, ensure_ascii=False)
                if policies is not None else None),
        ))
    return samples

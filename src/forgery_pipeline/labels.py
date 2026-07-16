"""HiFi-Net 风格层级标签体系（报告 §9）。"""
from __future__ import annotations
from typing import Optional

LEVEL1 = {"whole_generated", "partial_manipulated"}

LEVEL2 = {
    "diffusion", "GAN", "autoregressive", "Photoshop-editing", "DeepFake",
    "AIGC-editing", "copy-move", "splicing", "removal",
}

LEVEL3 = {
    "conditional_generation", "unconditional_generation", "text_guided_editing",
    "image_guided_editing", "mask_guided_inpainting", "object_replacement",
    "object_removal", "face_swap", "text_editing",
}

# whole_generated 不应配局部编辑类 level2
_WHOLE_ONLY_L2 = {"diffusion", "GAN", "autoregressive"}

# 多任务 loss 字段（报告 §9，仅文档化，训练时使用）
LOSS_TERMS = [
    "detection_loss", "localization_loss", "manipulation_type_loss",
    "generator_family_loss", "optional_explanation_loss",
]


def validate_labels(is_fake: int, task_type: str, mask_path: Optional[str],
                    l1: Optional[str], l2: Optional[str],
                    l3: Optional[str]) -> list[str]:
    """返回标签一致性错误列表，空列表表示通过。"""
    errs: list[str] = []
    if is_fake == 0:
        if any([l1, l2, l3]):
            errs.append("真实图（is_fake=0）不应带 manipulation_level 标签")
        if task_type != "real_pristine":
            errs.append("真实图 task_type 必须为 real_pristine")
        return errs

    # is_fake == 1
    if l1 not in LEVEL1:
        errs.append(f"level1 非法: {l1!r}")
    if l2 is not None and l2 not in LEVEL2:
        errs.append(f"level2 非法: {l2!r}")
    if l3 is not None and l3 not in LEVEL3:
        errs.append(f"level3 非法: {l3!r}")

    if l1 == "partial_manipulated":
        if not mask_path:
            errs.append("partial_manipulated 必须提供 mask_path")
        if task_type not in ("localization", "explainable"):
            errs.append("partial_manipulated 的 task_type 应为 localization 或 explainable")
    elif l1 == "whole_generated":
        if l2 is not None and l2 not in _WHOLE_ONLY_L2:
            errs.append(f"whole_generated 不应配 level2={l2!r}")
    return errs


# 显式编辑算子词表（对齐闸门 {img2img, inpaint, outpaint, replace, background}）
# instruct_edit：InstructPix2Pix 系指令编辑算子（PATCH 9 addendum §8.2 新增枚举，
# 校验口径见 validate.py V6：op_params 须含 image_guidance_scale）。
EDIT_OPERATORS = {
    "img2img", "inpaint", "outpaint",
    "object_insertion", "object_replacement", "object_removal",
    "background_editing", "attribute_editing", "text_editing", "face_editing",
    "instruct_edit",
}


def validate_operator(op: Optional[str]) -> list[str]:
    if op is None or op in EDIT_OPERATORS:
        return []
    return [f"operator 非法: {op!r}"]

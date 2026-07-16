"""Manifest 数据契约：每行一个 Sample（对应报告 §13）。"""
from __future__ import annotations
from enum import Enum
from typing import Literal, Optional, Union
from pydantic import BaseModel, Field, field_validator, model_validator
from forgery_pipeline.labels import validate_labels, validate_operator


class TaskType(str, Enum):
    real_pristine = "real_pristine"
    whole_image_detection = "whole_image_detection"
    localization = "localization"
    explainable = "explainable"


class Postprocess(BaseModel):
    jpeg_quality: Union[int, Literal["none"]] = "none"
    resize: str = "none"
    blur: str = "none"
    noise: str = "none"


class Explanation(BaseModel):
    location_description: str
    visual_artifact_description: str
    semantic_reasoning: str
    forensic_conclusion: str


class Sample(BaseModel):
    image_id: str
    image_path: str
    real_image_path: Optional[str] = None
    mask_path: Optional[str] = None
    is_fake: int
    task_type: TaskType
    manipulation_level1: Optional[str] = None
    manipulation_level2: Optional[str] = None
    manipulation_level3: Optional[str] = None
    manipulation_level4: Optional[str] = None
    source_dataset: Optional[str] = None
    generator_name: Optional[str] = None
    generator_family: Optional[str] = None
    prompt: Optional[str] = None
    negative_prompt: Optional[str] = None
    seed: Optional[int] = None
    sampler: Optional[str] = None
    steps: Optional[int] = None
    cfg_scale: Optional[float] = None
    mask_source: Optional[str] = None
    mask_area_ratio: Optional[float] = None
    postprocess: Postprocess = Field(default_factory=Postprocess)
    quality_score: Optional[float] = None
    quality_bucket: Optional[str] = None
    split: Optional[str] = None
    license: Optional[str] = None
    explanation: Optional[Explanation] = None
    # 新增：编辑算子参数（Gate 1/2 与算子逆估计所需）
    strength: Optional[float] = None        # img2img/SDEdit 去噪强度 ∈ [0,1]，≈ 起始噪声级 t0/T
    init_timestep: Optional[int] = None     # SDEdit 起始 timestep（可选，便于直接读 t0）
    operator: Optional[str] = None          # 显式编辑算子（对齐闸门口径），取值见 labels.EDIT_OPERATORS
    op_params: Optional[str] = None         # 算子参数容器（JSON string，如 cfg_scale/steps；PATCH 8.2 约定）
    io_chain: Optional[str] = None          # 逐节点处理链（PATCH 7.1），如 decode>rs512>edit:sd15_inpaint>png；旧行=legacy
    sample_kind: Optional[str] = None       # real / real_vae_rt / edited（PATCH 7.2）
    compositing: Optional[str] = None       # none / paste / paste_feather（PATCH 7.3，masked 算子必填）
    feather_px: Optional[int] = None        # paste_feather 羽化 σ（像素）
    probe_group: Optional[str] = None       # 成对 probe 组名（compositing_pair / nd_pair，PATCH 7.3/8.1）
    pair_id: Optional[str] = None           # 成对样本回链 id
    base_id: Optional[str] = None  # 底图组键（V8 split 互斥用；D0=自身 image_id，衍生行=底图 image_id）
    postprocess_of: Optional[str] = None    # 退化样本回链原始 fake 的 image_id；原图为 None

    @field_validator("is_fake")
    @classmethod
    def _check_is_fake(cls, v: int) -> int:
        if v not in (0, 1):
            raise ValueError("is_fake 必须是 0 或 1")
        return v

    @field_validator("mask_area_ratio", "quality_score", "strength")
    @classmethod
    def _check_unit_interval(cls, v):
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError("取值必须落在 [0, 1]")
        return v

    @model_validator(mode="after")
    def _check_label_consistency(self):
        errs = validate_labels(
            self.is_fake, self.task_type.value, self.mask_path,
            self.manipulation_level1, self.manipulation_level2,
            self.manipulation_level3,
        )
        errs += validate_operator(self.operator)
        if errs:
            raise ValueError("; ".join(errs))
        return self

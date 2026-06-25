"""真实分割适配器骨架（SAM/Grounded-SAM）。需 `pip install .[sam]` 与权重。"""
from __future__ import annotations
from forgery_pipeline.backends import base


class SAMSegmenter(base.Segmenter):
    def __init__(self, checkpoint: str, device: str = "cuda"):
        try:
            from segment_anything import sam_model_registry  # noqa: F401
        except ImportError as e:
            raise NotImplementedError(
                "未安装 segment-anything：`pip install .[sam]`。") from e
        raise NotImplementedError("参考骨架：加载 SAM 并实现 propose_masks()。")

    def propose_masks(self, image, k):
        raise NotImplementedError

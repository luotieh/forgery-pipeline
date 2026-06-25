"""Backend 抽象接口（模型契约）。所有重型 ML 阶段经此解耦。"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Iterator, Optional
import numpy as np
from forgery_pipeline.schema import Explanation

# 类型约定：图像 (H,W,3) uint8 RGB；掩码 (H,W) uint8 取值 {0,255}
Image = np.ndarray
Mask = np.ndarray


class ImageSource(ABC):
    """真实图像源（D0/D3 底图来源）。"""
    @abstractmethod
    def iter_images(self, n: int) -> Iterator[tuple[Image, dict]]:
        ...


class WholeImageGenerator(ABC):
    """整图生成器（D1）。"""
    @abstractmethod
    def generate(self, prompt: str, params: dict) -> tuple[Image, dict]:
        ...


class Inpainter(ABC):
    """局部重绘模型（D2）。"""
    @abstractmethod
    def inpaint(self, image: Image, mask: Mask, prompt: str,
                params: dict) -> tuple[Image, dict]:
        ...


class Segmenter(ABC):
    """分割/候选 mask 生成（D2 候选、D3 细化）。"""
    @abstractmethod
    def propose_masks(self, image: Image, k: int) -> list[Mask]:
        ...


class Explainer(ABC):
    """MLLM 解释生成（D4）。"""
    @abstractmethod
    def explain(self, image: Image, mask: Optional[Mask],
                context: dict) -> Explanation:
        ...

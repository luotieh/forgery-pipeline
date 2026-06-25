"""确定性 mock backend：让全流程在 CPU 上跑通且可复现。"""
from __future__ import annotations
import hashlib
from typing import Iterator, Optional
import cv2
import numpy as np
from forgery_pipeline.backends import base
from forgery_pipeline.schema import Explanation


def stable_hash(s: str) -> int:
    """跨进程稳定的字符串散列（不可用内置 hash）。"""
    return int.from_bytes(hashlib.sha1(s.encode("utf-8")).digest()[:4], "big")


def synth_image(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    """每个种子各异的低频平面背景 + 若干随机色块的合成图。

    低频部分按种子随机生成（保证不同种子的图像在 pHash 上可区分），
    高频色块提供局部细节。
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    img = np.zeros((h, w, 3), dtype=np.float64)
    for c in range(3):
        ax = float(rng.choice([-1.0, 1.0])) * float(rng.uniform(0.4, 1.0))
        ay = float(rng.choice([-1.0, 1.0])) * float(rng.uniform(0.4, 1.0))
        off = float(rng.uniform(0, 255))
        img[..., c] = off + ax * (xx / max(w, 1) * 255) + ay * (yy / max(h, 1) * 255)
    img = np.clip(img, 0, 255).astype(np.uint8)
    for _ in range(int(rng.integers(2, 5))):
        cy, cx = int(rng.integers(0, h)), int(rng.integers(0, w))
        rad = int(rng.integers(max(4, min(h, w) // 16), max(8, min(h, w) // 6)))
        color = rng.integers(0, 256, size=3).astype(np.uint8)
        dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
        img[dist2 <= rad ** 2] = color
    return img


def rect_mask(rng: np.random.Generator, h: int, w: int, frac: float) -> np.ndarray:
    """生成面积约为 frac*H*W 的矩形二值掩码。"""
    area = max(1.0, frac * h * w)
    aspect = float(rng.uniform(0.5, 2.0))
    bh = max(1, min(h, int(np.sqrt(area / aspect))))
    bw = max(1, min(w, int(aspect * bh)))
    y0 = int(rng.integers(0, max(1, h - bh + 1)))
    x0 = int(rng.integers(0, max(1, w - bw + 1)))
    mask = np.zeros((h, w), np.uint8)
    mask[y0:y0 + bh, x0:x0 + bw] = 255
    return mask


class MockImageSource(base.ImageSource):
    DATASETS = ["COCO", "ImageNet", "OpenImages", "FFHQ", "Places"]

    def __init__(self, seed: int = 0, size: tuple[int, int] = (256, 256)):
        self.seed = seed
        self.size = size

    def iter_images(self, n: int) -> Iterator[tuple[np.ndarray, dict]]:
        h, w = self.size
        for i in range(n):
            rng = np.random.default_rng(self.seed + i)
            img = synth_image(rng, h, w)
            meta = {
                "source_dataset": self.DATASETS[i % len(self.DATASETS)],
                "camera_model": None,
                "resolution": [w, h],
                "license": "research-only",
            }
            yield img, meta


class MockWholeImageGenerator(base.WholeImageGenerator):
    def __init__(self, name: str = "mock-gen", family: str = "diffusion"):
        self.name, self.family = name, family

    def generate(self, prompt: str, params: dict) -> tuple[np.ndarray, dict]:
        seed = int(params.get("seed", 0))
        h, w = int(params.get("height", 256)), int(params.get("width", 256))
        pseed = (seed * 1000003 + stable_hash(prompt)) & 0x7FFFFFFF
        img = synth_image(np.random.default_rng(pseed), h, w)
        meta = {
            "generator_name": self.name, "generator_family": self.family,
            "seed": seed, "sampler": params.get("sampler", "DPM++ 2M"),
            "steps": int(params.get("steps", 30)),
            "cfg_scale": float(params.get("cfg_scale", 7.5)),
        }
        return img, meta


class MockInpainter(base.Inpainter):
    def __init__(self, name: str = "stable-diffusion-inpaint",
                 family: str = "diffusion"):
        self.name, self.family = name, family

    def inpaint(self, image: np.ndarray, mask: np.ndarray, prompt: str,
                params: dict) -> tuple[np.ndarray, dict]:
        seed = int(params.get("seed", 0))
        rng = np.random.default_rng((seed + stable_hash(prompt)) & 0x7FFFFFFF)
        out = image.copy()
        m = mask > 127
        color = rng.integers(0, 256, size=3).astype(np.uint8)
        out[m] = color
        # 轻微羽化边界，模拟重绘痕迹
        blurred = cv2.GaussianBlur(out, (5, 5), 0)
        k = np.ones((5, 5), np.uint8)
        edge = cv2.dilate(mask, k) - cv2.erode(mask, k)
        out[edge > 127] = blurred[edge > 127]
        meta = {"generator_name": self.name, "generator_family": self.family,
                "seed": seed}
        return out, meta


class MockSegmenter(base.Segmenter):
    def __init__(self, seed: int = 0):
        self.seed = seed

    def propose_masks(self, image: np.ndarray, k: int) -> list[np.ndarray]:
        h, w = image.shape[:2]
        base_seed = self.seed + int.from_bytes(
            hashlib.sha1(np.ascontiguousarray(image).tobytes()).digest()[:4], "big")
        fracs = np.linspace(0.02, 0.45, max(k, 1))
        masks = []
        for j in range(k):
            rng = np.random.default_rng((base_seed + j) & 0x7FFFFFFF)
            masks.append(rect_mask(rng, h, w, float(fracs[j])))
        return masks


class MockExplainer(base.Explainer):
    def explain(self, image: np.ndarray, mask: Optional[np.ndarray],
                context: dict) -> Explanation:
        region = context.get("region", "the masked region")
        mtype = context.get("manipulation_level3", "local AIGC inpainting")
        return Explanation(
            location_description=f"The manipulated region is located at {region}.",
            visual_artifact_description=("The object boundary appears overly smooth "
                                         "and inconsistent with the surrounding texture."),
            semantic_reasoning=("The lighting direction and noise of the edited region "
                                "do not match the rest of the scene."),
            forensic_conclusion=f"The image is likely manipulated by {mtype}.",
        )

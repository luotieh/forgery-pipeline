"""YAML 配置加载与校验。"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import yaml


@dataclass
class GeneratorSpec:
    name: str
    family: str
    kind: str


@dataclass
class StageScales:
    d0: int = 40
    d1_per_generator: int = 3
    d2: int = 24
    d3: int = 12
    d4: int = 8


@dataclass
class PipelineConfig:
    out_dir: str
    seed: int
    backend: str
    stages: dict
    scales: StageScales
    generators: list[GeneratorSpec] = field(default_factory=list)
    inpainters: list[GeneratorSpec] = field(default_factory=list)
    img2img: list[GeneratorSpec] = field(default_factory=list)
    postprocess_prob: float = 0.5
    split_config: str = "configs/split.yaml"
    vae_rt_frac: float = 0.15


def load_generators(path) -> tuple[list[GeneratorSpec], list[GeneratorSpec],
                                    list[GeneratorSpec]]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    gens = [GeneratorSpec(**g) for g in data.get("generators", [])]
    inps = [GeneratorSpec(**g) for g in data.get("inpainters", [])]
    imgs = [GeneratorSpec(**g) for g in data.get("img2img", [])]
    return gens, inps, imgs


def load_config(path) -> PipelineConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    gens, inps, imgs = load_generators(data["generators_config"])
    return PipelineConfig(
        out_dir=data["out_dir"],
        seed=int(data["seed"]),
        backend=data.get("backend", "mock"),
        stages=data["stages"],
        scales=StageScales(**data["scales"]),
        generators=gens,
        inpainters=inps,
        img2img=imgs,
        postprocess_prob=float(data.get("postprocess_prob", 0.5)),
        split_config=data.get("split_config", "configs/split.yaml"),
        vae_rt_frac=float(data.get("vae_rt_frac", 0.15)),
    )

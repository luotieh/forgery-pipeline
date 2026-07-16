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
    compositing_feather_px: int = 8
    # PATCH 9 Wave 2：生成时采样政策全部进 config（9.7 零硬编码验收）；新字段全部有默认，
    # 追加在既有字段之后，向后兼容（旧 yaml/旧构造点不受影响）。
    nuisance_cfg_grid: list[float] = field(default_factory=lambda: [5.0, 7.5, 10.0])
    nuisance_steps_grid: list[int] = field(default_factory=lambda: [30, 50])
    strength_range: tuple[float, float] = (0.1, 0.95)
    area_buckets: list[float] = field(default_factory=lambda: [0.05, 0.15, 0.35, 0.7])  # 桶界
    outpaint_border_fracs: list[float] = field(default_factory=lambda: [0.125, 0.25])
    resolution_groups: dict[int, list[str]] = field(default_factory=dict)  # {512:[生成器名...],1024:[...]}；空=单组现状
    prompt_bank: str = "configs/prompt_bank.yaml"
    grid_per_op: int = 0  # grid builder 每算子底图数，0=关


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
        compositing_feather_px=int(data.get("compositing_feather_px", 8)),
        nuisance_cfg_grid=list(data.get("nuisance_cfg_grid", [5.0, 7.5, 10.0])),
        nuisance_steps_grid=list(data.get("nuisance_steps_grid", [30, 50])),
        strength_range=tuple(data.get("strength_range", (0.1, 0.95))),
        area_buckets=list(data.get("area_buckets", [0.05, 0.15, 0.35, 0.7])),
        outpaint_border_fracs=list(data.get("outpaint_border_fracs", [0.125, 0.25])),
        resolution_groups={int(k): v for k, v in
                           data.get("resolution_groups", {}).items()},
        prompt_bank=data.get("prompt_bank", "configs/prompt_bank.yaml"),
        grid_per_op=int(data.get("grid_per_op", 0)),
    )

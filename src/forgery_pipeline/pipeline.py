"""阶段编排：D0→{D1,D2,D3}→D4→postprocess→split→manifest/stats（报告 §3）。"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import yaml
from forgery_pipeline import image_io, manifest
from forgery_pipeline.backends.mock import stable_hash
from forgery_pipeline.builders.d0_real import build_d0
from forgery_pipeline.builders.d1_whole import build_d1
from forgery_pipeline.builders.d2_local import build_d2
from forgery_pipeline.builders.d3_web import build_d3
from forgery_pipeline.builders.d4_explain import build_d4
from forgery_pipeline.config import PipelineConfig
from forgery_pipeline.postprocess.degradations import sample_and_apply
from forgery_pipeline.split.leakage import check_leakage
from forgery_pipeline.split.splitter import assign_splits
from forgery_pipeline.schema import Sample


def apply_postprocess(out_dir, samples: list[Sample], prob: float, seed: int) -> None:
    out_dir = Path(out_dir)
    for s in samples:
        if s.is_fake != 1:
            continue
        rng = np.random.default_rng((seed + stable_hash(s.image_id)) & 0x7FFFFFFF)
        if rng.random() >= prob:
            continue
        img = image_io.load_image(out_dir / s.image_path)
        degraded, pp = sample_and_apply(img, rng)
        image_io.save_image(degraded, out_dir / s.image_path)
        s.postprocess = pp


def run_pipeline(cfg: PipelineConfig) -> dict:
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    st = cfg.stages
    seed = cfg.seed

    d0 = build_d0(out, cfg.scales.d0, cfg.backend, seed) if st.get("d0") else []
    d1 = (build_d1(out, cfg.generators, cfg.scales.d1_per_generator, cfg.backend, seed)
          if st.get("d1") else [])
    d2 = (build_d2(out, d0, cfg.scales.d2, cfg.inpainters, cfg.backend, seed)
          if st.get("d2") else [])
    d3 = build_d3(out, d0, cfg.scales.d3, cfg.backend, seed) if st.get("d3") else []
    d4 = build_d4(out, d2 + d3, cfg.scales.d4, cfg.backend) if st.get("d4") else []

    for name, lib in [("d0", d0), ("d1", d1), ("d2", d2), ("d3", d3), ("d4", d4)]:
        manifest.write_jsonl(out / f"{name}.jsonl", lib)

    samples = d0 + d1 + d2 + d3 + d4

    if st.get("postprocess"):
        apply_postprocess(out, samples, cfg.postprocess_prob, seed)

    if st.get("split"):
        rules = yaml.safe_load(Path(cfg.split_config).read_text(encoding="utf-8"))
        assign_splits(
            samples,
            holdout_generators=rules.get("holdout_generators", []),
            holdout_manipulation=rules.get("holdout_manipulation", []),
            holdout_domains=rules.get("holdout_domains", ["Places"]),
            seed=seed,
        )
        leaks = check_leakage(samples)
        if leaks:
            raise RuntimeError("检测到数据泄漏: " + "; ".join(leaks))

    manifest.write_jsonl(out / "manifest.jsonl", samples)
    st_out = manifest.stats(samples)
    (out / "stats.json").write_text(
        json.dumps(st_out, ensure_ascii=False, indent=2), encoding="utf-8")
    return st_out

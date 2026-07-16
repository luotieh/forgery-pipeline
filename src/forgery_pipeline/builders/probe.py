"""Probe 受控子集生成：强度网格（Gate 1）+ 算子×族网格（Gate 2）。

与主数据集分离：realism 不重要，重要的是受控、带 `strength`/`operator` 标签，
供 gate_experiments 分析脚本直接读取。
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from forgery_pipeline import image_io, ids, manifest
from forgery_pipeline.backends import registry
from forgery_pipeline.backends.mock import stable_hash
from forgery_pipeline.builders.d0_real import build_d0
from forgery_pipeline.compositing import composite
from forgery_pipeline.config import GeneratorSpec
from forgery_pipeline.schema import Sample, TaskType

_NUM_TRAIN_TIMESTEPS = 1000


def _split_for(name, holdout) -> str:
    return "test_b" if name in set(holdout) else "train"


# 算子 -> (level1, level2, level3, 掩码类型)；img2img 为全图、无掩码
_OP_SPEC = {
    "img2img":            ("whole",   "diffusion",   None,                   None),
    "inpaint":            ("partial", "AIGC-editing", "mask_guided_inpainting", "box"),
    "outpaint":           ("partial", "AIGC-editing", "image_guided_editing",   "border"),
    "object_replacement": ("partial", "AIGC-editing", "object_replacement",     "box"),
    "background_editing": ("partial", "AIGC-editing", "image_guided_editing",   "invert"),
}


def _box(h, w, rng, frac=0.2):
    side = max(8, int((frac * h * w) ** 0.5))
    y = int(rng.integers(0, max(1, h - side))); x = int(rng.integers(0, max(1, w - side)))
    m = np.zeros((h, w), np.uint8); m[y:y + side, x:x + side] = 255
    return m


def _border(h, w, b_frac=0.25):
    m = np.full((h, w), 255, np.uint8); b = int(min(h, w) * b_frac)
    m[b:h - b, b:w - b] = 0
    return m


def _mask_for(kind, h, w, rng):
    if kind == "box":
        return _box(h, w, rng)
    if kind == "border":
        return _border(h, w)
    if kind == "invert":
        return 255 - _box(h, w, rng, frac=0.3)
    raise ValueError(kind)


def build_probe_strength(out_dir, bases: list[Sample], img2img_specs: list[GeneratorSpec],
                         strengths, backend: str, seed: int,
                         holdout_generators=(), cfg_grid=None, steps_grid=None) -> list[Sample]:
    """Gate 1：每个底图 × 每个强度做一次 img2img，记录 strength + init_timestep + split。

    cfg_grid/steps_grid（可选）：nuisance 抖动网格（prereg v2 §5 补充 probe）——
    每个 (base, strength) 再展开 CFG × steps 单元，参数记入 op_params（JSON）。
    两者皆 None 时行为与旧版逐字节一致（同 seed/同 image_id/op_params=None）。
    """
    out = Path(out_dir)
    cells = [(c, st) for c in (cfg_grid or [None]) for st in (steps_grid or [None])]
    samples: list[Sample] = []
    for bi, base in enumerate(bases):
        img = image_io.load_image(out / base.image_path)
        for s in strengths:
            spec = img2img_specs[bi % len(img2img_specs)]
            gen = registry.get_img2img(backend, spec.name, spec.family)
            base_sd = seed + bi * 1000 + int(round(float(s) * 100))
            for cfg, nsteps in cells:
                sd, key, params, opp = base_sd, f"{base.image_id}-{spec.name}-{s}", None, None
                if cfg is not None or nsteps is not None:
                    sd = base_sd + 100_000 + stable_hash(f"{cfg}-{nsteps}") % 99_991
                    op = {}
                    if cfg is not None:
                        op["cfg_scale"] = float(cfg); key += f"-c{cfg:g}"
                    if nsteps is not None:
                        op["steps"] = int(nsteps); key += f"-st{nsteps}"
                    params = {"seed": sd, **op}
                    opp = json.dumps(op, sort_keys=True)
                fake, meta = gen.img2img(img, "", float(s), params or {"seed": sd})
                iid = ids.make_image_id("probe_s", key)
                rel = f"probe/gate1_strength/{iid}.png"
                image_io.save_canonical(fake, out / rel)
                st = float(meta.get("strength", s))
                samples.append(Sample(
                    image_id=iid, image_path=rel, real_image_path=base.image_path, is_fake=1,
                    task_type=TaskType.whole_image_detection,
                    manipulation_level1="whole_generated", manipulation_level2="diffusion",
                    manipulation_level4=spec.name, generator_name=spec.name,
                    generator_family=spec.family, operator="img2img", op_params=opp,
                    strength=st, init_timestep=int(round(st * _NUM_TRAIN_TIMESTEPS)),
                    seed=sd, split=_split_for(spec.name, holdout_generators),
                    source_dataset=base.source_dataset,
                    sample_kind="edited", base_id=base.image_id,
                    io_chain=image_io.chain("decode", f"rs{img.shape[0]}", f"edit:{spec.name}", "png"),
                ))
    return samples


def build_probe_operator(out_dir, bases: list[Sample], img2img_specs: list[GeneratorSpec],
                         inpainter_specs: list[GeneratorSpec], operators,
                         backend: str, seed: int, holdout_generators=()) -> list[Sample]:
    """Gate 2：每个底图 × 每个算子 × 每个生成器，记录 operator + generator_family + split。"""
    out = Path(out_dir)
    seg = registry.get_segmenter(backend, seed=seed)  # noqa: F841 (预留真实分割)
    samples: list[Sample] = []
    for bi, base in enumerate(bases):
        img = image_io.load_image(out / base.image_path)
        h, w = img.shape[:2]
        for op in operators:
            l1, l2, l3, mkind = _OP_SPEC[op]
            specs = img2img_specs if op == "img2img" else inpainter_specs
            for spec in specs:
                sd = seed + bi * 1000 + (stable_hash(f"{op}-{spec.name}") % 500)
                rng = np.random.default_rng(sd & 0x7FFFFFFF)
                iid = ids.make_image_id("probe_op", f"{base.image_id}-{op}-{spec.name}")
                sp = _split_for(spec.name, holdout_generators)
                if op == "img2img":
                    gen = registry.get_img2img(backend, spec.name, spec.family)
                    fake, meta = gen.img2img(img, "", 0.6, {"seed": sd})
                    rel = f"probe/gate2_operator/{iid}.png"
                    image_io.save_canonical(fake, out / rel)
                    st = float(meta.get("strength", 0.6))
                    samples.append(Sample(
                        image_id=iid, image_path=rel, real_image_path=base.image_path,
                        is_fake=1, task_type=TaskType.whole_image_detection,
                        manipulation_level1="whole_generated", manipulation_level2="diffusion",
                        manipulation_level4=spec.name, generator_name=spec.name,
                        generator_family=spec.family, operator="img2img",
                        strength=st, init_timestep=int(round(st * _NUM_TRAIN_TIMESTEPS)),
                        seed=sd, split=sp, source_dataset=base.source_dataset,
                        sample_kind="edited", base_id=base.image_id,
                        io_chain=image_io.chain("decode", f"rs{img.shape[0]}", f"edit:{spec.name}", "png"),
                    ))
                else:
                    mask = _mask_for(mkind, h, w, rng)
                    painter = registry.get_inpainter(backend, spec.name, spec.family)
                    fake, _ = painter.inpaint(img, mask, op, {"seed": sd})
                    rel = f"probe/gate2_operator/{iid}.png"
                    mrel = f"probe/gate2_operator/masks/{iid}.png"
                    image_io.save_canonical(fake, out / rel)
                    image_io.save_mask(mask, out / mrel)
                    samples.append(Sample(
                        image_id=iid, image_path=rel, real_image_path=base.image_path,
                        mask_path=mrel, is_fake=1, task_type=TaskType.localization,
                        manipulation_level1="partial_manipulated", manipulation_level2=l2,
                        manipulation_level3=l3, manipulation_level4=spec.name,
                        generator_name=spec.name, generator_family=spec.family,
                        operator=op, mask_source="probe", seed=sd, split=sp,
                        source_dataset=base.source_dataset, base_id=base.image_id,
                        # masked 行整图直出（未走 composite），compositing 显式记为 none（PATCH 7.3）
                        compositing="none", sample_kind="edited",
                        io_chain=image_io.chain("decode", f"rs{img.shape[0]}", f"edit:{spec.name}", "png"),
                    ))
    return samples


def build_compositing_pairs(out_dir, bases: list[Sample], inpainter_specs: list[GeneratorSpec],
                            n_pairs: int, backend: str, seed: int) -> list[Sample]:
    """成对 compositing 探针（PATCH 7.3）：同一 base/mask/seed 只让 compositing 不同
    （none vs paste_feather），供下游隔离估计"回贴痕迹"对判别的边际贡献。

    每对共用同一次 painter.inpaint 输出（"同一 painter+seed 生成一次 gen"），
    仅在 compositing() 混合阶段分叉出两行，避免生成器随机性混入回贴变量。
    """
    out = Path(out_dir)
    samples: list[Sample] = []
    if n_pairs <= 0 or not bases or not inpainter_specs:
        return samples          # 与同文件其余 build_* 一致：空输入静默返回空列表而非报错
    for k in range(n_pairs):
        base = bases[k % len(bases)]
        img = image_io.load_image(out / base.image_path)
        h, w = img.shape[:2]
        rng = np.random.default_rng((seed + k) & 0x7FFFFFFF)
        mask = _mask_for("box", h, w, rng)
        spec = inpainter_specs[k % len(inpainter_specs)]
        painter = registry.get_inpainter(backend, spec.name, spec.family)
        sd = seed + k
        gen, _ = painter.inpaint(img, mask, "inpaint", {"seed": sd})
        mask01 = (mask > 127).astype(np.float32)
        pid = f"cp{k:04d}"
        mrel = f"probe/compositing_pairs/masks/{pid}.png"
        image_io.save_mask(mask, out / mrel)
        # 成对 probe 是固定诊断配置（要复查的是回贴痕迹本身），feather_px 固定为 8，
        # 不接 cfg.compositing_feather_px——主 run 的羽化强度调整不应改变这条诊断探针的口径。
        for mode, feather_px in (("none", None), ("paste_feather", 8)):
            fake = composite(img, gen, mask01, mode, feather_px=feather_px or 8)
            iid = ids.make_image_id("probe_cp", f"{pid}-{mode}")
            rel = f"probe/compositing_pairs/{iid}.png"
            image_io.save_canonical(fake, out / rel)
            samples.append(Sample(
                image_id=iid, image_path=rel, real_image_path=base.image_path,
                mask_path=mrel, is_fake=1, task_type=TaskType.localization,
                manipulation_level1="partial_manipulated", manipulation_level2="AIGC-editing",
                manipulation_level3="mask_guided_inpainting", manipulation_level4=spec.name,
                generator_name=spec.name, generator_family=spec.family,
                operator="inpaint", mask_source="probe", seed=sd,
                source_dataset=base.source_dataset, base_id=base.image_id,
                compositing=mode, feather_px=feather_px,
                probe_group="compositing_pair", pair_id=pid,
                sample_kind="edited",
                io_chain=image_io.chain("decode", f"rs{img.shape[0]}", f"edit:{spec.name}", "png"),
            ))
    return samples


def run_probe(out_dir, *, n_base: int, strengths, operators,
              img2img_specs: list[GeneratorSpec], inpainter_specs: list[GeneratorSpec],
              holdout_generators=(), backend: str = "mock", seed: int = 0,
              cfg_grid=None, steps_grid=None, compositing_pairs: int = 0) -> dict:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    bases = build_d0(out, n_base, backend, seed)
    for b in bases:
        b.split = "train"
    g1 = build_probe_strength(out, bases, img2img_specs, strengths, backend, seed,
                              holdout_generators, cfg_grid=cfg_grid, steps_grid=steps_grid)
    g2 = build_probe_operator(out, bases, img2img_specs, inpainter_specs, operators,
                              backend, seed, holdout_generators)
    manifest.write_jsonl(out / "gate1_strength.jsonl", g1)
    manifest.write_jsonl(out / "gate2_operator.jsonl", g2)
    pairs = (build_compositing_pairs(out, bases, inpainter_specs, compositing_pairs, backend, seed)
             if compositing_pairs > 0 else [])
    samples = bases + g1 + g2 + pairs
    manifest.write_jsonl(out / "manifest.jsonl", samples)
    return manifest.stats(samples)

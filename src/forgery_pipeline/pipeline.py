"""阶段编排：D0→{D1,D2,D3}→D4→postprocess→split→manifest/stats（报告 §3）。"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import yaml
from forgery_pipeline import image_io, manifest
from forgery_pipeline.backends import registry
from forgery_pipeline.backends.mock import stable_hash
from forgery_pipeline.builders.d0_real import build_d0
from forgery_pipeline.builders.d1_whole import build_d1
from forgery_pipeline.builders.d2_local import build_d2
from forgery_pipeline.builders.d3_web import build_d3
from forgery_pipeline.builders.d4_explain import build_d4
from forgery_pipeline.builders.grid_ops import build_grid
from forgery_pipeline.config import PipelineConfig
from forgery_pipeline.postprocess.degradations import sample_and_apply
from forgery_pipeline.split.leakage import check_leakage
from forgery_pipeline.split.splitter import assign_splits
from forgery_pipeline.schema import Sample


def apply_postprocess(out_dir, samples: list[Sample], prob: float, seed: int) -> list[Sample]:
    """退化版另存为新文件 + 新 Sample（postprocess_of 回链），原图与原行保持不变。"""
    out_dir = Path(out_dir)
    new_samples: list[Sample] = []
    for s in samples:
        if s.is_fake != 1:
            continue
        rng = np.random.default_rng((seed + stable_hash(s.image_id)) & 0x7FFFFFFF)
        if rng.random() >= prob:
            continue
        img = image_io.load_image(out_dir / s.image_path)
        degraded, pp = sample_and_apply(img, rng)
        p = Path(s.image_path)
        deg_rel = str(p.with_name(p.stem + "__deg" + p.suffix))
        image_io.save_image(degraded, out_dir / deg_rel)
        d = s.model_copy(deep=True)
        d.image_id = s.image_id + "__deg"
        d.image_path = deg_rel
        d.postprocess = pp
        d.postprocess_of = s.image_id     # 回链原图
        new_samples.append(d)
    return new_samples


def run_pipeline(cfg: PipelineConfig) -> dict:
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    st = cfg.stages
    seed = cfg.seed
    rules = {}
    if Path(cfg.split_config).exists():
        rules = yaml.safe_load(Path(cfg.split_config).read_text(encoding="utf-8")) or {}
    holdout_gen = set(rules.get("holdout_generators", []))

    # 多分辨率组摄取（PATCH 9 Wave2 9.2c）：resolutions 由 cfg.resolution_groups 键集派生
    # 并排序（空 = 单组现状，resolutions=None，build_d0 逐字节保持 HEAD 行为）。
    resolutions = sorted(cfg.resolution_groups.keys()) if cfg.resolution_groups else None
    d0 = (build_d0(out, cfg.scales.d0, cfg.backend, seed, resolutions=resolutions)
          if st.get("d0") else [])
    d1 = (build_d1(out, cfg.generators, cfg.scales.d1_per_generator, cfg.backend, seed)
          if st.get("d1") else [])
    # D2/D3/grid 的"底图池"只消费基准组（resolutions[0]，分辨率最小的一组）行——其余
    # 分辨率组（如 SDXL@1024）只出 real（+vae_rt）行，不参与 D2/D3 的局部编辑/网页伪造。
    # fake 侧对非基准分辨率组的覆盖改由 grid 按 policies.resolution_groups 对 img2img
    # spec 名分组路由负责（build_grid 的 resolution_pool 参数，见下方 grid 调用与
    # builders/grid_ops.py 模块 docstring）——这样 V2（split 内 real/fake 非生成链集合
    # 相等）才能在两个分辨率组间同时成立，而不需要 D2/D3 感知分辨率。resolutions=None
    # 时 d0_base is d0，下面的过滤是 no-op（单组现状不变）。
    if resolutions:
        base_res = resolutions[0]
        d0_base = [s for s in d0 if image_io.chain_resolution(s.io_chain) == base_res]
    else:
        d0_base = d0
    # D2 与 D3 使用不相交的底图子集：否则同一 origin-group 会同时含 D2 的 holdout 生成器
    # 与 D3 的 manual-web-edit，导致非 holdout 的 manual-web-edit 被拖入 test_b 触发泄漏。
    # 这补全 PATCH 6 的不变式「每个 origin-group 只含一类（holdout/非 holdout）生成器」。
    _half = len(d0_base) // 2
    d2_bases = d0_base[:_half] or d0_base
    d3_bases = d0_base[_half:] or d0_base
    d2 = (build_d2(out, d2_bases, cfg.scales.d2, cfg.inpainters, cfg.backend, seed,
                   holdout_inpainters=holdout_gen, feather_px=cfg.compositing_feather_px,
                   policies=cfg)
          if st.get("d2") else [])
    d3 = build_d3(out, d3_bases, cfg.scales.d3, cfg.backend, seed) if st.get("d3") else []
    # grid（img2img/outpaint 主 run 算子轴，PATCH 9 Wave2 T3）：底图取自 d3_bases（而非
    # d0 头部/d2_bases）——沿用上面 D2/D3 已建立的"不相交底图池"不变式。若改用
    # `d0[:grid_per_op]`，会与 d2_bases 重叠：build_d2 对约 20% 底图独立分配 holdout
    # inpainter（其 pool_hold，与 grid 无关、grid 不知情），一旦 grid 与 D2 共享底图，
    # grid 对该底图统一施加的 img2img 行（非 holdout 生成器）就会被同一 origin-group 的
    # D2 holdout 行拖进 test_b——一旦同一生成器名在别的底图上落在 train，
    # check_leakage 规则4（cross-generator 生成器不得同时出现在 train 与 test_b）即炸
    # （已实测复现，非假设）。d3_bases 上的行只有 generator_name="manual-web-edit"
    # （非 holdout），与 grid 共享安全。D4 explain 只消费 d2+d3（不变），grid 不喂给
    # D4，只并入主 samples 流。
    # resolution_pool=d0（全部分辨率行，非仅 d0_base）：grid 按 img2img spec 名反查
    # policies.resolution_groups 所属分辨率组时，需要在其中查到 base 的同源分辨率兄弟行
    # （见 build_grid docstring）。resolutions=None 时 d0_base is d0，反查恒落空，等价现状。
    grid = (build_grid(out, d3_bases[:cfg.grid_per_op] or d0_base, cfg.img2img, cfg.inpainters,
                       cfg, cfg.backend, seed, holdout_generators=holdout_gen,
                       feather_px=cfg.compositing_feather_px, resolution_pool=d0)
            if st.get("grid") and cfg.grid_per_op > 0 else [])
    d4 = build_d4(out, d2 + d3, cfg.scales.d4, cfg.backend) if st.get("d4") else []

    for name, lib in [("d0", d0), ("d1", d1), ("d2", d2), ("d3", d3), ("d4", d4), ("grid", grid)]:
        manifest.write_jsonl(out / f"{name}.jsonl", lib)

    samples = d0 + d1 + d2 + d3 + d4 + grid

    if st.get("postprocess"):
        samples += apply_postprocess(out, samples, cfg.postprocess_prob, seed)

    if st.get("split"):
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

    # 基准分辨率组成规则（PATCH 9 Wave2 T4 裁决执行——第四案「组成规则过滤」，V2/D2/V8
    # 语义零改动）：Test-C 测算子泛化，分辨率非其轴——多分辨率摄取时
    # base_resolution_only_splits（configs/split.yaml）所列 split 只保留基准分辨率
    # （resolutions[0]）行，其余行为同底图的分辨率副本，剔除无信息损失。若不过滤：D2 底图
    # 池与 grid 底图池互斥（PATCH 6 不变式）且 D2 不做分辨率路由 → test_c 永远只有基准
    # 分辨率 fake，real 侧却有全部分辨率兄弟行 → V2（split 内 real/fake 非生成链集合相等）
    # 结构性必红（B3 真实矩阵下同样成立：1024 inpainter 均为 holdout→test_b）。
    # 时点：split+leakage 之后、vae_rt 插入之前——vae_rt 只采 train/test_a/test_f 的
    # real 行，与被过滤 split 无交集；过滤只删行不改行，泄漏检查的各集合只会收缩，之后的
    # leakage/V 检照常。postprocess 子行与母行 io_chain 相同（model_copy 继承 rs 节点）且
    # 同 split（test_a→test_e carve-out 不涉及所列 split），故不会出现母行被留、子行被剔
    # 的拆散（e2e 有留存行 postprocess_of 完整性断言）。rs 节点缺失（None/legacy）的行
    # 防御性保留（run profile 下 V5 已禁止此类行进入主 run）。
    if st.get("split") and resolutions:
        only_base_splits = set(rules.get("base_resolution_only_splits") or [])
        if only_base_splits:
            base_res = resolutions[0]

            def _keep(s: Sample) -> bool:
                if s.split not in only_base_splits:
                    return True
                res = image_io.chain_resolution(s.io_chain)
                return res is None or res == base_res

            samples = [s for s in samples if _keep(s)]

    if st.get("split") and cfg.vae_rt_frac > 0:
        rt = registry.get_vae_rt(cfg.backend)
        extra: list[Sample] = []
        for s in samples:
            if s.sample_kind != "real" or s.split not in {"train", "test_a", "test_f"}:
                continue
            if (stable_hash(s.image_id + "vaert") % 1000) / 1000.0 >= cfg.vae_rt_frac:
                continue
            img = image_io.load_image(out / s.image_path)
            rel = str(Path(s.image_path).with_name(Path(s.image_path).stem + "__vaert.png"))
            image_io.save_canonical(rt.roundtrip(img), out / rel)
            v = s.model_copy(deep=True)
            v.image_id = s.image_id + "__vaert"; v.image_path = rel
            v.sample_kind = "real_vae_rt"; v.real_image_path = s.image_path
            v.base_id = s.base_id or s.image_id
            v.io_chain = s.io_chain.replace(">png", f">vae_rt:{rt.name}>png") if s.io_chain else f"vae_rt:{rt.name}"
            extra.append(v)
        samples += extra
        leaks = check_leakage(samples)
        if leaks:
            raise RuntimeError("vae_rt 插入后泄漏: " + "; ".join(leaks))

    manifest.write_jsonl(out / "manifest.jsonl", samples)
    st_out = manifest.stats(samples)
    (out / "stats.json").write_text(
        json.dumps(st_out, ensure_ascii=False, indent=2), encoding="utf-8")
    return st_out

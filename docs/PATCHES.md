# PATCHES — forgery-pipeline 修补清单（供 Claude Code）

> 目标仓库：`forgery-pipeline/`（你上传的版本）。
> 这些补丁解锁闸门 1/2、修两个真 bug、补齐"实现了但没接进流水线"的项。
> **说明**：代码按仓库现有约定（pydantic v2 / registry / mock 后端 / image_io）编写，但**未在生成环境运行**（缺 pydantic/imagehash 且无网络）。**应用后请 `pytest -q` 验证。**
> 优先级：**P0** 解锁闸门 + 修 bug；**P1** 完整性。逐 PATCH 应用。

---

## PATCH 1 (P0) — schema 增加 strength / init_timestep / operator / postprocess_of

`src/forgery_pipeline/schema.py`，在 `Sample` 现有字段后加入四个可选字段：

```python
    # 新增：编辑算子参数（Gate 1/2 与算子逆估计所需）
    strength: Optional[float] = None        # img2img/SDEdit 去噪强度 ∈ [0,1]，≈ 起始噪声级 t0/T
    init_timestep: Optional[int] = None     # SDEdit 起始 timestep（可选，便于直接读 t0）
    operator: Optional[str] = None          # 显式编辑算子（对齐闸门口径），取值见 labels.EDIT_OPERATORS
    postprocess_of: Optional[str] = None    # 退化样本回链原始 fake 的 image_id；原图为 None
```

把单位区间校验扩到 `strength`（改装饰器）：

```python
    @field_validator("mask_area_ratio", "quality_score", "strength")
    @classmethod
    def _check_unit_interval(cls, v):
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError("取值必须落在 [0, 1]")
        return v
```

在 `model_validator` 末尾追加 operator 校验（先 import）：

```python
from forgery_pipeline.labels import validate_labels, validate_operator  # 改这一行
```
```python
    @model_validator(mode="after")
    def _check_label_consistency(self):
        errs = validate_labels(
            self.is_fake, self.task_type.value, self.mask_path,
            self.manipulation_level1, self.manipulation_level2,
            self.manipulation_level3,
        )
        errs += validate_operator(self.operator)   # 新增
        if errs:
            raise ValueError("; ".join(errs))
        return self
```

---

## PATCH 2 (P1) — labels 增加算子词表

`src/forgery_pipeline/labels.py` 末尾追加：

```python
# 显式编辑算子词表（对齐闸门 {img2img, inpaint, outpaint, replace, background}）
EDIT_OPERATORS = {
    "img2img", "inpaint", "outpaint",
    "object_insertion", "object_replacement", "object_removal",
    "background_editing", "attribute_editing", "text_editing", "face_editing",
}


def validate_operator(op: Optional[str]) -> list[str]:
    if op is None or op in EDIT_OPERATORS:
        return []
    return [f"operator 非法: {op!r}"]
```

---

## PATCH 3 (P0) — Img2Img 后端接口 + mock 实现 + registry + 真实骨架

**`src/forgery_pipeline/backends/base.py`** 追加抽象类：

```python
class Img2ImgGenerator(ABC):
    """img2img / SDEdit 重绘（probe / 全图算子）。strength 控制起始噪声级 t0。"""
    @abstractmethod
    def img2img(self, image: Image, prompt: str, strength: float,
                params: dict) -> tuple[Image, dict]:
        ...
```

**`src/forgery_pipeline/backends/mock.py`** 追加（复用 `synth_image` / `stable_hash`）：

```python
class MockImg2Img(base.Img2ImgGenerator):
    def __init__(self, name: str = "stable-diffusion-img2img",
                 family: str = "diffusion"):
        self.name, self.family = name, family

    def img2img(self, image: np.ndarray, prompt: str, strength: float,
                params: dict) -> tuple[np.ndarray, dict]:
        seed = int(params.get("seed", 0))
        rng = np.random.default_rng((seed + stable_hash(prompt)) & 0x7FFFFFFF)
        h, w = image.shape[:2]
        regen = synth_image(rng, h, w)                       # 模型“重绘”内容
        a = float(np.clip(strength, 0.0, 1.0))               # 强度越大越偏离原图
        out = np.clip((1 - a) * image.astype(np.float64)
                      + a * regen.astype(np.float64), 0, 255).astype(np.uint8)
        meta = {"generator_name": self.name, "generator_family": self.family,
                "seed": seed, "strength": a,
                "sampler": params.get("sampler", "DPM++ 2M"),
                "steps": int(params.get("steps", 30)),
                "cfg_scale": float(params.get("cfg_scale", 7.5))}
        return out, meta
```

**`src/forgery_pipeline/backends/registry.py`** 追加解析函数：

```python
def get_img2img(backend: str, name: str, family: str) -> base.Img2ImgGenerator:
    if backend == "mock":
        return mock.MockImg2Img(name=name, family=family)
    _unsupported(backend)
```

**`src/forgery_pipeline/backends/real/diffusers_gen.py`** 追加真实骨架（与现有风格一致）：

```python
class DiffusersImg2Img(base.Img2ImgGenerator):
    def __init__(self, model_id: str, device: str = "cuda"):
        try:
            from diffusers import AutoPipelineForImage2Image  # noqa: F401
        except ImportError as e:
            raise NotImplementedError("未安装 diffusers：`pip install .[real]`。") from e
        # TODO: self.pipe = AutoPipelineForImage2Image.from_pretrained(model_id).to(device)
        raise NotImplementedError("参考骨架：加载 pipeline 并实现 img2img()。")

    def img2img(self, image, prompt, strength, params):
        # TODO: return self.pipe(prompt=prompt, image=PIL(image),
        #            strength=strength, num_inference_steps=params.get("steps", 30)) ...
        raise NotImplementedError
```

---

## PATCH 4 (P0) — 新增 probe builder（强度网格 + 算子×族网格）

**新建文件 `src/forgery_pipeline/builders/probe.py`**（已通过语法检查，可直接落盘）：

```python
"""Probe 受控子集生成：强度网格（Gate 1）+ 算子×族网格（Gate 2）。

与主数据集分离：realism 不重要，重要的是受控、带 `strength`/`operator` 标签，
供 gate_experiments 分析脚本直接读取。
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from forgery_pipeline import image_io, ids, manifest
from forgery_pipeline.backends import registry
from forgery_pipeline.backends.mock import stable_hash
from forgery_pipeline.builders.d0_real import build_d0
from forgery_pipeline.config import GeneratorSpec
from forgery_pipeline.schema import Sample, TaskType

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
                         strengths, backend: str, seed: int) -> list[Sample]:
    """Gate 1：每个底图 × 每个强度做一次 img2img，记录 strength。"""
    out = Path(out_dir)
    samples: list[Sample] = []
    for bi, base in enumerate(bases):
        img = image_io.load_image(out / base.image_path)
        for s in strengths:
            spec = img2img_specs[bi % len(img2img_specs)]
            gen = registry.get_img2img(backend, spec.name, spec.family)
            sd = seed + bi * 1000 + int(round(float(s) * 100))
            fake, meta = gen.img2img(img, "", float(s), {"seed": sd})
            iid = ids.make_image_id("probe_s", f"{base.image_id}-{spec.name}-{s}")
            rel = f"probe/gate1_strength/{iid}.png"
            image_io.save_image(fake, out / rel)
            samples.append(Sample(
                image_id=iid, image_path=rel, real_image_path=base.image_path, is_fake=1,
                task_type=TaskType.whole_image_detection,
                manipulation_level1="whole_generated", manipulation_level2="diffusion",
                manipulation_level4=spec.name, generator_name=spec.name,
                generator_family=spec.family, operator="img2img",
                strength=float(meta.get("strength", s)), seed=sd,
                source_dataset=base.source_dataset,
            ))
    return samples


def build_probe_operator(out_dir, bases: list[Sample], img2img_specs: list[GeneratorSpec],
                         inpainter_specs: list[GeneratorSpec], operators,
                         backend: str, seed: int) -> list[Sample]:
    """Gate 2：每个底图 × 每个算子 × 每个生成器族各一份，记录 operator + generator_family。"""
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
                if op == "img2img":
                    gen = registry.get_img2img(backend, spec.name, spec.family)
                    fake, meta = gen.img2img(img, "", 0.6, {"seed": sd})
                    rel = f"probe/gate2_operator/{iid}.png"
                    image_io.save_image(fake, out / rel)
                    samples.append(Sample(
                        image_id=iid, image_path=rel, real_image_path=base.image_path,
                        is_fake=1, task_type=TaskType.whole_image_detection,
                        manipulation_level1="whole_generated", manipulation_level2="diffusion",
                        manipulation_level4=spec.name, generator_name=spec.name,
                        generator_family=spec.family, operator="img2img",
                        strength=float(meta.get("strength", 0.6)), seed=sd,
                        source_dataset=base.source_dataset,
                    ))
                else:
                    mask = _mask_for(mkind, h, w, rng)
                    painter = registry.get_inpainter(backend, spec.name, spec.family)
                    fake, _ = painter.inpaint(img, mask, op, {"seed": sd})
                    rel = f"probe/gate2_operator/{iid}.jpg"
                    mrel = f"probe/gate2_operator/masks/{iid}.png"
                    image_io.save_image(fake, out / rel)
                    image_io.save_mask(mask, out / mrel)
                    samples.append(Sample(
                        image_id=iid, image_path=rel, real_image_path=base.image_path,
                        mask_path=mrel, is_fake=1, task_type=TaskType.localization,
                        manipulation_level1="partial_manipulated", manipulation_level2=l2,
                        manipulation_level3=l3, manipulation_level4=spec.name,
                        generator_name=spec.name, generator_family=spec.family,
                        operator=op, mask_source="probe", seed=sd,
                        source_dataset=base.source_dataset,
                    ))
    return samples


def run_probe(out_dir, *, n_base: int, strengths, operators,
              img2img_specs: list[GeneratorSpec], inpainter_specs: list[GeneratorSpec],
              backend: str = "mock", seed: int = 0) -> dict:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    bases = build_d0(out, n_base, backend, seed)
    g1 = build_probe_strength(out, bases, img2img_specs, strengths, backend, seed)
    g2 = build_probe_operator(out, bases, img2img_specs, inpainter_specs, operators, backend, seed)
    manifest.write_jsonl(out / "gate1_strength.jsonl", g1)
    manifest.write_jsonl(out / "gate2_operator.jsonl", g2)
    samples = bases + g1 + g2
    manifest.write_jsonl(out / "manifest.jsonl", samples)
    return manifest.stats(samples)
```

**`src/forgery_pipeline/config.py`**：让 `load_generators` 也读 img2img 列表，并加到 `PipelineConfig`：

```python
@dataclass
class PipelineConfig:
    # ...现有字段...
    img2img: list[GeneratorSpec] = field(default_factory=list)   # 新增
```
```python
def load_generators(path):
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    gens = [GeneratorSpec(**g) for g in data.get("generators", [])]
    inps = [GeneratorSpec(**g) for g in data.get("inpainters", [])]
    imgs = [GeneratorSpec(**g) for g in data.get("img2img", [])]   # 新增
    return gens, inps, imgs
```
（调用处 `load_config` 改成三元解包：`gens, inps, imgs = load_generators(...)`，并在构造 `PipelineConfig` 时传 `img2img=imgs`。）

**`src/forgery_pipeline/cli.py`** 增加 `probe` 子命令：

```python
def _cmd_probe(args) -> int:
    from forgery_pipeline.builders.probe import run_probe
    from forgery_pipeline.config import load_config
    cfg = load_config(args.config)
    st = run_probe(args.out, n_base=args.n_base,
                   strengths=[round(0.1 * i, 1) for i in range(1, 10)],
                   operators=["img2img", "inpaint", "outpaint",
                              "object_replacement", "background_editing"],
                   img2img_specs=cfg.img2img, inpainter_specs=cfg.inpainters,
                   backend=cfg.backend, seed=cfg.seed)
    import json; print(json.dumps(st, ensure_ascii=False, indent=2)); return 0
```
```python
    p_probe = sub.add_parser("probe", help="生成 Gate 1/2 受控 probe 子集")
    p_probe.add_argument("--config", required=True)
    p_probe.add_argument("--out", default="data/probe")
    p_probe.add_argument("--n-base", type=int, default=40, dest="n_base")
    p_probe.set_defaults(func=_cmd_probe)
```

---

## PATCH 5 (P0) — 修退化原地覆盖（保留原图 + 退化另存 + 回链）

**`src/forgery_pipeline/pipeline.py`**：替换 `apply_postprocess`，让它**返回新的退化样本**而非就地覆盖：

```python
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
```

`run_pipeline` 里改调用为**追加**：

```python
    if st.get("postprocess"):
        samples += apply_postprocess(out, samples, cfg.postprocess_prob, seed)
```

> 说明：退化样本沿用原图的 `real_image_path`，故 `origin_key` 仍与原图同组、同 split，不引入泄漏；`splitter` 的 `test_e`（退化 fake）逻辑照常生效。

**新增测试 `tests/test_postprocess_provenance.py`**：

```python
import numpy as np
from forgery_pipeline.schema import Sample, TaskType
from forgery_pipeline.pipeline import apply_postprocess
from forgery_pipeline import image_io


def test_degradation_keeps_original_and_links(tmp_path):
    img = (np.random.default_rng(0).integers(0, 256, (64, 64, 3))).astype(np.uint8)
    rel = "D2/x.jpg"; image_io.save_image(img, tmp_path / rel)
    orig = Sample(image_id="x", image_path=rel, real_image_path="D0/r.jpg",
                  mask_path="D2/m/x.png", is_fake=1, task_type=TaskType.localization,
                  manipulation_level1="partial_manipulated",
                  manipulation_level2="AIGC-editing",
                  manipulation_level3="object_replacement")
    deg = apply_postprocess(tmp_path, [orig], prob=1.0, seed=1)
    assert (tmp_path / rel).exists()                      # 原图仍在
    assert len(deg) == 1 and deg[0].postprocess_of == "x" # 退化样本回链
    assert deg[0].image_path != rel                       # 退化版独立成文件
```

---

## PATCH 6 (P1) — 修 splitter × D2 混合生成器泄漏（按底图分配生成器池）

问题：`d2_local` 对同一底图轮换不同 inpainter，而 `splitter` 整组判定 `test_b`；一旦某 inpainter 进 holdout，混合组会让非 holdout 生成器混入 `test_b`，触发 `check_leakage` 抛错。
修法：**生成时按底图把 inpainter 划入互斥池**，保证每个 origin-group 只用一类生成器。

**`src/forgery_pipeline/builders/d2_local.py`**：`build_d2` 增加 `holdout_inpainters` 参数与池路由：

```python
from forgery_pipeline.backends.mock import stable_hash  # 顶部新增

def build_d2(out_dir, base_samples, n, inpainters, backend="mock", seed=0,
             holdout_inpainters=()):
    hold = {i.name for i in inpainters if i.name in set(holdout_inpainters)}
    pool_hold = [i for i in inpainters if i.name in hold]
    pool_train = [i for i in inpainters if i.name not in hold] or inpainters
    # ...原循环内，取 base 后按底图选池：
    #   okey = (base.real_image_path or base.image_path)
    #   use_hold = pool_hold and (stable_hash(okey) % 5 == 0)   # ~20% 底图走 holdout 池
    #   pool = pool_hold if use_hold else pool_train
    #   inp = pool[len(samples) % len(pool)]
```

`pipeline.run_pipeline` 调用处把 holdout 传进去（从 split.yaml 读）：

```python
    rules = yaml.safe_load(Path(cfg.split_config).read_text(encoding="utf-8"))
    holdout_gen = set(rules.get("holdout_generators", []))
    d2 = (build_d2(out, d0, cfg.scales.d2, cfg.inpainters, cfg.backend, seed,
                   holdout_inpainters=holdout_gen) if st.get("d2") else [])
```

**补测试**到 `tests/test_split_leakage.py`：构造同一底图、混合 holdout/非 holdout 生成器，断言修复后 `check_leakage` 不抛（或断言每个 origin-group 生成器同池）。

---

## PATCH 7 (P1) — 把 quality_score 接进 D2

`d2_local` 目前不写 `quality_score`（恒 None），§7.5 路由对定位数据未生效。仿照 `d3_web` 接入：

`src/forgery_pipeline/builders/d2_local.py`，在保存样本前计算并路由：

```python
from forgery_pipeline.qc.quality_score import qes_score, route_from_score, bucket_from_score, area_validity

    # 在 ratio 计算之后、构造 Sample 之前：
    score = qes_score(
        confidence=0.9,                              # mock 占位；真实后端用模型/差异置信度
        boundary_sharpness=0.8,                      # 可由 mask 边界梯度估
        mask_consistency=1.0 if 0.01 <= ratio <= 0.50 else 0.5,
        semantic_consistency=0.8,
        area_validity=area_validity(ratio),
    )
    if route_from_score(score) == "reject":
        continue
    # 构造 Sample 时补两个字段：
    #   quality_score=round(score, 4), quality_bucket=bucket_from_score(score),
```

> 真实后端落地后，把 `confidence`/`boundary_sharpness` 换成实测信号（如重绘区与背景的残差一致性、mask 边界锐度）。

---

## PATCH 8 (P1) — 配置：补 img2img 列表 + 不同族 inpainter + holdout

**`configs/generators.yaml`** 增补：

```yaml
inpainters:
  - {name: stable-diffusion-inpaint, family: diffusion,       kind: inpaint}
  - {name: glide-inpaint,            family: diffusion,       kind: inpaint}
  - {name: kandinsky-inpaint,        family: kandinsky,       kind: inpaint}   # 不同族
  - {name: brushnet-sdxl,            family: diffusion-sdxl,   kind: inpaint}
img2img:                                                                       # 新增
  - {name: stable-diffusion-img2img, family: diffusion,       kind: img2img}
  - {name: sdxl-img2img,             family: diffusion-sdxl,   kind: img2img}
```

**`configs/split.yaml`** 把一个 inpainter 放进 holdout，让 D2 有 cross-generator `test_b`：

```yaml
holdout_generators: [ideogram, progan, kandinsky-inpaint]
```

---

## 验收

```bash
cd forgery-pipeline
pip install -e ".[dev]"            # 或装 pydantic/imagehash/pyyaml/opencv-python/numpy/pillow
pytest -q                         # 全绿；新增 test_postprocess_provenance 等通过

# 受控 probe（mock 即可跑通）
forgery-pipeline probe --config configs/pipeline.example.yaml --out data/probe --n-base 8
forgery-pipeline validate-manifest --path data/probe/manifest.jsonl
# 检查 gate1_strength.jsonl 每行有 strength + operator=img2img；
# gate2_operator.jsonl 覆盖 5 算子 × ≥2 generator_family。
```

应用顺序建议：**1 → 3 → 4 → 5（解锁闸门 + 修 bug）**，再 **2 → 6 → 7 → 8（完整性）**。

---

## PATCH 完成记录（2026-07-16）

**PATCH 8.3 ✅ 完成**：`checking/testc_geometry_probe.py`（5 维掩码几何 + 手写 OvR logistic，sklearn-free）+ 5 测试。n=1600 掩码（mock probe n_base=100；probe 掩码由 `_mask_for` 几何原语生成，mock/real 同代码路径，几何分布与后端无关）。**结果**：outpaint 1.000 / background_editing 1.000 → geometry-trivial，不得作 Test-C holdout；inpaint 0.829 / object_replacement 0.829 → eligible；决定性配对 inpaint↔object_replacement AUC=0.487≈机会线。**裁定：Test-C holdout = object_replacement**（默认确认），已写回 experiment plan §3 B3。失效条件：Phase B 更改算子×掩码约定须重跑。报告 `checking/testc_geometry_report_2026-07-16.md`。

**PATCH 6 ⤳ 被预注册 v2 路线取代**：其目标（gate1 回归口径、饱和诊断、预注册阈值）已由 `docs/PREREG_gate1_v2_2026-07-15.md`（锁定）+ `checking/gate1_confirmatory.py` 全部吸收并以更严格形式落地（cluster bootstrap / 嵌套 isotonic / 一次性评估）；相邻档位 AUC 曲线证实"单调+饱和"。不再单独实施。

---

**PATCH 7 ✅ 完成（2026-07-16，分支 feat/patch7-canonical-io，Tasks 1–8 子代理驱动 + 逐任务审查）**

- **落地**：Sample 新字段 io_chain/sample_kind/compositing/feather_px/probe_group/pair_id（回填脚本 `scripts/backfill_manifest_v7.py`）；`compositing.composite()`（none/paste/paste_feather）；canonical I/O（`image_io.chain/load_and_resize/save_canonical`，D0–D4+probe 全 PNG、真假共享非生成链）；VAE 往返硬负样本（`base.VaeRoundtrip`+Mock/SD1.5 实现+pipeline split 后分层插入+泄漏复查，`vae_rt_frac` 默认 0.15）；D2 掩码算子 50/50 显式回贴+probe 成对样本（`run_probe(compositing_pairs=N)`）+diffusers 隐式回贴审计守卫；validator **V1–V7**（`validate.check_all`，CLI `--profile auto|run`）+ stats 扩展（by_sample_kind/by_compositing/io_chain_by_fake_split）；e2e 冒烟（mock 全链 V1–V7 过 + `scripts/assert_compositing_pairs.py` 成对断言 + vae_rt 残差分布记录）。测试 131→**176 passed**。
- **对 spec 的已记录偏差**：①V2 非生成链 strip 规则含 `gen:*` 且忽略首 `decode`（D1 全生成行无源可解码，字面规则结构性 FAIL）；②V4 增 `min_real=10` 守卫（小 n 时比值离散取值结构性落不进 band）；③成对断言两常数实证修正——羽化带膨胀 `8×feather_px+1`（cv2 float32 高斯核半宽=4σ，字面值 4× 实测 90% 假阳）与 none 行差异阈值 0.15（mock 合成图结构地板≈0.20，0.5 系随机噪声外推不适用）；④MockInpainter 输出加全局 mock-VAE 印记（忠实模拟真实管线整图 VAE 直出，使 7.5 断言在 mock 可测）。
- **派生修复**：d4_explain 行补 io_chain/sample_kind 等字段回链（否则 V2 对含 D4 的 split 假阳）；labels.EDIT_OPERATORS 增 instruct_edit（V6 需要，§8.2 已预告）。

---

**PATCH 9 Wave 1 ✅ 完成记录（2026-07-16，分支 `feat/patch9-wave1`）**

- **9.3 全部**（split 防泄漏）：`base_id` 底图组键全链落地（`d082b90`，六个 builder 构造点 + pipeline 的 vae_rt/postprocess 行继承 + `backfill_manifest_v7` 回填扩展）；`validate.py` 新增 V8–V10 + 各配注毒负例单测 + `--split-config`（`0366147`），含**裁决A**（V8 豁免既有 test_a→test_e 退化 carve-out——母行 test_a、退化子行 test_e 视为 eval→eval 移动、非训练泄漏，不算 split 不一致）与**裁决B**（V8/V10 仅在 `profile=="run"` 时执行，probe 产物是受控仪器，故意让同一算子×生成器网格进 train，validator 不罚仪器设计）；`testc_holdout: object_replacement` 写死 `configs/split.yaml`（PATCH 8.3 几何探针裁定结果落地为唯一 config 源）。
- **9.4 工具层**（B3 驱动加固的可本地部分）：`scripts/b3_preflight.py` 三断言（HEAD 一致性/工作区净树/磁盘余量）+ `src/forgery_pipeline/rundir.py` 断点续跑幂等原语（`append_jsonl_fsync` / `mark_done`+`is_done` / `detect_incomplete_tail`）（`63c4caa`），`detect_incomplete_tail` 后续修复为字节级处理多字节 UTF-8 残尾（`21e0761`，审查发现）。评估禁令（事故B）无法被 preflight 机械检测，以模块 docstring 公约 + 代码审查双保险固化。latent 复用与 COCO fetch 的 GPU 侧驱动本体不在本 wave（见下方待办）。
- **9.5 草案**：`docs/PREREG_gate2_v3_draft.md`（本 deliverable，占位符 P1–P5 未填、未锁定；design freeze config 与评估前锁定另行）。
- **9.6 已另行执行**：`28ff0d0`（`checking/gate1_nuisance_decomposition_2026-07-16.md`）——nuisance 敏感维全部落在 steps（st30 ρ=0.707 ≈ 主 confirmatory 复现，st50 ρ=0.514，Δ≈−0.19），(7.5,30) 重拟合 nuis_effect=+0.113>0.10 → 按预定规则「固定 CFG/steps」限定机械升级为正文 limitation。

测试 **180 → 203 passed**（本 wave 新增 `test_base_id.py` / `test_validate_v8_v10.py` / `test_rundir.py` / `test_b3_preflight.py` 等）。

**偏差记录**（相对 PATCH 9 原文字面的修正，如实记）：
- 原文 9.3 字面「退化行继承母行 split」修正为**承认既有 test_a→test_e Test-E carve-out**（裁决A）——退化样本本就有一条既定的 test_a→test_e 特例通路（评测侧转移、非训练泄漏），V8 若不豁免会对这条既有合法路径产生结构性假阳；母行在 train/val 时退化行仍须同 split，豁免不适用于该情形。
- V8/V10 限定为**仅 run profile 执行**（裁决B）——`data/probe` 产物的性质是"受控仪器"：探针网格故意让所有算子×生成器组合都在同一批数据里出现（包括主 run 会 holdout 的 `object_replacement`），以便探针自身测出算子的可分性/几何平凡性；对 probe 数据套用主 run 的 split 互斥断言会误伤仪器设计本身。

**待办（Wave 2）**：9.1/9.2 builder 采样政策（CFG/steps 逐图抖动、强度连续采样、prompt bank、掩码面积分层、分辨率组配套 real/vae_rt 行）；9.5 设计冻结 config（cell 网格 commit 进 config）+ PREREG_gate2_v3 评估前锁定；latent 复用（GPU 侧，`diffusers_gen` 内 VAE encode 缓存）；COCO fetch（`scripts/fetch_real_images.py` 新数据源路径的 socket timeout 生效性确认 + 归档 checksum）。
- **GPU 侧待复核**：SDVaeRoundtrip 真实冒烟（下次开机随 PATCH 9/B3 一并）。

---

**PATCH 9 Wave 2 ✅ 完成（2026-07-16，分支 `feat/patch9-wave2`，Tasks 1–6）**

- **9.1**（nuisance 逐图采样 + 强度连续 + V11）：`PipelineConfig` 新增采样政策字段——`nuisance_cfg_grid`/`nuisance_steps_grid`/`strength_range`/`area_buckets`/`outpaint_border_fracs`/`resolution_groups`/`prompt_bank`/`grid_per_op`（`e3755a7`）；D2/grid 行按 `stable_hash(iid+salt)` 逐图确定性抽样 CFG{5,7.5,10}×steps{30,50}（`09bd1b7`/`2a3a9a9`）；grid 的 img2img 行强度连续 `U(0.1,0.95)`（`s=0.1+0.85·(stable_hash(iid+"s")%10000)/10000`，确定性）；`validate.check_v11` 扩散编辑行 nuisance 记录完备性 + 逐 split 单元下限（`d9f7c56`，`cea93dc` 补 steps 侧非数值守卫对称化）。
- **9.2a**（prompt bank v1）：`configs/prompt_bank.yaml`（img2img/inpaint/object/background 四节英文模板）+ `prompts.py`（`bank_version()` sha1[:12] / `pick_prompt()` 确定性抽取），逐行记入 `op_params.prompt`/`prompt_bank_version`（`e3755a7`）。
- **9.2b**（面积分桶 + V12 + outpaint 边带网格）：D2 按 `area_buckets` 对候选 mask 分层轮转选桶（空桶顺延），`mask_area_ratio` 恢复无条件落行（`09bd1b7`，`d22c46c` 修正 Task2 规格误标）；`grid_ops.build_grid` 新增 outpaint 边带宽度网格行（`outpaint_border_fracs`）；`validate.check_v12` masked 算子行 `mask_area_ratio` 完备性 + 面积分桶下限（`d9f7c56`）。
- **9.2c**（多分辨率组）：`build_d0(resolutions=...)` 多分辨率摄取（`resize_square` 抽出接线（`load_and_resize` 委托同核，B2 COCO 摄取时启用））+ grid 按组路由同源分辨率兄弟行（`1c981b6`）；`configs/split.yaml` 新增 `base_resolution_only_splits: [test_c]`（test_c 基准分辨率组成规则，`f34e3c6`）+ test_b 覆盖设计约束文档化（`af20d73`）——**V2（split 内 real/fake 非生成链集合相等）天然承担 9.2c 断言，零新判据**。
- **9.5**：`configs/gate2_probe.yaml` 设计冻结骨架入库（本任务，Task 6）；`docs/PREREG_gate2_v3_draft.md` P2/P3 关联注记指向该 config。
- **+ grid_ops 主 run 算子轴**：新 `builders/grid_ops.py::build_grid`（每底图×每 img2img spec 一行 img2img + 每底图一行 outpaint，`2a3a9a9`），pipeline stage `"grid"` 接线；**SDXL 映射条目**：`diffusers_gen.py` MODEL 映射补 `sdxl-img2img`/`sdxl-inpaint`（代码级，GPU 冒烟另行）。

**裁决记录**：D2 七类操纵→operator 映射五合一进 `inpaint`（`object_replacement`/`background_editing` 保留原名，其余五类粗分进 `inpaint`，level3/level4 保留细粒度）；grid 池分离恢复 PATCH 6 不变式——img2img/outpaint 按 `holdout_generators` 二分 `pool_hold`/`pool_train`（镜像 `d2_local.py`，B3 holdout 形态可构造）+ `d3_bases` 二分为不相交的 grid 池（前半）/D3 池（后半）（`05e58c5`）；**「每个分辨率组须同时有 holdout 与非 holdout 成员」B3 config 约束**——非 holdout 侧缺位时 train/val/test_a 的 real `{rs64,rs128}` vs fake `{rs64}` 结构性触发 V2，故 `test_b` 不进 `base_resolution_only_splits`（`af20d73`）；**测试反模式教训：小 n 全栈 `check_all()==[]` 是掷币，机制作用域断言 + firing 锚是正解**——三条多分辨率 e2e 原用全局空断言，把无关 split 的小 n 组合噪声一并背上（seed/规模掷币），重写为确定性性质（不变量本体 + 定向前缀断言 + 条件守卫 firing 计数），验证 sweep `seed∈{0..5}×d0∈{16,20,28}` = 54/54 全过、firing 统计 A `grid_hold` 18/18、A `d2_hold` 18/18、B `fired` 18/18（`4beab3c`）。

**偏差记录**（相对 PATCH 9 Wave 2 原文字面的修正）：
- V11 cell-floor 仅检查 split 内出现过的单元（结构性——check_v11 不接收网格枚举；B3 驱动应另从 stats.by_nuisance_cell 断言全网格在场）
- mask_area_ratio 现统一 round(4)（含 policies=None 路径，与旧原始 float 有精度级差异，科学无害）

测试 **207 → 266 passed**。

**待办（B3/GPU 侧）**：COCO fetch 层（9.4 事故A：socket 超时对新数据源生效性确认 + 归档 checksum）随 B3 驱动/B2 摄取 + SDXL/SDVaeRoundtrip GPU 冒烟、B1 矩阵 SDXL 双栖张力解、连续 CFG 采样、LaMa/IP2P/Flux + grid seed 命名空间已修（本 commit）。

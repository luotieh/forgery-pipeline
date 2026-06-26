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

# 闸门数据支持 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 补齐管线对 `EXECUTION_CHECKLIST.md` 闸门 0–4 的数据覆盖：probe 留出生成器（`split=train/test_b`）、img2img 填 `init_timestep`、`stats` 增按生成器/算子计数、`configs/probe.yaml` + CLI 对齐、`docs/GATE_DATA.md` 映射表。

**Architecture:** 全部改动集中在 `builders/probe.py`（split/init_timestep/run_probe 签名）、`manifest.py`（stats 两键）、`cli.py`（`_cmd_probe` 读 probe 配置）+ 新增 `configs/probe.yaml`、`tests/test_probe.py`、`docs/GATE_DATA.md`。复用 `split` 字段，不改 schema/backend/主 run。

**Tech Stack:** Python ≥3.10、pydantic v2、numpy、pyyaml、pytest（均已有）。

## Global Constraints

- 不新增运行时依赖；不改 backend 抽象 / mock / 主 `run` / schema（复用 `split`，不加新字段）。
- 注释/文档中文、标识符 English、确定性。
- 向后兼容：`probe` 子命令对旧 config（无 `n_base`/`strengths`/`holdout_generators`）走默认值，holdout 默认空 → 全 `split=train`。
- `init_timestep = round(strength × 1000)`（`_NUM_TRAIN_TIMESTEPS = 1000`）。
- probe 留出默认 `{kandinsky-inpaint, sdxl-img2img}`（由 `configs/probe.yaml` 提供）。

---

## File Structure

| 文件 | 改动 |
|---|---|
| `src/forgery_pipeline/builders/probe.py` | 改：split 标记 + init_timestep + `run_probe` 增 `holdout_generators` |
| `src/forgery_pipeline/manifest.py` | 改：`stats()` 增 `by_generator_name` / `by_operator` |
| `src/forgery_pipeline/cli.py` | 改：`_cmd_probe` 读 probe 配置；argparse 默认改 None |
| `configs/probe.yaml` | 新增 |
| `tests/test_probe.py` | 新增（probe split/init_timestep/gate 覆盖 + CLI） |
| `tests/test_manifest.py` | 改：补 stats 新键测试 |
| `docs/GATE_DATA.md` | 新增（闸门→产物→字段→命令） |
| `README.md` | 改：快速开始指向 GATE_DATA.md |

---

## Task 1: probe 留出生成器标记 + init_timestep

**Files:**
- Modify: `src/forgery_pipeline/builders/probe.py`
- Create: `tests/test_probe.py`

**Interfaces:**
- Produces:
  - `build_probe_strength(out_dir, bases, img2img_specs, strengths, backend, seed, holdout_generators=()) -> list[Sample]`
  - `build_probe_operator(out_dir, bases, img2img_specs, inpainter_specs, operators, backend, seed, holdout_generators=()) -> list[Sample]`
  - `run_probe(out_dir, *, n_base, strengths, operators, img2img_specs, inpainter_specs, holdout_generators=(), backend="mock", seed=0) -> dict`
  - probe 样本：`generator_name ∈ holdout` → `split="test_b"`，否则 `"train"`；底图 `split="train"`；img2img 样本 `init_timestep=round(strength*1000)`，inpaint 样本 `init_timestep=None`。

- [ ] **Step 1: 写失败测试** `tests/test_probe.py`

```python
from forgery_pipeline.config import GeneratorSpec
from forgery_pipeline.builders.probe import run_probe
from forgery_pipeline import manifest

_IMG2IMG = [GeneratorSpec("sd-img2img", "diffusion", "img2img"),
            GeneratorSpec("sdxl-img2img", "diffusion-sdxl", "img2img")]
_INPS = [GeneratorSpec("sd-inpaint", "diffusion", "inpaint"),
         GeneratorSpec("kandinsky-inpaint", "kandinsky", "inpaint")]
_OPS = ["img2img", "inpaint", "outpaint", "object_replacement", "background_editing"]


def _run(tmp_path):
    run_probe(tmp_path / "probe", n_base=2, strengths=[0.2, 0.5, 0.8], operators=_OPS,
              img2img_specs=_IMG2IMG, inpainter_specs=_INPS,
              holdout_generators={"sdxl-img2img", "kandinsky-inpaint"}, seed=0)
    return manifest.read_jsonl(tmp_path / "probe" / "manifest.jsonl")


def test_probe_split_marks_holdout_as_test_b(tmp_path):
    rows = _run(tmp_path)
    seen = [r for r in rows if r.split == "train"]
    held = [r for r in rows if r.split == "test_b"]
    assert seen and held
    assert all(r.generator_name in {"sdxl-img2img", "kandinsky-inpaint"}
               for r in held if r.generator_name)


def test_probe_init_timestep_for_img2img_only(tmp_path):
    rows = _run(tmp_path)
    i2i = [r for r in rows if r.operator == "img2img"]
    assert i2i and all(r.init_timestep is not None for r in i2i)
    assert all(abs(r.init_timestep - round((r.strength or 0) * 1000)) <= 1 for r in i2i)
    inpaint = [r for r in rows if r.operator == "inpaint"]
    assert inpaint and all(r.init_timestep is None for r in inpaint)


def test_probe_gate_files_coverage(tmp_path):
    run_probe(tmp_path / "probe", n_base=2, strengths=[0.2, 0.5, 0.8], operators=_OPS,
              img2img_specs=_IMG2IMG, inpainter_specs=_INPS,
              holdout_generators={"sdxl-img2img", "kandinsky-inpaint"}, seed=0)
    g1 = manifest.read_jsonl(tmp_path / "probe" / "gate1_strength.jsonl")
    assert g1 and all(r.operator == "img2img" and r.strength is not None for r in g1)
    g2 = manifest.read_jsonl(tmp_path / "probe" / "gate2_operator.jsonl")
    assert {r.operator for r in g2} == set(_OPS)
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_probe.py -q`
Expected: FAIL（`run_probe` 不接受 `holdout_generators` / split 未标记）

- [ ] **Step 3: 修改 `builders/probe.py`**

在文件顶部常量区（`_OP_SPEC` 之前）加：

```python
_NUM_TRAIN_TIMESTEPS = 1000


def _split_for(name, holdout) -> str:
    return "test_b" if name in set(holdout) else "train"
```

把 `build_probe_strength` 整体替换为：

```python
def build_probe_strength(out_dir, bases: list[Sample], img2img_specs: list[GeneratorSpec],
                         strengths, backend: str, seed: int,
                         holdout_generators=()) -> list[Sample]:
    """Gate 1：每个底图 × 每个强度做一次 img2img，记录 strength + init_timestep + split。"""
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
            st = float(meta.get("strength", s))
            samples.append(Sample(
                image_id=iid, image_path=rel, real_image_path=base.image_path, is_fake=1,
                task_type=TaskType.whole_image_detection,
                manipulation_level1="whole_generated", manipulation_level2="diffusion",
                manipulation_level4=spec.name, generator_name=spec.name,
                generator_family=spec.family, operator="img2img",
                strength=st, init_timestep=int(round(st * _NUM_TRAIN_TIMESTEPS)),
                seed=sd, split=_split_for(spec.name, holdout_generators),
                source_dataset=base.source_dataset,
            ))
    return samples
```

把 `build_probe_operator` 整体替换为：

```python
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
                    image_io.save_image(fake, out / rel)
                    st = float(meta.get("strength", 0.6))
                    samples.append(Sample(
                        image_id=iid, image_path=rel, real_image_path=base.image_path,
                        is_fake=1, task_type=TaskType.whole_image_detection,
                        manipulation_level1="whole_generated", manipulation_level2="diffusion",
                        manipulation_level4=spec.name, generator_name=spec.name,
                        generator_family=spec.family, operator="img2img",
                        strength=st, init_timestep=int(round(st * _NUM_TRAIN_TIMESTEPS)),
                        seed=sd, split=sp, source_dataset=base.source_dataset,
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
                        operator=op, mask_source="probe", seed=sd, split=sp,
                        source_dataset=base.source_dataset,
                    ))
    return samples
```

把 `run_probe` 整体替换为：

```python
def run_probe(out_dir, *, n_base: int, strengths, operators,
              img2img_specs: list[GeneratorSpec], inpainter_specs: list[GeneratorSpec],
              holdout_generators=(), backend: str = "mock", seed: int = 0) -> dict:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    bases = build_d0(out, n_base, backend, seed)
    for b in bases:
        b.split = "train"
    g1 = build_probe_strength(out, bases, img2img_specs, strengths, backend, seed,
                              holdout_generators)
    g2 = build_probe_operator(out, bases, img2img_specs, inpainter_specs, operators,
                              backend, seed, holdout_generators)
    manifest.write_jsonl(out / "gate1_strength.jsonl", g1)
    manifest.write_jsonl(out / "gate2_operator.jsonl", g2)
    samples = bases + g1 + g2
    manifest.write_jsonl(out / "manifest.jsonl", samples)
    return manifest.stats(samples)
```

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/test_probe.py -q` → PASS

- [ ] **Step 5: 提交**

```bash
git add src/forgery_pipeline/builders/probe.py tests/test_probe.py
git commit -m "feat(probe): 留出生成器标 split=test_b + img2img init_timestep"
```

---

## Task 2: stats 增按生成器/算子计数

**Files:**
- Modify: `src/forgery_pipeline/manifest.py`
- Modify: `tests/test_manifest.py`

**Interfaces:**
- Produces：`stats(samples)` 返回值新增 `by_generator_name: dict` 与 `by_operator: dict`。

- [ ] **Step 1: 追加失败测试** `tests/test_manifest.py`

```python
def test_stats_includes_generator_name_and_operator():
    from forgery_pipeline.schema import Sample, TaskType
    def _i2i(i):
        return Sample(image_id=f"a{i}", image_path=f"x{i}.png", is_fake=1,
                      task_type=TaskType.whole_image_detection,
                      manipulation_level1="whole_generated", manipulation_level2="diffusion",
                      generator_name="sd-img2img", operator="img2img")
    s = manifest.stats([_i2i(0), _i2i(1)])
    assert s["by_generator_name"]["sd-img2img"] == 2
    assert s["by_operator"]["img2img"] == 2
```

- [ ] **Step 2: 运行确认失败** → FAIL（KeyError）

- [ ] **Step 3: 修改 `manifest.py` 的 `stats()`** —— 在 return 字典里 `by_generator_family` 之后追加两键：

```python
        "by_generator_name": dict(
            Counter(s.generator_name for s in samples if s.generator_name)),
        "by_operator": dict(Counter(s.operator for s in samples if s.operator)),
```

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/test_manifest.py -q` → PASS

- [ ] **Step 5: 提交**

```bash
git add src/forgery_pipeline/manifest.py tests/test_manifest.py
git commit -m "feat(manifest): stats 增 by_generator_name/by_operator（均衡可验证）"
```

---

## Task 3: configs/probe.yaml + CLI 读配置

**Files:**
- Create: `configs/probe.yaml`
- Modify: `src/forgery_pipeline/cli.py`
- Modify: `tests/test_probe.py`

**Interfaces:**
- Consumes：`config.load_generators`、`builders.probe.run_probe`
- Produces：`forgery-pipeline probe --config configs/probe.yaml [--out DIR] [--n-base N]`；从 config 读 `n_base/strengths/operators/holdout_generators/backend/seed`，CLI `--out/--n-base` 可覆盖。

- [ ] **Step 1: 写 `configs/probe.yaml`**

```yaml
out_dir: data/probe
seed: 1234
backend: mock
generators_config: configs/generators.yaml
n_base: 40
strengths: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
operators: [img2img, inpaint, outpaint, object_replacement, background_editing]
# 留出族（cross-generator 测试，闸门 3b）：一个 inpainter + 一个 img2img 模型
holdout_generators: [kandinsky-inpaint, sdxl-img2img]
```

- [ ] **Step 2: 追加失败测试** `tests/test_probe.py`

```python
def test_probe_cli_via_probe_yaml(tmp_path):
    from forgery_pipeline.cli import main
    out = tmp_path / "probe"
    assert main(["probe", "--config", "configs/probe.yaml",
                 "--out", str(out), "--n-base", "2"]) == 0
    rows = manifest.read_jsonl(out / "manifest.jsonl")
    assert any(r.split == "test_b" for r in rows)        # 留出生成器被标记
    assert any(r.init_timestep is not None for r in rows)  # img2img 带 init_timestep
    assert main(["validate-manifest", "--path", str(out / "manifest.jsonl")]) == 0
```

- [ ] **Step 3: 运行确认失败** → FAIL（旧 `_cmd_probe` 用 `load_config`，无 holdout）

- [ ] **Step 4: 改 `cli.py` 的 `_cmd_probe`** —— 整体替换为：

```python
def _cmd_probe(args) -> int:
    import yaml
    from forgery_pipeline.config import load_generators
    from forgery_pipeline.builders.probe import run_probe
    data = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    _, inps, imgs = load_generators(data["generators_config"])
    out = args.out or data.get("out_dir", "data/probe")
    n_base = args.n_base if args.n_base is not None else int(data.get("n_base", 40))
    st = run_probe(
        out, n_base=n_base,
        strengths=data.get("strengths", [round(0.1 * i, 1) for i in range(1, 10)]),
        operators=data.get("operators",
                           ["img2img", "inpaint", "outpaint",
                            "object_replacement", "background_editing"]),
        img2img_specs=imgs, inpainter_specs=inps,
        holdout_generators=set(data.get("holdout_generators", [])),
        backend=data.get("backend", "mock"), seed=int(data.get("seed", 0)),
    )
    print(json.dumps(st, ensure_ascii=False, indent=2))
    return 0
```

并把 `probe` 子命令的 argparse 默认改为 None（让 config 生效）：

```python
    p_probe.add_argument("--out", default=None, help="输出目录，默认取 config 的 out_dir")
    p_probe.add_argument("--n-base", type=int, default=None, dest="n_base",
                         help="底图数，默认取 config 的 n_base")
```

- [ ] **Step 5: 运行确认通过**

Run: `pytest tests/test_probe.py -q` → PASS

- [ ] **Step 6: 提交**

```bash
git add configs/probe.yaml src/forgery_pipeline/cli.py tests/test_probe.py
git commit -m "feat(cli): probe 读 configs/probe.yaml（含 holdout/strengths/operators）"
```

---

## Task 4: docs/GATE_DATA.md + README

**Files:**
- Create: `docs/GATE_DATA.md`
- Modify: `README.md`

- [ ] **Step 1: 写 `docs/GATE_DATA.md`**

```markdown
# GATE_DATA — 闸门数据对照（forgery-pipeline → EXECUTION_CHECKLIST.md）

本管线产出 `docs/EXECUTION_CHECKLIST.md` 闸门 0–4 所需的全部受控数据。
分析脚本在独立仓 `gate_experiments/`；本表给出「闸门 → 用哪个产物 → 关键字段 → 命令」。

## 生成命令

```bash
forgery-pipeline run   --config configs/pipeline.example.yaml --out data/run     # 主数据集（Test-A..F）
forgery-pipeline probe --config configs/probe.yaml            --out data/probe   # 受控 probe（Gate 1/2/3b）
forgery-pipeline validate-manifest --path data/probe/manifest.jsonl
```

## 对照表

| 闸门 | 用哪个产物 | 关键字段 | 备注 |
|---|---|---|---|
| 0 信号地基 | `data/probe`（gate2 inpaint）/ `data/run`（D2/D3） | `image_path` `real_image_path` `mask_path` | 成对 + 已知掩码，调 latent 对齐 |
| 1 t0 可恢复 | `data/probe/gate1_strength.jsonl` | `strength`(0.1–0.9) `init_timestep`(=round(strength·1000)) `operator=img2img` | 强度网格 |
| 2 算子可分 | `data/probe/gate2_operator.jsonl` | `operator`(5 类) `generator_family`/`generator_name` | 5 算子 × ≥2 族；跨模型按 family/name 切 |
| 3a 多 σ 增量 | 任意上面数据 | — | 消融，分析侧 |
| 3b 跨生成器 | `data/probe`（`split=test_b`）/ `data/run`（`split=test_b`） | `split` `generator_name` | 留出族：`kandinsky-inpaint` `sdxl-img2img`（probe）；`ideogram/progan/kandinsky-inpaint`（run） |
| 4 Test-A..F | `data/run/manifest.jsonl` | `split`(train/val/test_a..f) | 8 路评测轴 |
| 4 均衡采样 | `data/*/stats.json` 或 `stats` 输出 | `by_generator_name` `by_operator` | 核验每生成器计数 |
| 鲁棒性(Test-E) | `data/run`（退化行） | `postprocess` `postprocess_of` | 退化样本独立成行 + 回链 |
```

- [ ] **Step 2: README 快速开始追加一行** —— 在「可视化检视生成的数据集」一节之后加：

```markdown
### 闸门实验数据（论文 Inverting the Edit）

```bash
forgery-pipeline probe --config configs/probe.yaml --out data/probe
```

产出受控 probe 子集（强度网格 + 算子×族网格 + 留出生成器 `test_b`）。每个闸门用哪个产物/字段见 [`docs/GATE_DATA.md`](docs/GATE_DATA.md)。
```

- [ ] **Step 3: 全量测试 + 真实 probe 冒烟**

```bash
pytest -q
forgery-pipeline probe --config configs/probe.yaml --out data/probe --n-base 8
forgery-pipeline validate-manifest --path data/probe/manifest.jsonl
```
Expected: 全绿；probe manifest 校验通过、含 `split=test_b` 与 `init_timestep`。

- [ ] **Step 4: 提交**

```bash
git add docs/GATE_DATA.md README.md
git commit -m "docs: GATE_DATA 闸门数据对照表 + README 指引"
```

---

## Self-Review

**1. Spec coverage：**
- §2.1 probe split + init_timestep + run_probe 签名 → Task 1 ✓
- §2.2 stats 两键 → Task 2 ✓
- §2.3 configs/probe.yaml + _cmd_probe → Task 3 ✓
- §2.4 GATE_DATA.md + README → Task 4 ✓
- §4 测试（split/init_timestep/gate 覆盖/stats/CLI）→ Task 1+2+3 ✓

无缺口。

**2. Placeholder scan：** 各步含完整代码/命令/期望，无 TBD/TODO。

**3. Type consistency：**
- `run_probe(..., holdout_generators=(), backend, seed)`（Task 1）被 Task 3 `_cmd_probe` 调用，关键字参数名一致 ✓
- `build_probe_strength/operator(..., holdout_generators=())`（Task 1）被 `run_probe` 调用一致 ✓
- `_split_for(name, holdout)`、`_NUM_TRAIN_TIMESTEPS`（Task 1）使用一致 ✓
- `stats` 新键名 `by_generator_name`/`by_operator`（Task 2）在 Task 4 GATE_DATA 文档引用一致 ✓
- `load_generators` 返回三元组（既有）在 `_cmd_probe` 解包 `_, inps, imgs` 一致 ✓

无不一致。

## 执行顺序
Task 1 → 2 → 3 → 4；每步 TDD，Task 4 跑全量 + 真实 probe 冒烟。

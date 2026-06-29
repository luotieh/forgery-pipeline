# 让管线产出全部闸门所需数据 — 设计文档（Spec）

- 日期：2026-06-29
- 关联：`docs/EXECUTION_CHECKLIST.md`（闸门 0–4）、`docs/PAPER_DESIGN.md`、`docs/PATCHES.md`
- 目标：补齐数据生成管线对闸门 0–4 的覆盖，使按 `EXECUTION_CHECKLIST.md` 能完成全部检查。

---

## 1. 目标与范围

### 1.1 背景与缺口
管线现状对闸门的覆盖：闸门 0（成对 real/edited+mask）、1（probe 强度网格带 `strength`）、2（probe 算子×族）、4（Test-A..F 划分）已覆盖。**缺口**：
- **闸门 3b**：probe 无「seen vs 留出生成器」标记，无法在受控 probe 上测跨生成器掉点。
- **闸门 4**：缺可验证的「按生成器均衡采样」统计。
- 小项：`init_timestep` 未填（PAPER §8 可选，便于直接读 t0）；`configs/probe.yaml` 被清单命令附录引用但未提供。

### 1.2 范围内（4 项）
A. probe 留出生成器：按 `holdout_generators` 给 probe 样本打 `split=train/test_b`。
B. img2img 样本填 `init_timestep = round(strength×1000)`。
C. `manifest.stats()` 增 `by_generator_name` / `by_operator`。
D. 新增 `configs/probe.yaml`，`probe` 子命令改为读该配置；统一命令文档。
E. 测试（probe split/`init_timestep`/stats/CLI）+ `docs/GATE_DATA.md`（闸门→产物→命令映射）。

### 1.3 非目标
- 不改 backend 抽象 / mock 实现 / 主 `run` 流程 / schema（复用 `split`，不加新字段）。
- 不实现闸门分析脚本（属 `gate_experiments/` 独立仓）。
- 不接真实模型。

### 1.4 约束
- 复用已有依赖，注释/文档中文、标识符 English，确定性。
- 向后兼容：旧 config（`pipeline.example.yaml`）仍能跑 `probe`（holdout 默认空 → 全 `split=train`）。

---

## 2. 详细设计

### 2.1 probe 留出生成器（A）— `builders/probe.py`
- `build_probe_strength(out_dir, bases, img2img_specs, strengths, backend, seed, holdout_generators=())`：
  每个样本 `split = "test_b" if spec.name in holdout else "train"`；
  `init_timestep = int(round(float(s)*1000))`（B）。
- `build_probe_operator(out_dir, bases, img2img_specs, inpainter_specs, operators, backend, seed, holdout_generators=())`：
  每个样本同样按 `spec.name` 设 `split`；img2img 分支再设 `init_timestep = int(round(0.6*1000)) = 600`。
- `run_probe(out_dir, *, n_base, strengths, operators, img2img_specs, inpainter_specs, holdout_generators=(), backend="mock", seed=0)`：
  `build_d0` 后对每个 base 设 `split="train"`；把 `holdout_generators` 透传给两个 builder。
- 常量 `_NUM_TRAIN_TIMESTEPS = 1000`。

> 留出默认 `{kandinsky-inpaint, sdxl-img2img}`（由 `configs/probe.yaml` 提供，非硬编码）。

### 2.2 均衡统计（C）— `manifest.py`
`stats()` 返回值新增两键（不破坏既有键）：
```python
"by_generator_name": dict(Counter(s.generator_name for s in samples if s.generator_name)),
"by_operator": dict(Counter(s.operator for s in samples if s.operator)),
```

### 2.3 probe 配置与 CLI（D）— `configs/probe.yaml` + `cli.py`
`configs/probe.yaml`：
```yaml
out_dir: data/probe
seed: 1234
backend: mock
generators_config: configs/generators.yaml
n_base: 40
strengths: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
operators: [img2img, inpaint, outpaint, object_replacement, background_editing]
holdout_generators: [kandinsky-inpaint, sdxl-img2img]
```
`_cmd_probe` 改为：`yaml.safe_load(config)` → `load_generators(data["generators_config"])` 取 `(gens, inps, imgs)` → 以 config 值（缺省回退默认）调用 `run_probe`，`--out`/`--n-base` CLI 可覆盖。对旧 config（无 `n_base`/`strengths`/`holdout_generators`）走默认值，`holdout_generators` 默认空。

### 2.4 文档（E）— `docs/GATE_DATA.md`
一张表：每个闸门（0–4）→ 用哪个产物（`gate1_strength.jsonl` / `gate2_operator.jsonl` / 主 `manifest.jsonl` 的 Test-A..F / probe `split=test_b`）→ 关键字段（`strength`/`operator`/`generator_family`/`init_timestep`/`postprocess_of`）→ 跑什么命令。README 快速开始加一行指向它。

---

## 3. 数据流（probe）
```
probe.yaml → _cmd_probe 读配置 → run_probe
  build_d0(bases, split=train)
  build_probe_strength(每强度 img2img；split 按 holdout；init_timestep=round(strength*1000))
  build_probe_operator(5 算子 × specs；split 按 holdout；img2img init_timestep=600)
  写 gate1_strength.jsonl / gate2_operator.jsonl / manifest.jsonl
  返回 stats（含 by_generator_name / by_operator / by_split）
```

## 4. 测试策略（`tests/test_probe.py` 新增）
- `run_probe` 小规模（n_base=2）：
  - 留出生成器样本 `split=="test_b"`，seen 样本 `split=="train"`；两类都非空。
  - img2img 样本 `init_timestep` 非空且 `≈ strength*1000`；inpaint 样本 `init_timestep` 为 None。
  - `gate1_strength.jsonl` 每行 `operator=="img2img"` 且带 `strength`；`gate2_operator.jsonl` 覆盖 5 算子。
- `manifest.stats`：返回含 `by_generator_name` 与 `by_operator`，计数正确。
- CLI：`main(["probe","--config","configs/probe.yaml","--out",tmp,"--n-base","2"])==0`，产出 manifest 且 `validate-manifest` 通过；其中存在 `split==test_b` 的样本。

验收：`pytest -q` 全绿；`forgery-pipeline probe --config configs/probe.yaml --out data/probe` 产出的 manifest 中，留出生成器样本标 `test_b`、img2img 带 `init_timestep`、`stats` 可见每生成器计数；`docs/GATE_DATA.md` 覆盖闸门 0–4。

---

## 5. 实施顺序
A(probe split+init_timestep) → C(stats) → D(probe.yaml+cli) → E(tests+GATE_DATA.md)。每步 TDD，最后全量 `pytest` + 真实 probe 冒烟。

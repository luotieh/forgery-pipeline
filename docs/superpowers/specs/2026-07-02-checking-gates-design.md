# checking/ — 闸门执行测试落地本仓 — 设计文档（Spec）

- 日期：2026-07-02
- 关联：`docs/EXECUTION_CHECKLIST.md`（闸门 0–4）、`docs/PAPER_DESIGN.md`、`docs/GATE_DATA.md`
- 目标：把清单的闸门 go/no-go 分析实验在本仓落地为 `./checking/` 代码，消费本管线产出的 probe/run manifest，端到端产出每个闸门的 VERDICT。

---

## 1. 目标与范围

### 1.1 目标
在 `./checking/` 实现闸门 0–3 的分析（+ Gate 4 评测轴骨架），读取本管线 `data/probe` 与 `data/run` 的 manifest，计算清单规定的指标与 **VERDICT**。信号提取可插拔：CPU 代理（本环境即跑）+ 真实 SD2（guarded stub，GPU）。

### 1.2 范围内
- `checking/` 包：`extractor / metrics / data / gate0 / gate1 / gate2 / gate3 / gate4_eval / run_gates` + `README.md`。
- `tests/test_checking.py`：metrics 单测 + 端到端在 mock probe/run 上跑通各闸门。
- 打包：把 `checking` 纳入可安装包，使 `python -m checking.run_gates` 与 `import checking` 可用。

### 1.3 非目标
- 不实现 Gate 4 的完整多任务模型与训练循环、SOTA baseline（仅评测轴接线）。
- 不实现真实 SD2 提取器本体（guarded stub + 参考骨架；需 diffusers+GPU）。
- 不改本管线数据生成代码（只消费其产物）。

### 1.4 关键约束（诚实边界）
- **CPU `MultiSigmaResidual` 是代理信号**：在 mock 数据上的 VERDICT 只验证「分析代码通路正确」，**非科学结论**。真实判定需 `extractor=real`（SD2）+ 真实扩散生成数据 + GPU。此话须写入 README 与 `report.json` 的 `caveat` 字段。
- sklearn-free：仅用 numpy/scipy/cv2/skimage（已装）；matplotlib 仅用于可选散点图（guarded import）。真实提取器走 `[real]` extra。
- 注释/文档中文、标识符 English、确定性（随机走显式 seed）。

---

## 2. 组件与接口

### 2.1 `checking/extractor.py`
```python
class ResidualExtractor(ABC):
    sigmas: list
    def residual_stack(self, image) -> np.ndarray   # (K,H,W) float[0,1] 每尺度残差图（抽象）
    def profile(self, image) -> np.ndarray           # (2K,) [每尺度 mean, 每尺度 std]
    def residual_map(self, image) -> np.ndarray      # (H,W) 各尺度均值，用于定位
```
- `MultiSigmaResidual(sigmas=(3,5,9,17,33))`：`residual_stack` = 对每个尺度 `k`，`recon=GaussianBlur(img, sigmaX=k)`，`resid=|img-recon|.mean(ch)/255`。多尺度「重建残差」CPU 代理，快、确定。
- `DiffusersSD2Residual(model_id, device, sigmas)`：guarded import diffusers/torch；构造即 `raise NotImplementedError("需 pip install .[real]…")`，附「VAE 编码→多 t 加噪→UNet ε̂→Tweedie 残差」参考骨架。
- `get_extractor(name)`：`"multisigma"`→`MultiSigmaResidual()`；`"real"`→`DiffusersSD2Residual()`（抛错含提示）。

### 2.2 `checking/metrics.py`（sklearn-free）
- `roc_auc(y_true, scores) -> float`：基于 `scipy.stats.rankdata`（Mann–Whitney），空类回退 0.5。
- `separability_auc(y_true, scores) -> float`：`max(roc_auc, 1-roc_auc)`（方向无关）。
- `balanced_accuracy(y_true, y_pred) -> float`。
- `spearman(a, b) -> float`：`scipy.stats.spearmanr`（退化返回 0.0）。
- `class NearestCentroid`：`fit(X,y)`（内部标准化）/`predict(X)`；空/单类稳健。
- `linear_fit_predict(Xtr, ytr, Xte) -> np.ndarray`：`np.linalg.lstsq` 最小二乘回归（strength 回归）。
- `pca_2d(X) -> np.ndarray (N,2)`：numpy SVD 降维（替代 t-SNE）。
- `group_split(keys, test_frac=0.4, seed=0) -> (train_idx, test_idx)`：按分组键（如底图/family）不重叠划分。

### 2.3 `checking/data.py`
- `load(manifest_path) -> list[Sample]`（`forgery_pipeline.manifest.read_jsonl`）。
- `image_of(root, s) -> np.ndarray`、`mask_of(root, s) -> np.ndarray|None`（`forgery_pipeline.image_io`）。
- `profiles(extractor, root, samples) -> (np.ndarray X, list[Sample])`：批量提取，跳过读失败。

### 2.4 闸门（每个返回 dict，含 `metrics` 与 `verdict`）
- `gate0.run(run_dir, extractor, max_n=200) -> dict`：
  检测 = `separability_auc(is_fake, 每图 residual_map.mean())`；定位 = 对有 mask 的 fake，逐图 `separability_auc(mask像素, residual_map)` 求均值。`verdict="PASS" if det≥0.6 and loc≥0.6 else "FAIL"`。
- `gate1.run(probe_dir, extractor, max_n=None) -> dict`：读 `gate1_strength.jsonl`；`profile` 特征；3 类桶（low `{0.1,0.2,0.3}` / mid `{0.4,0.5,0.6}` / high `{0.7,0.8,0.9}`）；按底图 `group_split`；多 σ：`NearestCentroid`→`balanced_accuracy`，`linear_fit_predict`→`spearman(pred,true)`；单 σ：仅用一个尺度的 mean 特征→`balanced_accuracy`。`verdict`：`PASS`(bal≥0.55 且 ρ≥0.30)/`WEAK`([0.45,0.55))/`FAIL`。
- `gate2.run(probe_dir, extractor, max_n=None) -> dict`：读 `gate2_operator.jsonl`；标签=operator；同模型（每个 family 内 `group_split` 训练/测试→bal_acc，跨 family 平均）；跨模型（train family A→test family B，**限两族共有的算子**，所有有序对平均）；`verdict`：`PASS`(same≥0.50 且 cross≥0.40)/`CONFOUND`(same 高且 cross<0.30)/`WEAK`(same<0.50)；`pca_2d`→散点 png（guarded matplotlib，缺则跳过）。
- `gate3.run(probe_dir, run_dir, extractor, max_n=None) -> dict`：(a) 多 σ 增量 = gate1 的 `multi_bal_acc - single_bal_acc`；(b) 跨生成器：probe 上 `split=train`(seen) 训练 operator 分类器，`split=test_b`(留出) 测试→bal_acc，`drop = seen_bal_acc - heldout_bal_acc`。`verdict="PASS" if 多σ增量>0 且 heldout_bal_acc>随机（=1/算子数）` 否则说明性结论（跨生成器崩→第二篇动机）。
- `gate4_eval.run(run_dir, extractor, max_n=None) -> dict`：Test-A..F 评测轴骨架——在 `split=train` 拟合检测器，对每个 `test_a..test_f` 报 detection `separability_auc`；`test_e` 额外报退化子集；`test_f`（全真实）报 false-positive-rate。**说明：这是评测轴接线，非论文模型**。

### 2.5 `checking/run_gates.py`（CLI）
```
python -m checking.run_gates --run data/run --probe data/probe [--extractor multisigma|real] [--max N] [--out checking/report.json]
```
依次跑 gate0(run)/gate1(probe)/gate2(probe)/gate3(probe,run)/gate4_eval(run)，打印每个 VERDICT，写 `report.json`（含各闸门 metrics + verdict + `caveat`）。`--extractor real` → 抛 stub 错误提示。

### 2.6 打包
`pyproject.toml` 的 `[tool.setuptools.packages.find]` 改为 `where=["src","."]`、`include=["forgery_pipeline*","checking*"]`，`pip install -e .` 后 `import checking` 与 `python -m checking.run_gates` 可用。

---

## 3. 数据流
```
data/probe/{gate1_strength,gate2_operator,manifest}.jsonl + data/run/manifest.jsonl
  → checking.data 载入 + extractor 提取 profile/residual_map
  → gate0/1/2/3/gate4_eval 各算指标 + VERDICT
  → run_gates 汇总打印 + report.json（含 caveat）
```

## 4. 测试策略（`tests/test_checking.py`）
- metrics 单测：`roc_auc` 完美可分=1.0、随机≈0.5；`balanced_accuracy` 已知；`spearman` 单调=1.0；`NearestCentroid` 线性可分数据准确；`pca_2d` 形状 (N,2)。
- extractor：`MultiSigmaResidual().profile(img)` 形状 (2K,)，`residual_map` 形状 (H,W) dtype float。
- 端到端：`run_probe`(mock, n_base=3) + `run_pipeline`(小 scales) 造 tmp 数据 → 每个 gate/gate4_eval 跑通，断言返回 dict 含 `verdict`（取值在允许集合）与 `metrics`（float 落在 [0,1]）。
- 验收：`pytest -q` 全绿；`python -m checking.run_gates --run data/run --probe data/probe` 打印 5 个 VERDICT 并写 `report.json`（含 caveat）。

## 5. 实施顺序
metrics → extractor → data → gate0 → gate1 → gate2 → gate3 → gate4_eval → run_gates(+打包) → README。每步 TDD，最后全量 + 真实数据冒烟。

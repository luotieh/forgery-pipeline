# checking/ P0 优化 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 落地 `docs/gate_results_analysis_2026-07-02.md` 的 P0 优化（不需 GPU、CPU 上可测）：区域/分位特征、gate1 回归分桶、gate4 共享真实负样本池 + 局部检测分数、bootstrap 置信区间。

**Architecture:** 改 `checking/` 的 `metrics`（加 bootstrap_ci）、`extractor`（profile 加区域/分位特征 + detection_score）、`gate1`（回归→分桶取优 + ba CI）、`gate0/gate4`（用 detection_score；gate4 共享真实负样本池 + per-split AUC CI + 方向性 FPR）。全部 sklearn-free、确定性、CPU 可跑。

**Tech Stack:** numpy、scipy、cv2（已装）。

## Global Constraints

- sklearn-free；确定性（bootstrap 走显式 seed）；注释中文/标识符 English。
- 不改数据生成管线，只改 `checking/`。诚实边界不变（mock+代理 → VERDICT 仅验证通路）。
- 特征维度变化会波及 gate1/2/3（都用 `profile`）——它们的既有测试只断言范围，应仍通过；`test_checking_extractor` 的 profile 形状断言需更新。

---

## File Structure

| 文件 | 改动 |
|---|---|
| `checking/metrics.py` | 加 `bootstrap_ci(a,b,metric,n_boot,seed,alpha)` |
| `checking/extractor.py` | `profile` 加区域+分位特征；加 `_region_descriptors`、`detection_score` |
| `checking/gate1.py` | 回归→分桶取优 ba + `ba_ci` + `direct_acc`/`regression_bucket_acc` |
| `checking/gate0.py` | 检测用 `detection_score` |
| `checking/gate4_eval.py` | 共享真实负样本池 + `detection_score` + per-split `auc_ci` + 方向性 FPR |
| `tests/test_checking_metrics.py` | 加 bootstrap_ci 测试 |
| `tests/test_checking_extractor.py` | 更新 profile 形状 + detection_score 测试 |
| `docs/gate_results_analysis_2026-07-02.md` | 追加「P0 优化后复测」小节 |

---

## Task 1: metrics.bootstrap_ci

**Files:** Modify `checking/metrics.py`, `tests/test_checking_metrics.py`

**Interfaces:**
- Produces：`bootstrap_ci(a, b, metric, n_boot=200, seed=0, alpha=0.05) -> [lo,hi]`（对 `metric(a[idx],b[idx])` 自助；n<4 或无值 → `[None,None]`）

- [ ] **Step 1: 追加失败测试** `tests/test_checking_metrics.py`

```python
def test_bootstrap_ci():
    from checking.metrics import bootstrap_ci, separability_auc
    y = [0, 0, 0, 0, 1, 1, 1, 1]
    s = [0.1, 0.2, 0.15, 0.05, 0.9, 0.8, 0.85, 0.95]
    lo, hi = bootstrap_ci(y, s, separability_auc, n_boot=100, seed=0)
    assert 0.5 <= lo <= hi <= 1.0
    assert bootstrap_ci([0, 1], [0.1, 0.9], separability_auc) == [None, None]
```

- [ ] **Step 2: 运行确认失败** → FAIL

- [ ] **Step 3: 追加到 `checking/metrics.py` 末尾**

```python
def bootstrap_ci(a, b, metric, n_boot: int = 200, seed: int = 0, alpha: float = 0.05):
    """对 metric(a[idx], b[idx]) 做自助法置信区间；样本<4 或无值返回 [None, None]。"""
    a = np.asarray(a, dtype=object); b = np.asarray(b, dtype=object)
    n = len(a)
    if n < 4:
        return [None, None]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            vals.append(float(metric(list(a[idx]), list(b[idx]))))
        except Exception:
            continue
    if not vals:
        return [None, None]
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return [round(float(lo), 4), round(float(hi), 4)]
```

- [ ] **Step 4: 运行确认通过** → PASS

- [ ] **Step 5: 提交**

```bash
git add checking/metrics.py tests/test_checking_metrics.py
git commit -m "feat(checking): metrics.bootstrap_ci（小样本置信区间护栏）"
```

---

## Task 2: extractor 区域/分位特征 + detection_score

**Files:** Modify `checking/extractor.py`, `tests/test_checking_extractor.py`

**Interfaces:**
- Produces：`ResidualExtractor.profile` 返回 `2K+7` 维（每尺度 mean/std + agg 的 p10/p50/p90 + 区域描述子 cy/cx/spread/border_center_ratio）；新增 `detection_score(image)->float`（残差图 top-decile 均值，对局部编辑更灵敏）；模块函数 `_region_descriptors(agg)->np.ndarray(4,)`

- [ ] **Step 1: 更新/追加测试** `tests/test_checking_extractor.py`（替换形状断言 + 加 detection_score）

```python
import numpy as np
import pytest
from checking.extractor import MultiSigmaResidual, get_extractor


def test_multisigma_profile_and_map_shapes():
    ext = MultiSigmaResidual(sigmas=(3, 5, 9))
    img = np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    assert ext.profile(img).shape == (13,)   # 2*3 + 3 分位 + 4 区域
    rm = ext.residual_map(img)
    assert rm.shape == (64, 64) and rm.dtype == np.float32
    assert ext.residual_stack(img).shape == (3, 64, 64)
    ds = ext.detection_score(img)
    assert isinstance(ds, float) and ds == ds  # 有限


def test_get_extractor_and_real_stub():
    assert isinstance(get_extractor("multisigma"), MultiSigmaResidual)
    with pytest.raises(NotImplementedError):
        get_extractor("real")
```

- [ ] **Step 2: 运行确认失败** → FAIL（profile 形状仍是 (6,)）

- [ ] **Step 3: 改 `checking/extractor.py`** —— 在 imports 后加模块函数，并替换基类 `profile` + 加 `detection_score`

模块函数（加在 `ResidualExtractor` 定义之前）：

```python
def _region_descriptors(agg: np.ndarray) -> np.ndarray:
    """高残差区（agg≥p90）的几何：归一化重心 cy/cx、扩散、边界带/中心区残差比。"""
    H, W = agg.shape
    thr = np.quantile(agg, 0.9)
    ys, xs = np.nonzero(agg >= thr)
    if len(ys) == 0:
        cy = cx = 0.5; spread = 0.0
    else:
        cy = float(ys.mean()) / H; cx = float(xs.mean()) / W
        spread = float(np.sqrt(((ys / H - cy) ** 2 + (xs / W - cx) ** 2).mean()))
    b = max(1, min(H, W) // 8)
    border = np.concatenate([agg[:b].ravel(), agg[-b:].ravel(),
                             agg[:, :b].ravel(), agg[:, -b:].ravel()])
    center = agg[b:H - b, b:W - b]
    bc = float(border.mean() / (center.mean() + 1e-8)) if center.size else 1.0
    return np.array([cy, cx, spread, bc], np.float32)
```

基类内替换 `profile` 并加 `detection_score`（`residual_map` 保持不变）：

```python
    def profile(self, image: np.ndarray) -> np.ndarray:
        rs = self.residual_stack(image)
        per_scale = np.concatenate([rs.mean(axis=(1, 2)), rs.std(axis=(1, 2))])
        agg = rs.mean(axis=0)
        q = np.quantile(agg, [0.1, 0.5, 0.9])
        return np.concatenate([per_scale, q, _region_descriptors(agg)]).astype(np.float32)

    def detection_score(self, image: np.ndarray) -> float:
        """残差图 top-decile 均值：对局部编辑比全局均值更灵敏。"""
        rm = self.residual_map(image)
        thr = float(np.quantile(rm, 0.90))
        top = rm[rm >= thr]
        return float(top.mean()) if top.size else float(rm.mean())
```

- [ ] **Step 4: 运行确认通过** → PASS

- [ ] **Step 5: 提交**

```bash
git add checking/extractor.py tests/test_checking_extractor.py
git commit -m "feat(checking): profile 加区域/分位特征 + detection_score（局部灵敏）"
```

---

## Task 3: gate1 回归分桶 + gate0 detection_score

**Files:** Modify `checking/gate1.py`, `checking/gate0.py`

**Interfaces:**
- Consumes：`metrics.bootstrap_ci`、`extractor.detection_score`
- Produces：`gate1.run` 的 `balanced_accuracy` 取 max(直接分类, 回归分桶)，metrics 增 `ba_ci`/`direct_acc`/`regression_bucket_acc`；`gate0` 检测分数改 `detection_score`

- [ ] **Step 1: 改 `checking/gate1.py`** —— import 增 `bootstrap_ci`，替换 `run` 内计算段

顶部 import 改为：

```python
from checking.metrics import (NearestCentroid, balanced_accuracy, spearman,
                              linear_fit_predict, group_split, bootstrap_ci)
```

把 `tr, te = group_split(...)` 之后到 `verdict = ...` 之前整段替换为：

```python
    ytr = [buckets[i] for i in tr]; yte = [buckets[i] for i in te]
    ok = bool(tr) and bool(te) and len(set(ytr)) >= 2
    direct_pred = NearestCentroid().fit(X[tr], ytr).predict(X[te]) if ok else []
    direct = balanced_accuracy(yte, direct_pred) if direct_pred else 0.0
    reg_pred = linear_fit_predict(X[tr], strengths[tr], X[te]) if tr and te else np.array([])
    reg_buckets = [_bucket(v) for v in reg_pred]
    reg_ba = balanced_accuracy(yte, reg_buckets) if reg_buckets else 0.0
    if reg_ba >= direct:
        multi, best_pred = reg_ba, reg_buckets
    else:
        multi, best_pred = direct, list(direct_pred)
    ba_ci = bootstrap_ci(yte, best_pred, balanced_accuracy) if best_pred else [None, None]
    rho = spearman(strengths[te], reg_pred) if len(reg_pred) else 0.0
    Xs = X[:, :1]
    single = (balanced_accuracy(yte, NearestCentroid().fit(Xs[tr], ytr).predict(Xs[te]))
              if ok else 0.0)
    verdict = ("PASS" if multi >= 0.55 and rho >= 0.30
               else "WEAK" if multi >= 0.45 else "FAIL")
    return {"gate": 1,
            "metrics": {"balanced_accuracy": round(multi, 4), "ba_ci": ba_ci,
                        "direct_acc": round(direct, 4),
                        "regression_bucket_acc": round(reg_ba, 4),
                        "spearman_rho": round(rho, 4), "multi_sigma_acc": round(multi, 4),
                        "single_sigma_acc": round(single, 4), "n": len(kept)},
            "verdict": verdict}
```

- [ ] **Step 2: 改 `checking/gate0.py`** —— 检测分数用 `detection_score`

把循环内：

```python
        rmap = extractor.residual_map(img)
        det_y.append(int(s.is_fake)); det_s.append(float(rmap.mean()))
```

改为：

```python
        rmap = extractor.residual_map(img)
        det_y.append(int(s.is_fake)); det_s.append(extractor.detection_score(img))
```

- [ ] **Step 3: 运行确认通过**

Run: `pytest tests/test_checking_gates.py -q` → PASS（既有断言仍满足）

- [ ] **Step 4: 提交**

```bash
git add checking/gate1.py checking/gate0.py
git commit -m "feat(checking): gate1 回归分桶取优+ba_ci；gate0 局部检测分数"
```

---

## Task 4: gate4 共享真实负样本池 + 局部分数 + CI

**Files:** Modify `checking/gate4_eval.py`

**Interfaces:**
- Consumes：`extractor.detection_score`、`metrics.separability_auc/bootstrap_ci`
- Produces：`gate4_eval.run` 每个 test split 用「共享训练真实负样本池」算 detection AUC（test_e/f 不再无定义）+ `auc_ci`；方向性 `test_f_fpr`

- [ ] **Step 1: 整体替换 `checking/gate4_eval.py`**

```python
"""闸门 4 评测轴骨架：Test-A..F 用共享真实负样本池 + 局部检测分数（非论文模型）。"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from checking import data
from checking.metrics import separability_auc, bootstrap_ci

_TESTS = ["test_a", "test_b", "test_c", "test_d", "test_e"]


def run(run_dir, extractor, max_n=None) -> dict:
    run_dir = Path(run_dir)
    samples = data.load(run_dir / "manifest.jsonl")
    if max_n:
        samples = samples[:max_n]
    rows = []
    for s in samples:
        try:
            img = data.image_of(run_dir, s)
        except Exception:
            continue
        rows.append((s.split, int(s.is_fake), extractor.detection_score(img)))
    train_reals = [sc for sp, f, sc in rows if sp == "train" and f == 0]
    train_fakes = [sc for sp, f, sc in rows if sp == "train" and f == 1]
    per = {}
    for sp in _TESTS:
        fakes = [sc for s2, f, sc in rows if s2 == sp and f == 1]
        reals_in = [sc for s2, f, sc in rows if s2 == sp and f == 0]
        neg = reals_in + train_reals  # 共享真实负样本池
        if fakes and neg:
            y = [1] * len(fakes) + [0] * len(neg); sc = fakes + neg
            per[sp] = {"detection_auc": round(separability_auc(y, sc), 4),
                       "auc_ci": bootstrap_ci(y, sc, separability_auc),
                       "n_fake": len(fakes), "n_neg": len(neg)}
        else:
            per[sp] = {"detection_auc": None, "n_fake": len(fakes)}
    hi_is_fake = (np.median(train_fakes) >= np.median(train_reals)
                  if train_fakes and train_reals else True)
    tf_reals = [sc for sp, f, sc in rows if sp == "test_f" and f == 0]
    fpr = None
    if train_reals and tf_reals:
        thr = float(np.quantile(train_reals, 0.95 if hi_is_fake else 0.05))
        pred = [(sc > thr) if hi_is_fake else (sc < thr) for sc in tf_reals]
        fpr = round(float(np.mean(pred)), 4)
    return {"gate": 4,
            "metrics": {"per_split": per, "test_f_fpr": fpr,
                        "detection_direction": "high=fake" if hi_is_fake else "low=fake"},
            "verdict": "EVAL-ONLY",
            "note": "共享真实负样本池 + 局部检测分数；完整多任务模型/训练/SOTA baseline 属论文系统"}
```

- [ ] **Step 2: 运行确认通过**

Run: `pytest tests/test_checking_gates.py::test_gate4_eval -q` → PASS

- [ ] **Step 3: 提交**

```bash
git add checking/gate4_eval.py
git commit -m "feat(checking): gate4 共享真实负样本池 + 局部检测分数 + AUC CI + 方向性 FPR"
```

---

## Task 5: 全量测试 + 复测 + 分析文档追加

**Files:** Modify `docs/gate_results_analysis_2026-07-02.md`

- [ ] **Step 1: 全量测试**

Run: `pytest -q`
Expected: 全绿（现 101 + bootstrap_ci 测试）。

- [ ] **Step 2: 真实数据复测**

```bash
forgery-pipeline run   --config configs/pipeline.example.yaml --out data/run   >/dev/null
forgery-pipeline probe --config configs/probe.yaml            --out data/probe >/dev/null
python -m checking.run_gates --run data/run --probe data/probe
```
记录 gate1–4 的新 VERDICT 与关键指标（尤其 gate1 的 `balanced_accuracy`/`regression_bucket_acc`、gate2 的 same/cross、gate4 各 split AUC 与 CI）。

- [ ] **Step 3: 在 `docs/gate_results_analysis_2026-07-02.md` 末尾追加「## 6. P0 优化后复测」**

内容（按 Step 2 实测填数）：优化前→后对照表（gate1 ba、gate2 same/cross、gate4 test_e 由 null→有值、各 CI），并说明「哪些改善是特征/口径带来的真实改进、哪些仍受 mock 限制」。诚实边界重申：仍需真实 SD2+GPU 出科学判定。

- [ ] **Step 4: 提交**

```bash
git add docs/gate_results_analysis_2026-07-02.md
git commit -m "docs: 追加 P0 优化后复测对照"
```

---

## Self-Review

**1. 覆盖 P0：** ①gate1 回归分桶→Task 3；②extractor 区域/分位特征→Task 2；③gate2/3 靠新特征自动受益（无结构改，Task 5 复测验证）；④gate4 共享池+局部分数→Task 4；⑤bootstrap CI→Task 1（gate1 ba_ci Task 3、gate4 auc_ci Task 4）。规模是配置旋钮（复测可用现规模，或提 n_base，Task 5 说明）。无缺口。

**2. Placeholder scan：** 各步含完整代码/命令；Task 5 Step 3 复测小节按实测填数（非占位，是执行时记录真实数据）。

**3. Type consistency：**
- `bootstrap_ci(a,b,metric,...)`（Task 1）被 gate1（Task 3）、gate4（Task 4）以 `(labels,preds,balanced_accuracy)` 与 `(y,scores,separability_auc)` 调用一致 ✓
- `detection_score(image)->float`（Task 2）被 gate0（Task 3）、gate4（Task 4）调用一致 ✓
- `profile` 维度变化只影响形状断言（Task 2 已更新）；gate1/2/3 用 `X` 不假设维度 ✓
- gate1 返回 `multi_sigma_acc/single_sigma_acc`（Task 3）被 gate3 读取，键名不变 ✓

无不一致。

## 执行顺序
Task 1 → 2 → 3 → 4 → 5；每步 TDD，Task 5 全量 + 复测 + 文档。

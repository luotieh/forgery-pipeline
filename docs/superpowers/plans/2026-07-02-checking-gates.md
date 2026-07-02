# checking/ 闸门执行测试 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `./checking/` 落地闸门 0–3 分析 + Gate 4 评测轴骨架，消费本管线 `data/probe`/`data/run` manifest，端到端产出每个闸门的 VERDICT，CPU 即跑（多尺度残差代理），真实 SD2 提取器留 guarded stub。

**Architecture:** `checking/` 顶层包，`python -m checking.run_gates` 运行、`conftest.py` 保证测试可 `import checking`。可插拔 `ResidualExtractor`（`MultiSigmaResidual` CPU 代理 + `DiffusersSD2Residual` stub）→ 每图 profile/residual_map → 各 gate 用 sklearn-free 指标算 VERDICT。

**Tech Stack:** numpy、scipy、opencv、skimage（已装）；matplotlib 可选（guarded）；复用 `forgery_pipeline.manifest/image_io`。

## Global Constraints

- **sklearn-free**：仅 numpy/scipy/cv2；matplotlib 仅可选散点（guarded import，缺则跳过）；无新增必需依赖。真实 SD2 走 `[real]` extra（stub）。
- `checking/` 可导入靠仓库根 `conftest.py`（`sys.path` 注入）+ `python -m checking.run_gates`（cwd 在仓库根）；**不改 pyproject**（避免动到已用的 -e 安装）。
- **诚实边界**：`multisigma` 是 CPU 代理，mock 数据上的 VERDICT 只验证代码通路、非科学结论——写入 `checking/README.md` 与 `report.json` 的 `caveat`。
- 注释/文档中文、标识符 English、确定性（随机走显式 seed）。
- 产物默认写 `data/`（已 gitignore）：`--out data/checking_report.json`、散点 `data/gate2_pca.png`。

---

## File Structure

| 文件 | 职责 |
|---|---|
| `conftest.py`（仓库根） | 注入仓库根到 sys.path，使测试可 `import checking` |
| `checking/__init__.py` | 包标记 |
| `checking/metrics.py` | roc_auc/separability_auc/balanced_accuracy/spearman/NearestCentroid/linear_fit_predict/pca_2d/group_split |
| `checking/extractor.py` | ResidualExtractor + MultiSigmaResidual + DiffusersSD2Residual(stub) + get_extractor |
| `checking/data.py` | load/image_of/mask_of/profiles（读本管线 manifest） |
| `checking/gate0.py` | 检测+定位 AUC |
| `checking/gate1.py` | 强度桶分类 + 回归 Spearman + 多σ vs 单σ |
| `checking/gate2.py` | 同模型/跨模型算子可分 + PCA 散点 |
| `checking/gate3.py` | 多σ增量 + 跨生成器掉点 |
| `checking/gate4_eval.py` | Test-A..F 评测轴骨架 |
| `checking/run_gates.py` | CLI：跑全部 + report.json |
| `checking/README.md` | 用法 + 诚实边界 |
| `tests/test_checking_metrics.py` | metrics 单测 |
| `tests/test_checking_extractor.py` | extractor/data 单测 |
| `tests/test_checking_gates.py` | 端到端各 gate + CLI（模块级 fixture 建一次数据） |

---

## Task 1: 包骨架 + metrics（sklearn-free）

**Files:**
- Create: `conftest.py`, `checking/__init__.py`, `checking/metrics.py`, `tests/test_checking_metrics.py`

**Interfaces:**
- Produces：`roc_auc(y,scores)->float`、`separability_auc(y,scores)->float`、`balanced_accuracy(y_true,y_pred)->float`、`spearman(a,b)->float`、`class NearestCentroid{fit(X,y);predict(X)->list}`、`linear_fit_predict(Xtr,ytr,Xte)->np.ndarray`、`pca_2d(X)->np.ndarray(N,2)`、`group_split(keys,test_frac=0.4,seed=0)->(list,list)`

- [ ] **Step 1: 写失败测试** `tests/test_checking_metrics.py`

```python
import numpy as np
from checking.metrics import (roc_auc, separability_auc, balanced_accuracy,
                              spearman, NearestCentroid, linear_fit_predict,
                              pca_2d, group_split)


def test_roc_and_separability():
    assert abs(roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) - 1.0) < 1e-9
    assert abs(roc_auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) - 0.0) < 1e-9
    assert abs(separability_auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) - 1.0) < 1e-9
    assert roc_auc([1, 1], [0.5, 0.6]) == 0.5  # 单类回退


def test_balanced_accuracy_and_spearman():
    assert balanced_accuracy([0, 0, 1, 1], [0, 0, 1, 1]) == 1.0
    assert abs(balanced_accuracy([0, 0, 1, 1], [0, 0, 0, 0]) - 0.5) < 1e-9
    assert spearman([1, 2, 3, 4], [1, 2, 3, 4]) > 0.99
    assert spearman([1, 1, 1], [1, 2, 3]) == 0.0  # 退化


def test_nearest_centroid_separable():
    X = np.array([[0, 0], [0.1, 0], [5, 5], [5.1, 5]])
    clf = NearestCentroid().fit(X, ["a", "a", "b", "b"])
    assert clf.predict([[0.05, 0], [5.05, 5]]) == ["a", "b"]


def test_linear_and_pca_and_split():
    y = linear_fit_predict([[0.0], [1.0], [2.0]], [0.0, 1.0, 2.0], [[3.0]])
    assert abs(y[0] - 3.0) < 1e-6
    assert pca_2d(np.random.default_rng(0).random((8, 5))).shape == (8, 2)
    tr, te = group_split(["g1", "g1", "g2", "g3"], test_frac=0.5, seed=0)
    assert set(tr) & set(te) == set() and len(tr) + len(te) == 4
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_checking_metrics.py -q`
Expected: FAIL（`ModuleNotFoundError: No module named 'checking'`）

- [ ] **Step 3: 写 `conftest.py`（仓库根）**

```python
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
```

- [ ] **Step 4: 写 `checking/__init__.py`**

```python
"""闸门执行测试（消费 forgery-pipeline 产出的受控数据）。"""
```

- [ ] **Step 5: 写 `checking/metrics.py`**

```python
"""闸门分析用的 sklearn-free 指标与轻量分类器。"""
from __future__ import annotations
import numpy as np
from scipy.stats import rankdata, spearmanr


def roc_auc(y_true, scores) -> float:
    y = np.asarray(y_true); s = np.asarray(scores, float)
    pos = int((y == 1).sum()); neg = int((y == 0).sum())
    if pos == 0 or neg == 0:
        return 0.5
    r = rankdata(s)
    return float((r[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def separability_auc(y_true, scores) -> float:
    a = roc_auc(y_true, scores)
    return max(a, 1.0 - a)


def balanced_accuracy(y_true, y_pred) -> float:
    y = np.asarray(y_true, dtype=object); p = np.asarray(y_pred, dtype=object)
    accs = [float((p[y == c] == c).mean()) for c in np.unique(y) if (y == c).any()]
    return float(np.mean(accs)) if accs else 0.0


def spearman(a, b) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 2 or len(np.unique(a)) < 2 or len(np.unique(b)) < 2:
        return 0.0
    rho = spearmanr(a, b).correlation
    return float(rho) if rho == rho else 0.0


class NearestCentroid:
    def fit(self, X, y):
        X = np.asarray(X, float)
        self.mu = X.mean(0); self.sd = X.std(0) + 1e-8
        Xs = (X - self.mu) / self.sd
        y = list(y)
        self.classes = sorted(set(y))
        ay = np.array(y, dtype=object)
        self.cent = {c: Xs[ay == c].mean(0) for c in self.classes}
        return self

    def predict(self, X):
        Xs = (np.asarray(X, float) - self.mu) / self.sd
        return [min(self.classes, key=lambda c: float(np.linalg.norm(x - self.cent[c])))
                for x in Xs]


def linear_fit_predict(Xtr, ytr, Xte) -> np.ndarray:
    Xtr = np.asarray(Xtr, float); Xte = np.asarray(Xte, float)
    A = np.hstack([Xtr, np.ones((len(Xtr), 1))])
    coef, *_ = np.linalg.lstsq(A, np.asarray(ytr, float), rcond=None)
    return np.hstack([Xte, np.ones((len(Xte), 1))]) @ coef


def pca_2d(X) -> np.ndarray:
    X = np.asarray(X, float)
    Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:2].T


def group_split(keys, test_frac: float = 0.4, seed: int = 0):
    keys = list(keys)
    uniq = sorted(set(keys))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq))
    n_test = max(1, int(round(len(uniq) * test_frac)))
    test_groups = {uniq[i] for i in perm[:n_test]}
    train_idx = [i for i, k in enumerate(keys) if k not in test_groups]
    test_idx = [i for i, k in enumerate(keys) if k in test_groups]
    return train_idx, test_idx
```

- [ ] **Step 6: 运行确认通过**

Run: `pytest tests/test_checking_metrics.py -q` → PASS

- [ ] **Step 7: 提交**

```bash
git add conftest.py checking/__init__.py checking/metrics.py tests/test_checking_metrics.py
git commit -m "feat(checking): 包骨架 + sklearn-free 指标（roc/ba/spearman/NC/pca/split）"
```

---

## Task 2: extractor + data

**Files:**
- Create: `checking/extractor.py`, `checking/data.py`, `tests/test_checking_extractor.py`

**Interfaces:**
- Produces：
  - `ResidualExtractor.residual_stack(img)->(K,H,W)` / `.profile(img)->(2K,)` / `.residual_map(img)->(H,W)`
  - `MultiSigmaResidual(sigmas=(3,5,9,17,33))`、`DiffusersSD2Residual(...)`(构造即抛 NotImplementedError)、`get_extractor(name)->ResidualExtractor`
  - `data.load(path)->list[Sample]`、`data.image_of(root,s)`、`data.mask_of(root,s)`、`data.profiles(extractor,root,samples)->(X:(N,D),kept)`

- [ ] **Step 1: 写失败测试** `tests/test_checking_extractor.py`

```python
import numpy as np
import pytest
from checking.extractor import MultiSigmaResidual, get_extractor


def test_multisigma_profile_and_map_shapes():
    ext = MultiSigmaResidual(sigmas=(3, 5, 9))
    img = np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    assert ext.profile(img).shape == (6,)          # 3 尺度 × [mean,std]
    rm = ext.residual_map(img)
    assert rm.shape == (64, 64) and rm.dtype == np.float32
    assert ext.residual_stack(img).shape == (3, 64, 64)


def test_get_extractor_and_real_stub():
    assert isinstance(get_extractor("multisigma"), MultiSigmaResidual)
    with pytest.raises(NotImplementedError):
        get_extractor("real")
```

- [ ] **Step 2: 运行确认失败** → FAIL

- [ ] **Step 3: 写 `checking/extractor.py`**

```python
"""闸门信号提取器：CPU 多尺度残差代理 + 真实 SD2 骨架。"""
from __future__ import annotations
from abc import ABC, abstractmethod
import cv2
import numpy as np


class ResidualExtractor(ABC):
    sigmas: list

    @abstractmethod
    def residual_stack(self, image: np.ndarray) -> np.ndarray:
        """返回 (K,H,W) float[0,1] 每尺度残差图。"""

    def profile(self, image: np.ndarray) -> np.ndarray:
        rs = self.residual_stack(image)
        return np.concatenate([rs.mean(axis=(1, 2)), rs.std(axis=(1, 2))]).astype(np.float32)

    def residual_map(self, image: np.ndarray) -> np.ndarray:
        return self.residual_stack(image).mean(axis=0).astype(np.float32)


class MultiSigmaResidual(ResidualExtractor):
    """多尺度「高斯重建残差」CPU 代理（快、确定）。"""
    def __init__(self, sigmas=(3, 5, 9, 17, 33)):
        self.sigmas = list(sigmas)

    def residual_stack(self, image: np.ndarray) -> np.ndarray:
        g = image.astype(np.float32)
        maps = []
        for k in self.sigmas:
            recon = cv2.GaussianBlur(g, (0, 0), sigmaX=float(k))
            maps.append(np.abs(g - recon).mean(axis=2) / 255.0)
        return np.stack(maps).astype(np.float32)


class DiffusersSD2Residual(ResidualExtractor):
    """真实多 σ Tweedie 残差骨架（冻结 SD2）。需 `pip install .[real]` + GPU。"""
    def __init__(self, model_id: str = "stabilityai/stable-diffusion-2-base",
                 device: str = "cuda", sigmas=(0.1, 0.2, 0.4, 0.6, 0.8)):
        self.sigmas = list(sigmas)
        try:
            import torch  # noqa: F401
            import diffusers  # noqa: F401
        except ImportError as e:
            raise NotImplementedError(
                "真实 SD2 提取器未启用：请 `pip install .[real]`（torch/diffusers）并提供 GPU。") from e
        raise NotImplementedError(
            "参考骨架：VAE 编码 z0 → 多 t 加噪 z_t → UNet ε̂ → r_ε(t)=‖ε−ε̂‖²、"
            "一步反演 ẑ0 → r_x(t)=‖z0−ẑ0‖²，堆叠成 residual_stack。")

    def residual_stack(self, image):
        raise NotImplementedError


def get_extractor(name: str = "multisigma") -> ResidualExtractor:
    if name == "multisigma":
        return MultiSigmaResidual()
    if name == "real":
        return DiffusersSD2Residual()
    raise ValueError(f"未知 extractor: {name!r}（可选 multisigma / real）")
```

- [ ] **Step 4: 写 `checking/data.py`**

```python
"""读取本管线 manifest 并批量提取特征。"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from forgery_pipeline import manifest, image_io


def load(manifest_path) -> list:
    return manifest.read_jsonl(manifest_path)


def image_of(root, s) -> np.ndarray:
    return image_io.load_image(Path(root) / s.image_path)


def mask_of(root, s):
    if not s.mask_path:
        return None
    return image_io.load_mask(Path(root) / s.mask_path)


def profiles(extractor, root, samples):
    """返回 (X:(N,D) float, kept:list[Sample])，跳过读失败的样本。"""
    X, kept = [], []
    for s in samples:
        try:
            img = image_of(root, s)
        except Exception:
            continue
        X.append(extractor.profile(img)); kept.append(s)
    return (np.array(X, float) if X else np.zeros((0, 1))), kept
```

- [ ] **Step 5: 运行确认通过**

Run: `pytest tests/test_checking_extractor.py -q` → PASS

- [ ] **Step 6: 提交**

```bash
git add checking/extractor.py checking/data.py tests/test_checking_extractor.py
git commit -m "feat(checking): 可插拔残差提取器（multisigma 代理 + SD2 stub）+ data 载入"
```

---

## Task 3: gate0 + gate1（含端到端 fixture）

**Files:**
- Create: `checking/gate0.py`, `checking/gate1.py`, `tests/test_checking_gates.py`

**Interfaces:**
- Consumes：`data.*`、`metrics.*`、`extractor.MultiSigmaResidual`
- Produces：`gate0.run(run_dir, extractor, max_n=200)->dict`、`gate1.run(probe_dir, extractor, max_n=None)->dict`（均含 `metrics` 与 `verdict`）

- [ ] **Step 1: 写失败测试** `tests/test_checking_gates.py`

```python
import dataclasses
import pytest
from forgery_pipeline.builders.probe import run_probe
from forgery_pipeline.config import GeneratorSpec, load_config, StageScales
from forgery_pipeline.pipeline import run_pipeline
from checking.extractor import MultiSigmaResidual

_I = [GeneratorSpec("sd-img2img", "diffusion", "img2img"),
      GeneratorSpec("sdxl-img2img", "diffusion-sdxl", "img2img")]
_P = [GeneratorSpec("sd-inpaint", "diffusion", "inpaint"),
      GeneratorSpec("kandinsky-inpaint", "kandinsky", "inpaint")]
_OPS = ["img2img", "inpaint", "outpaint", "object_replacement", "background_editing"]


@pytest.fixture(scope="module")
def dd(tmp_path_factory):
    root = tmp_path_factory.mktemp("gates")
    probe, run = root / "probe", root / "run"
    run_probe(probe, n_base=3, strengths=[0.2, 0.5, 0.8], operators=_OPS,
              img2img_specs=_I, inpainter_specs=_P,
              holdout_generators={"sdxl-img2img", "kandinsky-inpaint"}, seed=0)
    cfg = load_config("configs/pipeline.example.yaml")
    cfg = dataclasses.replace(cfg, out_dir=str(run),
                              scales=StageScales(d0=12, d1_per_generator=1, d2=8, d3=4, d4=3))
    run_pipeline(cfg)
    return {"probe": probe, "run": run, "ext": MultiSigmaResidual()}


def test_gate0(dd):
    from checking import gate0
    r = gate0.run(dd["run"], dd["ext"])
    assert r["verdict"] in {"PASS", "FAIL"}
    assert 0.0 <= r["metrics"]["detection_auc"] <= 1.0
    assert 0.0 <= r["metrics"]["localization_auc"] <= 1.0


def test_gate1(dd):
    from checking import gate1
    r = gate1.run(dd["probe"], dd["ext"])
    assert r["verdict"] in {"PASS", "WEAK", "FAIL"}
    m = r["metrics"]
    assert 0.0 <= m["balanced_accuracy"] <= 1.0
    assert -1.0 <= m["spearman_rho"] <= 1.0
    assert "single_sigma_acc" in m
```

- [ ] **Step 2: 运行确认失败** → FAIL

- [ ] **Step 3: 写 `checking/gate0.py`**

```python
"""闸门 0：残差能否分开 真实 vs 编辑区（检测 + 定位）。"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from checking import data
from checking.metrics import separability_auc


def run(run_dir, extractor, max_n: int = 200) -> dict:
    run_dir = Path(run_dir)
    samples = data.load(run_dir / "manifest.jsonl")[:max_n]
    det_y, det_s, loc = [], [], []
    for s in samples:
        try:
            img = data.image_of(run_dir, s)
        except Exception:
            continue
        rmap = extractor.residual_map(img)
        det_y.append(int(s.is_fake)); det_s.append(float(rmap.mean()))
        m = data.mask_of(run_dir, s)
        if m is not None and m.shape == rmap.shape and (m > 127).any() and (m <= 127).any():
            loc.append(separability_auc((m > 127).astype(int).ravel(), rmap.ravel()))
    det = separability_auc(det_y, det_s) if det_y else 0.5
    loc_auc = float(np.mean(loc)) if loc else 0.5
    return {"gate": 0,
            "metrics": {"detection_auc": round(det, 4),
                        "localization_auc": round(loc_auc, 4),
                        "n_localization": len(loc)},
            "verdict": "PASS" if det >= 0.6 and loc_auc >= 0.6 else "FAIL"}
```

- [ ] **Step 4: 写 `checking/gate1.py`**

```python
"""闸门 1：t0 可恢复性（多 σ 剖面 → 强度桶分类 + 回归）。"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from checking import data
from checking.metrics import (NearestCentroid, balanced_accuracy, spearman,
                              linear_fit_predict, group_split)


def _bucket(s: float) -> str:
    return "low" if s < 0.35 else ("mid" if s < 0.65 else "high")


def run(probe_dir, extractor, max_n=None) -> dict:
    probe_dir = Path(probe_dir)
    samples = data.load(probe_dir / "gate1_strength.jsonl")
    if max_n:
        samples = samples[:max_n]
    X, kept = data.profiles(extractor, probe_dir, samples)
    if len(kept) < 6:
        return {"gate": 1, "metrics": {"balanced_accuracy": 0.0, "spearman_rho": 0.0,
                "multi_sigma_acc": 0.0, "single_sigma_acc": 0.0, "n": len(kept)},
                "verdict": "FAIL", "note": "样本不足"}
    strengths = np.array([s.strength for s in kept], float)
    buckets = [_bucket(v) for v in strengths]
    groups = [s.real_image_path or s.image_id for s in kept]
    tr, te = group_split(groups, test_frac=0.4, seed=0)
    ytr = [buckets[i] for i in tr]; yte = [buckets[i] for i in te]
    multi = balanced_accuracy(yte, NearestCentroid().fit(X[tr], ytr).predict(X[te])) \
        if tr and te and len(set(ytr)) >= 2 else 0.0
    rho = spearman(strengths[te], linear_fit_predict(X[tr], strengths[tr], X[te])) \
        if tr and te else 0.0
    Xs = X[:, :1]
    single = balanced_accuracy(yte, NearestCentroid().fit(Xs[tr], ytr).predict(Xs[te])) \
        if tr and te and len(set(ytr)) >= 2 else 0.0
    verdict = ("PASS" if multi >= 0.55 and rho >= 0.30
               else "WEAK" if multi >= 0.45 else "FAIL")
    return {"gate": 1,
            "metrics": {"balanced_accuracy": round(multi, 4), "spearman_rho": round(rho, 4),
                        "multi_sigma_acc": round(multi, 4), "single_sigma_acc": round(single, 4),
                        "n": len(kept)},
            "verdict": verdict}
```

- [ ] **Step 5: 运行确认通过**

Run: `pytest tests/test_checking_gates.py -q` → PASS

- [ ] **Step 6: 提交**

```bash
git add checking/gate0.py checking/gate1.py tests/test_checking_gates.py
git commit -m "feat(checking): gate0 检测/定位 + gate1 t0 可恢复（含端到端 fixture）"
```

---

## Task 4: gate2 + gate3 + gate4_eval

**Files:**
- Create: `checking/gate2.py`, `checking/gate3.py`, `checking/gate4_eval.py`
- Modify: `tests/test_checking_gates.py`

**Interfaces:**
- Produces：`gate2.run(probe_dir, extractor, max_n=None, plot_path=None)->dict`、`gate3.run(probe_dir, run_dir, extractor, max_n=None)->dict`、`gate4_eval.run(run_dir, extractor, max_n=None)->dict`

- [ ] **Step 1: 追加失败测试** `tests/test_checking_gates.py`

```python
def test_gate2(dd):
    from checking import gate2
    r = gate2.run(dd["probe"], dd["ext"])
    assert r["verdict"] in {"PASS", "CONFOUND", "WEAK"}
    assert 0.0 <= r["metrics"]["same_model_acc"] <= 1.0
    assert 0.0 <= r["metrics"]["cross_model_acc"] <= 1.0


def test_gate3(dd):
    from checking import gate3
    r = gate3.run(dd["probe"], dd["run"], dd["ext"])
    assert r["verdict"] in {"PASS", "PARTIAL"}
    assert "multi_sigma_delta" in r["metrics"]
    assert 0.0 <= r["metrics"]["heldout_acc"] <= 1.0


def test_gate4_eval(dd):
    from checking import gate4_eval
    r = gate4_eval.run(dd["run"], dd["ext"])
    assert r["verdict"] == "EVAL-ONLY"
    assert isinstance(r["metrics"]["per_split"], dict)
```

- [ ] **Step 2: 运行确认失败** → FAIL

- [ ] **Step 3: 写 `checking/gate2.py`**

```python
"""闸门 2：算子可分性 + 操作 vs 模型指纹。"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from checking import data
from checking.metrics import NearestCentroid, balanced_accuracy, group_split, pca_2d


def _acc_within(X, y, groups, seed=0):
    tr, te = group_split(groups, test_frac=0.4, seed=seed)
    if not tr or not te or len(set(y[i] for i in tr)) < 2:
        return None
    clf = NearestCentroid().fit(X[tr], [y[i] for i in tr])
    return balanced_accuracy([y[i] for i in te], clf.predict(X[te]))


def _plot(X, labels, path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    xy = pca_2d(X)
    fig, ax = plt.subplots(figsize=(6, 5))
    for lab in sorted(set(labels)):
        m = [i for i, l in enumerate(labels) if l == lab]
        ax.scatter(xy[m, 0], xy[m, 1], s=8, label=lab)
    ax.legend(fontsize=7); ax.set_title("operator PCA-2D")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100); plt.close(fig)
    return True


def run(probe_dir, extractor, max_n=None, plot_path=None) -> dict:
    probe_dir = Path(probe_dir)
    samples = data.load(probe_dir / "gate2_operator.jsonl")
    if max_n:
        samples = samples[:max_n]
    X, kept = data.profiles(extractor, probe_dir, samples)
    ops = [s.operator for s in kept]
    fams = [s.generator_family for s in kept]
    groups = [s.real_image_path or s.image_id for s in kept]
    same = []
    for f in sorted(set(fams)):
        idx = [i for i, x in enumerate(fams) if x == f]
        if len(set(ops[i] for i in idx)) < 2:
            continue
        a = _acc_within(X[idx], [ops[i] for i in idx], [groups[i] for i in idx])
        if a is not None:
            same.append(a)
    same_acc = float(np.mean(same)) if same else 0.0
    cross = []
    fam_set = sorted(set(fams))
    for fa in fam_set:
        for fb in fam_set:
            if fa == fb:
                continue
            ia = [i for i, x in enumerate(fams) if x == fa]
            ib = [i for i, x in enumerate(fams) if x == fb]
            shared = set(ops[i] for i in ia) & set(ops[i] for i in ib)
            if len(shared) < 2:
                continue
            ia = [i for i in ia if ops[i] in shared]
            ib = [i for i in ib if ops[i] in shared]
            clf = NearestCentroid().fit(X[ia], [ops[i] for i in ia])
            cross.append(balanced_accuracy([ops[i] for i in ib], clf.predict(X[ib])))
    cross_acc = float(np.mean(cross)) if cross else 0.0
    verdict = ("PASS" if same_acc >= 0.50 and cross_acc >= 0.40
               else "CONFOUND" if same_acc >= 0.50 and cross_acc < 0.30 else "WEAK")
    plotted = _plot(X, ops, plot_path) if plot_path else False
    return {"gate": 2,
            "metrics": {"same_model_acc": round(same_acc, 4),
                        "cross_model_acc": round(cross_acc, 4), "n": len(kept)},
            "verdict": verdict, "plot": plotted}
```

- [ ] **Step 4: 写 `checking/gate3.py`**

```python
"""闸门 3：多 σ 增量 + 跨生成器掉点。"""
from __future__ import annotations
from pathlib import Path
from checking import data, gate1
from checking.metrics import NearestCentroid, balanced_accuracy, group_split


def run(probe_dir, run_dir, extractor, max_n=None) -> dict:
    probe_dir = Path(probe_dir)
    g1 = gate1.run(probe_dir, extractor, max_n=max_n)
    delta = round(g1["metrics"]["multi_sigma_acc"] - g1["metrics"]["single_sigma_acc"], 4)
    samples = data.load(probe_dir / "gate2_operator.jsonl")
    if max_n:
        samples = samples[:max_n]
    X, kept = data.profiles(extractor, probe_dir, samples)
    ops = [s.operator for s in kept]
    seen = [i for i, s in enumerate(kept) if s.split == "train"]
    held = [i for i, s in enumerate(kept) if s.split == "test_b"]
    n_ops = len(set(ops)) or 1
    random_acc = 1.0 / n_ops
    seen_acc = heldout_acc = 0.0
    if seen and held and len(set(ops[i] for i in seen)) >= 2:
        Xseen = X[seen]; yseen = [ops[i] for i in seen]
        gseen = [kept[i].real_image_path or kept[i].image_id for i in seen]
        tr, te = group_split(gseen, test_frac=0.4, seed=0)
        if tr and te and len(set(yseen[i] for i in tr)) >= 2:
            seen_acc = balanced_accuracy(
                [yseen[i] for i in te],
                NearestCentroid().fit(Xseen[tr], [yseen[i] for i in tr]).predict(Xseen[te]))
        heldout_acc = balanced_accuracy(
            [ops[i] for i in held], NearestCentroid().fit(Xseen, yseen).predict(X[held]))
    verdict = "PASS" if delta > 0 and heldout_acc > random_acc else "PARTIAL"
    return {"gate": 3,
            "metrics": {"multi_sigma_delta": delta, "seen_acc": round(seen_acc, 4),
                        "heldout_acc": round(heldout_acc, 4),
                        "cross_generator_drop": round(seen_acc - heldout_acc, 4),
                        "random_acc": round(random_acc, 4)},
            "verdict": verdict,
            "note": "跨生成器崩得厉害→第二篇泛化动机；部分掉点可接受"}
```

- [ ] **Step 5: 写 `checking/gate4_eval.py`**

```python
"""闸门 4 评测轴骨架：Test-A..F 用简单检测器算指标（非论文模型）。"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from checking import data
from checking.metrics import separability_auc

_TESTS = ["test_a", "test_b", "test_c", "test_d", "test_e"]


def run(run_dir, extractor, max_n=None) -> dict:
    run_dir = Path(run_dir)
    samples = data.load(run_dir / "manifest.jsonl")
    if max_n:
        samples = samples[:max_n]
    by_split = {}
    for s in samples:
        try:
            img = data.image_of(run_dir, s)
        except Exception:
            continue
        sc = float(extractor.residual_map(img).mean())
        by_split.setdefault(s.split, []).append((int(s.is_fake), sc))
    per = {}
    for sp in _TESTS:
        rows = by_split.get(sp, [])
        ys = [r[0] for r in rows]; ss = [r[1] for r in rows]
        if len(set(ys)) >= 2:
            per[sp] = {"detection_auc": round(separability_auc(ys, ss), 4), "n": len(rows)}
        elif rows:
            per[sp] = {"detection_auc": None, "n": len(rows), "note": "单一类别"}
    trainf = [sc for f, sc in by_split.get("train", []) if f == 0]
    thr = float(np.median(trainf)) if trainf else 0.0
    tf = by_split.get("test_f", [])
    fpr = float(np.mean([1.0 if sc > thr else 0.0 for _, sc in tf])) if tf else None
    return {"gate": 4,
            "metrics": {"per_split": per, "test_f_fpr": round(fpr, 4) if fpr is not None else None},
            "verdict": "EVAL-ONLY",
            "note": "评测轴接线；完整多任务模型/训练/SOTA baseline 属论文系统，非本检查范畴"}
```

- [ ] **Step 6: 运行确认通过**

Run: `pytest tests/test_checking_gates.py -q` → PASS

- [ ] **Step 7: 提交**

```bash
git add checking/gate2.py checking/gate3.py checking/gate4_eval.py tests/test_checking_gates.py
git commit -m "feat(checking): gate2 算子可分 + gate3 跨生成器 + gate4 评测轴骨架"
```

---

## Task 5: run_gates CLI + README + 端到端

**Files:**
- Create: `checking/run_gates.py`, `checking/README.md`
- Modify: `tests/test_checking_gates.py`, `.gitignore`

**Interfaces:**
- Produces：`run_gates.main(argv=None)->int`；CLI `python -m checking.run_gates --run DIR --probe DIR [--extractor multisigma|real] [--max N] [--out PATH]`

- [ ] **Step 1: 追加失败测试** `tests/test_checking_gates.py`

```python
def test_run_gates_cli(dd, tmp_path):
    import json
    from checking.run_gates import main
    out = tmp_path / "report.json"
    rc = main(["--run", str(dd["run"]), "--probe", str(dd["probe"]), "--out", str(out)])
    assert rc == 0 and out.exists()
    rep = json.loads(out.read_text(encoding="utf-8"))
    assert "caveat" in rep
    assert set(rep["gates"]) == {"gate0", "gate1", "gate2", "gate3", "gate4"}
    assert all("verdict" in rep["gates"][g] for g in rep["gates"])
```

- [ ] **Step 2: 运行确认失败** → FAIL

- [ ] **Step 3: 写 `checking/run_gates.py`**

```python
"""闸门执行入口：跑 gate0-3 + gate4_eval，打印 VERDICT，写 report.json。"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from checking.extractor import get_extractor
from checking import gate0, gate1, gate2, gate3, gate4_eval

_CAVEAT = ("extractor=multisigma 是 CPU 代理信号：在 mock 数据上的 VERDICT 仅验证分析代码通路，"
           "非科学结论（甚至可能假阳性）。真实判定需 extractor=real（SD2）+ 真实扩散生成数据 + GPU。")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="checking.run_gates", description="闸门执行测试")
    ap.add_argument("--run", default="data/run")
    ap.add_argument("--probe", default="data/probe")
    ap.add_argument("--extractor", default="multisigma")
    ap.add_argument("--max", type=int, default=None)
    ap.add_argument("--out", default="data/checking_report.json")
    args = ap.parse_args(argv)
    ext = get_extractor(args.extractor)
    plot = str(Path(args.out).with_name("gate2_pca.png"))
    gates = {
        "gate0": gate0.run(args.run, ext, max_n=args.max or 200),
        "gate1": gate1.run(args.probe, ext, max_n=args.max),
        "gate2": gate2.run(args.probe, ext, max_n=args.max, plot_path=plot),
        "gate3": gate3.run(args.probe, args.run, ext, max_n=args.max),
        "gate4": gate4_eval.run(args.run, ext, max_n=args.max),
    }
    for k, r in gates.items():
        print(f"[{k}] VERDICT={r['verdict']}  {json.dumps(r['metrics'], ensure_ascii=False)}")
    print("CAVEAT:", _CAVEAT)
    report = {"extractor": args.extractor, "caveat": _CAVEAT, "gates": gates}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("report ->", args.out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
```

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/test_checking_gates.py -q` → PASS

- [ ] **Step 5: 写 `checking/README.md`**（要点，中文）
  - 一句话：消费 `data/probe`/`data/run` 跑闸门 0–3 + Gate4 评测轴。
  - **诚实边界**（置顶）：`multisigma` 是 CPU 代理，mock 上 VERDICT 仅验证通路、非科学结论；真实判定需 `--extractor real`（SD2）+ 真实数据 + GPU。
  - 用法：先 `forgery-pipeline run` + `forgery-pipeline probe` 产数据，再 `python -m checking.run_gates --run data/run --probe data/probe`。
  - 每闸门产出/阈值一览（引 `docs/GATE_DATA.md` 与 `docs/EXECUTION_CHECKLIST.md`）。
  - 接真实信号：实现 `DiffusersSD2Residual` + `pip install .[real]` + GPU。

- [ ] **Step 6: `.gitignore` 追加**（防提交产物）

```gitignore
data/checking_report.json
data/gate2_pca.png
```
（`data/` 已整体忽略，此为显式说明；若已被 `data/` 覆盖可跳过。）

- [ ] **Step 7: 全量测试 + 真实数据冒烟**

```bash
pytest -q
forgery-pipeline run   --config configs/pipeline.example.yaml --out data/run   >/dev/null
forgery-pipeline probe --config configs/probe.yaml            --out data/probe >/dev/null
python -m checking.run_gates --run data/run --probe data/probe
```
Expected: `pytest` 全绿；CLI 打印 gate0–4 五个 VERDICT + CAVEAT，并写 `data/checking_report.json`。

- [ ] **Step 8: 提交**

```bash
git add checking/run_gates.py checking/README.md tests/test_checking_gates.py .gitignore
git commit -m "feat(checking): run_gates CLI + README（诚实边界）+ 端到端"
```

---

## Self-Review

**1. Spec coverage：**
- §2.1 extractor（抽象+multisigma+SD2 stub+get_extractor）→ Task 2 ✓
- §2.2 metrics（roc/sep/ba/spearman/NC/lstsq/pca/split）→ Task 1 ✓
- §2.3 data（load/image_of/mask_of/profiles）→ Task 2 ✓
- §2.4 gate0/1/2/3/gate4_eval → Task 3+4 ✓
- §2.5 run_gates CLI + report+caveat → Task 5 ✓
- §2.6 可导入（conftest + module-run；不改 pyproject）→ Task 1（注：以 conftest 替代打包，规避 -e 安装风险）✓
- §4 测试（metrics/extractor/端到端/CLI）→ Task 1/2/3/4/5 ✓
- 诚实边界（README+report caveat）→ Task 5 ✓

无缺口。

**2. Placeholder scan：** 各步含完整代码/命令/期望，无 TBD/TODO。README 为要点清单（Task 5 Step 5），实现时按点成文，非占位。

**3. Type consistency：**
- `ResidualExtractor.profile/residual_map/residual_stack`（Task 2）被 gate0/1/2/3/4（Task 3/4）调用一致 ✓
- `data.profiles(extractor,root,samples)->(X,kept)`（Task 2）被各 gate 解包一致 ✓
- metrics 函数签名（Task 1）在各 gate 调用一致（`separability_auc/balanced_accuracy/spearman/NearestCentroid/linear_fit_predict/group_split/pca_2d`）✓
- `gate1.run` 返回 `metrics` 含 `multi_sigma_acc/single_sigma_acc`，被 `gate3` 读取一致 ✓
- `run_gates` 调 `gateX.run(...)` 参数与各 gate 定义一致（gate0(run,ext,max_n)/gate1(probe,ext,max_n)/gate2(probe,ext,max_n,plot_path)/gate3(probe,run,ext,max_n)/gate4_eval(run,ext,max_n)）✓

无不一致。

## 执行顺序
Task 1 → 2 → 3 → 4 → 5；每步 TDD，Task 5 跑全量 + 真实数据冒烟。

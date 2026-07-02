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

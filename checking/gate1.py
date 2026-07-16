"""闸门 1：t0 可恢复性（多 σ 剖面 → 强度桶分类 + 回归）。"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from checking import data
from checking.metrics import (NearestCentroid, balanced_accuracy, spearman,
                              linear_fit_predict, group_split, bootstrap_ci)


def _bucket(s: float, lo: float = 0.35, hi: float = 0.65) -> str:
    return "low" if s < lo else ("mid" if s < hi else "high")


def _bucket2(s: float, cut: float) -> str:
    """2 桶：以 median 切点二分强度（探索性降级运行点，非官方 verdict）。"""
    return "high" if s >= cut else "low"


# 桶边界敏感性网格：BA 读数不能只依赖单一桶界
_BOUNDARY_GRID = [(0.30, 0.60), (0.35, 0.65), (0.40, 0.70)]


def run(probe_dir, extractor, max_n=None) -> dict:
    probe_dir = Path(probe_dir)
    samples = data.load(probe_dir / "gate1_strength.jsonl")
    if max_n:
        samples = samples[:max_n]
    X, kept = data.profiles(extractor, probe_dir, samples)
    if len(kept) < 6:
        return {"gate": 1,
                "metrics": {"balanced_accuracy": 0.0, "spearman_rho": 0.0,
                            "multi_sigma_acc": 0.0, "single_sigma_acc": 0.0, "n": len(kept)},
                "verdict": "FAIL", "note": "样本不足"}
    strengths = np.array([s.strength for s in kept], float)
    buckets = [_bucket(v) for v in strengths]
    groups = [s.real_image_path or s.image_id for s in kept]
    tr, te = group_split(groups, test_frac=0.4, seed=0)
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
    sens = {}
    for lo, hi in _BOUNDARY_GRID:
        yb = [_bucket(strengths[i], lo, hi) for i in te]
        pb = [_bucket(v, lo, hi) for v in reg_pred]
        sens[f"{lo:.2f}/{hi:.2f}"] = round(balanced_accuracy(yb, pb), 4) if pb else 0.0
    # 2 桶 median 切点（探索性降级运行点，PATCH 6 粗桶方向的最简版；不参与官方 verdict）
    cut2 = float(np.median(strengths))
    yte2 = [_bucket2(strengths[i], cut2) for i in te]
    pred2 = [_bucket2(v, cut2) for v in reg_pred]
    two_ba = balanced_accuracy(yte2, pred2) if pred2 else 0.0
    two_ci = bootstrap_ci(yte2, pred2, balanced_accuracy) if pred2 else [None, None]
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
                        "single_sigma_acc": round(single, 4),
                        "bucket_sensitivity": sens,
                        "two_bucket_median": {
                            "ba": round(two_ba, 4), "ba_ci": two_ci, "cut": round(cut2, 4),
                            "note": "探索性降级运行点(median 2-bucket, 随机=0.5)；非官方 verdict，"
                                    "判据待 n>=200 强度网格验证性复测"},
                        "n": len(kept)},
            "verdict": verdict}

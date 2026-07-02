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
        return {"gate": 1,
                "metrics": {"balanced_accuracy": 0.0, "spearman_rho": 0.0,
                            "multi_sigma_acc": 0.0, "single_sigma_acc": 0.0, "n": len(kept)},
                "verdict": "FAIL", "note": "样本不足"}
    strengths = np.array([s.strength for s in kept], float)
    buckets = [_bucket(v) for v in strengths]
    groups = [s.real_image_path or s.image_id for s in kept]
    tr, te = group_split(groups, test_frac=0.4, seed=0)
    ytr = [buckets[i] for i in tr]; yte = [buckets[i] for i in te]
    multi = (balanced_accuracy(yte, NearestCentroid().fit(X[tr], ytr).predict(X[te]))
             if tr and te and len(set(ytr)) >= 2 else 0.0)
    rho = (spearman(strengths[te], linear_fit_predict(X[tr], strengths[tr], X[te]))
           if tr and te else 0.0)
    Xs = X[:, :1]
    single = (balanced_accuracy(yte, NearestCentroid().fit(Xs[tr], ytr).predict(Xs[te]))
              if tr and te and len(set(ytr)) >= 2 else 0.0)
    verdict = ("PASS" if multi >= 0.55 and rho >= 0.30
               else "WEAK" if multi >= 0.45 else "FAIL")
    return {"gate": 1,
            "metrics": {"balanced_accuracy": round(multi, 4), "spearman_rho": round(rho, 4),
                        "multi_sigma_acc": round(multi, 4), "single_sigma_acc": round(single, 4),
                        "n": len(kept)},
            "verdict": verdict}

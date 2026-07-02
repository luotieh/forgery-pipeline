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
            "metrics": {"per_split": per,
                        "test_f_fpr": round(fpr, 4) if fpr is not None else None},
            "verdict": "EVAL-ONLY",
            "note": "评测轴接线；完整多任务模型/训练/SOTA baseline 属论文系统，非本检查范畴"}

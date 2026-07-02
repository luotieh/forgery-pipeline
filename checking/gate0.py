"""闸门 0：残差能否分开 真实 vs 编辑区（检测 + 定位）。"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from checking import data
from checking.metrics import separability_auc


def run(run_dir, extractor, max_n: int = 200) -> dict:
    run_dir = Path(run_dir)
    samples = data.load(run_dir / "manifest.jsonl")
    if max_n and len(samples) > max_n:
        # 等间隔取样：manifest 分段有序（pristine→gate1→gate2），头部截断会漏掉整段掩码样本
        idx = np.linspace(0, len(samples) - 1, max_n).round().astype(int)
        samples = [samples[i] for i in dict.fromkeys(idx.tolist())]
    det_y, det_s, loc = [], [], []
    for s in samples:
        try:
            img = data.image_of(run_dir, s)
        except Exception:
            continue
        rmap = extractor.residual_map(img)
        det_y.append(int(s.is_fake)); det_s.append(extractor.detection_score(img))
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

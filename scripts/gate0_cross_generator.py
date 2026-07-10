"""gate0 跨生成器迁移检验：冻结 SD1.5 残差能否检出 **Kandinsky** 生成的伪造。

判决性实验（见 docs/next_optimization_plan_2026-07-09.md P0）：
  正样本 = generator_family=kandinsky 的 fake；负样本 = pristine real。
  若 detection_auc 仍显著 >0.5，gate0 是可迁移真信号；若塌到 0.5，
  则 0.688 主要是「SD1.5 检 SD1.5」的自家先验偏置。

复用 checking.data / extractor / metrics，不改动闸门代码。
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from checking import data
from checking.extractor import get_extractor
from checking.metrics import separability_auc


def run(run_dir, extractor, family: str = "kandinsky", max_n: int | None = None) -> dict:
    run_dir = Path(run_dir)
    samples = data.load(run_dir / "manifest.jsonl")
    pos = [s for s in samples if s.is_fake and s.generator_family == family]
    neg = [s for s in samples if not s.is_fake]
    if max_n:  # 等间隔取样，保持正负两段都取到
        def thin(xs):
            if len(xs) <= max_n:
                return xs
            idx = np.linspace(0, len(xs) - 1, max_n).round().astype(int)
            return [xs[i] for i in dict.fromkeys(idx.tolist())]
        pos, neg = thin(pos), thin(neg)
    det_y, det_s, loc = [], [], []
    for s in pos + neg:
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
    return {"gate": "0_cross_generator", "family": family,
            "metrics": {"detection_auc": round(det, 4),
                        "localization_auc": round(loc_auc, 4),
                        "n_pos": sum(det_y), "n_neg": len(det_y) - sum(det_y),
                        "n_localization": len(loc)},
            "interpretation": ("可迁移真信号" if det >= 0.6 else
                               "弱迁移" if det >= 0.55 else "疑似先验偏置（迁移塌陷）")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="gate0_cross_generator")
    ap.add_argument("--run", default="data/probe_real")
    ap.add_argument("--family", default="kandinsky", help="被检生成器族（异族先验）")
    ap.add_argument("--extractor", default="real")
    ap.add_argument("--max", type=int, default=None)
    ap.add_argument("--out", default="data/gate0_cross_generator.json")
    args = ap.parse_args(argv)
    r = run(args.run, get_extractor(args.extractor), family=args.family, max_n=args.max)
    print(json.dumps(r, ensure_ascii=False, indent=2))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
    print("report ->", args.out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

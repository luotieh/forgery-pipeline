"""只跑 gate1（跳过 gate0/2/3/4）→ 快速 t0 复算/验证，只提取强度网格图。

用法：python scripts/gate1_only.py <probe_dir> <out.json>
extractor 固定 real；方向特征由环境变量 CHECKING_DIRECTION_FEATURES=0/1 控制。
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from checking import gate1
from checking.extractor import get_extractor


def main() -> int:
    probe_dir, out = sys.argv[1], sys.argv[2]
    r = gate1.run(probe_dir, get_extractor("real"))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
    m = r["metrics"]; tb = m["two_bucket_median"]
    print(f"{out}: verdict={r['verdict']} 3桶BA={m['balanced_accuracy']:.4f} "
          f"rho={m['spearman_rho']:.4f} 2桶BA={tb['ba']:.4f} CI={tb['ba_ci']} "
          f"cut={tb['cut']} n={m['n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

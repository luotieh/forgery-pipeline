"""旧 manifest 回填 PATCH 7 字段（sample_kind 按 is_fake；compositing=none 仅 masked 编辑行；io_chain=legacy）。"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from forgery_pipeline import manifest

def backfill(in_path, out_path) -> int:
    rows = manifest.read_jsonl(in_path)
    for r in rows:
        r.sample_kind = r.sample_kind or ("edited" if r.is_fake else "real")
        r.io_chain = r.io_chain or "legacy"
        if r.is_fake and r.mask_path and r.compositing is None:
            r.compositing = "none"          # 历史 masked 编辑均为整图直出
    manifest.write_jsonl(out_path, rows)
    return len(rows)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); print("backfilled", backfill(a.inp, a.out), "rows")

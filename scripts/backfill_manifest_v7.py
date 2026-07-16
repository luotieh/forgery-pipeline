"""旧 manifest 回填 PATCH 7 字段（sample_kind 按 is_fake；compositing=none 仅 masked 编辑行；io_chain=legacy）。"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from forgery_pipeline import manifest

def backfill(in_path, out_path) -> int:
    rows = manifest.read_jsonl(in_path)
    # base_id 回填（PATCH 9.3 前置）：先按 real 行建 image_path -> image_id 映射，
    # 衍生行（含历史 real_vae_rt，其 is_fake==0 但带 real_image_path）经 real_image_path 查表，
    # 查不到就退化用 real_image_path 字符串本身当组键，再兜底自身 image_id。
    mapping = {r.image_path: r.image_id for r in rows if r.is_fake == 0}
    for r in rows:
        r.sample_kind = r.sample_kind or ("edited" if r.is_fake else "real")
        r.io_chain = r.io_chain or "legacy"
        if r.is_fake and r.mask_path and r.compositing is None:
            r.compositing = "none"          # 历史 masked 编辑均为整图直出
        r.base_id = r.base_id or (
            r.image_id if not r.real_image_path and r.is_fake == 0
            else mapping.get(r.real_image_path, r.real_image_path) or r.image_id
        )
    manifest.write_jsonl(out_path, rows)
    return len(rows)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); print("backfilled", backfill(a.inp, a.out), "rows")

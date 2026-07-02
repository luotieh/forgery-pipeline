"""从 picsum.photos 确定性下载真实底图（真实 Unsplash 照片，经 env 代理）。"""
from __future__ import annotations
import argparse
import urllib.request
from pathlib import Path


def fetch(out_dir="data/real_base", n=200, size=512, start_id=0) -> int:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    got, i = 0, start_id
    limit = start_id + n * 4
    while got < n and i < limit:
        url = f"https://picsum.photos/id/{i}/{size}/{size}.jpg"
        try:
            urllib.request.urlretrieve(url, out / f"real_{got:04d}.jpg")
            got += 1
        except Exception:
            pass
        i += 1
    print(f"fetched {got} images -> {out}")
    return got


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", dest="out_dir", default="data/real_base")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--size", type=int, default=512)
    raise SystemExit(0 if fetch(**vars(ap.parse_args())) else 1)

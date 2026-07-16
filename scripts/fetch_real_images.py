"""从 picsum.photos 确定性下载真实底图（真实 Unsplash 照片，经 env 代理）。"""
from __future__ import annotations
import argparse
import socket
import urllib.request
from pathlib import Path


def fetch(out_dir="data/real_base", n=200, size=512, start_id=0, timeout=15) -> int:
    # picsum/fastly 从数据中心 IP 会间歇性挂起（302 后 0 字节久拖）；urlretrieve 无超时会永久阻塞
    # → 设 socket 默认超时，挂起请求超时即抛异常被 except 跳过，换下一个 id 重试。
    socket.setdefaulttimeout(timeout)
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
    ap.add_argument("--timeout", type=int, default=15, help="单请求超时秒数，防 picsum 挂起")
    raise SystemExit(0 if fetch(**vars(ap.parse_args())) else 1)

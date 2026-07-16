"""B3 生成驱动起跑前预检（PATCH 9 §9.4「B3 驱动加固」，`docs/PATCH_9_for_addendum_2026-07-15.md`
§9.4：「把事故 A/B/C 从教训固化为断言」）。

`preflight()` 的三条断言与事故 A/B/C 的对应关系：

- ①HEAD 断言 + ②工作区净树断言 —— 均对应**事故 C**。9.4 原文："驱动起跑前 assert
  `git rev-parse HEAD == 期望 commit`（config 值），工作区不净则拒跑或显式记录。"二者合起来
  保证驱动实际跑的代码版本与预期 commit 完全一致、且没有任何未提交的本地改动混入产出——
  防止"跑的代码和以为跑的代码不是同一份"这一类事故重演。
- ③磁盘余量断言 —— 对应 9.4「磁盘预检」条目："起跑前 assert 数据盘余量 ≥ 配置估计（60–80k
  PNG + 各配置 npz 残差缓存，预留 ≥100GB），双备份策略延续。"防止长跑中途因磁盘写满而产生
  半批残留，是 `forgery_pipeline.rundir` 断点续跑幂等设计的前置防线（磁盘写满导致的半行写入
  正是 `rundir.detect_incomplete_tail` 要处理的残尾场景之一）。

**事故 A**（COCO 底图 fetch 层 socket 挂起超时）不在本模块断言范围内：已在
`scripts/fetch_real_images.py` 的 `socket.setdefaulttimeout` 处理，本预检不重复。

**评估禁令公约（事故 B 的固化）**，引 addendum §9.4「评估禁令」条目原文（字面一致，含原文
着重星号）：「生成驱动内**不得内置**任何对 probe/confirmatory 数据的评估步骤；评估脚本独立、
锁定后手动触发。以代码审查 + 驱动内注释断言双保险。」（释义："锁定"指预注册锁定——同文档
§9.7 "v3 锁定 commit 先于任何 gate2 评估"；此处的驱动即 B3 生成驱动。）这一条纪律无法被
preflight 机械检测——它约束的是"驱动代码里写了什么逻辑"，不是任何运行时可观测的文件系统/
进程状态，只能靠原文规定的"代码审查 + 驱动内注释断言"双保险落实。本 docstring 即该公约在
驱动侧的注释断言载体：任何 import 本模块的 B3 驱动代码，在拿到 `preflight()` 的同时也一并
引入这份提醒。

用法：python scripts/b3_preflight.py --min-free-gb 100 [--expected-head <commit>]
      [--data-dir <path>] [--repo-root <path>]
退出码：0 = 三条断言全部通过；1 = 至少一条未通过（各条违规消息打印到 stderr）。
"""
from __future__ import annotations
import argparse
import shutil
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def preflight(expected_head: str | None, min_free_gb: float, repo_root=".", data_dir=None) -> list[str]:
    """三条起跑前断言，返回违规消息列表（空列表 = 全部通过）。"""
    repo_root = str(repo_root)
    errs: list[str] = []

    # ①HEAD 断言（事故 C）：驱动实际跑的 commit 必须与期望一致（前缀匹配，大小写不敏感）。
    if expected_head is not None:
        actual = subprocess.run(
            ["git", "-C", repo_root, "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        if not actual.lower().startswith(expected_head.lower()):
            errs.append(f"PREFLIGHT: HEAD 不匹配: 期望 {expected_head} 实际 {actual}")

    # ②工作区净树断言（事故 C）：不得有未提交改动混入本次驱动产出。
    status = subprocess.run(
        ["git", "-C", repo_root, "status", "--porcelain"],
        capture_output=True, text=True, check=True,
    ).stdout
    dirty_lines = [line for line in status.splitlines() if line.strip()]
    if dirty_lines:
        errs.append(f"PREFLIGHT: 工作区不净: {len(dirty_lines)} 项")

    # ③磁盘余量断言（9.4 磁盘预检）：数据盘余量须 ≥ min_free_gb，防止长跑中途写满。
    usage = shutil.disk_usage(data_dir or repo_root)
    min_free_bytes = min_free_gb * 2**30
    if usage.free < min_free_bytes:
        free_gb = usage.free / 2**30
        errs.append(f"PREFLIGHT: 磁盘余量不足: {free_gb:.1f}GB < {min_free_gb}GB")

    return errs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="B3 驱动起跑前预检（HEAD/净树/磁盘三条断言）")
    ap.add_argument("--expected-head", default=None,
                    help="期望的 git HEAD（前缀匹配，大小写不敏感；不传则跳过该项）")
    ap.add_argument("--min-free-gb", type=float, required=True, help="数据盘最小余量（GB）")
    ap.add_argument("--data-dir", default=None, help="磁盘余量检查目录（默认 --repo-root）")
    ap.add_argument("--repo-root", default=".", help="git 仓库根目录（默认当前目录）")
    args = ap.parse_args(argv)

    errs = preflight(args.expected_head, args.min_free_gb,
                     repo_root=args.repo_root, data_dir=args.data_dir)
    if errs:
        for e in errs:
            print(e, file=sys.stderr)
        return 1
    print("PREFLIGHT: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""B3 驱动断点续跑幂等原语（PATCH 9 §9.4「断点续跑幂等」，`docs/PATCH_9_for_addendum_2026-07-15.md`
§9.4）。

9.4 原文对该设计的要求："按 cell（生成器×算子×nuisance 单元×批次）落 done-marker；manifest
原子追加（临时文件 + rename，或 JSONL append+fsync）；重启跳过已完成 cell、检测半批残留。
验收含"中途 kill → 重启 → 无重复行、计数吻合"的演练。"本模块提供三组原语，各自对应上述设计
的一部分：

- `append_jsonl_fsync`：manifest 原子追加采用的是 "JSONL append+fsync" 方案——每写完一行立即
  flush + os.fsync，保证该行在函数返回前已真正落盘（不止是进了 OS 页缓存）。驱动被 kill 时，
  最坏情况只丢失"正在写"的那一行，不会出现该行写到一半、下一次追加又接着写从而产生交错脏
  数据的情况；已成功返回的历史行不受后续崩溃影响。
- `mark_done` / `is_done`：按 cell 粒度的 done-marker。cell_key 由驱动自行拼出（例如
  "生成器×算子×nuisance 单元×批次" 的复合字符串），任意内容都安全——落盘前先做 sha1 摘要，
  文件名只使用十六进制摘要，不受 key 本身含斜杠/空格/Unicode 等字符影响文件系统合法性。驱动
  重启后对每个待生成 cell 先查 `is_done`，已完成即跳过，从而实现断点续跑而不重新生成。
- `detect_incomplete_tail`：重启时对 manifest.jsonl 做的"半批残留"检测。若上一次运行恰好在
  某一行 fsync 之前被 kill，该行会以无法被 json.loads 解析的残缺文本形式留在文件末尾。本函数
  负责发现它、把原文件整份逐字节备份、再截掉残尾，让驱动能在一个干净的行边界上继续追加，不
  产生重复行或损坏行。

三者组合起来即"中途 kill → 重启 → 无重复行、计数吻合"验收路径的实现基础：重启时先对
manifest 跑一次 `detect_incomplete_tail` 清残尾，再对每个 cell 用 `is_done` 判断是否需要
重新生成，新产出用 `append_jsonl_fsync` 写回。全程不依赖系统时间/随机数，同样的输入总是
产生同样的输出（确定性，可反复演练、可单测）。

纯 stdlib：json/os/hashlib/pathlib/shutil。
"""
from __future__ import annotations
import hashlib
import json
import os
import shutil
from pathlib import Path


def append_jsonl_fsync(path, obj: dict) -> None:
    """把 obj 序列化为一行 JSON 追加进 path（"a" 模式打开）。

    write → flush → os.fsync 三步落盘，父目录不存在时自动创建（run_dir 首次写入时可能
    还未建立）。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(obj, ensure_ascii=False) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def _marker_path(run_dir, cell_key: str) -> Path:
    digest = hashlib.sha1(cell_key.encode("utf-8")).hexdigest()
    return Path(run_dir) / "_done" / f"{digest}.marker"


def mark_done(run_dir, cell_key: str) -> None:
    """把 cell_key 标记为已完成：在 `{run_dir}/_done/` 下落一个以 sha1(cell_key) 命名的空文件。"""
    marker = _marker_path(run_dir, cell_key)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()


def is_done(run_dir, cell_key: str) -> bool:
    """cell_key 是否已被 mark_done 标记过。`_done` 目录尚不存在时天然返回 False。"""
    return _marker_path(run_dir, cell_key).exists()


def detect_incomplete_tail(path, backup_suffix: str = ".bak") -> bool:
    """检测 path 末行是否是一次被中断的写入（json.loads 失败），是则备份+截断。

    文件不存在，或末行能被正常 json.loads 解析（干净文件）→ 直接返回 False，不触碰文件。

    检测到残尾时：
    1. 把原文件整份、逐字节原样 copy 到 `{path}{backup_suffix}`（不重新格式化，供事后排查/
       找回被截掉的残尾）；
    2. 把 path 原地改写为去掉残尾行之后的内容——保留的其余行使用原始文本、不经
       json.loads/json.dumps 往返改写（避免键序、空格等被悄悄改变）；
    3. 返回 True。

    全程不读系统时间、不用随机数：同样的输入总是产生同样的输出（确定性，便于测试与重复
    演练）。
    """
    path = Path(path)
    if not path.exists():
        return False

    text = path.read_text(encoding="utf-8")
    if text == "":
        return False

    parts = text.split("\n")
    if parts[-1] == "":
        parts.pop()  # 正常结尾换行符产生的空 artifact，不是一行数据
    if not parts:
        return False

    last_line = parts[-1]
    try:
        json.loads(last_line)
        return False  # 末行可解析：文件干净，不动它
    except ValueError:
        pass  # json.JSONDecodeError 是 ValueError 子类，一并覆盖

    backup_path = Path(str(path) + backup_suffix)
    shutil.copy2(path, backup_path)

    remaining_lines = parts[:-1]
    new_text = "".join(line + "\n" for line in remaining_lines)
    path.write_text(new_text, encoding="utf-8")
    return True

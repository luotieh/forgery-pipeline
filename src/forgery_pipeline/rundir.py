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
  某一行 fsync 之前被 kill，该行会以残缺字节的形式留在文件末尾——可能是 JSON 不完整，也可能
  截断点恰好落在多字节 UTF-8 字符中间（连解码都过不了）。本函数按字节发现它、把原文件整份
  逐字节备份、再截掉残尾，让驱动能在一个干净的行边界上继续追加，不产生重复行或损坏行。

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
    """检测 path 末行是否是一次被中断的写入，是则备份+截断（字节级，单次只截一条残尾行）。

    残尾判定：末行字节先尝试 UTF-8 解码、再 json.loads，**任一失败**即视为残尾——中断（kill/
    磁盘写满）可能恰好切在多字节 UTF-8 字符（CJK 等；append_jsonl_fsync 以 ensure_ascii=False
    写入）的中间，此时末行连解码都过不了，须与"可解码但 JSON 不完整"同样处理。因此本函数
    全程按字节操作（read_bytes / 字节偏移截断），绝不对整个文件做解码——否则半个多字节字符
    会让函数自己先抛 UnicodeDecodeError、且此时备份还没来得及创建（审查修复，见对应回归测试）。

    文件不存在，或末行可正常解码+解析（干净文件）→ 直接返回 False，不触碰文件。

    检测到残尾时：
    1. 把原文件整份、逐字节原样 copy 到 `{path}{backup_suffix}`（不重新格式化，供事后排查/
       找回被截掉的残尾）；
    2. 按字节偏移截去残尾行：此前所有行保留原始字节、不经解码/重编码往返（键序、空格、
       非 ASCII 字节都不会被悄悄改变）；
    3. 返回 True。

    单次调用只截一条残尾行（append+fsync 的崩溃模型下最多只产生一条）；重启续跑的正确顺序：
    先调本函数清残尾，再开始 append_jsonl_fsync 追加。

    全程不读系统时间、不用随机数：同样的输入总是产生同样的输出（确定性，便于测试与重复
    演练）。
    """
    path = Path(path)
    if not path.exists():
        return False

    raw = path.read_bytes()
    if raw == b"":
        return False

    # 结尾单个 b"\n" 是正常行终止符，不是一行数据；剥掉后最末一段字节即"末行"。
    body = raw[:-1] if raw.endswith(b"\n") else raw
    nl = body.rfind(b"\n")
    last_line = body[nl + 1:]
    try:
        json.loads(last_line.decode("utf-8"))
        return False  # 末行可解码且可解析：文件干净，不动它
    except ValueError:
        # json.JSONDecodeError 与 UnicodeDecodeError 均为 ValueError 子类，一并覆盖
        # （半个 JSON 与半个多字节字符都是残尾）。
        pass

    backup_path = Path(str(path) + backup_suffix)
    shutil.copy2(path, backup_path)
    # nl == -1（整个文件只有一条残尾行）时 raw[:nl+1] == raw[:0] == b""，即清空文件。
    path.write_bytes(raw[:nl + 1])
    return True

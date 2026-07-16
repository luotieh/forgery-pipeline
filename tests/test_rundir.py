"""src/forgery_pipeline/rundir.py 单元测试（PATCH 9 §9.4 断点续跑幂等原语）。

覆盖 append_jsonl_fsync 的追加读回 roundtrip、mark_done/is_done 的 done-marker roundtrip
（含特殊字符 key）、detect_incomplete_tail 的残尾截断+备份 与 干净文件 noop 两条分支。
"""
from __future__ import annotations
import json
from forgery_pipeline import rundir


def test_append_and_reread_roundtrip(tmp_path):
    """追加 3 行 → 逐行读回须与写入顺序/内容一致。"""
    p = tmp_path / "manifest.jsonl"
    rows = [{"i": 0, "tag": "a"}, {"i": 1, "tag": "b"}, {"i": 2, "tag": "c"}]
    for r in rows:
        rundir.append_jsonl_fsync(p, r)

    lines = p.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert [json.loads(l) for l in lines] == rows


def test_done_marker_roundtrip(tmp_path):
    """mark_done 前 is_done 为 False；mark_done 后为 True；未标记的其他 key 不受影响；
    key 含斜杠/空格/中文/emoji 等特殊字符也须安全（落盘前先 sha1 摘要，不直接当文件名）。
    """
    run_dir = tmp_path / "run"
    key = "gen=sd15/op=inpaint area=0.2 中文 key 🎯"

    assert rundir.is_done(run_dir, key) is False
    rundir.mark_done(run_dir, key)
    assert rundir.is_done(run_dir, key) is True
    # 未标记的另一个 key 不应被误判为已完成。
    assert rundir.is_done(run_dir, "gen=sd15/op=inpaint area=0.2 中文 key 🎯 other") is False


def test_detect_incomplete_tail_truncates_and_backs_up(tmp_path):
    """手写 2 完整行 + 1 半行（模拟 kill 中断）：detect 须返回 True、原文件只剩 2 完整行、
    备份文件存在且完整包含原始 3 段内容（逐字节保真，不经 json 往返改写）。
    """
    p = tmp_path / "manifest.jsonl"
    row0 = json.dumps({"i": 0})
    row1 = json.dumps({"i": 1})
    half = '{"i": 2, "half"'  # 无收尾括号、无换行：典型的写到一半被 kill 的残尾
    with open(p, "w", encoding="utf-8") as f:
        f.write(row0 + "\n")
        f.write(row1 + "\n")
        f.write(half)

    assert rundir.detect_incomplete_tail(p) is True

    remaining = p.read_text(encoding="utf-8").splitlines()
    assert len(remaining) == 2
    assert [json.loads(l) for l in remaining] == [{"i": 0}, {"i": 1}]

    backup = p.with_name(p.name + ".bak")
    assert backup.exists()
    backup_text = backup.read_text(encoding="utf-8")
    assert row0 in backup_text
    assert row1 in backup_text
    assert half in backup_text


def test_detect_clean_file_noop(tmp_path):
    """末行可正常解析（干净文件）：detect 须返回 False 且完全不动文件（内容不变、不产生备份）。"""
    p = tmp_path / "manifest.jsonl"
    rows = [{"i": 0}, {"i": 1}]
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    before = p.read_text(encoding="utf-8")

    assert rundir.detect_incomplete_tail(p) is False

    after = p.read_text(encoding="utf-8")
    assert after == before
    assert not p.with_name(p.name + ".bak").exists()

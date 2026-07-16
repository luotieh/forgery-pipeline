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


def test_detect_incomplete_tail_cut_inside_multibyte_char(tmp_path):
    """残尾恰好切在多字节 UTF-8 字符中间（磁盘写满/kill 中断写入的现实形态；
    append_jsonl_fsync 以 ensure_ascii=False 写 CJK，字节级截断完全可能落在字符内部）：
    须与普通残尾同样处理——返回 True、备份含原始全部字节、此前完整行的字节逐一原样保留——
    而不是抛 UnicodeDecodeError 且不创建备份（审查修复的回归测试）。"""
    p = tmp_path / "manifest.jsonl"
    row0 = json.dumps({"i": 0, "t": "中文"}, ensure_ascii=False).encode("utf-8")
    row1 = json.dumps({"i": 1, "t": "样本"}, ensure_ascii=False).encode("utf-8")
    # "中" 的 UTF-8 编码为 e4 b8 ad（3 字节）；截去最后 1 字节留下 e4 b8——无法解码的半字符。
    half = '{"i": 2, "t": "中'.encode("utf-8")[:-1]
    raw = row0 + b"\n" + row1 + b"\n" + half
    p.write_bytes(raw)

    assert rundir.detect_incomplete_tail(p) is True

    # 保留行必须逐字节原样（不经解码/重编码往返）。
    assert p.read_bytes() == row0 + b"\n" + row1 + b"\n"
    backup = p.with_name(p.name + ".bak")
    assert backup.exists()
    assert backup.read_bytes() == raw


def test_detect_incomplete_tail_missing_trailing_newline_appends_newline(tmp_path):
    """末行 JSON 内容完整、但整份文件缺收尾换行符（append-only 写入器每行必带 "\\n"；短写可能
    恰好停在收尾的 "}" 之后、"\\n" 之前，产出一个"看起来完整"实则残缺的末行）：detect 须仍
    返回 True——不能因为末行"可解析"就误判为干净文件，否则下次 append 会紧贴着这行继续写，
    串出 {"i":2}{"i":3} 式的行，且下一轮 detect 会把两行一起当残尾静默截掉，悄悄丢失一条已
    完整落盘的记录。处理方式与"末行不可解析"不同：保留该行内容，只补写缺失的换行符；备份
    文件保存的是补写前的原始字节（审查修复的回归测试）。
    """
    p = tmp_path / "manifest.jsonl"
    row0 = json.dumps({"i": 0}).encode("utf-8")
    row1 = json.dumps({"i": 1}).encode("utf-8")
    raw = row0 + b"\n" + row1  # 两行内容均完整，但整份文件末尾缺换行符

    p.write_bytes(raw)

    assert rundir.detect_incomplete_tail(p) is True

    # 该行被保留，只是补上了缺失的换行符——不是被当成残尾截掉。
    assert p.read_bytes() == raw + b"\n"

    backup = p.with_name(p.name + ".bak")
    assert backup.exists()
    # 备份是补写前的原始字节（逐字节保真，不含补写的换行符）。
    assert backup.read_bytes() == raw


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

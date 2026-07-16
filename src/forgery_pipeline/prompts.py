"""Prompt bank 加载、版本哈希与确定性抽取（PATCH 9.2a）。

prompt 是喂给生成器（img2img/inpaint/object/background 四类算子）的英文模板；
抽取一律走 stable_hash(key)，同 key 永远选中同一模板，保证实验可复现。
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import yaml
from forgery_pipeline.backends.mock import stable_hash

REQUIRED_KINDS = ("img2img", "inpaint", "object", "background")


def load_bank(path: str = "configs/prompt_bank.yaml") -> dict:
    """加载 prompt bank 并校验四节均存在且为非空字符串列表；否则 raise ValueError。"""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    missing = [k for k in REQUIRED_KINDS
               if not isinstance(data.get(k), list) or len(data.get(k)) == 0]
    if missing:
        raise ValueError(f"prompt bank 缺少非空节: {missing}（path={path}）")
    return data


def bank_version(path: str = "configs/prompt_bank.yaml") -> str:
    """bank 文件版本 = 文件字节内容 sha1 的前 12 位十六进制；改一字节即变化。"""
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()[:12]


def pick_prompt(bank: dict, kind: str, key: str) -> str:
    """按 kind 与 key 的 stable_hash 确定性选取模板；kind 未知（或该节为空）报错。"""
    templates = bank.get(kind)
    if not templates:
        raise ValueError(f"未知或空的 prompt kind: {kind!r}")
    return templates[stable_hash(key) % len(templates)]

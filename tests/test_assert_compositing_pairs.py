"""scripts/assert_compositing_pairs.py 单元测试（PATCH 7.5 成对断言脚本）。

正/负例覆盖 check() 的两条分支（paste_feather 带外逐像素相等 / none 全图非平凡差异）+ CLI
退出码 + --plot 在 probe manifest（无 real_vae_rt 行）上的安全跳过。
"""
from __future__ import annotations
import numpy as np
from forgery_pipeline.config import GeneratorSpec
from forgery_pipeline.builders.probe import run_probe
from forgery_pipeline import image_io, manifest
from scripts.assert_compositing_pairs import check, main


def _run_probe(tmp_path, n_pairs=3):
    probe_dir = tmp_path / "p"
    run_probe(probe_dir, n_base=3, strengths=[0.5], operators=["inpaint"],
              img2img_specs=[GeneratorSpec("g", "diffusion", "img2img")],
              inpainter_specs=[GeneratorSpec("i", "diffusion", "inpaint")],
              seed=0, compositing_pairs=n_pairs)
    return probe_dir


def test_check_passes_on_valid_compositing_pairs_probe(tmp_path):
    probe_dir = _run_probe(tmp_path)
    assert check(probe_dir) == []


def test_check_detects_corrupted_paste_feather_row(tmp_path):
    """人为把 paste_feather 行整图替换为纯黑：羽化带外必然不再逐像素等于底图，须 FAIL。"""
    probe_dir = _run_probe(tmp_path)
    rows = manifest.read_jsonl(probe_dir / "manifest.jsonl")
    victim = next(r for r in rows if r.probe_group == "compositing_pair"
                 and r.compositing == "paste_feather")
    img = image_io.load_image(probe_dir / victim.image_path)
    image_io.save_canonical(np.zeros_like(img), probe_dir / victim.image_path)
    errs = check(probe_dir)
    assert any(victim.pair_id in e and "paste_feather" in e for e in errs)


def test_check_detects_none_row_identical_to_base(tmp_path):
    """人为把 none 行图像替换为底图本身：差异比例应降为 0，须 FAIL（真实回归场景：
    compositing 分支写反/生成图路径接错都会表现为此形态）。"""
    probe_dir = _run_probe(tmp_path)
    rows = manifest.read_jsonl(probe_dir / "manifest.jsonl")
    victim = next(r for r in rows if r.probe_group == "compositing_pair"
                 and r.compositing == "none")
    base = image_io.load_image(probe_dir / victim.real_image_path)
    image_io.save_canonical(base, probe_dir / victim.image_path)
    errs = check(probe_dir)
    assert any(victim.pair_id in e and "none" in e for e in errs)


def test_cli_exit_codes(tmp_path):
    probe_dir = _run_probe(tmp_path)
    assert main(["--probe", str(probe_dir)]) == 0

    rows = manifest.read_jsonl(probe_dir / "manifest.jsonl")
    victim = next(r for r in rows if r.probe_group == "compositing_pair"
                 and r.compositing == "none")
    base = image_io.load_image(probe_dir / victim.real_image_path)
    image_io.save_canonical(base, probe_dir / victim.image_path)
    assert main(["--probe", str(probe_dir)]) == 1


def test_check_handles_pair_id_none_without_crashing(tmp_path):
    """残缺 compositing_pair 行（pair_id=None，理论上不应由 build_compositing_pairs 产生，但
    manifest 是纯数据、不能假设总是完整）与正常 pair_id 混排时，排序不得因 None/str 在
    Python 3 下不可比较而抛 TypeError，应正常报出该组行数≠2。"""
    probe_dir = _run_probe(tmp_path)
    rows = manifest.read_jsonl(probe_dir / "manifest.jsonl")
    victim = next(r for r in rows if r.probe_group == "compositing_pair")
    orphan = victim.model_copy(deep=True)
    orphan.image_id = victim.image_id + "-orphan"
    orphan.pair_id = None
    manifest.write_jsonl(probe_dir / "manifest.jsonl", rows + [orphan])
    errs = check(probe_dir)  # 不应抛 TypeError
    assert any("行数≠2" in e for e in errs)


def test_plot_is_safe_noop_on_probe_manifest_without_vae_rt_rows(tmp_path):
    """probe manifest（run_probe 产出）不含 real_vae_rt 行——--plot 须安全跳过而非报错，
    且不产生输出文件（该分支在触碰 matplotlib 之前就已短路返回，天然不依赖 matplotlib 是否安装）。
    """
    probe_dir = _run_probe(tmp_path)
    out_png = tmp_path / "out" / "vae_rt_residual.png"
    assert main(["--probe", str(probe_dir), "--plot", str(out_png)]) == 0
    assert not out_png.exists()

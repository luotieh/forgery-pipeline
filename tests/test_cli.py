import dataclasses
from pathlib import Path
from forgery_pipeline.cli import main
from forgery_pipeline.config import load_config, StageScales
from forgery_pipeline.pipeline import run_pipeline


def _make_run(tmp_path):
    cfg = load_config("configs/pipeline.example.yaml")
    # PATCH 7 V4（real_vae_rt 占比 ∈ [0.05,0.35]）在 train/test_a/test_f 上逐 split 校验；
    # 在过小的规模下（如原 d0=12）某些 split 的 real 行数太少（如 2），vae_rt_frac=0.15 的
    # 逐行插入在该 split 上可能落到 0 命中或命中过密，其占比在离散计数下根本落不进
    # [0.05,0.35] 区间（n_real=2 时可达比例只有 {0, 0.5, 1.0}，无一落在区间内）。
    # d0=60（配 d2/d3/d4 等比放大）是本仓库固定 seed=1234 下经验证的最小可行规模：
    # train/test_a/test_f 三个 split 的 real_vae_rt 占比均落入 [0.05,0.35]（确定性，非概率保证）。
    cfg = dataclasses.replace(cfg, out_dir=str(tmp_path / "run"),
                              scales=StageScales(d0=60, d1_per_generator=4, d2=30, d3=15, d4=6))
    run_pipeline(cfg)
    return Path(cfg.out_dir) / "manifest.jsonl"


def test_validate_manifest_ok(tmp_path):
    mani = _make_run(tmp_path)
    assert main(["validate-manifest", "--path", str(mani)]) == 0


def test_stats_prints(tmp_path, capsys):
    mani = _make_run(tmp_path)
    assert main(["stats", "--path", str(mani)]) == 0
    out = capsys.readouterr().out
    assert "total" in out


def test_validate_manifest_missing_file_returns_nonzero(tmp_path):
    assert main(["validate-manifest", "--path", str(tmp_path / "nope.jsonl")]) != 0

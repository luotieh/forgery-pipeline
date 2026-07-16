import dataclasses
from pathlib import Path
from forgery_pipeline.cli import main
from forgery_pipeline.config import load_config, StageScales
from forgery_pipeline.pipeline import run_pipeline


def _make_run(tmp_path):
    cfg = load_config("configs/pipeline.example.yaml")
    # vae_rt_frac=0 → V4 在 auto profile 下跳过：CLI 冒烟只测接线，V4 的 e2e 覆盖在 test_e2e_patch7 与 test_validate_v7
    cfg = dataclasses.replace(cfg, out_dir=str(tmp_path / "run"),
                              scales=StageScales(d0=12, d1_per_generator=1, d2=6, d3=3, d4=2),
                              vae_rt_frac=0.0)
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

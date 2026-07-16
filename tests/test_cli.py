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


def test_validate_manifest_auto_profile_output_is_honest(tmp_path, capsys):
    """诚实性回归锚（终审修复 Fix 2）：默认 profile=auto 时 V8/V10 实际不执行（裁决B），
    输出不得笼统宣称"V1–V10 通过"——必须显式提示 V8/V10 未执行，且成功行只能列出真正
    执行过的检查项，不能把 V10 混进去冒充已校验。
    """
    mani = _make_run(tmp_path)
    assert main(["validate-manifest", "--path", str(mani)]) == 0
    out = capsys.readouterr().out
    assert "V8/V10 未执行" in out
    success_line = next(line for line in out.splitlines() if line.startswith("OK:"))
    assert "V10" not in success_line
    assert "V8" not in success_line
    assert "V1–V10 通过" not in out


def test_validate_manifest_empty_testc_holdout_string_skips_v10(tmp_path, capsys):
    """终审修复 Fix 1：split-config 中 testc_holdout: "" 空串应被归一为 None，
    使 V10 跳过（同 holdout_generators 的归一逻辑），输出不应声称 V10 已执行。
    """
    import yaml
    mani = _make_run(tmp_path)
    split_cfg = tmp_path / "split.yaml"
    split_cfg.write_text(yaml.dump({
        "holdout_generators": ["mock"],  # 非空，防止 V9 因缺 holdout 而跳过
        "testc_holdout": ""  # 空串应被视同 None
    }), encoding="utf-8")
    # 用 --profile run 启用 V10 校验，以便观察 testc_holdout 空串的影响
    assert main(["validate-manifest", "--path", str(mani), "--split-config", str(split_cfg), "--profile", "run"]) == 0
    out = capsys.readouterr().out
    assert "V10 跳过" in out  # 应输出 V10 因缺 testc_holdout 而跳过
    success_line = next(line for line in out.splitlines() if line.startswith("OK:"))
    assert "V10" not in success_line  # V10 不应在已执行清单中

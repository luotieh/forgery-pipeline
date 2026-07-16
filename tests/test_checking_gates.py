import dataclasses
import json
import pytest
from forgery_pipeline.builders.probe import run_probe
from forgery_pipeline.config import GeneratorSpec, load_config, StageScales
from forgery_pipeline.pipeline import run_pipeline
from checking.extractor import MultiSigmaResidual

_I = [GeneratorSpec("sd-img2img", "diffusion", "img2img"),
      GeneratorSpec("sdxl-img2img", "diffusion-sdxl", "img2img")]
_P = [GeneratorSpec("sd-inpaint", "diffusion", "inpaint"),
      GeneratorSpec("kandinsky-inpaint", "kandinsky", "inpaint")]
_OPS = ["img2img", "inpaint", "outpaint", "object_replacement", "background_editing"]


@pytest.fixture(scope="module")
def dd(tmp_path_factory):
    root = tmp_path_factory.mktemp("gates")
    probe, run = root / "probe", root / "run"
    run_probe(probe, n_base=3, strengths=[0.2, 0.5, 0.8], operators=_OPS,
              img2img_specs=_I, inpainter_specs=_P,
              holdout_generators={"sdxl-img2img", "kandinsky-inpaint"}, seed=0)
    cfg = load_config("configs/pipeline.example.yaml")
    cfg = dataclasses.replace(cfg, out_dir=str(run),
                              scales=StageScales(d0=12, d1_per_generator=1, d2=8, d3=4, d4=3))
    run_pipeline(cfg)
    return {"probe": probe, "run": run, "ext": MultiSigmaResidual()}


def test_gate0(dd):
    from checking import gate0
    r = gate0.run(dd["run"], dd["ext"])
    assert r["verdict"] in {"PASS", "FAIL"}
    assert 0.0 <= r["metrics"]["detection_auc"] <= 1.0
    assert 0.0 <= r["metrics"]["localization_auc"] <= 1.0


def test_gate0_max_n_covers_masked_samples(dd):
    # manifest 头部是 pristine + gate1 全图样本；头部截断会取不到任何掩码样本
    from checking import gate0
    r = gate0.run(dd["probe"], dd["ext"], max_n=12)
    assert r["metrics"]["n_localization"] > 0


def test_gate1(dd):
    from checking import gate1
    r = gate1.run(dd["probe"], dd["ext"])
    assert r["verdict"] in {"PASS", "WEAK", "FAIL"}
    m = r["metrics"]
    assert 0.0 <= m["balanced_accuracy"] <= 1.0
    assert -1.0 <= m["spearman_rho"] <= 1.0
    assert "single_sigma_acc" in m
    # 桶边界敏感性：三组边界下的 reg-bucket BA（判定不能只依赖单一桶界）
    sens = m["bucket_sensitivity"]
    assert set(sens) == {"0.30/0.60", "0.35/0.65", "0.40/0.70"}
    assert all(0.0 <= v <= 1.0 for v in sens.values())
    # 2 桶 median 探索性运行点（附加，不参与 verdict）
    tb = m["two_bucket_median"]
    assert 0.0 <= tb["ba"] <= 1.0 and "cut" in tb and "note" in tb


def test_gate2(dd):
    from checking import gate2
    r = gate2.run(dd["probe"], dd["ext"])
    assert r["verdict"] in {"PASS", "CONFOUND", "WEAK"}
    assert 0.0 <= r["metrics"]["same_model_acc"] <= 1.0
    assert 0.0 <= r["metrics"]["cross_model_acc"] <= 1.0


def test_gate3(dd):
    from checking import gate3
    r = gate3.run(dd["probe"], dd["run"], dd["ext"])
    assert r["verdict"] in {"PASS", "PARTIAL"}
    assert "multi_sigma_delta" in r["metrics"]
    assert 0.0 <= r["metrics"]["heldout_acc"] <= 1.0


def test_gate4_eval(dd):
    from checking import gate4_eval
    r = gate4_eval.run(dd["run"], dd["ext"])
    assert r["verdict"] == "EVAL-ONLY"
    assert isinstance(r["metrics"]["per_split"], dict)


def test_run_gates_cli(dd, tmp_path):
    from checking.run_gates import main
    out = tmp_path / "report.json"
    rc = main(["--run", str(dd["run"]), "--probe", str(dd["probe"]), "--out", str(out)])
    assert rc == 0 and out.exists()
    rep = json.loads(out.read_text(encoding="utf-8"))
    assert "caveat" in rep
    assert set(rep["gates"]) == {"gate0", "gate1", "gate2", "gate3", "gate4"}
    assert all("verdict" in rep["gates"][g] for g in rep["gates"])

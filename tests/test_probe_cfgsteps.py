"""CFG/steps 抖动网格（prereg v2 §5 补充 probe 所需的 builder 支持）。"""
import json
from forgery_pipeline.config import GeneratorSpec
from forgery_pipeline.builders.probe import run_probe
from forgery_pipeline import manifest

_IMG2IMG = [GeneratorSpec("sd-img2img", "diffusion", "img2img")]
_INPS = [GeneratorSpec("sd-inpaint", "diffusion", "inpaint")]


def _g1(tmp_path, **kw):
    run_probe(tmp_path / "probe", n_base=2, strengths=[0.3, 0.7], operators=[],
              img2img_specs=_IMG2IMG, inpainter_specs=_INPS, seed=0, **kw)
    return manifest.read_jsonl(tmp_path / "probe" / "gate1_strength.jsonl")


def test_cfgsteps_grid_expands_rows_and_records_op_params(tmp_path):
    rows = _g1(tmp_path, cfg_grid=[5.0, 7.5], steps_grid=[30])
    assert len(rows) == 2 * 2 * 2 * 1              # base × strength × cfg × steps
    assert len({r.image_id for r in rows}) == len(rows)
    for r in rows:
        p = json.loads(r.op_params)
        assert p["cfg_scale"] in (5.0, 7.5) and p["steps"] == 30
        assert r.strength in (0.3, 0.7) and r.operator == "img2img"
    # 同一 (base, strength) 下不同 cell 的 seed 必须不同（确定性区分）
    seeds = {(r.real_image_path, r.strength, json.dumps(json.loads(r.op_params), sort_keys=True)): r.seed
             for r in rows}
    assert len(set(seeds.values())) == len(seeds)


def test_no_grid_keeps_legacy_behavior(tmp_path):
    rows = _g1(tmp_path)
    assert len(rows) == 2 * 2                      # base × strength，无网格膨胀
    assert all(r.op_params is None for r in rows)


def test_steps_only_grid(tmp_path):
    rows = _g1(tmp_path, steps_grid=[30, 50])
    assert len(rows) == 2 * 2 * 2
    assert all(json.loads(r.op_params)["steps"] in (30, 50) for r in rows)
    assert all("cfg_scale" not in json.loads(r.op_params) for r in rows)

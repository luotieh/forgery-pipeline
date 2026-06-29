import dataclasses
import json
from pathlib import Path

import numpy as np

from forgery_pipeline.viewer import render_overlay, make_thumb, build_viewer
from forgery_pipeline.cli import main
from forgery_pipeline import manifest
from forgery_pipeline.config import load_config, StageScales
from forgery_pipeline.pipeline import run_pipeline


# ---------- Task 1: 渲染辅助 ----------

def test_render_overlay_tints_mask_region():
    fake = np.full((50, 50, 3), 100, np.uint8)
    mask = np.zeros((50, 50), np.uint8)
    mask[10:30, 10:30] = 255
    out = render_overlay(fake, mask)
    assert out.shape == (50, 50, 3) and out.dtype == np.uint8
    assert out[20, 20, 0] > 100                    # mask 内 R 通道被染红
    assert tuple(out[45, 45]) == (100, 100, 100)   # mask 外不变


def test_make_thumb_long_side_and_dtype():
    t = make_thumb(np.zeros((300, 150, 3), np.uint8), size=224)
    assert max(t.shape[:2]) == 224
    assert t.shape[2] == 3 and t.dtype == np.uint8


# ---------- Task 2: build_viewer ----------

def _tiny_run(tmp_path):
    cfg = load_config("configs/pipeline.example.yaml")
    cfg = dataclasses.replace(cfg, out_dir=str(tmp_path / "run"),
                              scales=StageScales(d0=8, d1_per_generator=1, d2=4, d3=2, d4=2))
    run_pipeline(cfg)
    return Path(cfg.out_dir)


def _embedded_samples(html_path):
    text = Path(html_path).read_text(encoding="utf-8")
    line = next(l for l in text.splitlines() if l.startswith("const SAMPLES = "))
    return json.loads(line[len("const SAMPLES = "):].rstrip(";"))


def test_build_viewer_emits_html_with_all_samples(tmp_path):
    run = _tiny_run(tmp_path)
    out = build_viewer(run)
    assert out.exists() and out.name == "viewer.html"
    data = _embedded_samples(out)
    assert len(data) == len(manifest.read_jsonl(run / "manifest.jsonl"))
    ov = [d for d in data if d["paths"]["overlay"]]
    assert ov
    assert (out.parent / ov[0]["paths"]["overlay"]).exists()
    assert (out.parent / data[0]["paths"]["thumb"]).exists()


def test_build_viewer_max_samples(tmp_path):
    run = _tiny_run(tmp_path)
    out = build_viewer(run, max_samples=3)
    assert len(_embedded_samples(out)) == 3


def test_viewer_includes_operator_strength(tmp_path):
    from forgery_pipeline.schema import Sample, TaskType
    from forgery_pipeline import image_io
    run = tmp_path / "run"
    image_io.save_image(np.zeros((32, 32, 3), np.uint8), run / "probe/g1/x.png")
    s = Sample(image_id="probe_s_x", image_path="probe/g1/x.png", is_fake=1,
               task_type=TaskType.whole_image_detection,
               manipulation_level1="whole_generated", manipulation_level2="diffusion",
               operator="img2img", strength=0.5)
    manifest.write_jsonl(run / "manifest.jsonl", [s])
    rec = _embedded_samples(build_viewer(run))[0]
    assert rec["operator"] == "img2img" and rec["strength"] == 0.5
    assert "postprocess_of" in rec


# ---------- Task 3: CLI ----------

def test_viewer_cli_ok(tmp_path):
    run = _tiny_run(tmp_path)
    assert main(["viewer", "--run", str(run)]) == 0
    assert (run / "viewer.html").exists()


def test_viewer_cli_missing_run(tmp_path):
    assert main(["viewer", "--run", str(tmp_path / "nope")]) != 0

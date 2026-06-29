from forgery_pipeline.config import GeneratorSpec
from forgery_pipeline.builders.probe import run_probe
from forgery_pipeline import manifest

_IMG2IMG = [GeneratorSpec("sd-img2img", "diffusion", "img2img"),
            GeneratorSpec("sdxl-img2img", "diffusion-sdxl", "img2img")]
_INPS = [GeneratorSpec("sd-inpaint", "diffusion", "inpaint"),
         GeneratorSpec("kandinsky-inpaint", "kandinsky", "inpaint")]
_OPS = ["img2img", "inpaint", "outpaint", "object_replacement", "background_editing"]


def _run(tmp_path):
    run_probe(tmp_path / "probe", n_base=2, strengths=[0.2, 0.5, 0.8], operators=_OPS,
              img2img_specs=_IMG2IMG, inpainter_specs=_INPS,
              holdout_generators={"sdxl-img2img", "kandinsky-inpaint"}, seed=0)
    return manifest.read_jsonl(tmp_path / "probe" / "manifest.jsonl")


def test_probe_split_marks_holdout_as_test_b(tmp_path):
    rows = _run(tmp_path)
    seen = [r for r in rows if r.split == "train"]
    held = [r for r in rows if r.split == "test_b"]
    assert seen and held
    assert all(r.generator_name in {"sdxl-img2img", "kandinsky-inpaint"}
               for r in held if r.generator_name)


def test_probe_init_timestep_for_img2img_only(tmp_path):
    rows = _run(tmp_path)
    i2i = [r for r in rows if r.operator == "img2img"]
    assert i2i and all(r.init_timestep is not None for r in i2i)
    assert all(abs(r.init_timestep - round((r.strength or 0) * 1000)) <= 1 for r in i2i)
    inpaint = [r for r in rows if r.operator == "inpaint"]
    assert inpaint and all(r.init_timestep is None for r in inpaint)


def test_probe_gate_files_coverage(tmp_path):
    run_probe(tmp_path / "probe", n_base=2, strengths=[0.2, 0.5, 0.8], operators=_OPS,
              img2img_specs=_IMG2IMG, inpainter_specs=_INPS,
              holdout_generators={"sdxl-img2img", "kandinsky-inpaint"}, seed=0)
    g1 = manifest.read_jsonl(tmp_path / "probe" / "gate1_strength.jsonl")
    assert g1 and all(r.operator == "img2img" and r.strength is not None for r in g1)
    g2 = manifest.read_jsonl(tmp_path / "probe" / "gate2_operator.jsonl")
    assert {r.operator for r in g2} == set(_OPS)


def test_probe_cli_via_probe_yaml(tmp_path):
    from forgery_pipeline.cli import main
    out = tmp_path / "probe"
    assert main(["probe", "--config", "configs/probe.yaml",
                 "--out", str(out), "--n-base", "2"]) == 0
    rows = manifest.read_jsonl(out / "manifest.jsonl")
    assert any(r.split == "test_b" for r in rows)          # 留出生成器被标记
    assert any(r.init_timestep is not None for r in rows)  # img2img 带 init_timestep
    assert main(["validate-manifest", "--path", str(out / "manifest.jsonl")]) == 0

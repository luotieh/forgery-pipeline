import numpy as np, pytest
from forgery_pipeline.image_io import load_and_resize, save_canonical, chain, load_image


def test_chain_joins_nodes():
    assert chain("decode", "rs512", "edit:m", "png") == "decode>rs512>edit:m>png"


def test_save_canonical_enforces_png(tmp_path):
    img = np.zeros((8, 8, 3), np.uint8)
    save_canonical(img, tmp_path / "a.png")
    assert (tmp_path / "a.png").exists()
    with pytest.raises(AssertionError):
        save_canonical(img, tmp_path / "b.jpg")


def test_load_and_resize_center_square(tmp_path):
    img = np.zeros((40, 80, 3), np.uint8); img[:, 40:] = 255   # 右半白
    save_canonical(img, tmp_path / "w.png")
    out = load_and_resize(tmp_path / "w.png", size=32)
    assert out.shape == (32, 32, 3)                            # 中心裁剪→方形→resize


def test_d0_real_rows_canonical(tmp_path):
    from forgery_pipeline.builders.d0_real import build_d0
    rows = build_d0(tmp_path, 3, backend="mock", seed=0)
    for r in rows:
        assert r.image_path.endswith(".png") and r.sample_kind == "real"
        assert r.io_chain and r.io_chain.startswith("decode>rs") and r.io_chain.endswith(">png")


def test_probe_strength_rows_canonical(tmp_path):
    from forgery_pipeline.config import GeneratorSpec
    from forgery_pipeline.builders.probe import run_probe
    from forgery_pipeline import manifest
    run_probe(tmp_path / "p", n_base=2, strengths=[0.5], operators=[],
              img2img_specs=[GeneratorSpec("g", "diffusion", "img2img")],
              inpainter_specs=[GeneratorSpec("i", "diffusion", "inpaint")], seed=0)
    rows = manifest.read_jsonl(tmp_path / "p" / "gate1_strength.jsonl")
    assert all(r.sample_kind == "edited" and "edit:g" in r.io_chain for r in rows)

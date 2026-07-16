import numpy as np, pytest
from forgery_pipeline.compositing import composite

def _pair(h=32, w=32):
    rng = np.random.default_rng(0)
    orig = rng.integers(0, 256, (h, w, 3), np.uint8)
    gen = rng.integers(0, 256, (h, w, 3), np.uint8)
    m = np.zeros((h, w), np.float32); m[8:20, 8:20] = 1.0
    return orig, gen, m

def test_none_returns_gen():
    o, g, m = _pair(); assert composite(o, g, m, "none") is g

def test_paste_exact_outside_inside():
    o, g, m = _pair(); out = composite(o, g, m, "paste")
    assert np.array_equal(out[m == 0], o[m == 0])     # 掩码外逐像素==orig
    assert np.array_equal(out[m == 1], g[m == 1])     # 掩码内==gen

def test_paste_feather_blends_band_exact_far_outside():
    o, g, m = _pair(); out = composite(o, g, m, "paste_feather", feather_px=2)
    assert np.array_equal(out[:2], o[:2])             # 远离羽化带 == orig
    band = out[7, 8:20]                               # 边界带为混合值
    assert not np.array_equal(band, o[7, 8:20]) and not np.array_equal(band, g[7, 8:20])

def test_shape_mismatch_raises():
    o, g, m = _pair()
    with pytest.raises(AssertionError):
        composite(o[:16], g, m, "paste")

def test_unknown_mode_raises():
    o, g, m = _pair()
    with pytest.raises(ValueError):
        composite(o, g, m, "bogus")

def test_d2_fifty_fifty_compositing(tmp_path):
    from forgery_pipeline.builders.d0_real import build_d0
    from forgery_pipeline.builders.d2_local import build_d2
    from forgery_pipeline.config import GeneratorSpec
    bases = build_d0(tmp_path, 8, backend="mock", seed=0)
    rows = build_d2(tmp_path, bases, 16, [GeneratorSpec("i1", "diffusion", "inpaint")],
                    backend="mock", seed=0)
    comps = {r.compositing for r in rows}
    assert comps == {"none", "paste_feather"}          # 两种都出现
    assert all(r.feather_px == 8 for r in rows if r.compositing == "paste_feather")
    assert all(r.sample_kind == "edited" and r.io_chain for r in rows)

def test_probe_compositing_pairs(tmp_path):
    from forgery_pipeline.builders.probe import run_probe
    from forgery_pipeline.config import GeneratorSpec
    from forgery_pipeline import manifest
    run_probe(tmp_path / "p", n_base=3, strengths=[0.5], operators=["inpaint"],
              img2img_specs=[GeneratorSpec("g", "diffusion", "img2img")],
              inpainter_specs=[GeneratorSpec("i", "diffusion", "inpaint")],
              seed=0, compositing_pairs=3)
    rows = [r for r in manifest.read_jsonl(tmp_path / "p" / "manifest.jsonl")
            if r.probe_group == "compositing_pair"]
    assert len(rows) == 6                               # 3 组 × 2 行
    by_pair = {}
    for r in rows: by_pair.setdefault(r.pair_id, []).append(r)
    for pid, pr in by_pair.items():
        assert len(pr) == 2 and {p.compositing for p in pr} == {"none", "paste_feather"}
        assert pr[0].mask_path and pr[0].seed == pr[1].seed
        assert pr[0].real_image_path == pr[1].real_image_path

"""PATCH 8.3：Test-C holdout 算子几何平凡性探针（零生成，仅 GT 掩码几何）。"""
import numpy as np
from forgery_pipeline.config import GeneratorSpec
from forgery_pipeline.builders.probe import run_probe
from checking.testc_geometry_probe import mask_geometry, logistic_ovr_scores, run

_IMG2IMG = [GeneratorSpec("sd-img2img", "diffusion", "img2img")]
_INPS = [GeneratorSpec("sd-inpaint", "diffusion", "inpaint")]
_OPS = ["img2img", "inpaint", "outpaint", "object_replacement", "background_editing"]


def _box_mask(h=64, w=64, y0=20, x0=20, side=16):
    m = np.zeros((h, w), np.uint8); m[y0:y0 + side, x0:x0 + side] = 255
    return m


def test_mask_geometry_centered_box():
    f = mask_geometry(_box_mask())
    assert f.shape == (5,)
    area, border, ncomp, convex, cdist = f
    assert abs(area - (16 * 16) / (64 * 64)) < 1e-3   # 面积比
    assert border == 0.0                              # 不触边
    assert ncomp == 1.0                               # 单连通域
    assert convex > 0.9                               # 矩形≈凸
    assert cdist < 0.15                               # 近中心


def test_mask_geometry_border_ring_touches_border():
    m = np.full((64, 64), 255, np.uint8); m[16:48, 16:48] = 0   # 边框环
    f = mask_geometry(m)
    assert f[1] > 0.9        # 边界接触率≈1
    assert f[3] < 0.9        # 环的凸性低（hull≈全图）


def test_mask_geometry_two_components_and_offset():
    m = np.zeros((64, 64), np.uint8)
    m[4:14, 4:14] = 255; m[44:58, 44:58] = 255
    f = mask_geometry(m)
    assert f[2] == 2.0
    off = mask_geometry(_box_mask(y0=2, x0=2, side=10))
    assert off[4] > 0.3      # 角落框质心偏移大


def test_logistic_ovr_separable_vs_chance():
    rng = np.random.default_rng(0)
    y = np.tile([0, 1], 60)                           # 交错 → train/test 都含两类
    X = y[:, None] * 3.0 + rng.normal(0, 0.3, (120, 3))
    s = logistic_ovr_scores(X[:80], y[:80], X[80:])
    assert s.shape == (40,)
    from checking.metrics import roc_auc
    assert roc_auc(y[80:], s) > 0.95                  # 可分 → AUC 高
    Xc = rng.normal(0, 1, (120, 3))                   # 同分布 → 机会线附近
    sc = logistic_ovr_scores(Xc[:80], y[:80], Xc[80:])
    assert 0.2 < roc_auc(y[80:], sc) < 0.8


def test_run_end_to_end_outpaint_trivial_object_replacement_not(tmp_path):
    run_probe(tmp_path / "p", n_base=24, strengths=[0.5], operators=_OPS,
              img2img_specs=_IMG2IMG, inpainter_specs=_INPS, seed=0)
    r = run(tmp_path / "p")
    ops = set(r["per_operator"])
    assert ops == {"inpaint", "outpaint", "object_replacement", "background_editing"}
    for k, v in r["per_operator"].items():
        assert 0.0 <= v["auc"] <= 1.0 and v["n_pos"] > 0
    # 边框环几何平凡；object_replacement 与 inpaint 同为 box → 不平凡
    assert r["per_operator"]["outpaint"]["auc"] >= 0.90
    assert r["decision"]["outpaint"] == "geometry-trivial"
    assert r["per_operator"]["object_replacement"]["auc"] < 0.90
    assert r["decision"]["object_replacement"] == "eligible"

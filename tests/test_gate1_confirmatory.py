"""gate1_confirmatory 统计核心单测（CPU，合成数据，不触 GPU/图像）。"""
import numpy as np
from checking.gate1_confirmatory import (
    ridge_fit_predict, pava_isotonic_fit, pava_predict, repeated_group_kfold,
    oof_predictions, cluster_bootstrap_indices, adjacent_aucs, evaluate)


def test_pava_pools_violators_and_monotone():
    ux, fit = pava_isotonic_fit([1, 2, 3], [1, 3, 2])
    assert np.allclose(fit, [1, 2.5, 2.5]) and np.all(np.diff(fit) >= -1e-12)


def test_pava_merges_duplicate_x_and_predict_clips():
    ux, fit = pava_isotonic_fit([0, 0, 1, 2], [0, 2, 1, 3])   # x=0 重复 → 先合并均值 1
    assert np.all(np.diff(fit) >= -1e-12)
    p = pava_predict(*pava_isotonic_fit([0, 1, 2], [0, 1, 2]), [-9, 0.5, 9])
    assert np.allclose(p, [0, 0.5, 2])                        # 两端 clip、中间线性内插


def test_ridge_recovers_linear_signal():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3)); y = X @ np.array([1.0, 2.0, 0.0]) + 5
    assert np.corrcoef(ridge_fit_predict(X, y, X), y)[0, 1] > 0.99


def test_repeated_group_kfold_integrity():
    groups = [f"g{i // 5}" for i in range(100)]               # 20 组 × 5 行
    g = np.asarray(groups, dtype=object)
    n_test = 0
    for tr, te in repeated_group_kfold(groups, n_splits=5, n_repeats=3, seed=0):
        assert not (set(g[tr]) & set(g[te]))                  # 同组永不跨折
        n_test += len(te)
    assert n_test == 100 * 3                                  # 每重复每行恰好一次 test


def test_oof_covers_all_rows():
    rng = np.random.default_rng(1)
    y = np.tile([0.1, 0.5, 0.9], 10)
    X = y[:, None] + rng.normal(0, 0.1, (30, 2))
    groups = [f"b{i // 3}" for i in range(30)]
    raw, cal = oof_predictions(X, y, groups, n_splits=3, n_repeats=2, seed=0)
    assert raw.shape == (30,) and np.isfinite(raw).all() and np.isfinite(cal).all()


def test_cluster_bootstrap_whole_clusters_only():
    groups = ["a"] * 3 + ["b"] * 3 + ["c"] * 3
    g = np.asarray(groups, dtype=object)
    for idx in cluster_bootstrap_indices(groups, B=20, seed=1):
        assert len(idx) == 9
        _, counts = np.unique(g[idx], return_counts=True)
        assert all(c % 3 == 0 for c in counts)                # 整簇进出


def test_adjacent_aucs_perfect_separation():
    y = np.array([0.1] * 5 + [0.3] * 5)
    proj = np.arange(10, dtype=float)                         # 完美排序
    assert adjacent_aucs(y, proj, [0.1, 0.3]) == {"0.1|0.3": 1.0}


def test_evaluate_synthetic_monotone_end_to_end():
    rng = np.random.default_rng(0)
    levels, nb = [0.1, 0.3, 0.5, 0.7, 0.9], 24
    y = np.array([s for _ in range(nb) for s in levels])
    groups = [f"b{j}" for j in range(nb) for _ in levels]
    X = np.column_stack([y + rng.normal(0, 0.35, len(y)) for _ in range(5)])
    r = evaluate(X, y, groups, prefix_dims={"main": 5, "amponly": 3, "single": 1},
                 B=100, n_splits=5, n_repeats=3, seed=0)
    assert r["n"] == nb * 5 and r["levels"] == levels
    m = r["configs"]["main"]
    assert m["rho"] > 0.6 and m["rho_ci"][0] < m["rho"] < m["rho_ci"][1] + 0.2
    assert 0.0 <= m["ba2"] <= 1.0 and 0.0 <= m["mae"]
    assert set(r["deltas"]) == {"amponly_minus_single", "main_minus_single",
                                "main_minus_amponly"}
    assert len(r["adjacent_aucs"]) == 4
    assert sum(map(sum, r["confusion_cut"]["matrix_low_high"])) == r["n"]
    assert isinstance(r["verdict_lines"], list) and len(r["verdict_lines"]) == 4
    assert set(r["gates"]) == {"primary", "aux", "tier_072", "mae_ok", "c4"}


def test_evaluate_deterministic_same_seed():
    rng = np.random.default_rng(2)
    y = np.tile([0.1, 0.5, 0.9], 12)
    X = y[:, None] + rng.normal(0, 0.2, (36, 2))
    groups = [f"b{i // 3}" for i in range(36)]
    kw = dict(prefix_dims={"main": 2, "amponly": 1, "single": 1},
              B=50, n_splits=3, n_repeats=2, seed=7)
    assert evaluate(X, y, groups, **kw) == evaluate(X, y, groups, **kw)

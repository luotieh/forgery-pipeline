import numpy as np
from checking.metrics import (roc_auc, separability_auc, balanced_accuracy,
                              spearman, NearestCentroid, linear_fit_predict,
                              pca_2d, group_split)


def test_roc_and_separability():
    assert abs(roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) - 1.0) < 1e-9
    assert abs(roc_auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) - 0.0) < 1e-9
    assert abs(separability_auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) - 1.0) < 1e-9
    assert roc_auc([1, 1], [0.5, 0.6]) == 0.5  # 单类回退


def test_balanced_accuracy_and_spearman():
    assert balanced_accuracy([0, 0, 1, 1], [0, 0, 1, 1]) == 1.0
    assert abs(balanced_accuracy([0, 0, 1, 1], [0, 0, 0, 0]) - 0.5) < 1e-9
    assert spearman([1, 2, 3, 4], [1, 2, 3, 4]) > 0.99
    assert spearman([1, 1, 1], [1, 2, 3]) == 0.0  # 退化


def test_nearest_centroid_separable():
    X = np.array([[0, 0], [0.1, 0], [5, 5], [5.1, 5]])
    clf = NearestCentroid().fit(X, ["a", "a", "b", "b"])
    assert clf.predict([[0.05, 0], [5.05, 5]]) == ["a", "b"]


def test_linear_and_pca_and_split():
    y = linear_fit_predict([[0.0], [1.0], [2.0]], [0.0, 1.0, 2.0], [[3.0]])
    assert abs(y[0] - 3.0) < 1e-6
    assert pca_2d(np.random.default_rng(0).random((8, 5))).shape == (8, 2)
    tr, te = group_split(["g1", "g1", "g2", "g3"], test_frac=0.5, seed=0)
    assert set(tr) & set(te) == set() and len(tr) + len(te) == 4


def test_bootstrap_ci():
    from checking.metrics import bootstrap_ci, separability_auc
    y = [0, 0, 0, 0, 1, 1, 1, 1]
    s = [0.1, 0.2, 0.15, 0.05, 0.9, 0.8, 0.85, 0.95]
    lo, hi = bootstrap_ci(y, s, separability_auc, n_boot=100, seed=0)
    assert 0.5 <= lo <= hi <= 1.0
    assert bootstrap_ci([0, 1], [0.1, 0.9], separability_auc) == [None, None]

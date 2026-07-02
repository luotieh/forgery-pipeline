"""闸门分析用的 sklearn-free 指标与轻量分类器。"""
from __future__ import annotations
import numpy as np
from scipy.stats import rankdata, spearmanr


def roc_auc(y_true, scores) -> float:
    y = np.asarray(y_true); s = np.asarray(scores, float)
    pos = int((y == 1).sum()); neg = int((y == 0).sum())
    if pos == 0 or neg == 0:
        return 0.5
    r = rankdata(s)
    return float((r[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def separability_auc(y_true, scores) -> float:
    a = roc_auc(y_true, scores)
    return max(a, 1.0 - a)


def balanced_accuracy(y_true, y_pred) -> float:
    y = np.asarray(y_true, dtype=object); p = np.asarray(y_pred, dtype=object)
    accs = [float((p[y == c] == c).mean()) for c in np.unique(y) if (y == c).any()]
    return float(np.mean(accs)) if accs else 0.0


def spearman(a, b) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 2 or len(np.unique(a)) < 2 or len(np.unique(b)) < 2:
        return 0.0
    rho = spearmanr(a, b).correlation
    return float(rho) if rho == rho else 0.0


class NearestCentroid:
    def fit(self, X, y):
        X = np.asarray(X, float)
        self.mu = X.mean(0); self.sd = X.std(0) + 1e-8
        Xs = (X - self.mu) / self.sd
        y = list(y)
        self.classes = sorted(set(y))
        ay = np.array(y, dtype=object)
        self.cent = {c: Xs[ay == c].mean(0) for c in self.classes}
        return self

    def predict(self, X):
        Xs = (np.asarray(X, float) - self.mu) / self.sd
        return [min(self.classes, key=lambda c: float(np.linalg.norm(x - self.cent[c])))
                for x in Xs]


def linear_fit_predict(Xtr, ytr, Xte) -> np.ndarray:
    Xtr = np.asarray(Xtr, float); Xte = np.asarray(Xte, float)
    A = np.hstack([Xtr, np.ones((len(Xtr), 1))])
    coef, *_ = np.linalg.lstsq(A, np.asarray(ytr, float), rcond=None)
    return np.hstack([Xte, np.ones((len(Xte), 1))]) @ coef


def pca_2d(X) -> np.ndarray:
    X = np.asarray(X, float)
    Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:2].T


def group_split(keys, test_frac: float = 0.4, seed: int = 0):
    keys = list(keys)
    uniq = sorted(set(keys))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq))
    n_test = max(1, int(round(len(uniq) * test_frac)))
    test_groups = {uniq[i] for i in perm[:n_test]}
    train_idx = [i for i, k in enumerate(keys) if k not in test_groups]
    test_idx = [i for i, k in enumerate(keys) if k in test_groups]
    return train_idx, test_idx


def bootstrap_ci(a, b, metric, n_boot: int = 200, seed: int = 0, alpha: float = 0.05):
    """对 metric(a[idx], b[idx]) 做自助法置信区间；样本<4 或无值返回 [None, None]。"""
    a = np.asarray(a, dtype=object); b = np.asarray(b, dtype=object)
    n = len(a)
    if n < 4:
        return [None, None]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            vals.append(float(metric(list(a[idx]), list(b[idx]))))
        except Exception:
            continue
    if not vals:
        return [None, None]
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return [round(float(lo), 4), round(float(hi), 4)]

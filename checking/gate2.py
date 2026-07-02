"""闸门 2：算子可分性 + 操作 vs 模型指纹。"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from checking import data
from checking.metrics import NearestCentroid, balanced_accuracy, group_split, pca_2d


def _acc_within(X, y, groups, seed=0):
    tr, te = group_split(groups, test_frac=0.4, seed=seed)
    if not tr or not te or len(set(y[i] for i in tr)) < 2:
        return None
    clf = NearestCentroid().fit(X[tr], [y[i] for i in tr])
    return balanced_accuracy([y[i] for i in te], clf.predict(X[te]))


def _plot(X, labels, path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    xy = pca_2d(X)
    fig, ax = plt.subplots(figsize=(6, 5))
    for lab in sorted(set(labels)):
        m = [i for i, l in enumerate(labels) if l == lab]
        ax.scatter(xy[m, 0], xy[m, 1], s=8, label=lab)
    ax.legend(fontsize=7); ax.set_title("operator PCA-2D")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100); plt.close(fig)
    return True


def run(probe_dir, extractor, max_n=None, plot_path=None) -> dict:
    probe_dir = Path(probe_dir)
    samples = data.load(probe_dir / "gate2_operator.jsonl")
    if max_n:
        samples = samples[:max_n]
    X, kept = data.profiles(extractor, probe_dir, samples)
    ops = [s.operator for s in kept]
    fams = [s.generator_family for s in kept]
    groups = [s.real_image_path or s.image_id for s in kept]
    same = []
    for f in sorted(set(fams)):
        idx = [i for i, x in enumerate(fams) if x == f]
        if len(set(ops[i] for i in idx)) < 2:
            continue
        a = _acc_within(X[idx], [ops[i] for i in idx], [groups[i] for i in idx])
        if a is not None:
            same.append(a)
    same_acc = float(np.mean(same)) if same else 0.0
    cross = []
    fam_set = sorted(set(fams))
    for fa in fam_set:
        for fb in fam_set:
            if fa == fb:
                continue
            ia = [i for i, x in enumerate(fams) if x == fa]
            ib = [i for i, x in enumerate(fams) if x == fb]
            shared = set(ops[i] for i in ia) & set(ops[i] for i in ib)
            if len(shared) < 2:
                continue
            ia = [i for i in ia if ops[i] in shared]
            ib = [i for i in ib if ops[i] in shared]
            clf = NearestCentroid().fit(X[ia], [ops[i] for i in ia])
            cross.append(balanced_accuracy([ops[i] for i in ib], clf.predict(X[ib])))
    cross_acc = float(np.mean(cross)) if cross else 0.0
    verdict = ("PASS" if same_acc >= 0.50 and cross_acc >= 0.40
               else "CONFOUND" if same_acc >= 0.50 and cross_acc < 0.30 else "WEAK")
    plotted = _plot(X, ops, plot_path) if plot_path else False
    return {"gate": 2,
            "metrics": {"same_model_acc": round(same_acc, 4),
                        "cross_model_acc": round(cross_acc, 4), "n": len(kept)},
            "verdict": verdict, "plot": plotted}

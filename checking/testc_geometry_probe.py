"""PATCH 8.3：Test-C holdout 算子几何平凡性探针（零生成，仅 GT 掩码几何）。

风险：若某算子仅凭掩码几何即可识别（如 outpaint 的边框环），它在 Test-C 上的
"成功"不构成 score 签名泛化的证据 → 不得作 Test-C holdout。
判定：geometry-only one-vs-rest AUC ≥ 0.90 → "geometry-trivial"，否则 "eligible"。

设计注记：主管线 D2 的七类操纵共用同一掩码机械（propose_masks→make_irregular，
类型轮转分配）——主库掩码几何构造上不携带算子信息；平凡性风险在 probe 网格
（outpaint=边框、background=反转框）与 Phase B 算子网格沿用的同类约定。
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from checking import data
from checking.metrics import balanced_accuracy, group_split, roc_auc

AUC_TRIVIAL = 0.90


def mask_geometry(mask: np.ndarray) -> np.ndarray:
    """5 维掩码几何：面积比 / 边界接触率 / 连通域数 / 凸性 / 质心偏移（均归一）。"""
    m = (np.asarray(mask) > 127).astype(np.uint8)
    H, W = m.shape
    area = float(m.mean())
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        return np.array([0.0, 0.0, 0.0, 1.0, 0.0], np.float32)
    border = np.concatenate([m[0], m[-1], m[:, 0], m[:, -1]])
    border_contact = float(border.mean())
    ncomp = float(cv2.connectedComponents(m)[0] - 1)
    hull = cv2.convexHull(np.stack([xs, ys], 1).astype(np.int32))
    convexity = float(min(len(ys) / max(cv2.contourArea(hull), 1.0), 1.5))
    cdist = float(np.hypot(ys.mean() / H - 0.5, xs.mean() / W - 0.5) / np.hypot(0.5, 0.5))
    return np.array([area, border_contact, ncomp, convexity, cdist], np.float32)


def logistic_ovr_scores(Xtr, ytr, Xte, iters: int = 300, lr: float = 0.5) -> np.ndarray:
    """手写 logistic（GD + 轻 L2，不罚截距），返回 test 概率分数。sklearn-free。"""
    Xtr = np.asarray(Xtr, float); Xte = np.asarray(Xte, float)
    y = np.asarray(ytr, float)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Ztr = np.hstack([(Xtr - mu) / sd, np.ones((len(Xtr), 1))])
    Zte = np.hstack([(Xte - mu) / sd, np.ones((len(Xte), 1))])
    w = np.zeros(Ztr.shape[1])
    reg = np.ones_like(w); reg[-1] = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-Ztr @ w))
        w -= lr * ((Ztr.T @ (p - y)) / len(y) + 1e-3 * reg * w / len(y))
    return 1.0 / (1.0 + np.exp(-Zte @ w))


def run(probe_dir) -> dict:
    probe_dir = Path(probe_dir)
    rows = [s for s in data.load(probe_dir / "gate2_operator.jsonl") if s.mask_path]
    X = np.array([mask_geometry(data.mask_of(probe_dir, s)) for s in rows])
    ops = [s.operator for s in rows]
    groups = [s.real_image_path or s.image_id for s in rows]
    tr, te = group_split(groups, test_frac=0.4, seed=0)
    per, dec = {}, {}
    for op in sorted(set(ops)):
        ytr = [1 if ops[i] == op else 0 for i in tr]
        yte = [1 if ops[i] == op else 0 for i in te]
        if sum(ytr) == 0 or sum(yte) == 0 or sum(ytr) == len(ytr):
            continue
        s = logistic_ovr_scores(X[tr], ytr, X[te])
        auc = roc_auc(yte, s)
        ba = balanced_accuracy(yte, list((s >= 0.5).astype(int)))
        per[op] = {"auc": round(auc, 4), "ba": round(ba, 4),
                   "n_pos": int(sum(ytr) + sum(yte))}
        dec[op] = "geometry-trivial" if auc >= AUC_TRIVIAL else "eligible"
    return {"per_operator": per, "decision": dec,
            "threshold": AUC_TRIVIAL, "n_masks": len(rows)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="testc_geometry_probe")
    ap.add_argument("--probe", default="data/probe")
    ap.add_argument("--out", default="data/testc_geometry.json")
    args = ap.parse_args(argv)
    r = run(args.probe)
    print(json.dumps(r, ensure_ascii=False, indent=2))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
    print("report ->", args.out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

"""PATCH 9.6：补充 probe 的 (cfg,steps) 单元分解（EXPLORATORY / ADDENDA 级）。

从 gate1_cfgsteps_features.npz + probe manifest 出三视图（全部锁定协议函数确定性复现，
不改任何已落盘 verdict）：
  ①切片视图：pooled 抖动模型 OOF 按 6 单元切 ρ + cluster CI；
  ②steps 边际 + seed 地板：CFG 惰性发现（空 prompt → CFG 项消去）下，同 steps 跨 cfg
    单元只差独立 seed → 其 ρ 波动即纯 seed 噪声地板；
  ③重拟合视图：仅 (7.5,30) 单元 250 行按主协议重拟合 → ρ_refit，
    分解 base_effect = ρ_main − ρ_refit（200→50 底图）与
    nuis_effect = ρ_refit − ρ_supp（同底图，固定 vs 抖动；n 250 vs 1500 为已知混淆）。
决策规则（PATCH 9.6 预定）：nuis_effect > 0.10 → method 稿脚注升级为正文 limitation。
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from checking.gate1_confirmatory import (SEED, cluster_bootstrap_indices,
                                         oof_predictions)
from checking.metrics import spearman

RHO_MAIN = 0.6999          # 主 confirmatory（n=1000，固定 nuisance）
RHO_SUPP = 0.6076          # 补充 probe pooled（n=1500，抖动）
NUIS_LIMIT = 0.10          # 9.6 决策阈值


def _ci(y, oof, groups, B=2000):
    boots = cluster_bootstrap_indices(groups, B, SEED + 1)
    lo, hi = np.percentile([spearman(y[i], oof[i]) for i in boots], [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def main() -> int:
    z = np.load("data/gate1_cfgsteps_features.npz", allow_pickle=True)
    X, y = np.asarray(z["X"], float), np.asarray(z["y"], float)
    groups, ids = list(z["groups"]), list(z["image_ids"])
    cell_of = {}
    for line in Path("data/probe_cfgsteps_gate1_strength.jsonl").read_text(
            encoding="utf-8").splitlines():
        r = json.loads(line)
        p = json.loads(r["op_params"])
        cell_of[r["image_id"]] = (float(p["cfg_scale"]), int(p["steps"]))
    cells = np.array([cell_of[i] for i in ids])
    # ---- pooled OOF（确定性复现，校验对齐补充报告） ----
    raw, _ = oof_predictions(X, y, groups)
    rho_pool = spearman(y, raw)
    assert abs(rho_pool - RHO_SUPP) < 0.01, f"pooled ρ={rho_pool} 偏离补充报告 {RHO_SUPP}"
    out = {"pooled_rho_reproduced": round(rho_pool, 4)}
    # ---- ①切片视图：6 单元 ----
    per_cell = {}
    for cfg in (5.0, 7.5, 10.0):
        for st in (30, 50):
            m = (cells[:, 0] == cfg) & (cells[:, 1] == st)
            g = [groups[i] for i in np.nonzero(m)[0]]
            per_cell[f"cfg{cfg:g}/st{st}"] = {
                "rho": round(spearman(y[m], raw[m]), 4),
                "ci": _ci(y[m], raw[m], g), "n": int(m.sum())}
    out["slice_per_cell"] = per_cell
    # ---- ②steps 边际 + seed 地板 ----
    marg = {}
    for st in (30, 50):
        m = cells[:, 1] == st
        g = [groups[i] for i in np.nonzero(m)[0]]
        marg[f"steps{st}"] = {"rho": round(spearman(y[m], raw[m]), 4),
                              "ci": _ci(y[m], raw[m], g), "n": int(m.sum())}
    out["steps_marginal"] = marg
    floor = {}
    for st in (30, 50):
        rhos = [per_cell[f"cfg{c:g}/st{st}"]["rho"] for c in (5.0, 7.5, 10.0)]
        floor[f"steps{st}"] = {"rhos": rhos, "spread": round(max(rhos) - min(rhos), 4),
                               "std": round(float(np.std(rhos)), 4)}
    out["seed_floor"] = floor
    # ---- ③重拟合视图：(7.5,30) 单元 ----
    m = (cells[:, 0] == 7.5) & (cells[:, 1] == 30)
    idx = np.nonzero(m)[0]
    g75 = [groups[i] for i in idx]
    raw75, _ = oof_predictions(X[idx], y[idx], g75)
    rho_refit = spearman(y[idx], raw75)
    base_eff = RHO_MAIN - rho_refit
    nuis_eff = rho_refit - RHO_SUPP
    out["refit_cell_7p5_30"] = {
        "rho_refit": round(rho_refit, 4), "ci": _ci(y[idx], raw75, g75),
        "base_effect(0.700-refit)": round(base_eff, 4),
        "nuis_effect(refit-0.608)": round(nuis_eff, 4),
        "decision": ("正文 limitation（nuis_effect>0.10）" if nuis_eff > NUIS_LIMIT
                     else "脚注不动（nuis_effect≤0.10）")}
    Path("data/gate1_nuisance_decomposition.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

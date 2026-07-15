"""gate1 验证性复测（一次性评估）。协议与判据锁定于 docs/PREREG_gate1_v2_2026-07-15.md。

sklearn-free：Ridge 闭式解 / PAVA isotonic / repeated group K-fold / cluster bootstrap 手写。
统计核心（evaluate 及以下纯函数）与 IO/提取分离，可 CPU 单测。
三配置共享同一折划分与同一 bootstrap 索引序列（固定 SEED，预注册 §2）。

C4 增量判定口径（脚本先于评估写死，防事后择优）：
  C4 主配对 = Δρ(amp-only − 单σ)——两者同为幅值特征、仅尺度数不同，是「多σ增量」的
  干净隔离；Δρ(main − 单σ) 与 Δρ(main − amp-only)（方向增量）如实并报、不设门。
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from checking.metrics import balanced_accuracy, roc_auc, spearman

# ---- 预注册常量（docs/PREREG_gate1_v2_2026-07-15.md；不接受运行时覆盖） ----
SEED = 20260715                 # 折划分 seed；bootstrap 用 SEED+1
B_BOOT = 2000                   # §2 cluster bootstrap 次数
N_SPLITS, N_REPEATS = 5, 20     # §2 OOF
ALPHA = 1.0                     # §2 Ridge
CUT = 0.4                       # P3：第 2/3 档之间（{0.1,0.3} vs {0.5,0.7,0.9}）
RHO_GATE = (0.50, 0.30)         # §3.1（点值, CI 下界）
BA_GATE = (0.66, 0.55)          # §3.2
BA_TIER = 0.72                  # §3.3
MAE_GATE = 0.15                 # §3.5
AUC_FLOOR = 0.55                # §3.6 s* 判定
AMP_DIMS, SINGLE_DIMS = 17, 1   # amp-only=基类幅值前缀；单σ=t50 均值（P1 profile 布局）


# ---------- 统计核心（纯 numpy/scipy，可单测） ----------

def ridge_fit_predict(Xtr, ytr, Xte, alpha: float = ALPHA) -> np.ndarray:
    """折内标准化 + 闭式 Ridge（含截距）。标准化统计只在训练折 fit，防泄漏。"""
    Xtr = np.asarray(Xtr, float); Xte = np.asarray(Xte, float)
    ytr = np.asarray(ytr, float)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
    yc = ytr - ytr.mean()
    w = np.linalg.solve(Ztr.T @ Ztr + alpha * np.eye(Ztr.shape[1]), Ztr.T @ yc)
    return Zte @ w + ytr.mean()


def pava_isotonic_fit(x, y):
    """递增 isotonic（PAVA）。返回 (ux, fit) 供 np.interp 预测；相同 x 先加权合并。"""
    x = np.asarray(x, float); y = np.asarray(y, float)
    order = np.argsort(x, kind="stable")
    ux, inv = np.unique(x[order], return_inverse=True)
    w = np.bincount(inv).astype(float)
    yy = np.bincount(inv, weights=y[order]) / w
    vs, ws, members = [], [], []
    for j in range(len(ux)):
        v, ww, mem = float(yy[j]), float(w[j]), [j]
        while vs and vs[-1] > v + 1e-15:          # 违反单调 → 池化
            pv, pw, pm = vs.pop(), ws.pop(), members.pop()
            v = (v * ww + pv * pw) / (ww + pw); ww += pw; mem = pm + mem
        vs.append(v); ws.append(ww); members.append(mem)
    fit = np.empty(len(ux))
    for v, mem in zip(vs, members):
        fit[mem] = v
    return ux, fit


def pava_predict(ux, fit, xnew) -> np.ndarray:
    """线性内插 + 两端 clip（与 sklearn IsotonicRegression 预测行为一致）。"""
    return np.interp(np.asarray(xnew, float), ux, fit)


def repeated_group_kfold(groups, n_splits: int = N_SPLITS, n_repeats: int = N_REPEATS,
                         seed: int = SEED):
    """按 group 的 repeated K-fold：同 group 永不跨折；每重复内每行恰好当一次 test。"""
    codes = np.unique(np.asarray(groups, dtype=object), return_inverse=True)[1]
    n_groups = codes.max() + 1
    rng = np.random.default_rng(seed)
    for _rep in range(n_repeats):
        perm = rng.permutation(n_groups)
        for fold in np.array_split(perm, n_splits):
            te = np.isin(codes, fold)
            yield np.nonzero(~te)[0], np.nonzero(te)[0]


def oof_predictions(X, y, groups, alpha: float = ALPHA, n_splits: int = N_SPLITS,
                    n_repeats: int = N_REPEATS, seed: int = SEED):
    """返回 (oof_raw, oof_cal)：对 repeats 取平均的 OOF 预测；校准嵌套在折内
    （isotonic 在训练折 in-sample 预测上 fit，apply 到 test 折）。"""
    X = np.asarray(X, float); y = np.asarray(y, float)
    acc_r = np.zeros(len(y)); acc_c = np.zeros(len(y)); cnt = np.zeros(len(y))
    for tr, te in repeated_group_kfold(groups, n_splits, n_repeats, seed):
        pr_te = ridge_fit_predict(X[tr], y[tr], X[te], alpha)
        pr_tr = ridge_fit_predict(X[tr], y[tr], X[tr], alpha)
        ux, fit = pava_isotonic_fit(pr_tr, y[tr])
        acc_r[te] += pr_te; acc_c[te] += pava_predict(ux, fit, pr_te); cnt[te] += 1
    return acc_r / cnt, acc_c / cnt


def cluster_bootstrap_indices(groups, B: int = B_BOOT, seed: int = SEED + 1):
    """按底图 cluster 重采样的行索引序列（整簇进出）；三配置共享同一序列。"""
    codes = np.unique(np.asarray(groups, dtype=object), return_inverse=True)[1]
    rows_of = [np.nonzero(codes == k)[0] for k in range(codes.max() + 1)]
    rng = np.random.default_rng(seed)
    return [np.concatenate([rows_of[k] for k in rng.integers(0, len(rows_of), len(rows_of))])
            for _ in range(B)]


def _boot_ci(idx_list, fn):
    vals = [v for v in (fn(i) for i in idx_list) if v == v]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def _ba2(y, pred_cal, idx, cut: float = CUT) -> float:
    bt = (y[idx] >= cut).astype(int); bp = (pred_cal[idx] >= cut).astype(int)
    return balanced_accuracy(list(bt), list(bp))


def adjacent_aucs(y, proj, levels):
    """相邻档位 (a,b) 上 1-D 投影的可分性 AUC（§3.6，每点 ~2×n_base 样本）。"""
    out = {}
    for a, b in zip(levels[:-1], levels[1:]):
        m = np.isclose(y, a) | np.isclose(y, b)
        out[f"{a:g}|{b:g}"] = round(roc_auc((np.isclose(y[m], b)).astype(int), proj[m]), 4)
    return out


def evaluate(X, y, groups, prefix_dims: dict | None = None, cut: float = CUT,
             B: int = B_BOOT, n_splits: int = N_SPLITS, n_repeats: int = N_REPEATS,
             seed: int = SEED, alpha: float = ALPHA) -> dict:
    """预注册 §3 全套指标 + §4 机械 verdict。prefix_dims: 各配置取 X 的前缀维数。"""
    X = np.asarray(X, float); y = np.asarray(y, float)
    dims = prefix_dims or {"main": X.shape[1], "amponly": AMP_DIMS, "single": SINGLE_DIMS}
    boots = cluster_bootstrap_indices(groups, B, seed + 1)
    levels = sorted(np.unique(np.round(y, 6)).tolist())
    res, oof = {}, {}
    for name, d in dims.items():
        raw, cal = oof_predictions(X[:, :d], y, groups, alpha, n_splits, n_repeats, seed)
        oof[name] = (raw, cal)
        res[name] = {
            "rho": round(spearman(y, raw), 4),
            "rho_ci": _boot_ci(boots, lambda i, r=raw: spearman(y[i], r[i])),
            "ba2": round(_ba2(y, cal, np.arange(len(y)), cut), 4),
            "ba2_ci": _boot_ci(boots, lambda i, c=cal: _ba2(y, c, i, cut)),
            "mae": round(float(np.mean(np.abs(cal - y))), 4),
            "mae_ci": _boot_ci(boots, lambda i, c=cal: float(np.mean(np.abs(c[i] - y[i])))),
        }
    deltas = {}
    for tag, (a, b) in {"amponly_minus_single": ("amponly", "single"),
                        "main_minus_single": ("main", "single"),
                        "main_minus_amponly": ("main", "amponly")}.items():
        ra, rb = oof[a][0], oof[b][0]
        deltas[tag] = {
            "delta_rho": round(spearman(y, ra) - spearman(y, rb), 4),
            "ci": _boot_ci(boots, lambda i: spearman(y[i], ra[i]) - spearman(y[i], rb[i])),
        }
    raw_m, cal_m = oof["main"]
    aucs = adjacent_aucs(y, raw_m, levels)
    s_star = "right-censored（全部边界 AUC≥0.55）"
    for (a, b), v in zip(zip(levels[:-1], levels[1:]), aucs.values()):
        if v < AUC_FLOOR:
            s_star = f"{(a + b) / 2:g}"
            break
    bt = (y >= cut).astype(int); bp = (cal_m >= cut).astype(int)
    conf = [[int(((bt == i) & (bp == j)).sum()) for j in (0, 1)] for i in (0, 1)]
    # ---- §4 机械 verdict（主配置；全部携带 P4=否限定） ----
    m = res["main"]
    primary = m["rho"] >= RHO_GATE[0] and m["rho_ci"][0] > RHO_GATE[1]
    aux = m["ba2"] >= BA_GATE[0] and m["ba2_ci"][0] > BA_GATE[1]
    tier = aux and m["ba2"] >= BA_TIER
    mae_ok = m["mae"] <= MAE_GATE
    q = "（固定 CFG/steps 条件下；P4=否）"
    if primary and mae_ok:
        v = f"t0 逆估计（粗分辨率）成立{q}；(t0,c,M) 三坐标全量，上行期权兑现"
    elif primary:
        v = f"强序数恢复 + 粗桶强度分级{q}；不以「逆估计」修饰 t0（MAE 未过 §3.5）"
    elif m["rho"] >= 0.30:
        v = (f"t0 降级为粗桶强度分级{q}（辅判据过）" if aux
             else f"t0 降级为序数陈述{q}（辅判据未过）") + "；G-A 走降级线"
    else:
        v = "t0 主张撤下；G-A 走兜底分支（定位 + 算子族）"
    c4 = deltas["amponly_minus_single"]["ci"][0] > 0
    lines = [v,
             ("C4 多σ增量成立（Δρ CI 下界>0）" if c4
              else "C4：论文删「多尺度对 t0 的增量」表述（Δρ CI 下界≤0；检测/定位主张不受管辖）"),
             f"s* = {s_star}",
             ("§3.3 加档：许可「信息量不低于三桶 BA 0.55」声明" if tier
              else "§3.3 加档未触发" if aux else "§3.2 辅判据未过，无加档")]
    return {"configs": res, "deltas": deltas, "adjacent_aucs": aucs, "s_star": s_star,
            "confusion_cut": {"cut": cut, "matrix_low_high": conf},
            "gates": {"primary": primary, "aux": aux, "tier_072": tier, "mae_ok": mae_ok,
                      "c4": c4},
            "verdict_lines": lines, "n": int(len(y)), "levels": levels}


# ---------- IO / 提取 / 报告（GPU 侧） ----------

def _extract(probe_dir: Path, npz_path: Path):
    """direction ON 单次提取；npz 缓存 profile + latent-native 64×64 逐 t 残差图（§3.6）。"""
    from checking import data
    from checking.extractor import get_extractor
    import cv2
    samples = data.load(probe_dir / "gate1_strength.jsonl")
    ids = [s.image_id for s in samples]
    if npz_path.exists():
        z = np.load(npz_path, allow_pickle=True)
        if list(z["image_ids"]) == ids:
            print(f"npz 缓存命中：{npz_path}")
            return z["X"], z["y"], list(z["groups"]), z
    ext = get_extractor("real")
    assert ext.direction_features, "预注册主配置=幅值+方向，方向开关必须为开"
    X, y, groups, eps64, x64 = [], [], [], [], []
    for i, s in enumerate(samples):
        img = data.image_of(probe_dir, s)
        X.append(ext.profile(img))
        y.append(float(s.strength))
        groups.append(s.real_image_path or s.image_id)
        eps64.append(np.stack([cv2.resize(m, (64, 64)) for m in ext._eps_stack]))
        x64.append(np.stack([cv2.resize(m, (64, 64)) for m in ext._x_stack]))
        if (i + 1) % 100 == 0:
            print(f"extract {i + 1}/{len(samples)}")
    X = np.asarray(X, np.float32)
    assert X.shape[1] == AMP_DIMS + 19, f"profile 维度 {X.shape[1]} ≠ 36（P1 布局）"
    np.savez_compressed(
        npz_path, X=X, y=np.asarray(y, np.float32),
        groups=np.asarray(groups, object), image_ids=np.asarray(ids, object),
        eps64=np.asarray(eps64, np.float16), x64=np.asarray(x64, np.float16),
        manifest_sha1=hashlib.sha1((probe_dir / "gate1_strength.jsonl").read_bytes()).hexdigest())
    return X, np.asarray(y, float), groups, None


def _report_md(r: dict, probe_dir: Path, npz_path: Path) -> str:
    sha = hashlib.sha1((probe_dir / "gate1_strength.jsonl").read_bytes()).hexdigest()
    L = [f"# gate1 验证性复测报告（2026-07-15）\n",
         "预注册：`docs/PREREG_gate1_v2_2026-07-15.md`（锁定版）；一次性评估。\n",
         "## DATA",
         f"- probe: `{probe_dir}`（n={r['n']}，档位 {r['levels']}），manifest sha1 `{sha[:12]}`",
         "- extractor: **real**（冻结 SD1.5 多σ Tweedie 残差，P1=54e207b，幅值+方向）",
         f"- 特征/残差图缓存: `{npz_path}`（64×64 latent-native 逐 t 图）",
         "- P4=否 → 所有结论限定「固定 CFG/steps（steps=30, cfg=7.5）」\n",
         "## CRITERIA（预注册 §3，数值锁定）",
         "主判据 ρ≥0.50 且 CI下界>0.30；辅判据 cut=0.4 2桶 BA≥0.66 且 CI下界>0.55；"
         "加档 BA≥0.72；MAE≤0.15；C4=Δρ(amp-only−单σ) CI下界>0；CI=按底图 cluster bootstrap B=2000\n",
         "## RESULTS",
         "| 配置 | ρ [95%CI] | 2桶BA(cut0.4) [95%CI] | MAE [95%CI] |", "|---|---|---|---|"]
    for k, lbl in [("main", "幅值+方向(主)"), ("amponly", "amp-only"), ("single", "单σ t50")]:
        c = r["configs"][k]
        L.append(f"| {lbl} | {c['rho']:.4f} {c['rho_ci']} | {c['ba2']:.4f} {c['ba2_ci']} "
                 f"| {c['mae']:.4f} {c['mae_ci']} |")
    L += ["", "| 配对 Δρ | 点值 | 95%CI |", "|---|---|---|"]
    for k, d in r["deltas"].items():
        L.append(f"| {k} | {d['delta_rho']:+.4f} | {d['ci']} |")
    L += ["", f"相邻档位 AUC：{json.dumps(r['adjacent_aucs'], ensure_ascii=False)}",
          f"s\\* = {r['s_star']}",
          f"切点混淆（行=true low/high，列=pred）：{r['confusion_cut']['matrix_low_high']}\n",
          "## VERDICT（§4 机械抄写）"]
    L += [f"- {x}" for x in r["verdict_lines"]]
    L += ["", "## EXPLORATORY-ADDENDA", "（此后任何追加分析置于此节并显式标注。当前：无。）"]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="gate1_confirmatory")
    ap.add_argument("--probe", default="data/probe_real_n200")
    ap.add_argument("--npz", default="data/gate1_n200_features.npz")
    ap.add_argument("--out", default="data/gate1_confirmatory.json")
    ap.add_argument("--report", default="checking/gate1_confirmatory_report_2026-07-15.md")
    a = ap.parse_args(argv)
    report = Path(a.report)
    if report.exists():
        print(f"REFUSE：{report} 已存在——预注册规定一次性评估，不得重跑覆盖。")
        return 2
    probe = Path(a.probe)
    X, y, groups, _ = _extract(probe, Path(a.npz))
    r = evaluate(X, y, groups)
    Path(a.out).write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
    report.write_text(_report_md(r, probe, Path(a.npz)), encoding="utf-8")
    print("\n===== VERDICT（§4 机械导出） =====")
    for line in r["verdict_lines"]:
        print(" -", line)
    print("report ->", report, "| json ->", a.out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

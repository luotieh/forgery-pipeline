# Test-C holdout 几何平凡性探针报告（PATCH 8.3，2026-07-16）

> **裁定**：**Test-C holdout = `object_replacement`**（增补默认选择被数据确认）。
> `outpaint` 与 `background_editing` 均 geometry-trivial（AUC=1.0），**不得**作 Test-C holdout。

## DATA
- mock probe `n_base=100`（`configs/probe.yaml`，本地 CPU 生成）→ **1600 张掩码**（4 masked 算子 × 400）。
- 掩码几何与后端无关的依据：probe 掩码由 `_mask_for`（box/border/invert 几何原语 + 固定 seed rng）生成，mock 与 real 后端走同一代码路径——几何分布逐字节同机械。零 GPU、零真实生成。

## METHOD
- 5 维纯几何特征：面积比 / 边界接触率 / 连通域数 / 凸性 / 质心偏移（`mask_geometry`）。
- one-vs-rest 手写 logistic（GD+轻L2，sklearn-free），按底图 group split（60/40，seed=0）。
- 预定判定规则（增补 8.3，先于实验写死）：geometry-only AUC ≥ **0.90** → geometry-trivial。

## RESULTS（n_pos=400/算子）

| 算子 | geometry-only AUC | BA | 判定 |
|---|---|---|---|
| outpaint | **1.000** | 0.995 | **geometry-trivial** ❌ |
| background_editing | **1.000** | 1.000 | **geometry-trivial** ❌ |
| inpaint | 0.829 | 0.671 | eligible ✅ |
| object_replacement | 0.829 | 0.656 | eligible ✅ |

**决定性配对**：inpaint vs object_replacement 两两几何 AUC = **0.487**（n=800）≈ 机会线——两者共用同一 box 机械，OvR 的 0.83 全部来自与环形算子（outpaint/background）的区分，**组内无几何信息**。

## DECISION（写回 experiment plan §3 B3）
- **Test-C holdout = `object_replacement`**：holdout 它时，模型无法凭掩码几何"作弊"识别（与留在训练里的 inpaint 几何同分布），Test-C 的成功才构成 score 签名泛化证据。
- outpaint / background_editing 留在训练可见集合；论文若报它们的归因准确率，须注明几何可分性上界（AUC=1.0）。

## 设计注记与有效期
- 主管线 D2 的七类操纵**共用同一掩码机械**（`propose_masks`→`make_irregular`，类型轮转分配）→ 主库掩码几何构造上不携带算子信息；几何平凡性风险只在算子网格的约定掩码（border/invert）。
- **失效条件**：Phase B 若更改算子×掩码约定（如 outpaint 改用非边框掩码），本报告失效，须重跑 `python -m checking.testc_geometry_probe`。
- 备选方案（增补 8.3，默认不选）：`instruct_edit` 作 Test-C holdout——留作 G-A 后可选强化。

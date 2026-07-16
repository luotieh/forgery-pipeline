# gate1 nuisance 单元分解（PATCH 9.6，2026-07-16）— EXPLORATORY / ADDENDA

> **地位**：探索性 ADDENDA，不改任何已落盘 verdict（prereg v2 与补充 probe 结论不动）。
> 输入：`gate1_cfgsteps_features.npz`（n=1500）+ probe manifest cell 标签；全部数字由锁定协议函数确定性复现（pooled ρ 复现 0.6076 与补充报告一致）。
> 前提发现：**空 prompt 下 CFG 惰性**（cond≡uncond → CFG 项消去），6 单元有效坍缩为 steps 30/50 两单元，同 steps 跨 cfg 差异=各单元独立 seed。

## ①切片视图（pooled OOF 按单元切，cluster CI，250 行/50 簇每单元）

| cell | ρ | 95% CI |
|---|---|---|
| cfg5/st30 | 0.650 | — |
| cfg7.5/st30 | 0.733 | — |
| cfg10/st30 | 0.744 | — |
| cfg5/st50 | 0.481 | — |
| cfg7.5/st50 | 0.511 | — |
| cfg10/st50 | 0.561 | — |

（逐单元 CI 见 `data/gate1_nuisance_decomposition.json`；50 簇 CI 偏宽如实注明。）

## ②steps 边际 + seed 地板 —— **敏感维几乎全在 steps**

| 边际 | ρ | 95% CI | n |
|---|---|---|---|
| **steps=30** | **0.7065** | [0.669, 0.744] | 750 |
| **steps=50** | **0.5141** | [0.443, 0.587] | 750 |

- steps=30 边际 ρ≈0.707 ≈ 主 confirmatory 的 0.700（CI 重叠）——**在"主场"步数下，抖动数据完全复现主结论**。
- steps 30→50：**Δρ ≈ −0.19**（边际 CI 不重叠）——补充 probe 的 pooled 跌幅 0.092 是双峰混合的平均，**低估了单维效应**。
- seed 地板：同 steps 跨 cfg 的 ρ 波动 spread 0.08–0.09（std 0.03–0.04），与单元级估计噪声（50 簇 CI 半宽）同量级 → **看不出超出抽样噪声的 seed 效应**。

## ③重拟合视图（(7.5,30) 单元 250 行按主协议重拟合）

- ρ_refit = **0.7207** [0.658, 0.782]
- base_effect = 0.700 − 0.721 = **−0.021**（50 底图子集不比 200 底图难；底图子集差异≈无）
- **nuis_effect = 0.721 − 0.608 = +0.113 > 0.10** ← 9.6 预定决策阈值

## 决策（9.6 预定规则机械执行）

**nuis_effect > 0.10 → 「固定 CFG/steps」限定由脚注升级为正文 limitation。**

正文 limitation 的精确措辞要素（按本分解收紧）：
1. t0 序数恢复对**去噪步数敏感**：steps 30→50 边际 ρ 0.707→0.514（空 prompt 域）；
2. **CFG 维在空 prompt 机制下惰性**（本轮所有 probe 均空 prompt），prompted 域的 CFG 敏感度**未测**；
3. 主库（Phase B）已裁定逐图抖动 CFG/steps + prompt bank（PATCH 9.1/9.2），Phase C/D 系统级结果将在抖动数据上训练评测，不再携带此限定——限定仅约束 gate1 的 t0 序数主张本身。

## 已知混淆（如实）

- 重拟合视图 n=250 vs 补充 pooled n=1500（n 差异为已知混淆，已在 9.6 设计中注明）；
- steps 边际的两组 750 行共享同 50 底图（簇间独立性由 cluster CI 处理）；
- 本分解全部在空 prompt 域内，结论不外推到 prompted 生成。

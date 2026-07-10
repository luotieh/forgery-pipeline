# 下一步优化方向与计划（2026-07-09）

> **一句话**：工程已跑通，瓶颈在**方法与实验覆盖**，不在代码。按杠杆排序有三条路径——P0 先跑已就绪的跨生成器实验拿到真信号，P1 若要救 gate1 必须换方向特征（扩样本无用，幅值已触顶），P2 是降级 gate1、把论文重心移到 gate0 + 跨生成器泛化 gap 的兜底方案。

## 现状锚点（核实自代码与 `real_gate_results_2026-07-02.md`）

| 闸门 | VERDICT | 关键指标 | 诊断 |
|---|---|---|---|
| gate0 残差可分 | PASS | det 0.688 / loc 0.643 | 过最低门槛但 <0.7，且含"SD1.5 检 SD1.5"自家先验偏置，跨模型未验 |
| gate1 t0 可恢复 | WEAK | BA 0.475 / ρ 0.476 / 多σ−单σ **+0.025** | **幅值特征触顶**——见下方根因 |
| gate2 算子可分 | WEAK | same 0.372 / cross 0.374 | 所有 inpainter 底层同一 SD1.5，cross_model 当前无信息 |
| gate3 跨生成器 | PARTIAL | delta +0.025 / heldout 结构性 0 | 单模型下 `holdout_generators:[]` → 恒 0，不可测 |

**gate1 触顶根因（代码级）**：`extractor.py:33-38` 的 `profile()` 只输出幅值统计——per-scale `mean/std` + 聚合分位数 + 区域几何。而 `DiffusersResidual.residual_stack`（`extractor.py:120`）把本来分离的 `r_eps`（ε 预测误差）与 `r_x`（x0 重建误差）**直接相加坍缩成一张标量残差图**，逐 t 的**方向/相位信息在提取器出口就丢了**。真实 SDEdit 去噪轨迹在低/高 strength 处饱和，幅值随 strength 单调但压缩，所以多σ相对单σ只多 +0.025。**扩样本救不了触顶——只有换特征能救。**

---

## P0 — 跑已就绪的 Kandinsky 跨生成器实验【最高杠杆，先做】

**为什么先做**：代码已提交（`474484e`，`configs/generators.real.yaml` + `holdout_generators:[kandinsky-inpaint]`），只差一次 GPU 运行（约 2h）。一次运行同时解锁三件事，且不依赖任何新方法：

1. **gate3 真实掉点**：Kandinsky 2.2 是异族 unCLIP 先验，heldout split 非空后 heldout_acc 首次可测 → 回答"跨生成器泛化 gap 到底多大"。
2. **gate2 cross_model 变得有信息**：SD1.5-inpaint vs Kandinsky-inpaint 是两个真实不同算子族，`cross_model` 不再是同一模型的标签幻觉。
3. **判定 gate0 的 0.688 是真信号还是先验偏置**：用冻结 SD1.5 残差去检 **Kandinsky 生成的伪造**——若 AUC 仍显著 >0.5，gate0 是可迁移的真信号；若塌到 0.5，则 0.688 主要是"自家先验检自家生成"的偏置。**这是 gate0 能否作为论文承重结论的判决性实验。**

**注意边界**：Kandinsky 只走 inpaint 支路，**只影响 gate2/3 和 gate0 的跨模型检验，修不了 gate1**——gate1 的 strength 网格仍是 SD1.5 img2img 单线。

**任务清单**
- [ ] 云 GPU 上按 `configs/generators.real.yaml` 放量生成（n_base=50，含 SD1.5 + Kandinsky-inpaint 两族）
- [ ] `python -m checking.run_gates --extractor real`，产出新的 `data/checking_report_real.json`（覆盖当前仍是单模型的旧报告）
- [ ] 额外单跑"SD1.5 残差检 Kandinsky 伪造"的 gate0 AUC（拆出 Kandinsky-only 子集）
- [ ] 记录 gate3 heldout_acc、gate2 same vs cross 的真实差、gate0 跨模型 AUC 三个数

**验收/成本**：约 2h GPU（80 min 生成 + 30 min 分析 + 拆子集）。产出直接改写四张 VERDICT 里的 gate2/3 与 gate0 迁移性结论。

**一键脚本（云 GPU 上跑）**：`bash scripts/run_cross_generator_p0.sh`（冒烟自检 `N_BASE=8 bash ...`）。它串起「装 `[real]` 依赖 → 下底图 → `forgery_pipeline.cli probe`（SD1.5+Kandinsky）→ `checking.run_gates --extractor real` → `scripts/gate0_cross_generator.py` 跨生成器迁移判决」，末尾打印 gate2 cross/gate3 heldout/gate0 迁移 AUC 三个关键数。跑完把 `data/checking_report_real.json` 与 `data/gate0_cross_generator.json` 拷回本机更新结论。

---

## P1 — gate1 方向特征升级【若要 gate1 当承重贡献，唯一出路】

**触发条件**：只有当论文需要"t0 可恢复 / 强度回归"作为**承重假设**时才做。否则走 P2 降级。

**核心改动**：不再把 `r_eps` 和 `r_x` 相加，改为保留**逐 t 的误差向量方向**，把方向/相位特征拼进 `profile()`。候选特征（按实现成本递增）：

1. **分离 r_eps / r_x 双通道**（最小改动）：`extractor.py:120` 不再 `r_eps + r_x`，两张图分别进 profile → 特征维度翻倍，先看 delta 是否回升。
2. **逐 t 残差的方向余弦**：相邻 t 之间 ε 误差图的空间相关/夹角，捕捉去噪轨迹形状而非幅值。
3. **DIRE 式反演重建距离**：DDIM 反演→重建，编辑区重建误差与 strength 更单调（文献验证过对 SDEdit 强度更敏感）。
4. **频域特征**：残差的径向功率谱斜率，img2img 全图重绘在高频段有特征印记。

**方法学纪律**：先做①（几乎零成本），在**现有 SD1.5 probe 数据**上离线复算 gate1（不用重新生成图！残差提取器可对已存图重跑）。若 ①的多σ−单σ delta 从 +0.025 回升到 >0.1，再投入 ②③。**每一步都要有 delta 回升证据才继续，避免在无效方向堆特征。**

**任务清单**
- [ ] `extractor.py`：`residual_stack` 增开 `return_channels` 选项，保留 r_eps/r_x 分离
- [ ] `profile()`：拼入分离双通道统计（保持向后兼容，旧 multisigma 走原路径）
- [ ] 在已有 `data/probe_real` 上离线复跑 gate1，对比 delta（无需 GPU 生成，仅残差提取）
- [ ] delta 回升才进 ②方向余弦；否则记录"幅值+方向均触顶"并转 P2

**验收**：多σ−单σ delta 显著回升（目标 >0.1）且 BA CI 下界超过 0.55 门槛，gate1 才够格 PASS。

---

## P2 — gate1 降级 + 论文重心转移【兜底，与 P1 二选一或并行准备】

若 P1 的 ① 证明方向特征同样触顶，**不要硬拗 gate1 回归**。改为：

- **gate1 降级为粗粒度强度分级**（2 桶 low/high 而非 3 桶回归），如实报告"强度信息部分留存但不足以精确回归"——这本身是诚实且可发表的负结果。
- **论文承重结论移到 gate0 + 跨生成器泛化 gap**：即"冻结单一扩散先验的多σ残差能检测/定位真实伪造，且跨异族生成器的掉点为 X"（X 由 P0 给出）。这条线不依赖 gate1 触顶的死结。
- gate2 的"算子指纹存在但弱"作为辅助证据，配 P0 的真实 cross_model 数据。

**任务清单**
- [ ] `gate1.py:_bucket` 增 2 桶模式，VERDICT 文案改为"粗粒度分级"口径
- [ ] 论文 outline（`docs/PAPER_DESIGN.md`）重排：承重假设 = gate0 可分性 + P0 泛化 gap；gate1 移为次要
- [ ] 所有 VERDICT 措辞与 `real_gate_results` 保持"诚实边界"一致，不复现 mock 高估

---

## 决策树（执行顺序）

```
P0 跑 Kandinsky 跨生成器实验（无条件先做，~2h GPU）
  └─ gate0 检 Kandinsky AUC 显著 >0.5 ?
       ├─ 是 → gate0 是可迁移真信号，作论文承重结论 ✅
       └─ 否 → gate0 主要是先验偏置，论文重心须另寻（强调 gate3 gap 本身）
  │
  ├─ 论文是否需要 gate1 当承重贡献 ?
  │    ├─ 是 → P1 ①（零成本离线复算）→ delta 回升?→ 是则②③，否则转 P2
  │    └─ 否 → P2 直接降级 gate1
```

## 约束与提醒（沿用项目纪律）

- **WSL 不跑 GPU**：本机只做实验设计 + 代码实现，P0/P1 的 GPU 运行交云服务器（见记忆 `no-gpu-in-wsl-scripts-for-cloud`）；长跑前起 Windows 侧保活 `wsl.exe`（见 `wsl-kills-background-gpu-runs`）。
- **诚实边界**：任何新 VERDICT 都区分 extractor（real vs multisigma 代理），不复现 mock 的乐观读数。
- P1 的离线复算是关键成本杠杆——**残差提取器可对已存图重跑，改特征不必重新生成数据集**。
</content>
</invoke>

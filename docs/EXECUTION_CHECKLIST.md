# EXECUTION_CHECKLIST — 风险闸门执行清单（供 Claude Code）

> 项目：**Inverting the Edit**（扩散式图像编辑的检测 / 定位 / 编辑算子归因）
> 用途：在投入完整系统与大量算力前，用便宜实验逐个**证伪或证实**承重假设。
> 原则：**每个闸门都是 go/no-go**，先把通过标准定死，再看数据；按"最致命且最便宜"排序。
> 两个代码基：
> - 闸门实验脚手架 `gate_experiments/`（独立，直接用 diffusers 生成 + 分析）。
> - 数据集 pipeline `forgery-pipeline/`（生成带 `strength`/`operator` 标签的受控 probe 子集，见 PATCHES.md 的 probe 补丁）。

---

## 0. 前置

- [ ] 准备真实底图目录（几百张即可，如 COCO-val 子集），填入 `gate_experiments/config.py:REAL_IMAGE_DIR`。
- [ ] 安装依赖：`pip install "torch>=2.1" torchvision "diffusers>=0.27" "transformers>=4.38" accelerate scikit-learn scipy matplotlib pillow numpy`。
- [ ] **先冒烟**：`config.py` 把 `G1_N_PER_STRENGTH=4`、`G2_N_PER_CELL=4`，端到端跑通四个脚本，清掉 Hub/版本问题，再恢复成 150。
- [ ] 算力预算：闸门总开销应 < 全部算力的 5%；前三关合计数个 GPU 小时。

---

## 闸门 0 — 信号地基（最先、最便宜）

**测**：复现 DIRE/AEROBLADE 的单尺度重建/去噪残差，在你的成对数据上能否把"真实 vs 编辑区"分开。
**为什么第一**：地基 + 别人已验证，作用是排查管线 bug（latent 对齐 / 归一化 / 掩码），不是验证新东西。

- [ ] 跑单尺度残差 → 检测 AUC + 定位热图肉眼可见。
- [ ] **过**：能分开 → 继续闸门 1。
- [ ] **不过**：是工程 bug（多半是 latent/分辨率处理），先修管线，**不要往下走**。

---

## 闸门 1 — t0 可恢复性（最承重，决定数学是否成立）

**测**：img2img/SDEdit 在已知强度 0.1–0.9 生成，多 σ 残差剖面能否回出 t0。
**承重点**："类型识别 = 算子逆估计"整套压在这一关。

- [ ] 生成数据（二选一）：`python gate_experiments/gate1_generate.py`；或用 pipeline 的 probe 强度网格（PATCHES.md PATCH 4）产出带 `strength` 的 manifest。
- [ ] 分析：`python gate_experiments/gate1_analyze.py` → 读 **GATE 1 VERDICT**。
- [ ] 同时看 `[multi-sigma]` vs `[VAE baseline]`：多 σ 是否优于单尺度（"多 σ"这个词能否用）。

| 结果 | 判据 | 行动 |
|---|---|---|
| **PASS** | 三分类平衡准确率 ≥ 0.55 **且** Spearman ρ ≥ 0.30 | 继续闸门 2 |
| **WEAK** | 平衡准确率 ∈ [0.45, 0.55) | 改粗桶 / 加数据，重测 |
| **FAIL** | ≈ 随机 (0.333) | **砍掉 t0 逆估计理论**，转判别式分类 + 流形/曲率定位（子创新 3） |

> 预期：粗桶大概率可分；细粒度回归未必准（低强度 + VAE 往返后变噪）。粗桶能分即算过，别要求 R² 漂亮。

---

## 闸门 2 — 算子可分性 + 操作 vs 模型指纹（审稿人必攻击点）

**测**：五类算子 {img2img, inpaint, outpaint, replace, background} 是否可分，且信号来自**操作**而非**模型指纹**。

- [ ] 生成数据：`python gate_experiments/gate2_generate.py`（5 算子 × ≥2 生成器族）。
- [ ] 分析：`python gate_experiments/gate2_analyze.py` → 读 **GATE 2 VERDICT** + `tsne_operators.png`。
- [ ] 关键：**同模型内** 与 **跨模型**（训 A 测 B，双向）一起跑，差距即混淆诊断。

| 结果 | 判据 | 含义 / 行动 |
|---|---|---|
| **PASS** | 同模型 ≥ 0.50 **且** 跨模型均值 ≥ 0.40 | 学到操作，继续 |
| **CONFOUND** | 同模型高、跨模型 < 0.30 | **学到模型指纹** → 加生成器族 + 把主张收窄为"算子族识别"（差距本身可写成发现） |
| **WEAK** | 同模型 < ~0.50 | 该信号下算子勉强可分 |

> 预期：算子族可分，但近似扩散 inpainter 之间会混淆（与归因文献"扩散-扩散难分"一致）；跨模型预期掉点，但不应到随机。

---

## 闸门 3 — 多 σ 增量 + 跨生成器掉点（确定"多尺度""泛化"两个词能否用）

**测**：(a) 多 σ 剖面相对 AEROBLADE 单尺度的增量；(b) 固定先验算的残差在**留出模型族**编辑上是否崩。

- [ ] 消融：多 σ vs 单尺度的检测/定位差值。
- [ ] 跨生成器：在留出族（pipeline 的 `test_b` / probe 的留出 inpainter）上测掉点曲线。

- [ ] **过**：多 σ 有增量、跨生成器部分掉点（可接受）→ 进闸门 4。
- [ ] **多 σ 无增量** → 删多尺度框架，故事改挂算子逆估计 + 定位。
- [ ] **跨生成器崩得厉害** → 这正是第二篇泛化论文的动机；第一篇把它写成"被刻画的局限 + 部分缓解"，不硬撑。

---

## 闸门 4 — 完整系统与规模化（前面全过才做）

**测**：完整多任务模型 + 像素级定位（latent→pixel 对齐在此集中解决）+ 全量训练 + 对齐 SOTA。

- [ ] Baseline 齐全：TruFor / PSCC / IML-ViT / FIRE / DIRE / AEROBLADE / CLIP 或 DINOv3 线性探针。
- [ ] 评测轴：跑 pipeline 设计的 **Test-A..F**（in-domain / cross-generator / cross-manipulation / cross-domain / degradation / real-only 误报率）。
- [ ] 每个创新点配消融（去门控 / 去逆估计 / 去多 σ 各掉多少）。
- [ ] 采样：生成器**均衡采样**而非按图数采样（呼应 Community Forensics）。

---

## 明确延后（第二篇地盘，别塞进第一篇）

- 跨生成器泛化的理论（风险不变 V-REx / 元-Fisher / 仿射包证书）：训练不稳、需分层采样 + shrinkage，且解决的是"泛化"，与第一篇"检测+定位+类型"是两个核心。第一篇里最多作为闸门 3(b) 的轻量正则出现。

## 兜底线（即使最性感结论都偏弱，仍能成一篇）

- score 残差场在 **AIGC 局部编辑定位** 上打过/接近 TruFor-FIRE 一档 + **算子族级别**类型识别 + **诚实刻画**的跨生成器局限。底盘成立即可投强会（CVPR/ICCV/ECCV）或 TIFS 一档；t0 逆估计（闸门 1 强版）与跨生成器缓解（闸门 3b）是上行期权。

---

## 命令附录

```bash
# 冒烟（改 config.py 计数为 4 后）
python gate_experiments/gate1_generate.py && python gate_experiments/gate1_analyze.py
python gate_experiments/gate2_generate.py && python gate_experiments/gate2_analyze.py

# pipeline 受控 probe（应用 PATCHES.md 后）
forgery-pipeline run --config configs/probe.yaml --out data/probe
forgery-pipeline validate-manifest --path data/probe/manifest.jsonl
```

# PAPER_DESIGN — Inverting the Edit（创新点与设计，供 Claude Code）

> **拟定标题**：*Inverting the Edit: Joint Detection, Localization, and Operator Attribution of Diffusion-Based Image Editing via the Score-Residual Field*
> **一句话主线**：用冻结扩散先验上的**多尺度 score 残差场**做检测/定位，并把**编辑类型识别**重表述为对**编辑算子参数**（噪声级 t0 × 条件机制 × 掩码支撑）的**反向估计**。
> **诚实边界**：标注为"待验证"的命题由 EXECUTION_CHECKLIST.md 的闸门检验；勿把任何机制写成 *provable*。

---

## 1. 问题与任务边界

- 局部编辑（inpainting / img2img / outpainting / 替换 / 背景重绘）视觉高度一致，传统 splicing 的边界、频率、相机指纹线索被削弱。
- **不走多域融合路线**：空间 + 频域 + 噪声 + 语义四分支融合在 2024–2025 已高度饱和（TruFor、一批空频/三域工作），作为创新点立不住；赛道已转向基础模型特征（CLIP / DINOv3）。
- **本篇任务**：一个框架内同时完成 **检测 + 像素级定位 + 编辑算子归因** —— 不止真假，还要"在哪、怎么编辑的"。
- **范围声明**：跨生成器泛化的理论（风险不变 / 元-Fisher）留作**第二篇**，不并入本篇。

---

## 2. 核心创新：score 残差场 + 算子逆估计

### 2.1 统一信号 — 多 σ Tweedie 去噪残差场

对冻结 SD2 先验，在多个噪声级 `t` 上测去噪残差：

- 前向加噪：`z_t = √(ᾱ_t)·z0 + √(1−ᾱ_t)·ε`,  `ε ~ N(0, I)`
- 去噪残差（score 匹配）：`r_ε(t) = E‖ ε − ε̂_θ(z_t, t) ‖²`
- 一步反演：`ẑ0 = (z_t − √(1−ᾱ_t)·ε̂_θ) / √(ᾱ_t)`,  `r_x(t) = E‖ z0 − ẑ0 ‖²`

由 **Tweedie 公式** `(x̂ − x)/σ² ≈ ∇ log p_σ(x)`，该残差即 score（对数密度梯度）的估计——**有理论根基的取证场**，而非手搓"噪声分支"。每张图 → 跨尺度剖面向量 `f(x) = [ r_ε(t₁:t_K), r_x(t₁:t_K) ]`。

### 2.2 三项输出

- **检测 / 定位**：真实像素与模型采样像素的 σ 剖面系统性不同（采样区接近去噪映射不动点、在模型流形上、模型密度高）。
- **算子归因（核心、最原创）**：每种编辑在 σ 剖面上留下可区分签名 → 类型识别 = 反解 `θe = (t0, 条件 c, 掩码 M)`：
  `θ̂e = g_ψ( f(x) )`,  `θe = (t0, c, M)`
  这是带生成过程语义的逆问题，而非"接个 softmax"。

### 2.3 为什么有数学支撑（机制，非定理）

- SDEdit 的**真实性–忠实性权衡**：加更多噪声、跑更长 SDE → 更真实更不忠实，故起始噪声级 t0 在残差里有签名（→ 闸门 1 检验）。
- inpaint（RePaint 式）每步把已知区**硬投影**回去 → 边界不连续 + 内部在流形上，签名与全图 img2img 不同。
- 由此解释公开难题"为何不同扩散方法之间难分"：它们的 `(t0, 条件)` 重叠 → 签名重叠；多 σ 剖面 + 边界算子正是拉开它们的维度。

---

## 3. 方法设计

- **冻结打分先验**：单个 SD2，与生成器无关（这本身是闸门 3 要压的跨生成器假设）。VAE 数学走 fp32 防 NaN；UNet 走 fp16。
- **表示**：多 σ 残差剖面 `f(x)`（latent 级，少数 σ 步即可，推理可控）。可选频域分支（DCT/FFT，读 JPEG/上采样签名）作为辅助，而非并行主分支。
- **四个头**：检测（图像级）、定位（像素掩码，U-Net/Segformer）、算子归因（逆估计 g_ψ）、可靠性/不确定性（可选）。
- **自洽约束**：算子的掩码-条件机制预测的掩码几何，应与定位结果一致（grounding 一致性损失）。

---

## 4. 子创新（每个带理论钩子）

1. **多 σ Tweedie 残差场**作为统一取证表示（替代四分支拼接）。理论：Tweedie + 去噪 score 匹配。
2. **编辑算子逆估计做类型识别**（`(t0, c, M)` 参数恢复）。理论：SDEdit 噪声级权衡 + RePaint 投影。**全场最原创**。
3. **流形/曲率定位**：编辑区贴近模型流形，去噪器雅可比 / 局部本征维数与真实区不同。理论：PF-ODE 收敛自适应内在维度。**注意**：multiLID 已用 LID 做扩散图检测，故落在"定位 + 去噪器雅可比"，与之区分。
4. **跨生成器鲁棒性 = score 先验集成 + 第二篇的不变目标**（第一篇仅作轻量轴）。

---

## 5. 与现有工作的关系（related work 骨架）

- **DIRE / AEROBLADE / FIRE / DRCT**：单尺度重建/反演做检测与 inpainting 定位。本方法差异：**多 σ score 剖面 + 算子逆估计 + 定位**，并给"扩散-扩散难分"机理解释。
- **MiraGe（生成器不变 + Fisher 式判别）/ CRDA（IRM × deepfake，环境=增强）**：均与第二篇的泛化主线相关；第一篇不与之正面竞争。
- **Community Forensics**：生成器多样性结论 → 第二篇用仿射包证书解释。
- 必须正面区分 multiLID（LID 检测）与 AEROBLADE（单尺度重建定位）。

---

## 6. 诚实的风险与边界

- score 残差需推理时跑扩散 → 有算力成本，但只需少数 σ 步。
- 跨生成器：残差在某先验下算，若编辑用差异极大模型族，签名可能对不上（子创新 4 / 第二篇解决，别假装没有）。
- `(t0, 算子)` 的可识别性是**假设**，写成"有理论动机 + 实验支撑"，**不写 provable**。
- "操作 vs 模型指纹"混淆：必须做同一算子跨多模型的归因测试解耦（闸门 2 cross-model）。

---

## 7. 验证计划

- 全部承重假设由 **EXECUTION_CHECKLIST.md** 的闸门 0–4 检验，先定 go/no-go 再看数据。
- Baseline：TruFor / PSCC / IML-ViT / FIRE / DIRE / AEROBLADE / CLIP 或 DINOv3 探针。
- 评测轴（来自 pipeline 划分）：Test-A in-domain、Test-B cross-generator、Test-C cross-manipulation、Test-D cross-domain、Test-E degradation、Test-F real-only 误报率。
- 每创新点配消融。

---

## 8. 数据依赖（对接 forgery-pipeline）

本篇数据由 `forgery-pipeline/` 生成，manifest 须包含（含 PATCHES.md 新增字段）：
- 定位/检测：`image_path`、`real_image_path`、`mask_path`、`is_fake`、`generator_name`、`generator_family`、`split`。
- 类型/逆估计：`operator`（img2img/inpaint/outpaint/object_replacement/background_editing…）、**`strength`**（img2img/SDEdit）、可选 `init_timestep`。
- 鲁棒性：退化样本须**独立成行** + `postprocess` 参数 + `postprocess_of` 回链原图（见 PATCHES.md PATCH 5）。
- 受控 probe 子集（强度网格 + 算子×族网格）用于闸门 1/2（PATCH 4）。

---

## 9. 论文范围切分

- **第一篇（本篇）**：检测 + 定位 + 算子归因。题眼：score 残差场 + 算子逆估计。
- **第二篇**：跨生成器泛化理论（V-REx 风险方差 + 元-Fisher 闭式 `S⁻¹δ̄` + 仿射包证书 + WRI 异方差修正）。
- 投稿取向：TIFS / TIP / Pattern Recognition 一档自然归宿；理论足够深可考虑 TPAMI；并行可投 CVPR/ICCV/ECCV。**分区每年刷新，投前按当年表确认。**

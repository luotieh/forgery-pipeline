# 完整实验计划 — *Inverting the Edit*（2026-07-15）

> **论文**：*Inverting the Edit: Joint Detection, Localization, and Operator Attribution of Diffusion-Based Image Editing via the Score-Residual Field*
> **主线**：冻结扩散先验上的**多 σ Tweedie score 残差场** → 检测 + 像素级定位 + **编辑算子逆估计 `(t0, c, M)`**。
> **本文档**：把 `PAPER_DESIGN.md` 的创新点 + `EXECUTION_CHECKLIST.md` 的闸门，落成一套**足以支撑完整一篇**的实验矩阵，含 go/no-go、算力预算、图表清单、兜底决策树。
> **对接**：`next_optimization_plan_2026-07-09.md` 的 P0/P1/P2 = 本计划 Phase A 的三条子线，此处并入并扩展。

---

## 0. 战略框架：锁死兜底线，争取上行期权

**当前真实闸门（真实 SD1.5，n=50，见 `real_gate_results_2026-07-02.md`）**：gate0 **PASS**（弱，det 0.688/loc 0.643）、gate1 **WEAK**（t0：BA 0.475/ρ 0.476，多σ 增量 +0.025）、gate2 **WEAK**（算子 0.372，单模型下 cross 无信息）、gate3 **PARTIAL**（跨生成器单模型不可测；Kandinsky 实验运行中）。

**张力**：论文最原创的一环（算子逆估计，尤其 t0）恰好现在最弱。故实验计划按**期权结构**编排——

| 层 | 内容 | 由哪些实验保证 | 目标会议/期刊档位 |
|---|---|---|---|
| **兜底线（必须拿下）** | score 残差场做 **AIGC 局部编辑定位** 打到/接近 TruFor-FIRE 一档 + **算子族级**识别 + **诚实刻画**跨生成器局限 | Phase B/C/D 的定位 + 算子族 + Test-A/E/F | 底盘成立即可投 CVPR/ICCV/ECCV 或 TIFS |
| **上行期权 1** | t0 逆估计（细粒度强度回归）成立 | Phase A 特征升级把 gate1 WEAK→PASS | 加分，强化「逆估计」题眼 |
| **上行期权 2** | 跨生成器缓解（先验集成/不变正则）部分回收掉点 | Phase F 跨生成器轴 + 轻量正则 | 加分，接第二篇 |

**决策门 G-A（Phase A 结束时）**：若特征升级仍无法把 t0/算子拉过闸门阈值 → **头条改为「定位 + 算子族识别」**，t0 逆估计降级为「有理论动机、实验部分支撑」的次要贡献，绝不写 provable。这一步决定论文叙事，必须先做。

> **G-A(t0) 已裁决（2026-07-15，预注册 confirmatory，n=1000，报告 `checking/gate1_confirmatory_report_2026-07-15.md`）**：主判据过（ρ=**0.700** [0.669,0.728]）、辅判据过且触发加档（2桶BA=**0.770**≥0.72 → 许可「信息量不低于原三桶 0.55 门槛」）、MAE 0.161>0.15 → **t0 = 强序数恢复 + 粗桶强度分级（固定 CFG/steps 条件下）**，不以「逆估计」修饰 t0。**C4 多σ增量成立**（Δρ(amp−单σ)=+0.075 [0.044,0.106]）；**方向特征增量确证**（Δρ=+0.124 [0.095,0.156]）。相邻档位 AUC 0.80/0.70/0.65/0.58 单调递减（饱和假说证实），全程≥0.55 → s\* 右删失。**待办**：P4=否 → 投稿前须跑 CFG/steps 抖动补充 probe（v2 §5）。G-A(算子) 待 B3 gate2 n≥200 复测。

---

## 1. 主张 → 实验 → 图表 追溯矩阵

| # | 论文主张 | 承重实验 | 成功判据 | 图/表 |
|---|---|---|---|---|
| C1 | 多σ score 残差场可检测扩散局部编辑 | Phase C 检测头 × Test-A..F | AUC/AP 优于 DIRE/AEROBLADE | Tab-Detect, Fig-ROC |
| C2 | 且能像素级定位编辑区 | Phase C 定位头 × Test-A..F | F1/IoU 接近或超 TruFor/FIRE | Tab-Loc, Fig-Heatmap |
| C3 | **算子逆估计做类型识别**（核心） | Phase A(特征) + Phase C(g_ψ头) | 算子族 BA≥0.50；t0 ρ≥0.30（上行） | Tab-Attr, Fig-Confusion, Fig-t0scatter |
| C4 | 多σ 优于单σ（AEROBLADE） | Phase E 消融 A1 | Δ(检测/定位/算子) 显著>0 | Tab-Ablation |
| C5 | 信号来自操作而非模型指纹 | Phase A gate2 cross-model（多族） | 跨模型 BA≥0.40 不塌到随机 | Tab-CrossModel |
| C6 | 跨生成器局限被诚实刻画 + 部分缓解 | Phase F Test-B + 先验集成 | 掉点曲线 + 集成回收量 | Fig-CrossGenDrop |
| C7 | 定位-算子自洽（grounding 一致性） | Phase E 消融 A4 | 去一致性损失掉点 | Tab-Ablation |
| C8 | 少数σ步即可（效率可控） | Phase E 消融 A5 | σ步数-精度曲线 | Fig-Efficiency |

---

## 2. Phase A — 信号地基与特征升级（**pivot，先做，决定叙事**）

**目的**：在小成本受控 probe 上把「t0/算子」尽力从 WEAK 拉过阈值；拉不动就锁定兜底叙事。全部离线可复算——残差提取器对**已存图**重跑，改特征**不必**重新生成数据。

### A0. 跨生成器判决（P0，运行中）
- [ ] Kandinsky 2.2 异族先验 probe（n_base=50）跑完 → 更新 gate2 cross-model / gate3 heldout / **gate0 检 Kandinsky 迁移 AUC**（判 gate0 0.688 是真信号还是自家先验偏置，支撑 C5/C6）
- 判据：gate0 迁移 AUC 显著>0.5→C5 立；gate2 cross≥0.40→操作>模型指纹

### A1. t0/算子 方向特征升级（把 gate1/2 WEAK→PASS 的唯一杠杆）
现 `extractor.py:profile()` 只用**幅值统计**，且 `residual_stack` 把 `r_ε`(ε误差) 与 `r_x`(x0误差) 相加**坍缩掉方向**（`extractor.py:120`）——多σ 增量仅 +0.025 即触顶。逐级升级、每级要有增量证据才继续：
- [ ] **①分离 r_ε / r_x 双通道**（零成本，在已存 probe 上离线复算）→ 看多σ−单σ delta 是否从 +0.025 回升
- [ ] **②逐 t 残差方向余弦**（相邻 σ 的 ε 误差图空间相关/夹角，捕捉去噪轨迹形状）
- [ ] **③DIRE 式反演重建距离**（DDIM 反演→重建，对 SDEdit 强度更单调）
- [ ] **④频域辅助分支**（残差径向功率谱斜率，读 img2img 全图重绘/JPEG 印记）
- 判据（闸门 1）：多σ BA≥0.55 **且** ρ≥0.30 → t0 逆估计 PASS（上行期权 1 兑现）；停在 [0.45,0.55) → 降级为**粗桶强度分级**
- 判据（闸门 2）：同模型 BA≥0.50 **且** 跨模型≥0.40 → 算子逆估计成立；否则收窄为**算子族识别**

### A2. 多σ 增量确权（决定「多尺度」一词能否用）
- [ ] 多σ vs 单σ（AEROBLADE 口径）在检测/定位/算子三处的增量 → 支撑 C4；无增量则删「多尺度」框架，故事挂算子逆估计+定位

### A3. gate1 t0 判据预注册（PATCH 6 方向的务实版）

**探索性观测（n=50 probe，2026-07-15，real extractor）——不得据此宣布 PASS**：
- amp-only：ρ=0.377，三桶 BA=0.483；+direction：**ρ=0.503**，三桶 BA=0.492（方向特征增益 ρ **+0.127**、BA +0.008）。
- 呈典型「ρ 显著、三桶 BA 不过」分裂 → 信号单调、均匀三桶画错的假说。
- 2 桶 median 切点探索性运行点：见 `data/report_*_2bucket`（gate1 `metrics.two_bucket_median`），**标注为降级分支运行点、探索性**（其 CI 为逐行 bootstrap，未按底图聚类，偏窄——v2 预注册已改为 cluster bootstrap；median 切点 0.5 与 v2 声明式切点 0.4 在离散档位下给出**同一划分** {0.1,0.3} vs {0.5,0.7,0.9}）。

**预注册（权威版）**：t0 验证性判据与协议以 **`docs/PREREG_gate1_v2_2026-07-15.md`（已锁定）** 为唯一权威，取代本节先前草案。要点：主判据 ρ≥0.50（cluster-bootstrap 95% CI 下界>0.30）；辅判据切点 **0.4** 的 2 桶 BA≥**0.66**（CI 下界>0.55），BA≥0.72 加档「信息量等效」声明；MAE≤0.15 决定「逆估计」措辞权；C4 挂 Δρ 配对 cluster bootstrap CI 下界>0；**P4=否** → 全部结论自动限定「固定 CFG/steps 条件下」，投稿前须补 CFG/steps 抖动 probe（n=50 × 网格 × CFG{5,7.5,10} × steps{30,50}）。协议：Ridge(α=1) OOF（5 折×20 重复 repeated **group** K-fold，按底图分组）+ 折内嵌套 isotonic + **按底图 cluster bootstrap**（B≥2000）；三配置共享折划分与 bootstrap 索引；一次性评估。执行脚本 `checking/gate1_confirmatory.py`，verdict 按预注册 §4 机械导出。

**Phase A 产物**：决策门 G-A 的结论 + 冻结的特征提取配置（`profile()` 最终形态）+ 闸门复测报告 + n≥200 confirmatory gate1。**算力**：几个 GPU 小时（多为已存图离线复算 + 一次 Kandinsky 生成 + 一次 n≥200 强度网格）。

---

## 3. Phase B — 主数据集构建（Test-A..F 全轴）

**目的**：产出支撑全部评测轴的规模化数据，生成器**均衡采样**（呼应 Community Forensics），退化样本**独立成行 + 回链**。

### B1. 生成器矩阵（4090 24GB 可承载；括号内为 holdout 归属）
| 角色 | 模型族 | 算子 | 备注 |
|---|---|---|---|
| 打分先验（冻结，唯一） | SD1.5 | — | 提取器；与生成器解耦是 gate3 假设 |
| 训练可见生成器 | SD1.5, SD2/SDXL | img2img/SDEdit | 多强度 |
| 训练可见 inpainter | SD1.5-inpaint | inpaint/outpaint/replace/background | 掩码支撑 |
| **cross-generator holdout (Test-B)** | **Kandinsky 2.2**(异族unCLIP), **SDXL**, (PixArt/Flux 若显存允许) | inpaint/img2img | 只出现在 test_b |
| 非扩散对照（可选） | LaMa / MAT | inpaint | 检验 score 场对非扩散编辑的行为 |

### B1.5 存储格式决定（2026-07-16 冻结，B3 前不再摇摆）

**主库 = PNG 母本**（维持 PATCH 7），拒绝"两边统一压 JPEG"。理由：统一 JPEG 只把混淆从「有无 JPEG 史」换成「双压 vs 单压」（经典取证特征，仍可学），却要付出三重代价——JPEG 量化吃掉多σ残差的高频信号（自残头牌方法）、Test-E 退化轴被二次压缩污染失去解释力、且不可逆（PNG 可派生 JPEG 评测变体，反向不行）。**源头 JPEG 史混淆的正解**（非存储层）：① Phase C 训练协议加两类同施的随机 Q JPEG 增广（Wang'20/Corvi'23 标准做法）；② B2 真实源掺 RAW 派生子集（RAISE/Dresden）并提升一片进 train；③ **压缩史混淆探针**（零 GPU：仅块效应/DCT 直方图特征的浅探针判 real/fake，量化残余可利用度，数字进论文）。工程：80k×512² PNG ≈ 24–36GB，**B3 前扩 AutoDL 数据盘**。

### B1.6 PATCH 9 阻断关系（2026-07-16 接受，详见 addendum PATCH 9）

B3 起跑前必须完成：**9.1** 主 run nuisance（CFG{5,7.5,10}×steps{30,50} 逐图采样）+ 强度连续采样 U(0.1,0.95)（probe 网格不动）· **9.2** prompt bank（版本 hash 入行）/掩码面积分桶（V12）/分辨率组配套 real+vae_rt 行 · **9.3** base_id 字段 + V8–V10 split 防泄漏（注毒负例单测）· **9.4** 驱动加固（HEAD 断言/评估禁令/断点续跑幂等/磁盘预检 ≥100GB）· **9.5** gate2 设计冻结 + PREREG_gate2_v3（锁定先于任何 gate2 评估）。**9.6**（(7.5,30) 单元分解，零 GPU）非阻断、最先做。

**⚠️ CFG 惰性发现（2026-07-16，影响 9.1/9.6 与措辞）**：全部 probe 的 img2img 均为空 prompt → CFG 项在 classifier-free guidance 中精确消去（cond≡uncond）——实验⑤的 CFG 维度实为惰性，6 单元有效坍缩为 steps 30/50 两单元，跨 CFG 差异=各单元独立 seed。后效：①补充 probe 结论内部自洽（主 confirmatory 同为空 prompt 域），但 v2 §5 脚注应精确化为「空 prompt 机制下 CFG 惰性、敏感度由 steps+seed 承载」；②主库上 prompt bank（9.2a）激活 CFG 后其敏感度未被覆盖 → 固定 CFG 无从声称安全，9.1 逐图抖动是唯一诚实选项；③9.6 免费升级：同 steps 跨 CFG 单元的 ρ 波动=纯 seed 噪声地板，分解为 base/steps/seed 三视图。

### B2. 真实底图域（供 Test-D cross-domain / Test-F real-only）
- [ ] in-domain：COCO-val 子集（替换现 picsum，场景分布更标准）
- [ ] cross-domain：另一域（RAISE/Dresden 相机原图 或 不同网图集）→ Test-D
- [ ] real-only：多来源纯真实图，测 FPR → Test-F

### B3. 规模与划分（目标，均衡采样）
- [ ] 主 `run`：每 生成器×算子 单元 ≥ 2–4k 编辑图 + 等量真实，总量 **~40–80k**；masks 齐全（定位监督）
- [ ] 8 路 split：train/val/test_a(in-domain)/test_b(cross-gen)/test_c(cross-manip holdout 算子)/test_d(cross-domain)/test_e(degradation)/test_f(real-only)
  - **Test-C holdout 已裁定（2026-07-16，PATCH 8.3 几何探针）= `object_replacement`**：outpaint/background_editing geometry-only AUC=1.0 → 几何平凡不得作 holdout；object_replacement 与 inpaint 配对几何 AUC=0.487≈机会线 → 合格。见 `checking/testc_geometry_report_2026-07-16.md`；Phase B 若改算子×掩码约定须重跑探针。
- [ ] 受控 probe：强度网格 n_base≥200（gate1）+ 算子×族网格（gate2）
- [ ] Test-E 退化：JPEG(Q40-90)/resize/高斯模糊/噪声/二次压缩，**独立成行** + `postprocess`/`postprocess_of` 回链
- [ ] `validate-manifest` + `stats` 核验 `by_generator_name`/`by_operator` 计数均衡

**算力**：主数据集生成是最大头。4090 上扩散编辑 ~2–4 img/s（含 VAE 往返），60k 图约 **6–12 GPU 小时/轮**，分批跑。

---

## 4. Phase C — 方法与四头

**冻结打分先验**：单 SD1.5，VAE fp32 防 NaN、UNet fp16（已实现于 `DiffusersResidual`）。**表示** `f(x)=[r_ε(t₁:K), r_x(t₁:K), 方向/频域特征]`（Phase A 冻结）。

- [ ] **H1 检测头**：残差剖面 → 图像级二分类（先线性/浅 MLP 探针，再轻量 CNN）；指标 AUC/AP
- [ ] **H2 定位头**：多σ 残差图堆叠 → U-Net/Segformer 解码器 → 像素掩码；**latent→pixel 对齐在此集中解决**（gate0 已验残差图可粗定位）；指标 pixel-F1/IoU/AUC
- [ ] **H3 算子逆估计头 g_ψ**：`f(x)→(t0 桶/回归, 算子类, 掩码几何)`；**对照「接 softmax」证明逆估计参数化更优**（C3/消融）
- [ ] **H4 可靠性/不确定性头**（可选）：预测置信，支撑 real-only 误报控制
- [ ] **自洽约束**：H3 预测的掩码几何与 H2 定位一致（grounding 一致性损失，C7）

**训练**：多任务联合 + 逐头消融权重。**算力**：残差场可预提取缓存（一次性），头训练轻量，~数 GPU 小时/配置。

---

## 5. Phase D — Baseline 与 Test-A..F 评测

- [ ] **Baseline 齐全**：DIRE、AEROBLADE、FIRE、TruFor、PSCC-Net、IML-ViT、CLIP/DINOv3 线性探针（检测+定位各取可比者）
- [ ] **统一评测协议**：同 train/test split、同指标、同图像预处理；开源权重优先，缺失则按原文复现并声明
- [ ] **六轴全跑**（每轴 × 每方法 × 检测/定位/算子）：
  - Test-A in-domain：主战场，须 ≥ 强 baseline
  - Test-B cross-generator：掉点曲线（seen→holdout 族），支撑 C6
  - Test-C cross-manipulation：holdout 算子上的泛化
  - Test-D cross-domain：换真实底图域
  - Test-E degradation：退化鲁棒性曲线
  - Test-F real-only：**FPR@真实**（审稿人必看）
- 判据：定位在 Test-A 达兜底线（接近/超 TruFor-FIRE）；检测全轴不塌；算子族识别显著>随机

**算力**：主要是 baseline 推理 + 本方法评测，~数 GPU 小时。

---

## 6. Phase E — 每创新点消融

| # | 消融 | 回答的主张 | 观测 |
|---|---|---|---|
| A1 | 多σ vs 单σ(AEROBLADE) | C4 | Δ 检测/定位/算子 |
| A2 | r_ε only / r_x only / both / +方向 / +频域 | C3 特征贡献 | 逐特征增量 |
| A3 | 打分先验换族（SD1.5 vs SDXL 当 scorer） | 方法对先验的敏感性 | 稳定性 |
| A4 | 去 grounding 一致性损失 | C7 | 定位/算子掉点 |
| A5 | σ 步数 {1,2,3,5,K} | C8 效率 | 精度-成本曲线 |
| A6 | 逆估计头 vs 纯 softmax | C3 参数化优越性 | 算子/t0 精度差 |

---

## 7. Phase F — 跨生成器鲁棒性与诚实边界（上行期权 2 + 审稿护城河）

- [ ] **跨生成器掉点刻画**：Test-B 上 seen→holdout 族的检测/定位/算子掉点曲线（C6 的「诚实局限」）
- [ ] **轻量缓解**：score 先验集成（多先验残差平均/拼接）+ 可选不变正则（作为闸门 3b 的轻量轴，**理论留第二篇**）→ 报告回收多少掉点
- [ ] **操作 vs 模型指纹解耦**：同一算子跨多模型归因（gate2 cross-model 的论文版），防审稿「学的是模型指纹」
- [ ] **可识别性诚实声明**：`(t0,算子)` 写成「有理论动机 + 实验支撑」，随 G-A 结论调整措辞

---

## 8. 算力预算与排期（4090 单卡，AutoDL 按量）

| Phase | 主要开销 | GPU 时估 | 关键产物 |
|---|---|---|---|
| A 信号/特征 | Kandinsky 生成 + 离线复算 | 3–6 h | 决策门 G-A、冻结特征 |
| B 数据集 | 60k 图生成（分批） | 6–12 h/轮 | 主 run + probe + 退化 |
| C 方法 | 残差预提取缓存 + 头训练 | 4–8 h | 四头 checkpoint |
| D 评测 | baseline + 六轴 | 4–8 h | 主结果表 |
| E 消融 | 复用缓存 | 3–6 h | 消融表 |
| F 鲁棒性 | 先验集成 | 2–4 h | 跨生成器曲线 |
| **合计** | | **~25–45 GPU 时** | 按 ¥2/时 ≈ ¥50–90 |

排期建议：**A 先行**（决定叙事，不做完不投入 B 的大生成）→ B/C 并行（数据边生成边预提取）→ D/E/F 收尾。每 Phase 存 checkpoint，AutoDL 用完**关机**。

---

## 9. 论文产物清单（写作即填空）

- **表**：Tab-Detect（六轴检测）、Tab-Loc（六轴定位）、Tab-Attr（算子/t0）、Tab-CrossModel、Tab-Ablation
- **图**：Fig-ROC、Fig-Heatmap（定位定性）、Fig-Confusion（算子混淆）、Fig-t0scatter（强度回归）、Fig-CrossGenDrop（掉点曲线）、Fig-Efficiency（σ步-精度）、Fig-Method（多σ残差场+四头架构图）
- **补充**：数据集统计（生成器×算子×域计数）、失败案例、退化鲁棒性全曲线

---

## 10. 风险登记与兜底决策树

```
G-A: Phase A 后 t0/算子能否过闸门?   ← t0 支已裁决(2026-07-15): 强序数恢复+粗桶分级,
 │                                      ρ=0.70 confirmatory；算子支待 B3 gate2 n≥200
 ├─ 能(上行) → 头条=「检测+定位+算子逆估计(含t0)」，逆估计做核心卖点
 └─ 不能 → 头条=「检测+定位+算子族识别」，t0 降为次要；兜底线仍成立 ✅

G-B: gate0 检 Kandinsky 迁移 AUC?
 ├─ 显著>0.5 → C5 立，score 场是可迁移真信号
 └─ 塌到0.5 → 强调「先验相关性」为被刻画局限，先验集成(Phase F)作缓解

G-C: 多σ 无增量?  → 删「多尺度」词，故事挂逆估计+定位(方法仍完整)
G-D: 跨生成器崩得厉害? → 写成「被刻画的局限+部分缓解」，正是第二篇动机，不硬撑
```

**红线（诚实边界，全程）**：任何机制不写 provable；VERDICT 区分 extractor(real vs multisigma 代理)；不复现 mock 高估；跨生成器泛化**理论**留第二篇。

---

## 附：与现有仓库/文档的接口

- 数据字段（`image_path/real_image_path/mask_path/is_fake/generator_name/generator_family/operator/strength/init_timestep/postprocess/postprocess_of/split`）已由 `PATCHES.md` 落地，`GATE_DATA.md` 给闸门→产物→字段对照。
- 闸门分析在 `checking/`（gate0-4 + `DiffusersResidual`），Phase A 的特征升级改的就是 `checking/extractor.py`。
- 生成后端 `forgery_pipeline.backends.real.diffusers_gen`（`AutoPipelineForImage2Image/Inpainting`，已支持 SD1.5 + Kandinsky 异族先验）。
- 云端运行：`scripts/run_cross_generator_p0.sh`（**注**：其 `pip install -e ".[real]"` 在 torch 2.3 镜像上会拉到需 torch≥2.5 的 diffusers，须锁 `diffusers==0.30 transformers==4.44`，待修）。
</content>

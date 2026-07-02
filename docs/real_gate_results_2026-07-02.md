# 真实扩散信号闸门结果（2026-07-02）

> **一句话结论**：真实信号下,gate0 残差可分性成立（检测 AUC 0.69、定位 AUC 0.64）,gate1 的 t0 可恢复性**存在但显著弱于 mock 代理暗示的水平**（ρ 0.48 vs 0.88,BA 0.475 vs 0.757）,gate2 算子可分性高于随机但弱（0.37 vs 随机 0.2）,gate3 多σ增量仅 +0.025。mock 代理此前的乐观读数大部分是 mock 假象。

## 环境

| 项 | 值 |
|---|---|
| GPU | RTX 4060 Laptop 8GB（WSL2） |
| 生成模型 | SD1.5 `stable-diffusion-v1-5/stable-diffusion-v1-5` + `stable-diffusion-inpainting`（fp16,attention slicing;**SD2 已从 HF 下架,由计划中的 SD2 改为 SD1.5**） |
| 残差 extractor | `--extractor real`:冻结 SD1.5,多σ Tweedie 残差（UNet fp16 + VAE fp32,t ∈ {50,150,300,500,700}） |
| 底图 | picsum.photos 真实照片 64 张,512×512 中心裁剪 |
| 规模 | n_base=50 → gate1 250 张（5 强度）,gate2 900 张（5 算子 × 18 生成器位）,manifest 共 1200 样本 |
| 耗时 | 生成约 80 分钟（~4 s/张）+ 闸门分析约 30 分钟（残差 ~1 s/张） |
| 判定口径 | gate1 三桶(low/mid/high)随机=0.333;gate2 五算子随机=0.2 |

## 每闸门真实 VERDICT（n_base=50）

| 闸门 | VERDICT | 关键指标 | 随机基线 | mock 代理对照 |
|---|---|---|---|---|
| gate0 残差可分 | **PASS** | detection_auc **0.688** · localization_auc **0.643** (n_loc=146) | 0.5 | PASS(det 0.631/loc 0.758) |
| gate1 t0 可恢复 | **WEAK** | BA **0.475** CI[0.397,0.546] · ρ **0.476** · 多σ 0.475 / 单σ 0.450 (n=250) | 0.333 | PASS(BA 0.757 · ρ 0.881 · 多σ−单σ +0.417) |
| gate2 算子可分 | **WEAK** | same_model **0.372** · cross_model 0.374 (n=900) | 0.2 | WEAK(0.454/0.478) |
| gate3 多σ增量+跨生成器 | **PARTIAL** | multi_sigma_delta **+0.025** · heldout 结构性 0 | — | PASS(delta +0.417) |
| gate4 评测口径 | EVAL-ONLY | probe 全部为 train split,各 test 桶 n_fake=0（预期,非缺陷） | — | EVAL-ONLY |

冒烟版（n_base=4）数据见 `data/checking_report_real_smoke_n4.json`;正式报告 `data/checking_report_real.json`。

## 诚实解读

**gate0「残差可分」成立,但强度有限。**
冻结 SD1.5 的多σ残差在真实数据上能区分 real/fake（AUC 0.688）并粗定位编辑区（AUC 0.643,146 个掩码样本）——承重假设的最低门槛过了。但两项都不到 0.7,且存在「SD1.5 检 SD1.5」的自家先验偏置,跨模型可分性未验证。

**gate1「t0 可恢复」在真实信号下部分成立,但被 mock 严重高估。**
BA 0.475 的 CI 下界 0.397 高于随机 0.333,ρ=0.476 达到 PASS 的相关性门槛（≥0.30）——强度信息确实留在多σ残差里。但分类准确率离 PASS 门槛（0.55）有距离,且**多σ相对单σ的增量只有 +0.025**（mock 上是 +0.417）。mock 的 ρ=0.89 是 mock 假象:mock 生成器把 strength 直接线性写进像素扰动,真实 SDEdit 的去噪轨迹在低/高强度处饱和。论文若以 t0 回归为承重假设,需要更强的特征（如逐 t 残差方向而非幅值）或降级为「粗粒度强度分级」。

**gate2「算子可分」高于随机但弱。**
0.372 vs 随机 0.2:算子指纹存在（inpaint/outpaint 的掩码边界痕迹、img2img 的全图重绘特征）,但 5 类平衡准确率不足 0.4。same≈cross（0.372 vs 0.374）不是好消息也不是坏消息——**当前 real 后端所有 inpainter 名字底层都是同一个 SD1.5-inpaint**,generators.yaml 的多名字只是标签,cross_model 在本设置下无信息量。

**gate3 跨生成器部分在单模型设置下结构性不可测。**
`holdout_generators: []` → 无 test_b split → heldout_acc 恒 0,gate3 在当前配置下不可能 PASS。多σ增量 +0.025 为正但微弱。要真正回答「跨生成器掉点」,必须接入第二个真实生成器（SDXL / Kandinsky 级别的异族先验）,这是下一步工作,不是本次冒烟能回答的。

**gate4 全空是口径问题而非信号问题**:probe 数据全部标 train,评测桶（test_a–f）没有样本。真实 gate4 需要走完整 pipeline 的 split 逻辑,超出 probe 范畴。

## 与 mock 代理的总对照

- mock 高估最严重的是 gate1（BA +0.28、ρ +0.41、多σ增量 +0.39）和 gate3（delta +0.39）;
- gate2 mock 与真实同为 WEAK,量级也接近（0.45 vs 0.37）——算子指纹这条线 mock 代理反而校准得不差;
- 此前分析文档（`gate_results_analysis_2026-07-02.md`）明确标注过「mock VERDICT 非科学结论」,本次真实复核证实了这一警示的必要性。

## 局限（写论文前必须解决）

1. **单一先验模型**:生成与提取都是 SD1.5,存在「自家先验检自家生成」的乐观偏置;跨生成器结论完全缺位。
2. **规模仍小**:n_base=50,底图全部来自 picsum（Unsplash 风格照片,场景分布窄）。
3. **gate0 曾有取样 bug**:max_n 头部截断在放量后取不到掩码样本,localization 假 FAIL;已修（等间隔取样)并复跑。
4. gate1 的强度桶边界（0.35/0.65)与 strengths 网格（0.1–0.9)的对齐会影响 BA 读数,换桶界应做敏感性检查。

## 过程记录

- 冒烟（n_base=4,92 张生成,~10 分钟):gate0 PASS(det 0.674/loc 0.666),gate1/2 WEAK,gate3 PARTIAL —— 与放量结论方向一致。
- 发现并修复:real 后端 registry 未缓存实例,probe 每样本重载扩散管线（上次运行中断的根因);gate0 头部截断取样;CAVEAT 文案按 extractor 区分。
- SD2 → SD1.5 切换:stabilityai 的 SD2 repo 已从 HF 下架,改用 SD1.5(512、ε-prediction,与 extractor 口径一致,方法与具体先验无关)。

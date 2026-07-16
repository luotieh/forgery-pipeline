# 组会汇报:伪造检测数据管线——闸门实验(2026-07-02)

> 汇报人:罗铁 · 项目:forgery-pipeline · 分支:`feat/real-diffusion-gates`

## 0. 一页结论

**今天完成了从「mock 代理信号」到「真实扩散信号」的闸门判定闭环**,这是论文承重假设的第一次真实检验:

- **gate0(残差可分)PASS**:冻结 SD1.5 的多σ Tweedie 残差在真实数据上,检测 AUC 0.688、定位 AUC 0.643——承重假设的最低门槛通过;
- **gate1(t0 可恢复)WEAK**:信号存在(BA 0.475 > 随机 0.333,ρ=0.476)但远弱于 mock 暗示(0.757/0.881)——**mock 高估被真实数据证伪**;
- **gate2(算子可分)WEAK**:0.372 vs 随机 0.2,算子指纹存在但弱;
- **gate3(多σ增量)PARTIAL**:增量仅 +0.025;跨生成器部分在单模型设置下结构性不可测。

**对论文的含义**:t0 回归不能作为承重主张,应降级为「粗粒度强度分级」或换更强特征;「残差可检测/可定位」这条主线是安全的。

---

## 1. 背景

- 论文思路:用冻结扩散先验的**多σ Tweedie 去噪残差**做伪造检测/定位/强度估计(设计见 `docs/PAPER_DESIGN.md`)。
- 在投入大规模实验前,设置 5 道**闸门**(gate0–4)对承重假设做 go/no-go 检验(`docs/EXECUTION_CHECKLIST.md`)。
- 此前所有 VERDICT 都基于 **mock 数据 + CPU 代理提取器**,只验证分析代码通路,非科学结论。

## 2. 今日工作总览(两轮)

| 轮次 | 内容 | 产出 |
|---|---|---|
| 上午 | **P0 分析代码优化 + mock 复测**:区域/分位特征、gate1 回归→分桶、gate4 评测口径修复、bootstrap CI | `docs/gate_results_analysis_2026-07-02.md` |
| 下午–晚间 | **真实扩散闸门**:真实后端(SD1.5 img2img/inpaint/残差)+ 真实底图 + RTX 4060 全流程,n_base=4 冒烟 → n_base=50 放量 | `docs/real_gate_results_2026-07-02.md`、`data/checking_report_real.json` |

## 3. 第一轮:P0 优化让分析代码「口径正确」(mock 复测)

同数据同代理,仅换分析代码:

| 闸门 | 优化前 | 优化后 | 改进来源 |
|---|---|---|---|
| gate1 | WEAK · BA 0.472 | **PASS** · BA 0.757 CI[0.68,0.82] | 回归→分桶(序数信号换对分类器) |
| gate2 | WEAK · 0.33 | WEAK · 0.454 | 区域几何特征 +0.13,仍不过线 |
| gate3 | PASS · +0.132 | PASS · +0.417 | 富特征放大多σ优势 |
| gate4 | test_e 无定义 | test_e 0.537 + 全 split 带 CI | 共享真实负样本池 + 局部检测分数 |

**关键教训**:mock 上的 PASS 只说明代码口径对了,不说明假设成立——这正是第二轮要回答的。

## 4. 第二轮:真实扩散闸门

### 4.1 实验设置

| 项 | 值 |
|---|---|
| 硬件 | RTX 4060 Laptop 8GB(WSL2) |
| 生成 | SD1.5 img2img + SD1.5-inpainting(fp16,attention slicing;**SD2 已从 HF 下架,计划中的 SD2 改为 SD1.5**) |
| 残差提取 | 冻结 SD1.5,多σ Tweedie 残差,t∈{50,150,300,500,700},UNet fp16 + VAE fp32 |
| 底图 | picsum.photos 真实照片 64 张,512² |
| 规模 | n_base=50 → gate1 250 张(5 强度)+ gate2 900 张(5 算子)= 1200 样本 |
| 耗时 | 生成 ~80 min(~4 s/张)+ 闸门分析 ~30 min(残差 ~1 s/张) |

### 4.2 真实 VERDICT vs mock 代理

| 闸门 | 真实(n=50 底图) | mock 代理(P0 后) | 差距解读 |
|---|---|---|---|
| gate0 残差可分 | **PASS** det 0.688 / loc 0.643 | PASS 0.631 / 0.758 | 方向一致,真实定位略弱 |
| gate1 t0 可恢复 | **WEAK** BA 0.475 · ρ 0.476 | PASS BA 0.757 · ρ 0.881 | **mock 高估 +0.28 / +0.41** |
| gate2 算子可分 | **WEAK** 0.372(随机 0.2) | WEAK 0.454 | mock 校准尚可 |
| gate3 多σ增量 | **PARTIAL** +0.025 | PASS +0.417 | **mock 高估 +0.39** |
| gate4 评测口径 | EVAL-ONLY(probe 全 train,预期) | EVAL-ONLY(各 split 有值) | 真实 gate4 需走完整 pipeline split |

### 4.3 诚实解读

1. **gate0 过线但不豪华**:0.688/0.643 都不到 0.7,且是「SD1.5 检 SD1.5」的自家先验设置,跨模型可分性未验证。
2. **gate1 被 mock 严重高估**:mock 把 strength 线性写进像素,真实 SDEdit 的去噪轨迹在低/高强度饱和。强度信息确实在(CI 下界 0.397 > 0.333),但多σ相对单σ的增量只剩 +0.025(mock 上 +0.417)。
3. **gate2 same≈cross(0.372/0.374)无信息量**:real 后端全部 inpainter 名字底层是同一个 SD1.5-inpaint,cross_model 在单模型设置下退化。
4. **gate3 跨生成器结构性不可测**:无 holdout 生成器 → heldout 恒 0,当前配置下 gate3 不可能 PASS。

### 4.4 过程中发现并修复的 3 个 bug

| bug | 症状 | 修复 |
|---|---|---|
| registry 不缓存 real 实例 | probe 每样本重载扩散管线(92 次),上次运行中断的根因 | 实例缓存,commit `62d5f00` |
| gate0 max_n 头部截断 | 放量后前 200 条全无掩码 → localization 假 FAIL | 等间隔取样,commit `4956a7b` |
| CAVEAT 不分 extractor | real 报告套用 mock 免责声明 | 按 extractor 区分,commit `a124e0f` |

## 5. 结论与下一步

**结论**:
- 承重主线「冻结扩散先验残差可检测/可定位」在真实信号下**成立**(gate0 PASS);
- 「t0 精细回归」**不成立为承重主张**,降级为粗粒度强度分级或换特征(逐 t 残差方向);
- mock 代理的作用重新定位:**验代码口径可以,预估效应量不行**。

**下一步(优先级序)**:
1. 接入第二个真实生成器(SDXL/Kandinsky 级异族先验)→ 解锁 gate3 跨生成器判定与 gate2 cross_model;
2. gate1 强度桶边界敏感性检查 + 更强特征;
3. 扩底图规模与场景多样性(picsum 分布窄);
4. 真实 gate4 走完整 pipeline split。

**局限声明**:单一先验模型(自家先验偏置)、n_base=50 规模、底图场景窄——本次是真实信号下的**初步**判定,写进论文前需扩样本 + 多先验复核。

---

## 附:产物索引

- 真实结果详文:`docs/real_gate_results_2026-07-02.md`
- mock 分析与 P0 优化:`docs/gate_results_analysis_2026-07-02.md`
- 报告 JSON:`data/checking_report_real.json`(放量)/ `data/checking_report_real_smoke_n4.json`(冒烟)
- 复现命令:
  ```bash
  python3 scripts/fetch_real_images.py --out data/real_base --n 64 --size 512
  FORGERY_REAL_IMAGE_DIR=data/real_base python3 -m forgery_pipeline.cli probe \
      --config configs/probe.real.yaml --out data/probe_real --n-base 50
  FORGERY_REAL_IMAGE_DIR=data/real_base python3 -m checking.run_gates \
      --run data/probe_real --probe data/probe_real --extractor real \
      --out data/checking_report_real.json
  ```
- 今日提交:`8c203d5`…`1832250`(10 commits,`feat/real-diffusion-gates`)

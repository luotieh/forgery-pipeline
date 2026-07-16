# 预注册 v3（草案）— gate2 验证性复测（算子机制可分性，same/cross-model，n_base≥200）

> **状态**：**草案**（§0 占位符 P1–P5 未填毕，未锁定）。本文件本次改动**不构成锁定**；后续对占位符表的填写与协议文本的任何改动，必须在专门的"锁定" commit 中一次性完成，且该 commit 信息须显式包含"锁定"字样，方可使本文件升格为 confirmatory 协议（机制同 `docs/PREREG_gate1_v2_2026-07-15.md`）。
> **锁定条件**：gate2 probe 数据的**生成**可先行于本文件锁定（生成不构成偷看——评估 = 计算或查看任何标签–特征关系，不含生成本身，与 v2 同一原则）；但**本文件的锁定 commit 必须先于对 gate2 probe 的任何标签–特征评估**（任何算子标签与残差/幅值/几何特征关系的查看，包括"部分数据先瞄一眼"）。
> **失效条件**：见 §8（沿用 `docs/PREREG_gate1_v2_2026-07-15.md` §7 全部条款，据 gate2 场景调整措辞）+ **一次性评估保护**——confirmatory 报告文件已存在即拒跑（§7）。任何一条触发 → 本轮降为探索性，需新 probe + 预注册 v3' 才能恢复 confirmatory 地位。
> **阈值声明**：主判据数值（same-model BA≥0.50、cross-model BA≥0.40）沿用 2026-07 原计划，**不因 gate1 t0 支验证性复测的大胜（ρ=0.700 confirmatory）而抬高或降低**（`docs/PATCHES_addendum_06_07_2026-07-15.md` §9.5「阈值不因 gate1 大胜而抬高或降低」/ §9.8「prereg v3 阈值沿用 7 月计划原值（0.50/0.40）」）。两支判据（t0 与算子）相互独立，互不构成对方放宽或收紧的证据。

---

## 0. 待填占位符（锁定前必须填完，共 5 处）

| # | 占位符 | 草案阶段状态（锁定时确认/替换） |
|---|---|---|
| P1 | 特征提取器 commit（冻结版本；覆盖 `profile()` 池化统计与供空间头读取的逐 σ 残差图两种输出面——二者共享同一次前向提取，只是读出粒度不同） | 待定——须晚于 A1 全部特征改动（r_ε/r_x 分离①、方向余弦②，必要时反演距离③/频域④，见 `paper_experiment_plan` §2 A1）落地后的最终 commit；不早于 gate1 v2 P1（`54e207b`）锁定时点。草案阶段留空。 |
| P2 | cell 网格：算子集 × 生成器集 × 掩码面积桶界 | 候选（9.5 原文字面枚举，锁定时以实测 cell 计数矩阵替换/确认）：算子集 **K 候选=6** `{inpaint, outpaint, object_replacement, background_editing, img2img, instruct_edit}`；生成器集 `{SD1.5, SDXL, Kandinsky 2.2}`；掩码面积桶界候选 = 9.2b 默认 `{(0,0.05], (0.05,0.15], (0.15,0.35], (0.35,0.7]}`（**仅对掩码类算子生效**——img2img/instruct_edit 为全图算子、M 坐标退化、不进面积桶维，见 §4 纤维化协议）。 |
| P3 | nuisance 采样声明 | 依据 9.1 网格：CFG∈{5,7.5,10}×steps∈{30,50}（或 config 连续采样 CFG~U(4.5,10.5)）。**须显式声明本轮 probe 生成时 prompt bank（9.2a）是否已接入**：若仍空 prompt，则据 CFG 惰性发现（`paper_experiment_plan` §3 B1.6 附注，2026-07-16：空 prompt 下 cond≡uncond，CFG 项在 classifier-free guidance 中精确消去）本轮 CFG 维实际不活性，敏感度只由 steps(+seed) 承载，须在报告 DATA 节写明"本轮 CFG 惰性"；prompt bank 生效后的后续轮次须重新声明本项状态，不得默认沿用旧声明。 |
| P4 | n_base 与每 cell 计数下限 | n_base ≥ 200（9.5 原文，覆盖 same-model/cross-model/方向性预言/null-exceedance 各判定边际）；每 cell 计数下限 = config 项，草案默认沿用 V12 同类设计原则（保证每个 one-vs-rest 二分类臂在两类上均有足够样本）；实际生成量受"同底图跨算子成对"设计约束（掩码类算子间同 base 同 mask 成对出现，非自由网格），具体数字与实测计数矩阵锁定时填。 |
| P5 | 方向性预言两估计器（空间头 CNN / 全局幅值 probe）的实现冻结 commit | 待定——两估计器须在**同一 commit** 下一起冻结（架构、超参、训练协议），冻结后不得为追平判据而调整；草案阶段留空。null baseline（`checking/testc_geometry_probe.py`）改动很小（见 §2），不占用本占位符，但若有改动仍须在同一冻结 commit 内一并锁定。 |

---

## 1. 主张与范围

检验主张：编辑算子在多σ score 残差场上留下可分类型签名，且该签名 (a) 在单一生成器内部可分（same-model）；(b) 部分迁移到未见生成器（cross-model，操作 > 模型指纹，呼应 C5）；(c) 空间读出（CNN 读残差图的 2D 结构）显著优于只读全局幅值统计的读出方式（方向性预言，呼应机制 (ii) RePaint 投影）；(d) 显著优于纯掩码几何 null baseline（排除"只是学到掩码形状"的混淆，呼应 PATCH 8.3）。

本预注册只管辖 B3 主生成随附、`probe_group=gate2` 的 n_base≥200 controlled probe（9.5 设计冻结 cell 网格，§0 P2）上的**一次性验证性评估**。此前所有 gate2 结果——`checking/gate2.py` 现行 5 类 NearestCentroid 结果（`docs/real_gate_results_2026-07-02.md`：same 0.372 / cross 0.374，真实数据 n=900；mock 阶段 same 0.327 / cross 0.328）、`docs/PAPER_DESIGN_v2_2026-07-15.md` E8 记载的 n=50 全局幅值 probe 探索值（≈0.48，压线未过 0.50）——均为探索性，不复述、不计入，也**不构成本轮任何判据的锚点或阈值来源**。

**与既有 `checking/gate2.py` / `docs/EXECUTION_CHECKLIST.md` 闸门 2 表的关系（延续与升级）**：延续其 same-model/cross-model 术语与 0.50/0.40 主阈值（旧表的 0.30 附加分界见 §5 落点表内嵌注）；升级点——①评估协议从一次性 60/40 `group_split` 换成 repeated group K-fold + cluster bootstrap（gate1 v2 协议模板同一套机制）；②分类器架构从单一 NearestCentroid 扩展为三配置对照（主=空间头 CNN，配对对照=全局幅值 probe，null=掩码几何 logistic）；③K 从 5 类扩到 6 类（新增 `instruct_edit`，PATCH 8.2 c 轴）；④新增 null-baseline 超出性要求，旧版本没有这条防线。

## 2. 冻结项

- **cell 网格（P2）**：见 §0；同底图跨算子成对（掩码类算子间同 base 同 mask）以控内容混淆，行打 `probe_group=gate2`，随 B3 一并生成（9.5 原文）。probe 是受控仪器（`validate.py` 模块 docstring 裁决B：V8/V10 只在 `profile=="run"` 时执行），gate2 probe 网格允许覆盖主 run split 中受限的算子（如 `testc_holdout=object_replacement`，`configs/split.yaml`）——这是仪器设计的一部分，不构成 split 泄漏。
- **三配置（分类器，互不作为彼此的"第二次及格机会"）**：
  1. **主配置** = 空间头（CNN 读多 σ 残差图的 2D 空间结构，P1 残差图输出面 + P5 架构冻结）——用于 §3.1 主判据（same-model / cross-model BA）。
  2. **配对对照** = 全局幅值 probe（对同一残差场做全局池化幅值统计的分类器，P1 `profile()` 池化输出面 + P5 架构冻结；功能上是 gate1 amp-only 概念在 gate2 场景下的类比）——只服务于 §3.2(a) 方向性预言的配对比较，**不作为主判据的候选**（"任一配置过线即 PASS"的解读预先排除，同 v2 §2 配对对照条款原话）。
  3. **null baseline** = 掩码几何-only 分类器（复用 `checking/testc_geometry_probe.py` 的 `mask_geometry()` 5 维特征 + `logistic_ovr_scores()` 手写 OvR logistic；由 K 个 one-vs-rest 分数取 argmax 得到多分类决策，以获得与上两个配置口径一致的多分类 BA——这是对原脚本"每算子独立报 AUC/BA"输出的**最小泛化**。原脚本自身的 60/40 `group_split`（seed=0）**须替换**为本文件下方"配对可比性"约定的共享折协议，不得沿用其独立 seed）——只服务于 §3.2(b) null-exceedance 的配对比较，同样不作为主判据候选。
- **分类协议**：same-model / cross-model 算子分类，OOF = repeated group K-fold（按 `base_id` 分组，同底图/同 pair 不跨折，沿用 gate1 v2 §2 `repeated_group_kfold` 机制模板）。
  - **same-model** = 单一生成器内部（对每个生成器分别做 group K-fold 分类，K 个算子的 argmax 决策，跨生成器取平均——延续 `gate2.py::_acc_within` 的"逐族内部"定义，只是把其一次性 60/40 换成 repeated K-fold）。
  - **cross-model** = 在训练可见生成器集合上拟合、在留出生成器（`configs/probe.yaml` 的 `holdout_generators`，gate2 probe 沿用既有留出族机制，样本标 `split=test_b`）上零样本评估，双向（"训 A 测 B"各方向都做——延续 `gate2.py::cross_acc` 的双向配对定义，只是把逐样本单次训练换成 group K-fold 下的重复估计）。
  - 分类器输出多分类标签直接计算 balanced accuracy，不需要连续量校准（与 v2 t0 桶化校准是不同结构）。
- **CI**：按 `base_id` cluster bootstrap（B ≥ 2000，percentile 95%），沿用 gate1 v2 `cluster_bootstrap_indices` 机制模板。
- **配对可比性**：三配置（空间头 CNN / 全局幅值 probe / 掩码几何 null）共享同一 `base_id` 分组的折划分与同一 bootstrap 索引序列（固定 seed，写入脚本，沿用 `gate1_confirmatory.py` 的 `SEED` / `SEED+1` 双常量模式）。
- **一次性评估**：全量 gate2 probe 数据出齐后运行一次；不看中间结果；verdict 由 §5 机械导出，不允许解释性调整。

## 3. 判据（全部数值现在写死，不因 gate1 大胜调整）

### 3.1 主判据（两级，均为点估计门槛；CI 强制报告但不额外叠加"CI 下界>0"门——与下方 3.2 的两个增量判据结构不同）

- **same-model**：算子 BA ≥ **0.50**（K = P2 候选算子集合基数，草案候选 K=6，机会线 = 1/K ≈ **0.167**；较 `docs/EXECUTION_CHECKLIST.md`/`checking/gate2.py` 旧文档标注的 1/5=0.20 更低，因新增 `instruct_edit`，报告中不得混用旧机会线数字）。
- **cross-model**：算子 BA ≥ **0.40**（K 可能小于 same-model 的 K——受限于留出生成器实际支持的算子子集，例如 `instruct_edit` 目前只在 SD1.5 下存在；报告须分别列出 same-model 与 cross-model 各自的 K 与机会线，不得混用）。
- 本节两条判据是**点估计门槛**，不要求 cluster CI 下界额外过线（与 addendum §9.5 原文一致：只述"BA≥0.50/0.40（注明 K 与机会线），cluster CI（按 base_id）"，未叠加 CI 下界条款）；cluster CI 仍必须报告（透明度要求），且是 §3.2 两个配对增量判据的计算基础。
- 阈值 **0.50 / 0.40 沿用 2026-07 原计划**，**不因 gate1 t0 支的验证性复测大胜而抬高或降低**（见文首「阈值声明」）。

### 3.2 增量判据（两份，各自独立，均为配对 cluster bootstrap 95% CI 下界 > 0 的门）

- **(a) 方向性预言**：ΔBA(空间头 CNN − 全局幅值 probe) 配对 cluster bootstrap 95% CI 下界 > **0** → 方向性预言成立，许可"空间读出显著优于全局幅值统计"一类表述（呼应机制 (ii) RePaint 投影的正式检验，`docs/PAPER_DESIGN_v2_2026-07-15.md` §2.3(ii) / E9）。
  背景（探索性，仅作动机说明，**非本轮判据锚点**）：E8 记载 n=50 探索轮全局幅值 probe same-model BA≈0.48，压线未过 0.50；本轮判据是在同一批 n≥200 数据上对两个估计器**重新估计**的配对 Δ，与该探索性数字无引用/继承关系，不得作为本轮阈值来源。
- **(b) null-exceedance**：ΔBA(空间头 CNN − 掩码几何 null) 配对 cluster bootstrap 95% CI 下界 > **0** → 算子主张可记在"score 签名"头上（`docs/PATCHES_addendum_06_07_2026-07-15.md` §9.5 原文："算子主张必须以配对 Δ（cluster CI 下界 > 0）显著超过它，才能记在 score 签名头上"）；不过 → 算子可分性无法排除"仅学到掩码几何"的混淆，即使 §3.1 两级 BA 都过线，也不得称信号来自 score 残差场，须收窄措辞（见 §5 落点表）。null-exceedance 在 same-model 与 cross-model 两个层级**分别检验**（各自配对、各自 CI），§5 落点表按"细粒度算子级 vs 仅族级超出"两种结果分流。

### 3.3 结构化产出（无阈值，必出）

- 算子混淆矩阵（K×K，same-model 与 cross-model 各一份；延续 `gate2.py::_plot` 的 PCA-2D 可视化传统，另加混淆矩阵热图）；
- 三配置（空间头 / 全局幅值 / 几何 null）BA 对照表 + 各自 cluster CI；
- 按 P2 面积桶切片的 BA 曲线（仅掩码类算子子集，呼应 9.2b 编辑面积–归因掉点曲线的评测钩子）；
- 逐算子 one-vs-rest AUC（复用 `testc_geometry_probe.py::run()` 的 `per_operator` 结构，null baseline 与主配置各出一份，便于逐算子比较而不止看聚合 BA）。

## 4. 纤维化指标协议

算子分类 BA（§3.1 主判据、§3.2 两个增量判据）在**全体** P2 候选算子集合上统一统计——算子类别坐标 `c` 对每一行恒定义，不存在纤维退化问题。

以下附属指标（若在 EXPLORATORY-ADDENDA 中报告）**仅在对应纤维子集**上统计，禁止用全集分母稀释：

- **t0 类指标**（如按算子分层的强度—残差相关探索性附录）仅在**strength 可变算子子集**（img2img 与掩码类算子：inpaint/outpaint/object_replacement/background_editing）上统计，**排除 `instruct_edit`**——其 t0≡T 恒定、无强度轴，是条件式参数化从建议固化为约束的直接后果（`docs/PATCHES_addendum_06_07_2026-07-15.md` §8.2："先预测算子类，t0 头只对 strength 可变算子有定义，t0 指标只在该子集上统计"）。
- **掩码/面积相关指标**（P2 面积桶切片 BA、掩码 IoU 等若涉及）仅在**掩码类算子子集**（M 非退化）上统计，全图算子（img2img/instruct_edit）的 M 视为退化、不进入该子集分母，也不进入 P2 面积桶维（§0 P2 已注明）。

协议细节（分折、bootstrap、一次性评估、共享折与 bootstrap 索引）沿用 §2/§3 本文件已述的 gate1 v2 协议模板，不重复给出。

## 5. verdict 落点表（机械抄写，禁止事后解读；三档 + 两条独立行）

判定顺序：先查 §3.1 两级 BA 点估计，再查 §3.2(b) null-exceedance（分 same-model / cross-model 两级）决定该档位是"细粒度算子级"还是"仅族级"成立。

| 落点 | 触发条件（机械） | 许可的论文句子 |
|---|---|---|
| **全量算子归因** | same-model BA≥0.50 **且** cross-model BA≥0.40 **且** null-exceedance 在 same-model 与 cross-model 两级均成立（细粒度算子级） | "算子机制可分且部分迁移到未见生成器（操作信号，非模型指纹或掩码几何混淆）"；`(t0,c,M)` 三坐标中 c 支全量落地；**G-A(算子)=能（上行）**，头条可用"算子逆估计"叙事（`docs/paper_experiment_plan_2026-07-15.md` §10 风险树"能(上行)"分支），与已裁决的 t0 支（强序数恢复+粗桶分级）共同构成"检测+定位+算子逆估计(含t0)"头条 |
| **算子族识别** | (same-model BA≥0.50 **且** cross-model BA<0.40) **或** (两级 BA 都过线，但 null-exceedance 仅在 same-model 级成立、cross-model 级不成立——即只能排除"同模型内学几何"，排不掉跨模型部分的几何混淆) | "算子在单一生成器内可分，跨生成器部分/未确证迁移"；若 cross-model BA < 0.30（`docs/EXECUTION_CHECKLIST.md`/`checking/gate2.py` 既有 CONFOUND 分界，本档内嵌套注明、不单列第四档）则额外许可"差距本身指向模型指纹混淆而非操作信号，为诚实局限"一句；**G-A(算子)=不能**，头条退到"检测+定位+算子族识别"（`docs/paper_experiment_plan_2026-07-15.md` §10 风险树"不能"分支），t0 已裁决的强序数恢复+粗桶分级作为独立支仍可写入，兜底线（定位+算子族+诚实刻画）成立 |
| **坍缩为定位** | same-model BA<0.50（不论 cross-model 结果，含 `gate2.py` 既有 WEAK 分界的字面延伸）**或** null-exceedance 在 same-model 级也不成立（即便 BA 点估计过线，也不能排除"学到的是掩码几何而非 score 签名"） | 算子主张全部撤下；只保留 C1/C2 兜底线本身（检测+像素级定位+诚实刻画跨生成器局限，见 `docs/paper_experiment_plan_2026-07-15.md` §0 表格"兜底线（必须拿下）"行）；`(t0,c,M)` 的 c 坐标降级为"未确证，留作 future work"；t0 支（strength 可变算子子集内）独立保留、不受本档牵连 |
| §3.2(a) 方向性预言 CI 下界 ≤ 0 | （独立行，不影响上面三档） | 论文删"空间读出显著优于全局幅值统计"一句；机制 (ii) RePaint 投影的空间签名预言未被证实，`docs/PAPER_DESIGN_v2_2026-07-15.md` E9 由"未决，预言已预注册"改记"预言未命中"；不影响上面三档的算子归因结论本身（该结论只依赖 §3.1+§3.2(b)，与方向性预言的成立与否解耦，同 gate1 v2 §4 中 C4 增量行与主判据行解耦的先例一致） |
| §3.2(b) null-exceedance 在 cross-model 级不成立、其余条件均够"全量" | （独立行，交叉情形显式兜底，防止执行者遗漏分支） | 归入"算子族识别"档（第二行已覆盖，此处仅作显式索引） |

## 6. nuisance 分支（按 P3 声明二选一，现在生效）

- **P3 声明 = prompt bank 已接入（CFG 活性）**：主张全额——"算子机制可分（若过线）"不携带 CFG 限定。
- **P3 声明 = prompt bank 未接入 / 仍空 prompt（CFG 惰性）**：本轮 confirmatory 主张自动限定为"算子机制可分（空 prompt / CFG 惰性条件下）"，verdict 与论文措辞必须携带该限定；并预注册一个补充 probe（非空 prompt 网格，仿 gate1 v2 §5 的 CFG/steps 抖动补充 probe 设计）在投稿前完成——若补充 probe 上 same-model/cross-model BA 相对本轮跌幅使某一级从"过线"变"不过线"，限定升级为正文 limitation 而非脚注（判定精神与 gate1 v2 §5、9.6 的"跌幅>0.10 升级"一致；数值待 P3 实际声明后另定，因算子分类 BA 与 t0 的 ρ 不同量纲，不能直接借用 0.10）。

## 7. 输出契约

单一报告 `checking/gate2_confirmatory_report_<date>.md`（执行脚本 `checking/gate2_confirmatory.py`，**尚未编写**，随 B3 数据落地后随附本预注册协议实现——本次为草案，不含脚本），固定结构 DATA（含 P1–P5 回填值、manifest hash、实测 cell 计数矩阵、same/cross 各自 K 与机会线）/ CRITERIA（本文件逐条引用）/ RESULTS（§3 全部数字 + CI）/ VERDICT（§5 对应行的抄写）/ EXPLORATORY-ADDENDA（此后任何追加分析置于此节且显式标注，含 §4 纤维子集指标）。

**一次性评估保护**：`checking/gate2_confirmatory.py` 起跑前必须检查输出报告路径是否已存在；**已存在即拒跑**（非 0 退出，不覆盖、不追加、不重算）；需要重算须人工删除该文件（删除行为进入 git 历史，不允许静默覆盖）——防止"跑了看一眼不满意再跑一次"式的隐蔽多次评估/择优。

## 8. 失效清单

以下任一发生即 confirmatory 地位失效（逐条移植自 `docs/PREREG_gate1_v2_2026-07-15.md` §7，据 gate2 场景调整措辞，全部条款保留）：

- 锁定前评估了 gate2 probe 数据（任何算子标签与残差/幅值/几何特征关系的查看，包括"部分数据先瞄一眼"）——数据**生成**不构成偷看，生成已在进行/完成不影响本预注册效力（同 v2 原则）；
- 锁定后修改阈值（0.50/0.40）、cell 网格（P2 算子集/生成器集/面积桶界）、协议（分折/bootstrap/估计器架构）或配置集合；
- 逐行（非按 `base_id`）分折或 bootstrap；
- 在三配置（空间头 CNN / 全局幅值 probe / 掩码几何 null）间事后择优（如临时把主判据换算到表现更好的配置上）；
- 查看部分数据后继续或调整（含"先看 same-model 结果决定要不要跑 cross-model"这类分阶段偷看）；
- verdict 偏离 §5 落点表（任何解释性调整）；
- **一次性评估保护被绕过**：confirmatory 报告文件已存在时仍重新运行并覆盖/追加（本条为新增条款，构成隐蔽的多次评估/择优；gate1 v2 未显式列出，本文件据 9.5 原文"一次性保护"要求补齐，精神与 v2 §7 一致）。

---
*本文件当前状态：**草案**，尚未锁定。锁定时须：①填毕 §0 全部占位符（P1–P5）；②将本段替换为「锁定人/时间/commit」记录（格式同 `docs/PREREG_gate1_v2_2026-07-15.md` 文末）；③将文首状态头由"草案"改为"已锁定"，并把"锁定条件"改写为已满足的事实陈述。锁定 commit 必须先于对 gate2 probe 的任何标签–特征评估——数据生成可先行，评估不可先行。*

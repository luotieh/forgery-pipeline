# PATCH 9 — B3 主生成前置项收口（2026-07-15，交 Claude Code 追加）

> ✅ **已于 2026-07-16 按下方插入说明并入 `PATCHES_addendum_06_07_2026-07-15.md`**（正文 + 同步清单增量）。本文件保留为原稿存档；以 addendum 内版本为准。

> **给 Claude Code 的插入说明**：本节追加到 `PATCHES_addendum_06_07_2026-07-15.md` 的 PATCH 8 之后、"文档同步清单"之前；文末"同步清单增量"合并进该清单。若编号 9 已被占用则顺延。命名/字段以实际仓库为准，按"定位方式"适配。
>
> **阻断关系**：9.6 最先（零成本、影响 9.1 论证与 method 脚注）→ 9.3/9.4（代码）与 9.1/9.2（配置决策）为 **B3 阻断项** → 9.5 的设计冻结先于 B3 生成（gate2 probe 行随 B3 产出）、其 prereg v3 锁定先于任何 gate2 评估。与 PATCH 7 编码可并行。

---

## 9.0 目的

PATCH 7 管住了"图怎么存"（压缩链/负样本/compositing），PATCH 8 管住了"矩阵有多大"。本 PATCH 收口三类**生成后不可回溯**的遗漏：写进像素的采样政策（9.1/9.2）、split 级防泄漏（9.3）、以及 12 小时批跑的生存性与纪律（9.4）；另将 G-A 唯一未决支（gate2）的设计与预注册前置（9.5），并清掉方法稿最后一条悬置项（9.6）。

## 9.1 主 run 的 nuisance 与强度采样政策（阻断；实验⑤的直接后果）

实验⑤已证明 CFG/steps 是真实敏感维（ρ 0.700→0.608）。主数据集若固定 (7.5, 30)：下游头在固定 nuisance 上训练、"固定 CFG/steps"限定被 6 万张图永久固化。抖动 plumbing（`293d7ef` 的 op_params/网格）已在手，边际成本为零。

- **CFG/steps**：全部扩散编辑行逐图随机采样。默认离散网格 CFG∈{5, 7.5, 10} × steps∈{30, 50} 均匀采（与补充 probe 同网格，保持可比性）；连续采样（CFG~U(4.5,10.5)）作为 config 可选项。逐行记入 `op_params`。
- **强度**：训练行 s ~ U(0.1, 0.95) **连续采样**（喂 ordinal 损失 + 为 Phase E 插值实验预留细粒度；留空格子在训练时子集化实现，不在生成层做）。probe 行保持离散网格（闸门口径不变）。
- **切片自由**：nuisance 已逐行记录 → 评测时可免费构造"固定 nuisance 子集"，无需生成侧特供。
- **validator V11**：所有扩散编辑行必含 op_params.cfg/steps；每 split 内每个网格单元计数 ≥ 配置下限（连续采样则改为分位数覆盖检查）。
- **措辞后效（记入计划文档）**：Phase C/D 的系统级结果在抖动数据上训练与评测，不再携带 gate1 的"固定 CFG/steps"限定；该限定仅约束 gate1 的 t0 序数主张本身。

## 9.2 写进像素的三个政策（阻断）

- **(a) prompt 来源**：定死并逐行记录。默认：img2img/instruct 用 COCO caption + 模板扰动库，inpaint/replace 用对象类模板库；prompt 记入既有 prompt 字段（无则 op_params.prompt），模板库版本以 `prompt_bank_version`（hash）入行，保证可复现。
- **(b) 掩码面积分层**：生成时计算并记录 `mask_area_ratio`；按桶分层采样（默认 {(0,0.05], (0.05,0.15], (0.15,0.35], (0.35,0.7]}，config 可调）；outpaint 边带宽度给网格（默认边长的 {12.5%, 25%}）。依据：编辑面积是已知的归因混淆维（Xu et al. 的面积–掉点曲线），评测必须可按面积切片。**validator V12**：masked 算子行必含 mask_area_ratio 且每桶计数 ≥ 下限。
- **(c) 多分辨率组**：按生成器分组（SD1.5@512 / SDXL@1024 等）。**每个分辨率组必须含走同一非生成链的 real 行与 vae_rt 行**，否则 PATCH 7 的 V2 断言在组内空转；V2 的"可比组"定义显式扩展为含分辨率维（io_chain 的 rs{N} 已记录，V2 实现按组执行即可）。

## 9.3 validator V8–V10：split 防泄漏（阻断；行级断言 V1–V7 覆盖不到的类别）

- **V8 底图互斥**：同一 `base_id` 的全部行（real、real_vae_rt、各算子编辑、成对 probe）必须落在同一 split；退化行（`postprocess_of`）继承母行 split，断言一致。若 manifest 尚无显式 `base_id` 字段，新增之（real_image_path 可作回填键），回填脚本扩展 `backfill_manifest_v7.py`。这是 gate1 group-fold 教训在数据集层的对应物。
- **V9 生成器 holdout 互斥**：test_b 的 holdout 生成器族/名不得出现在 train/val 任何行。
- **V10 算子 holdout 互斥**：test_c 的 holdout 算子不得出现在 train/val；holdout 算子值来自唯一 config 源（`testc_holdout`），PATCH 8.3 判定完成后即写死。
- 三条各配**注毒负例单测**：构造一条违例行，断言 validator FAIL 并打印违例键。

## 9.4 B3 驱动加固（阻断；把事故 A/B/C 从教训固化为断言）

- **HEAD 断言**（事故 C）：驱动起跑前 assert `git rev-parse HEAD == 期望 commit`（config 值），工作区不净则拒跑或显式记录。
- **评估禁令**（事故 B）：生成驱动内**不得内置**任何对 probe/confirmatory 数据的评估步骤；评估脚本独立、锁定后手动触发。以代码审查 + 驱动内注释断言双保险。
- **断点续跑幂等**：按 cell（生成器×算子×nuisance 单元×批次）落 done-marker；manifest 原子追加（临时文件 + rename，或 JSONL append+fsync）；重启跳过已完成 cell、检测半批残留。验收含"中途 kill → 重启 → 无重复行、计数吻合"的演练。
- **fetch 层**（事故 A）：COCO 底图下载走已修复的超时层（确认 socket timeout 对新数据源路径生效）+ 归档 checksum 校验。
- **latent 复用**（日志已记待办）：每底图每分辨率组 VAE encode 一次，跨强度/nuisance 单元复用（`diffusers_gen` 内实现，缓存键 (base_id, resolution)）。**纯优化红线**：同 seed 下输出必须与不复用逐字节等价（encode 确定性保证；冒烟对比验证），不等价则弃用。预期省 VAE 往返 3–5 成。
- **磁盘预检**：起跑前 assert 数据盘余量 ≥ 配置估计（60–80k PNG + 各配置 npz 残差缓存，预留 ≥100GB），双备份策略延续。

## 9.5 gate2 probe 设计冻结 + PREREG_gate2_v3（设计先于生成、锁定先于评估）

**设计冻结（B3 起跑前 commit 进 config）**：
- cell = 算子 {inpaint, outpaint, replace, background, img2img, instruct_edit} × 生成器 {SD1.5, SDXL, Kandinsky} × 掩码面积桶（9.2b）；nuisance 按 9.1 采样；n_base ≥ 200 覆盖各判定边际。
- **同底图跨算子成对**（掩码类算子间同 base 同 mask）以控内容；行打 `probe_group=gate2`，随 B3 一并生成。

**PREREG_gate2_v3 草案（以 v2 为模板另立文件，本 PATCH 只规定必备要素，评估前锁定）**：
- 主判据沿用原计划：same-model 算子 BA ≥ 0.50、cross-model ≥ 0.40（注明类数 K 与机会线），cluster CI（按 base_id）。**阈值不因 gate1 大胜而抬高或降低。**
- **方向性预言预注册**：空间头（CNN 读残差图）应显著超过全局幅值 probe（同数据、共享折、两估计器先冻结）——此前理论预测（RePaint 投影的判别量被全局统计错过）的正式检验。
- **null baseline**：掩码几何-only 分类器（直接复用 PATCH 8.3 脚本）——算子主张必须以配对 Δ（cluster CI 下界 > 0）显著超过它，才能记在 score 签名头上。
- 纤维化指标协议：逐纤维统计（t0 类指标仅 strength 可变子集），沿用 gate1 协议模板（group K-fold、cluster bootstrap、一次性评估、共享折与 bootstrap 索引）。
- verdict 落点表：全量算子归因 / 算子族识别 / 坍缩为定位 三档，各档许可措辞与 G-A 算子支的叙事映射，机械抄写。

## 9.6 清尾：(7.5, 30) 单元分解（非阻断，最先做；本地 npz，零 GPU）

从 `gate1_cfgsteps_features.npz` 出两组数，全部标注 exploratory、写入补充报告 ADDENDA 节：
- **切片视图（主）**：pooled 抖动模型的 OOF 预测按 6 个 (cfg, steps) 单元切片各算 ρ + cluster CI（每单元 250 行 / 50 簇，CI 偏宽如实注明）。若 (7.5,30) 切片 ρ ≈ 其余单元 → 0.092 主要来自底图子集/估计器差，nuisance 本身敏感度低；若显著高于其余 → 真敏感。
- **重拟合视图（辅）**：仅 (7.5,30) 单元 250 行按主协议重拟合 OOF → ρ_refit。分解：base_effect = 0.700 − ρ_refit（同条件、200→50 底图）；nuis_effect = ρ_refit − 0.608（同 50 底图、固定 vs 抖动；n 差异 250 vs 1500 为已知混淆，注明）。
- **决策规则**：nuis_effect > 0.10 → method 稿脚注**主动升级**为 limitation 正文（中英两稿同步 v3，实验日志记录）；否则脚注不动、页边注待办划除。

## 9.7 验收标准

- [ ] 9.6 报告落盘（ADDENDA 标注），脚注升级与否的结论回写 method 文档决策记录与页边注。
- [ ] V8–V12 各配注毒负例单测，违例必 FAIL；旧 manifest 经回填后 V8 通过。
- [ ] 驱动演练：中途 kill → 重启 → 无重复行；HEAD 断言演示生效（改一位 commit 期望值须拒跑）。
- [ ] nuisance/强度/prompt/掩码政策全部落 config（零硬编码）；冒烟 run 的 `stats` 显示 op_params 网格、mask 面积桶、prompt_bank_version 覆盖。
- [ ] latent 复用同 seed 逐字节等价验证通过（不等价则关闭该开关并记录）。
- [ ] gate2 设计冻结 config + PREREG_gate2_v3 草案（占位符齐）在 B3 起跑前 commit；v3 锁定 commit 先于任何 gate2 评估（沿用 v2 的失效清单与一次性保护）。
- [ ] 磁盘预检通过。

## 9.8 不改什么

- PATCH 7 的 io_chain/sample_kind/compositing 语义零改动（9.2 只是新增分层字段与组定义扩展）。
- gate1 已落盘结论与 prereg v2 不动；9.6 为 ADDENDA 级探索，不改 verdict。
- prereg v3 阈值沿用 7 月计划原值（0.50 / 0.40）。
- probe 行的离散强度网格不动（闸门口径连续性）。

## 同步清单增量（合并进 addendum 末尾清单）

- `paper_experiment_plan_2026-07-15.md`：§3 B3 加入 nuisance/强度连续采样、prompt bank、掩码面积分层、分辨率组 real 行配套；§2 gate2 判据行 → 指向 PREREG_gate2_v3；§8 预算注明 latent 复用后的时耗修正。
- `GATE_DATA.md` 字段对照表加入：`base_id / mask_area_ratio / prompt_bank_version`（op_params 已由 PATCH 8 登记）。
- `PATCHES.md` 追加 PATCH 9 条目；RUNBOOK（若无则新建）固化事故 A/B/C 三断言。
- method 中文/英文稿：9.6 结论落定后按 D2 遗留项同步（脚注不动或升 limitation，二选一）。

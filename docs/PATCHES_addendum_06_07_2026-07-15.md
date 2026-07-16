# PATCHES 增补 — PATCH 6 / 7 / 8（2026-07-15）

> 对接：`paper_experiment_plan_2026-07-15.md` §2（Phase A）与 §3（Phase B）；延续 `PATCHES.md` 编号（现至 PATCH 5；若 6/7/8 已被占用则顺延，并同步改本文档标题）。
>
> **给 Claude Code 的执行原则**：
> 1. 文中的文件名/函数名/字段名以**实际仓库为准**——按"定位方式"找到目标，命名不同就适配，不要为了对齐本文档而重命名既有代码。
> 2. 两个 PATCH **互不依赖，可并行**；但 PATCH 6 必须在 A1-①（r_ε/r_x 分离）等任何特征改动**之前**完成并出报告——否则"换度量"与"换特征"两个变量混在一起，增量归因失效。
> 3. PATCH 7 是 Phase B3 主生成的**阻断项**：未合并 + 未过冒烟验收前，不启动主数据集生成。
> 4. PATCH 8 分批执行：8.3（Test-C holdout 判定）零生成成本，须在 B3 前出结论；8.1/8.2 随主生成实现；8.4（Flux）走独立环境作业、可在主生成后补跑——但全部新字段与 split 约定须与 PATCH 7 **一次性定死**，避免 manifest 二次迁移。
> 5. 每个 PATCH 完成后在 `PATCHES.md` 追加条目，并按文末"文档同步清单"更新相关文档。

---

## PATCH 6 — gate1 改回归口径 + 混淆结构复判（零新生成，纯离线复算）

### 6.0 目的与科学口径

- **主张重述**：gate1 检验的科学主张是"t0/strength 在多σ残差剖面上留下**单调可恢复**的签名"。该主张的原生度量是秩相关与回归误差（ρ / MAE），而非均匀分桶上的 balanced accuracy。现状 BA 0.475 / ρ 0.476（n=50）呈"ρ 显著、BA 不过"的典型分裂，最可能的解释是**信号单调但高强度端饱和**（SDEdit strength→1 时趋于全图重生成，剖面收敛）——本 PATCH 用混淆结构诊断直接检验这一预测。
- **诚实边界（必须写进报告 VERDICT）**：本次在既有 n=50 probe 上的复算属**探索性诊断**。由此产生的任何新阈值须在报告中**预注册**，并在 Phase B3 的 n_base≥200 强度网格上做**验证性**复测后，方可作为决策门 G-A 的依据。禁止仅凭本次复算宣布 gate1 PASS。
- **变量隔离**：本 PATCH **冻结当前特征**（`checking/extractor.py:profile()` 现状，含 residual_stack 对 r_ε/r_x 的坍缩）。A1-① 完成后重跑同一脚本，两份报告的差值即特征升级的增量证据。

### 6.1 改动清单

**新增** `checking/gate1_regression.py`（若仓库 gate 脚本另有命名惯例，按惯例取名，如 `gate1_v2.py`；**不覆盖**原 gate1 脚本，二者并列可复跑）。

**定位与输入**：
- 复用既有 gate1 probe manifest（带 `strength` 的 img2img/SDEdit 行）与**已缓存**的残差特征/profile 向量。若特征未缓存，仅允许对已存图离线重提取；**禁止任何新生成**。
- 单σ对照特征（AEROBLADE 口径）的 σ 选择：复用原 gate1 脚本中已有的单σ对照定义；若原脚本没有，取现行多σ集合中的单点（选与 AEROBLADE 论文最接近的中等噪声级），并在报告 METHOD 里写明选点依据。
- 桶数 K 与桶边界：**从现有 gate1 脚本读取**，勿另设。若现行做法是把强度网格的离散水平直接当类别，则本文所有"桶"分析按序数类别同样执行。

**功能模块**（参考实现见 6.2，仓库耦合处自行适配）：

(a) **回归主指标（out-of-fold）**
- n=50 → RepeatedKFold(5 折 × 20 重复) 或 LOO；模型用 Ridge（每折内独立标准化，避免泄漏），不上重模型。
- 目标变量两套：① raw strength `s`；② log-SNR(t0) 变换——`t0 = round(s·(T−1))`，`SNR = ᾱ_t0/(1−ᾱ_t0)`，ᾱ 从仓库现有 SD1.5 scheduler 对象读取（`scheduler.alphas_cumprod`），**不要硬编码 beta 表**。
- 报告：Spearman ρ（OOF）、Pearson r、MAE(s)、MAE(logSNR)，各配 bootstrap 95% CI（≥2000 次按样本重采样）。

(b) **校准后分桶 BA**（直接检验"信号单调、桶画错"假说）
- 训练折上用 isotonic 回归把 1-D OOF 预测校准到 strength 轴，然后在三套桶方案上算 BA：
  ① 现行均匀桶（原口径，作对照）；② 等频分位数桶；③ 粗三桶（低/中/高——切点先按 (c) 的饱和分析定，默认候选 [min,0.45)/[0.45,0.7)/[0.7,1.0]）。

(c) **混淆结构诊断**（对"现有分类器预测"与"校准回归器分桶"两套输出都做）
- 混淆矩阵热图（K×K）。
- **邻接错误占比** = |pred−true|==1 的错误数 / 全部错误数。
- 上三角 vs 下三角错误质量比；按 true bucket 的 recall 曲线（饱和预测：高强度桶 recall 最低，错误集中上三角邻带）。
- **相邻桶两两可分性**：对每对相邻桶 (k, k+1)，用同一 1-D 投影算 AUC，画 AUC-vs-k 曲线（饱和预测：随 k 递减）。
- 判定规则（写死在脚本里，输出到 DIAGNOSIS）：邻接错误占比 ≥ 0.70 **且** 相邻桶 AUC 随 k 单调或近单调递减 → 判"**单调 + 饱和**"；否则判"信号弥散"（此时问题不在桶而在特征，A1 升级为必需路径）。

(d) **多σ增量在回归口径下复测**
- Δρ(多σ − 单σ) 用**配对 bootstrap** 给 95% CI；同时报校准后 ΔBA 作对照。原 +0.025 是 BA 口径，必须给出 ρ 口径的对应数——这直接决定 C4"多尺度"一词的存废证据形态。

(e) **报告输出**
- `checking/gate1_regression_report_2026-07-15.md` + 图：pred-vs-true 散点（OOF）、两套混淆热图、相邻桶 AUC 曲线、多σ vs 单σ 对比。
- 报告固定结构：`DATA`（含样本数、强度网格、**extractor 标注**——按既有红线区分 real extractor vs multisigma 代理）/ `METHOD` / `RESULTS`（全部数字带 CI）/ `DIAGNOSIS`（饱和判定结论）/ `VERDICT`（探索性结论 + 下方预注册块）/ `NEXT`。

(f) **预注册块**（写入报告，并同步修改 experiment plan §2 A1 的闸门 1 判据行，引用本报告路径）
- 建议的新 gate1 判据（**在 n_base≥200 网格上验证性执行后才生效**）：
  - 主判据：OOF Spearman ρ ≥ 0.50 且 95% CI 下界 > 0.30；
  - 辅判据：分位数桶或粗三桶 BA ≥ 0.55；
  - sanity：MAE(s) ≤ 0.15。
- 落点解释规则保持与原计划一致：ρ ∈ [0.30, 0.50) → t0 降级为"粗桶强度分级"；ρ < 0.30 → t0 主张撤下，G-A 走兜底分支。
- C4 多σ主张改挂 Δρ：**Δρ 的 95% CI 下界 > 0** 才可在论文中称"多σ对 t0 估计有增量"。

### 6.2 参考实现（仓库无关的统计核心，装载部分自行适配）

```python
import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

def oof_ridge(X, y, alpha=1.0, n_splits=5, n_repeats=20, seed=0):
    """折内标准化的 repeated-KFold OOF 预测，返回对重复取平均后的 OOF 向量。"""
    X, y = np.asarray(X, float), np.asarray(y, float)
    acc = np.zeros(len(y)); cnt = np.zeros(len(y))
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    for tr, te in rkf.split(X):
        sc = StandardScaler().fit(X[tr])            # 只在训练折 fit，防泄漏
        m = Ridge(alpha=alpha).fit(sc.transform(X[tr]), y[tr])
        acc[te] += m.predict(sc.transform(X[te])); cnt[te] += 1
    return acc / cnt

def boot_ci(y, p, fn, B=2000, seed=0):
    rng = np.random.default_rng(seed); n = len(y); vals = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        vals.append(fn(y[i], p[i]))
    return np.percentile(vals, [2.5, 50, 97.5])

def paired_boot_delta_rho(y, p_multi, p_single, B=2000, seed=0):
    """Δρ = ρ(multi) − ρ(single) 的配对 bootstrap CI。"""
    rng = np.random.default_rng(seed); n = len(y); d = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        d.append(stats.spearmanr(y[i], p_multi[i]).statistic
                 - stats.spearmanr(y[i], p_single[i]).statistic)
    return np.percentile(d, [2.5, 50, 97.5])

def calibrated_buckets(y_true, p_oof, edges):
    """isotonic 校准 OOF 预测后按 edges 分桶，返回 (true_bucket, pred_bucket, BA)。
    注：n=50 下允许全量 isotonic（单调映射不改秩，对 BA 的乐观偏差有限），
    报告中注明；若要求严格，可在 KFold 内嵌套校准。"""
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_oof, y_true)
    p_cal = iso.predict(p_oof)
    bt = np.digitize(y_true, edges[1:-1])
    bp = np.digitize(p_cal,  edges[1:-1])
    return bt, bp, balanced_accuracy_score(bt, bp)

def confusion_diagnostics(bt, bp, K):
    C = np.zeros((K, K), int)
    for a, b in zip(bt, bp): C[a, b] += 1
    E = C.copy(); np.fill_diagonal(E, 0)
    adj = sum(E[k, k+d] for k in range(K) for d in (-1, 1) if 0 <= k+d < K)
    return dict(
        C=C,
        adjacency_ratio = adj / max(E.sum(), 1),
        upper_mass = int(np.triu(E, 1).sum()),
        lower_mass = int(np.tril(E, -1).sum()),
        per_bucket_recall = np.diag(C) / np.maximum(C.sum(1), 1),
    )

def adjacent_pair_aucs(bt, proj_1d, K):
    """相邻桶 (k, k+1) 上用同一 1-D 投影的可分性 AUC。"""
    out = []
    for k in range(K - 1):
        m = np.isin(bt, [k, k+1])
        if m.sum() < 4 or len(np.unique(bt[m])) < 2:
            out.append(np.nan); continue
        out.append(roc_auc_score((bt[m] == k+1).astype(int), proj_1d[m]))
    return out

def logsnr_of_strength(s, scheduler):
    """s∈(0,1] → log-SNR(t0)。scheduler 为仓库现用 SD1.5 diffusers scheduler。"""
    T = scheduler.config.num_train_timesteps
    t0 = int(round(float(s) * (T - 1)))
    abar = float(scheduler.alphas_cumprod[t0])
    return float(np.log(abar / (1.0 - abar)))
```

### 6.3 验收标准

- [ ] 不产生任何新图；在已缓存特征上端到端运行 < 数分钟。
- [ ] 报告含 6.1(a)–(d) 全部数字与 CI、四类图齐全，结构符合 (e)。
- [ ] VERDICT 明确区分"探索性结论"与"预注册待验证阈值"，且 experiment plan §2 A1 判据行已同步（附报告路径引用）。
- [ ] 原 gate1 脚本未被修改、仍可复跑；gate2 判据未动（`confusion_diagnostics`/`adjacent_pair_aucs` 可被 gate2 复用，但复用属后续工作，不在本 PATCH 范围）。

### 6.4 不改什么

- `extractor.py` 特征逻辑零改动（A1-① 另行提交）。
- 不动闸门 0/2/3/4 的判据与脚本。

---

## PATCH 7 — 压缩链对齐 + compositing 显式化（Phase B 主生成前的阻断项）

### 7.0 目的

1. **消除压缩历史混淆**：现状是"真实 = COCO web JPEG（单次未知 Q 压缩，直接入库）vs 编辑 = 解码 → 扩散/VAE → 重存"。检测头可以只学压缩/存储指纹就在 Test-A 上拿高分——取证经典陷阱，且是**训练集层面**的混淆，Test-E 的推理时退化评测治不了。TIFS 审稿人必查。
2. **paste-back 显式受控**：diffusers inpaint 默认整图 VAE 解码直出（无回贴），意味着编辑图的"未编辑区"也经过 VAE 往返——这压缩了编辑/未编辑区的残差对比（检测偏易、定位偏难），可能是 gate0 定位 0.643 偏弱的成因之一；而真实世界工具（如 A1111 系）普遍做回贴。两种形态都存在于野外，必须变成显式变量而非隐式默认。

### 7.1 canonical I/O（真假共享同一条非生成处理链）

**原则**：真实图与编辑图共享**同一条**载入→缩放→存储路径，唯一差异是中间是否插入编辑算子。

**定位**：`forgery_pipeline` 中真实底图入库与编辑输出保存的写出层。若真/假两侧目前各自散落保存逻辑，先重构出公共出口 `save_canonical(img, meta) -> path`，两侧统一走它。

**改动**：
1. **统一载入**：真实底图与编辑管线输入使用同一个 `load_and_resize()`（同解码器、同插值、同目标分辨率与 crop 策略）。分辨率策略按生成器分组时（SD1.5=512 / SDXL=1024 等），策略本身写入 io_chain（见 4），同一可比组内必须一致。
2. **统一存储**：主库一律 **PNG**（无损）。理由：不再引入一层有损压缩，把 JPEG 鲁棒性完整留给既有 PATCH 5 的 Test-E 独立退化行——保持退化轴的独立性与回链机制不动。
3. **真实图也走全链**：web 原件不得直接拷贝入库，必须 decode → resize → save_canonical，与编辑输入共享前两步。
4. **新字段 `io_chain`**（string）：记录逐节点处理链，语法示例：
   - 真实：`decode>rs512>png`
   - VAE 往返负样本：`decode>rs512>vae_rt:sd15>png`
   - 编辑：`decode>rs512>edit:sd15_inpaint>png`
   `stats` 命令新增按 `io_chain × is_fake × split` 的计数输出。

**附注**：paste-back 变体（见 7.3）的未编辑区保留源图解码后的像素统计，与同链真实图天然对齐——7.1 与 7.3 相互加强，一起把"可被 is_fake 预测的非生成差异"压到最低。

### 7.2 VAE 往返硬负样本（DRCT 式）

1. **新样本类型**：真实图过 VAE encode→decode（无扩散、无编辑、无掩码），标签 `is_fake=0`。
2. **新字段 `sample_kind`** ∈ `{real, real_vae_rt, edited}`（全行必填；比只在 real 侧挂布尔更干净，下游切片直接 group-by）。
3. **VAE 选择**：默认用 scorer 同款 **SD1.5 VAE**（最可能泄漏进 scorer 残差的正是它；fp32 走既有防 NaN 约定）。io_chain 记 `vae_rt:sd15`。可选扩展：少量 `vae_rt:sdxl` 行，非必需。
4. **配比（配置项，勿硬编码）**：train 中 `real_vae_rt` ≈ 真实图行数的 10–20%；`test_a` 与 `test_f` 必含。
5. **下游钩子**：Test-F 的 FPR 按 `sample_kind` 分列报告（论文表新增一列：FPR@plain_real 与 FPR@vae_rt_real）——这一列同时充当"信号来自生成/编辑痕迹而非 VAE 重采样"的正面证据。

### 7.3 compositing（paste-back）显式化

1. **生成后端改动**（`forgery_pipeline.backends.real.diffusers_gen`，masked 算子：inpaint/outpaint/replace/background）：新增参数
   - `compositing ∈ {none, paste, paste_feather}`，`feather_px: int`（仅 paste_feather 必填，默认 8）。
   - `none`：现状，整图 VAE 解码直出；
   - `paste`：像素域硬回贴 `out = orig·(1−M) + gen·M`（M=1 为编辑区）;
   - `paste_feather`：M 先高斯羽化（σ=feather_px）再混合。
   - **审计**：检查现有调用是否触发 diffusers 的 `padding_mask_crop` / overlay 路径——该路径会**隐式** paste-back，必须确保 `compositing=none` 分支真的没有隐式回贴，否则字段失真。

   参考实现：
   ```python
   import numpy as np, cv2

   def composite(orig_rgb_u8, gen_rgb_u8, mask01, mode="none", feather_px=8):
       """mask01: HxW float/bool，1=编辑区(生成)。orig/gen 必须同尺寸——若管线内部
       有 resize，先把 gen 对齐回 orig 分辨率再混合，并 assert 形状一致。"""
       if mode == "none":
           return gen_rgb_u8
       m = mask01.astype(np.float32)
       if mode == "paste_feather":
           m = np.clip(cv2.GaussianBlur(m, (0, 0), float(feather_px)), 0.0, 1.0)
       m = m[..., None]
       out = orig_rgb_u8.astype(np.float32) * (1 - m) + gen_rgb_u8.astype(np.float32) * m
       return np.clip(np.round(out), 0, 255).astype(np.uint8)
   ```
2. **manifest 新字段**：`compositing`、`feather_px`。约定：img2img/SDEdit 等全图算子固定 `compositing=none`；masked 算子必填。定位 GT 不变（mask_path 仍是语义编辑掩码）；`compositing=none` 行的"全图经 VAE"属性由该字段隐含，分析时据此切片。
3. **历史数据回填**：写 `scripts/backfill_manifest_v7.py`，给既有 manifest 补 `sample_kind`（按 is_fake 推断 real/edited）、`compositing=none`、`io_chain=legacy`，使其通过新 validator（V5 向后兼容）。在 PATCHES.md 记录回填约定。
4. **主 run 配比**（配置项）：masked 算子按全局 50/50 分层生成 `none` / `paste_feather` 两变体——同一 (base, mask, seed) **只取一种**，不成对，避免主库翻倍。
5. **成对 probe**（供 gate0 定位复查）：单独生成 ~100 组成对样本（同 base/mask/seed，两种 compositing 各一行），字段 `probe_group=compositing_pair` + `pair_id` 回链。用途：复查 gate0 定位 0.643 是否主要由 `none` 变体的全图 VAE 地板造成。
6. **下游钩子**：Tab-Loc / Tab-Detect 增加按 `compositing` 切片的行；预期方向写进分析笔记（paste 系 → 边界信号强、定位偏易；none 系 → 全图 VAE 地板、检测偏易而精细定位偏难），两种都必须如实报告。

### 7.4 validator 扩展（`validate-manifest` 新增断言）

- **V1** 存储格式与分辨率策略在同一可比组内唯一（Test-E 退化行凭 `postprocess` 字段豁免）。
- **V2** 每个 split 内，`io_chain` 去掉 `edit:*` 与 `vae_rt:*` 节点后的**非生成段**，在 real 与 fake 之间分布一致——即压缩/处理历史不可由 `is_fake` 预测。实现为：对每个 split，`set(nongen_chain | is_fake=0) == set(nongen_chain | is_fake=1)`，不一致则 FAIL 并打印差集。
- **V3** masked 算子行必有 `compositing`；`paste_feather` 行必有 `feather_px`。
- **V4** `train/test_a/test_f` 含 `real_vae_rt` 且占比落在配置区间。
- **V5** 向后兼容：旧 manifest 经回填脚本后必须通过 V1–V4（`io_chain=legacy` 行豁免 V2，但 `stats` 单列计数，主 run 中不得出现 legacy）。

### 7.5 验收标准

- [ ] 冒烟 run（每类 ~20 图：plain_real / real_vae_rt / img2img / inpaint×{none,paste,paste_feather}）通过 V1–V5；`stats` 能按 `sample_kind` / `compositing` / `io_chain` 出计数。
- [ ] 成对 probe 抽样目检：`paste` 变体在羽化带外的未编辑区与 orig **逐像素相等**；`none` 变体全图与 orig 不等（VAE 往返所致）。写成脚本断言而非人工看图。
- [ ] `real_vae_rt` 行的残差探针分数分布图输出一张（仅作记录，不设通过阈值——它是科学结果，不是工程验收）。
- [ ] 执行顺序确认：本 PATCH 合并 → 冒烟通过 → 才允许启动 B3 主生成。

### 7.6 不改什么

- PATCH 5 的退化机制、`postprocess`/`postprocess_of` 字段与回链逻辑零改动。
- 不引入新的有损压缩步——压缩混淆用"全库 PNG + 共享非生成链 + vae_rt 负样本"解决，**不是**"两边都压 JPEG"（后者会污染 Test-E 退化轴的独立性）。
- 闸门脚本不动（gate0 定位复查用成对 probe 跑现有脚本即可）。

---

## PATCH 8 — 数据矩阵升级（c 轴补全、非扩散边界、Test-C 判定、holdout 现代化）

### 8.0 目的与执行时机

四项都是 B1 生成器×算子矩阵的**范围决策**，必须在 B3 主生成前定案（实现可分批）：

| 项 | 内容 | 服务的主张 | 时机 |
|---|---|---|---|
| 8.1 | LaMa 非扩散对照：可选→**必做** | 界定 C1/C2 适用边界 | 随 B 主生成（test-only，可稍后补） |
| 8.2 | InstructPix2Pix 指令编辑 | 补 (t0, c, M) 中 **c 轴**唯一缺失的变化源 | 随 B 主生成（训练可见） |
| 8.3 | Test-C holdout 算子判定 | 防掩码几何平凡可分使 Test-C 失去证明力 | **立即**，B3 前出结论 |
| 8.4 | Flux 落实进 Test-B holdout | 2026 审稿口径下 Kandinsky+SDXL 偏旧 | 独立环境，可后补 |

### 8.1 非扩散对照：LaMa 必做，MAT 次选（test-only）

- **后端**：新增 `backends.real.nondiffusion_gen`。LaMa 经 simple-lama-inpainting 或 IOPaint 推理（big-lama 权重）；MAT 用官方仓库，有余力再加。
- **成对设计**：生成专用成对 probe——同一 (base, mask, seed) 让 LaMa 与 SD1.5-inpaint 各出一行，`probe_group=nd_pair` + `pair_id` 回链。**两侧都是新生成的 test-only 行**，不复用 train 行（避免跨 split 纠缠）。成对是为了对比"score 场对非扩散编辑亮不亮"时排除内容/掩码混淆。
- **split 与规模**：test-only、**不入 train**（要回答的是 score 场的零样本行为，不是可学性）；`generator_family=non_diffusion`，归入 test_b（论文表内单独成行），每算子 1–2k + 成对 probe ~100 组。
- **compositing**：如实记录所用包装器行为（IOPaint 默认回贴 → `paste`；原生全图输出 → `none`），关键是记录一致，不强行统一。
- **论文钩子**：成对残差热图进补充材料。亮 → "适用面比扩散编辑更广"的经验证据；不亮 → 干净的边界声明（方法限定扩散编辑，与通用 IML 互补而非竞争）。两种结果都可写，写法随结果定，不预设立场。

### 8.2 InstructPix2Pix：c 轴（训练可见）

- **pipeline**：`StableDiffusionInstructPix2PixPipeline`（timbrooks/instruct-pix2pix，SD1.5 系，24GB 无压力）；算子枚举新增 `instruct_edit`。
- **定位口径**：全图算子，随 img2img 惯例排除出像素定位指标（mask=全图）。
- **t0 语义（重要）**：IP2P 恒从纯噪声起步（t0≡T），无 strength 轴 → 借此把 H3 的**条件式参数化**从建议固化为约束：先预测算子类，t0 头只对 strength 可变算子有定义，t0 指标只在该子集上统计。写进 Phase C H3 设计注记。
- **参数与字段**：`image_guidance_scale` × `guidance_scale` 网格采样，记入新字段 **`op_params`**（JSON string，通用算子参数容器，后续算子专属旋钮都进这里）；instruction 用 ~50 条模板库（对象替换/天气/风格/颜色等）按底图内容填充，同记入 op_params。
- **probe**：`instruct_edit` 纳入 B3 受控算子×族网格，使 gate2 的 n≥200 复测覆盖它。
- **论文钩子**：同族（SD1.5）不同条件机制——归因若能分开 img2img vs instruct_edit，即"操作>模型指纹"的**族内**证据（补强 C5）；Fig-Confusion 增加该类。

### 8.3 Test-C holdout 算子判定（零生成成本，B3 前出结论）

- **风险**：outpaint 的边缘带掩码几何平凡可分——若 holdout 算子仅凭掩码几何即可识别，Test-C 的成功不构成 score 签名泛化的证据。
- **新脚本** `checking/testc_geometry_probe.py`：仅用 GT 掩码的几何特征（面积比、边界接触率、连通域数、凸性、质心偏移）→ logistic 回归做算子分类，对每个候选算子报 one-vs-rest AUC/BA（用既有掩码即可，零生成）。
- **决策规则**：候选 geometry-only AUC ≥ 0.90 → 判"几何平凡"，**不得**作 Test-C holdout。默认取 `object_replacement`（对象形掩码，与常规 inpaint 同构，预期不平凡），以脚本数字确认后写回 experiment plan §3 B3。
- **备选方案**（写入 §10 决策树，默认不选）：把 8.2 的 `instruct_edit` 留作 Test-C holdout——换来"未见条件机制"的更强命题，代价是 train 失去 c 轴多样性、g_ψ 无该类（此时 Test-C 评检测/定位泛化 + 归因的近邻退化行为，预期混淆到 img2img——本身是可写的发现）。留作 G-A 之后的可选强化，与默认方案二选一，不并行。

### 8.4 Flux 进 Test-B holdout（现代化）

- **必做**：Flux.1-schnell img2img（Apache-2.0，4 步蒸馏，24GB 可跑）+ Flux.1-Fill-dev inpaint（唯一官方 Fill 管线）。**许可注意**：dev 系为非商业许可——数据集若公开发布，dataset card 须注明许可与用途限制；schnell 侧无此负担。
- **PixArt-Σ**：无官方 img2img，需自写 SDEdit 包装（encode → 按 σ 加噪 → PixArt 去噪）；列**可选**，有余力再做。
- **环境冲突（重要）**：附录已注明现镜像锁 `diffusers==0.30 transformers==4.44`（torch 2.3），而 Flux Fill 需 diffusers≥0.32 + 更高 torch → Flux 生成走**独立 AutoDL 作业 + 独立 lockfile**，产出 manifest 事后合并；合并后必须**复跑 validate-manifest 全部断言**（跨作业合并是 V2 非生成链一致性最容易破的地方）。
- **规模与预算**：holdout 量每算子 0.5–1k；Fill-dev 需 offload/fp8，约 30–60s/图 → 预算追加 **+6–15 GPU·h**，同步更新 §8 预算表。
- **字段**：io_chain/compositing 遵循 PATCH 7 约定（`edit:flux_fill` / `edit:flux_img2img`）。

### 8.5 validator 增量（叠加在 PATCH 7 的 V1–V5 之上）

- **V6** `operator=instruct_edit` 行必含合法 JSON 的 `op_params` 且含 `image_guidance_scale` 键。
- **V7** 凡 `probe_group` 为成对组（`compositing_pair` / `nd_pair`）的行：同一 `pair_id` 恰好出现两次，组内 base/mask/seed 一致，仅目标维度不同（compositing_pair → 仅 compositing 不同；nd_pair → 仅 generator_name/family 不同）。

### 8.6 验收标准

- [ ] geometry probe 报告落盘，Test-C holdout 最终选择（含各候选 AUC 数字）写回 experiment plan §3 B3。
- [ ] LaMa 成对冒烟（~20 组）通过 V1–V7；脚本断言编辑区确被重建、pair 回链完整。
- [ ] instruct 冒烟行通过 V6；定位统计脚本确认 instruct_edit/img2img 已排除出 pixel-F1。
- [ ] Flux 独立环境 smoke：两算子各 5 图跑通，manifest 合并后 V1–V7 全过。
- [ ] experiment plan §8 预算表与 §3 B1 生成器矩阵表已更新。

### 8.7 不改什么

- scorer 不动（仍单 SD1.5）：LaMa/IP2P/Flux 全部只在生成侧。
- test_b 既有 Kandinsky/SDXL 行不动——Flux/LaMa 是**增行**不是替换（Kandinsky 的 unCLIP 异族价值仍在）。
- g_ψ 输出空间只固化"条件式参数化 + t0 子集统计"这一条约束，不引入其他头部改动。

---

## 文档同步清单（三个 PATCH 落地后）

1. `PATCHES.md`：追加 PATCH 6 / 7 / 8 条目（含日期、涉及文件、验收结果）。
2. `paper_experiment_plan_2026-07-15.md`：
   - §2 A1 闸门 1 判据行 → 替换为 PATCH 6 预注册判据（标注"待 n≥200 验证性复测"）；
   - §3 B1 生成器矩阵 → LaMa 由"可选"改"必做（test-only）"；新增 `instruct_edit` 行（训练可见，SD1.5 系）；holdout 行落实 Flux（schnell-img2img + Fill-dev-inpaint），PixArt 标注"可选（需自写 SDEdit 包装）"；
   - §3 B2/B3 → 加入 canonical I/O、`sample_kind` 配比、compositing 50/50 与成对 probe 条目；Test-C holdout 最终选择（附 geometry probe AUC 与报告路径）；
   - §8 预算表 → Phase B 追加 Flux 独立作业 +6–15 GPU·h；
   - §9 产物清单 → Tab-Loc/Tab-Detect 增加 compositing 切片行；Test-F 表按 sample_kind 分列 FPR；Fig-Confusion 增加 instruct_edit 类；补充材料增加 LaMa vs SD-inpaint 成对残差热图；
   - §10 决策树 → 增补 8.3 备选方案（instruct_edit 作 Test-C holdout）为 G-A 后可选强化分支。
3. `GATE_DATA.md`：字段对照表加入 `io_chain / sample_kind / compositing / feather_px / probe_group / pair_id / op_params`。
4. （若 gate1 复判确认"单调+饱和"）在 experiment plan §10 决策树 G-A 分支处补注：t0 主指标已切换为回归口径，粗桶方案为已验证的降级路径。
5. `scripts/run_cross_generator_p0.sh` 的依赖锁修复（附录已知问题）与 PATCH 8.4 的 Flux 独立 lockfile **分开处理**——前者是既有 SD/Kandinsky 作业的修复，后者是新环境，不要合并成一个 requirements。

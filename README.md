# forgery-pipeline · 伪造检测数据集生成 Pipeline

基于 **GIM、MIML、TruFor、HiFi-Net、FakeShield、Community Forensics** 六篇论文提炼出的数据集构建方法，构建一个面向 **图像伪造检测与定位** 的统一数据基础设施。

数据被组织为 **5 个子库 D0–D4**，产出带 **HiFi-Net 层级标签** 的统一 JSONL `manifest`，并完成 **质量控制（QC）→ 后处理增强 → 防泄漏 8-way 划分**。所有重型 ML 阶段（整图生成 / 局部重绘 / SAM 分割 / MLLM 解释）走 **可插拔 backend**；自带 **mock backend**，让整条流水线在 CPU 上 **开箱即跑、可复现、并通过 pytest**。

> 设计文档见 [`docs/superpowers/specs/2026-06-25-forgery-pipeline-design.md`](docs/superpowers/specs/2026-06-25-forgery-pipeline-design.md)，实施计划见 [`docs/superpowers/plans/2026-06-25-forgery-pipeline.md`](docs/superpowers/plans/2026-06-25-forgery-pipeline.md)，来源报告见 [`docs/forgery_detection_data_pipeline_report.pdf`](docs/forgery_detection_data_pipeline_report.pdf)。

---

## 五个子库

| 子库 | 内容 | 对应论文启发 |
|---|---|---|
| `D0_real_pristine` | 真实图像池：真实负样本 + 局部篡改底图 | TruFor（真实相机痕迹） |
| `D1_whole_generated` | 整图 AIGC 生成（强调**生成器多样性**） | Community Forensics |
| `D2_local_aigc_edit` | 局部 AIGC 篡改（像素级 mask，7 类篡改） | GIM |
| `D3_web_human_forgery` | 网页人工篡改伪标注（差分 → 伪 mask → QES） | MIML |
| `D4_explainable_subset` | 可解释取证子集（image-mask-description 三元组） | FakeShield |
| 统一层级标签 | Level0–4 真假 / 整图-局部 / 方法 / 具体模型 | HiFi-Net |

## 总体流程

```
真实图像池 D0
  → 整图生成 D1 / 局部篡改 D2 / 网页人工篡改 D3
  → 可解释子集 D4（取自 D2/D3）
  → 统一层级标签 + manifest
  → QC（图像/mask/生成质量 + 泄漏检查）
  → 后处理增强（JPEG/resize/blur/noise/社媒压缩）
  → 8-way 划分（train/val/test_a..f）
  → 统计与产出（manifest.jsonl + stats.json）
```

---

## 安装

需要 Python ≥ 3.10。

```bash
pip install -e .            # 核心依赖（mock 全流程即可运行）
# 可选：接入真实模型时按需安装
pip install -e ".[real]"    # diffusers / torch / transformers
pip install -e ".[sam]"     # segment-anything
pip install -e ".[mllm]"    # openai / anthropic
```

> 若系统 Python 为外部托管（PEP 668），可加 `--user --break-system-packages`，或先建虚拟环境。

## 快速开始

```bash
# 用 mock 后端端到端跑通，产物写到 data/run/
forgery-pipeline run --config configs/pipeline.example.yaml

# 查看统计 / 校验 manifest
forgery-pipeline stats --path data/run/manifest.jsonl
forgery-pipeline validate-manifest --path data/run/manifest.jsonl

# 或一键示例脚本
bash examples/run_mock_pipeline.sh
```

产物：
- `data/run/d0.jsonl … d4.jsonl`：各子库 manifest
- `data/run/manifest.jsonl`：合并后的统一 manifest（每行一个样本）
- `data/run/stats.json`：规模与划分统计
- `data/run/D0_real_pristine/ … D4_*`：图像与 mask 文件

### 可视化检视生成的数据集

```bash
forgery-pipeline viewer --run data/run        # 生成 data/run/viewer.html
# 或生成后自动用浏览器打开：
forgery-pipeline viewer --run data/run --open
```

浏览器打开 `data/run/viewer.html`：缩略图画廊 + 按 `split / 篡改类型 / 生成器 / 真假` 筛选 + image_id 搜索；点击样本看 **原图 | 伪造图 | mask 叠加** 三联 + 层级标签 + 解释文本。纯静态页面（无服务器、零新依赖）。

## 目录结构

```
src/forgery_pipeline/
  schema.py        Sample/Postprocess/Explanation（manifest 数据契约）
  labels.py        HiFi-Net Level0-4 层级标签 + 一致性校验
  ids.py manifest.py config.py dedup.py image_io.py
  backends/        base(抽象接口) · mock(确定性合成) · registry · real/(适配器骨架)
  masks/           morphology · candidates(尺度分桶) · pseudo_mask(差分伪标注)
  qc/              image_qc · mask_qc · gen_qc · quality_score(QES)
  postprocess/     degradations(退化增强)
  builders/        d0_real · d1_whole · d2_local · d3_web · d4_explain
  split/           grouping · leakage(5 条规则) · splitter(8-way)
  pipeline.py cli.py
configs/           pipeline.example.yaml · generators.yaml · split.yaml
tests/             各模块单测 + 端到端 smoke
```

## 层级标签与 manifest

- **Level 0**: real / fake
- **Level 1**: whole_generated / partial_manipulated
- **Level 2**: diffusion / GAN / autoregressive / Photoshop-editing / DeepFake / AIGC-editing / copy-move / splicing / removal
- **Level 3**: conditional/unconditional generation、text/image-guided editing、mask_guided_inpainting、object_replacement/removal、face_swap、text_editing
- **Level 4**: 具体生成器/方法（如 stable-diffusion-inpaint、midjourney-v6）

`manipulation_type_loss + generator_family_loss + detection_loss + localization_loss + optional_explanation_loss` 多任务 loss 字段在 `labels.LOSS_TERMS` 中文档化（训练阶段使用，本仓库不含训练）。

manifest 每行一个 `Sample`（pydantic 校验），字段含 image/mask 路径、真假、task_type、四级标签、生成器与 prompt/seed、mask_area_ratio、postprocess 参数、quality_score、split、explanation 等。

## 8-way 划分与泄漏检查

| Split | 目标 |
|---|---|
| train / val | 训练 / 调参 |
| test_a | In-domain 常规性能 |
| test_b | Cross-generator（未见生成器） |
| test_c | Cross-manipulation（未见篡改类型） |
| test_d | Cross-domain（未见来源域） |
| test_e | Degradation（退化鲁棒性） |
| test_f | Real-only（误报率） |

划分**按原图分组**（同一真实底图及其衍生样本不跨 split），并强制 5 条泄漏检查（原图/压缩版本/prompt+seed 不跨 split、cross-generator 生成器不入 train、公开 benchmark 不入 train）。划分后若检出泄漏，pipeline 直接报错中止。

## 接入真实后端

1. 安装对应 extra（见上）并准备模型权重 / API key。
2. 在 `src/forgery_pipeline/backends/real/` 下按骨架实现适配器（`diffusers_gen.py` / `sam_segmenter.py` / `mllm_explainer.py`），并在 `backends/registry.py` 中接线。
3. 把 `configs/pipeline.example.yaml` 的 `backend` 从 `mock` 改为 `real:diffusers` 等。
4. D0 可改为从本地真实数据集目录读取（自定义 `ImageSource`）。

接口契约（`backends/base.py`）：`ImageSource / WholeImageGenerator / Inpainter / Segmenter / Explainer`，图像统一为 `(H,W,3) uint8 RGB`，掩码为 `(H,W) uint8 {0,255}`。

## 测试

```bash
pytest -q
```

## 局限与后续

- 当前为**框架 + mock 后端**：真实百万级数据需接入真实生成/分割/MLLM 后端与真实图像源。
- 后处理变体当前为**就地退化**；如需鲁棒性评估的多版本落库，可扩展为每个退化生成独立 manifest 行。
- D3 的人工复核池（QES 0.60–0.75）与真实网页 pair 采集尚为占位，可按 MIML 进一步完善。

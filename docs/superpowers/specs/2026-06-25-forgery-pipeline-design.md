# 伪造检测数据集生成 Pipeline —— 设计文档（Spec）

- 日期：2026-06-25
- 来源需求：`docs/forgery_detection_data_pipeline_report.pdf`（基于 GIM、MIML、TruFor、HiFi-Net、FakeShield、Community Forensics 六篇论文提炼的数据集构建方法）
- 目标仓库：`git@github.com:luotieh/forgery-pipeline.git`

---

## 1. 目标与范围

### 1.1 目标
构建一个可运行的 **伪造检测数据集生成 pipeline**，把数据拆成 5 个子库（D0–D4），产出统一的 JSONL `manifest`（含 HiFi-Net 层级标签），并完成质量控制（QC）、后处理增强（degradation）与防泄漏的 8-way 数据划分。

支持的下游任务：图像级真假检测、像素级篡改定位、伪造类型分类、跨生成器/跨域/退化鲁棒性评估、可解释取证。

### 1.2 本次范围（In scope）
- 完整的**流水线编排骨架**：manifest schema、层级标签体系、QC、后处理、防泄漏划分、伪 mask 生成，全部在 CPU 上**可运行、可测试**。
- **可插拔 backend**：整图生成 / 局部重绘 / 分割（SAM）/ MLLM 解释 / 真实图像源。自带 **mock backend**（确定性合成数据），让整条流水线开箱即跑、端到端通过 pytest。
- 真实模型适配器（`diffusers` / `segment-anything` / MLLM API）以**接口 + stub + 参考骨架**形式提供，可在具备 GPU/权重/API key 时启用。

### 1.3 非目标（Out of scope）
- 不在本环境真实下载 COCO/ImageNet 等大型数据集，不真实运行 Stable Diffusion / SAM / MLLM（无 GPU、无权重、无 key）。这些通过 backend 接口对接。
- 不实现模型训练。多任务 loss 字段清单仅作为文档常量保留，供后续训练使用。

### 1.4 关键决策（已与用户确认）
- 配置格式：**YAML**。
- CLI：**argparse**（不引入额外依赖）。
- manifest 存储：先按子库写 `d0.jsonl … d4.jsonl`，QC + 划分后合并为最终 `manifest.jsonl`（溯源清晰，契合"5 子库"框架）。
- 文档/注释/README：**中文**；代码标识符：English。

---

## 2. 总体架构

```
真实图像池 D0
  → 整图生成 D1 / 局部篡改 D2 / 网页人工篡改 D3
  → 可解释子集 D4（取自 D2/D3）
  → 统一层级标签 + manifest
  → QC（图像/mask/生成质量 + 泄漏检查）
  → 后处理增强（degradation）
  → 8-way 划分（Train/Val/Test-A..F）
  → 统计与产出
```

每个阶段是一个独立、可单测的单元，通过 manifest（数据契约）与 backend（模型契约）两类接口通信，可单独开关与替换。

---

## 3. 目录结构

```
forgery-pipeline/
├── README.md                      中文说明 + 快速开始
├── pyproject.toml                 包元数据、依赖、可选 extras
├── .gitignore                     忽略 data/ 产物、venv、缓存
├── configs/
│   ├── pipeline.example.yaml      主配置：规模/阶段开关/路径/后端选择
│   ├── generators.yaml            生成器清单（family/name/类型）
│   └── split.yaml                 8 个 split 的定义与规则
├── src/forgery_pipeline/
│   ├── __init__.py
│   ├── schema.py                  Manifest 字段（pydantic）：Sample / Postprocess / Explanation
│   ├── labels.py                  Level 0–4 层级标签 + 一致性校验 + loss 字段常量
│   ├── manifest.py                JSONL 读写 / 累加 / 合并 / 统计
│   ├── config.py                  配置加载与校验（dataclass/pydantic）
│   ├── ids.py                     稳定 image_id 生成（内容哈希 + 前缀）
│   ├── dedup.py                   pHash 去重（必装）+ CLIP embedding 去重（可选）
│   ├── logging_util.py            统一日志
│   ├── backends/
│   │   ├── base.py                抽象接口：ImageSource/WholeImageGenerator/Inpainter/Segmenter/Explainer
│   │   ├── registry.py            按名字解析 backend（配置驱动）
│   │   ├── mock.py                合成实现（CPU，确定性，可运行）
│   │   └── real/
│   │       ├── diffusers_gen.py   Stable Diffusion / SDXL（stub + guarded import）
│   │       ├── sam_segmenter.py   SAM / Grounded-SAM（stub）
│   │       └── mllm_explainer.py  OpenAI/Anthropic MLLM（stub）
│   ├── builders/
│   │   ├── d0_real.py             真实图像池：摄取 + 清洗 + 去重
│   │   ├── d1_whole.py            整图 AIGC 生成（跨生成器）
│   │   ├── d2_local.py            局部 AIGC 篡改（mask→prompt→inpaint）
│   │   ├── d3_web.py              网页人工篡改伪标注（差分→伪mask→QES）
│   │   └── d4_explain.py          可解释子集（image+mask→MLLM 解释三元组）
│   ├── qc/
│   │   ├── image_qc.py            解码/分辨率/边框/重复/license
│   │   ├── mask_qc.py             面积比/碎片化/边界一致性
│   │   ├── gen_qc.py              生成失败/prompt 不一致/畸变
│   │   └── quality_score.py       QES-like 评分（D3）
│   ├── masks/
│   │   ├── candidates.py          候选 mask 采样 + 尺度分桶
│   │   ├── morphology.py          dilation/erosion/boundary blur/irregular
│   │   └── pseudo_mask.py         D3 差分(RGB/SSIM/LPIPS)→coarse→细化→连通域
│   ├── postprocess/
│   │   └── degradations.py        jpeg/resize/blur/noise/social/recapture/strip-exif
│   ├── split/
│   │   ├── grouping.py            按原图/生成器/篡改类型/后处理分组
│   │   ├── leakage.py             5 条泄漏检查
│   │   └── splitter.py            8-way split（Train/Val/Test-A..F）
│   └── pipeline.py                阶段编排（顶层 run）
├── cli.py                         命令行入口（argparse）
├── tests/                         pytest：各模块单测 + 端到端 smoke
└── examples/
    └── run_mock_pipeline.sh       用 mock 后端跑通全流程示例
```

---

## 4. 数据模型：Manifest Schema（`schema.py`）

用 pydantic 实现。每行一个样本，写入前做 schema + 标签一致性校验。

### 4.1 完整字段（对应报告 §13.1）

| 字段 | 类型 | 说明 |
|---|---|---|
| `image_id` | str | 全局唯一，内容哈希派生 |
| `image_path` | str | 样本图像路径 |
| `real_image_path` | str? | 对应真实底图（D2/D3 有，D0/D1 无） |
| `mask_path` | str? | 篡改 mask（定位任务必有） |
| `is_fake` | int(0/1) | 真假标签 |
| `task_type` | enum | `whole_image_detection` / `localization` / `real_pristine` / `explainable` |
| `manipulation_level1` | enum? | `whole_generated` / `partial_manipulated` |
| `manipulation_level2` | enum? | diffusion/GAN/autoregressive/Photoshop-editing/DeepFake/AIGC-editing/copy-move/splicing/removal |
| `manipulation_level3` | enum? | conditional/unconditional/text_guided/image_guided/mask_guided_inpainting/object_replacement/object_removal/face_swap/text_editing |
| `manipulation_level4` | str? | 具体生成器/方法（如 stable-diffusion-inpaint） |
| `source_dataset` | str? | 来源数据集 |
| `generator_name` | str? | 生成器名 |
| `generator_family` | str? | 生成器族（diffusion/GAN/...） |
| `prompt` | str? | 生成/编辑 prompt |
| `negative_prompt` | str? | 负向 prompt |
| `seed` | int? | 随机种子 |
| `sampler` | str? | 采样器 |
| `steps` | int? | 步数 |
| `cfg_scale` | float? | CFG |
| `mask_source` | enum? | SAM/Grounded-SAM/manual/synthetic/diff |
| `mask_area_ratio` | float? | mask 面积比 [0,1] |
| `postprocess` | Postprocess | jpeg_quality/resize/blur/noise（无则 "none"） |
| `quality_score` | float? | QC/QES 评分 [0,1] |
| `quality_bucket` | enum? | high/mid/low |
| `split` | enum? | train/val/test_a..test_f（划分后填） |
| `license` | str? | 许可信息 |
| `explanation` | Explanation? | 四段文本（D4） |

子模型：
- `Postprocess { jpeg_quality:int|"none", resize:str, blur:str, noise:str }`
- `Explanation { location_description, visual_artifact_description, semantic_reasoning, forensic_conclusion }`

### 4.2 MVP 必需字段（对应报告 §13.2，强制校验）
`image_id, image_path, real_image_path, mask_path, is_fake, task_type, manipulation_level1, manipulation_level2, generator_name, source_dataset, prompt, postprocess, quality_score, split`（其中 real_image_path/mask_path 对 D0/D1 可为 null）。

---

## 5. 层级标签体系（`labels.py`，借鉴 HiFi-Net §9）

- **Level 0**: real / fake
- **Level 1**: whole_generated / partial_manipulated
- **Level 2**: diffusion / GAN / autoregressive / Photoshop-editing / DeepFake / AIGC-editing / copy-move / splicing / removal
- **Level 3**: conditional_generation / unconditional_generation / text_guided_editing / image_guided_editing / mask_guided_inpainting / object_replacement / object_removal / face_swap / text_editing
- **Level 4**: 具体生成器或方法（自由字符串，给定枚举建议值）

**一致性校验规则**：
- `is_fake==0` ⇒ level1..4 均为 null，task_type=`real_pristine`。
- `is_fake==1 & level1==whole_generated` ⇒ 不要求 mask，task_type 通常 `whole_image_detection`。
- `is_fake==1 & level1==partial_manipulated` ⇒ **必须有 mask_path**，task_type=`localization`。
- level2/3/4 与 level1 的组合受白名单约束（例如 whole_generated 不应配 copy-move）。

**多任务 loss 字段常量**（仅文档化，不实现训练）：
`loss = detection_loss + localization_loss + manipulation_type_loss + generator_family_loss + optional_explanation_loss`。

---

## 6. 可插拔 Backend（`backends/`）

抽象接口（`base.py`），均为最小契约，返回值附带可写入 manifest 的元数据：

- `ImageSource.iter_images(n) -> Iterable[(image, meta)]`：真实图像源。mock=确定性合成照片或读本地目录。
- `WholeImageGenerator.generate(prompt, params) -> (image, gen_meta)`：整图生成。mock=按 prompt+seed 确定性合成图。
- `Inpainter.inpaint(image, mask, prompt, params) -> (image, gen_meta)`：局部重绘。mock=在 mask 区域贴合成补丁。
- `Segmenter.propose_masks(image, k) -> list[mask]`：候选 mask。mock=几何/随机区域。
- `Explainer.explain(image, mask, context) -> Explanation`：MLLM 解释。mock=模板填充。

`registry.py` 按配置里的名字解析具体 backend（`mock` / `real:diffusers` 等）。real 适配器默认 stub：调用即抛清晰错误「请 `pip install .[real]` 并提供权重/key」，并保留 guarded-import 的参考实现骨架。

---

## 7. 五个 Builder（`builders/`）

### 7.1 D0 真实图像池（§4）
摄取（mock 合成 or 读本地目录）→ 清洗：删解码失败、短边<256、极端长宽比；pHash/CLIP 去重；标记水印/截图/边框；尽量保留 EXIF/相机型号/分辨率/license → 写 `d0.jsonl`（is_fake=0, task_type=real_pristine）。

### 7.2 D1 整图 AIGC 生成（§5）
**强调生成器多样性**（Community Forensics）：从 `generators.yaml` 遍历多生成器，每个生成少量图像。prompt 源：COCO/LAION caption、业务 prompt、LLM 扩写、人工模板（mock 用内置模板库）。gen_qc 后写 `d1.jsonl`（is_fake=1, level1=whole_generated, level2=generator_family）。

### 7.3 D2 局部 AIGC 篡改（§6，借鉴 GIM）
底图取自 D0 → Segmenter 候选 mask → mask 过滤与采样（面积<1% 或 >50% 删；保留 小1–5%/中5–20%/大20–50%；形态学扰动 dilation/erosion/boundary blur/irregular；记录 mask_source）→ LLM/模板生成编辑 prompt → Inpainter 局部重绘 → mask_qc → 写 `d2.jsonl`。
覆盖 **7 类篡改**：object insertion / replacement / removal / attribute editing / background editing / text editing / face editing。

### 7.4 D3 网页人工篡改伪标注（§7，借鉴 MIML）
real-fake pair（mock 用合成 pair）→ 配准/尺寸对齐 → 差分（RGB + SSIM，可选 LPIPS）→ coarse mask → 语义过滤 + SAM 边界细化 + 连通域去噪 → 伪 mask →
**QES-like 评分**：`0.3*confidence + 0.2*boundary_sharpness + 0.2*mask_consistency + 0.2*semantic_consistency + 0.1*area_validity`；≥0.75 入训练、0.60–0.75 待人工复核、<0.60 删除 → 写 `d3.jsonl`。

### 7.5 D4 可解释取证子集（§8，借鉴 FakeShield）
从 D2/D3 取 image+mask → Explainer(MLLM) 生成四段解释（位置/伪迹/语义推理/取证结论）→ 规则过滤 → 写 `d4.jsonl`（task_type=explainable，带 explanation 字段）。

---

## 8. QC 质量控制（`qc/`，§11）
- **图像质量**：可解码；短边≥256；无大面积黑/白/纯色边；不重复；license 合法。
- **mask 质量**：`0.01≤area_ratio≤0.50`；不过度碎片化；边界与变化区域基本一致；不覆盖整图（整图生成除外）。
- **生成质量**：删明显失败图；删 prompt-图像严重不一致；删大面积畸变；保留少量低质 AIGC 但单独标 `quality_bucket`。
- **QES 评分**：见 §7.4，用于 D3。

---

## 9. 后处理增强（`postprocess/degradations.py`，§10）
类型：JPEG(q=50/60/70/80/90/95)、resize(0.5/0.67/0.75/1.5)、Gaussian blur(k=3/5)、Gaussian noise(σ=3/5/10)、社媒压缩（近似）、截屏重采样、抹 EXIF。
**保存策略**：保留原始 fake 图 + 后处理参数；变体命名 `image_jpeg_q70` / `image_resize_0.5` 等；参数写入 manifest `postprocess` 字段，便于鲁棒性评估。退化操作需**确定性**（固定种子）以便测试。

---

## 10. 数据划分（`split/`，§12）

| Split | 目标 | 构建方法 |
|---|---|---|
| Train | 常规训练 | seen images / generators / manipulation types |
| Val | 调参 | 同分布、图像不重复 |
| Test-A | In-domain | seen generator family，unseen images |
| Test-B | Cross-generator | 完全未见生成器 |
| Test-C | Cross-manipulation | train 不含某类篡改，test 专测该类 |
| Test-D | Cross-domain | 新闻/电商/社媒/证件等新场景 |
| Test-E | Degradation | JPEG/resize/blur/noise/社媒压缩 |
| Test-F | Real-only | 全真实图，测误报率 |

**5 条泄漏检查**（`leakage.py`，划分后强制断言）：
1. 原图不同时出现在 train 和 test；
2. 同一生成图的不同压缩版本不跨 split；
3. 同一 prompt+seed 结果不跨 split；
4. cross-generator test 的生成器不出现在 train；
5. 公开 benchmark 图像不混入训练集。

分组策略：按 原图ID / 生成器 / 篡改类型 / 后处理类型 分组后再分配。

---

## 11. 编排与 CLI

`pipeline.py` 按依赖顺序执行：`D0 → {D1, D2, D3} → D4 → QC → postprocess → split → stats`，每阶段可在配置开关。

CLI（`cli.py`，argparse）：
- `forgery-pipeline run --config configs/pipeline.example.yaml`（全流程）
- 子命令：`build-d0 / build-d1 / build-d2 / build-d3 / build-d4 / qc / postprocess / split / stats / validate-manifest`
- 默认 **mock backend**，开箱即跑。

---

## 12. 测试策略（`tests/`）
pytest：
- 单元：schema 校验、标签一致性、ids 稳定性、dedup、mask 形态学/面积分桶、QES 评分、退化确定性、泄漏检测、splitter 分配。
- **端到端 smoke**：用极小合成集跑通全流程，断言：manifest 全部行通过 schema、各 split 非空、`leakage.py` 零泄漏、D2 行均有 mask。

验收标准：`pytest` 全绿；`examples/run_mock_pipeline.sh` 能产出合法 manifest 与统计。

---

## 13. 依赖（`pyproject.toml`）
- 核心（必跑）：`pydantic, numpy, Pillow, opencv-python-headless, scikit-image, imagehash, pyyaml, tqdm`。
- 可选 extras：`[real]`(torch/diffusers/transformers)、`[sam]`(segment-anything)、`[clip]`(open-clip-torch)、`[mllm]`(openai/anthropic)。
- LPIPS 缺 torch 时回退到 SSIM+RGB。
- Python ≥ 3.10。

---

## 14. 实施优先级（对应报告 §14.2）
P0 D0 → P1 D1 → P2 D2 → P3 标签+manifest → P4 cross-generator 测试 → P5 D3 → P6 D4。
（共享基础设施 schema/labels/manifest/qc/split 先行，使各 builder 可独立挂接。）

---

## 15. 需避免的问题（报告 §14.3，作为设计约束）
1. 只用少数生成器 → 记住指纹：D1 强制多生成器配置。
2. 只做真假标签：强制层级标签。
3. 随机划分泄漏：强制分组 + 泄漏断言。
4. 忽略真实传播退化：内置后处理增强 + Test-E。
5. 不记录 prompt/seed/generator/后处理：manifest 强制溯源字段。

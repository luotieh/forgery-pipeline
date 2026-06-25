# Forgery Detection Dataset Pipeline 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一个可运行的伪造检测数据集生成 pipeline，把数据组织为 D0–D4 五个子库，产出带 HiFi-Net 层级标签的统一 JSONL manifest，并完成 QC、后处理增强与防泄漏 8-way 划分；所有重型 ML 阶段走可插拔 backend，自带 mock backend 使全流程在 CPU 上端到端跑通并通过 pytest。

**Architecture:** 分层单元，通过两类契约通信——manifest（数据契约，pydantic `Sample`）与 backend（模型契约，ABC）。基础层（schema/labels/ids/manifest/config）→ backend 层（base/mock/registry）→ 工具层（masks/qc/dedup/postprocess）→ builder 层（D0–D4）→ 编排层（split/pipeline/cli）。默认 mock backend，开箱即跑。

**Tech Stack:** Python ≥3.10、pydantic v2、numpy、Pillow、opencv-python-headless、scikit-image、imagehash、pyyaml、tqdm、pytest。

## Global Constraints

> 每个任务都隐含遵守本节。

- Python ≥ 3.10。源码标识符用 English；注释/文档字符串/README 用中文。
- 核心依赖仅限：`pydantic>=2`, `numpy`, `Pillow`, `opencv-python-headless`, `scikit-image`, `imagehash`, `pyyaml`, `tqdm`, `pytest`（测试）。重型依赖（torch/diffusers/segment-anything/openai/anthropic）只能出现在 `[real]/[sam]/[clip]/[mllm]` 可选 extras 与 `backends/real/` 的 guarded import 里。
- 图像统一表示：`np.ndarray`，shape `(H, W, 3)`，dtype `uint8`，**RGB** 通道序。掩码统一表示：`np.ndarray`，shape `(H, W)`，dtype `uint8`，取值 `{0, 255}`。
- **确定性**：所有随机操作必须由显式 `seed` 驱动（`np.random.default_rng(seed)`），禁止使用 wall-clock 随机；相同输入产出相同结果（测试依赖此性质）。
- manifest 存储：各 builder 写各自 `d0.jsonl … d4.jsonl`；QC+划分后合并为 `manifest.jsonl`。每行一个 `Sample` 的 JSON。
- 默认 backend = `mock`。
- mask 面积比有效范围 `[0.01, 0.50]`；D2 采样尺度桶：small `1%–5%`、mid `5%–20%`、large `20%–50%`。
- QES 评分权重固定：`0.3*confidence + 0.2*boundary_sharpness + 0.2*mask_consistency + 0.2*semantic_consistency + 0.1*area_validity`；分流阈值 `≥0.75` 入训练、`0.60–0.75` 待复核、`<0.60` 删除。
- 8 个 split：`train, val, test_a, test_b, test_c, test_d, test_e, test_f`；5 条泄漏规则必须在划分后断言通过。
- 纪律：TDD（先写失败测试）、小步提交、DRY、YAGNI。每个任务结束 `git commit`。

---

## File Structure

| 文件 | 职责 |
|---|---|
| `pyproject.toml` | 包元数据、核心依赖、可选 extras、pytest 配置 |
| `.gitignore` | 忽略 `data/`、`__pycache__`、`*.egg-info`、venv |
| `src/forgery_pipeline/schema.py` | `Sample`/`Postprocess`/`Explanation` pydantic 模型 + 基础字段校验 |
| `src/forgery_pipeline/labels.py` | Level0–4 标签集合、`validate_labels`、`LOSS_TERMS` |
| `src/forgery_pipeline/ids.py` | 稳定 `image_id` 生成 |
| `src/forgery_pipeline/manifest.py` | JSONL 读/写/追加/合并/统计 |
| `src/forgery_pipeline/config.py` | YAML → `PipelineConfig` |
| `src/forgery_pipeline/dedup.py` | pHash 去重器 |
| `src/forgery_pipeline/backends/base.py` | 5 个 backend ABC + 图像/掩码类型约定 |
| `src/forgery_pipeline/backends/mock.py` | 5 个确定性 mock backend |
| `src/forgery_pipeline/backends/registry.py` | 按名解析 backend；real stub |
| `src/forgery_pipeline/masks/morphology.py` | dilation/erosion/boundary blur/irregular |
| `src/forgery_pipeline/masks/candidates.py` | area_ratio、尺度分桶、过滤采样 |
| `src/forgery_pipeline/masks/pseudo_mask.py` | 配准/差分/coarse/细化/连通域/编排 |
| `src/forgery_pipeline/qc/image_qc.py` | 图像质量过滤 |
| `src/forgery_pipeline/qc/mask_qc.py` | mask 质量过滤 |
| `src/forgery_pipeline/qc/gen_qc.py` | 生成质量过滤 |
| `src/forgery_pipeline/qc/quality_score.py` | QES 评分与分桶 |
| `src/forgery_pipeline/postprocess/degradations.py` | 后处理退化与参数记录 |
| `src/forgery_pipeline/builders/d0_real.py` | D0 真实图像池 |
| `src/forgery_pipeline/builders/d1_whole.py` | D1 整图生成 |
| `src/forgery_pipeline/builders/d2_local.py` | D2 局部篡改 |
| `src/forgery_pipeline/builders/d3_web.py` | D3 网页人工篡改伪标注 |
| `src/forgery_pipeline/builders/d4_explain.py` | D4 可解释子集 |
| `src/forgery_pipeline/split/grouping.py` | 分组键 |
| `src/forgery_pipeline/split/leakage.py` | 5 条泄漏检查 |
| `src/forgery_pipeline/split/splitter.py` | 8-way 划分 |
| `src/forgery_pipeline/pipeline.py` | 阶段编排 |
| `src/forgery_pipeline/cli.py` | argparse 命令行 |
| `configs/*.yaml` | 主配置 / 生成器清单 / 划分规则 |
| `tests/**` | 每模块单测 + 端到端 smoke |
| `examples/run_mock_pipeline.sh` | mock 全流程示例 |

约定：所有源码在 `src/` 布局下，包名 `forgery_pipeline`，通过 `pip install -e .` 安装；测试用 `from forgery_pipeline... import ...`。

---

## Phase 0 — 项目骨架

### Task 1: 包骨架与打包配置

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `src/forgery_pipeline/__init__.py`, `tests/test_smoke_import.py`
- Create: `src/forgery_pipeline/backends/__init__.py`, `src/forgery_pipeline/masks/__init__.py`, `src/forgery_pipeline/qc/__init__.py`, `src/forgery_pipeline/postprocess/__init__.py`, `src/forgery_pipeline/builders/__init__.py`, `src/forgery_pipeline/split/__init__.py`（均为空文件）

**Interfaces:**
- Produces: 可安装包 `forgery_pipeline`，`__version__: str`。

- [ ] **Step 1: 写失败测试** `tests/test_smoke_import.py`

```python
def test_package_imports_and_has_version():
    import forgery_pipeline
    assert isinstance(forgery_pipeline.__version__, str)
    assert forgery_pipeline.__version__
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_smoke_import.py -q`
Expected: FAIL（`ModuleNotFoundError: No module named 'forgery_pipeline'`）

- [ ] **Step 3: 写 `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "forgery-pipeline"
version = "0.1.0"
description = "Forgery detection dataset generation pipeline (D0-D4, manifest, QC, split)"
requires-python = ">=3.10"
dependencies = [
    "pydantic>=2",
    "numpy",
    "Pillow",
    "opencv-python-headless",
    "scikit-image",
    "imagehash",
    "pyyaml",
    "tqdm",
]

[project.optional-dependencies]
dev = ["pytest"]
real = ["torch", "diffusers", "transformers"]
sam = ["segment-anything"]
clip = ["open-clip-torch"]
mllm = ["openai", "anthropic"]

[project.scripts]
forgery-pipeline = "forgery_pipeline.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 4: 写 `.gitignore`**

```gitignore
__pycache__/
*.py[cod]
*.egg-info/
.eggs/
build/
dist/
.venv/
venv/
.pytest_cache/
data/
!data/.gitkeep
```

- [ ] **Step 5: 写 `src/forgery_pipeline/__init__.py`**

```python
"""Forgery detection 数据集生成 pipeline。"""
__version__ = "0.1.0"
```

创建其余空 `__init__.py`（backends/masks/qc/postprocess/builders/split 各一个）。

- [ ] **Step 6: 安装并运行测试**

Run: `pip install -e . && pytest tests/test_smoke_import.py -q`
Expected: PASS

- [ ] **Step 7: 提交**

```bash
git add pyproject.toml .gitignore src tests
git commit -m "chore: 初始化 forgery_pipeline 包骨架与打包配置"
```

---

## Phase 1 — 基础层（数据契约）

### Task 2: Manifest schema（`schema.py`）

**Files:**
- Create: `src/forgery_pipeline/schema.py`, `tests/test_schema.py`

**Interfaces:**
- Produces:
  - `class TaskType(str, Enum)`: `real_pristine, whole_image_detection, localization, explainable`
  - `class Postprocess(BaseModel)`: `jpeg_quality: int|Literal["none"]="none"`, `resize: str="none"`, `blur: str="none"`, `noise: str="none"`
  - `class Explanation(BaseModel)`: `location_description, visual_artifact_description, semantic_reasoning, forensic_conclusion: str`
  - `class Sample(BaseModel)`: 全字段（见下）；`is_fake ∈ {0,1}`，`mask_area_ratio ∈ [0,1] 或 None`，`quality_score ∈ [0,1] 或 None`。标签一致性校验在 Task 3 接入（此处先不做跨字段标签校验）。

- [ ] **Step 1: 写失败测试** `tests/test_schema.py`

```python
import pytest
from pydantic import ValidationError
from forgery_pipeline.schema import Sample, Postprocess, Explanation, TaskType


def test_minimal_real_sample_ok():
    s = Sample(image_id="real_0001", image_path="D0/real_0001.jpg",
              is_fake=0, task_type=TaskType.real_pristine)
    assert s.postprocess.jpeg_quality == "none"
    assert s.mask_path is None


def test_is_fake_must_be_binary():
    with pytest.raises(ValidationError):
        Sample(image_id="x", image_path="x.jpg", is_fake=2,
               task_type=TaskType.real_pristine)


def test_mask_area_ratio_range():
    with pytest.raises(ValidationError):
        Sample(image_id="x", image_path="x.jpg", is_fake=1,
               task_type=TaskType.localization, mask_area_ratio=1.5)


def test_roundtrip_json():
    s = Sample(image_id="g1", image_path="D1/g1.png", is_fake=1,
               task_type=TaskType.whole_image_detection,
               postprocess=Postprocess(jpeg_quality=70))
    data = s.model_dump()
    s2 = Sample(**data)
    assert s2.postprocess.jpeg_quality == 70
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_schema.py -q` → FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 实现 `schema.py`**

```python
"""Manifest 数据契约：每行一个 Sample（对应报告 §13）。"""
from __future__ import annotations
from enum import Enum
from typing import Literal, Optional, Union
from pydantic import BaseModel, Field, field_validator


class TaskType(str, Enum):
    real_pristine = "real_pristine"
    whole_image_detection = "whole_image_detection"
    localization = "localization"
    explainable = "explainable"


class Postprocess(BaseModel):
    jpeg_quality: Union[int, Literal["none"]] = "none"
    resize: str = "none"
    blur: str = "none"
    noise: str = "none"


class Explanation(BaseModel):
    location_description: str
    visual_artifact_description: str
    semantic_reasoning: str
    forensic_conclusion: str


class Sample(BaseModel):
    image_id: str
    image_path: str
    real_image_path: Optional[str] = None
    mask_path: Optional[str] = None
    is_fake: int
    task_type: TaskType
    manipulation_level1: Optional[str] = None
    manipulation_level2: Optional[str] = None
    manipulation_level3: Optional[str] = None
    manipulation_level4: Optional[str] = None
    source_dataset: Optional[str] = None
    generator_name: Optional[str] = None
    generator_family: Optional[str] = None
    prompt: Optional[str] = None
    negative_prompt: Optional[str] = None
    seed: Optional[int] = None
    sampler: Optional[str] = None
    steps: Optional[int] = None
    cfg_scale: Optional[float] = None
    mask_source: Optional[str] = None
    mask_area_ratio: Optional[float] = None
    postprocess: Postprocess = Field(default_factory=Postprocess)
    quality_score: Optional[float] = None
    quality_bucket: Optional[str] = None
    split: Optional[str] = None
    license: Optional[str] = None
    explanation: Optional[Explanation] = None

    @field_validator("is_fake")
    @classmethod
    def _check_is_fake(cls, v: int) -> int:
        if v not in (0, 1):
            raise ValueError("is_fake 必须是 0 或 1")
        return v

    @field_validator("mask_area_ratio", "quality_score")
    @classmethod
    def _check_unit_interval(cls, v):
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError("取值必须落在 [0, 1]")
        return v
```

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/test_schema.py -q` → PASS

- [ ] **Step 5: 提交**

```bash
git add src/forgery_pipeline/schema.py tests/test_schema.py
git commit -m "feat(schema): Sample/Postprocess/Explanation 模型与基础字段校验"
```

---

### Task 3: 层级标签体系（`labels.py`）

**Files:**
- Create: `src/forgery_pipeline/labels.py`, `tests/test_labels.py`
- Modify: `src/forgery_pipeline/schema.py`（在 `Sample` 末尾加 `model_validator` 调用 `validate_labels`）

**Interfaces:**
- Produces:
  - 常量集合：`LEVEL1: set[str]`, `LEVEL2: set[str]`, `LEVEL3: set[str]`（Level4 自由字符串）
  - `LOSS_TERMS: list[str]`（多任务 loss 字段名，仅文档化）
  - `validate_labels(is_fake:int, task_type:str, mask_path:str|None, l1,l2,l3) -> list[str]`：返回错误信息列表（空=通过）
- Consumes（Task 2）：`Sample`, `TaskType`

- [ ] **Step 1: 写失败测试** `tests/test_labels.py`

```python
import pytest
from pydantic import ValidationError
from forgery_pipeline.labels import validate_labels, LEVEL1, LEVEL2, LOSS_TERMS
from forgery_pipeline.schema import Sample, TaskType


def test_real_must_have_no_manip_labels():
    errs = validate_labels(0, "real_pristine", None, "whole_generated", None, None)
    assert errs  # 真实图不应带 level1


def test_partial_requires_mask():
    errs = validate_labels(1, "localization", None, "partial_manipulated", "AIGC-editing", None)
    assert any("mask" in e for e in errs)


def test_whole_generated_ok_without_mask():
    errs = validate_labels(1, "whole_image_detection", None, "whole_generated", "diffusion", None)
    assert errs == []


def test_level2_must_be_known():
    errs = validate_labels(1, "whole_image_detection", None, "whole_generated", "bogus", None)
    assert any("level2" in e for e in errs)


def test_constants_present():
    assert "whole_generated" in LEVEL1 and "partial_manipulated" in LEVEL1
    assert "diffusion" in LEVEL2
    assert "detection_loss" in LOSS_TERMS


def test_sample_model_rejects_inconsistent_labels():
    with pytest.raises(ValidationError):
        Sample(image_id="x", image_path="x.jpg", is_fake=1,
               task_type=TaskType.localization,
               manipulation_level1="partial_manipulated")  # 缺 mask_path
```

- [ ] **Step 2: 运行确认失败** → FAIL

- [ ] **Step 3: 实现 `labels.py`**

```python
"""HiFi-Net 风格层级标签体系（报告 §9）。"""
from __future__ import annotations
from typing import Optional

LEVEL1 = {"whole_generated", "partial_manipulated"}

LEVEL2 = {
    "diffusion", "GAN", "autoregressive", "Photoshop-editing", "DeepFake",
    "AIGC-editing", "copy-move", "splicing", "removal",
}

LEVEL3 = {
    "conditional_generation", "unconditional_generation", "text_guided_editing",
    "image_guided_editing", "mask_guided_inpainting", "object_replacement",
    "object_removal", "face_swap", "text_editing",
}

# whole_generated 不应配局部编辑类 level2
_WHOLE_ONLY_L2 = {"diffusion", "GAN", "autoregressive"}

# 多任务 loss 字段（报告 §9，仅文档化，训练时使用）
LOSS_TERMS = [
    "detection_loss", "localization_loss", "manipulation_type_loss",
    "generator_family_loss", "optional_explanation_loss",
]


def validate_labels(is_fake: int, task_type: str, mask_path: Optional[str],
                    l1: Optional[str], l2: Optional[str],
                    l3: Optional[str]) -> list[str]:
    """返回标签一致性错误列表，空列表表示通过。"""
    errs: list[str] = []
    if is_fake == 0:
        if any([l1, l2, l3]):
            errs.append("真实图（is_fake=0）不应带 manipulation_level 标签")
        if task_type != "real_pristine":
            errs.append("真实图 task_type 必须为 real_pristine")
        return errs

    # is_fake == 1
    if l1 not in LEVEL1:
        errs.append(f"level1 非法: {l1!r}")
    if l2 is not None and l2 not in LEVEL2:
        errs.append(f"level2 非法: {l2!r}")
    if l3 is not None and l3 not in LEVEL3:
        errs.append(f"level3 非法: {l3!r}")

    if l1 == "partial_manipulated":
        if not mask_path:
            errs.append("partial_manipulated 必须提供 mask_path")
        if task_type not in ("localization", "explainable"):
            errs.append("partial_manipulated 的 task_type 应为 localization 或 explainable")
    elif l1 == "whole_generated":
        if l2 is not None and l2 not in _WHOLE_ONLY_L2:
            errs.append(f"whole_generated 不应配 level2={l2!r}")
    return errs
```

- [ ] **Step 4: 接入 `schema.py`** —— 在 imports 加 `from pydantic import model_validator`、`from forgery_pipeline.labels import validate_labels`，并在 `Sample` 类末尾追加：

```python
    @model_validator(mode="after")
    def _check_label_consistency(self):
        errs = validate_labels(
            self.is_fake, self.task_type.value, self.mask_path,
            self.manipulation_level1, self.manipulation_level2,
            self.manipulation_level3,
        )
        if errs:
            raise ValueError("; ".join(errs))
        return self
```

- [ ] **Step 5: 运行确认通过**

Run: `pytest tests/test_labels.py tests/test_schema.py -q` → PASS

- [ ] **Step 6: 提交**

```bash
git add src/forgery_pipeline/labels.py src/forgery_pipeline/schema.py tests/test_labels.py
git commit -m "feat(labels): Level0-4 层级标签与一致性校验并接入 Sample"
```

---

### Task 4: 稳定 ID 生成（`ids.py`）

**Files:**
- Create: `src/forgery_pipeline/ids.py`, `tests/test_ids.py`

**Interfaces:**
- Produces:
  - `make_image_id(prefix: str, payload: bytes | str) -> str` → `f"{prefix}_{sha1(payload)[:12]}"`，确定性。
  - `content_hash(img: np.ndarray) -> bytes`：对图像字节做 sha1 摘要原始 bytes。

- [ ] **Step 1: 写失败测试** `tests/test_ids.py`

```python
import numpy as np
from forgery_pipeline.ids import make_image_id, content_hash


def test_make_image_id_deterministic():
    a = make_image_id("real", "hello")
    b = make_image_id("real", "hello")
    assert a == b
    assert a.startswith("real_") and len(a) == len("real_") + 12


def test_make_image_id_varies_with_payload():
    assert make_image_id("real", "a") != make_image_id("real", "b")


def test_content_hash_stable():
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    assert content_hash(img) == content_hash(img.copy())
```

- [ ] **Step 2: 运行确认失败** → FAIL

- [ ] **Step 3: 实现 `ids.py`**

```python
"""稳定、确定性的样本 ID 生成。"""
from __future__ import annotations
import hashlib
import numpy as np


def content_hash(img: np.ndarray) -> bytes:
    """对图像原始字节求 sha1 摘要。"""
    return hashlib.sha1(np.ascontiguousarray(img).tobytes()).digest()


def make_image_id(prefix: str, payload: bytes | str) -> str:
    """生成 `<prefix>_<sha1[:12]>` 形式的稳定 ID。"""
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    digest = hashlib.sha1(payload).hexdigest()[:12]
    return f"{prefix}_{digest}"
```

- [ ] **Step 4: 运行确认通过** → PASS

- [ ] **Step 5: 提交**

```bash
git add src/forgery_pipeline/ids.py tests/test_ids.py
git commit -m "feat(ids): 确定性 image_id 与内容哈希"
```

---

### Task 5: Manifest 读写与统计（`manifest.py`）

**Files:**
- Create: `src/forgery_pipeline/manifest.py`, `tests/test_manifest.py`

**Interfaces:**
- Consumes（Task 2）：`Sample`
- Produces:
  - `write_jsonl(path: str|Path, samples: Iterable[Sample]) -> int`（返回写入行数，每行校验）
  - `append_jsonl(path, samples) -> int`
  - `read_jsonl(path) -> list[Sample]`（逐行校验）
  - `merge(paths: list[str|Path], out_path) -> int`
  - `stats(samples: list[Sample]) -> dict`（计数：总数、is_fake、task_type、generator_family、split、有 mask 数）

- [ ] **Step 1: 写失败测试** `tests/test_manifest.py`

```python
from forgery_pipeline.schema import Sample, TaskType
from forgery_pipeline import manifest


def _real(i):
    return Sample(image_id=f"real_{i}", image_path=f"D0/real_{i}.jpg",
                  is_fake=0, task_type=TaskType.real_pristine,
                  generator_family=None)


def _fake(i):
    return Sample(image_id=f"gen_{i}", image_path=f"D1/gen_{i}.png", is_fake=1,
                  task_type=TaskType.whole_image_detection,
                  manipulation_level1="whole_generated",
                  manipulation_level2="diffusion", generator_family="diffusion")


def test_write_read_roundtrip(tmp_path):
    p = tmp_path / "d0.jsonl"
    n = manifest.write_jsonl(p, [_real(0), _real(1)])
    assert n == 2
    got = manifest.read_jsonl(p)
    assert [s.image_id for s in got] == ["real_0", "real_1"]


def test_append_and_merge(tmp_path):
    p0, p1 = tmp_path / "d0.jsonl", tmp_path / "d1.jsonl"
    manifest.write_jsonl(p0, [_real(0)])
    manifest.write_jsonl(p1, [_fake(0)])
    out = tmp_path / "manifest.jsonl"
    n = manifest.merge([p0, p1], out)
    assert n == 2
    s = manifest.stats(manifest.read_jsonl(out))
    assert s["total"] == 2 and s["fake"] == 1 and s["real"] == 1


def test_stats_counts(tmp_path):
    s = manifest.stats([_real(0), _fake(0), _fake(1)])
    assert s["by_task_type"]["whole_image_detection"] == 2
    assert s["by_generator_family"]["diffusion"] == 2
```

- [ ] **Step 2: 运行确认失败** → FAIL

- [ ] **Step 3: 实现 `manifest.py`**

```python
"""统一 manifest 的 JSONL 读写、合并与统计。"""
from __future__ import annotations
from collections import Counter
from pathlib import Path
from typing import Iterable
from forgery_pipeline.schema import Sample


def write_jsonl(path, samples: Iterable[Sample], mode: str = "w") -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, mode, encoding="utf-8") as f:
        for s in samples:
            f.write(s.model_dump_json() + "\n")
            n += 1
    return n


def append_jsonl(path, samples: Iterable[Sample]) -> int:
    return write_jsonl(path, samples, mode="a")


def read_jsonl(path) -> list[Sample]:
    out: list[Sample] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(Sample.model_validate_json(line))
    return out


def merge(paths, out_path) -> int:
    all_samples: list[Sample] = []
    for p in paths:
        if Path(p).exists():
            all_samples.extend(read_jsonl(p))
    return write_jsonl(out_path, all_samples)


def stats(samples: list[Sample]) -> dict:
    return {
        "total": len(samples),
        "real": sum(1 for s in samples if s.is_fake == 0),
        "fake": sum(1 for s in samples if s.is_fake == 1),
        "with_mask": sum(1 for s in samples if s.mask_path),
        "by_task_type": dict(Counter(s.task_type.value for s in samples)),
        "by_generator_family": dict(
            Counter(s.generator_family for s in samples if s.generator_family)),
        "by_split": dict(Counter(s.split for s in samples if s.split)),
    }
```

- [ ] **Step 4: 运行确认通过** → PASS

- [ ] **Step 5: 提交**

```bash
git add src/forgery_pipeline/manifest.py tests/test_manifest.py
git commit -m "feat(manifest): JSONL 读写/合并/统计"
```

---

### Task 6: 配置加载（`config.py` + `configs/*.yaml`）

**Files:**
- Create: `src/forgery_pipeline/config.py`, `tests/test_config.py`
- Create: `configs/pipeline.example.yaml`, `configs/generators.yaml`, `configs/split.yaml`

**Interfaces:**
- Produces:
  - `@dataclass GeneratorSpec{ name:str, family:str, kind:str }`
  - `@dataclass StageScales{ d0:int, d1_per_generator:int, d2:int, d3:int, d4:int }`
  - `@dataclass PipelineConfig{ out_dir:str, seed:int, backend:str, stages:dict[str,bool], scales:StageScales, generators:list[GeneratorSpec], postprocess_prob:float }`
  - `load_config(path) -> PipelineConfig`
  - `load_generators(path) -> list[GeneratorSpec]`

- [ ] **Step 1: 写失败测试** `tests/test_config.py`

```python
from forgery_pipeline.config import load_config, PipelineConfig


def test_load_example_config():
    cfg = load_config("configs/pipeline.example.yaml")
    assert isinstance(cfg, PipelineConfig)
    assert cfg.seed == 1234
    assert cfg.backend == "mock"
    assert cfg.stages["d0"] is True
    assert cfg.scales.d1_per_generator >= 1
    assert len(cfg.generators) >= 3
    fams = {g.family for g in cfg.generators}
    assert "diffusion" in fams
```

- [ ] **Step 2: 运行确认失败** → FAIL

- [ ] **Step 3: 写 `configs/generators.yaml`**

```yaml
# 生成器清单（强调多样性；mock 后端忽略具体权重，仅用 name/family 打标签）
generators:
  - {name: stable-diffusion-1-5, family: diffusion, kind: txt2img}
  - {name: stable-diffusion-xl,  family: diffusion, kind: txt2img}
  - {name: flux-1,               family: diffusion, kind: txt2img}
  - {name: openjourney,          family: diffusion, kind: txt2img}
  - {name: stylegan2,            family: GAN,       kind: unconditional}
  - {name: biggan,               family: GAN,       kind: conditional}
  - {name: progan,               family: GAN,       kind: unconditional}
  - {name: dalle-3,              family: autoregressive, kind: txt2img}
  - {name: midjourney-v6,        family: diffusion, kind: txt2img}
  - {name: ideogram,             family: diffusion, kind: txt2img}
inpainters:
  - {name: stable-diffusion-inpaint, family: diffusion, kind: inpaint}
  - {name: glide-inpaint,            family: diffusion, kind: inpaint}
```

- [ ] **Step 4: 写 `configs/split.yaml`**

```yaml
# 8-way 划分比例与规则（详见 split/splitter.py）
ratios: {train: 0.70, val: 0.10, test_a: 0.05, test_b: 0.05, test_c: 0.03,
         test_d: 0.03, test_e: 0.02, test_f: 0.02}
# 指定哪些生成器仅出现在 cross-generator 测试集（Test-B）
holdout_generators: [ideogram, progan]
# 指定哪类篡改仅出现在 cross-manipulation 测试集（Test-C）
holdout_manipulation: [text_editing]
# 指定哪些来源域仅出现在 cross-domain 测试集（Test-D）
holdout_domains: [Places]
```

- [ ] **Step 5: 写 `configs/pipeline.example.yaml`**

```yaml
out_dir: data/run
seed: 1234
backend: mock
postprocess_prob: 0.5
stages: {d0: true, d1: true, d2: true, d3: true, d4: true,
         qc: true, postprocess: true, split: true}
scales: {d0: 40, d1_per_generator: 3, d2: 24, d3: 12, d4: 8}
generators_config: configs/generators.yaml
split_config: configs/split.yaml
```

- [ ] **Step 6: 实现 `config.py`**

```python
"""YAML 配置加载与校验。"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import yaml


@dataclass
class GeneratorSpec:
    name: str
    family: str
    kind: str


@dataclass
class StageScales:
    d0: int = 40
    d1_per_generator: int = 3
    d2: int = 24
    d3: int = 12
    d4: int = 8


@dataclass
class PipelineConfig:
    out_dir: str
    seed: int
    backend: str
    stages: dict
    scales: StageScales
    generators: list[GeneratorSpec] = field(default_factory=list)
    inpainters: list[GeneratorSpec] = field(default_factory=list)
    postprocess_prob: float = 0.5
    split_config: str = "configs/split.yaml"


def load_generators(path) -> tuple[list[GeneratorSpec], list[GeneratorSpec]]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    gens = [GeneratorSpec(**g) for g in data.get("generators", [])]
    inps = [GeneratorSpec(**g) for g in data.get("inpainters", [])]
    return gens, inps


def load_config(path) -> PipelineConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    gens, inps = load_generators(data["generators_config"])
    return PipelineConfig(
        out_dir=data["out_dir"],
        seed=int(data["seed"]),
        backend=data.get("backend", "mock"),
        stages=data["stages"],
        scales=StageScales(**data["scales"]),
        generators=gens,
        inpainters=inps,
        postprocess_prob=float(data.get("postprocess_prob", 0.5)),
        split_config=data.get("split_config", "configs/split.yaml"),
    )
```

- [ ] **Step 7: 运行确认通过**

Run: `pytest tests/test_config.py -q` → PASS

- [ ] **Step 8: 提交**

```bash
git add src/forgery_pipeline/config.py tests/test_config.py configs/
git commit -m "feat(config): YAML 配置加载与示例配置（pipeline/generators/split）"
```

---

## Phase 2 — Backend 层与 mask 工具

### Task 7: Backend 抽象接口（`backends/base.py`）

**Files:**
- Create: `src/forgery_pipeline/backends/base.py`, `tests/test_backends_base.py`

**Interfaces:**
- Produces（类型别名与 ABC，供后续所有 backend/builder 使用）：
  - `Image = np.ndarray`（(H,W,3) uint8 RGB）、`Mask = np.ndarray`（(H,W) uint8 {0,255}）
  - `ImageSource.iter_images(n:int) -> Iterator[tuple[Image, dict]]`
  - `WholeImageGenerator.generate(prompt:str, params:dict) -> tuple[Image, dict]`
  - `Inpainter.inpaint(image:Image, mask:Mask, prompt:str, params:dict) -> tuple[Image, dict]`
  - `Segmenter.propose_masks(image:Image, k:int) -> list[Mask]`
  - `Explainer.explain(image:Image, mask:Optional[Mask], context:dict) -> Explanation`
- Consumes（Task 2）：`Explanation`

- [ ] **Step 1: 写失败测试** `tests/test_backends_base.py`

```python
import pytest
from forgery_pipeline.backends import base


def test_abstract_cannot_instantiate():
    with pytest.raises(TypeError):
        base.ImageSource()


def test_subclass_must_implement():
    class Bad(base.Segmenter):
        pass
    with pytest.raises(TypeError):
        Bad()

    class Good(base.Segmenter):
        def propose_masks(self, image, k):
            return []
    assert Good().propose_masks(None, 0) == []
```

- [ ] **Step 2: 运行确认失败** → FAIL

- [ ] **Step 3: 实现 `backends/base.py`**

```python
"""Backend 抽象接口（模型契约）。所有重型 ML 阶段经此解耦。"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Iterator, Optional
import numpy as np
from forgery_pipeline.schema import Explanation

# 类型约定：图像 (H,W,3) uint8 RGB；掩码 (H,W) uint8 取值 {0,255}
Image = np.ndarray
Mask = np.ndarray


class ImageSource(ABC):
    """真实图像源（D0/D3 底图来源）。"""
    @abstractmethod
    def iter_images(self, n: int) -> Iterator[tuple[Image, dict]]:
        ...


class WholeImageGenerator(ABC):
    """整图生成器（D1）。"""
    @abstractmethod
    def generate(self, prompt: str, params: dict) -> tuple[Image, dict]:
        ...


class Inpainter(ABC):
    """局部重绘模型（D2）。"""
    @abstractmethod
    def inpaint(self, image: Image, mask: Mask, prompt: str,
                params: dict) -> tuple[Image, dict]:
        ...


class Segmenter(ABC):
    """分割/候选 mask 生成（D2 候选、D3 细化）。"""
    @abstractmethod
    def propose_masks(self, image: Image, k: int) -> list[Mask]:
        ...


class Explainer(ABC):
    """MLLM 解释生成（D4）。"""
    @abstractmethod
    def explain(self, image: Image, mask: Optional[Mask],
                context: dict) -> Explanation:
        ...
```

- [ ] **Step 4: 运行确认通过** → PASS

- [ ] **Step 5: 提交**

```bash
git add src/forgery_pipeline/backends/base.py tests/test_backends_base.py
git commit -m "feat(backends): 抽象接口 ImageSource/Generator/Inpainter/Segmenter/Explainer"
```

---

### Task 8: Mock backend（`backends/mock.py`）

**Files:**
- Create: `src/forgery_pipeline/backends/mock.py`, `tests/test_backends_mock.py`

**Interfaces:**
- Consumes（Task 7）：`base.*`；（Task 2）：`Explanation`
- Produces（确定性合成实现）：
  - `MockImageSource(seed:int=0, size:tuple=(256,256))`
  - `MockWholeImageGenerator(name:str, family:str)`
  - `MockInpainter(name:str, family:str)`
  - `MockSegmenter(seed:int=0)`
  - `MockExplainer()`
  - 辅助：`stable_hash(s:str)->int`、`synth_image(rng,h,w)->Image`、`rect_mask(rng,h,w,frac)->Mask`

> 关键：所有随机性走 `np.random.default_rng(seed)`；字符串散列用 `stable_hash`（基于 hashlib，跨进程稳定），**禁止**用内置 `hash()`。

- [ ] **Step 1: 写失败测试** `tests/test_backends_mock.py`

```python
import numpy as np
from forgery_pipeline.backends import mock
from forgery_pipeline.schema import Explanation


def test_image_source_deterministic_and_shape():
    a = list(mock.MockImageSource(seed=7).iter_images(3))
    b = list(mock.MockImageSource(seed=7).iter_images(3))
    assert len(a) == 3
    img, meta = a[0]
    assert img.shape == (256, 256, 3) and img.dtype == np.uint8
    assert np.array_equal(a[0][0], b[0][0])  # 确定性
    assert meta["source_dataset"]


def test_whole_generator_deterministic():
    g = mock.MockWholeImageGenerator("stable-diffusion-xl", "diffusion")
    i1, m1 = g.generate("a dog", {"seed": 5})
    i2, _ = g.generate("a dog", {"seed": 5})
    i3, _ = g.generate("a cat", {"seed": 5})
    assert np.array_equal(i1, i2)
    assert not np.array_equal(i1, i3)  # prompt 改变结果
    assert m1["generator_name"] == "stable-diffusion-xl"


def test_inpainter_changes_only_masked_region():
    img = np.full((64, 64, 3), 100, np.uint8)
    mask = np.zeros((64, 64), np.uint8)
    mask[10:30, 10:30] = 255
    out, meta = mock.MockInpainter().inpaint(img, mask, "replace", {"seed": 1})
    assert out.shape == img.shape
    assert not np.array_equal(out[10:30, 10:30], img[10:30, 10:30])  # 区域被改
    assert np.array_equal(out[40:60, 40:60], img[40:60, 40:60])      # 区域外不变


def test_segmenter_masks_binary_and_count():
    img = np.zeros((128, 128, 3), np.uint8)
    masks = mock.MockSegmenter(seed=3).propose_masks(img, 5)
    assert len(masks) == 5
    for m in masks:
        assert m.shape == (128, 128) and m.dtype == np.uint8
        assert set(np.unique(m)).issubset({0, 255})


def test_explainer_returns_explanation():
    e = mock.MockExplainer().explain(np.zeros((8, 8, 3), np.uint8), None,
                                     {"manipulation_level3": "object_replacement"})
    assert isinstance(e, Explanation)
    assert "object_replacement" in e.forensic_conclusion
```

- [ ] **Step 2: 运行确认失败** → FAIL

- [ ] **Step 3: 实现 `backends/mock.py`**

```python
"""确定性 mock backend：让全流程在 CPU 上跑通且可复现。"""
from __future__ import annotations
import hashlib
from typing import Iterator, Optional
import cv2
import numpy as np
from forgery_pipeline.backends import base
from forgery_pipeline.schema import Explanation


def stable_hash(s: str) -> int:
    """跨进程稳定的字符串散列（不可用内置 hash）。"""
    return int.from_bytes(hashlib.sha1(s.encode("utf-8")).digest()[:4], "big")


def synth_image(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    """渐变背景 + 若干随机色块的合成图。"""
    yy, xx = np.mgrid[0:h, 0:w]
    r = (xx / max(w, 1) * 255).astype(np.uint8)
    g = (yy / max(h, 1) * 255).astype(np.uint8)
    b = ((xx + yy) / max(w + h, 1) * 255).astype(np.uint8)
    img = np.stack([r, g, b], axis=-1).astype(np.uint8)
    for _ in range(int(rng.integers(2, 5))):
        cy, cx = int(rng.integers(0, h)), int(rng.integers(0, w))
        rad = int(rng.integers(max(4, min(h, w) // 16), max(8, min(h, w) // 6)))
        color = rng.integers(0, 256, size=3).astype(np.uint8)
        dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
        img[dist2 <= rad ** 2] = color
    return img


def rect_mask(rng: np.random.Generator, h: int, w: int, frac: float) -> np.ndarray:
    """生成面积约为 frac*H*W 的矩形二值掩码。"""
    area = max(1.0, frac * h * w)
    aspect = float(rng.uniform(0.5, 2.0))
    bh = max(1, min(h, int(np.sqrt(area / aspect))))
    bw = max(1, min(w, int(aspect * bh)))
    y0 = int(rng.integers(0, max(1, h - bh + 1)))
    x0 = int(rng.integers(0, max(1, w - bw + 1)))
    mask = np.zeros((h, w), np.uint8)
    mask[y0:y0 + bh, x0:x0 + bw] = 255
    return mask


class MockImageSource(base.ImageSource):
    DATASETS = ["COCO", "ImageNet", "OpenImages", "FFHQ", "Places"]

    def __init__(self, seed: int = 0, size: tuple[int, int] = (256, 256)):
        self.seed = seed
        self.size = size

    def iter_images(self, n: int) -> Iterator[tuple[np.ndarray, dict]]:
        h, w = self.size
        for i in range(n):
            rng = np.random.default_rng(self.seed + i)
            img = synth_image(rng, h, w)
            meta = {
                "source_dataset": self.DATASETS[i % len(self.DATASETS)],
                "camera_model": None,
                "resolution": [w, h],
                "license": "research-only",
            }
            yield img, meta


class MockWholeImageGenerator(base.WholeImageGenerator):
    def __init__(self, name: str = "mock-gen", family: str = "diffusion"):
        self.name, self.family = name, family

    def generate(self, prompt: str, params: dict) -> tuple[np.ndarray, dict]:
        seed = int(params.get("seed", 0))
        h, w = int(params.get("height", 256)), int(params.get("width", 256))
        pseed = (seed * 1000003 + stable_hash(prompt)) & 0x7FFFFFFF
        img = synth_image(np.random.default_rng(pseed), h, w)
        meta = {
            "generator_name": self.name, "generator_family": self.family,
            "seed": seed, "sampler": params.get("sampler", "DPM++ 2M"),
            "steps": int(params.get("steps", 30)),
            "cfg_scale": float(params.get("cfg_scale", 7.5)),
        }
        return img, meta


class MockInpainter(base.Inpainter):
    def __init__(self, name: str = "stable-diffusion-inpaint",
                 family: str = "diffusion"):
        self.name, self.family = name, family

    def inpaint(self, image: np.ndarray, mask: np.ndarray, prompt: str,
                params: dict) -> tuple[np.ndarray, dict]:
        seed = int(params.get("seed", 0))
        rng = np.random.default_rng((seed + stable_hash(prompt)) & 0x7FFFFFFF)
        out = image.copy()
        m = mask > 127
        color = rng.integers(0, 256, size=3).astype(np.uint8)
        out[m] = color
        # 轻微羽化边界，模拟重绘痕迹
        blurred = cv2.GaussianBlur(out, (5, 5), 0)
        edge = cv2.dilate(mask, np.ones((5, 5), np.uint8)) - cv2.erode(mask, np.ones((5, 5), np.uint8))
        out[edge > 127] = blurred[edge > 127]
        meta = {"generator_name": self.name, "generator_family": self.family,
                "seed": seed}
        return out, meta


class MockSegmenter(base.Segmenter):
    def __init__(self, seed: int = 0):
        self.seed = seed

    def propose_masks(self, image: np.ndarray, k: int) -> list[np.ndarray]:
        h, w = image.shape[:2]
        base_seed = self.seed + int.from_bytes(
            hashlib.sha1(np.ascontiguousarray(image).tobytes()).digest()[:4], "big")
        fracs = np.linspace(0.02, 0.45, max(k, 1))
        masks = []
        for j in range(k):
            rng = np.random.default_rng((base_seed + j) & 0x7FFFFFFF)
            masks.append(rect_mask(rng, h, w, float(fracs[j])))
        return masks


class MockExplainer(base.Explainer):
    def explain(self, image: np.ndarray, mask: Optional[np.ndarray],
                context: dict) -> Explanation:
        region = context.get("region", "the masked region")
        mtype = context.get("manipulation_level3", "local AIGC inpainting")
        return Explanation(
            location_description=f"The manipulated region is located at {region}.",
            visual_artifact_description=("The object boundary appears overly smooth "
                                         "and inconsistent with the surrounding texture."),
            semantic_reasoning=("The lighting direction and noise of the edited region "
                                "do not match the rest of the scene."),
            forensic_conclusion=f"The image is likely manipulated by {mtype}.",
        )
```

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/test_backends_mock.py -q` → PASS

- [ ] **Step 5: 提交**

```bash
git add src/forgery_pipeline/backends/mock.py tests/test_backends_mock.py
git commit -m "feat(backends): 确定性 mock backend（source/gen/inpaint/segment/explain）"
```

---

### Task 9: Backend 注册表与 real stub（`backends/registry.py` + `backends/real/`）

**Files:**
- Create: `src/forgery_pipeline/backends/registry.py`, `tests/test_backends_registry.py`
- Create: `src/forgery_pipeline/backends/real/__init__.py`, `src/forgery_pipeline/backends/real/diffusers_gen.py`, `src/forgery_pipeline/backends/real/sam_segmenter.py`, `src/forgery_pipeline/backends/real/mllm_explainer.py`

**Interfaces:**
- Consumes：`mock.*`
- Produces:
  - `get_image_source(backend:str, seed:int=0) -> base.ImageSource`
  - `get_whole_generator(backend:str, name:str, family:str) -> base.WholeImageGenerator`
  - `get_inpainter(backend:str, name:str, family:str) -> base.Inpainter`
  - `get_segmenter(backend:str, seed:int=0) -> base.Segmenter`
  - `get_explainer(backend:str) -> base.Explainer`
  - 非 `mock` 后端抛 `NotImplementedError`，信息含安装提示。

- [ ] **Step 1: 写失败测试** `tests/test_backends_registry.py`

```python
import pytest
from forgery_pipeline.backends import registry, mock


def test_mock_resolves():
    assert isinstance(registry.get_image_source("mock"), mock.MockImageSource)
    assert isinstance(registry.get_segmenter("mock"), mock.MockSegmenter)
    g = registry.get_whole_generator("mock", "sdxl", "diffusion")
    assert g.name == "sdxl"


def test_real_backend_raises_with_hint():
    with pytest.raises(NotImplementedError) as ei:
        registry.get_whole_generator("real:diffusers", "sdxl", "diffusion")
    assert "pip install" in str(ei.value)
```

- [ ] **Step 2: 运行确认失败** → FAIL

- [ ] **Step 3: 实现 `backends/registry.py`**

```python
"""按名称解析 backend。默认 mock；真实后端给出清晰的启用提示。"""
from __future__ import annotations
from forgery_pipeline.backends import base, mock

_HINTS = {
    "real:diffusers": "[real]",
    "real:sam": "[sam]",
    "real:mllm": "[mllm]",
}


def _unsupported(backend: str):
    extra = _HINTS.get(backend, "[real]")
    raise NotImplementedError(
        f"backend {backend!r} 未启用：请 `pip install .{extra}` 安装依赖、提供模型权重/API key，"
        f"并在 forgery_pipeline/backends/real/ 中完成适配器实现。当前可用：'mock'。")


def get_image_source(backend: str, seed: int = 0) -> base.ImageSource:
    if backend == "mock":
        return mock.MockImageSource(seed=seed)
    _unsupported(backend)


def get_whole_generator(backend: str, name: str, family: str) -> base.WholeImageGenerator:
    if backend == "mock":
        return mock.MockWholeImageGenerator(name=name, family=family)
    _unsupported(backend)


def get_inpainter(backend: str, name: str, family: str) -> base.Inpainter:
    if backend == "mock":
        return mock.MockInpainter(name=name, family=family)
    _unsupported(backend)


def get_segmenter(backend: str, seed: int = 0) -> base.Segmenter:
    if backend == "mock":
        return mock.MockSegmenter(seed=seed)
    _unsupported(backend)


def get_explainer(backend: str) -> base.Explainer:
    if backend == "mock":
        return mock.MockExplainer()
    _unsupported(backend)
```

- [ ] **Step 4: 写 real stub（guarded import，参考骨架）**

`src/forgery_pipeline/backends/real/__init__.py`：空文件。

`src/forgery_pipeline/backends/real/diffusers_gen.py`：

```python
"""真实整图/重绘生成器适配器骨架（diffusers）。需 `pip install .[real]` 与 GPU/权重。"""
from __future__ import annotations
from forgery_pipeline.backends import base


class DiffusersWholeGenerator(base.WholeImageGenerator):
    def __init__(self, model_id: str, device: str = "cuda"):
        try:
            from diffusers import AutoPipelineForText2Image  # noqa: F401
        except ImportError as e:
            raise NotImplementedError(
                "未安装 diffusers：`pip install .[real]`。") from e
        raise NotImplementedError("参考骨架：在此加载 pipeline 并实现 generate()。")

    def generate(self, prompt, params):
        raise NotImplementedError
```

`sam_segmenter.py` 与 `mllm_explainer.py` 同样的骨架结构（分别 guarded import `segment_anything` 与 `openai`/`anthropic`，`__init__` 抛 `NotImplementedError` 并附安装提示）。

- [ ] **Step 5: 运行确认通过**

Run: `pytest tests/test_backends_registry.py -q` → PASS

- [ ] **Step 6: 提交**

```bash
git add src/forgery_pipeline/backends/registry.py src/forgery_pipeline/backends/real tests/test_backends_registry.py
git commit -m "feat(backends): 注册表与 real 适配器骨架（guarded import）"
```

---

### Task 10: mask 形态学扰动（`masks/morphology.py`）

**Files:**
- Create: `src/forgery_pipeline/masks/morphology.py`, `tests/test_masks_morphology.py`

**Interfaces:**
- Produces（输入/输出均为 (H,W) uint8 {0,255}）：
  - `dilate(mask, ksize=5) -> Mask`
  - `erode(mask, ksize=5) -> Mask`
  - `boundary_blur(mask, ksize=5) -> Mask`
  - `make_irregular(mask, seed=0, ksize=5) -> Mask`

- [ ] **Step 1: 写失败测试** `tests/test_masks_morphology.py`

```python
import numpy as np
from forgery_pipeline.masks import morphology as mo


def _square():
    m = np.zeros((100, 100), np.uint8)
    m[30:70, 30:70] = 255
    return m


def test_dilate_grows_erode_shrinks():
    m = _square()
    base = int((m > 127).sum())
    assert int((mo.dilate(m, 5) > 127).sum()) > base
    assert int((mo.erode(m, 5) > 127).sum()) < base


def test_outputs_binary():
    for fn in (mo.dilate, mo.erode, mo.boundary_blur):
        out = fn(_square(), 5)
        assert set(np.unique(out)).issubset({0, 255})


def test_make_irregular_deterministic_and_binary():
    a = mo.make_irregular(_square(), seed=1)
    b = mo.make_irregular(_square(), seed=1)
    assert np.array_equal(a, b)
    assert set(np.unique(a)).issubset({0, 255})
```

- [ ] **Step 2: 运行确认失败** → FAIL

- [ ] **Step 3: 实现 `masks/morphology.py`**

```python
"""mask 形态学扰动（报告 §6.4：dilation/erosion/boundary blur/irregular）。"""
from __future__ import annotations
import cv2
import numpy as np


def _binarize(mask: np.ndarray) -> np.ndarray:
    return ((mask > 127).astype(np.uint8)) * 255


def _kernel(ksize: int) -> np.ndarray:
    ksize = max(1, ksize | 1)  # 取奇数
    return np.ones((ksize, ksize), np.uint8)


def dilate(mask: np.ndarray, ksize: int = 5) -> np.ndarray:
    return _binarize(cv2.dilate(_binarize(mask), _kernel(ksize)))


def erode(mask: np.ndarray, ksize: int = 5) -> np.ndarray:
    return _binarize(cv2.erode(_binarize(mask), _kernel(ksize)))


def boundary_blur(mask: np.ndarray, ksize: int = 5) -> np.ndarray:
    """高斯模糊后再二值化，得到轻微抖动的边界。"""
    k = max(3, ksize | 1)
    blurred = cv2.GaussianBlur(_binarize(mask), (k, k), 0)
    return _binarize(blurred)


def make_irregular(mask: np.ndarray, seed: int = 0, ksize: int = 5) -> np.ndarray:
    """在膨胀与腐蚀之间按随机场切换，制造不规则边界。"""
    rng = np.random.default_rng(seed)
    d = cv2.dilate(_binarize(mask), _kernel(ksize))
    e = cv2.erode(_binarize(mask), _kernel(ksize))
    choice = rng.random(mask.shape) < 0.5
    out = np.where(choice, d, e).astype(np.uint8)
    return _binarize(out)
```

- [ ] **Step 4: 运行确认通过** → PASS

- [ ] **Step 5: 提交**

```bash
git add src/forgery_pipeline/masks/morphology.py tests/test_masks_morphology.py
git commit -m "feat(masks): 形态学扰动 dilate/erode/boundary_blur/irregular"
```

---

### Task 11: mask 面积与尺度分桶（`masks/candidates.py`）

**Files:**
- Create: `src/forgery_pipeline/masks/candidates.py`, `tests/test_masks_candidates.py`

**Interfaces:**
- Produces:
  - `area_ratio(mask) -> float`（前景像素占比）
  - `bucket_for_ratio(r:float) -> str|None`：`small`(0.01≤r<0.05)/`mid`(0.05≤r<0.20)/`large`(0.20≤r≤0.50)，否则 `None`
  - `filter_and_sample(masks:list[Mask]) -> list[tuple[Mask, float, str]]`：剔除 ratio∉[0.01,0.50]，返回 (mask, ratio, bucket)

- [ ] **Step 1: 写失败测试** `tests/test_masks_candidates.py`

```python
import numpy as np
from forgery_pipeline.masks import candidates as ca


def _mask_with_frac(frac, h=100, w=100):
    m = np.zeros((h, w), np.uint8)
    n = int(frac * h * w)
    m.flat[:n] = 255
    return m


def test_area_ratio():
    assert abs(ca.area_ratio(_mask_with_frac(0.1)) - 0.1) < 1e-6


def test_buckets():
    assert ca.bucket_for_ratio(0.02) == "small"
    assert ca.bucket_for_ratio(0.10) == "mid"
    assert ca.bucket_for_ratio(0.30) == "large"
    assert ca.bucket_for_ratio(0.005) is None  # 太小
    assert ca.bucket_for_ratio(0.7) is None     # 太大


def test_filter_and_sample_drops_invalid():
    masks = [_mask_with_frac(f) for f in (0.005, 0.03, 0.10, 0.30, 0.70)]
    kept = ca.filter_and_sample(masks)
    buckets = {b for _, _, b in kept}
    assert buckets == {"small", "mid", "large"}
    assert len(kept) == 3
```

- [ ] **Step 2: 运行确认失败** → FAIL

- [ ] **Step 3: 实现 `masks/candidates.py`**

```python
"""候选 mask 的面积比与尺度分桶（报告 §6.4）。"""
from __future__ import annotations
from typing import Optional
import numpy as np

MIN_RATIO = 0.01
MAX_RATIO = 0.50


def area_ratio(mask: np.ndarray) -> float:
    return float((mask > 127).sum()) / float(mask.size)


def bucket_for_ratio(r: float) -> Optional[str]:
    if 0.01 <= r < 0.05:
        return "small"
    if 0.05 <= r < 0.20:
        return "mid"
    if 0.20 <= r <= 0.50:
        return "large"
    return None


def filter_and_sample(masks: list[np.ndarray]) -> list[tuple[np.ndarray, float, str]]:
    out: list[tuple[np.ndarray, float, str]] = []
    for m in masks:
        r = area_ratio(m)
        b = bucket_for_ratio(r)
        if b is not None and MIN_RATIO <= r <= MAX_RATIO:
            out.append((m, r, b))
    return out
```

- [ ] **Step 4: 运行确认通过** → PASS

- [ ] **Step 5: 提交**

```bash
git add src/forgery_pipeline/masks/candidates.py tests/test_masks_candidates.py
git commit -m "feat(masks): 面积比与尺度分桶 + 候选过滤"
```

---

### Task 12: 差分伪 mask 流程（`masks/pseudo_mask.py`，D3 核心）

**Files:**
- Create: `src/forgery_pipeline/masks/pseudo_mask.py`, `tests/test_masks_pseudo_mask.py`

**Interfaces:**
- Produces:
  - `align(real:Image, fake:Image) -> tuple[Image, Image]`（mock：resize 对齐）
  - `diff_map(real, fake) -> tuple[np.ndarray(float32, HxW, [0,1]), float ssim_score]`
  - `coarse_mask(diff, thresh=0.15) -> Mask`
  - `connected_component_filter(mask, min_frac=0.005) -> Mask`
  - `refine(mask) -> Mask`（形态学闭运算）
  - `pseudo_mask(real, fake, thresh=0.15) -> tuple[Mask, dict metrics]`，metrics 含 `confidence, ssim_score, boundary_sharpness, area_ratio`
- Consumes：`skimage.metrics.structural_similarity`, `cv2`

- [ ] **Step 1: 写失败测试** `tests/test_masks_pseudo_mask.py`

```python
import numpy as np
from forgery_pipeline.masks import pseudo_mask as pm
from forgery_pipeline.masks.candidates import area_ratio


def _scene(seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(128, 128, 3), dtype=np.uint8)


def test_diff_zero_when_identical():
    img = _scene()
    diff, score = pm.diff_map(img, img)
    assert diff.max() < 1e-3
    assert score > 0.99


def test_pseudo_mask_recovers_edited_region():
    real = _scene(1)
    fake = real.copy()
    fake[40:80, 50:90] = 0  # 已知篡改矩形
    mask, metrics = pm.pseudo_mask(real, fake, thresh=0.1)
    assert set(np.unique(mask)).issubset({0, 255})
    # 召回：篡改矩形内大部分被标记
    region = mask[40:80, 50:90]
    assert (region > 127).mean() > 0.7
    # 精确：整体面积不离谱
    assert area_ratio(mask) < 0.3
    assert metrics["confidence"] > 0
```

- [ ] **Step 2: 运行确认失败** → FAIL

- [ ] **Step 3: 实现 `masks/pseudo_mask.py`**

```python
"""真实 real-fake pair 的差分伪标注（报告 §7.3/§7.4，MIML 思路工程近似）。"""
from __future__ import annotations
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


def align(real: np.ndarray, fake: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if fake.shape[:2] != real.shape[:2]:
        fake = cv2.resize(fake, (real.shape[1], real.shape[0]),
                          interpolation=cv2.INTER_AREA)
    return real, fake


def diff_map(real: np.ndarray, fake: np.ndarray) -> tuple[np.ndarray, float]:
    """RGB L1 + (1-SSIM) 融合差异图，范围 [0,1]。"""
    r = real.astype(np.float32)
    f = fake.astype(np.float32)
    rgb = np.abs(r - f).mean(axis=2) / 255.0
    rg = cv2.cvtColor(real, cv2.COLOR_RGB2GRAY)
    fg = cv2.cvtColor(fake, cv2.COLOR_RGB2GRAY)
    score, smap = ssim(rg, fg, full=True, data_range=255)
    ssim_diff = np.clip(1.0 - smap, 0.0, 1.0)
    diff = np.clip(0.5 * rgb + 0.5 * ssim_diff, 0.0, 1.0).astype(np.float32)
    return diff, float(score)


def coarse_mask(diff: np.ndarray, thresh: float = 0.15) -> np.ndarray:
    return ((diff >= thresh).astype(np.uint8)) * 255


def refine(mask: np.ndarray) -> np.ndarray:
    k = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx((mask > 127).astype(np.uint8), cv2.MORPH_CLOSE, k)
    return (closed * 255).astype(np.uint8)


def connected_component_filter(mask: np.ndarray, min_frac: float = 0.005) -> np.ndarray:
    binm = (mask > 127).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binm, 8)
    out = np.zeros_like(mask)
    total = mask.size
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_frac * total:
            out[labels == i] = 255
    return out


def _boundary_sharpness(diff: np.ndarray, mask: np.ndarray) -> float:
    """边界处差异梯度均值，作为边界清晰度代理，归一化到 [0,1]。"""
    edges = cv2.Canny((mask > 127).astype(np.uint8) * 255, 50, 150)
    if edges.sum() == 0:
        return 0.0
    grad = cv2.magnitude(cv2.Sobel(diff, cv2.CV_32F, 1, 0),
                         cv2.Sobel(diff, cv2.CV_32F, 0, 1))
    return float(np.clip(grad[edges > 0].mean(), 0.0, 1.0))


def pseudo_mask(real: np.ndarray, fake: np.ndarray,
                thresh: float = 0.15) -> tuple[np.ndarray, dict]:
    real, fake = align(real, fake)
    diff, ssim_score = diff_map(real, fake)
    coarse = coarse_mask(diff, thresh)
    refined = refine(coarse)
    final = connected_component_filter(refined)
    fg = final > 127
    confidence = float(diff[fg].mean()) if fg.any() else 0.0
    metrics = {
        "confidence": confidence,
        "ssim_score": ssim_score,
        "boundary_sharpness": _boundary_sharpness(diff, final),
        "area_ratio": float(fg.sum()) / float(final.size),
    }
    return final, metrics
```

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/test_masks_pseudo_mask.py -q` → PASS

- [ ] **Step 5: 提交**

```bash
git add src/forgery_pipeline/masks/pseudo_mask.py tests/test_masks_pseudo_mask.py
git commit -m "feat(masks): 差分伪 mask 流程（diff/coarse/refine/cc + 指标）"
```

---

## Phase 3 — 去重、QC 与后处理

### Task 13: pHash 去重（`dedup.py`）

**Files:**
- Create: `src/forgery_pipeline/dedup.py`, `tests/test_dedup.py`

**Interfaces:**
- Produces:
  - `class PHashDeduper(hamming_threshold:int=5)`：`add(img:np.ndarray) -> bool`（新图返回 True 并记录；重复返回 False）、`is_duplicate(img) -> bool`

- [ ] **Step 1: 写失败测试** `tests/test_dedup.py`

```python
import numpy as np
from forgery_pipeline.dedup import PHashDeduper


def test_dedup_detects_repeat():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    d = PHashDeduper()
    assert d.add(img) is True          # 首次为新
    assert d.add(img.copy()) is False  # 重复
    assert d.is_duplicate(img) is True


def test_dedup_accepts_distinct():
    d = PHashDeduper()
    a = np.zeros((64, 64, 3), np.uint8)
    b = np.full((64, 64, 3), 255, np.uint8)
    assert d.add(a) is True
    assert d.add(b) is True
```

- [ ] **Step 2: 运行确认失败** → FAIL

- [ ] **Step 3: 实现 `dedup.py`**

```python
"""基于感知哈希（pHash）的近重复检测（报告 §4.3）。"""
from __future__ import annotations
import imagehash
import numpy as np
from PIL import Image as PILImage


class PHashDeduper:
    def __init__(self, hamming_threshold: int = 5):
        self.threshold = hamming_threshold
        self._hashes: list = []

    def _hash(self, img: np.ndarray):
        return imagehash.phash(PILImage.fromarray(img))

    def is_duplicate(self, img: np.ndarray) -> bool:
        h = self._hash(img)
        return any((h - prev) <= self.threshold for prev in self._hashes)

    def add(self, img: np.ndarray) -> bool:
        h = self._hash(img)
        if any((h - prev) <= self.threshold for prev in self._hashes):
            return False
        self._hashes.append(h)
        return True
```

- [ ] **Step 4: 运行确认通过** → PASS

- [ ] **Step 5: 提交**

```bash
git add src/forgery_pipeline/dedup.py tests/test_dedup.py
git commit -m "feat(dedup): pHash 近重复检测"
```

---

### Task 14: 图像质量过滤（`qc/image_qc.py`）

**Files:**
- Create: `src/forgery_pipeline/qc/image_qc.py`, `tests/test_qc_image.py`

**Interfaces:**
- Produces:
  - `check_image(img:np.ndarray, min_short_side:int=256, max_aspect:float=4.0) -> tuple[bool, list[str]]`

- [ ] **Step 1: 写失败测试** `tests/test_qc_image.py`

```python
import numpy as np
from forgery_pipeline.qc.image_qc import check_image


def _good():
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(300, 300, 3), dtype=np.uint8)


def test_good_image_passes():
    ok, reasons = check_image(_good())
    assert ok and reasons == []


def test_short_side_rejected():
    ok, reasons = check_image(np.zeros((100, 400, 3), np.uint8))
    assert not ok and any("短边" in r for r in reasons)


def test_solid_image_rejected():
    ok, reasons = check_image(np.full((300, 300, 3), 128, np.uint8))
    assert not ok and any("纯色" in r for r in reasons)


def test_extreme_aspect_rejected():
    img = np.random.default_rng(1).integers(0, 256, (300, 2000, 3), dtype=np.uint8)
    ok, reasons = check_image(img)
    assert not ok and any("长宽比" in r for r in reasons)
```

- [ ] **Step 2: 运行确认失败** → FAIL

- [ ] **Step 3: 实现 `qc/image_qc.py`**

```python
"""图像质量过滤（报告 §11.1）。"""
from __future__ import annotations
import cv2
import numpy as np


def check_image(img: np.ndarray, min_short_side: int = 256,
                max_aspect: float = 4.0) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if img is None or img.ndim != 3 or img.shape[2] != 3:
        return False, ["解码失败或非 RGB 三通道"]
    h, w = img.shape[:2]
    if min(h, w) < min_short_side:
        reasons.append(f"短边过小 (<{min_short_side})")
    if max(h, w) / max(min(h, w), 1) > max_aspect:
        reasons.append(f"极端长宽比 (>{max_aspect})")
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if float(gray.std()) < 3.0:
        reasons.append("大面积纯色/空白")
    return (len(reasons) == 0, reasons)
```

- [ ] **Step 4: 运行确认通过** → PASS

- [ ] **Step 5: 提交**

```bash
git add src/forgery_pipeline/qc/image_qc.py tests/test_qc_image.py
git commit -m "feat(qc): 图像质量过滤（分辨率/长宽比/纯色）"
```

---

### Task 15: mask 质量过滤（`qc/mask_qc.py`）

**Files:**
- Create: `src/forgery_pipeline/qc/mask_qc.py`, `tests/test_qc_mask.py`

**Interfaces:**
- Consumes（Task 11）：`candidates.area_ratio`
- Produces:
  - `num_components(mask) -> int`
  - `check_mask(mask:np.ndarray, max_components:int=15) -> tuple[bool, list[str]]`（面积比∈[0.01,0.50]、不过度碎片化、不覆盖整图）

- [ ] **Step 1: 写失败测试** `tests/test_qc_mask.py`

```python
import numpy as np
from forgery_pipeline.qc.mask_qc import check_mask


def test_valid_mask_passes():
    m = np.zeros((100, 100), np.uint8)
    m[30:60, 30:60] = 255  # 9% 面积，单连通
    ok, reasons = check_mask(m)
    assert ok and reasons == []


def test_too_small_rejected():
    m = np.zeros((100, 100), np.uint8)
    m[:2, :2] = 255  # 0.04%
    ok, reasons = check_mask(m)
    assert not ok and any("面积" in r for r in reasons)


def test_full_image_rejected():
    ok, reasons = check_mask(np.full((100, 100), 255, np.uint8))
    assert not ok


def test_fragmented_rejected():
    rng = np.random.default_rng(0)
    m = ((rng.random((100, 100)) < 0.06) * 255).astype(np.uint8)  # 大量散点
    ok, reasons = check_mask(m)
    assert not ok and any("碎片" in r for r in reasons)
```

- [ ] **Step 2: 运行确认失败** → FAIL

- [ ] **Step 3: 实现 `qc/mask_qc.py`**

```python
"""mask 质量过滤（报告 §11.2）。"""
from __future__ import annotations
import cv2
import numpy as np
from forgery_pipeline.masks.candidates import area_ratio


def num_components(mask: np.ndarray) -> int:
    n, _ = cv2.connectedComponents((mask > 127).astype(np.uint8), 8)
    return max(0, n - 1)  # 去掉背景


def check_mask(mask: np.ndarray, max_components: int = 15) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    r = area_ratio(mask)
    if not (0.01 <= r <= 0.50):
        reasons.append(f"面积比越界 ({r:.3f} ∉ [0.01, 0.50])")
    if r > 0.99:
        reasons.append("mask 覆盖整图")
    if num_components(mask) > max_components:
        reasons.append("mask 过度碎片化")
    return (len(reasons) == 0, reasons)
```

- [ ] **Step 4: 运行确认通过** → PASS

- [ ] **Step 5: 提交**

```bash
git add src/forgery_pipeline/qc/mask_qc.py tests/test_qc_mask.py
git commit -m "feat(qc): mask 质量过滤（面积比/碎片化/整图）"
```

---

### Task 16: 生成质量过滤（`qc/gen_qc.py`）

**Files:**
- Create: `src/forgery_pipeline/qc/gen_qc.py`, `tests/test_qc_gen.py`

**Interfaces:**
- Produces:
  - `check_generation(img:np.ndarray, prompt:str|None=None) -> tuple[bool, list[str], str]`，第三返回值为 `quality_bucket ∈ {high, mid, low}`

- [ ] **Step 1: 写失败测试** `tests/test_qc_gen.py`

```python
import numpy as np
from forgery_pipeline.qc.gen_qc import check_generation


def test_normal_image_ok_with_bucket():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(128, 128, 3), dtype=np.uint8)
    ok, reasons, bucket = check_generation(img, "a dog")
    assert ok and reasons == []
    assert bucket in {"high", "mid", "low"}


def test_solid_image_flagged_failure():
    ok, reasons, bucket = check_generation(np.full((128, 128, 3), 10, np.uint8))
    assert not ok and bucket == "low"
```

- [ ] **Step 2: 运行确认失败** → FAIL

- [ ] **Step 3: 实现 `qc/gen_qc.py`**

```python
"""生成质量过滤（报告 §11.3）。mock 用低层统计量近似；真实可接入美学/一致性模型。"""
from __future__ import annotations
import cv2
import numpy as np


def check_generation(img: np.ndarray, prompt: str | None = None
                     ) -> tuple[bool, list[str], str]:
    reasons: list[str] = []
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    std = float(gray.std())
    if std < 5.0:
        reasons.append("近乎纯色，疑似生成失败")
    bucket = "high" if std > 40 else "mid" if std > 15 else "low"
    # 真实后端可在此加入 prompt-图像一致性（如 CLIPScore）；mock 跳过。
    return (len(reasons) == 0, reasons, bucket)
```

- [ ] **Step 4: 运行确认通过** → PASS

- [ ] **Step 5: 提交**

```bash
git add src/forgery_pipeline/qc/gen_qc.py tests/test_qc_gen.py
git commit -m "feat(qc): 生成质量过滤与 quality_bucket"
```

---

### Task 17: QES 质量评分（`qc/quality_score.py`）

**Files:**
- Create: `src/forgery_pipeline/qc/quality_score.py`, `tests/test_qc_quality_score.py`

**Interfaces:**
- Produces:
  - `area_validity(area_ratio:float) -> float`（∈[0.01,0.50] 返回 1.0，否则 0.0）
  - `qes_score(confidence, boundary_sharpness, mask_consistency, semantic_consistency, area_validity) -> float`（权重 0.3/0.2/0.2/0.2/0.1）
  - `route_from_score(score:float) -> str`：`accept`(≥0.75)/`review`(0.60–0.75)/`reject`(<0.60)
  - `bucket_from_score(score:float) -> str`：`high`(≥0.75)/`mid`(0.60–0.75)/`low`(<0.60)

- [ ] **Step 1: 写失败测试** `tests/test_qc_quality_score.py`

```python
from forgery_pipeline.qc.quality_score import (
    qes_score, route_from_score, bucket_from_score, area_validity)


def test_weighted_sum():
    s = qes_score(1.0, 1.0, 1.0, 1.0, 1.0)
    assert abs(s - 1.0) < 1e-9
    s2 = qes_score(1.0, 0.0, 0.0, 0.0, 0.0)
    assert abs(s2 - 0.3) < 1e-9


def test_area_validity():
    assert area_validity(0.1) == 1.0
    assert area_validity(0.005) == 0.0
    assert area_validity(0.7) == 0.0


def test_routing_thresholds():
    assert route_from_score(0.80) == "accept"
    assert route_from_score(0.65) == "review"
    assert route_from_score(0.50) == "reject"
    assert bucket_from_score(0.80) == "high"
    assert bucket_from_score(0.65) == "mid"
    assert bucket_from_score(0.50) == "low"
```

- [ ] **Step 2: 运行确认失败** → FAIL

- [ ] **Step 3: 实现 `qc/quality_score.py`**

```python
"""QES-like 质量评分（报告 §7.5）。"""
from __future__ import annotations

WEIGHTS = {
    "confidence": 0.3,
    "boundary_sharpness": 0.2,
    "mask_consistency": 0.2,
    "semantic_consistency": 0.2,
    "area_validity": 0.1,
}


def area_validity(area_ratio: float) -> float:
    return 1.0 if 0.01 <= area_ratio <= 0.50 else 0.0


def qes_score(confidence: float, boundary_sharpness: float,
              mask_consistency: float, semantic_consistency: float,
              area_validity: float) -> float:
    return float(
        WEIGHTS["confidence"] * confidence
        + WEIGHTS["boundary_sharpness"] * boundary_sharpness
        + WEIGHTS["mask_consistency"] * mask_consistency
        + WEIGHTS["semantic_consistency"] * semantic_consistency
        + WEIGHTS["area_validity"] * area_validity
    )


def route_from_score(score: float) -> str:
    if score >= 0.75:
        return "accept"
    if score >= 0.60:
        return "review"
    return "reject"


def bucket_from_score(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.60:
        return "mid"
    return "low"
```

- [ ] **Step 4: 运行确认通过** → PASS

- [ ] **Step 5: 提交**

```bash
git add src/forgery_pipeline/qc/quality_score.py tests/test_qc_quality_score.py
git commit -m "feat(qc): QES 质量评分与分流/分桶"
```

---

### Task 18: 后处理退化（`postprocess/degradations.py`）

**Files:**
- Create: `src/forgery_pipeline/postprocess/degradations.py`, `tests/test_postprocess.py`

**Interfaces:**
- Consumes（Task 2）：`Postprocess`
- Produces（输入/输出图像同为 (H,W,3) uint8，**形状保持不变**以兼容 mask）：
  - `apply_jpeg(img, q:int) -> Image`
  - `apply_resize(img, scale:float) -> Image`（先缩放再还原至原尺寸，模拟重采样损失）
  - `apply_blur(img, k:int) -> Image`
  - `apply_noise(img, sigma:float, seed:int) -> Image`
  - `apply_social(img) -> Image`（强 JPEG + 轻重采样近似社媒转码）
  - `sample_and_apply(img, rng:np.random.Generator) -> tuple[Image, Postprocess]`（随机选一种退化并返回参数记录）

- [ ] **Step 1: 写失败测试** `tests/test_postprocess.py`

```python
import numpy as np
from forgery_pipeline.postprocess import degradations as dg
from forgery_pipeline.schema import Postprocess


def _img():
    return np.random.default_rng(0).integers(0, 256, (128, 128, 3), dtype=np.uint8)


def test_jpeg_changes_pixels_keeps_shape():
    img = _img()
    out = dg.apply_jpeg(img, 50)
    assert out.shape == img.shape and out.dtype == np.uint8
    assert not np.array_equal(out, img)


def test_resize_preserves_shape():
    img = _img()
    assert dg.apply_resize(img, 0.5).shape == img.shape


def test_noise_deterministic():
    img = _img()
    a = dg.apply_noise(img, 10, seed=3)
    b = dg.apply_noise(img, 10, seed=3)
    assert np.array_equal(a, b)


def test_sample_and_apply_records_one_param():
    img = _img()
    out, pp = dg.sample_and_apply(img, np.random.default_rng(1))
    assert out.shape == img.shape
    assert isinstance(pp, Postprocess)
    changed = [pp.jpeg_quality != "none", pp.resize != "none",
               pp.blur != "none", pp.noise != "none"]
    assert sum(changed) >= 1
```

- [ ] **Step 2: 运行确认失败** → FAIL

- [ ] **Step 3: 实现 `postprocess/degradations.py`**

```python
"""后处理退化增强（报告 §10）。保持图像尺寸不变以兼容 mask；参数写入 Postprocess。"""
from __future__ import annotations
import cv2
import numpy as np
from forgery_pipeline.schema import Postprocess

JPEG_QUALITIES = [50, 60, 70, 80, 90, 95]
RESIZE_SCALES = [0.5, 0.67, 0.75, 1.5]
BLUR_KERNELS = [3, 5]
NOISE_SIGMAS = [3, 5, 10]


def apply_jpeg(img: np.ndarray, q: int) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                           [int(cv2.IMWRITE_JPEG_QUALITY), int(q)])
    dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)


def apply_resize(img: np.ndarray, scale: float) -> np.ndarray:
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                       interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def apply_blur(img: np.ndarray, k: int) -> np.ndarray:
    k = max(3, k | 1)
    return cv2.GaussianBlur(img, (k, k), 0)


def apply_noise(img: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def apply_social(img: np.ndarray) -> np.ndarray:
    return apply_jpeg(apply_resize(img, 0.75), 60)


def sample_and_apply(img: np.ndarray, rng: np.random.Generator
                     ) -> tuple[np.ndarray, Postprocess]:
    kind = rng.choice(["jpeg", "resize", "blur", "noise"])
    pp = Postprocess()
    if kind == "jpeg":
        q = int(rng.choice(JPEG_QUALITIES))
        img, pp.jpeg_quality = apply_jpeg(img, q), q
    elif kind == "resize":
        s = float(rng.choice(RESIZE_SCALES))
        img, pp.resize = apply_resize(img, s), str(s)
    elif kind == "blur":
        k = int(rng.choice(BLUR_KERNELS))
        img, pp.blur = apply_blur(img, k), f"k{k}"
    else:
        sigma = int(rng.choice(NOISE_SIGMAS))
        seed = int(rng.integers(0, 2**31 - 1))
        img, pp.noise = apply_noise(img, sigma, seed), f"sigma{sigma}"
    return img, pp
```

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/test_postprocess.py -q` → PASS

- [ ] **Step 5: 提交**

```bash
git add src/forgery_pipeline/postprocess/degradations.py tests/test_postprocess.py
git commit -m "feat(postprocess): JPEG/resize/blur/noise/social 退化与参数记录"
```

---

## Phase 4 — 五个子库 builder

> 所有 builder 用显式参数（而非整个 cfg）以便单测；`pipeline.py`（Task 27）负责把 `PipelineConfig` 映射到这些调用。图像路径在 manifest 中是相对 `out_dir` 的相对路径。

### Task 19: 图像 IO 辅助 + D0 真实图像池（`image_io.py` + `builders/d0_real.py`）

**Files:**
- Create: `src/forgery_pipeline/image_io.py`, `src/forgery_pipeline/builders/d0_real.py`
- Create: `tests/test_image_io.py`, `tests/test_builder_d0.py`

**Interfaces:**
- Produces（`image_io`）：`save_image(img, path)`, `save_mask(mask, path)`, `load_image(path)->np.ndarray(HxWx3 uint8 RGB)`, `load_mask(path)->np.ndarray(HxW uint8)`
- Produces（`d0_real`）：`build_d0(out_dir, n:int, backend:str="mock", seed:int=0) -> list[Sample]`（is_fake=0, task_type=real_pristine；保存到 `D0_real_pristine/`）

- [ ] **Step 1: 写失败测试** `tests/test_image_io.py`

```python
import numpy as np
from forgery_pipeline import image_io


def test_image_roundtrip(tmp_path):
    img = np.random.default_rng(0).integers(0, 256, (32, 48, 3), dtype=np.uint8)
    p = tmp_path / "a.png"
    image_io.save_image(img, p)
    got = image_io.load_image(p)
    assert got.shape == img.shape and np.array_equal(got, img)


def test_mask_roundtrip(tmp_path):
    m = np.zeros((20, 20), np.uint8)
    m[5:15, 5:15] = 255
    p = tmp_path / "m.png"
    image_io.save_mask(m, p)
    got = image_io.load_mask(p)
    assert got.shape == (20, 20) and set(np.unique(got)).issubset({0, 255})
```

`tests/test_builder_d0.py`:

```python
from pathlib import Path
from forgery_pipeline.builders.d0_real import build_d0


def test_build_d0_produces_real_samples(tmp_path):
    samples = build_d0(tmp_path, n=5, seed=0)
    assert len(samples) == 5
    for s in samples:
        assert s.is_fake == 0
        assert s.task_type.value == "real_pristine"
        assert (Path(tmp_path) / s.image_path).exists()
        assert s.manipulation_level1 is None
```

- [ ] **Step 2: 运行确认失败** → FAIL

- [ ] **Step 3: 实现 `image_io.py`**

```python
"""图像/掩码的磁盘读写。"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import Image as PILImage


def save_image(img: np.ndarray, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    PILImage.fromarray(img).save(path)


def save_mask(mask: np.ndarray, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    PILImage.fromarray(mask).save(path)


def load_image(path) -> np.ndarray:
    return np.asarray(PILImage.open(path).convert("RGB"), dtype=np.uint8)


def load_mask(path) -> np.ndarray:
    return np.asarray(PILImage.open(path).convert("L"), dtype=np.uint8)
```

- [ ] **Step 4: 实现 `builders/d0_real.py`**

```python
"""D0 真实图像池：摄取 → 清洗 → 去重（报告 §4）。"""
from __future__ import annotations
from pathlib import Path
from forgery_pipeline import image_io, ids
from forgery_pipeline.backends import registry
from forgery_pipeline.dedup import PHashDeduper
from forgery_pipeline.qc.image_qc import check_image
from forgery_pipeline.schema import Sample, TaskType


def build_d0(out_dir, n: int, backend: str = "mock", seed: int = 0) -> list[Sample]:
    out_dir = Path(out_dir)
    src = registry.get_image_source(backend, seed=seed)
    dedup = PHashDeduper()
    samples: list[Sample] = []
    for img, meta in src.iter_images(n * 3):  # 过采样以容忍 QC 丢弃
        if len(samples) >= n:
            break
        ok, _ = check_image(img)
        if not ok or not dedup.add(img):
            continue
        iid = ids.make_image_id("real", ids.content_hash(img))
        rel = f"D0_real_pristine/{iid}.jpg"
        image_io.save_image(img, out_dir / rel)
        samples.append(Sample(
            image_id=iid, image_path=rel, is_fake=0,
            task_type=TaskType.real_pristine,
            source_dataset=meta.get("source_dataset"),
            license=meta.get("license"),
        ))
    return samples
```

- [ ] **Step 5: 运行确认通过**

Run: `pytest tests/test_image_io.py tests/test_builder_d0.py -q` → PASS

- [ ] **Step 6: 提交**

```bash
git add src/forgery_pipeline/image_io.py src/forgery_pipeline/builders/d0_real.py tests/test_image_io.py tests/test_builder_d0.py
git commit -m "feat(builders): D0 真实图像池 + 图像 IO 辅助"
```

---

### Task 20: D1 整图生成（`builders/d1_whole.py`）

**Files:**
- Create: `src/forgery_pipeline/builders/d1_whole.py`, `tests/test_builder_d1.py`

**Interfaces:**
- Consumes：`config.GeneratorSpec`, `registry.get_whole_generator`, `qc.gen_qc.check_generation`
- Produces：`build_d1(out_dir, generators:list[GeneratorSpec], per_generator:int, backend="mock", seed=0) -> list[Sample]`（is_fake=1, level1=whole_generated, level2=family）

- [ ] **Step 1: 写失败测试** `tests/test_builder_d1.py`

```python
from pathlib import Path
from forgery_pipeline.config import GeneratorSpec
from forgery_pipeline.builders.d1_whole import build_d1


def test_build_d1_multi_generator(tmp_path):
    gens = [GeneratorSpec("stable-diffusion-xl", "diffusion", "txt2img"),
            GeneratorSpec("stylegan2", "GAN", "unconditional")]
    samples = build_d1(tmp_path, gens, per_generator=2, seed=0)
    assert len(samples) == 4
    fams = {s.generator_family for s in samples}
    assert fams == {"diffusion", "GAN"}
    for s in samples:
        assert s.is_fake == 1
        assert s.manipulation_level1 == "whole_generated"
        assert s.manipulation_level2 in {"diffusion", "GAN", "autoregressive"}
        assert (Path(tmp_path) / s.image_path).exists()
        assert s.prompt and s.seed is not None
```

- [ ] **Step 2: 运行确认失败** → FAIL

- [ ] **Step 3: 实现 `builders/d1_whole.py`**

```python
"""D1 整图 AIGC 生成：强调生成器多样性（报告 §5，Community Forensics）。"""
from __future__ import annotations
from pathlib import Path
from forgery_pipeline import image_io, ids
from forgery_pipeline.backends import registry
from forgery_pipeline.config import GeneratorSpec
from forgery_pipeline.qc.gen_qc import check_generation
from forgery_pipeline.schema import Sample, TaskType

PROMPTS = [
    "A realistic photo of a dog running on the beach at sunset.",
    "A news-style photo of a crowded street after heavy rain.",
    "A product photography image of a black backpack on a white background.",
    "A portrait of a smiling person in soft natural light.",
    "A landscape photo of mountains under a clear blue sky.",
]
_FAMILY_TO_L2 = {"diffusion": "diffusion", "GAN": "GAN",
                 "autoregressive": "autoregressive"}


def build_d1(out_dir, generators: list[GeneratorSpec], per_generator: int,
             backend: str = "mock", seed: int = 0) -> list[Sample]:
    out_dir = Path(out_dir)
    samples: list[Sample] = []
    for gi, gen in enumerate(generators):
        g = registry.get_whole_generator(backend, gen.name, gen.family)
        for j in range(per_generator):
            s = seed + gi * 1000 + j
            prompt = PROMPTS[(gi + j) % len(PROMPTS)]
            img, meta = g.generate(prompt, {"seed": s})
            ok, _, bucket = check_generation(img, prompt)
            if not ok:
                continue
            iid = ids.make_image_id("whole_gen", f"{gen.name}-{s}-{prompt}")
            rel = f"D1_whole_generated/{iid}.png"
            image_io.save_image(img, out_dir / rel)
            samples.append(Sample(
                image_id=iid, image_path=rel, is_fake=1,
                task_type=TaskType.whole_image_detection,
                manipulation_level1="whole_generated",
                manipulation_level2=_FAMILY_TO_L2.get(gen.family, "diffusion"),
                manipulation_level4=gen.name,
                generator_name=gen.name, generator_family=gen.family,
                prompt=prompt, seed=meta["seed"], sampler=meta["sampler"],
                steps=meta["steps"], cfg_scale=meta["cfg_scale"],
                quality_bucket=bucket,
            ))
    return samples
```

- [ ] **Step 4: 运行确认通过** → PASS

- [ ] **Step 5: 提交**

```bash
git add src/forgery_pipeline/builders/d1_whole.py tests/test_builder_d1.py
git commit -m "feat(builders): D1 整图生成（多生成器，层级标签）"
```

---

### Task 21: D2 局部篡改（`builders/d2_local.py`）

**Files:**
- Create: `src/forgery_pipeline/builders/d2_local.py`, `tests/test_builder_d2.py`

**Interfaces:**
- Consumes：`registry.get_segmenter/get_inpainter`, `masks.candidates.filter_and_sample/area_ratio`, `masks.morphology.make_irregular`, `qc.mask_qc.check_mask`, D0 的 `base_samples`
- Produces：`build_d2(out_dir, base_samples:list[Sample], n:int, inpainters:list[GeneratorSpec], backend="mock", seed=0) -> list[Sample]`（is_fake=1, level1=partial_manipulated, level2=AIGC-editing, 必有 mask）
- 覆盖 7 类篡改：`MANIP_TYPES`（name, level3, prompt 模板）

- [ ] **Step 1: 写失败测试** `tests/test_builder_d2.py`

```python
from pathlib import Path
from forgery_pipeline.config import GeneratorSpec
from forgery_pipeline.builders.d0_real import build_d0
from forgery_pipeline.builders.d2_local import build_d2, MANIP_TYPES


def test_build_d2_localization(tmp_path):
    base = build_d0(tmp_path, n=6, seed=0)
    inps = [GeneratorSpec("stable-diffusion-inpaint", "diffusion", "inpaint")]
    samples = build_d2(tmp_path, base, n=4, inpainters=inps, seed=0)
    assert len(samples) == 4
    for s in samples:
        assert s.is_fake == 1
        assert s.task_type.value == "localization"
        assert s.manipulation_level1 == "partial_manipulated"
        assert s.mask_path and (Path(tmp_path) / s.mask_path).exists()
        assert 0.01 <= s.mask_area_ratio <= 0.50
        assert s.real_image_path is not None


def test_manip_types_cover_seven():
    assert len(MANIP_TYPES) == 7
```

- [ ] **Step 2: 运行确认失败** → FAIL

- [ ] **Step 3: 实现 `builders/d2_local.py`**

```python
"""D2 局部 AIGC 篡改：mask → prompt → inpaint（报告 §6，借鉴 GIM）。"""
from __future__ import annotations
from pathlib import Path
from forgery_pipeline import image_io, ids
from forgery_pipeline.backends import registry
from forgery_pipeline.config import GeneratorSpec
from forgery_pipeline.masks.candidates import filter_and_sample, area_ratio
from forgery_pipeline.masks import morphology
from forgery_pipeline.qc.mask_qc import check_mask
from forgery_pipeline.schema import Sample, TaskType

# (篡改类型, level3, 编辑 prompt 模板)；level3 取自 LEVEL3 合法值
MANIP_TYPES = [
    ("object_insertion", "mask_guided_inpainting",
     "Insert a new realistic object into the masked region."),
    ("object_replacement", "object_replacement",
     "Replace the object in the masked region with a different realistic object."),
    ("object_removal", "object_removal",
     "Remove the object in the masked region and fill the background naturally."),
    ("attribute_editing", "text_guided_editing",
     "Change the color or attribute of the object in the masked region."),
    ("background_editing", "image_guided_editing",
     "Repaint the background within the masked region."),
    ("text_editing", "text_editing",
     "Modify the text content within the masked region."),
    ("face_editing", "face_swap",
     "Edit the face in the masked region (expression/glasses/hair)."),
]


def build_d2(out_dir, base_samples: list[Sample], n: int,
             inpainters: list[GeneratorSpec], backend: str = "mock",
             seed: int = 0) -> list[Sample]:
    out_dir = Path(out_dir)
    seg = registry.get_segmenter(backend, seed=seed)
    samples: list[Sample] = []
    attempts = 0
    max_attempts = max(n * 8, 8)
    while len(samples) < n and base_samples and attempts < max_attempts:
        base = base_samples[attempts % len(base_samples)]
        attempts += 1
        img = image_io.load_image(out_dir / base.image_path)
        valid = filter_and_sample(seg.propose_masks(img, 6))
        if not valid:
            continue
        mask = morphology.make_irregular(valid[len(samples) % len(valid)][0],
                                         seed=seed + attempts)
        ok, _ = check_mask(mask)
        if not ok:
            continue
        ratio = area_ratio(mask)
        mtype, level3, tmpl = MANIP_TYPES[len(samples) % len(MANIP_TYPES)]
        inp = inpainters[len(samples) % len(inpainters)]
        painter = registry.get_inpainter(backend, inp.name, inp.family)
        s = seed + attempts
        fake, _ = painter.inpaint(img, mask, tmpl, {"seed": s})
        iid = ids.make_image_id("local_edit", f"{base.image_id}-{mtype}-{s}")
        img_rel = f"D2_local_aigc_edit/{iid}.jpg"
        mask_rel = f"D2_local_aigc_edit/masks/{iid}.png"
        image_io.save_image(fake, out_dir / img_rel)
        image_io.save_mask(mask, out_dir / mask_rel)
        samples.append(Sample(
            image_id=iid, image_path=img_rel,
            real_image_path=base.image_path, mask_path=mask_rel, is_fake=1,
            task_type=TaskType.localization,
            manipulation_level1="partial_manipulated",
            manipulation_level2="AIGC-editing",
            manipulation_level3=level3, manipulation_level4=inp.name,
            generator_name=inp.name, generator_family=inp.family,
            mask_source="SAM", mask_area_ratio=ratio, prompt=tmpl, seed=s,
            source_dataset=base.source_dataset,
        ))
    return samples
```

- [ ] **Step 4: 运行确认通过** → PASS

- [ ] **Step 5: 提交**

```bash
git add src/forgery_pipeline/builders/d2_local.py tests/test_builder_d2.py
git commit -m "feat(builders): D2 局部篡改（候选mask→形态学→inpaint→QC，7类）"
```

---

### Task 22: D3 网页人工篡改伪标注（`builders/d3_web.py`）

**Files:**
- Create: `src/forgery_pipeline/builders/d3_web.py`, `tests/test_builder_d3.py`

**Interfaces:**
- Consumes：`masks.pseudo_mask.pseudo_mask`, `qc.mask_qc.check_mask`, `qc.quality_score.*`，D0 的 `base_samples`
- Produces：`build_d3(out_dir, base_samples, n, backend="mock", seed=0) -> list[Sample]`（level1=partial_manipulated，level2∈{splicing,copy-move,removal}，mask_source="diff"，带 quality_score，route≠reject 才入库）

> mock 用 D0 真实图合成「人工篡改 pair」（copy-move 拼贴），流水线**不使用**真值，只通过 `pseudo_mask` 差分还原 mask，演示 MIML 自动标注流程。

- [ ] **Step 1: 写失败测试** `tests/test_builder_d3.py`

```python
from pathlib import Path
from forgery_pipeline.builders.d0_real import build_d0
from forgery_pipeline.builders.d3_web import build_d3


def test_build_d3_pseudo_mask_localization(tmp_path):
    base = build_d0(tmp_path, n=8, seed=0)
    samples = build_d3(tmp_path, base, n=4, seed=0)
    assert len(samples) >= 1  # 部分可能被 QES 过滤
    for s in samples:
        assert s.manipulation_level1 == "partial_manipulated"
        assert s.mask_source == "diff"
        assert s.mask_path and (Path(tmp_path) / s.mask_path).exists()
        assert s.quality_score is not None and s.quality_score >= 0.60
        assert s.task_type.value == "localization"
```

- [ ] **Step 2: 运行确认失败** → FAIL

- [ ] **Step 3: 实现 `builders/d3_web.py`**

```python
"""D3 网页人工篡改：real-fake pair → 差分伪 mask → QES（报告 §7，借鉴 MIML）。"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from forgery_pipeline import image_io, ids
from forgery_pipeline.masks.pseudo_mask import pseudo_mask
from forgery_pipeline.qc.mask_qc import check_mask
from forgery_pipeline.qc.quality_score import (
    qes_score, route_from_score, bucket_from_score, area_validity)
from forgery_pipeline.schema import Sample, TaskType

_L2 = ["splicing", "copy-move", "removal"]
_L3 = {"splicing": "image_guided_editing", "copy-move": "image_guided_editing",
       "removal": "object_removal"}


def _synthesize_web_fake(real: np.ndarray, seed: int) -> np.ndarray:
    """模拟人工拼贴（copy-move）：把一块区域复制到另一处，制造已知篡改。"""
    rng = np.random.default_rng(seed)
    h, w = real.shape[:2]
    bh, bw = h // 4, w // 4
    sy, sx = int(rng.integers(0, h - bh)), int(rng.integers(0, w - bw))
    ty, tx = int(rng.integers(0, h - bh)), int(rng.integers(0, w - bw))
    fake = real.copy()
    fake[ty:ty + bh, tx:tx + bw] = real[sy:sy + bh, sx:sx + bw]
    return fake


def build_d3(out_dir, base_samples: list[Sample], n: int,
             backend: str = "mock", seed: int = 0) -> list[Sample]:
    out_dir = Path(out_dir)
    samples: list[Sample] = []
    attempts = 0
    max_attempts = max(n * 8, 8)
    while len(samples) < n and base_samples and attempts < max_attempts:
        base = base_samples[attempts % len(base_samples)]
        attempts += 1
        real = image_io.load_image(out_dir / base.image_path)
        fake = _synthesize_web_fake(real, seed + attempts)
        mask, metrics = pseudo_mask(real, fake, thresh=0.1)
        ok, _ = check_mask(mask)
        if not ok:
            continue
        ratio = metrics["area_ratio"]
        score = qes_score(
            confidence=min(metrics["confidence"], 1.0),
            boundary_sharpness=metrics["boundary_sharpness"],
            mask_consistency=1.0 if 0.01 <= ratio <= 0.50 else 0.5,
            semantic_consistency=0.8,
            area_validity=area_validity(ratio),
        )
        if route_from_score(score) == "reject":
            continue
        l2 = _L2[len(samples) % len(_L2)]
        iid = ids.make_image_id("web_forgery", f"{base.image_id}-{attempts}")
        img_rel = f"D3_web_human_forgery/{iid}.jpg"
        mask_rel = f"D3_web_human_forgery/masks/{iid}.png"
        image_io.save_image(fake, out_dir / img_rel)
        image_io.save_mask(mask, out_dir / mask_rel)
        samples.append(Sample(
            image_id=iid, image_path=img_rel,
            real_image_path=base.image_path, mask_path=mask_rel, is_fake=1,
            task_type=TaskType.localization,
            manipulation_level1="partial_manipulated",
            manipulation_level2=l2, manipulation_level3=_L3[l2],
            generator_name="manual-web-edit", generator_family="editing",
            mask_source="diff", mask_area_ratio=ratio,
            quality_score=round(score, 4), quality_bucket=bucket_from_score(score),
            source_dataset=base.source_dataset,
        ))
    return samples
```

- [ ] **Step 4: 运行确认通过** → PASS

- [ ] **Step 5: 提交**

```bash
git add src/forgery_pipeline/builders/d3_web.py tests/test_builder_d3.py
git commit -m "feat(builders): D3 网页人工篡改伪标注（差分→伪mask→QES）"
```

---

### Task 23: D4 可解释子集（`builders/d4_explain.py`）

**Files:**
- Create: `src/forgery_pipeline/builders/d4_explain.py`, `tests/test_builder_d4.py`

**Interfaces:**
- Consumes：`registry.get_explainer`，带 mask 的源样本（D2/D3）
- Produces：`build_d4(out_dir, source_samples:list[Sample], n:int, backend="mock") -> list[Sample]`（task_type=explainable，携带 `explanation`，复用源 image/mask 与层级标签）

- [ ] **Step 1: 写失败测试** `tests/test_builder_d4.py`

```python
from forgery_pipeline.config import GeneratorSpec
from forgery_pipeline.builders.d0_real import build_d0
from forgery_pipeline.builders.d2_local import build_d2
from forgery_pipeline.builders.d4_explain import build_d4


def test_build_d4_explanations(tmp_path):
    base = build_d0(tmp_path, n=6, seed=0)
    inps = [GeneratorSpec("stable-diffusion-inpaint", "diffusion", "inpaint")]
    d2 = build_d2(tmp_path, base, n=4, inpainters=inps, seed=0)
    d4 = build_d4(tmp_path, d2, n=3)
    assert len(d4) == 3
    for s in d4:
        assert s.task_type.value == "explainable"
        assert s.explanation is not None
        assert s.explanation.forensic_conclusion
        assert s.mask_path is not None
```

- [ ] **Step 2: 运行确认失败** → FAIL

- [ ] **Step 3: 实现 `builders/d4_explain.py`**

```python
"""D4 可解释取证子集：image+mask → MLLM 文本解释（报告 §8，借鉴 FakeShield）。"""
from __future__ import annotations
from pathlib import Path
from forgery_pipeline import image_io, ids
from forgery_pipeline.backends import registry
from forgery_pipeline.schema import Sample, TaskType


def build_d4(out_dir, source_samples: list[Sample], n: int,
             backend: str = "mock") -> list[Sample]:
    out_dir = Path(out_dir)
    explainer = registry.get_explainer(backend)
    cands = [s for s in source_samples if s.mask_path][:n]
    samples: list[Sample] = []
    for s in cands:
        img = image_io.load_image(out_dir / s.image_path)
        mask = image_io.load_mask(out_dir / s.mask_path)
        expl = explainer.explain(
            img, mask,
            {"manipulation_level3": s.manipulation_level3 or "local AIGC inpainting"})
        iid = ids.make_image_id("explain", s.image_id)
        samples.append(Sample(
            image_id=iid, image_path=s.image_path,
            real_image_path=s.real_image_path, mask_path=s.mask_path, is_fake=1,
            task_type=TaskType.explainable,
            manipulation_level1=s.manipulation_level1,
            manipulation_level2=s.manipulation_level2,
            manipulation_level3=s.manipulation_level3,
            manipulation_level4=s.manipulation_level4,
            generator_name=s.generator_name, generator_family=s.generator_family,
            mask_source=s.mask_source, mask_area_ratio=s.mask_area_ratio,
            explanation=expl,
        ))
    return samples
```

- [ ] **Step 4: 运行确认通过** → PASS

- [ ] **Step 5: 提交**

```bash
git add src/forgery_pipeline/builders/d4_explain.py tests/test_builder_d4.py
git commit -m "feat(builders): D4 可解释子集（image-mask-description 三元组）"
```

---

## Phase 5 — 划分、编排、CLI 与端到端

### Task 24: 分组键（`split/grouping.py`）

**Files:**
- Create: `src/forgery_pipeline/split/grouping.py`, `tests/test_split_grouping.py`

**Interfaces:**
- Produces:
  - `origin_key(s:Sample) -> str`：取 `real_image_path or image_path` 的文件名 stem，使同一真实底图的 D0 基图与其 D2/D3 衍生样本归入同一组。
  - `is_degraded(pp:Postprocess) -> bool`

- [ ] **Step 1: 写失败测试** `tests/test_split_grouping.py`

```python
from forgery_pipeline.schema import Sample, TaskType, Postprocess
from forgery_pipeline.split.grouping import origin_key, is_degraded


def test_origin_links_base_and_edit():
    base = Sample(image_id="real_abc", image_path="D0_real_pristine/real_abc.jpg",
                  is_fake=0, task_type=TaskType.real_pristine)
    edit = Sample(image_id="local_x", image_path="D2_local_aigc_edit/local_x.jpg",
                  real_image_path="D0_real_pristine/real_abc.jpg",
                  mask_path="D2_local_aigc_edit/masks/local_x.png", is_fake=1,
                  task_type=TaskType.localization,
                  manipulation_level1="partial_manipulated",
                  manipulation_level2="AIGC-editing",
                  manipulation_level3="object_replacement")
    assert origin_key(base) == origin_key(edit) == "real_abc"


def test_is_degraded():
    assert is_degraded(Postprocess(jpeg_quality=70)) is True
    assert is_degraded(Postprocess(noise="sigma5")) is True
    assert is_degraded(Postprocess()) is False
```

- [ ] **Step 2: 运行确认失败** → FAIL

- [ ] **Step 3: 实现 `split/grouping.py`**

```python
"""划分分组键（报告 §12.1：按原图 ID 分组，避免同源泄漏）。"""
from __future__ import annotations
from pathlib import Path
from forgery_pipeline.schema import Sample, Postprocess


def origin_key(s: Sample) -> str:
    ref = s.real_image_path or s.image_path
    return Path(ref).stem


def is_degraded(pp: Postprocess) -> bool:
    return (pp.jpeg_quality != "none" or pp.resize != "none"
            or pp.blur != "none" or pp.noise != "none")
```

- [ ] **Step 4: 运行确认通过** → PASS

- [ ] **Step 5: 提交**

```bash
git add src/forgery_pipeline/split/grouping.py tests/test_split_grouping.py
git commit -m "feat(split): 原图分组键与退化判定"
```

---

### Task 25: 泄漏检查（`split/leakage.py`）

**Files:**
- Create: `src/forgery_pipeline/split/leakage.py`, `tests/test_split_leakage.py`

**Interfaces:**
- Consumes（Task 24）：`origin_key`
- Produces:
  - `BENCHMARK_SOURCES: set[str]`
  - `check_leakage(samples:list[Sample]) -> list[str]`（报告 §11.4 五条规则；空列表=无泄漏）

- [ ] **Step 1: 写失败测试** `tests/test_split_leakage.py`

```python
from forgery_pipeline.schema import Sample, TaskType
from forgery_pipeline.split.leakage import check_leakage


def _fake(iid, real, split, gen="sd", prompt="p", seed=1):
    return Sample(image_id=iid, image_path=f"D2/{iid}.jpg", real_image_path=real,
                  mask_path=f"D2/m/{iid}.png", is_fake=1,
                  task_type=TaskType.localization,
                  manipulation_level1="partial_manipulated",
                  manipulation_level2="AIGC-editing",
                  manipulation_level3="object_replacement",
                  generator_name=gen, prompt=prompt, seed=seed, split=split)


def test_clean_split_has_no_leak():
    a = _fake("a", "D0/real_1.jpg", "train", seed=1)
    b = _fake("b", "D0/real_2.jpg", "test_a", seed=2)
    assert check_leakage([a, b]) == []


def test_same_origin_train_and_test_flagged():
    a = _fake("a", "D0/real_1.jpg", "train", seed=1)
    b = _fake("b", "D0/real_1.jpg", "test_a", seed=2)  # 同原图跨 train/test
    errs = check_leakage([a, b])
    assert any("原图" in e for e in errs)


def test_cross_generator_generator_in_train_flagged():
    a = _fake("a", "D0/r1.jpg", "train", gen="ideogram", seed=1)
    b = _fake("b", "D0/r2.jpg", "test_b", gen="ideogram", seed=2)
    errs = check_leakage([a, b])
    assert any("生成器" in e for e in errs)
```

- [ ] **Step 2: 运行确认失败** → FAIL

- [ ] **Step 3: 实现 `split/leakage.py`**

```python
"""数据泄漏检查（报告 §11.4）。"""
from __future__ import annotations
from collections import defaultdict
from forgery_pipeline.schema import Sample
from forgery_pipeline.split.grouping import origin_key

# 公开取证 benchmark，不得混入训练集
BENCHMARK_SOURCES = {"CASIA", "Columbia", "Coverage", "NIST16", "DSO-1", "IMD2020"}


def check_leakage(samples: list[Sample]) -> list[str]:
    errs: list[str] = []

    # 规则 1&2：train 的原图不得出现在任何非 train split（同源/压缩版本泄漏）
    train_o = {origin_key(s) for s in samples if s.split == "train"}
    other_o = {origin_key(s) for s in samples if s.split and s.split != "train"}
    shared = train_o & other_o
    if shared:
        errs.append(f"原图跨越 train 与非 train: {sorted(shared)[:3]}")

    # 规则 3：同一 prompt+seed 不得跨 train/非 train
    ps = defaultdict(set)
    for s in samples:
        if s.split and s.prompt is not None and s.seed is not None:
            ps[(s.prompt, s.seed)].add("train" if s.split == "train" else "other")
    if any("train" in v and "other" in v for v in ps.values()):
        errs.append("存在 prompt+seed 跨 train/非 train")

    # 规则 4：cross-generator 测试集（test_b）的生成器不得出现在 train
    train_gen = {s.generator_name for s in samples
                 if s.split == "train" and s.generator_name}
    tb_gen = {s.generator_name for s in samples
              if s.split == "test_b" and s.generator_name}
    if train_gen & tb_gen:
        errs.append(f"cross-generator 生成器出现在 train: {sorted(train_gen & tb_gen)}")

    # 规则 5：公开 benchmark 不得混入 train
    bench = {s.source_dataset for s in samples
             if s.split == "train" and s.source_dataset in BENCHMARK_SOURCES}
    if bench:
        errs.append(f"benchmark 数据混入 train: {sorted(bench)}")

    return errs
```

- [ ] **Step 4: 运行确认通过** → PASS

- [ ] **Step 5: 提交**

```bash
git add src/forgery_pipeline/split/leakage.py tests/test_split_leakage.py
git commit -m "feat(split): 五条数据泄漏检查"
```

---

### Task 26: 8-way 划分（`split/splitter.py`）

**Files:**
- Create: `src/forgery_pipeline/split/splitter.py`, `tests/test_split_splitter.py`

**Interfaces:**
- Consumes：`origin_key`, `is_degraded`
- Produces:
  - `SPLITS: list[str]`（8 个）
  - `assign_splits(samples, holdout_generators, holdout_manipulation, holdout_domains=("Places",), seed=0) -> list[Sample]`（就地设置 `.split` 并返回）
- 路由（按原图组，整组同一 split）：含 holdout 生成器→`test_b`；含 holdout 篡改→`test_c`；来源域∈holdout→`test_d`；纯真实组→哈希分到 `train/val/test_f`；含 fake 组→哈希分到 `train/val/test_a`；最后把落在 `test_a` 的退化 fake 改判 `test_e`。

- [ ] **Step 1: 写失败测试** `tests/test_split_splitter.py`

```python
from forgery_pipeline.schema import Sample, TaskType, Postprocess
from forgery_pipeline.split.splitter import assign_splits, SPLITS
from forgery_pipeline.split.leakage import check_leakage


def _real(i, ds="COCO"):
    return Sample(image_id=f"real_{i}", image_path=f"D0/real_{i}.jpg",
                  is_fake=0, task_type=TaskType.real_pristine, source_dataset=ds)


def _fake(i, real_i, gen="sd", manip="object_replacement", ds="COCO", pp=None):
    return Sample(image_id=f"f_{i}", image_path=f"D2/f_{i}.jpg",
                  real_image_path=f"D0/real_{real_i}.jpg",
                  mask_path=f"D2/m/f_{i}.png", is_fake=1,
                  task_type=TaskType.localization,
                  manipulation_level1="partial_manipulated",
                  manipulation_level2="AIGC-editing", manipulation_level3=manip,
                  generator_name=gen, source_dataset=ds, seed=i,
                  postprocess=pp or Postprocess())


def test_holdout_routing():
    s_b = _fake(1, 1, gen="ideogram")
    s_c = _fake(2, 2, manip="text_editing")
    s_d = _real(3, ds="Places")
    assign_splits([s_b, s_c, s_d], holdout_generators=["ideogram"],
                  holdout_manipulation=["text_editing"], holdout_domains=["Places"])
    assert s_b.split == "test_b"
    assert s_c.split == "test_c"
    assert s_d.split == "test_d"


def test_degraded_testa_becomes_teste():
    # 构造多个纯 fake 原图，至少一个落到 test_a，且退化
    samples = [_fake(i, i, pp=Postprocess(jpeg_quality=70)) for i in range(40)]
    assign_splits(samples, holdout_generators=[], holdout_manipulation=[],
                  holdout_domains=[])
    assert any(s.split == "test_e" for s in samples)
    assert all(s.split in SPLITS for s in samples)


def test_no_leakage_after_split():
    samples = [_real(i) for i in range(20)] + [_fake(i, i) for i in range(20, 40)]
    assign_splits(samples, holdout_generators=["ideogram"],
                  holdout_manipulation=["text_editing"], holdout_domains=["Places"])
    assert check_leakage(samples) == []
```

- [ ] **Step 2: 运行确认失败** → FAIL

- [ ] **Step 3: 实现 `split/splitter.py`**

```python
"""8-way 数据划分（报告 §12）。确定性、按原图分组、防泄漏。"""
from __future__ import annotations
import hashlib
from collections import defaultdict
from forgery_pipeline.schema import Sample
from forgery_pipeline.split.grouping import origin_key, is_degraded

SPLITS = ["train", "val", "test_a", "test_b", "test_c", "test_d", "test_e", "test_f"]


def _hash01(key: str, salt: str) -> float:
    digest = hashlib.sha1(f"{salt}|{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") / 2 ** 32


def assign_splits(samples: list[Sample], holdout_generators, holdout_manipulation,
                  holdout_domains=("Places",), seed: int = 0) -> list[Sample]:
    hg, hm, hd = set(holdout_generators), set(holdout_manipulation), set(holdout_domains)
    groups: dict[str, list[Sample]] = defaultdict(list)
    for s in samples:
        groups[origin_key(s)].append(s)

    for okey, members in groups.items():
        gens = {m.generator_name for m in members if m.generator_name}
        manips = {m.manipulation_level3 for m in members if m.manipulation_level3}
        domains = {m.source_dataset for m in members if m.source_dataset}
        real_only = all(m.is_fake == 0 for m in members)
        if gens & hg:
            split = "test_b"
        elif manips & hm:
            split = "test_c"
        elif domains & hd:
            split = "test_d"
        elif real_only:
            r = _hash01(okey, f"real-{seed}")
            split = "train" if r < 0.60 else "val" if r < 0.75 else "test_f"
        else:
            r = _hash01(okey, f"fake-{seed}")
            split = "train" if r < 0.70 else "val" if r < 0.80 else "test_a"
        for m in members:
            m.split = split

    # 从 test_a 的退化 fake 中切出 degradation 测试集 test_e（仍属非 train，不破坏 train 隔离）
    for s in samples:
        if s.split == "test_a" and s.is_fake == 1 and is_degraded(s.postprocess):
            s.split = "test_e"
    return samples
```

- [ ] **Step 4: 运行确认通过** → PASS

- [ ] **Step 5: 提交**

```bash
git add src/forgery_pipeline/split/splitter.py tests/test_split_splitter.py
git commit -m "feat(split): 8-way 划分（holdout 路由 + 退化切分）"
```

---

### Task 27: 阶段编排（`pipeline.py`）

**Files:**
- Create: `src/forgery_pipeline/pipeline.py`, `tests/test_pipeline.py`

**Interfaces:**
- Consumes：所有 builders、`manifest.*`、`split.*`、`postprocess.degradations.sample_and_apply`、`config.PipelineConfig`
- Produces:
  - `run_pipeline(cfg:PipelineConfig) -> dict`（执行各阶段、写 `d0..d4.jsonl`、`manifest.jsonl`、`stats.json`，返回 stats；划分后若有泄漏抛 `RuntimeError`）
  - `apply_postprocess(out_dir, samples, prob, seed) -> None`（对部分 fake 就地退化并更新 `postprocess`）

- [ ] **Step 1: 写失败测试** `tests/test_pipeline.py`

```python
import dataclasses
from pathlib import Path
from forgery_pipeline.config import load_config, StageScales
from forgery_pipeline.pipeline import run_pipeline
from forgery_pipeline import manifest
from forgery_pipeline.split.leakage import check_leakage


def test_run_pipeline_end_to_end(tmp_path):
    cfg = load_config("configs/pipeline.example.yaml")
    cfg = dataclasses.replace(
        cfg, out_dir=str(tmp_path / "run"),
        scales=StageScales(d0=16, d1_per_generator=1, d2=8, d3=4, d4=3))
    st = run_pipeline(cfg)
    assert st["total"] > 0
    mani = Path(cfg.out_dir) / "manifest.jsonl"
    assert mani.exists()
    samples = manifest.read_jsonl(mani)            # 全部行通过 schema 校验
    assert len(samples) == st["total"]
    assert check_leakage(samples) == []            # 无泄漏
    # 局部篡改样本必须有 mask
    assert all(s.mask_path for s in samples
               if s.manipulation_level1 == "partial_manipulated")
    assert st["by_split"].get("train", 0) > 0
```

- [ ] **Step 2: 运行确认失败** → FAIL

- [ ] **Step 3: 实现 `pipeline.py`**

```python
"""阶段编排：D0→{D1,D2,D3}→D4→postprocess→split→manifest/stats（报告 §3）。"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import yaml
from forgery_pipeline import image_io, manifest
from forgery_pipeline.backends.mock import stable_hash
from forgery_pipeline.builders.d0_real import build_d0
from forgery_pipeline.builders.d1_whole import build_d1
from forgery_pipeline.builders.d2_local import build_d2
from forgery_pipeline.builders.d3_web import build_d3
from forgery_pipeline.builders.d4_explain import build_d4
from forgery_pipeline.config import PipelineConfig
from forgery_pipeline.postprocess.degradations import sample_and_apply
from forgery_pipeline.split.leakage import check_leakage
from forgery_pipeline.split.splitter import assign_splits
from forgery_pipeline.schema import Sample


def apply_postprocess(out_dir, samples: list[Sample], prob: float, seed: int) -> None:
    out_dir = Path(out_dir)
    for s in samples:
        if s.is_fake != 1:
            continue
        rng = np.random.default_rng((seed + stable_hash(s.image_id)) & 0x7FFFFFFF)
        if rng.random() >= prob:
            continue
        img = image_io.load_image(out_dir / s.image_path)
        degraded, pp = sample_and_apply(img, rng)
        image_io.save_image(degraded, out_dir / s.image_path)
        s.postprocess = pp


def run_pipeline(cfg: PipelineConfig) -> dict:
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    st = cfg.stages
    seed = cfg.seed

    d0 = build_d0(out, cfg.scales.d0, cfg.backend, seed) if st.get("d0") else []
    d1 = (build_d1(out, cfg.generators, cfg.scales.d1_per_generator, cfg.backend, seed)
          if st.get("d1") else [])
    d2 = (build_d2(out, d0, cfg.scales.d2, cfg.inpainters, cfg.backend, seed)
          if st.get("d2") else [])
    d3 = build_d3(out, d0, cfg.scales.d3, cfg.backend, seed) if st.get("d3") else []
    d4 = build_d4(out, d2 + d3, cfg.scales.d4, cfg.backend) if st.get("d4") else []

    for name, lib in [("d0", d0), ("d1", d1), ("d2", d2), ("d3", d3), ("d4", d4)]:
        manifest.write_jsonl(out / f"{name}.jsonl", lib)

    samples = d0 + d1 + d2 + d3 + d4

    if st.get("postprocess"):
        apply_postprocess(out, samples, cfg.postprocess_prob, seed)

    if st.get("split"):
        rules = yaml.safe_load(Path(cfg.split_config).read_text(encoding="utf-8"))
        assign_splits(
            samples,
            holdout_generators=rules.get("holdout_generators", []),
            holdout_manipulation=rules.get("holdout_manipulation", []),
            holdout_domains=rules.get("holdout_domains", ["Places"]),
            seed=seed,
        )
        leaks = check_leakage(samples)
        if leaks:
            raise RuntimeError("检测到数据泄漏: " + "; ".join(leaks))

    manifest.write_jsonl(out / "manifest.jsonl", samples)
    st_out = manifest.stats(samples)
    (out / "stats.json").write_text(
        json.dumps(st_out, ensure_ascii=False, indent=2), encoding="utf-8")
    return st_out
```

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/test_pipeline.py -q` → PASS

- [ ] **Step 5: 提交**

```bash
git add src/forgery_pipeline/pipeline.py tests/test_pipeline.py
git commit -m "feat(pipeline): 阶段编排（builders→postprocess→split→manifest/stats）"
```

---

### Task 28: 命令行入口（`cli.py`）

**Files:**
- Create: `src/forgery_pipeline/cli.py`, `tests/test_cli.py`

**Interfaces:**
- Produces:
  - `main(argv:list[str]|None=None) -> int`
  - 子命令：`run --config PATH [--out DIR]`、`stats --path MANIFEST`、`validate-manifest --path MANIFEST`

- [ ] **Step 1: 写失败测试** `tests/test_cli.py`

```python
import dataclasses
from pathlib import Path
from forgery_pipeline.cli import main
from forgery_pipeline.config import load_config, StageScales
from forgery_pipeline.pipeline import run_pipeline


def _make_run(tmp_path):
    cfg = load_config("configs/pipeline.example.yaml")
    cfg = dataclasses.replace(cfg, out_dir=str(tmp_path / "run"),
                              scales=StageScales(d0=12, d1_per_generator=1, d2=6, d3=3, d4=2))
    run_pipeline(cfg)
    return Path(cfg.out_dir) / "manifest.jsonl"


def test_validate_manifest_ok(tmp_path):
    mani = _make_run(tmp_path)
    assert main(["validate-manifest", "--path", str(mani)]) == 0


def test_stats_prints(tmp_path, capsys):
    mani = _make_run(tmp_path)
    assert main(["stats", "--path", str(mani)]) == 0
    out = capsys.readouterr().out
    assert "total" in out


def test_validate_manifest_missing_file_returns_nonzero(tmp_path):
    assert main(["validate-manifest", "--path", str(tmp_path / "nope.jsonl")]) != 0
```

- [ ] **Step 2: 运行确认失败** → FAIL

- [ ] **Step 3: 实现 `cli.py`**

```python
"""命令行入口（argparse）。默认 mock backend，开箱即跑。"""
from __future__ import annotations
import argparse
import dataclasses
import json
import sys
from pathlib import Path
from forgery_pipeline import manifest
from forgery_pipeline.config import load_config
from forgery_pipeline.pipeline import run_pipeline


def _cmd_run(args) -> int:
    cfg = load_config(args.config)
    if args.out:
        cfg = dataclasses.replace(cfg, out_dir=args.out)
    st = run_pipeline(cfg)
    print(json.dumps(st, ensure_ascii=False, indent=2))
    return 0


def _cmd_stats(args) -> int:
    samples = manifest.read_jsonl(args.path)
    print(json.dumps(manifest.stats(samples), ensure_ascii=False, indent=2))
    return 0


def _cmd_validate(args) -> int:
    path = Path(args.path)
    if not path.exists():
        print(f"manifest 不存在: {path}", file=sys.stderr)
        return 2
    try:
        samples = manifest.read_jsonl(path)  # 逐行 schema 校验
    except Exception as e:  # noqa: BLE001
        print(f"manifest 校验失败: {e}", file=sys.stderr)
        return 1
    print(f"OK: {len(samples)} 条样本全部通过校验")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="forgery-pipeline",
                                     description="伪造检测数据集生成 pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="运行完整 pipeline")
    p_run.add_argument("--config", required=True)
    p_run.add_argument("--out", default=None, help="覆盖 out_dir")
    p_run.set_defaults(func=_cmd_run)

    p_stats = sub.add_parser("stats", help="统计 manifest")
    p_stats.add_argument("--path", required=True)
    p_stats.set_defaults(func=_cmd_stats)

    p_val = sub.add_parser("validate-manifest", help="逐行校验 manifest")
    p_val.add_argument("--path", required=True)
    p_val.set_defaults(func=_cmd_validate)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
```

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/test_cli.py -q` → PASS

- [ ] **Step 5: 提交**

```bash
git add src/forgery_pipeline/cli.py tests/test_cli.py
git commit -m "feat(cli): run/stats/validate-manifest 子命令"
```

---

### Task 29: 端到端 smoke 测试 + 示例脚本

**Files:**
- Create: `tests/test_end_to_end.py`, `examples/run_mock_pipeline.sh`, `data/.gitkeep`

**Interfaces:**
- Consumes：`run_pipeline`, `manifest`, `check_leakage`

- [ ] **Step 1: 写端到端测试** `tests/test_end_to_end.py`

```python
import dataclasses
from pathlib import Path
from forgery_pipeline.config import load_config, StageScales
from forgery_pipeline.pipeline import run_pipeline
from forgery_pipeline import manifest
from forgery_pipeline.split.leakage import check_leakage
from forgery_pipeline.split.splitter import SPLITS


def test_full_mock_pipeline(tmp_path):
    cfg = load_config("configs/pipeline.example.yaml")
    cfg = dataclasses.replace(
        cfg, out_dir=str(tmp_path / "run"),
        scales=StageScales(d0=40, d1_per_generator=3, d2=24, d3=12, d4=8))
    st = run_pipeline(cfg)

    samples = manifest.read_jsonl(Path(cfg.out_dir) / "manifest.jsonl")
    # 1) 全部样本 schema 合法（read_jsonl 已校验）
    assert len(samples) == st["total"] > 0
    # 2) 无数据泄漏
    assert check_leakage(samples) == []
    # 3) 局部篡改样本必有 mask 且文件存在
    for s in samples:
        if s.manipulation_level1 == "partial_manipulated":
            assert s.mask_path and (Path(cfg.out_dir) / s.mask_path).exists()
    # 4) 训练/验证非空，且划分覆盖足够多类别
    present = set(st["by_split"])
    assert present <= set(SPLITS)
    assert st["by_split"].get("train", 0) > 0
    assert st["by_split"].get("val", 0) > 0
    assert len(present) >= 5
    # 5) 五个子库各有产出（real + 四类 fake 任务）
    assert st["real"] > 0 and st["fake"] > 0
    assert st["by_task_type"].get("explainable", 0) > 0
```

- [ ] **Step 2: 运行确认通过**

Run: `pytest tests/test_end_to_end.py -q` → PASS（首次写后即应通过；若某 split 为空导致 `>=5` 失败，提高 `scales` 或检查 splitter 路由）

- [ ] **Step 3: 写 `examples/run_mock_pipeline.sh`**

```bash
#!/usr/bin/env bash
# 用 mock 后端端到端跑通 pipeline 并打印统计。
set -euo pipefail
forgery-pipeline run --config configs/pipeline.example.yaml
forgery-pipeline stats --path data/run/manifest.jsonl
forgery-pipeline validate-manifest --path data/run/manifest.jsonl
```

并 `chmod +x examples/run_mock_pipeline.sh`；创建空文件 `data/.gitkeep`。

- [ ] **Step 4: 运行全量测试**

Run: `pytest -q`
Expected: 全绿。

- [ ] **Step 5: 提交**

```bash
git add tests/test_end_to_end.py examples/run_mock_pipeline.sh data/.gitkeep
git commit -m "test: 端到端 mock 全流程 smoke + 示例脚本"
```

---

### Task 30: README（中文）与收尾

**Files:**
- Create: `README.md`

**Interfaces:** 无代码接口，纯文档。

- [ ] **Step 1: 写 `README.md`（中文）**，至少包含：
  - 项目简介（基于六篇论文的伪造检测数据集生成 pipeline；五子库 D0–D4）。
  - 安装：`pip install -e .`（核心依赖）；可选 `pip install -e .[real]` 等。
  - 快速开始：`forgery-pipeline run --config configs/pipeline.example.yaml`，产物在 `data/run/`（`manifest.jsonl` + `stats.json`），随后 `stats` / `validate-manifest`。
  - 目录与子库说明（D0–D4 各自职责、对应论文）。
  - manifest 字段与层级标签简介（指向 `docs/superpowers/specs/2026-06-25-forgery-pipeline-design.md`）。
  - 8-way 划分与泄漏检查说明。
  - 接入真实后端：在 `backends/real/` 实现适配器并安装对应 extra，把 `configs/pipeline.example.yaml` 的 `backend` 改为 `real:diffusers` 等；当前默认 `mock`。
  - 测试：`pytest -q`。
  - 局限与后续：真实数据接入、postprocess 变体落库、人工复核池等。

- [ ] **Step 2: 运行全量测试确认绿**

Run: `pip install -e . && pytest -q` → 全部 PASS

- [ ] **Step 3: 提交**

```bash
git add README.md
git commit -m "docs: 中文 README（安装/快速开始/子库/接入真实后端）"
```

---

## Self-Review

**1. Spec coverage（逐节核对 spec → 任务）：**
- §3 五子库 / 总体流程 → Tasks 19–23 + 27 ✓
- §4 D0 来源/清洗/字段 → Task 19（QC=Task 14，dedup=Task 13）✓
- §5 D1 多生成器/prompt/字段 → Task 20（config 生成器清单=Task 6）✓
- §6 D2 mask→prompt→inpaint/7 类/尺度分桶/形态学 → Tasks 21 + 10 + 11 ✓
- §7 D3 差分伪 mask/QES → Tasks 22 + 12 + 17 ✓
- §8 D4 可解释三元组 → Task 23（Explainer=Task 8）✓
- §9 层级标签 + loss 常量 → Task 3 ✓
- §10 后处理增强 → Task 18（编排接入=Task 27）✓
- §11 QC（图像/mask/生成）+ 泄漏 → Tasks 14/15/16 + 25 ✓
- §12 8-way 划分 → Task 26 ✓
- §13 manifest 字段 + JSONL → Tasks 2 + 5 ✓
- 可插拔 backend + mock + real stub → Tasks 7/8/9 ✓
- CLI/编排/端到端 → Tasks 27/28/29；README=Task 30 ✓

无未覆盖 spec 条目。

**2. Placeholder scan：** 各步含完整代码/命令/期望输出，无 TBD/TODO/“类似上文”。real 适配器 `sam_segmenter.py`/`mllm_explainer.py` 在 Task 9 Step 4 以“与 diffusers_gen 同构（分别 guarded import segment_anything 与 openai/anthropic）”描述——实现时照搬骨架即可，非占位。

**3. Type consistency（跨任务签名核对）：**
- `Sample` 字段在 Tasks 2/3 定义，被所有 builder/split 使用，名称一致 ✓
- 图像/掩码类型 (H,W,3)uint8 / (H,W)uint8{0,255} 全程一致 ✓
- backend 方法签名（generate/inpaint/propose_masks/explain/iter_images）在 Task 7 定义，Task 8 实现、registry(9)/builders(19–23) 调用一致 ✓
- `filter_and_sample -> list[(mask,ratio,bucket)]`（Task 11）被 Task 21 解包一致 ✓
- `pseudo_mask -> (mask, metrics{confidence,ssim_score,boundary_sharpness,area_ratio})`（Task 12）被 Task 22 使用一致 ✓
- `qes_score(...)/route_from_score/bucket_from_score`（Task 17）被 Task 22 调用一致 ✓
- `origin_key/is_degraded`（Task 24）被 25/26 使用一致；`assign_splits` 签名（Task 26）被 Task 27 调用一致 ✓
- `build_d*` 签名（Tasks 19–23）被 Task 27 调用一致 ✓

无签名不一致。

## 执行顺序与依赖
严格按 Task 1 → 30 顺序执行（后者依赖前者的接口）。每个任务自带失败测试→实现→通过→提交；任务末尾 `pytest` 局部绿，Task 29/30 跑全量绿。

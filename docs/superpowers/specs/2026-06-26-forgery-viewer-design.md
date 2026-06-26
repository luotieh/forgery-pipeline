# 数据集可视化前端（静态 HTML 生成器）— 设计文档（Spec）

- 日期：2026-06-26
- 关联：`forgery-pipeline` 主项目（见 `docs/superpowers/specs/2026-06-25-forgery-pipeline-design.md`）
- 目标：为生成的数据集提供一个**离线、零新依赖**的可视化检视界面，便于人工核查伪造数据质量。

---

## 1. 目标与范围

### 1.1 目标
新增 CLI 子命令 `forgery-pipeline viewer --run <dir>`，读取某次运行的 `manifest.jsonl` 与图像/掩码，生成一个**自包含的 `viewer.html`**：缩略图画廊 + 多维筛选 + 详情（原图 | 伪造图 | mask 叠加 + 层级标签 + 解释文本 + 元数据）。在浏览器中打开即可浏览，无需服务器。

### 1.2 范围内
- 读取 `manifest.jsonl`（复用 `manifest.read_jsonl`）。
- 用已有 opencv/PIL 预渲染 **mask 叠加图** 与 **缩略图**。
- 生成内联 CSS/JS、内嵌精简样本 JSON 的 `viewer.html`，图像以**相对路径**引用。
- CLI 子命令 `viewer`，参数 `--run/--out/--max/--open`。
- 单元测试 + CLI 测试。

### 1.3 非目标（YAGNI）
- 不做服务器（Flask/Streamlit）、不做在线编辑/标注、不做分页/虚拟滚动（数百样本直接渲染足够；超大数据集用 `--max` 兜底）。
- 不引入任何新的运行时依赖。

### 1.4 约束
- 仅用 stdlib + 项目已有依赖（numpy/Pillow/opencv-python-headless）。
- 注释/文档/界面文案用中文；代码标识符用 English。
- 确定性：缩略图/叠加图渲染对相同输入产出相同结果。

---

## 2. 文件结构

| 文件 | 职责 |
|---|---|
| `src/forgery_pipeline/viewer.py` | 渲染叠加/缩略图 + 生成 viewer.html 的全部逻辑 |
| `src/forgery_pipeline/cli.py`（改） | 新增 `viewer` 子命令 |
| `tests/test_viewer.py` | 叠加渲染、build_viewer、CLI 测试 |

产物（落在 run 目录，已被 `.gitignore` 覆盖）：
```
<run>/viewer.html                  自包含页面
<run>/viewer_assets/thumb/<id>.jpg   画廊缩略图
<run>/viewer_assets/overlay/<id>.jpg 伪造图 + mask 叠加
```

---

## 3. 接口（`viewer.py`）

- `render_overlay(fake: np.ndarray, mask: np.ndarray, color=(255,0,0), alpha=0.35) -> np.ndarray`
  在伪造图上叠加 mask：半透明红色填充 + 红色轮廓（`cv2.findContours`+`drawContours`），返回同尺寸 (H,W,3) uint8。
- `make_thumb(img: np.ndarray, size: int = 224) -> np.ndarray`
  等比缩放使长边=size（短边按比例），返回缩略图。
- `build_viewer(run_dir, out_html=None, max_samples=None) -> Path`
  编排：读 `<run>/manifest.jsonl` → 对每个样本生成 thumb；对有 mask 的样本生成 overlay → 把精简后的样本记录序列化为 JSON 内嵌进 HTML → 写 `out_html`（默认 `<run>/viewer.html`）→ 返回路径。`max_samples` 截断。

**内嵌样本记录字段**（仅 UI 需要）：`image_id, is_fake, task_type, split, manipulation_level1, manipulation_level3, generator_family, generator_name, quality_score, prompt, explanation, paths{real, fake, overlay, thumb}`。每个 path 用 `os.path.relpath(图像绝对路径, out_html 所在目录)` 计算，因此无论 `--out` 指向哪里，页面都能正确加载图像；缺失项为 null。

**HTML 结构**：单文件，`<style>` 内联，`<script>` 内联一个 `const SAMPLES=[...]` 与渲染逻辑。
- 顶部筛选条：`split / task_type / manipulation_level1 / manipulation_level3 / generator_family / is_fake` 下拉（选项由数据动态去重生成）+ image_id 文本搜索 + 实时计数。
- 左侧画廊：缩略图网格，点击选中（真/假以边框色区分）。
- 右侧详情：有真实底图与 mask 的样本显示 **原图 | 伪造图 | 叠加** 三联；整图生成/真实图仅显示单图。下方层级标签表 + 解释四段 + 元数据。

---

## 4. 数据流

```
pipeline run → <run>/manifest.jsonl + 图像/掩码
  → viewer.build_viewer 读 manifest
  → 逐样本生成 thumb；有 mask 的生成 overlay
  → 精简样本 → 内嵌 JSON
  → 写 viewer.html（引用相对图像路径）
  → 浏览器打开（--open 可选，best-effort）
```

`render_overlay` 对每个定位样本读取 `image_path`(fake) 与 `mask_path`，叠加后存到 `viewer_assets/overlay/`。真实/整图样本无 mask，跳过 overlay。

---

## 5. CLI（`cli.py` 新增子命令）

```
forgery-pipeline viewer --run <dir> [--out viewer.html] [--max N] [--open]
```
- `--run`：必填，run 目录（含 manifest.jsonl）。
- `--out`：输出 html 路径，默认 `<run>/viewer.html`。
- `--max`：最多渲染样本数（兜底大数据集）。
- `--open`：best-effort 调 `webbrowser.open` 打开；无论是否成功都打印 html 绝对路径。
- 返回 0；`--run` 不存在或无 manifest.jsonl 时打印错误并返回非 0。

---

## 6. 错误处理
- run 目录不存在 / 无 `manifest.jsonl` → 返回码 2 并提示。
- 单个样本图像缺失/解码失败 → 跳过该样本的 thumb/overlay，但不中断整体（记录计数，最终打印"跳过 N 个"）。
- `--open` 在无图形环境（如纯 WSL 无浏览器）失败 → 吞掉异常，仅打印路径。

---

## 7. 测试策略（`tests/test_viewer.py`）
- `render_overlay`：构造已知 mask，断言叠加图在 mask 区域出现红色通道增强、整体尺寸不变、dtype uint8。
- `make_thumb`：长边等于 size、等比、uint8。
- `build_viewer`：在小型生成 run（用 `run_pipeline` 造 tmp 数据，小 scales）上生成 html；断言：① 文件存在；② html 含内嵌 `SAMPLES` 且样本数==manifest 行数（或==max）；③ 内嵌 JSON 可被解析；④ 至少一个 overlay/thumb 文件确实生成且被引用的相对路径在磁盘存在。
- CLI：`main(["viewer","--run",tmp])==0` 且 `viewer.html` 生成；`--run` 指向空目录返回非 0。

验收：`pytest -q` 全绿；对真实 `data/run` 跑 `forgery-pipeline viewer --run data/run` 能产出可在浏览器打开的 viewer.html。

---

## 8. 依赖
无新增。复用 `numpy / Pillow / opencv-python-headless` 与 stdlib（`json/html/webbrowser/pathlib`）。

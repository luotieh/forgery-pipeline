# 数据集可视化前端 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增 CLI 子命令 `forgery-pipeline viewer --run <dir>`，读取某次运行的 `manifest.jsonl` + 图像/掩码，生成一个自包含的 `viewer.html`（缩略图画廊 + 多维筛选 + 原图/伪造/mask 叠加详情），浏览器打开即可检视数据集。

**Architecture:** 单模块 `viewer.py`：用已有 opencv 预渲染 mask 叠加图与缩略图，把精简样本记录序列化为内嵌 JSON 注入到带内联 CSS/JS 的 HTML 模板，图像以相对路径引用；`cli.py` 加一个 `viewer` 子命令调用它。零新依赖、无服务器。

**Tech Stack:** Python ≥3.10、numpy、opencv-python-headless、Pillow（均为已有依赖）、stdlib（json/os/html/webbrowser/pathlib）、vanilla JS。

## Global Constraints

- **不新增任何运行时依赖**，仅用 stdlib + 已有 `numpy/Pillow/opencv-python-headless`。
- 注释/文档/界面文案用中文；代码标识符用 English。
- 图像 `(H,W,3) uint8 RGB`；掩码 `(H,W) uint8 {0,255}`（复用 `image_io.load_image/load_mask/save_image`）。
- 确定性：相同输入产出相同 thumb/overlay。
- 产物落在 run 目录（已被 `.gitignore` 覆盖）：`viewer.html`、`viewer_assets/thumb/*.jpg`、`viewer_assets/overlay/*.jpg`。
- 嵌入 JSON 前把 `<` 转义为 `<`，防止数据中出现 `</script>` 截断脚本。
- 内嵌样本字段：`image_id, is_fake, task_type, split, manipulation_level1, manipulation_level3, generator_family, generator_name, quality_score, prompt, explanation, paths{real,fake,overlay,thumb}`；path 用 `os.path.relpath(图像绝对路径, out_html 目录)` 计算；缺失为 null。

---

## File Structure

| 文件 | 职责 |
|---|---|
| `src/forgery_pipeline/viewer.py` | `render_overlay` / `make_thumb` / `build_viewer` + HTML 模板 |
| `src/forgery_pipeline/cli.py`（改） | 新增 `viewer` 子命令 |
| `tests/test_viewer.py` | 叠加渲染、缩略图、build_viewer、CLI 测试 |

---

## Task 1: 渲染辅助（`render_overlay` + `make_thumb`）

**Files:**
- Create: `src/forgery_pipeline/viewer.py`
- Create: `tests/test_viewer.py`

**Interfaces:**
- Produces:
  - `render_overlay(fake:np.ndarray, mask:np.ndarray, color=(255,0,0), alpha=0.35) -> np.ndarray`：mask 区域半透明红色填充 + 红色轮廓，返回同尺寸 (H,W,3) uint8。
  - `make_thumb(img:np.ndarray, size:int=224) -> np.ndarray`：等比缩放使长边=size（不放大），返回 uint8。

- [ ] **Step 1: 写失败测试** `tests/test_viewer.py`

```python
import numpy as np
from forgery_pipeline.viewer import render_overlay, make_thumb


def test_render_overlay_tints_mask_region():
    fake = np.full((50, 50, 3), 100, np.uint8)
    mask = np.zeros((50, 50), np.uint8)
    mask[10:30, 10:30] = 255
    out = render_overlay(fake, mask)
    assert out.shape == (50, 50, 3) and out.dtype == np.uint8
    assert out[20, 20, 0] > 100          # mask 内 R 通道被染红
    assert tuple(out[45, 45]) == (100, 100, 100)  # mask 外不变


def test_make_thumb_long_side_and_dtype():
    t = make_thumb(np.zeros((300, 150, 3), np.uint8), size=224)
    assert max(t.shape[:2]) == 224
    assert t.shape[2] == 3 and t.dtype == np.uint8
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_viewer.py -q` → FAIL（`ModuleNotFoundError`/无函数）

- [ ] **Step 3: 创建 `viewer.py`（先实现两个辅助）**

```python
"""数据集可视化：生成自包含 viewer.html（静态 HTML 生成器）。"""
from __future__ import annotations
import json
import os
from pathlib import Path
import cv2
import numpy as np
from forgery_pipeline import image_io, manifest


def render_overlay(fake: np.ndarray, mask: np.ndarray,
                   color: tuple = (255, 0, 0), alpha: float = 0.35) -> np.ndarray:
    """在伪造图上叠加 mask：mask 区域半透明红色填充 + 红色轮廓。"""
    out = fake.astype(np.float32).copy()
    m = mask > 127
    col = np.array(color, np.float32)
    out[m] = (1.0 - alpha) * out[m] + alpha * col
    out = out.astype(np.uint8)
    contours, _ = cv2.findContours(m.astype(np.uint8) * 255,
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, 2)
    return out


def make_thumb(img: np.ndarray, size: int = 224) -> np.ndarray:
    """等比缩放使长边=size（不放大）。"""
    h, w = img.shape[:2]
    scale = min(1.0, size / max(h, w))
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
```

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/test_viewer.py -q` → PASS

- [ ] **Step 5: 提交**

```bash
git add src/forgery_pipeline/viewer.py tests/test_viewer.py
git commit -m "feat(viewer): mask 叠加与缩略图渲染辅助"
```

---

## Task 2: `build_viewer` + HTML 模板

**Files:**
- Modify: `src/forgery_pipeline/viewer.py`（追加模板常量与 `build_viewer`）
- Modify: `tests/test_viewer.py`（追加 build_viewer 测试）

**Interfaces:**
- Consumes（Task 1）：`render_overlay`, `make_thumb`；`manifest.read_jsonl`, `image_io.*`
- Produces：`build_viewer(run_dir, out_html=None, max_samples=None) -> Path`，读 `<run>/manifest.jsonl`，生成 thumb/overlay 与 `viewer.html`，返回 html 路径。

- [ ] **Step 1: 追加失败测试** `tests/test_viewer.py`

```python
import json
import dataclasses
from pathlib import Path
from forgery_pipeline.viewer import build_viewer
from forgery_pipeline import manifest
from forgery_pipeline.config import load_config, StageScales
from forgery_pipeline.pipeline import run_pipeline


def _tiny_run(tmp_path):
    cfg = load_config("configs/pipeline.example.yaml")
    cfg = dataclasses.replace(cfg, out_dir=str(tmp_path / "run"),
                              scales=StageScales(d0=8, d1_per_generator=1, d2=4, d3=2, d4=2))
    run_pipeline(cfg)
    return Path(cfg.out_dir)


def test_build_viewer_emits_html_with_all_samples(tmp_path):
    run = _tiny_run(tmp_path)
    out = build_viewer(run)
    assert out.exists() and out.name == "viewer.html"
    text = out.read_text(encoding="utf-8")
    line = next(l for l in text.splitlines() if l.startswith("const SAMPLES = "))
    data = json.loads(line[len("const SAMPLES = "):].rstrip(";"))
    assert len(data) == len(manifest.read_jsonl(run / "manifest.jsonl"))
    # 至少一个定位样本生成了 overlay，且引用路径在磁盘存在
    ov = [d for d in data if d["paths"]["overlay"]]
    assert ov
    assert (out.parent / ov[0]["paths"]["overlay"]).exists()
    assert (out.parent / data[0]["paths"]["thumb"]).exists()


def test_build_viewer_max_samples(tmp_path):
    run = _tiny_run(tmp_path)
    out = build_viewer(run, max_samples=3)
    text = out.read_text(encoding="utf-8")
    line = next(l for l in text.splitlines() if l.startswith("const SAMPLES = "))
    assert len(json.loads(line[len("const SAMPLES = "):].rstrip(";"))) == 3
```

- [ ] **Step 2: 运行确认失败** → FAIL

- [ ] **Step 3: 追加模板与 `build_viewer` 到 `viewer.py`**

在文件末尾追加（`_HTML_TEMPLATE` 用 `__SAMPLES__` 占位，单独成行便于解析）：

```python
def _rel(path_abs: Path, start_dir: Path) -> str:
    return os.path.relpath(path_abs, start_dir).replace(os.sep, "/")


def build_viewer(run_dir, out_html=None, max_samples=None) -> Path:
    run_dir = Path(run_dir)
    samples = manifest.read_jsonl(run_dir / "manifest.jsonl")
    if max_samples:
        samples = samples[:max_samples]
    out_html = Path(out_html) if out_html else run_dir / "viewer.html"
    out_dir = out_html.parent
    thumb_dir = run_dir / "viewer_assets" / "thumb"
    overlay_dir = run_dir / "viewer_assets" / "overlay"

    records, skipped = [], 0
    for s in samples:
        fake_abs = run_dir / s.image_path
        try:
            fimg = image_io.load_image(fake_abs)
        except Exception:
            skipped += 1
            continue
        thumb_abs = thumb_dir / f"{s.image_id}.jpg"
        image_io.save_image(make_thumb(fimg), thumb_abs)

        overlay_rel = None
        if s.mask_path and (run_dir / s.mask_path).exists():
            try:
                ov = render_overlay(fimg, image_io.load_mask(run_dir / s.mask_path))
                overlay_abs = overlay_dir / f"{s.image_id}.jpg"
                image_io.save_image(ov, overlay_abs)
                overlay_rel = _rel(overlay_abs, out_dir)
            except Exception:
                overlay_rel = None

        real_rel = None
        if s.real_image_path and (run_dir / s.real_image_path).exists():
            real_rel = _rel(run_dir / s.real_image_path, out_dir)

        records.append({
            "image_id": s.image_id, "is_fake": s.is_fake,
            "task_type": s.task_type.value, "split": s.split,
            "manipulation_level1": s.manipulation_level1,
            "manipulation_level3": s.manipulation_level3,
            "generator_family": s.generator_family,
            "generator_name": s.generator_name,
            "quality_score": s.quality_score, "prompt": s.prompt,
            "explanation": s.explanation.model_dump() if s.explanation else None,
            "paths": {"real": real_rel, "fake": _rel(fake_abs, out_dir),
                      "overlay": overlay_rel, "thumb": _rel(thumb_abs, out_dir)},
        })

    payload = json.dumps(records, ensure_ascii=False).replace("<", "\\u003c")
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(_HTML_TEMPLATE.replace("__SAMPLES__", payload), encoding="utf-8")
    if skipped:
        print(f"viewer: 跳过 {skipped} 个无法读取的样本")
    return out_html
```

并在文件顶部（imports 之后）加入模板常量：

```python
_HTML_TEMPLATE = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><title>Forgery Dataset Viewer</title>
<style>
body{margin:0;font-family:system-ui,Arial,sans-serif;background:#0f1115;color:#e6e6e6}
header{padding:10px 14px;background:#171a21;position:sticky;top:0;border-bottom:1px solid #2a2f3a}
header h1{font-size:15px;margin:0 0 8px}
.filters{display:flex;flex-wrap:wrap;gap:8px;align-items:center}
select,input{background:#0f1115;color:#e6e6e6;border:1px solid #2a2f3a;border-radius:6px;padding:4px 6px;font-size:12px}
.count{font-size:12px;color:#9aa4b2;margin-left:auto}
main{display:flex;height:calc(100vh - 92px)}
.gallery{width:46%;overflow:auto;padding:8px;display:grid;grid-template-columns:repeat(auto-fill,minmax(110px,1fr));gap:8px;align-content:start}
.card{cursor:pointer;border:2px solid transparent;border-radius:8px;overflow:hidden;background:#171a21}
.card.fake{border-color:#7a2230}.card.real{border-color:#234e2a}.card.sel{outline:2px solid #4f8cff}
.card img{width:100%;display:block;aspect-ratio:1/1;object-fit:cover}
.card .cap{font-size:10px;padding:2px 4px;color:#9aa4b2;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.detail{flex:1;overflow:auto;padding:14px;border-left:1px solid #2a2f3a}
.imgs{display:flex;gap:10px;flex-wrap:wrap}.imgs figure{margin:0}.imgs figcaption{font-size:11px;color:#9aa4b2;text-align:center}
.imgs img{max-width:240px;max-height:240px;border:1px solid #2a2f3a;border-radius:6px}
table{border-collapse:collapse;margin-top:12px;font-size:12px}td{border:1px solid #2a2f3a;padding:3px 8px}.k{color:#9aa4b2}
.expl{margin-top:12px;font-size:12px;line-height:1.5;background:#171a21;padding:10px;border-radius:8px}
.badge{display:inline-block;font-size:11px;padding:1px 6px;border-radius:10px;margin-right:4px}.b-fake{background:#7a2230}.b-real{background:#234e2a}
</style></head><body>
<header><h1>Forgery Dataset Viewer</h1><div class="filters" id="filters"></div></header>
<main><div class="gallery" id="gallery"></div>
<div class="detail" id="detail"><p style="color:#9aa4b2">点击左侧缩略图查看详情</p></div></main>
<script>
const SAMPLES = __SAMPLES__;
const FIELDS=[["split","split"],["task_type","task"],["manipulation_level1","level1"],["manipulation_level3","level3"],["generator_family","gen"]];
const state={};
function esc(x){return String(x==null?'':x).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
function distinct(f){return [...new Set(SAMPLES.map(s=>s[f]).filter(v=>v!=null))].sort();}
function buildFilters(){
 const c=document.getElementById('filters');
 for(const [f,label] of FIELDS){const sel=document.createElement('select');
  sel.innerHTML=`<option value="">${label}: 全部</option>`+distinct(f).map(v=>`<option>${esc(v)}</option>`).join('');
  sel.onchange=()=>{state[f]=sel.value;render();};c.appendChild(sel);}
 const fk=document.createElement('select');
 fk.innerHTML='<option value="">真/假: 全部</option><option value="1">fake</option><option value="0">real</option>';
 fk.onchange=()=>{state.is_fake=fk.value;render();};c.appendChild(fk);
 const q=document.createElement('input');q.placeholder='搜索 image_id';q.oninput=()=>{state.q=q.value;render();};c.appendChild(q);
 const cnt=document.createElement('span');cnt.className='count';cnt.id='count';c.appendChild(cnt);}
function match(s){
 for(const [f] of FIELDS){if(state[f]&&String(s[f])!==state[f])return false;}
 if(state.is_fake!==undefined&&state.is_fake!==''&&String(s.is_fake)!==state.is_fake)return false;
 if(state.q&&!s.image_id.includes(state.q))return false;return true;}
function render(){
 const g=document.getElementById('gallery');const list=SAMPLES.filter(match);
 document.getElementById('count').textContent=`${list.length} / ${SAMPLES.length} 样本`;g.innerHTML='';
 for(const s of list){const d=document.createElement('div');d.className='card '+(s.is_fake?'fake':'real');
  d.innerHTML=`<img loading="lazy" src="${s.paths.thumb}"><div class="cap">${esc(s.image_id)}</div>`;
  d.onclick=()=>{showDetail(s);[...g.children].forEach(x=>x.classList.remove('sel'));d.classList.add('sel');};
  g.appendChild(d);}}
function fig(src,cap){return src?`<figure><img src="${src}"><figcaption>${cap}</figcaption></figure>`:'';}
function row(k,v){return v==null||v===''?'':`<tr><td class="k">${k}</td><td>${esc(v)}</td></tr>`;}
function showDetail(s){const e=s.explanation;
 const imgs=fig(s.paths.real,'原图 real')+fig(s.paths.fake,'伪造 fake')+fig(s.paths.overlay,'mask 叠加');
 const expl=e?`<div class="expl"><b>解释</b><br>位置: ${esc(e.location_description)}<br>伪迹: ${esc(e.visual_artifact_description)}<br>推理: ${esc(e.semantic_reasoning)}<br>结论: ${esc(e.forensic_conclusion)}</div>`:'';
 document.getElementById('detail').innerHTML=
  `<div><span class="badge ${s.is_fake?'b-fake':'b-real'}">${s.is_fake?'FAKE':'REAL'}</span><b>${esc(s.image_id)}</b></div>`+
  `<div class="imgs">${imgs||'<p style=\\'color:#9aa4b2\\'>无图像</p>'}</div>`+
  `<table>${row('task_type',s.task_type)}${row('split',s.split)}${row('level1',s.manipulation_level1)}${row('level3',s.manipulation_level3)}${row('generator',s.generator_name)}${row('family',s.generator_family)}${row('quality_score',s.quality_score)}${row('prompt',s.prompt)}</table>`+expl;}
buildFilters();render();
</script></body></html>
"""
```

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/test_viewer.py -q` → PASS

- [ ] **Step 5: 提交**

```bash
git add src/forgery_pipeline/viewer.py tests/test_viewer.py
git commit -m "feat(viewer): build_viewer 生成自包含 viewer.html"
```

---

## Task 3: CLI `viewer` 子命令 + 真实数据验证 + README

**Files:**
- Modify: `src/forgery_pipeline/cli.py`（加 `viewer` 子命令、`webbrowser` import）
- Modify: `tests/test_viewer.py`（追加 CLI 测试）
- Modify: `README.md`（快速开始加 viewer 一节）

**Interfaces:**
- Consumes：`viewer.build_viewer`
- Produces：`forgery-pipeline viewer --run <dir> [--out PATH] [--max N] [--open]`，返回 0；run 无 manifest 返回 2。

- [ ] **Step 1: 追加失败测试** `tests/test_viewer.py`

```python
from forgery_pipeline.cli import main


def test_viewer_cli_ok(tmp_path):
    run = _tiny_run(tmp_path)
    assert main(["viewer", "--run", str(run)]) == 0
    assert (run / "viewer.html").exists()


def test_viewer_cli_missing_run(tmp_path):
    assert main(["viewer", "--run", str(tmp_path / "nope")]) != 0
```

- [ ] **Step 2: 运行确认失败** → FAIL（无 viewer 子命令）

- [ ] **Step 3: 修改 `cli.py`**

顶部 import 区加 `import webbrowser`。新增命令函数：

```python
def _cmd_viewer(args) -> int:
    from pathlib import Path
    from forgery_pipeline.viewer import build_viewer
    run = Path(args.run)
    if not (run / "manifest.jsonl").exists():
        print(f"run 目录缺少 manifest.jsonl: {run}", file=sys.stderr)
        return 2
    out = build_viewer(run, out_html=args.out, max_samples=args.max)
    print(f"已生成 {out.resolve()}")
    if args.open:
        try:
            webbrowser.open(out.resolve().as_uri())
        except Exception:
            pass
    return 0
```

在 `main()` 的子命令区追加（放在 `validate-manifest` 之后）：

```python
    p_view = sub.add_parser("viewer", help="生成数据集可视化 viewer.html")
    p_view.add_argument("--run", required=True, help="run 目录（含 manifest.jsonl）")
    p_view.add_argument("--out", default=None, help="输出 html 路径，默认 <run>/viewer.html")
    p_view.add_argument("--max", type=int, default=None, help="最多渲染样本数")
    p_view.add_argument("--open", action="store_true", help="生成后尝试用浏览器打开")
    p_view.set_defaults(func=_cmd_viewer)
```

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/test_viewer.py -q` → PASS

- [ ] **Step 5: 全量测试 + 真实数据冒烟**

```bash
pytest -q
python3 -m forgery_pipeline.cli run --config configs/pipeline.example.yaml >/dev/null
python3 -m forgery_pipeline.cli viewer --run data/run
test -f data/run/viewer.html && echo "viewer.html OK"
```
Expected: 全绿 + 打印 `viewer.html OK`。

- [ ] **Step 6: README 加一节**（`## 快速开始` 末尾追加）

```markdown
### 可视化检视生成的数据集

```bash
forgery-pipeline run --config configs/pipeline.example.yaml
forgery-pipeline viewer --run data/run        # 生成 data/run/viewer.html
# 浏览器打开 data/run/viewer.html：缩略图画廊 + 按 split/篡改类型/生成器筛选
# + 原图/伪造图/mask 叠加 + 层级标签 + 解释文本
```
```

- [ ] **Step 7: 提交**

```bash
git add src/forgery_pipeline/cli.py tests/test_viewer.py README.md
git commit -m "feat(cli): viewer 子命令 + README 可视化说明"
```

---

## Self-Review

**1. Spec coverage：**
- §3 `render_overlay/make_thumb/build_viewer` → Task 1+2 ✓
- §3 内嵌字段 + relpath → Task 2 `build_viewer` records ✓
- §3 HTML：筛选条/画廊/三联详情/标签/解释 → Task 2 模板 ✓
- §5 CLI `viewer --run/--out/--max/--open` → Task 3 ✓
- §6 错误处理：无 manifest 返回 2、单样本读失败跳过、--open best-effort → Task 3 `_cmd_viewer` + Task 2 try/except + skipped 计数 ✓
- §7 测试：overlay/thumb/build_viewer/CLI → Task 1/2/3 ✓
- §8 零新依赖 → 仅用已有库 ✓

无缺口。

**2. Placeholder scan：** 各步均含完整代码/命令/期望，无 TBD/TODO。

**3. Type consistency：**
- `render_overlay(fake,mask,...)->ndarray`、`make_thumb(img,size)->ndarray`（Task 1）被 Task 2 `build_viewer` 调用，签名一致 ✓
- `build_viewer(run_dir,out_html,max_samples)->Path`（Task 2）被 Task 3 `_cmd_viewer` 调用，参数名一致（out_html=args.out, max_samples=args.max）✓
- 内嵌 JSON 标记 `const SAMPLES = ` 在模板与测试解析中一致 ✓

无不一致。

## 执行顺序
Task 1 → 2 → 3 顺序执行；每任务自带失败测试→实现→通过→提交；Task 3 跑全量 + 真实数据冒烟。

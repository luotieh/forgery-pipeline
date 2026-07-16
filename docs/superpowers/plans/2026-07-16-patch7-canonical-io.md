# PATCH 7 canonical I/O + VAE 往返负样本 + compositing 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 消除"压缩/处理历史可预测 is_fake"的训练集混淆 + 把 paste-back 变成显式受控变量，使 Phase B 主生成解除阻断（spec：`docs/PATCHES_addendum_06_07_2026-07-15.md` PATCH 7 + 8.5 字段一次性定死）。

**Architecture:** 真/假共享同一条非生成处理链（`load_and_resize`→编辑(可选)→`save_canonical` 全 PNG），逐行 `io_chain` 记录处理链并由 validator V2 断言其对 is_fake 不可预测；VAE 往返负样本（`sample_kind=real_vae_rt`）在 split 后按 split 分层插入；`composite()` 纯函数实现 none/paste/paste_feather，D2 主生成 50/50 分配、probe 可生成成对样本。Mock 后端为 inpaint 输出加全局 mock-VAE 印记以忠实模拟"整图 VAE 直出"。

**Tech Stack:** numpy/cv2/PIL/pydantic（零新依赖）；mock 后端 CPU 全链可测；真实 SD1.5 VAE 懒加载（本地不跑 GPU）。

## Global Constraints

- 注释/文档中文、标识符 English、确定性（显式 seed）；sklearn-free（checking/ 纪律沿用）。
- 主库一律 **PNG**；不引入新的有损压缩步（PATCH 5 退化机制/字段零改动，spec 7.6）。
- `sample_kind ∈ {real, real_vae_rt, edited}`；`compositing ∈ {none, paste, paste_feather}`；`io_chain` 节点语法 `decode>rs{S}>[edit:{gen}|vae_rt:{vae}]>png`，legacy 行 `io_chain=legacy`。
- **V2 判定的谱系适配**（相对 spec 字面的已记录偏差）：非生成段 = 去掉 `edit:*`/`vae_rt:*`/`gen:*` 节点且忽略首节点 `decode`（D1 全生成行无源可解码，字面规则会结构性 FAIL；V2 意图=管线附加处理不可预测 is_fake，源头 JPEG 史由 vae_rt 负样本对冲——写入 validator docstring）。
- V4（vae_rt 占比）默认 **auto**：仅当 manifest 含 `real_vae_rt` 行或 CLI `--profile run` 时强制（probe manifest 无 test_a，不适用）。
- 每任务 TDD：先写失败测试→跑红→最小实现→跑绿→全套 pytest→commit。

---

## File Structure

| 文件 | 职责 |
|---|---|
| `src/forgery_pipeline/schema.py` (M) | +6 字段：io_chain/sample_kind/compositing/feather_px/probe_group/pair_id |
| `src/forgery_pipeline/compositing.py` (C) | `composite()` 纯函数（spec 7.3 参考实现） |
| `src/forgery_pipeline/image_io.py` (M) | `load_and_resize()` + `save_canonical()`（强制 PNG）+ `chain()` |
| `src/forgery_pipeline/backends/base.py` (M) | `VaeRoundtrip` ABC |
| `src/forgery_pipeline/backends/mock.py` (M) | `MockVaeRoundtrip`；`MockInpainter` 输出加全局印记 |
| `src/forgery_pipeline/backends/real/diffusers_gen.py` (M) | `SDVaeRoundtrip`（懒加载 fp32）；inpaint 无隐式回贴的审计注释 |
| `src/forgery_pipeline/backends/registry.py` (M) | `get_vae_rt(backend)` |
| `src/forgery_pipeline/builders/d0_real.py` (M) | canonical 摄取（PNG + io_chain + sample_kind=real） |
| `src/forgery_pipeline/builders/d1_whole.py`/`d3_web.py` (M) | 行补 io_chain/sample_kind=edited、输出 PNG |
| `src/forgery_pipeline/builders/d2_local.py` (M) | compositing 50/50 + composite() + 字段 |
| `src/forgery_pipeline/builders/probe.py` (M) | 行补字段；`compositing_pairs` 成对 probe 选项 |
| `src/forgery_pipeline/pipeline.py` (M) | split 后插入 vae_rt 负样本行（分层） |
| `src/forgery_pipeline/config.py` (M) | `vae_rt_frac`（默认 0.15）、`compositing_feather_px`（默认 8） |
| `src/forgery_pipeline/validate.py` (C) | V1–V7 断言集 |
| `src/forgery_pipeline/cli.py` (M) | validate-manifest 接 V1–V7（--profile auto/run）；stats 已在 manifest.py 扩 |
| `src/forgery_pipeline/manifest.py` (M) | stats + io_chain×is_fake×split / by_sample_kind / by_compositing |
| `scripts/backfill_manifest_v7.py` (C) | 旧 manifest 回填 |
| `scripts/assert_compositing_pairs.py` (C) | 7.5 成对断言（paste 带外逐像素==orig；none 全图≠orig） |
| tests (C/M) | test_compositing / test_canonical_io / test_vae_rt / test_validate_v7 / test_backfill_v7 + 既有测试适配 |

---

### Task 1: schema 字段一次性定死 + 回填脚本

**Files:** Modify `src/forgery_pipeline/schema.py`；Create `scripts/backfill_manifest_v7.py`、`tests/test_backfill_v7.py`

**Interfaces (Produces):** `Sample.io_chain/sample_kind/compositing/feather_px/probe_group/pair_id: Optional[str|int]`；`backfill(in_path, out_path) -> int`（返回回填行数）。

- [ ] **Step 1: 失败测试**

```python
# tests/test_backfill_v7.py
import json
from forgery_pipeline import manifest
from forgery_pipeline.schema import Sample, TaskType
from scripts.backfill_manifest_v7 import backfill

def _legacy_rows(tmp_path):
    rows = [Sample(image_id="r0", image_path="a.jpg", is_fake=0,
                   task_type=TaskType.real_pristine),
            Sample(image_id="f0", image_path="b.jpg", is_fake=1, operator="inpaint",
                   mask_path="m.png", task_type=TaskType.localization)]
    p = tmp_path / "old.jsonl"; manifest.write_jsonl(p, rows); return p

def test_new_fields_roundtrip(tmp_path):
    s = Sample(image_id="x", image_path="x.png", is_fake=1,
               task_type=TaskType.localization, io_chain="decode>rs256>edit:m>png",
               sample_kind="edited", compositing="paste_feather", feather_px=8,
               probe_group="compositing_pair", pair_id="p0")
    p = tmp_path / "m.jsonl"; manifest.write_jsonl(p, [s])
    r = manifest.read_jsonl(p)[0]
    assert (r.io_chain, r.sample_kind, r.compositing, r.feather_px,
            r.probe_group, r.pair_id) == ("decode>rs256>edit:m>png", "edited",
                                          "paste_feather", 8, "compositing_pair", "p0")

def test_backfill_fills_legacy(tmp_path):
    p = _legacy_rows(tmp_path); out = tmp_path / "new.jsonl"
    assert backfill(p, out) == 2
    rows = manifest.read_jsonl(out)
    assert [r.sample_kind for r in rows] == ["real", "edited"]     # 按 is_fake 推断
    assert all(r.io_chain == "legacy" for r in rows)
    assert rows[1].compositing == "none" and rows[0].compositing is None  # 仅 masked 编辑行
```

- [ ] **Step 2: 跑红** `python3 -m pytest tests/test_backfill_v7.py -q` → ImportError/AttributeError。
- [ ] **Step 3: 实现** schema.py 在 `op_params` 行后追加：

```python
    io_chain: Optional[str] = None          # 逐节点处理链（PATCH 7.1），如 decode>rs512>edit:sd15_inpaint>png；旧行=legacy
    sample_kind: Optional[str] = None       # real / real_vae_rt / edited（PATCH 7.2）
    compositing: Optional[str] = None       # none / paste / paste_feather（PATCH 7.3，masked 算子必填）
    feather_px: Optional[int] = None        # paste_feather 羽化 σ（像素）
    probe_group: Optional[str] = None       # 成对 probe 组名（compositing_pair / nd_pair，PATCH 7.3/8.1）
    pair_id: Optional[str] = None           # 成对样本回链 id
```

`scripts/backfill_manifest_v7.py`：

```python
"""旧 manifest 回填 PATCH 7 字段（sample_kind 按 is_fake；compositing=none 仅 masked 编辑行；io_chain=legacy）。"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from forgery_pipeline import manifest

def backfill(in_path, out_path) -> int:
    rows = manifest.read_jsonl(in_path)
    for r in rows:
        r.sample_kind = r.sample_kind or ("edited" if r.is_fake else "real")
        r.io_chain = r.io_chain or "legacy"
        if r.is_fake and r.mask_path and r.compositing is None:
            r.compositing = "none"          # 历史 masked 编辑均为整图直出
    manifest.write_jsonl(out_path, rows)
    return len(rows)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); print("backfilled", backfill(a.inp, a.out), "rows")
```

- [ ] **Step 4: 跑绿**（本文件 + `python3 -m pytest -q` 全套）。
- [ ] **Step 5: Commit** `git commit -m "feat(schema): PATCH7/8 字段一次性定死 + 旧manifest回填脚本"`

### Task 2: composite() 纯函数

**Files:** Create `src/forgery_pipeline/compositing.py`、`tests/test_compositing.py`

**Interfaces (Produces):** `composite(orig_rgb_u8, gen_rgb_u8, mask01, mode="none", feather_px=8) -> np.ndarray(uint8)`；mask01 为 HxW float/bool，1=编辑区。

- [ ] **Step 1: 失败测试**

```python
# tests/test_compositing.py
import numpy as np, pytest
from forgery_pipeline.compositing import composite

def _pair(h=32, w=32):
    rng = np.random.default_rng(0)
    orig = rng.integers(0, 256, (h, w, 3), np.uint8)
    gen = rng.integers(0, 256, (h, w, 3), np.uint8)
    m = np.zeros((h, w), np.float32); m[8:20, 8:20] = 1.0
    return orig, gen, m

def test_none_returns_gen():
    o, g, m = _pair(); assert composite(o, g, m, "none") is g

def test_paste_exact_outside_inside():
    o, g, m = _pair(); out = composite(o, g, m, "paste")
    assert np.array_equal(out[m == 0], o[m == 0])     # 掩码外逐像素==orig
    assert np.array_equal(out[m == 1], g[m == 1])     # 掩码内==gen

def test_paste_feather_blends_band_exact_far_outside():
    o, g, m = _pair(); out = composite(o, g, m, "paste_feather", feather_px=2)
    assert np.array_equal(out[:2], o[:2])             # 远离羽化带 == orig
    band = out[7, 8:20]                               # 边界带为混合值
    assert not np.array_equal(band, o[7, 8:20]) and not np.array_equal(band, g[7, 8:20])

def test_shape_mismatch_raises():
    o, g, m = _pair()
    with pytest.raises(AssertionError):
        composite(o[:16], g, m, "paste")
```

- [ ] **Step 2: 跑红**（模块不存在）。
- [ ] **Step 3: 实现**（spec 7.3 参考实现 + 形状断言）：

```python
"""paste-back 显式化（PATCH 7.3）：none=整图直出 / paste=硬回贴 / paste_feather=羽化混合。"""
from __future__ import annotations
import cv2
import numpy as np

def composite(orig_rgb_u8, gen_rgb_u8, mask01, mode: str = "none",
              feather_px: int = 8) -> np.ndarray:
    if mode == "none":
        return gen_rgb_u8
    assert orig_rgb_u8.shape == gen_rgb_u8.shape, "orig/gen 分辨率必须一致（先对齐再混合）"
    assert orig_rgb_u8.shape[:2] == np.asarray(mask01).shape, "mask 与图像 HxW 不一致"
    m = np.asarray(mask01, np.float32)
    if mode == "paste_feather":
        m = np.clip(cv2.GaussianBlur(m, (0, 0), float(feather_px)), 0.0, 1.0)
    elif mode != "paste":
        raise ValueError(f"未知 compositing: {mode!r}")
    m = m[..., None]
    out = orig_rgb_u8.astype(np.float32) * (1 - m) + gen_rgb_u8.astype(np.float32) * m
    return np.clip(np.round(out), 0, 255).astype(np.uint8)
```

- [ ] **Step 4: 跑绿 + 全套。**
- [ ] **Step 5: Commit** `feat(compositing): composite() 纯函数（none/paste/paste_feather）`

### Task 3: canonical I/O（load_and_resize / save_canonical / chain）+ D0/D1/D3/probe 走全链

**Files:** Modify `src/forgery_pipeline/image_io.py`、`builders/d0_real.py`、`builders/d1_whole.py`、`builders/d3_web.py`、`builders/probe.py`；Create `tests/test_canonical_io.py`；Modify 受影响断言的既有测试（d2 在 Task 5）。

**Interfaces (Produces):** `load_and_resize(path, size:int|None=None)->np.ndarray`（同解码器+LANCZOS+中心裁剪方形可选）；`save_canonical(img, path)->None`（强制 `.png` 后缀，非 png 即 raise）；`chain(*nodes)->str`（`"decode","rs512","edit:x","png"` → `"decode>rs512>edit:x>png"`）。行字段约定：真实行 `io_chain=chain("decode", f"rs{S}", "png"), sample_kind="real"`；编辑行 `io_chain=chain("decode", f"rs{S}", f"edit:{generator_name}", "png"), sample_kind="edited"`；D1 全生成行 `chain(f"gen:{name}", f"rs{S}", "png")`。

- [ ] **Step 1: 失败测试**

```python
# tests/test_canonical_io.py
import numpy as np, pytest
from forgery_pipeline.image_io import load_and_resize, save_canonical, chain, load_image

def test_chain_joins_nodes():
    assert chain("decode", "rs512", "edit:m", "png") == "decode>rs512>edit:m>png"

def test_save_canonical_enforces_png(tmp_path):
    img = np.zeros((8, 8, 3), np.uint8)
    save_canonical(img, tmp_path / "a.png")
    assert (tmp_path / "a.png").exists()
    with pytest.raises(AssertionError):
        save_canonical(img, tmp_path / "b.jpg")

def test_load_and_resize_center_square(tmp_path):
    img = np.zeros((40, 80, 3), np.uint8); img[:, 40:] = 255   # 右半白
    save_canonical(img, tmp_path / "w.png")
    out = load_and_resize(tmp_path / "w.png", size=32)
    assert out.shape == (32, 32, 3)                            # 中心裁剪→方形→resize

def test_d0_real_rows_canonical(tmp_path):
    from forgery_pipeline.builders.d0_real import build_d0
    rows = build_d0(tmp_path, 3, backend="mock", seed=0)
    for r in rows:
        assert r.image_path.endswith(".png") and r.sample_kind == "real"
        assert r.io_chain and r.io_chain.startswith("decode>rs") and r.io_chain.endswith(">png")

def test_probe_strength_rows_canonical(tmp_path):
    from forgery_pipeline.config import GeneratorSpec
    from forgery_pipeline.builders.probe import run_probe
    from forgery_pipeline import manifest
    run_probe(tmp_path / "p", n_base=2, strengths=[0.5], operators=[],
              img2img_specs=[GeneratorSpec("g", "diffusion", "img2img")],
              inpainter_specs=[GeneratorSpec("i", "diffusion", "inpaint")], seed=0)
    rows = manifest.read_jsonl(tmp_path / "p" / "gate1_strength.jsonl")
    assert all(r.sample_kind == "edited" and "edit:g" in r.io_chain for r in rows)
```

- [ ] **Step 2: 跑红。**
- [ ] **Step 3: 实现** image_io.py 追加：

```python
def chain(*nodes: str) -> str:
    """io_chain 组装：节点用 '>' 连接（PATCH 7.1）。"""
    return ">".join(nodes)

def load_and_resize(path, size: int | None = None) -> np.ndarray:
    """统一载入：同解码器；size 给定时中心裁剪为方形后 LANCZOS 缩放（真/假共享）。"""
    img = load_image(path)
    if size is not None:
        h, w = img.shape[:2]; side = min(h, w)
        y0, x0 = (h - side) // 2, (w - side) // 2
        img = img[y0:y0 + side, x0:x0 + side]
        img = np.asarray(PILImage.fromarray(img).resize((size, size), PILImage.LANCZOS))
    return img

def save_canonical(img: np.ndarray, path) -> None:
    """统一存储出口：主库一律 PNG（无损，不再引入有损层，PATCH 7.1）。"""
    assert str(path).endswith(".png"), f"canonical 存储必须 PNG: {path}"
    save_image(img, path)
```

d0_real.py：`rel = f"D0_real_pristine/{iid}.png"`；`image_io.save_canonical(img, out_dir / rel)`；Sample 增 `sample_kind="real", io_chain=image_io.chain("decode", f"rs{img.shape[0]}", "png")`（mock 源图已方形；真实源经 fetch 已 512——`rs{H}` 记录实际入库分辨率）。
d1_whole.py：找到 Sample(...) 构造处，图像保存改 `save_canonical`（路径 .png），行增 `sample_kind="edited", io_chain=image_io.chain(f"gen:{spec.name}", f"rs{img.shape[0]}", "png")`。
d3_web.py：同理 `sample_kind="edited", io_chain=chain("decode", f"rs{H}", f"edit:{gen_name}", "png")`，输出改 .png/save_canonical。
probe.py：`build_probe_strength` 行增 `sample_kind="edited", io_chain=image_io.chain("decode", f"rs{img.shape[0]}", f"edit:{spec.name}", "png")`；`build_probe_operator` img2img 行同 strength；masked 行在 Task 5 一并加（含 compositing）。D0 由 build_d0 已覆盖。
既有测试适配：凡断言 `.jpg` 后缀/路径的（`test_builder_d0/d1/d3`、`test_end_to_end` 若有）改 `.png`——逐个跑失败清单修正，**不放宽任何行为断言**。

- [ ] **Step 4: 跑绿 + 全套**（预期需修 3–6 处后缀断言）。
- [ ] **Step 5: Commit** `feat(io): canonical I/O（PNG+io_chain+sample_kind）真假共享同链`

### Task 4: VAE 往返负样本（mock+real 后端 + pipeline 分层插入）

**Files:** Modify `backends/base.py`、`backends/mock.py`、`backends/real/diffusers_gen.py`、`backends/registry.py`、`pipeline.py`、`config.py`；Create `tests/test_vae_rt.py`。

**Interfaces (Produces):** `base.VaeRoundtrip.roundtrip(img: np.ndarray) -> np.ndarray`；`registry.get_vae_rt(backend) -> VaeRoundtrip`（mock→`MockVaeRoundtrip`（vae 名 `mock`），real→`SDVaeRoundtrip`（vae 名 `sd15`，懒加载 fp32））；`PipelineConfig.vae_rt_frac: float = 0.15`；pipeline 在 split 后对 `split ∈ {train, test_a, test_f}` 的 `sample_kind=="real"` 行按 frac 确定性抽样（stable_hash(image_id)），生成新行：`is_fake=0, sample_kind="real_vae_rt", real_image_path=源.image_path, io_chain=源链插入 vae_rt:{vae} 于 png 前, split=源.split, image_id=源+"__vaert"`。

- [ ] **Step 1: 失败测试**

```python
# tests/test_vae_rt.py
import numpy as np
from forgery_pipeline.backends import registry

def test_mock_vae_rt_global_deterministic():
    rt = registry.get_vae_rt("mock")
    img = np.random.default_rng(0).integers(0, 256, (32, 32, 3), np.uint8)
    a, b = rt.roundtrip(img), rt.roundtrip(img)
    assert np.array_equal(a, b)                       # 确定性
    assert a.shape == img.shape and a.dtype == np.uint8
    assert (a != img).mean() > 0.5                    # 全局印记：过半像素被触碰

def test_real_vae_rt_lazy():
    rt = registry.get_vae_rt("real")
    assert rt._vae is None                            # 构造不加载模型

def test_pipeline_inserts_vae_rt_rows(tmp_path):
    from forgery_pipeline.config import PipelineConfig, StageScales, GeneratorSpec
    from forgery_pipeline.pipeline import run_pipeline
    cfg = PipelineConfig(
        out_dir=str(tmp_path / "run"), seed=0, backend="mock",
        stages={"d0": True, "d1": False, "d2": True, "d3": True, "d4": False,
                "postprocess": False, "split": True},
        scales=StageScales(d0=12, d2=6, d3=4),
        inpainters=[GeneratorSpec("i1", "diffusion", "inpaint")],
        vae_rt_frac=0.5)
    run_pipeline(cfg)
    from forgery_pipeline import manifest
    rows = manifest.read_jsonl(tmp_path / "run" / "manifest.jsonl")
    rt = [r for r in rows if r.sample_kind == "real_vae_rt"]
    assert rt and all(r.is_fake == 0 and "vae_rt:mock" in r.io_chain for r in rt)
    assert all(r.split == next(x.split for x in rows if x.image_path == r.real_image_path)
               for r in rt)                            # 与源同 split（同 origin-group 防泄漏）
```

- [ ] **Step 2: 跑红。**
- [ ] **Step 3: 实现**
  base.py 追加：

```python
class VaeRoundtrip(ABC):
    """真实图过 VAE encode→decode（无扩散/编辑/掩码）→ DRCT 式硬负样本（PATCH 7.2）。"""
    @abstractmethod
    def roundtrip(self, img: np.ndarray) -> np.ndarray: ...
```

  mock.py 追加（并给 `MockInpainter.inpaint` 的 return 前加一行全局印记——见 docstring 注明"模拟整图 VAE 直出"）：

```python
class MockVaeRoundtrip(base.VaeRoundtrip):
    """确定性全局印记：轻高斯 + uint8 往返，模拟 VAE 重采样足迹。"""
    name = "mock"
    def roundtrip(self, img: np.ndarray) -> np.ndarray:
        out = cv2.GaussianBlur(img, (0, 0), 0.6)
        return np.clip(out.astype(np.int16) + ((img.astype(np.int16) - out) // 3), 0, 255).astype(np.uint8)
```

  `MockInpainter.inpaint` 在 `meta = {...}` 前加：`out = MockVaeRoundtrip().roundtrip(out)  # 模拟真实管线整图 VAE 直出（PATCH 7.3 审计口径）`。
  diffusers_gen.py 追加：

```python
class SDVaeRoundtrip:
    """SD1.5 VAE encode→decode（fp32 防 NaN，懒加载；PATCH 7.2）。名称 sd15 记入 io_chain。"""
    name = "sd15"
    def __init__(self, model_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
                 device: str = "cuda"):
        self.model_id, self.device, self._vae = model_id, device, None
    def roundtrip(self, img):
        import torch
        from diffusers import AutoencoderKL
        if self._vae is None:
            self._vae = AutoencoderKL.from_pretrained(
                self.model_id, subfolder="vae", torch_dtype=torch.float32
            ).to(self.device).eval()
        import numpy as np
        x = (torch.from_numpy(img).float().permute(2, 0, 1)[None] / 127.5 - 1.0).to(self.device)
        with torch.no_grad():
            y = self._vae.decode(self._vae.encode(x).latent_dist.mean).sample
        y = ((y.clamp(-1, 1) + 1) * 127.5).round().byte()[0].permute(1, 2, 0).cpu().numpy()
        return y
```

  registry.py 追加 `get_vae_rt(backend)`（mock/real 分派，签名同其余 get_*）。
  config.py `PipelineConfig` 追加 `vae_rt_frac: float = 0.15`；`load_config` 读 `data.get("vae_rt_frac", 0.15)`。
  pipeline.py 在 `assign_splits(...)`+`check_leakage` 之后、`write_jsonl(manifest)` 之前插入：

```python
    if st.get("split") and cfg.vae_rt_frac > 0:
        rt = registry.get_vae_rt(cfg.backend)
        extra: list[Sample] = []
        for s in samples:
            if s.sample_kind != "real" or s.split not in {"train", "test_a", "test_f"}:
                continue
            if (stable_hash(s.image_id + "vaert") % 1000) / 1000.0 >= cfg.vae_rt_frac:
                continue
            img = image_io.load_image(out / s.image_path)
            rel = str(Path(s.image_path).with_name(Path(s.image_path).stem + "__vaert.png"))
            image_io.save_canonical(rt.roundtrip(img), out / rel)
            v = s.model_copy(deep=True)
            v.image_id = s.image_id + "__vaert"; v.image_path = rel
            v.sample_kind = "real_vae_rt"; v.real_image_path = s.image_path
            v.io_chain = s.io_chain.replace(">png", f">vae_rt:{rt.name}>png") if s.io_chain else f"vae_rt:{rt.name}"
            extra.append(v)
        samples += extra
        leaks = check_leakage(samples)
        if leaks:
            raise RuntimeError("vae_rt 插入后泄漏: " + "; ".join(leaks))
```

  （registry import 已在 pipeline？无则加 `from forgery_pipeline.backends import registry`。）
- [ ] **Step 4: 跑绿 + 全套**（MockInpainter 全局印记可能触发 2–3 处"掩码外不变"类断言——按 spec 语义更新那些断言为"掩码外≈原图（全局轻印记）/paste 后逐像素相等"，在测试注释注明 PATCH 7）。
- [ ] **Step 5: Commit** `feat(vae-rt): VAE 往返硬负样本（mock/real 后端 + split 后分层插入 + 泄漏复查）`

### Task 5: compositing 进生成路径（D2 50/50 + probe 成对 + 字段）

**Files:** Modify `builders/d2_local.py`、`builders/probe.py`、`config.py`（feather 默认）、`backends/real/diffusers_gen.py`（inpaint 审计注释）；Modify `tests/test_probe.py`/`test_builder_d2.py` 相应断言；Create 测试于 `tests/test_compositing.py` 追加。

**Interfaces (Produces):** D2 masked 行：`compositing ∈ {"none","paste_feather"}`（`stable_hash(iid+"comp")%2` 50/50）、`feather_px=8`（仅 feather）、io_chain 的 edit 节点后缀 `+paste_feather` 不加（compositing 已有独立字段，链保持 edit:{name}）；probe `run_probe(..., compositing_pairs: int = 0)`：>0 时额外生成 N 组成对样本（同 base/mask/seed，两行 compositing=none/paste_feather，`probe_group="compositing_pair"`、`pair_id=f"cp{k:04d}"`），写入 `gate2_operator.jsonl` 之后、manifest 一并。

- [ ] **Step 1: 失败测试**（追加到 tests/test_compositing.py）

```python
def test_d2_fifty_fifty_compositing(tmp_path):
    from forgery_pipeline.builders.d0_real import build_d0
    from forgery_pipeline.builders.d2_local import build_d2
    from forgery_pipeline.config import GeneratorSpec
    bases = build_d0(tmp_path, 8, backend="mock", seed=0)
    rows = build_d2(tmp_path, bases, 16, [GeneratorSpec("i1", "diffusion", "inpaint")],
                    backend="mock", seed=0)
    comps = {r.compositing for r in rows}
    assert comps == {"none", "paste_feather"}          # 两种都出现
    assert all(r.feather_px == 8 for r in rows if r.compositing == "paste_feather")
    assert all(r.sample_kind == "edited" and r.io_chain for r in rows)

def test_probe_compositing_pairs(tmp_path):
    from forgery_pipeline.builders.probe import run_probe
    from forgery_pipeline.config import GeneratorSpec
    from forgery_pipeline import manifest
    run_probe(tmp_path / "p", n_base=3, strengths=[0.5], operators=["inpaint"],
              img2img_specs=[GeneratorSpec("g", "diffusion", "img2img")],
              inpainter_specs=[GeneratorSpec("i", "diffusion", "inpaint")],
              seed=0, compositing_pairs=3)
    rows = [r for r in manifest.read_jsonl(tmp_path / "p" / "manifest.jsonl")
            if r.probe_group == "compositing_pair"]
    assert len(rows) == 6                               # 3 组 × 2 行
    by_pair = {}
    for r in rows: by_pair.setdefault(r.pair_id, []).append(r)
    for pid, pr in by_pair.items():
        assert len(pr) == 2 and {p.compositing for p in pr} == {"none", "paste_feather"}
        assert pr[0].mask_path and pr[0].seed == pr[1].seed
        assert pr[0].real_image_path == pr[1].real_image_path
```

- [ ] **Step 2: 跑红。**
- [ ] **Step 3: 实现**
  d2_local.py：`painter.inpaint(...)` 之后：

```python
        mode = "paste_feather" if stable_hash(iid + "comp") % 2 else "none"
        fake = composite(img, fake, (mask > 127).astype(np.float32), mode, feather_px=8)
```

  行增 `compositing=mode, feather_px=(8 if mode == "paste_feather" else None), sample_kind="edited", io_chain=image_io.chain("decode", f"rs{img.shape[0]}", f"edit:{inp.name}", "png")`；输出改 `.png` + `save_canonical`（掩码路径不变）。iid 在 composite 前已知（先算 iid 再 composite）。
  probe.py：`run_probe` 增参 `compositing_pairs: int = 0`；新函数 `build_compositing_pairs(out, bases, inpainter_specs, n_pairs, backend, seed)`：取前 n_pairs 个 base，box 掩码（`_mask_for("box", ...)`, rng=seed+k），同一 painter+seed 生成一次 gen，两行分别 `composite(..., "none")` / `composite(..., "paste_feather", 8)` 存两个 PNG，Sample 字段含 `probe_group="compositing_pair", pair_id=f"cp{k:04d}", compositing, feather_px, mask_path, operator="inpaint", sample_kind="edited", io_chain=...`；`samples = bases + g1 + g2 + pairs` 入 manifest。
  diffusers_gen.py `inpaint()`：调用处加审计注释与守卫：`assert "padding_mask_crop" not in params, "禁止隐式 paste-back（PATCH 7.3 审计）：compositing 必须显式"`（现调用不传该参，注明 diffusers 默认无 overlay 路径）。
- [ ] **Step 4: 跑绿 + 全套**（d2 既有测试的 .jpg 断言/字段数适配）。
- [ ] **Step 5: Commit** `feat(compositing): D2 50/50 显式回贴 + probe 成对样本 + 隐式回贴审计守卫`

### Task 6: validator V1–V7 + stats 扩展

**Files:** Create `src/forgery_pipeline/validate.py`、`tests/test_validate_v7.py`；Modify `manifest.py`（stats）、`cli.py`（validate-manifest 接入 + `--profile`）。

**Interfaces (Produces):** `validate.check_all(samples, profile="auto") -> list[str]`（空=通过；每条 `"V2: ..."` 带差集）；stats 增 `by_sample_kind`、`by_compositing`、`io_chain_by_fake_split`（`{split: {io_chain_nongen: {"real": n, "fake": n}}}`）。非生成链归一函数 `nongen_chain(io_chain) -> str`（去 edit:*/vae_rt:*/gen:* 节点、忽略首个 decode；legacy → "legacy"）。

- [ ] **Step 1: 失败测试**

```python
# tests/test_validate_v7.py
from forgery_pipeline.validate import check_all, nongen_chain
from forgery_pipeline.schema import Sample, TaskType

def _r(**kw):
    d = dict(image_id=kw.pop("i"), image_path=kw.pop("p"), is_fake=kw.pop("f"),
             task_type=TaskType.real_pristine if not kw.get("mask_path")
             else TaskType.localization, split="train",
             sample_kind="real" if kw["f"] == 0 else "edited") if False else None
    ...
```

（实际测试写直白构造，覆盖：V1 同组混 png/jpg → FAIL 且 postprocess 行豁免；V2 real 与 fake 非生成链不等 → FAIL 打印差集、`io_chain=legacy` 豁免；V3 masked 行缺 compositing / feather 缺 feather_px → FAIL；V4 profile="run" 时 train 无 real_vae_rt → FAIL、占比越界 → FAIL、profile="auto" 且无 vae_rt 行 → 跳过；V5 backfill 后旧 manifest V1–V4 过；V6 instruct_edit 行 op_params 非法 JSON/缺 image_guidance_scale → FAIL；V7 pair_id 出现次数≠2 / 组内 seed 不一致 / compositing_pair 组内除 compositing 外 generator_name 不同 → FAIL。每条一个测试函数，共 ~12 个。）

- [ ] **Step 2: 跑红。**
- [ ] **Step 3: 实现** validate.py：`nongen_chain` 拆 `>`、过滤 `edit:`/`vae_rt:`/`gen:` 前缀、首节点 `decode` 丢弃、重组；V1 按（split 内非退化行）后缀集合唯一（`postprocess` 非空行豁免）；V2 对每 split 比较 real/fake 的 `nongen_chain` 集合（legacy 豁免、差集入消息）；V3/V6/V7 逐行断言；V4 对 profile 与 vae_rt 行存在性决定是否执行，占比区间默认 `[0.05, 0.35]`（`check_all(..., vae_rt_range=(0.05,0.35))`）。cli.py `_cmd_validate` 在 schema 校验后调 `check_all(samples, profile=args.profile)`，非空则打印并 return 1；`p_val.add_argument("--profile", default="auto", choices=["auto","run"])`。manifest.stats 增三键（`nongen_chain` 从 validate import——validate 不得反向 import manifest，避免环）。
- [ ] **Step 4: 跑绿 + 全套。**
- [ ] **Step 5: Commit** `feat(validate): V1–V7 断言集 + stats io_chain/sample_kind/compositing 计数`

### Task 7: 7.5 冒烟验收（e2e 测试 + 成对断言脚本 + vae_rt 分布图）

**Files:** Create `scripts/assert_compositing_pairs.py`；Modify `tests/test_end_to_end.py`（或新增 `tests/test_e2e_patch7.py`）。

**Interfaces (Produces):** `assert_compositing_pairs.check(probe_dir) -> list[str]`（空=通过）：对每个 `compositing_pair`——paste_feather 行在 `mask 膨胀(4×feather_px) 之外`逐像素 == 对应底图；none 行与底图全图不等比例 > 0.5。脚本 CLI：`python scripts/assert_compositing_pairs.py --probe <dir>`，退出码 0/1。

- [ ] **Step 1: 失败测试**（e2e：mock 全链 → V1–V7 全过 + 成对断言过 + stats 键在）

```python
# tests/test_e2e_patch7.py
def test_mock_smoke_passes_v1_v7_and_pair_assertions(tmp_path):
    from forgery_pipeline.config import PipelineConfig, StageScales, GeneratorSpec
    from forgery_pipeline.pipeline import run_pipeline
    from forgery_pipeline import manifest
    from forgery_pipeline.validate import check_all
    cfg = PipelineConfig(out_dir=str(tmp_path / "run"), seed=0, backend="mock",
                         stages={"d0": True, "d1": True, "d2": True, "d3": True,
                                 "d4": False, "postprocess": True, "split": True},
                         scales=StageScales(d0=20, d1_per_generator=2, d2=10, d3=6),
                         generators=[GeneratorSpec("g1", "gan", "whole")],
                         inpainters=[GeneratorSpec("i1", "diffusion", "inpaint")],
                         vae_rt_frac=0.4)
    st = run_pipeline(cfg)
    rows = manifest.read_jsonl(tmp_path / "run" / "manifest.jsonl")
    assert check_all(rows, profile="run") == []
    assert "by_sample_kind" in st and "io_chain_by_fake_split" in st
    # 成对 probe
    from forgery_pipeline.builders.probe import run_probe
    run_probe(tmp_path / "p", n_base=4, strengths=[0.5], operators=["inpaint"],
              img2img_specs=[GeneratorSpec("g", "diffusion", "img2img")],
              inpainter_specs=[GeneratorSpec("i", "diffusion", "inpaint")],
              seed=0, compositing_pairs=4)
    from scripts.assert_compositing_pairs import check
    assert check(tmp_path / "p") == []
```

- [ ] **Step 2: 跑红。**
- [ ] **Step 3: 实现** scripts/assert_compositing_pairs.py：读 manifest 中 `probe_group=="compositing_pair"` 行按 pair_id 配对；对 feather 行：`band = cv2.dilate(mask, ones(k,k))`（k=4×feather_px+1），断言 `np.array_equal(img[band==0], base[band==0])`；对 none 行：`(img != base).mean() > 0.5`（mock 全局印记/real VAE 足迹均满足）；guarded matplotlib：`--plot data/vae_rt_residual.png` 时画 real vs real_vae_rt 的 |img−base| 直方图（7.5 记录项，无阈值）。
- [ ] **Step 4: 跑绿 + 全套。**
- [ ] **Step 5: Commit** `test(e2e): PATCH7 mock 冒烟（V1–V7 + 成对断言 + 残差分布记录）`

### Task 8: 文档同步 + 合并

**Files:** Modify `docs/PATCHES.md`、`docs/paper_experiment_plan_2026-07-15.md`（§3 B2/B3 勾选 canonical/sample_kind/compositing 项）、`docs/GATE_DATA.md`（字段表 += io_chain/sample_kind/compositing/feather_px/probe_group/pair_id/op_params）。

- [ ] **Step 1:** PATCHES.md 追加"PATCH 7 ✅ 完成记录"（涉及文件/验收结果/V2 谱系适配说明——D1 无 decode 的 strip 规则偏差及理由）。
- [ ] **Step 2:** experiment plan §3 B2/B3 对应条目更新（canonical I/O、`real_vae_rt` 配比 0.15、compositing 50/50、成对 probe 已就绪——B3 阻断解除，标注"待 GPU 侧 real 冒烟复核 SDVaeRoundtrip"）。
- [ ] **Step 3:** GATE_DATA.md 字段速查表补 7 个新字段一行一句。
- [ ] **Step 4:** `python3 -m pytest -q` 全绿后：`git checkout main && git merge feat/patch7-canonical-io && python3 -m pytest -q && git push origin main && git branch -d feat/patch7-canonical-io`。

---

## Self-Review 结论

- **覆盖**：7.1（Task 3）、7.2（Task 4）、7.3 全部四点+回填（Task 1/2/5）、7.4 V1–V5（Task 6）、8.5 V6/V7（Task 6）、7.5 三项验收（Task 7）、文档同步（Task 8）。7.6"不改什么"以约束句写入 Global Constraints。
- **偏差已记录**：V2 非生成链 strip 规则含 `gen:*`+首 `decode`（D1 结构性问题）；mock inpaint 加全局印记（忠实模拟整图 VAE 直出，使 7.5 断言在 mock 上可测）。
- **类型一致**：`composite(orig,gen,mask01,mode,feather_px)`、`chain(*nodes)`、`get_vae_rt(backend).roundtrip(img)`、`check_all(samples,profile)`、`run_probe(...,compositing_pairs=0)` 在各任务间签名一致。
- **GPU 留白**：`SDVaeRoundtrip` 与 real 后端 compositing 只写代码不本地运行，GPU 冒烟归 Phase B 开机首日清单。

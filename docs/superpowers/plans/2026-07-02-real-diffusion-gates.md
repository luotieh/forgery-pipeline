# 真实扩散信号跑真实闸门 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 落地真实底图源 + SD2 img2img/inpaint 生成 + SD2 多σ Tweedie 残差 extractor，在 RTX 4060 上用 `--extractor real` 跑出 gate1/2/3 的真实 VERDICT。

**Architecture:** 真实后端全部**懒加载**（`__init__` 不 import diffusers、不占显存；首次使用才 `_ensure` 加载）→ pytest 保持 CPU-only 快绿（只测构造与 `LocalImageSource`）；真实 GPU 计算走 Task 4 执行冒烟。fp16 UNet + fp32 VAE + attention slicing + 512。

**Tech Stack:** diffusers/transformers/accelerate/safetensors（新装）、torch 2.12+cu130（已装）、SD2-base + SD2-inpainting 权重、picsum.photos 底图。

## Global Constraints

- 真实后端模块顶层**不 import diffusers/torch**（只在 `_ensure`/方法内 import）；`__init__` 仅存配置 → 无 GPU/无 diffusers 也能构造，pytest CPU-only。
- fp16 UNet、**fp32 VAE**（防 NaN）、`enable_attention_slicing()`、512 分辨率；随机走显式 `torch.Generator` seed。
- `backend="real"`（区别于旧 `real:diffusers` 提示串，后者仍走 `_unsupported`）。
- 残差 residual_stack 输出 `(K,H,W) float[0,1]`：latent 64×64 计算 → 上采样到图像分辨率 → 按 p99 clip 归一化。
- 产物写 `data/`（gitignore）；权重缓存 `~/.cache/huggingface`（一次性 ~10GB）。
- 真实 gate 判定**如实报告**（PASS/WEAK/FAIL 皆可），不美化。

---

## File Structure

| 文件 | 职责 |
|---|---|
| `scripts/fetch_real_images.py` | 从 picsum 下载真实底图到 data/real_base |
| `src/forgery_pipeline/backends/real/local_source.py` | LocalImageSource（读本地图目录） |
| `src/forgery_pipeline/backends/real/diffusers_gen.py`（改） | DiffusersImg2Img + DiffusersInpainter（懒加载真实实现） |
| `src/forgery_pipeline/backends/registry.py`（改） | `backend="real"` 解析 |
| `checking/extractor.py`（改） | DiffusersSD2Residual 真实多σ残差 + get_extractor("real") 惰性 |
| `configs/probe.real.yaml` | 真实 probe 配置 |
| `pyproject.toml`（改） | `[real]` extra 增 accelerate/safetensors |
| `tests/test_backends_real.py` | LocalImageSource + registry real 构造（CPU） |
| `tests/test_checking_extractor.py`（改） | get_extractor("real") 惰性返回实例 |
| `docs/real_gate_results_2026-07-02.md` | Task 4 GPU 冒烟真实结果记录 |

---

## Task 1: 真实底图源（fetch 脚本 + LocalImageSource）

**Files:** Create `scripts/fetch_real_images.py`, `src/forgery_pipeline/backends/real/local_source.py`, `tests/test_backends_real.py`; Modify `src/forgery_pipeline/backends/registry.py`

**Interfaces:**
- Produces：`LocalImageSource(root, size=512, seed=0)` 实现 `iter_images(n)`；`registry.get_image_source("real", seed)` → LocalImageSource（root 取 env `FORGERY_REAL_IMAGE_DIR` 或 `data/real_base`）

- [x] **Step 1: 写失败测试** `tests/test_backends_real.py`

```python
import numpy as np
from forgery_pipeline import image_io
from forgery_pipeline.backends.real.local_source import LocalImageSource
from forgery_pipeline.backends import registry


def test_local_image_source_reads_and_crops(tmp_path):
    for i in range(2):
        img = np.random.default_rng(i).integers(0, 256, (300, 400, 3), dtype=np.uint8)
        image_io.save_image(img, tmp_path / f"p{i}.jpg")
    got = list(LocalImageSource(tmp_path, size=128).iter_images(2))
    assert len(got) == 2
    im, meta = got[0]
    assert im.shape == (128, 128, 3) and im.dtype == np.uint8
    assert meta["source_dataset"] == "local"


def test_registry_real_image_source(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGERY_REAL_IMAGE_DIR", str(tmp_path))
    assert isinstance(registry.get_image_source("real"), LocalImageSource)
```

- [x] **Step 2: 运行确认失败** → FAIL

- [x] **Step 3: 写 `src/forgery_pipeline/backends/real/local_source.py`**

```python
"""真实底图源：读本地图目录，中心裁剪 + resize 到工作分辨率。"""
from __future__ import annotations
from pathlib import Path
from typing import Iterator
import cv2
import numpy as np
from forgery_pipeline import image_io
from forgery_pipeline.backends import base

_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _center_crop_resize(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    return cv2.resize(img[y0:y0 + s, x0:x0 + s], (size, size))


class LocalImageSource(base.ImageSource):
    def __init__(self, root, size: int = 512, seed: int = 0):
        self.root = Path(root); self.size = size; self.seed = seed

    def iter_images(self, n: int) -> Iterator[tuple[np.ndarray, dict]]:
        files = sorted(p for p in self.root.rglob("*") if p.suffix.lower() in _EXTS)
        count = 0
        for p in files:
            if count >= n:
                break
            try:
                img = image_io.load_image(p)
            except Exception:
                continue
            yield _center_crop_resize(img, self.size), {
                "source_dataset": "local", "camera_model": None,
                "resolution": [self.size, self.size], "license": "unknown"}
            count += 1
```

- [x] **Step 4: 改 `registry.py` 的 `get_image_source`**

```python
def get_image_source(backend: str, seed: int = 0) -> base.ImageSource:
    if backend == "mock":
        return mock.MockImageSource(seed=seed)
    if backend == "real":
        import os
        from forgery_pipeline.backends.real.local_source import LocalImageSource
        return LocalImageSource(os.environ.get("FORGERY_REAL_IMAGE_DIR", "data/real_base"),
                                seed=seed)
    _unsupported(backend)
```

- [x] **Step 5: 写 `scripts/fetch_real_images.py`**

```python
"""从 picsum.photos 确定性下载真实底图（真实 Unsplash 照片，经 env 代理）。"""
from __future__ import annotations
import argparse
import urllib.request
from pathlib import Path


def fetch(out_dir="data/real_base", n=200, size=512, start_id=0) -> int:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    got, i = 0, start_id
    limit = start_id + n * 4
    while got < n and i < limit:
        url = f"https://picsum.photos/id/{i}/{size}/{size}.jpg"
        try:
            urllib.request.urlretrieve(url, out / f"real_{got:04d}.jpg")
            got += 1
        except Exception:
            pass
        i += 1
    print(f"fetched {got} images -> {out}")
    return got


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/real_base")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--size", type=int, default=512)
    raise SystemExit(0 if fetch(**vars(ap.parse_args())) else 1)
```

- [x] **Step 6: 运行确认通过**

Run: `pytest tests/test_backends_real.py -q` → PASS

- [x] **Step 7: 提交**

```bash
git add scripts/fetch_real_images.py src/forgery_pipeline/backends/real/local_source.py src/forgery_pipeline/backends/registry.py tests/test_backends_real.py
git commit -m "feat(real): LocalImageSource + picsum 底图下载 + registry real 源"
```

---

## Task 2: 真实 diffusers img2img / inpaint（懒加载）

**Files:** Modify `src/forgery_pipeline/backends/real/diffusers_gen.py`, `src/forgery_pipeline/backends/registry.py`, `tests/test_backends_real.py`

**Interfaces:**
- Produces：`DiffusersImg2Img(model_id, device, dtype)`（`img2img(image,prompt,strength,params)->(img,meta)`）；`DiffusersInpainter(...)`（`inpaint(image,mask,prompt,params)->(img,meta)`）；registry `get_img2img/get_inpainter/get_segmenter("real")`。构造懒加载（不 import diffusers）。

- [x] **Step 1: 追加失败测试** `tests/test_backends_real.py`

```python
def test_registry_real_generators_lazy():
    from forgery_pipeline.backends.real.diffusers_gen import DiffusersImg2Img, DiffusersInpainter
    from forgery_pipeline.backends import mock
    assert isinstance(registry.get_img2img("real", "x", "y"), DiffusersImg2Img)
    assert isinstance(registry.get_inpainter("real", "x", "y"), DiffusersInpainter)
    # probe 用几何掩码，real segmenter 占位为 mock
    assert isinstance(registry.get_segmenter("real"), mock.MockSegmenter)
    # 构造不触发模型加载（无 GPU/无 diffusers 也能构造）
    g = DiffusersImg2Img()
    assert g._pipe is None
```

- [x] **Step 2: 运行确认失败** → FAIL

- [x] **Step 3: 整体替换 `src/forgery_pipeline/backends/real/diffusers_gen.py`**

```python
"""真实 diffusers 生成后端（SD2）。懒加载：__init__ 不 import diffusers/不占显存。"""
from __future__ import annotations
import cv2
import numpy as np
from PIL import Image
from forgery_pipeline.backends import base

_SD2 = "stabilityai/stable-diffusion-2-base"
_SD2_INPAINT = "stabilityai/stable-diffusion-2-inpainting"


def _to_uint8_like(pil_img, ref: np.ndarray) -> np.ndarray:
    arr = np.asarray(pil_img.convert("RGB"), np.uint8)
    if arr.shape[:2] != ref.shape[:2]:
        arr = cv2.resize(arr, (ref.shape[1], ref.shape[0]))
    return arr


class DiffusersImg2Img(base.Img2ImgGenerator):
    def __init__(self, model_id: str = _SD2, device: str = "cuda", dtype: str = "fp16"):
        self.model_id, self.device, self.dtype = model_id, device, dtype
        self.name, self.family = "stable-diffusion-2-img2img", "diffusion"
        self._pipe = None

    def _ensure(self):
        if self._pipe is not None:
            return
        import torch
        from diffusers import AutoPipelineForImage2Image
        self._pipe = AutoPipelineForImage2Image.from_pretrained(
            self.model_id, torch_dtype=torch.float16).to(self.device)
        self._pipe.set_progress_bar_config(disable=True)
        self._pipe.enable_attention_slicing()

    def img2img(self, image, prompt, strength, params):
        import torch
        self._ensure()
        seed = int(params.get("seed", 0))
        g = torch.Generator(self.device).manual_seed(seed)
        out = self._pipe(prompt=prompt or "a realistic high quality photo",
                         image=Image.fromarray(image), strength=float(strength),
                         num_inference_steps=int(params.get("steps", 30)),
                         guidance_scale=float(params.get("cfg_scale", 7.5)), generator=g)
        meta = {"generator_name": self.name, "generator_family": self.family, "seed": seed,
                "strength": float(strength), "steps": int(params.get("steps", 30)),
                "cfg_scale": float(params.get("cfg_scale", 7.5)), "sampler": "default"}
        return _to_uint8_like(out.images[0], image), meta


class DiffusersInpainter(base.Inpainter):
    def __init__(self, model_id: str = _SD2_INPAINT, device: str = "cuda", dtype: str = "fp16"):
        self.model_id, self.device, self.dtype = model_id, device, dtype
        self.name, self.family = "stable-diffusion-2-inpaint", "diffusion"
        self._pipe = None

    def _ensure(self):
        if self._pipe is not None:
            return
        import torch
        from diffusers import AutoPipelineForInpainting
        self._pipe = AutoPipelineForInpainting.from_pretrained(
            self.model_id, torch_dtype=torch.float16).to(self.device)
        self._pipe.set_progress_bar_config(disable=True)
        self._pipe.enable_attention_slicing()

    def inpaint(self, image, mask, prompt, params):
        import torch
        self._ensure()
        seed = int(params.get("seed", 0))
        g = torch.Generator(self.device).manual_seed(seed)
        out = self._pipe(prompt=prompt or "a realistic object, high quality",
                         image=Image.fromarray(image), mask_image=Image.fromarray(mask),
                         num_inference_steps=int(params.get("steps", 30)),
                         guidance_scale=float(params.get("cfg_scale", 7.5)), generator=g)
        meta = {"generator_name": self.name, "generator_family": self.family, "seed": seed}
        return _to_uint8_like(out.images[0], image), meta
```

- [x] **Step 4: 改 `registry.py`** —— `get_img2img/get_inpainter/get_segmenter` 加 real 分支

```python
def get_img2img(backend: str, name: str, family: str) -> base.Img2ImgGenerator:
    if backend == "mock":
        return mock.MockImg2Img(name=name, family=family)
    if backend == "real":
        from forgery_pipeline.backends.real.diffusers_gen import DiffusersImg2Img
        return DiffusersImg2Img()
    _unsupported(backend)


def get_inpainter(backend: str, name: str, family: str) -> base.Inpainter:
    if backend == "mock":
        return mock.MockInpainter(name=name, family=family)
    if backend == "real":
        from forgery_pipeline.backends.real.diffusers_gen import DiffusersInpainter
        return DiffusersInpainter()
    _unsupported(backend)


def get_segmenter(backend: str, seed: int = 0) -> base.Segmenter:
    if backend == "mock":
        return mock.MockSegmenter(seed=seed)
    if backend == "real":
        return mock.MockSegmenter(seed=seed)  # probe 用几何掩码，占位无害
    _unsupported(backend)
```

- [x] **Step 5: 运行确认通过**

Run: `pytest tests/test_backends_real.py -q` → PASS

- [x] **Step 6: 提交**

```bash
git add src/forgery_pipeline/backends/real/diffusers_gen.py src/forgery_pipeline/backends/registry.py tests/test_backends_real.py
git commit -m "feat(real): SD2 img2img/inpaint 懒加载后端 + registry real 接线"
```

---

## Task 3: 真实 SD2 多σ残差 extractor + 配置/依赖

**Files:** Modify `checking/extractor.py`, `tests/test_checking_extractor.py`; Create `configs/probe.real.yaml`; Modify `pyproject.toml`

**Interfaces:**
- Produces：`DiffusersSD2Residual(model_id, device, timesteps)`（`residual_stack(image)->(K,H,W)float[0,1]`，懒加载）；`get_extractor("real")` 惰性返回该实例

- [x] **Step 1: 更新测试** `tests/test_checking_extractor.py`（`get_extractor("real")` 改为惰性返回实例）

```python
def test_get_extractor_and_real_lazy():
    from checking.extractor import get_extractor, DiffusersSD2Residual, MultiSigmaResidual
    assert isinstance(get_extractor("multisigma"), MultiSigmaResidual)
    ext = get_extractor("real")               # 惰性：构造不加载模型
    assert isinstance(ext, DiffusersSD2Residual)
    assert ext._unet is None
```

（删除旧的 `test_get_extractor_and_real_stub` 里 `pytest.raises(NotImplementedError)` 那条，替换为上面。）

- [x] **Step 2: 运行确认失败** → FAIL

- [x] **Step 3: 替换 `checking/extractor.py` 的 `DiffusersSD2Residual`**（整类替换现有 stub）

```python
class DiffusersSD2Residual(ResidualExtractor):
    """真实多 σ Tweedie 残差（冻结 SD2）。懒加载；需 diffusers + GPU。"""
    def __init__(self, model_id: str = "stabilityai/stable-diffusion-2-base",
                 device: str = "cuda", timesteps=(50, 150, 300, 500, 700)):
        self.model_id, self.device = model_id, device
        self.sigmas = list(timesteps)          # 多 σ = 多 t
        self._unet = None

    def _ensure(self):
        if self._unet is not None:
            return
        import torch
        from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
        from transformers import CLIPTextModel, CLIPTokenizer
        m = self.model_id
        self._vae = AutoencoderKL.from_pretrained(m, subfolder="vae",
                    torch_dtype=torch.float32).to(self.device).eval()
        self._unet = UNet2DConditionModel.from_pretrained(m, subfolder="unet",
                     torch_dtype=torch.float16).to(self.device).eval()
        self._abar = DDPMScheduler.from_pretrained(m, subfolder="scheduler").alphas_cumprod
        tok = CLIPTokenizer.from_pretrained(m, subfolder="tokenizer")
        te = CLIPTextModel.from_pretrained(m, subfolder="text_encoder",
             torch_dtype=torch.float16).to(self.device).eval()
        ids = tok("", padding="max_length", max_length=tok.model_max_length,
                  return_tensors="pt").input_ids.to(self.device)
        with torch.no_grad():
            self._null = te(ids)[0]

    def residual_stack(self, image: np.ndarray) -> np.ndarray:
        import hashlib
        import torch
        self._ensure()
        H, W = image.shape[:2]
        img512 = cv2.resize(image, (512, 512))
        x = (torch.from_numpy(img512).float().permute(2, 0, 1)[None] / 127.5 - 1.0
             ).to(self.device, torch.float32)
        seed_base = int.from_bytes(
            hashlib.sha1(np.ascontiguousarray(img512).tobytes()).digest()[:4], "big")
        maps = []
        with torch.no_grad():
            z0 = self._vae.encode(x).latent_dist.mean * self._vae.config.scaling_factor
            for t in self.sigmas:
                g = torch.Generator(self.device).manual_seed((seed_base + int(t)) & 0x7FFFFFFF)
                eps = torch.randn(z0.shape, generator=g, device=self.device, dtype=torch.float32)
                abar = self._abar[int(t)].to(self.device).float()
                zt = abar.sqrt() * z0 + (1 - abar).sqrt() * eps
                eps_hat = self._unet(zt.half(), int(t),
                                     encoder_hidden_states=self._null).sample.float()
                r_eps = ((eps - eps_hat) ** 2).mean(dim=1)[0]
                z0_hat = (zt - (1 - abar).sqrt() * eps_hat) / abar.sqrt()
                r_x = ((z0 - z0_hat) ** 2).mean(dim=1)[0]
                m = (r_eps + r_x).cpu().numpy()
                p99 = float(np.percentile(m, 99)) + 1e-8
                m = np.clip(m / p99, 0.0, 1.0)
                maps.append(cv2.resize(m.astype(np.float32), (W, H)))
        return np.stack(maps).astype(np.float32)
```

（`get_extractor` 已有 `if name == "real": return DiffusersSD2Residual()` 分支——现在它惰性、不再抛错；无需改 `get_extractor` 本体。）

- [x] **Step 4: 写 `configs/probe.real.yaml`**

```yaml
out_dir: data/probe_real
seed: 1234
backend: real
generators_config: configs/generators.yaml
n_base: 8
strengths: [0.1, 0.3, 0.5, 0.7, 0.9]
operators: [img2img, inpaint, outpaint, object_replacement, background_editing]
holdout_generators: []
```

- [x] **Step 5: 改 `pyproject.toml` 的 `[real]` extra**

```toml
real = ["torch", "diffusers", "transformers", "accelerate", "safetensors"]
```

- [x] **Step 6: 运行确认通过**

Run: `pytest tests/test_checking_extractor.py -q` → PASS

- [x] **Step 7: 全量 CPU 测试**

Run: `pytest -q` → 全绿（真实后端全懒加载，未触发 diffusers/GPU）

- [x] **Step 8: 提交**

```bash
git add checking/extractor.py tests/test_checking_extractor.py configs/probe.real.yaml pyproject.toml
git commit -m "feat(real): SD2 多σ Tweedie 残差 extractor + probe.real 配置 + [real] extra"
```

---

## Task 4: GPU 执行冒烟 + 真实结果记录

**Files:** Create `docs/real_gate_results_2026-07-02.md`

> 本任务在 RTX 4060 上执行真实生成与残差；非 pytest。每步记录真实输出。

- [x] **Step 1: 装 diffusers 栈**

Run: `pip install --user --break-system-packages diffusers transformers accelerate safetensors`
Expected: 成功；`python3 -c "import diffusers, transformers, accelerate; print('ok')"` 打印 ok。

- [x] **Step 2: 抓真实底图（先小量）**

Run: `python3 scripts/fetch_real_images.py --out data/real_base --n 16 --size 512`
Expected: `fetched 16 images -> data/real_base`；`ls data/real_base | wc -l` ≥ 16。

- [x] **Step 3: 真实 probe 冒烟（n_base=4，触发权重下载）**

Run: `FORGERY_REAL_IMAGE_DIR=data/real_base python3 -m forgery_pipeline.cli probe --config configs/probe.real.yaml --out data/probe_real --n-base 4`
Expected: 首次会下载 SD2-base + SD2-inpainting（~10GB，数分钟）；随后生成真实图，打印 stats（fake>0），无 OOM。抽查 `data/probe_real/probe/gate1_strength/*.png` 是真实编辑图（非合成色块）。
> OOM 兜底：在 `diffusers_gen._ensure` 临时加 `self._pipe.enable_sequential_cpu_offload()`（去掉 `.to(device)`）；或把 `configs/probe.real.yaml` 分辨率/步数调小。

- [x] **Step 4: 真实闸门冒烟**

Run: `FORGERY_REAL_IMAGE_DIR=data/real_base python3 -m checking.run_gates --run data/probe_real --probe data/probe_real --extractor real --out data/checking_report_real.json`
Expected: 打印 gate0/1/2/3(/4) 的 VERDICT + CAVEAT（extractor=real）；写 report。首个残差调用会加载 SD2 UNet/VAE/text_encoder（fp16/fp32），无 OOM。

- [x] **Step 5: 放量复跑（n_base≈50）**

Run:
```bash
python3 scripts/fetch_real_images.py --out data/real_base --n 64 --size 512
FORGERY_REAL_IMAGE_DIR=data/real_base python3 -m forgery_pipeline.cli probe --config configs/probe.real.yaml --out data/probe_real --n-base 50
FORGERY_REAL_IMAGE_DIR=data/real_base python3 -m checking.run_gates --run data/probe_real --probe data/probe_real --extractor real --out data/checking_report_real.json
```
Expected: 真实 gate1/2/3 指标与 VERDICT（数分钟到十几分钟）。

- [x] **Step 6: 写 `docs/real_gate_results_2026-07-02.md`**

按 Step 4/5 实测填：环境（4060/SD2/n_base/分辨率/耗时）、每闸门真实 VERDICT + 指标、与 mock 代理结果的对照、诚实解读（真实 gate1 t0 可恢复是否成立、gate2 算子可分、gate3 多σ增量），以及局限（单模型无跨生成器、规模小）。诚实边界：这是**真实信号**下的初步判定，规模仍小、需扩样本与多模型确认。

- [x] **Step 7: 提交**

```bash
git add docs/real_gate_results_2026-07-02.md
git commit -m "docs: RTX 4060 真实 SD2 闸门冒烟结果记录"
```

---

## Self-Review

**1. Spec coverage：**
- §2.1 fetch 脚本 → Task 1 ✓；§2.2 LocalImageSource → Task 1 ✓；§2.3 img2img/inpaint → Task 2 ✓；§2.4 registry real → Task 1/2 ✓；§2.5 SD2 extractor → Task 3 ✓；§2.6 probe.real.yaml → Task 3 ✓；§2.7 依赖/权重 → Task 3(extra)+Task 4(install/download) ✓
- §4 测试（LocalImageSource + 懒构造 CPU；GPU 冒烟执行）→ Task 1/2/3(pytest) + Task 4(GPU) ✓
- §5 风险兜底（OOM offload、fp32 VAE、p99 归一化、如实报告）→ Task 3 代码 + Task 4 兜底注 + Task 6 文档 ✓

无缺口。

**2. Placeholder scan：** Tasks 1–3 各步含完整代码/命令；Task 4 为执行步骤（真实命令+期望），Step 6 文档按实测填数（执行时记录真实数据，非占位）。

**3. Type consistency：**
- 懒加载哨兵：`DiffusersImg2Img._pipe`（Task 2 测试断言 `._pipe is None`）与实现一致；`DiffusersSD2Residual._unet`（Task 3 测试断言 `._unet is None`）与实现一致 ✓
- registry `get_image_source/get_img2img/get_inpainter/get_segmenter("real")`（Task 1/2）返回类型与 probe 调用一致（backend 字符串 "real" 贯穿 configs/probe.real.yaml → run_probe → registry）✓
- `residual_stack->(K,H,W)float[0,1]`（Task 3）满足 `ResidualExtractor` 契约，基类 profile/residual_map/detection_score 复用 ✓
- img2img meta 含 `strength`（Task 2）被 `build_probe_strength`（既有）读取一致 ✓

无不一致。

## 执行顺序
Task 1 → 2 → 3（CPU，TDD，pytest 全绿）→ Task 4（GPU 执行冒烟 + 结果文档）。

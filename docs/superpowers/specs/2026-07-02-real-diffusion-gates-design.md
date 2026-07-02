# 真实扩散信号跑真实闸门（RTX 4060/8GB）— 设计文档（Spec）

- 日期：2026-07-02
- 关联：`docs/EXECUTION_CHECKLIST.md`、`docs/PAPER_DESIGN.md`（§2.1 多σ Tweedie 残差、§3 冻结 SD2、VAE fp32/UNet fp16）、`docs/multi_sigma_tweedie_forensics.md`、`checking/`
- 目标：在本机 GPU 上落地**真实底图 + 真实 SD2 生成 + 真实 SD2 多σ残差**，用 `--extractor real` 跑出 gate1/2/3 的**真实** go/no-go 判定（最小可跑路径）。

---

## 1. 目标与范围

### 1.1 环境（已探测）
RTX 4060 / 8 GB、`torch 2.12+cu130` CUDA 可用、HuggingFace 可达（proxy）、`picsum.photos` 可达；`diffusers/transformers/accelerate/safetensors` 待装；COCO 直链不可达。

### 1.2 范围内（最小可跑真实闸门）
- 真实底图下载脚本（picsum）。
- 真实后端：`LocalImageSource` + `DiffusersImg2Img`(SD2-base) + `DiffusersInpainter`(SD2-inpainting) + registry `backend="real"`。
- 真实 `checking/extractor.py:DiffusersSD2Residual`（多σ Tweedie 残差）。
- `configs/probe.real.yaml`；依赖安装 + 权重下载（一次性）。
- GPU 冒烟：`probe --config configs/probe.real.yaml`（n_base=4→~50）+ `run_gates --extractor real` → 真实 gate1/2/3(+gate0)。

### 1.3 非目标
- 主管线 D0–D4 全切真实、真实 SAM/MLLM、SDXL/商业模型 → 后续。
- 不美化判定：真实 gate 可能 WEAK/FAIL，按实测如实报告。

### 1.4 约束
- fp16 UNet + **fp32 VAE**（防 NaN，PAPER §3）+ attention slicing；512 分辨率；模型**懒加载**（构造不下载/不占显存）。
- pytest 保持 **CPU-only 快**：只测 `LocalImageSource` 与懒构造；**真实 GPU 验证走执行步骤（手动冒烟）**，非 pytest。
- 确定性：生成与残差的随机走显式 seed（`torch.Generator`）。
- 复用已有依赖 + 新增 `diffusers/transformers/accelerate/safetensors`（`[real]` extra）。

---

## 2. 组件与接口

### 2.1 底图下载 `scripts/fetch_real_images.py`
`fetch(out_dir="data/real_base", n=200, size=512, start_id=0)`：从 `https://picsum.photos/id/{i}/{size}/{size}.jpg` 确定性拉 n 张（跳过失败 id），存 `real_{i:04d}.jpg`。CLI：`python scripts/fetch_real_images.py --out data/real_base --n 200`。`data/` 已 gitignore。

### 2.2 `backends/real/local_source.py`
```python
class LocalImageSource(base.ImageSource):
    def __init__(self, root, size=512, seed=0)
    def iter_images(self, n) -> Iterator[(img(size,size,3)uint8, meta)]
```
读 `root` 下 `*.jpg/*.png`（排序、确定性），中心裁剪 + resize 到 `size`；`meta={source_dataset:"local", camera_model:None, resolution:[size,size], license:"unknown"}`。不足 n 则产出全部。

### 2.3 `backends/real/diffusers_gen.py`（实现现有 stub）
```python
class DiffusersImg2Img(base.Img2ImgGenerator):
    def __init__(self, model_id="stabilityai/stable-diffusion-2-base", device="cuda", dtype="fp16")
    def img2img(self, image, prompt, strength, params) -> (img, meta)

class DiffusersInpainter(base.Inpainter):
    def __init__(self, model_id="stabilityai/stable-diffusion-2-inpainting", device="cuda", dtype="fp16")
    def inpaint(self, image, mask, prompt, params) -> (img, meta)
```
- 懒加载 `AutoPipelineForImage2Image`/`AutoPipelineForInpainting`，fp16、`enable_attention_slicing()`、`set_progress_bar_config(disable=True)`；OOM 兜底 `enable_sequential_cpu_offload()`（配置开关）。
- prompt 为空时用默认 `"a realistic high quality photo"`；`generator=torch.Generator(device).manual_seed(seed)`。
- img2img：`pipe(prompt, image=PIL(img), strength, num_inference_steps=params.get("steps",30), guidance_scale=params.get("cfg_scale",7.5))`；inpaint：`pipe(prompt, image=PIL(img), mask_image=PIL(mask))`（mask 白=重绘区，与本仓 255=编辑区一致）。
- 返回 `np.array(out.images[0])(uint8 RGB)` + `meta{generator_name, generator_family, seed, strength(仅img2img), steps, cfg_scale}`。

### 2.4 `backends/registry.py` 加 `backend="real"`
- `get_image_source("real", seed)` → `LocalImageSource(root=env FORGERY_REAL_IMAGE_DIR 或 "data/real_base", seed=seed)`
- `get_img2img("real", name, family)` → `DiffusersImg2Img()`（name/family 仅打标签）
- `get_inpainter("real", name, family)` → `DiffusersInpainter()`
- `get_segmenter("real", seed)` → `mock.MockSegmenter`（probe 用几何掩码，占位无害）
- `get_whole_generator("real",...)/get_explainer("real")` → 保持 `_unsupported`（最小路径不用）
- 模型 id/dtype 经模块默认 + 环境变量（`FORGERY_SD2_MODEL` 等）覆盖。

### 2.5 `checking/extractor.py:DiffusersSD2Residual`（核心科学实现）
```python
class DiffusersSD2Residual(ResidualExtractor):
    def __init__(self, model_id="stabilityai/stable-diffusion-2-base", device="cuda",
                 timesteps=(50,150,300,500,700))  # 多 σ = 多 t
    def residual_stack(self, image) -> (K,H,W) float[0,1]
```
- 懒加载：`AutoencoderKL`(fp32) + `UNet2DConditionModel`(fp16) + `DDPMScheduler`(取 `alphas_cumprod`) + `CLIPTextModel/Tokenizer`（算空 prompt 嵌入，UNet cross-attn 需要）。
- 预处理：resize 512、`[-1,1]` 张量；VAE encode → `z0 = latent_dist.mean * vae.config.scaling_factor`（fp32，64×64×4）。
- 对每个 `t`：`eps~N(0,I)`（`torch.Generator` 按图内容+ t 定种，确定性）；`z_t=√ᾱ_t·z0+√(1−ᾱ_t)·eps`；`eps_hat=unet(z_t.half(), t, null_emb).sample`(→fp32)；`r_eps=mean_ch((eps−eps_hat)²)`；`z0_hat=(z_t−√(1−ᾱ_t)·eps_hat)/√ᾱ_t`，`r_x=mean_ch((z0−z0_hat)²)`；本尺度残差图 `= r_eps + r_x`（latent 64×64）。
- 每尺度图**上采样到图像 (H,W)**（`cv2.resize`）并**按自身 p99 归一化并 clip 到 [0,1]**；`np.stack` 成 (K,H,W)。基类 `profile/residual_map/detection_score` 直接复用。
- `get_extractor("real")` 改为**惰性返回** `DiffusersSD2Residual()`（构造不加载模型），既有「抛错」测试相应更新。

### 2.6 配置 `configs/probe.real.yaml`
```yaml
out_dir: data/probe_real
seed: 1234
backend: real
generators_config: configs/generators.yaml
n_base: 8
strengths: [0.1, 0.3, 0.5, 0.7, 0.9]
operators: [img2img, inpaint, outpaint, object_replacement, background_editing]
holdout_generators: []          # 单模型 SD2；跨生成器留后续多模型
```
（`probe` 子命令已读这些键；`generators.yaml` 的 img2img/inpainters 的 name/family 仅打标签，真实后端统一用 SD2/SD2-inpaint。）

### 2.7 依赖
`pyproject.toml` `[real]` extra 增 `accelerate`、`safetensors`（已含 torch/diffusers/transformers）。安装：`pip install --user --break-system-packages diffusers transformers accelerate safetensors`。权重首次 `from_pretrained` 自动下载（SD2-base + SD2-inpainting，约 10 GB，一次性缓存 `~/.cache/huggingface`）。

---

## 3. 数据流（真实路径）
```
scripts/fetch_real_images → data/real_base/*.jpg
forgery-pipeline probe --config configs/probe.real.yaml --out data/probe_real
  build_d0(LocalImageSource) → 真实底图
  build_probe_strength(DiffusersImg2Img, strengths) → 真实 SDEdit
  build_probe_operator(DiffusersInpainter, operators) → 真实局部编辑
python -m checking.run_gates --run data/probe_real --probe data/probe_real --extractor real
  DiffusersSD2Residual 提取真实多σ残差 → gate1/2/3(+gate0) 真实 VERDICT + report.json
```

## 4. 测试策略
- **pytest（CPU-only，保持快绿）**：
  - `LocalImageSource`：造 tmp 图目录 → `iter_images(2)` 产 (512,512,3) uint8 + meta（无需 GPU/diffusers）。
  - `get_extractor("real")` 返回 `DiffusersSD2Residual` 实例（惰性，不加载）；真实 img2img/inpaint 构造不触发下载。
  - 既有 `test_get_extractor_and_real_stub` 更新为「返回实例、不抛错」。
- **GPU 冒烟（执行步骤，非 pytest；本机跑、如实记录）**：
  1. 装依赖 + `fetch_real_images --n 64` + 触发权重下载；
  2. `probe --config configs/probe.real.yaml --out data/probe_real --n-base 4`（验证真实生成、无 OOM、图像是真实编辑）；
  3. `run_gates --run data/probe_real --probe data/probe_real --extractor real`（+ 放到 n_base~50 复跑）→ 记录真实 gate1/2/3(+gate0) VERDICT。
- 验收：pytest 全绿；GPU 冒烟产出真实生成图 + 真实 extractor 残差 + 真实 gate VERDICT（无论 PASS/WEAK/FAIL，如实写入结果文档）。

## 5. 诚实风险与兜底
- **显存**：8GB 在 512 + fp16 + slicing 下 SD2 img2img/inpaint 与残差提取应可跑；若 OOM → 开 `sequential_cpu_offload` / 降分辨率 / 减 batch。
- **数值**：VAE fp32 防 NaN；残差归一化按 p99 clip 防极值。
- **耗时**：权重下载 ~10–20 min（一次性）；每图 K 次 UNet 前向，n_base=50 的 probe 数分钟到十几分钟。
- **结论**：真实 gate1（t0 可恢复）可能达不到 PASS——这正是闸门要证伪/证实的，按实测报告，不硬撑。

## 6. 实施顺序
LocalImageSource + fetch 脚本 → diffusers img2img/inpaint → registry real → SD2 extractor → configs + 依赖 → GPU 冒烟 + 结果文档。每步 TDD（CPU 部分）；GPU 部分为执行冒烟。

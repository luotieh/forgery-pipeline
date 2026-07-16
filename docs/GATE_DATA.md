# GATE_DATA — 闸门数据对照（forgery-pipeline → EXECUTION_CHECKLIST.md）

本管线产出 `docs/EXECUTION_CHECKLIST.md` 闸门 0–4 所需的全部受控数据。
分析脚本在独立仓 `gate_experiments/`；本表给出「闸门 → 用哪个产物 → 关键字段 → 命令」。

## 生成命令

```bash
forgery-pipeline run   --config configs/pipeline.example.yaml --out data/run     # 主数据集（Test-A..F）
forgery-pipeline probe --config configs/probe.yaml            --out data/probe   # 受控 probe（Gate 1/2/3b）
forgery-pipeline validate-manifest --path data/probe/manifest.jsonl
```

## 对照表

| 闸门 | 用哪个产物 | 关键字段 | 备注 |
|---|---|---|---|
| 0 信号地基 | `data/probe`（gate2 inpaint）/ `data/run`（D2/D3） | `image_path` `real_image_path` `mask_path` | 成对 + 已知掩码，调 latent 对齐 |
| 1 t0 可恢复 | `data/probe/gate1_strength.jsonl` | `strength`(0.1–0.9) `init_timestep`(=round(strength·1000)) `operator=img2img` | 强度网格 |
| 2 算子可分 | `data/probe/gate2_operator.jsonl` | `operator`(5 类) `generator_family`/`generator_name` | 5 算子 × ≥2 族；跨模型按 family/name 切 |
| 3a 多 σ 增量 | 任意上面数据 | — | 消融，分析侧 |
| 3b 跨生成器 | `data/probe`（`split=test_b`）/ `data/run`（`split=test_b`） | `split` `generator_name` | 留出族：`kandinsky-inpaint` `sdxl-img2img`（probe）；`ideogram/progan/kandinsky-inpaint`（run） |
| 4 Test-A..F | `data/run/manifest.jsonl` | `split`(train/val/test_a..f) | 8 路评测轴 |
| 4 均衡采样 | `stats` 输出（run/probe 末尾打印） | `by_generator_name` `by_operator` | 核验每生成器计数 |
| 鲁棒性(Test-E) | `data/run`（退化行） | `postprocess` `postprocess_of` | 退化样本独立成行 + 回链 |

## 留出生成器（闸门 3b）

- **probe**：`configs/probe.yaml` 的 `holdout_generators`（默认 `kandinsky-inpaint` + `sdxl-img2img`）。这些生成器的 probe 样本标 `split=test_b`，其余标 `split=train`；分析端 `split=="test_b"` 即留出族，做 seen→留出 掉点曲线。
- **主数据集**：`configs/split.yaml` 的 `holdout_generators`（`ideogram/progan/kandinsky-inpaint`）→ 主 `run` 的 `test_b`。

## 字段速查（PATCHES.md 新增）

- `operator`：img2img / inpaint / outpaint / object_replacement / background_editing …（见 `labels.EDIT_OPERATORS`）。
- `strength`：img2img/SDEdit 去噪强度 ∈ [0,1]，≈ t0/T。
- `init_timestep`：= round(strength × 1000)，直接读 SDEdit 起始 timestep。
- `postprocess_of`：退化样本回链原始 fake 的 `image_id`（退化样本独立成行）。
- `io_chain`：逐节点处理链（`decode>rs512>edit:sd15_inpaint>png`；旧行=`legacy`）——V2 断言真假非生成链一致（PATCH 7.1）。
- `sample_kind`：real / real_vae_rt / edited（VAE 往返硬负样本用，PATCH 7.2）。
- `compositing` / `feather_px`：掩码算子回贴方式 none/paste/paste_feather 与羽化 σ（PATCH 7.3）。
- `probe_group` / `pair_id`：成对 probe 回链（compositing_pair / nd_pair，PATCH 7.3/8.1）。
- `op_params`：算子参数 JSON 容器（cfg_scale/steps 等，PATCH 8.2；CFG/steps 抖动 probe 已用）。
- `base_id`：底图组键（V8 split 互斥断言用；D0=自身 `image_id`，衍生行=底图 `image_id`，PATCH 9.3）。
- `op_params` 扩展键（cfg_scale/steps/prompt/prompt_bank_version——9.1/9.2a 记录）。

# checking/ — 闸门执行测试

消费本管线产出的受控数据（`data/probe`、`data/run`），运行 `docs/EXECUTION_CHECKLIST.md` 的闸门 **0–3** 分析 + **Gate 4 评测轴骨架**，产出每个闸门的 VERDICT。

## ⚠️ 诚实边界（先读）
默认 `--extractor multisigma` 是 **CPU 代理信号**（多尺度高斯重建残差）。在 **mock 数据**上的 VERDICT **只验证分析代码通路正确，不是科学结论**（甚至可能假阳性，因为 mock img2img 强度与像素偏差单调相关）。

要得到**真实**的闸门判定，需三样：
1. `--extractor real`（`DiffusersSD2Residual`，真实冻结 SD2 的多 σ Tweedie 残差）——现为 guarded 骨架，需 `pip install .[real]` + GPU 落地实现；
2. **真实扩散生成的数据**（把 `configs/*.yaml` 的 `backend` 从 `mock` 切到真实后端，或用真实底图）；
3. GPU + 权重 + 数个 GPU 小时。

## 用法

```bash
# 1) 先用管线产数据
forgery-pipeline run   --config configs/pipeline.example.yaml --out data/run
forgery-pipeline probe --config configs/probe.yaml            --out data/probe

# 2) 跑闸门（CPU 代理）
python -m checking.run_gates --run data/run --probe data/probe
# → 打印 gate0-4 五个 VERDICT + CAVEAT，写 data/checking_report.json（+ data/gate2_pca.png）
```

## 闸门与判据

| 闸门 | 测什么 | 关键指标 | 通过判据 |
|---|---|---|---|
| 0 | 残差分开 真实 vs 编辑区 | detection_auc / localization_auc | 均 ≥ 0.6 |
| 1 | t0 可恢复（强度桶+回归） | balanced_accuracy / spearman_rho / 多σ vs 单σ | PASS: ba≥0.55 且 ρ≥0.30 |
| 2 | 算子可分 + 操作vs模型 | same_model_acc / cross_model_acc | PASS: same≥0.50 且 cross≥0.40；CONFOUND: cross<0.30 |
| 3 | 多σ增量 + 跨生成器掉点 | multi_sigma_delta / heldout_acc | PASS: 增量>0 且 heldout>随机 |
| 4 | Test-A..F 评测轴（骨架） | per_split detection_auc / test_f_fpr | EVAL-ONLY（非论文模型） |

数据字段对照见 [`../docs/GATE_DATA.md`](../docs/GATE_DATA.md)。

## 结构
- `extractor.py`：`ResidualExtractor`（`MultiSigmaResidual` CPU 代理 / `DiffusersSD2Residual` 真实骨架）。
- `metrics.py`：sklearn-free（roc/balanced_accuracy/spearman/NearestCentroid/pca_2d/group_split）。
- `data.py`：读 manifest + 批量提特征。
- `gate0..3.py` / `gate4_eval.py` / `run_gates.py`。

## 接真实信号
实现 `checking/extractor.py:DiffusersSD2Residual.residual_stack`（VAE 编码→多 t 加噪→UNet ε̂→Tweedie 残差），`pip install .[real]`，`--extractor real` 即可。

#!/usr/bin/env bash
# P0 跨生成器实验一键脚本（云 GPU 上跑，见 docs/next_optimization_plan_2026-07-09.md）
#
# 一次运行解锁：gate3 真实掉点 + gate2 真实 cross_model + gate0 跨生成器迁移判决。
# 用法：
#   bash scripts/run_cross_generator_p0.sh            # n_base=50 正式
#   N_BASE=8 bash scripts/run_cross_generator_p0.sh   # 冒烟自检
#
# 前置：CUDA GPU（Kandinsky 2.2 + SD1.5 fp16，建议 ≥12GB 显存；8GB 需已启用 cpu offload）、
#       HF 可访问（首次下载 SD1.5 + Kandinsky 权重）、Python ≥3.10。
set -euo pipefail
cd "$(dirname "$0")/.."

N_BASE="${N_BASE:-50}"
PROBE_DIR="data/probe_real"
REPORT="data/checking_report_real.json"
CROSS_REPORT="data/gate0_cross_generator.json"

echo "==> [0/5] 安装依赖（real extra）"
pip install -e ".[real]" >/dev/null

echo "==> [1/5] 下载真实底图（picsum，n=${N_BASE}）"
python scripts/fetch_real_images.py --out data/real_base --n "${N_BASE}"

echo "==> [2/5] 生成 probe（SD1.5 + Kandinsky 异族先验，n_base=${N_BASE}）"
# holdout_generators=[kandinsky-inpaint] 已写在 configs/probe.real.yaml → gate3 heldout split 非空
python -m forgery_pipeline.cli probe \
  --config configs/probe.real.yaml --out "${PROBE_DIR}" --n-base "${N_BASE}"

echo "==> [3/5] 跑闸门 gate0-4（真实扩散残差 extractor）"
python -m checking.run_gates \
  --run "${PROBE_DIR}" --probe "${PROBE_DIR}" \
  --extractor real --out "${REPORT}"

echo "==> [4/5] gate0 跨生成器迁移判决（SD1.5 残差检 Kandinsky 伪造）"
python scripts/gate0_cross_generator.py \
  --run "${PROBE_DIR}" --family kandinsky --extractor real --out "${CROSS_REPORT}"

echo "==> [5/5] 关键数字汇总"
python - "$REPORT" "$CROSS_REPORT" <<'PY'
import json, sys
rep = json.load(open(sys.argv[1], encoding="utf-8"))
cross = json.load(open(sys.argv[2], encoding="utf-8"))
g = rep["gates"]
print("\n===== P0 关键结论 =====")
print(f"gate2 same_model : {g['gate2']['metrics'].get('same_model_acc', g['gate2']['metrics'])}")
print(f"gate2 cross_model: {g['gate2']['metrics'].get('cross_model_acc', '见上')}")
print(f"gate3 heldout    : {g['gate3']['metrics']}  VERDICT={g['gate3']['verdict']}")
print(f"gate0 迁移(Kandinsky): det={cross['metrics']['detection_auc']} "
      f"loc={cross['metrics']['localization_auc']} "
      f"n_pos={cross['metrics']['n_pos']} → {cross['interpretation']}")
print("完整报告:", sys.argv[1], "|", sys.argv[2])
PY
echo "==> 完成。把 ${REPORT} 和 ${CROSS_REPORT} 拷回本机更新文档结论。"

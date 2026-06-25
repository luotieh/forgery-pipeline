#!/usr/bin/env bash
# 用 mock 后端端到端跑通 pipeline 并打印统计。
set -euo pipefail
forgery-pipeline run --config configs/pipeline.example.yaml
forgery-pipeline stats --path data/run/manifest.jsonl
forgery-pipeline validate-manifest --path data/run/manifest.jsonl

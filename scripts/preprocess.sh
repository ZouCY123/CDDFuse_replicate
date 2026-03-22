#!/usr/bin/env bash
set -e

CONFIG="${1:-configs/base.yaml}"

echo ">>> 使用配置: $CONFIG"
python -m src.datasets.preprocess --config "$CONFIG"
echo "✓ 预处理完成"
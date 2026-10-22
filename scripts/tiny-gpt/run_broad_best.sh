#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PY=../../.venv/bin/python
for arm in full ctrl; do
    echo "=== train broad-4L256d-$arm-long  $(date -u +%H:%M:%S) ==="
    $PY -u src/train.py --config "configs/broad-4L256d-$arm-long.yaml"
    echo "=== test broad-4L256d-$arm-long  $(date -u +%H:%M:%S) ==="
    $PY -u src/evaluate.py --checkpoint "artifacts/runs/broad-4L256d-$arm-long/best.pt" \
        --split test --per-document --threads 2
done
echo "BEST PIPELINE DONE $(date -u +%H:%M:%S)"

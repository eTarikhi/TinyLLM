#!/usr/bin/env bash
# Capacity x mixture: the big model on both corpora, completing the 2x2.
set -euo pipefail
cd "$(dirname "$0")"
PY=../../.venv/bin/python
for arm in full ctrl; do
    echo "=== train broad-4L256d-$arm  $(date -u +%H:%M:%S) ==="
    $PY -u src/train.py --config "configs/broad-4L256d-$arm.yaml"
    echo "=== test broad-4L256d-$arm  $(date -u +%H:%M:%S) ==="
    $PY -u src/evaluate.py --checkpoint "artifacts/runs/broad-4L256d-$arm/best.pt" \
        --split test --per-document --threads 2
done
echo "SIZE PIPELINE DONE $(date -u +%H:%M:%S)"

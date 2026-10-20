#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PY=../../.venv/bin/python
echo "=== train broad-full-9600  $(date -u +%H:%M:%S) ==="
$PY -u src/train.py --config configs/broad-full-9600.yaml
echo "=== test broad-full-9600  $(date -u +%H:%M:%S) ==="
$PY -u src/evaluate.py --checkpoint artifacts/runs/broad-full-9600/best.pt \
    --split test --per-document --threads 2
echo "LONG PIPELINE DONE $(date -u +%H:%M:%S)"

#!/usr/bin/env bash
# Follows queue4: the 6e-3 LR arm, last input to the release configuration.
set -euo pipefail
cd "$(dirname "$0")"
PY=../../.venv/bin/python
while pgrep -f 'run_queue4.sh|src/train.py|src/evaluate.py|src/data.py' >/dev/null 2>&1; do sleep 60; done
echo "=== train big-lr60  $(date -u +%H:%M:%S) ==="
$PY -u src/train.py --config configs/big-lr60.yaml
echo "=== test big-lr60  $(date -u +%H:%M:%S) ==="
$PY -u src/evaluate.py --checkpoint artifacts/runs/big-lr60/best.pt \
    --split test --per-document --threads 2
echo "QUEUE5 DONE $(date -u +%H:%M:%S)"

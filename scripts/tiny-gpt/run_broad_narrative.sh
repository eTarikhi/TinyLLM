#!/usr/bin/env bash
# Third arm of the corpus-composition experiment, queued behind the first two.
# Waits for the running pipeline rather than for a human, then trains the
# volume-matched narrative control and scores it per document.
set -euo pipefail
cd "$(dirname "$0")"
PY=../../.venv/bin/python

while pgrep -f 'src/train.py|src/evaluate.py' >/dev/null 2>&1; do sleep 60; done
echo "=== train broad-narrative  $(date -u +%H:%M:%S) ==="
$PY -u src/train.py --config configs/broad-narrative.yaml
echo "=== test broad-narrative  $(date -u +%H:%M:%S) ==="
$PY -u src/evaluate.py --checkpoint artifacts/runs/broad-narrative/best.pt \
    --split test --per-document --threads 2
echo "NARRATIVE PIPELINE DONE $(date -u +%H:%M:%S)"

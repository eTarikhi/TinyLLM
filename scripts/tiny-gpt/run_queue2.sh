#!/usr/bin/env bash
# Follows the capacity x horizon pair automatically: first the held-out
# transfer probe on both checkpoints, then the seed replication of the
# composition headline (broad-*-s1338). Waits on the running pipeline
# rather than on a human.
set -euo pipefail
cd "$(dirname "$0")"
PY=../../.venv/bin/python

while pgrep -f 'run_broad_best.sh|src/train.py|src/evaluate.py' >/dev/null 2>&1; do sleep 60; done

for arm in full ctrl; do
    echo "=== held-out probe broad-4L256d-$arm-long  $(date -u +%H:%M:%S) ==="
    $PY -u src/evaluate.py --checkpoint "artifacts/runs/broad-4L256d-$arm-long/best.pt" \
        --data-dir data/prepared-heldout --split test --per-document --threads 2 \
        --out-dir "artifacts/runs/broad-4L256d-$arm-long/heldout"
done

for arm in full ctrl; do
    echo "=== train broad-$arm-s1338  $(date -u +%H:%M:%S) ==="
    $PY -u src/train.py --config "configs/broad-$arm-s1338.yaml"
    echo "=== test broad-$arm-s1338  $(date -u +%H:%M:%S) ==="
    $PY -u src/evaluate.py --checkpoint "artifacts/runs/broad-$arm-s1338/best.pt" \
        --split test --per-document --threads 2
done
echo "QUEUE2 DONE $(date -u +%H:%M:%S)"

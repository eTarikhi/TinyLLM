#!/usr/bin/env bash
# Third in the chain: prepare the broad50 corpus, train it at 3.74M on the
# long clock, then score it on both the seen 71 and the held-out 40.
set -euo pipefail
cd "$(dirname "$0")"
PY=../../.venv/bin/python

while pgrep -f 'run_queue2.sh|src/train.py|src/evaluate.py' >/dev/null 2>&1; do sleep 60; done

echo "=== prepare broad50  $(date -u +%H:%M:%S) ==="
$PY -u src/data.py --corpus gutenberg-mixed-broad50 --tokenizer byte_bpe \
    --vocab-size 2048 --out-dir data/prepared-broad50 --cache-dir data/raw/broad \
    --tokenizer-from data/prepared-broad-full
echo "=== train broad50-4L256d-long  $(date -u +%H:%M:%S) ==="
$PY -u src/train.py --config configs/broad50-4L256d-long.yaml
echo "=== test broad50-4L256d-long (seen 71)  $(date -u +%H:%M:%S) ==="
$PY -u src/evaluate.py --checkpoint artifacts/runs/broad50-4L256d-long/best.pt \
    --split test --per-document --threads 2
echo "=== held-out probe broad50-4L256d-long  $(date -u +%H:%M:%S) ==="
$PY -u src/evaluate.py --checkpoint artifacts/runs/broad50-4L256d-long/best.pt \
    --data-dir data/prepared-heldout --split test --per-document --threads 2 \
    --out-dir artifacts/runs/broad50-4L256d-long/heldout
echo "QUEUE3 DONE $(date -u +%H:%M:%S)"

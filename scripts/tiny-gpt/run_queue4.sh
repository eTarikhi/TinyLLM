#!/usr/bin/env bash
# Probe queue: three short-clock arms at 3.74M on the broad corpus, each
# moving one variable off broad-4L256d-full (test 1.6467). Their outcomes
# choose the configuration of the long release run that follows.
set -euo pipefail
cd "$(dirname "$0")"
PY=../../.venv/bin/python

echo "=== prepare broad-full vocab 4096  $(date -u +%H:%M:%S) ==="
$PY -u src/data.py --corpus gutenberg-mixed-broad-full --tokenizer byte_bpe \
    --vocab-size 4096 --out-dir data/prepared-broad-full-v4096 --cache-dir data/raw/broad

for cfg in big-lr15 big-ctx512 big-vocab4096; do
  echo "=== train $cfg  $(date -u +%H:%M:%S) ==="
  $PY -u src/train.py --config configs/$cfg.yaml
  echo "=== test $cfg  $(date -u +%H:%M:%S) ==="
  $PY -u src/evaluate.py --checkpoint artifacts/runs/$cfg/best.pt \
      --split test --per-document --threads 2
done
echo "QUEUE4 DONE $(date -u +%H:%M:%S)"

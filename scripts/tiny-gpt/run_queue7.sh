#!/usr/bin/env bash
# Probe queue 2: vocab 8192, shape 8Lx192d, seed-1338 vocab replica.
# Each arm is a short-clock (2,571-step) single-variable move read
# against big-vocab4096 (1.6236). Outcomes choose the release v3 config.
set -euo pipefail
cd "$(dirname "$0")"
PY=../../.venv/bin/python

echo "=== prepare broad-full vocab 4096  $(date -u +%H:%M:%S) ==="
$PY -u src/data.py --corpus gutenberg-mixed-broad-full --tokenizer byte_bpe \
    --vocab-size 4096 --out-dir data/prepared-broad-full-v4096 --cache-dir data/raw/broad
echo "=== prepare broad-full vocab 8192  $(date -u +%H:%M:%S) ==="
$PY -u src/data.py --corpus gutenberg-mixed-broad-full --tokenizer byte_bpe \
    --vocab-size 8192 --out-dir data/prepared-broad-full-v8192 --cache-dir data/raw/broad

for cfg in big-vocab8192 big-shape-8L192d big-vocab4096-s1338; do
  echo "=== train $cfg  $(date -u +%H:%M:%S) ==="
  $PY -u src/train.py --config configs/$cfg.yaml
  echo "=== test $cfg  $(date -u +%H:%M:%S) ==="
  $PY -u src/evaluate.py --checkpoint artifacts/runs/$cfg/best.pt \
      --split test --per-document --threads 2
done
echo "QUEUE7 DONE $(date -u +%H:%M:%S)"

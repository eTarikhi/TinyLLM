#!/usr/bin/env bash
# Probe queue 3: vocab 16384 (fourth doubling) and the shape arm
# (re-run: the original died at step 2400/2571 in a sandbox wipe
# before its eval). Outcomes choose the release v3 configuration.
set -euo pipefail
cd "$(dirname "$0")"
PY=../../.venv/bin/python

for V in 4096 8192 16384; do
  echo "=== prepare broad-full vocab $V  $(date -u +%H:%M:%S) ==="
  $PY -u src/data.py --corpus gutenberg-mixed-broad-full --tokenizer byte_bpe \
      --vocab-size $V --out-dir data/prepared-broad-full-v$V --cache-dir data/raw/broad
done

for cfg in big-vocab16384 big-shape-8L192d; do
  echo "=== train $cfg  $(date -u +%H:%M:%S) ==="
  $PY -u src/train.py --config configs/$cfg.yaml
  echo "=== test $cfg  $(date -u +%H:%M:%S) ==="
  $PY -u src/evaluate.py --checkpoint artifacts/runs/$cfg/best.pt \
      --split test --per-document --threads 2
done
echo "QUEUE8 DONE $(date -u +%H:%M:%S)"

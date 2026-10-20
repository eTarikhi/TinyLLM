#!/usr/bin/env bash
# Corpus-composition experiment: train both arms, then score both on the test
# split per document. One script so the second job starts the moment the first
# finishes rather than waiting on a human.
#
#   broad-full  151 books, 25,952,894 train tokens, 67.8% narrative prose
#   broad-ctrl   71 books, 18,939,699 train tokens, 90.7% narrative prose
#
# val.bin, test.bin and tokenizer.json are byte-identical between the two
# prepared directories, so the arms differ only in what they trained on.
set -euo pipefail
cd "$(dirname "$0")"
PY=../../.venv/bin/python

for arm in full ctrl; do
    echo "=== train broad-$arm  $(date -u +%H:%M:%S) ==="
    $PY -u src/train.py --config "configs/broad-$arm.yaml"
done

for arm in full ctrl; do
    echo "=== test broad-$arm  $(date -u +%H:%M:%S) ==="
    $PY -u src/evaluate.py \
        --checkpoint "artifacts/runs/broad-$arm/best.pt" \
        --split test --per-document --threads 2
done

echo "BROAD PIPELINE DONE $(date -u +%H:%M:%S)"

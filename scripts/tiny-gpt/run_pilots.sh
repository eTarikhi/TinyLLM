#!/bin/bash
# Pilot matrix: one variable changed per run, identical token budget (400 steps
# x 4,096 tokens = 1.64M tokens) so the comparison is at a fixed budget.
set -e
PY="../../.venv/bin/python -u"
export OMP_NUM_THREADS=2

$PY src/train.py --config configs/pilot.yaml --run-id p1-char-baseline
$PY src/train.py --config configs/pilot.yaml --run-id p2-bpe512 --data-dir data/prepared-bpe512
$PY src/train.py --config configs/pilot.yaml --run-id p3-untied --tie-embeddings false
$PY src/train.py --config configs/pilot.yaml --run-id p4-lr3e-3 --learning-rate 3e-3
$PY src/train.py --config configs/pilot.yaml --run-id p5-nodropout --dropout 0.0
echo "ALL PILOTS DONE"

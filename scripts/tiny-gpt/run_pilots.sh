#!/bin/bash
# Pilot matrix: one variable changed per run, identical token budget
# (1,638,400 tokens) so every comparison is at a fixed budget rather than a
# fixed step count. Results and the decision each pilot drove:
# experiments/2026-10-17-tiny-shakespeare/pilots.md
#
# ~35 minutes total on 2 CPU cores. Run from scripts/tiny-gpt.
set -e
PY="${PY:-../../.venv/bin/python} -u"
export OMP_NUM_THREADS=2

# --- single-variable pilots, 400 steps x 4,096 tokens ----------------------
$PY src/train.py --config configs/pilot.yaml --run-id p1-char-baseline
$PY src/train.py --config configs/pilot.yaml --run-id p2-bpe512 --data-dir data/prepared-bpe512
$PY src/train.py --config configs/pilot.yaml --run-id p3-untied --tie-embeddings false
$PY src/train.py --config configs/pilot.yaml --run-id p4-lr3e-3 --learning-rate 3e-3
$PY src/train.py --config configs/pilot.yaml --run-id p5-nodropout --dropout 0.0

# --- confirmation pilots -----------------------------------------------------
# Three pilots won separately; that does not prove they win together. These two
# run the winning combination at the main run's batch size (8,192 tokens per
# update, so 200 steps for the same 1,638,400-token budget) and test whether
# nanochat's eta ~ sqrt(B/B_ref) rule justifies raising the LR with the batch.
$PY src/train.py --config configs/pilot.yaml --run-id p6-bpe-lr3e-3 \
    --data-dir data/prepared-bpe512 --learning-rate 3e-3 --dropout 0.0 \
    --grad-accum 2 --max-steps 200 --warmup-steps 50 --eval-every 50 --log-every 25

$PY src/train.py --config configs/pilot.yaml --run-id p7-bpe-lr4.2e-3 \
    --data-dir data/prepared-bpe512 --learning-rate 4.2e-3 --dropout 0.0 \
    --grad-accum 2 --max-steps 200 --warmup-steps 50 --eval-every 50 --log-every 25

echo "ALL PILOTS DONE -- compare with:"
echo "  python src/plot_metrics.py artifacts/runs/p*/metrics.jsonl --metric val_bpb --out pilots.svg"

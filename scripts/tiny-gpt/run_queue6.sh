#!/usr/bin/env bash
# Release queue: waits for the probe queues, picks the peak LR from the
# big-lr60 result, then trains broad-v4096-release and scores it on the
# seen 71 and the held-out 40 (the latter re-prepared at vocab 4096).
set -euo pipefail
cd "$(dirname "$0")"
PY=../../.venv/bin/python

while pgrep -f 'run_queue4.sh|run_queue5.sh|src/train.py|src/evaluate.py|src/data.py' >/dev/null 2>&1; do sleep 60; done

LR=$($PY - <<'PYEOF'
import json, pathlib
p = pathlib.Path("artifacts/runs/big-lr60/eval-test.json")
try:
    v = json.loads(p.read_text())["model"]["bpb"]
    print("6.0e-3" if v < 1.6467 * (1 - 0.0037) else "3.0e-3")
except Exception:
    print("3.0e-3")
PYEOF
)
echo "=== release peak LR selected: $LR  $(date -u +%H:%M:%S) ==="
if [ "$LR" != "3.0e-3" ]; then
  sed -i "s/learning_rate: 3.0e-3/learning_rate: $LR/" configs/broad-v4096-release.yaml
fi

echo "=== prepare heldout at vocab 4096  $(date -u +%H:%M:%S) ==="
$PY -u src/data.py --corpus gutenberg-heldout-forms --tokenizer byte_bpe \
    --vocab-size 4096 --out-dir data/prepared-heldout-v4096 --cache-dir data/raw/broad \
    --tokenizer-from data/prepared-broad-full-v4096

echo "=== train broad-v4096-release  $(date -u +%H:%M:%S) ==="
$PY -u src/train.py --config configs/broad-v4096-release.yaml
echo "=== test broad-v4096-release (seen 71)  $(date -u +%H:%M:%S) ==="
$PY -u src/evaluate.py --checkpoint artifacts/runs/broad-v4096-release/best.pt \
    --split test --per-document --threads 2
echo "=== held-out probe broad-v4096-release  $(date -u +%H:%M:%S) ==="
$PY -u src/evaluate.py --checkpoint artifacts/runs/broad-v4096-release/best.pt \
    --data-dir data/prepared-heldout-v4096 --split test --per-document --threads 2 \
    --out-dir artifacts/runs/broad-v4096-release/heldout
echo "QUEUE6 DONE $(date -u +%H:%M:%S)"

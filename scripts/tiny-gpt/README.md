# tiny-gpt — the reference implementation for this skill

A complete, tested, CPU-runnable pretraining project: tokenizers written from
scratch, a provenance-checked data pipeline, a decoder-only Transformer, a
training loop that can be interrupted and resumed exactly, and an evaluation
that reports bits per byte next to the baselines a model has to beat.

It exists so that `SKILL.md`'s Track 1 is not advice. It is the smallest thing
that passes every gate the skill defines, and it was run end to end on two CPU
cores — the measurements below are from that run, not from a datasheet.

**Copy it into the user's project and adapt it.** Do not paste it in unread,
and do not quote its numbers as predictions for different hardware or data.

## Layout

```text
scripts/tiny-gpt/
  requirements.txt            pinned, with the CPU-only install note
  configs/                    tiny.yaml (smoke), pilot.yaml, main.yaml
  src/
    tokenizer.py              CharTokenizer + ByteBPETokenizer (no tokenizer library)
    data.py                   fetch + checksum + audit + split + encode + batch
    model.py                  GPT: pre-norm blocks, SDPA causal attention, tying
    train.py                  AdamW, warmup+cosine, clipping, eval, resume
    evaluate.py               held-out loss/PPL/BPB, baselines, memorization probe
    generate.py               sampling with recorded settings
    plot_metrics.py           metrics.jsonl -> SVG, no plotting dependency
  tests/                      the seven gates SKILL.md names, plus tokenizer tests
```

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# On a CPU-only Linux box the PyPI torch wheel drags in several GB of CUDA
# runtime packages it will never use. Prefer:
#   pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
```

## The gate sequence, as commands

Run them in this order. Each one is a decision point, not a step to skip.

```bash
# Gate 1 + 2 — correctness and a learning proof (~2 s on 2 CPU cores)
python -m unittest discover -s tests -t tests -v

# Gate 0 — fetch, verify, audit, split, tokenize. Writes data/prepared-char/manifest.json
python src/data.py --tokenizer char     --out-dir data/prepared-char
python src/data.py --tokenizer byte_bpe --vocab-size 512 --out-dir data/prepared-bpe512

# Gate 3 — pilots: one variable per run, identical budget
bash run_pilots.sh

# Gate 4 — the main run
python -u src/train.py --config configs/main.yaml --run-id main

# Evaluate against baselines on the held-out test split, and sample
python src/evaluate.py --checkpoint artifacts/runs/main/best.pt --split test
python src/generate.py --checkpoint artifacts/runs/main/best.pt --num-samples 3 \
    --temperature 0.8 --top-k 40 --out artifacts/runs/main/samples.txt
python src/plot_metrics.py artifacts/runs/*/metrics.jsonl --metric val_bpb --out curve.svg
```

Every run writes `metrics.jsonl` (one JSON record per log, eval, sample and
summary event), `summary.json`, and atomic `last.pt` / `best.pt` checkpoints
carrying model, optimizer, config, step, token count, RNG state and data
position.

## What the design choices are, and why

| Choice | Value | Reason |
| --- | --- | --- |
| Architecture | pre-norm decoder-only, GELU MLP, learned positions | SKILL.md Track 1: keep the first architecture conventional and readable |
| Attention | `F.scaled_dot_product_attention(..., is_causal=True)` | correct masking for free, and the fastest kernel the runtime has |
| Embeddings | tied by default, `tie_embeddings: false` to compare | the tied/untied choice changes the parameter count and the plan records it |
| Optimizer | AdamW, decay on matrices only | one global LR; `beta2=0.99` because tokens/update is small |
| Schedule | linear warmup, cosine to `0.1 × peak`, horizon = run length | both nanoGPT anchors in `references/training-defaults.md` do this |
| Clipping | 1.0, pre-clip norm logged | so it is visible when clipping is actually binding |
| Data split | contiguous 90/5/5, boundaries on blank lines | a shuffled split of one continuous text leaks overlapping windows |
| Tokenizer training | train split only | validation text must not shape the vocabulary |
| Batching | `batch(step)` is a pure function of `(seed, step)` | makes resume exact and the data position auditable |
| Metric | bits per byte, next to per-token loss | per-token loss is not comparable across vocabularies |

## Reproducibility recipe

1. `git rev-parse HEAD` — the data manifest records it automatically.
2. `pip install -r requirements.txt` — exact pins; the run records the resolved
   versions, platform, thread count and peak RSS in `metrics.jsonl`.
3. `python src/data.py ...` — the corpus is checksum-verified before use, so a
   changed upstream file fails loudly instead of silently training on new data.
4. `python src/train.py --config configs/<name>.yaml` — seeded; batches are a
   pure function of `(seed, step)`.
5. Compare runs at a fixed **token** budget, not a fixed step count, and never
   compare per-token loss across tokenizers — use the `bpb` column.

## Known limitations

- One small public-domain corpus of early modern English. The model is a
  demonstration of the pipeline, not a useful language model.
- CPU-only measurements. Throughput, memory and wall clock do not transfer to a
  GPU, and the MFU figure is not comparable to published large-model numbers.
- No instruction tuning, no chat template, no safety tuning. This is a base
  model trained on next-token prediction; it will not follow instructions.
- The memorization probe uses a public-domain corpus, so it calibrates
  behaviour rather than demonstrating privacy safety.

See [`../../experiments/`](../../experiments/) for the filled-in experiment
plan, the pilot matrix, the evaluation report and the model card from the run
recorded below.

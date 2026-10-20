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
  configs/                    tiny.yaml (smoke), pilot*.yaml, main*.yaml,
                              size-*/ctx-*/lr-* sweep points
  src/
    tokenizer.py              CharTokenizer + ByteBPETokenizer (no tokenizer library)
    data.py                   fetch + checksum + audit + split + encode + batch
    model.py                  GPT: pre-norm blocks, SDPA causal attention, tying
    train.py                  AdamW, warmup+cosine, clipping, eval, resume
    evaluate.py               held-out loss/PPL/BPB, baselines, memorization probe
    generate.py               sampling with recorded settings
    plot_metrics.py           metrics.jsonl -> SVG, no plotting dependency
    size_probe.py             params + tokens/s per config, for compute-matched sweeps
  tests/                      the seven gates SKILL.md names, plus tokenizer and
                              multi-document corpus tests
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

# Gate 0 — fetch, verify, audit, split, tokenize. Writes <out-dir>/manifest.json
#   tiny-shakespeare    one 1.1 MB file, the fastest way to prove the pipeline works
#   gutenberg-mixed-27  27 public-domain books, 18.8 MB, per-document provenance
python src/data.py --tokenizer char     --out-dir data/prepared-char
python src/data.py --tokenizer byte_bpe --vocab-size 512 --out-dir data/prepared-bpe512
python src/data.py --corpus gutenberg-mixed-27 --tokenizer byte_bpe --vocab-size 2048 \
    --out-dir data/prepared-gb-bpe2048 --tokenizer-sample-chars 4000000

# Gate 3 — pilots: one variable per run, identical budget
bash run_pilots.sh                                      # iteration 1, tiny-shakespeare
for p in g1-vocab1024 g2-vocab2048 g3-small g4-wide; do # iteration 2, 27 books
  python -u src/train.py --config configs/pilot-$p.yaml
done

# Gate 4 — the main run
python -u src/train.py --config configs/main.yaml            # iteration 1
python -u src/train.py --config configs/main-gutenberg.yaml  # iteration 2

# Evaluate against baselines on the held-out test split, and sample
python src/evaluate.py --checkpoint artifacts/runs/main/best.pt --split test
python src/generate.py --checkpoint artifacts/runs/main/best.pt --num-samples 3 \
    --temperature 0.8 --top-k 40 --out artifacts/runs/main/samples.txt
python src/plot_metrics.py artifacts/runs/*/metrics.jsonl --metric val_bpb --out curve.svg

# Gate 5 — sweep one lever at a matched budget. size_probe.py sizes each point
# from measured throughput so every run completes its cosine schedule.
python src/size_probe.py --sweep context --budget-s 2926 --reference-tok-s 13437
for c in ctx-128 ctx-512 ctx-1024; do python -u src/train.py --config configs/$c.yaml; done
# Curves a percent apart converge visually; --since-tokens zooms on the tail.
python src/plot_metrics.py artifacts/runs/lr-*/metrics.jsonl --metric val_bpb \
    --since-tokens 19660800 --out lr-bracket.svg
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
| Data split | contiguous 90/5/5, boundaries on blank lines, applied *per document* | a shuffled split leaks overlapping windows; holding out whole documents would make validation a different set of authors, measuring domain transfer instead of language modelling |
| Document joins | the tokenizer's EOS id between documents | concatenating unrelated books teaches transitions that do not exist; EOS contributes no bytes, so bits-per-byte is unaffected |
| Boilerplate | Project Gutenberg header/footer stripped, statistics cached beside each document | the boilerplate is the one part of the file that is not public domain, and it is identical across books, so leaving it in teaches the model to emit licence text |
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

- Public-domain literary English only, and at most 18.8 MB of it. The models
  are demonstrations of the pipeline, not useful language models.
- CPU-only measurements. Throughput, memory and wall clock do not transfer to a
  GPU, and the MFU figure is not comparable to published large-model numbers.
- No instruction tuning, no chat template, no safety tuning. This is a base
  model trained on next-token prediction; it will not follow instructions.
- The memorization probe uses a public-domain corpus, so it calibrates
  behaviour rather than demonstrating privacy safety.

## The run this code actually performed

Everything below was measured on 2026-10-18 on two Xeon cores, not estimated.
The full evidence trail — plan, pilot matrix, metrics, samples, evaluation
report and model card — is in
[`../../experiments/2026-10-17-tiny-shakespeare/`](../../experiments/2026-10-17-tiny-shakespeare/plan.md).

| | |
| --- | --- |
| Corpus | tiny-shakespeare, 1,115,394 bytes, public domain, contiguous 90/5/5 split |
| Tokenizer | byte-level BPE-512 trained on the train split, 1.889 bytes/token |
| Model | 3,344,640 parameters: 4 layers, `d_model` 256, 4 heads, context 256, tied embeddings |
| Pilots | 7 runs, one variable each, 1,638,400 tokens apiece, ~35 min total |
| Main run | 1,500 steps × 8,192 tokens = 12,288,000 tokens in **25.6 min** at **8,003 tok/s**, peak RSS 1,269 MB, 0 non-finite steps |
| Selected checkpoint | step 750 (6,144,000 tokens), chosen by validation loss, not by being last |
| Held-out result | **2.3284 bits/byte** on the test split, against 2.8233 for a bigram counter, 4.8619 for the same model untrained, and 2.5583 for `xz -9` primed with the training text |

Three results worth carrying away, because they are the kind that do not
appear in a tutorial:

1. **The tokenizer beat every hyperparameter.** Byte-level BPE-512 against a
   67-symbol character vocabulary was worth 19% in bits per byte at an equal
   token budget — more than the learning rate (11%) and far more than tying or
   dropout. On per-token loss it looks 53% *worse*, which is why the metric was
   fixed as bits per byte before any run started.
2. **Dropout costs 40% of throughput on CPU** (5,080 → 7,129 tok/s with
   `dropout=0.0`), a trade that does not exist on a GPU where the matmuls
   dominate. Taking it was correct for the pilots and wrong by step 900 of the
   main run.
3. **Validation turned at 11.9 epochs while training loss kept falling.** The
   final weights score 14.6% worse on held-out text than the selected ones.
   Keeping `best.pt` is not a detail; it is half the value of the run.

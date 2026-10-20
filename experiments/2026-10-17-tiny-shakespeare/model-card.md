# Model card — tiny-gpt-shakespeare-3.3M

| | |
| --- | --- |
| **Model** | `tiny-gpt-shakespeare-3.3M`, run id `main` |
| **Date** | 2026-10-18 (trained 00:16–00:42 UTC) |
| **Type** | Base language model, decoder-only Transformer, trained from random init |
| **Parameters** | 3,344,640 total / 3,279,104 non-embedding (input and output embeddings tied) |
| **Tokenizer** | Byte-level BPE, vocabulary 512, trained on the train split only, 1.889 bytes/token |
| **Context** | 256 tokens (~484 bytes) |
| **Training data** | tiny-shakespeare, 1,003,857-byte train split, public domain |
| **Training budget** | 6,144,000 tokens at the selected checkpoint (11.9 epochs); the run continued to 12,288,000 and got worse |
| **Compute** | 2 vCPU Intel Xeon @ 2.60 GHz, FP32, 25.6 min wall clock, 8,003 tok/s, 185.8 GFLOP/s, peak RSS 1,269 MB, $0 |
| **Held-out quality** | **2.3284 bits per byte** on the test split (perplexity 20.25/token) vs 2.8233 for a bigram counter and 4.8298 for uniform |
| **License** | MIT (code). Weights are not distributed. |
| **Contact / provenance** | `scripts/tiny-gpt` in this repository, commit recorded in the run manifest |

## What it is

A deliberately small GPT trained end to end to demonstrate the workflow in
this skill: tokenizer from scratch, audited data pipeline, correctness tests,
a pilot matrix, one main run, and an evaluation against baselines. The model
is the *by-product*; the evidence trail is the deliverable.

Architecture: 4 pre-norm blocks, `d_model` 256, 4 heads of dimension 64,
GELU MLP with 4× expansion, learned absolute positions, LayerNorm without
bias, causal `scaled_dot_product_attention`, tied input/output embeddings,
FP32. Trained with AdamW (β 0.9/0.99, weight decay 0.1 on matrices only),
peak LR 3e-3 with 150 steps of linear warmup then cosine decay to 3e-4,
gradient clipping at 1.0, 8,192 tokens per optimizer update.

## Intended use

Teaching and reference: reading the code, reproducing the run, and comparing
your own changes against a measured baseline on hardware you already have.

## Out of scope

Not for any user-facing or production use. It is a base model with no
instruction tuning, no chat template, no refusal behaviour and no content
filtering; it cannot answer questions, follow instructions, or be trusted for
any factual claim. It writes Elizabethan-flavoured English only, and its
output is frequently ungrammatical and always meaningless beyond a few lines.

## Training data

tiny-shakespeare (`karpathy/char-rnn`, file MIT-licensed, text public domain),
1,115,394 bytes, split contiguously 90/5/5 with boundaries on blank lines.
Full data card — provenance, license reasoning, audit results, split rationale
and hashes — is in [`plan.md`](plan.md). The corpus is checksum-verified
(`sha256:86c4e6aa…`) at every run, and the audit found no email addresses,
phone numbers, IP addresses, URLs or key-like strings. Duplication was
measured (1.02% of 512-byte blocks) and deliberately not removed, because
repeated short lines are a property of play dialogue.

Nothing was filtered for toxicity or bias. Shakespeare's text contains
violence, misogyny, antisemitism and racialized language characteristic of its
period, and a model trained on it can reproduce that register.

## Evaluation

Full report with protocol, baselines and caveats:
[`eval-report.md`](eval-report.md).

| Metric (test split, 55,349 bytes) | Value |
| --- | --- |
| Bits per byte | **2.3284** |
| Loss / perplexity per token | 3.0082 nats / 20.25 |
| vs uniform (4.8298 bpb) | 2.07× better |
| vs unigram (4.0068) | 1.72× better |
| vs bigram (2.8233) | 1.21× better |
| vs same architecture, random init (4.8619) | 2.09× better |
| vs `xz -9` primed with the training text (2.5583) | 1.10× better |
| Longest verbatim match vs training text, 4×240 tokens | 27 characters |

No downstream, instruction-following or safety benchmarks were run. Reporting
any would be misleading at this scale.

## Known failure modes

- Loses grammatical agreement and referential consistency within a few lines.
- Invents word-like strings (`musicious`, `softed`) — expected from a 512-entry
  byte-level BPE vocabulary with too little data to constrain compositions.
- Repeats speaker headings and short dialogue turns, the dominant pattern in
  the corpus.
- Degrades sharply above temperature ~1.0 with top-k disabled.
- Quality depends on the checkpoint: the final weights of the same run score
  14.6% worse on held-out text than the selected ones, because the run was
  continued past its validation minimum on purpose, to measure that.

## Reproducibility

```bash
cd scripts/tiny-gpt
pip install -r requirements.txt
python src/data.py --tokenizer byte_bpe --vocab-size 512 --out-dir data/prepared-bpe512
python -u src/train.py --config configs/main.yaml --run-id main      # ~26 min on 2 cores
python src/evaluate.py --checkpoint artifacts/runs/main/best.pt --split test
```

Seed 1337; batches are a pure function of `(seed, step)`; the data manifest
records the corpus hash, the tokenizer hash, the split boundaries and the code
revision. The run manifest in [`plan.md`](plan.md) records the commit, the
environment, every checkpoint and the full metric stream
([`metrics/main-metrics.jsonl`](metrics/main-metrics.jsonl)).

Weights are not committed: `best.pt` and `last.pt` are 38 MB each of FP32
weights plus AdamW moments, and regenerating them takes 26 minutes.

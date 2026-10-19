# Model card — `tiny-gpt / main-gutenberg-4800`

A teaching artifact: the smallest honest end-to-end language model this
repository's Track 1 produces. Trained from random initialization on two CPU
cores in 49 minutes.

This card describes the 4,800-step model, which is the released one. Its
2,400-step predecessor `main-gutenberg` is still committed and documented in
[`eval-report.md`](eval-report.md); the two differ only in horizon.

## Model details

| | |
| --- | --- |
| Architecture | Decoder-only transformer (pre-norm, learned positional embeddings, tied input/output embeddings) |
| Layers / width / heads | 3 / 192 / 3 |
| Context length | 256 tokens |
| Parameters | **1,770,816** total; 1,721,664 non-embedding |
| Vocabulary | 2,048, byte-level BPE (trained on a 4 M-character sample of the training split) |
| Dropout | 0.0 |
| Precision | fp32 |
| Checkpoint | `scripts/tiny-gpt/artifacts/runs/main-gutenberg-4800/best.pt` (81 MB with optimizer state; weights identical to `last.pt`) |
| Date | 2026-10-18 |
| License | MIT, same as this repository |

## Intended use

Teaching and reference. It exists to make the Track 1 walkthrough falsifiable:
every number in `eval-report.md` can be reproduced from the pinned config and
manifest on commodity hardware without a GPU.

**Out of scope:** any production use, any use where output accuracy matters,
and any task-based claim. It has no instruction tuning, no alignment, no
safety filtering of any kind. It is a next-token predictor over 19th-century
novels.

## Training data

`gutenberg-mixed-27`: 27 public-domain books from Project Gutenberg via the
GITenberg mirrors, 18,777,471 characters.

- Corpus SHA-256 `1960eb6375c67d9e0f0b4fb96f0ab0724def0d74c547b3baba83faff851dd236`
- Gutenberg license headers and footers stripped: 524,204 characters removed across 27 files
- Split **within each document**, contiguous 90/5/5, snapped to blank lines, documents joined with an EOS token
- Tokens: train 5,771,560 / val 317,418 / test 323,284
- Audit: zero matches for email, phone, IPv4, URL, AWS-key and private-key patterns; duplicate 512-character blocks 1,372 / 57,115 (2.40%)

Splitting per document rather than globally is deliberate: a global split would
have put whole books on one side of the boundary and measured
*book-to-book* transfer, not language modelling.

Full provenance — per-document IDs, titles, authors, years, source repositories,
character counts and boilerplate removal stats — is in `data-manifest.json`.

## Training procedure

| | |
| --- | --- |
| Steps / tokens | 4,800 / 39,321,600 (6.81 epochs) |
| Batch | 16 sequences × 256 tokens × 2 accumulation = 8,192 tokens/step |
| Optimizer | AdamW, β = (0.9, 0.99), weight decay 0.1 (decoupled; no decay on norms/biases) |
| LR schedule | 3e-3 peak, 240-step linear warmup, cosine to 3e-4 over the full 4,800 steps |
| Gradient clipping | 1.0 (engaged on 63 steps, 1.3%) |
| Seed | 1337 |
| Hardware | 2 vCPU Intel Xeon, 3 GB RAM, no GPU |
| Wall clock | 2,926 s (48.8 min) at 13,437 tokens/s, 166.5 GFLOP/s, peak RSS 1,166 MB |

Architecture and vocabulary were chosen by a four-configuration pilot matrix at
8 minutes each, documented with its arithmetic in `pilots.md`. The horizon was
doubled from 2,400 steps afterwards, because that run ended compute-limited —
see [`eval-report-4800.md`](eval-report-4800.md).

Four of those pilot choices have since been re-tested at the full budget.
The released configuration survived three of them and tied the fourth. Read them
against the measured noise floor: three runs that differ only in seed span
**0.37%** of test bits per byte ([`data-scaling-and-seed-bar.md`](data-scaling-and-seed-bar.md)),
so any margin below that is not a result.

- **Model size** ([`size-sweep.md`](size-sweep.md)) — against 3.74 M and 5.64 M parameter models, this size tied the first (0.2%) and beat the second (2.1%). Released on a tie rather than a clear win.
- **Context length** ([`context-sweep.md`](context-sweep.md)) — bracketed at 128, 512 and 1024 tokens. 256 wins all three comparisons, by 0.64%, 1.56% and 10.30% of test bits per byte. It is an interior optimum, not an edge.
- **Peak learning rate** ([`lr-bracket.md`](lr-bracket.md)) — bracketed at 1.5e-3 and 6.0e-3. 3e-3 wins both, by 0.83% and 0.08%, but the margin over the higher rate is inside single-seed noise and the response is asymmetric: this value sits on the lower edge of a plateau rather than at its peak.
- **Training data** ([`clean-data-scaling.md`](clean-data-scaling.md)) — the largest effect in this project, and the one that took two experiments to read correctly. Against a 3.53× larger corpus *evaluated only on the original 27 books*, this corpus appeared to win by 0.53% of test bits per byte. Re-run with the evaluation covering all 71 books, the larger corpus wins by 7.30%, and the decomposition says why: the extra text costs +0.35% on the original 27 books — inside the 0.37% seed bar, so a tie — and gains 10.47% on the rest. **A model trained on all 71 books is the better artifact.** This one is released for continuity: every number in this directory is measured against the 27-book test split.

Two settings in the table below have *not* been swept at this budget: the
8,192-token optimizer batch and `grad_clip` 1.0. The learning-rate bracket
showed those two interact — at 1.5e-3 the clip binds on 54.8% of steps — so
neither should be assumed neutral.

## Evaluation

Held-out test split, 323,046 tokens / 931,467 bytes. Full detail in
[`eval-report-4800.md`](eval-report-4800.md).

| | Bits per byte |
| --- | ---: |
| **This model** | **1.6569** |
| bzip2 -9 (primed) | 2.4497 |
| Bigram counts | 2.1953 |
| Unigram counts | 3.1435 |
| Uniform | 3.8149 |

Perplexity 27.43 (comparable only against the same 2,048-token vocabulary).
Validation bits per byte 1.6158, still falling when the step budget ended.
The 2,400-step predecessor scored 1.7201 test bits per byte, so doubling the
compute bought 3.7%.

Memorization probe: longest verbatim match against the training text **24
characters** over 4 × 240-token samples — slightly *shorter* than the
2,400-step model's 26, despite twice the epochs.

## Limitations

- **Semantically incoherent beyond a clause.** It produces well-formed English sentences that do not mean anything and do not stay consistent across paragraphs. This is a property of the scale, not a bug to be fixed by sampling settings.
- **Undertrained.** Validation loss was still improving at the final step, though the last four evaluations bought only 0.0098 bpb against the previous run's 0.0302. The published numbers are a floor, not a converged result — but the next doubling of steps would buy little. The open lever is model size, not steps.
- **Narrow domain.** 19th-century literary English. Expect sharp degradation on modern prose, code, dialogue, or any other language.
- **256-token context.** It cannot attend beyond roughly a page.
- **Will reproduce its training data's biases.** The corpus is 19th-century fiction, with that era's attitudes on race, gender and class present in the text. Nothing in this pipeline detects or mitigates them.
- **No safety behaviour whatsoever.** No tuning, no filtering, no refusals.
- **Single seed.** No variance estimate on any reported number.

## Reproducing

```bash
cd scripts/tiny-gpt
python src/data.py --corpus gutenberg-mixed-27 --tokenizer byte_bpe --vocab-size 2048 \
  --out-dir data/prepared-gb-bpe2048
python src/train.py --config configs/main-gutenberg-4800.yaml
python src/evaluate.py --checkpoint artifacts/runs/main-gutenberg-4800/best.pt --split test
```

The data step has now been run twice on different machine states, once after
the corpus cache had been wiped entirely: it re-downloaded all 27 books and
reproduced the corpus SHA-256 and every split size exactly, and an earlier
re-run reproduced `train.bin` byte-identically. `data-manifest.json` describes
exactly the bytes behind these weights.
Everything needed to check this card — config, manifest, raw metrics for all
five runs, evaluation JSON, compressor baselines, samples — is committed
alongside it.

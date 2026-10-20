# Model card — `tiny-gpt / data71-full-9600`

The released model as of 2026-10-20. Same architecture as its predecessor
`main-gutenberg-4800`, trained on 3.3× more text for 2× as many steps: 1.77M
parameters, two CPU cores, 102 minutes.

It supersedes [`model-card.md`](model-card.md), which stays in place as the
card for the 27-book model and as the record of every number published before
this one.

## Why it replaces the previous release

The 27-book model was kept as the release through two earlier experiments on
the grounds that the 71-book model was merely *tied* with it on the 27-book
test split (+0.35%, inside the 0.37% seed bar) while being better elsewhere —
continuity was free. Doubling the horizon removed that argument. Scored on the
exact bytes the old release is scored on, this model is **2.60% better**, or
7.0× the seed bar. It is now better on the old benchmark, the new benchmark,
and every slice in between. See
[`clean-data-scaling.md`](clean-data-scaling.md#horizon--spending-the-compute-the-capacity-test-said-to-spend).

## Model details

| | |
| --- | --- |
| Architecture | Decoder-only transformer (pre-norm, learned positional embeddings, tied input/output embeddings) |
| Layers / width / heads | 3 / 192 / 3 |
| Context length | 256 tokens |
| Parameters | **1,770,816** total; 1,721,664 non-embedding |
| Vocabulary | 2,048, byte-level BPE |
| Dropout | 0.0 |
| Precision | fp32 |
| Config | [`config-data71-full-9600.yaml`](config-data71-full-9600.yaml) |
| Checkpoint | `scripts/tiny-gpt/artifacts/runs/data71-full-9600/best.pt` (step 9,600; the argmin of the validation curve is its final point, so `best.pt` and `last.pt` hold the same weights) |
| Date | 2026-10-20 |
| License | MIT, same as this repository |

Identical architecture to the previous release — deliberately. Two separate
experiments in this directory found that adding parameters at a matched
compute budget makes things *worse* on this corpus (+1.35%), so the extra
budget went into steps instead.

## Intended use

Teaching and reference, exactly as before. It exists to make the Track 1
walkthrough falsifiable: every number here is reproducible from the pinned
config and manifest on commodity hardware without a GPU.

**Out of scope:** any production use, any use where output accuracy matters,
and any task-based claim. No instruction tuning, no alignment, no safety
filtering. It is a next-token predictor over 19th-century prose.

## Training data

`gutenberg-mixed-71`: 71 public-domain books from Project Gutenberg via the
GITenberg mirrors, 60,467,324 characters, manifest SHA-256 prefix `52290d35`.
Per-document provenance and per-split hashes are in
[`data-manifest-71-full.json`](data-manifest-71-full.json).

| split | tokens |
| --- | ---: |
| train | 18,939,699 |
| validation | 1,042,315 |
| test | 1,038,419 |

**Split policy, stated precisely because it bounds what the evaluation
means.** Every book is split contiguously 90/5/5 and the pieces are
concatenated per split, so *every book contributes to all three splits*. The
test split is the last 5% of each of the 71 books, never shuffled, with
boundaries snapped to a blank line. The behaviour is pinned by
`tests/test_multidoc.py::MultiDocumentSplitTest::test_every_document_appears_in_every_split`,
which is how the first draft of this card was caught claiming the opposite.

This is a deliberate choice and it is the weaker of the two options for
leakage. Holding out whole books would be cleaner, but it would make the
evaluation a different set of authors from training and would therefore
measure domain transfer rather than language modelling. The consequence is
that **1.5662 bits per byte is generalization to unseen passages of seen
books, not to unseen books.** The unseen-book number is measured separately:
the distribution-shift probe in
[`distribution-shift.json`](distribution-shift.json) puts books outside the
training corpus **6.93% harder** than the model's own test split. Quote the
figure that matches the question being asked.

## Training

| | |
| --- | --- |
| Steps | 9,600 |
| Tokens seen | 78,643,200 (4.15 epochs) |
| Batch | 16 sequences × 2 accumulation × 256 tokens = 8,192 tokens/step |
| Optimizer | AdamW, β 0.9/0.99, weight decay 0.1 |
| Learning rate | 3e-3 peak, cosine to 3e-4, 240-step warmup |
| Gradient clipping | 1.0; engaged on 38 steps |
| Non-finite steps | 0 |
| Seed | 1337 |
| Hardware | 2 vCPU, fp32, no GPU |
| Wall clock | 6,131 s (102.2 min) at 12,827 tokens/s |
| Peak RSS | 1,183 MB |

## Evaluation

Full held-out test split, 1,038,266 tokens / 3,025,186 bytes. Bits per byte,
so the figures are comparable across tokenizers.

| | bpb | vs model |
| --- | ---: | ---: |
| **This model** | **1.5662** | — |
| Bigram | 2.1560 | 1.377× |
| Unigram | 3.1056 | 1.983× |
| Untrained, same architecture | 3.7859 | 2.417× |
| Uniform over the vocabulary | 3.7753 | 2.410× |

Final validation bits per byte: 1.5417.

### Per document

All 71 test documents scored separately (`evaluate.py --per-document`):

| | book | bpb |
| --- | --- | ---: |
| easiest | Pride and Prejudice | 1.3511 |
| hardest | On the Origin of Species | 2.1509 |

Mean 1.5798, standard deviation 0.1612, range 0.7998 — **50.6% of the mean**.
The spread between books is larger than any effect this project has measured
by changing the model. Read the aggregate figure as a property of the corpus
mixture as much as of the model, and never compare it against a number
measured on a different mixture.

### Comparability

* Against the previous release on the **27-book** test bytes: 1.6139 vs
  1.6569, **2.60% better**.
* Against `data71-full` (same corpus, half the steps) on the **71-book** test
  split: 1.5662 vs 1.6147, **3.00% better**.
* Against any validation number published before 2026-10-19: **not
  comparable.** Those were measured on a prefix of the validation split
  covering 8 of 27 books. Test numbers are unaffected.

## Known failure modes

Unchanged in kind from the previous release, and worth restating because a
better bits-per-byte figure does not fix any of them:

* Fluent-sounding sentences with no referential content; it mixes characters
  and settings from different books within a paragraph.
* No factual reliability of any kind.
* 256-token context: it cannot track anything across a page.
* Degrades on non-19th-century, non-English, or structured text. A
  distribution-shift probe measured +6.93% on books outside its training
  corpus.

Memorization probe: longest verbatim match against the training split across
4 samples of 240 tokens was **29 characters** (mean 27.5). That is phrase
length, not passage length, and it is the expected scale for a 1.77M-parameter
model over four epochs. The corpus is public domain, so this is a calibration
of memorization behaviour rather than a privacy finding.

## Reproducibility

```bash
cd scripts/tiny-gpt
python src/data.py prepare --corpus gutenberg-mixed-71 --out data/prepared-71-full
python src/train.py --config configs/data71-full-9600.yaml
python src/evaluate.py --checkpoint artifacts/runs/data71-full-9600/best.pt \
    --split test --per-document
```

Takes about 2 hours on two cores, plus roughly 10 minutes to fetch and prepare
the corpus. The data preparation is deterministic: re-running it in a fresh
checkout reproduced the manifest hash and all three split token counts exactly.

# LLM pretraining experiment plan

Filled-in copy of [`templates/experiment-plan.md`](../../templates/experiment-plan.md).
Run family: **2026-10-18-gutenberg-mixed**. Iteration 2 of the bundled
implementation, and a direct answer to the roadmap item that
[iteration 1](../2026-10-17-tiny-shakespeare/plan.md) ended on.

## Goal and constraints

**What iteration 1 found.** A 3.3M-parameter GPT trained on 1 MB of Shakespeare
reached 2.3284 bits/byte on held-out text, beating every baseline — but its
validation loss turned at step 750 of 1500, **11.9 epochs** into a corpus of
517,246 training tokens. Past that point the run was not learning language, it
was memorizing a small text: the last checkpoint scored 2.6685 bpb against the
selected checkpoint's 2.3284, 14.6% worse. The bottleneck was data, not
architecture, optimizer, or compute. The roadmap item was explicit: *get more
text before touching anything else.*

**Goal of this run.** Train the same architecture family on a corpus large
enough that the same compute budget is spent learning rather than memorizing,
and quantify what that buys. Success is not "a lower number than last time" —
the corpora differ, so the numbers are not comparable — it is a run whose
validation loss is still falling when the compute runs out, and a held-out
result that beats a baseline ladder measured on *this* corpus.

| Constraint | Value |
| --- | --- |
| Hardware | 2 vCPU Intel Xeon (Sapphire Rapids), 3 GB RAM, **no GPU**, `torch.get_num_threads()` forced to 2 |
| Framework | PyTorch 2.14.1, FP32 (no AMP: `torch.autocast` on CPU offers nothing here) |
| Wall-clock budget | ~35 min of pilots + ~30 min for the main run |
| Data | public domain only, with per-document provenance |
| Reproducibility | fixed seed, pinned deps, checksummed corpus, manifest for every artifact |

**Success criteria, fixed before the run:**

1. Validation bits/byte **still improving at the final step** — the opposite of iteration 1's turn at 11.9 epochs.
2. Held-out bits/byte beats uniform, unigram, bigram, and an untrained model of the same shape, on this corpus.
3. The pilot matrix answers the two questions iteration 1 could not: **vocabulary size** and **model capacity**, both decided at a matched wall-clock budget rather than a matched token budget.
4. Every number in this directory is reproducible from the committed configs, the pinned corpus hash, and the recorded seed.

## Data card

| Field | Value |
| --- | --- |
| Name | `gutenberg-mixed-27` |
| Composition | 27 public-domain books, 1605–1908: novels (Austen ×3, Dickens ×2, Brontë ×2, Twain ×2, Wells ×2, Melville, Shelley, Stoker, Stevenson, Wilde, Doyle, Conrad, Alcott, Montgomery, Carroll, Cervantes, Dostoyevsky), non-fiction (Darwin, Douglass), poetry (Whitman), folk tales (Grimm) |
| Source | one GITenberg repository per book, `master` branch tarball from `codeload.github.com`, member `<slug>_<pg-id>-master/<pg-id>-0.txt` |
| Access date | 2026-10-18 |
| Size | **18,777,471 characters** after boilerplate removal (17× iteration 1) |
| Corpus SHA-256 | `1960eb6375c67d9e0f0b4fb96f0ab0724def0d74c547b3baba83faff851dd236` — pinned in `src/data.py`; a mismatch aborts preparation |
| Per-document record | title, author, year, source URL, archive member, chars, bytes, SHA-256, and boilerplate-removal statistics, all in `manifest.json` |
| Text licence | public domain (all works pre-1929, authors long deceased) |
| File licence | the GITenberg packaging is itself public-domain/CC0-equivalent; the Project Gutenberg **header and footer are not** part of the public-domain work and are removed |
| Boilerplate removed | **524,204 characters** across 27 files, 19,249–19,674 per file — the tight range is the evidence that the header/footer regex matched every document rather than silently missing some |
| Sensitive-pattern scan | email 0, phone-like 0, IPv4 0, URL 0, AWS-key-like 0, private-key header 0 |
| Duplicate 512-char blocks | 1,372 of 57,115 = **2.40%** (iteration 1: 1.02%) — repeated chapter headings and formulaic phrasing across 27 books; nothing removed, the rate is reported so the decision is visible |
| Longest / shortest document | Don Quixote 2,288,489 chars / Alice in Wonderland 144,519 chars |

**Split policy.** Each document is split contiguously **90/5/5 within itself**,
and the pieces are concatenated per split. Boundaries snap to a blank line so
no training window begins mid-paragraph.

The alternative — holding out whole books — was rejected deliberately. It
would be cleaner for leakage, but validation would then be a different set of
authors from training, and the metric would measure domain transfer rather
than language modelling. Splitting within each document keeps all 27 authors
in all three splits; splitting *contiguously* keeps overlapping context windows
off both sides of each boundary. Both failure modes are avoided; the residual
cost is a mild distribution shift between a book's middle and its end.

Documents are joined with the tokenizer's EOS id. EOS contributes zero bytes,
so it is excluded from both the numerator and the denominator of bits-per-byte.

| Split | Tokens (vocab 2048) | Bytes | Documents | EOS separators |
| --- | --- | --- | --- | --- |
| train | 5,771,560 | 17,052,870 | 27 | 26 |
| val | 317,418 | 934,984 | 27 | 26 |
| test | 323,284 | 932,099 | 27 | 26 |

## Tokenizer

Byte-level BPE, implemented in `src/tokenizer.py` without a tokenizer library,
trained on the **training split only** and capped at a 4,000,000-character
sample of it (merge counting is O(corpus × merges); the vocabulary a few MB
produces is indistinguishable from the one the whole corpus produces, and the
cap is recorded in the manifest).

| Vocabulary | Bytes/token on val | Train tokens | SHA-256 (first 16) |
| --- | --- | --- | --- |
| 1024 | 2.5456 | 6,686,737 | `09d8fecc43675f9f` |
| **2048** | **2.9456** | 5,771,560 | `6c223a63a523a7a0` |

Iteration 1 used vocab 512 at 1.889 bytes/token on a much smaller corpus. Which
of these two is better is a measurement, not a preference — see pilots g1/g2.

## Model configuration

Decoder-only GPT, pre-norm, learned position embeddings, tied input/output
embeddings, no biases, GELU MLP at 4× width — unchanged from iteration 1 except
where a pilot says otherwise.

| | g1/g2 (baseline) | g3-small | g4-wide |
| --- | --- | --- | --- |
| Layers | 4 | 3 | 6 |
| `d_model` | 256 | 192 | 384 |
| Heads | 4 | 3 | 6 |
| Context | 256 | 256 | 256 |
| Vocabulary | 1024 / 2048 | 2048 | 2048 |

**Why capacity is a question at all.** The compute budget is ~12M tokens at
~23 MFLOP/token, about 2.9e14 FLOPs. Setting `C = 6ND` with Chinchilla's
`D = 20N` gives `N = sqrt(C/120) ≈ 1.5M` parameters — *smaller* than the 3.3M
model iteration 1 used. The budget argues the model was oversized for its
compute, not undersized. That is a hypothesis about a ratio derived on a very
different scale, so it gets tested in both directions rather than believed.

## Pilot matrix

Iteration 1 already measured, on this hardware, that byte-BPE beats a character
vocabulary (19% bpb), that lr 3e-3 beats 1e-3 (11%) and 4.2e-3 (0.8%), that
dropout 0.0 beats 0.1 at this budget *and* runs 40% faster, and that untying
embeddings is not worth its throughput. Those are carried over unchanged rather
than re-measured; `configs/pilot-gb-base.yaml` records them with their evidence.

This matrix tests only what changed — a corpus 17× larger:

| Pilot | Question | Change | Budget |
| --- | --- | --- | --- |
| g1-vocab1024 | does a bigger vocabulary pay for its slower output layer? | vocab 1024 | 400 steps |
| g2-vocab2048 | " | vocab 2048 | 400 steps |
| g3-small | is the model oversized for the compute, as `6ND` suggests? | 3L/192d, ~0.42× FLOPs/token | 800 steps |
| g4-wide | or undersized, now that data is no longer the constraint? | 6L/384d, ~3.4× FLOPs/token | 130 steps |

**Pilots are matched on wall-clock, not on tokens.** At a fixed token budget a
bigger model simply gets more compute, and the comparison answers nothing a
practitioner can act on. The step counts above were chosen so each pilot costs
roughly eight minutes at its own measured throughput; the comparison is then
"what is the lowest validation bits/byte reachable in eight minutes", which is
the decision actually facing a 30-minute budget.

All pilots: seed 1337, 8,192 tokens per optimizer update (batch 16 × grad-accum
2 × context 256), lr 3e-3 cosine to 3e-4 with 10% warmup, AdamW β=(0.9, 0.99),
weight decay 0.1 on matrices only, grad clip 1.0, dropout 0.0, FP32, 2 threads,
validation every 50 steps on a fixed 65,536-token slice.

## Pilot measurement

See [`pilots.md`](pilots.md).

## Run manifest

Executed 2026-10-18 as `main-gutenberg` from [`config-main.yaml`](config-main.yaml).

| | |
| --- | --- |
| Model | 3 layers / 192 width / 3 heads, context 256, tied embeddings, 1,770,816 params |
| Data | `prepared-gb-bpe2048` — 27 books, vocab 2,048, train 5,771,560 tokens |
| Steps / tokens | 2,400 / 19,660,800 (3.41 epochs) |
| Schedule | lr 3e-3, 240-step warmup, cosine to 3e-4; AdamW β=(0.9,0.99), wd 0.1, clip 1.0 |
| Wall clock | 1,702 s (28.4 min) at 11,552 tok/s, 143.2 GFLOP/s, peak RSS 1,147 MB |
| Stability | 70 clipped steps (2.9%), zero non-finite losses |
| Result | val **1.6815** bpb, still falling at the final evaluation |

Raw record in [`metrics/main-gutenberg.jsonl`](metrics/main-gutenberg.jsonl)
and [`metrics/main-gutenberg-summary.json`](metrics/main-gutenberg-summary.json).

## Evaluation and release gate

Full detail in [`eval-report.md`](eval-report.md). Against the gates set above:

| Gate | Target | Result | |
| --- | --- | --- | --- |
| Validation improves to the end | no turn-up | monotone across all 20 evaluations | pass |
| Beats the bigram counter | < 2.1953 bpb | **1.7201** (1.28×) | pass |
| Beats the best classical compressor | < 2.4497 bpb | **1.7201** (1.42×) | pass |
| Beats random init | < 3.8269 bpb | 1.7201 | pass |
| Memorization probe | reported, not zero-by-accident | 26 chars, after fixing a probe that measured nothing | pass |
| Reproducible from the manifest | byte-identical re-prepare | `train.bin` reproduced byte-identically | pass |
| Tests green | all | 58/58 | pass |

Released: evaluation report, model card, samples, curves, all five runs'
metrics, data manifest, compressor baselines.

## Next-step roadmap

Ordered by what the evidence says is limiting, not by what is interesting:

1. ~~**More steps on this exact configuration.**~~ **Done** — see
   [`eval-report-4800.md`](eval-report-4800.md). Doubling to 4,800 steps
   bought 3.7% on test bits per byte (1.7201 → 1.6569) with no overfitting at
   6.81 epochs; the train/val gap narrowed by two thirds. The prediction held,
   but the final four evaluations bought 0.0098 bpb against the shorter run's
   0.0302, so a third doubling is no longer the best use of the next hour.
2. **Re-run the size sweep at the longer budget — now the top item.** The
   pilots picked 1.77 M over the ~3.3 M the 6ND rule suggests, measured at 8
   minutes, and that model has since been given 6× that budget. A model deep
   into diminishing returns whose generalization gap keeps *shrinking* is the
   signature of one that is too small for its compute, so this is where the
   next improvement most likely is.
3. **Context length.** Fixed at 256 for every run so far, never varied. Prose
   plausibly rewards more; attention cost grows quadratically, so measure
   rather than assume.
4. **A second seed.** Every number in this directory is a single draw. One
   repeat of the main run would put a bar on the headline.
5. **Vocabulary above 2,048.** 2,048 beat 1,024 on token-matched loss. The
   trend's direction is known; its end is not.

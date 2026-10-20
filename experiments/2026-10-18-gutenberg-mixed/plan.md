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
2. ~~**Re-run the size sweep at the longer budget.**~~ **Done** — see
   [`size-sweep.md`](size-sweep.md). Three sizes at a matched 2,926 s:
   1.77 M scored 1.6569 test bpb, 3.74 M scored 1.6605, 5.64 M scored 1.6920.
   The pilots' 11% margin for the small model has collapsed to 0.2%, which is
   a tie at single-seed precision rather than a win, and 5.64 M is clearly too
   big. The optimum sits in the 1.8–3.7 M band and the crossover is at or just
   beyond this budget — recorded as a prediction, not a result.
3. ~~**Context length.**~~ **Done** — see [`context-sweep.md`](context-sweep.md).
   256 is an interior optimum, bracketed on both sides: T=128 costs +0.64%
   test bpb, T=512 +1.56%, T=1024 +10.30%. The two sides fail differently.
   Above 256 the penalty is intrinsic — it survives token-matching. Below it
   the penalty is pure throughput: at matched tokens T=128 ties T=256 to
   +0.03%, and it loses only because halving the context made the run 9.3%
   *slower*, not faster. Attention at 192 dimensions is not the bottleneck,
   so shrinking it saves nothing while doubling the batch adds overhead.
4. ~~**T=128.**~~ **Done** — folded into item 3 above. Recorded separately
   because it is the one place in this directory where a measured prediction
   came out with the wrong *sign*: the throughput probe promised +5.8% and
   delivered −9.3%, and the step budget had to be re-derived mid-flight from
   a 300-step trial of the real configuration.
5. ~~**The learning-rate recipe.**~~ **Done** — see
   [`lr-bracket.md`](lr-bracket.md). 3e-3 survives a half-decade bracket:
   1.5e-3 costs +0.83% test bpb, 6.0e-3 costs +0.08%. The response is sharply
   asymmetric, so the inherited value sits on the lower edge of a plateau
   rather than at a peak, and a log-quadratic through the three points puts
   the vertex at 3.5e-3. This retroactively strengthens the size and context
   sweeps: both were run within 1.3× of the best rate in the bracket. One
   caveat found along the way — at 1.5e-3 the fixed `grad_clip: 1.0` binds on
   54.8% of steps, so the low arm is partly a clipping experiment.
6. **Batch size, which the bracket did not move.** Learning rate and batch
   size trade off against each other and only the first has been swept;
   8,192 tokens per step is still an inherited number.
7. ~~**A second seed.**~~ **Done** — see
   [`data-scaling-and-seed-bar.md`](data-scaling-and-seed-bar.md). Seeds 1338
   and 1339 of the released configuration put the run-to-run range at **0.37%**
   of test bits per byte (0.24% on val). That swallows exactly the two margins
   predicted here — size 3.74M at +0.22% and lr 6e-3 at +0.08% — and confirms
   both as ties. It also leaves T=128's +0.64% at only 1.7× the bar, so that
   one claim is thinner than the context sweep implied. The released seed,
   1337, is not the best of the three; 1339 would have been quotable at 1.6531.
8. ~~**More data.**~~ **Done, and the answer is no** — see
   [`data-scaling-and-seed-bar.md`](data-scaling-and-seed-bar.md). 44 more
   public-domain books take the training split from 5.77M to **20.4M tokens**
   (3.53×, 6.81 → 1.93 epochs) with val/test and the tokenizer held
   byte-identical. At fixed compute the larger corpus is **+0.53% worse** on
   test and +1.10% worse on val, leads at no point after step 120, and the
   deficit is still ≈1% at the end of the cosine. A probe explains why: a clean
   27-book model finds the 44 additions **6.93% harder** than its own test
   split, so they are a different distribution rather than more of the same,
   and training on them moves the new-book score 17× more than the test score.
   At 1.77M parameters the model cannot hold both. The clean restatement of the
   question — val/test drawn from all 71 books — is now the open item, and it
   would break comparability with every number above, so it needs its own
   baseline.
9. **Vocabulary above 2,048.** 2,048 beat 1,024 on token-matched loss. The
   trend's direction is known; its end is not.
10. ~~**"More data" asked properly.**~~ **Done** — see
    [`clean-data-scaling.md`](clean-data-scaling.md). Item 8 pinned the
    evaluation to the original 27 books, so it could only ever measure
    whether the additions help *there*. Two new corpora,
    `gutenberg-mixed-71-full` and `-71-ctrl`, carry the same 71 documents and
    the same val and test splits — byte-identical, shared vocabulary — and
    differ only in which documents feed training (18,939,699 vs 5,771,560
    tokens). Running both over the identical horizon answers the general
    question and, by decomposing the test split into the 27 both arms train
    on and the 44 only one arm trains on, reproduces item 8's result inside
    the same pair of runs.
11. ~~**Does the size tie break on more data?**~~ **Done, no** — folded into
    [`clean-data-scaling.md`](clean-data-scaling.md). The size sweep's tie was
    measured at 6.81 epochs, so it only ever tested capacity *at that data
    volume*. Re-run at 2.08 epochs on the 71-book corpus and wall-clock
    matched, 3.74M parameters is +1.35% worse than 1.77M, against +0.22%
    before: the tie widened rather than broke. More data makes a *step* more
    valuable, not a parameter, and the bigger model pays for its capacity with
    46% fewer steps. Grow the model after the compute, not instead of it.
12. ~~**Spend the compute on steps instead, then.**~~ **Done** — folded into
    [`clean-data-scaling.md`](clean-data-scaling.md). Item 11 said the binding
    constraint is optimization steps, so the prediction was that doubling the
    horizon should pay. It paid 3.00% of test bits per byte (8.1× the seed
    bar) for 2× the compute, and the resulting model is better than the
    previous release on the previous release's own test bytes by 2.60%, which
    is why it is now the released model
    ([`model-card-71.md`](model-card-71.md)). The marginal return is falling —
    3.67% for the previous doubling, 3.00% for this one, and 3.2× flatter
    inside the run — so one more doubling is worth roughly 2% and the step
    axis is close to exhausted. The per-document evaluation added alongside it
    says where the remaining signal is: 50.6% of spread between the easiest
    and hardest book, against under 8% for any lever tried so far. **More
    books beats more passes from here.**
13. ~~**Batch size, still untouched.**~~ **Done** — written up in
    [`batch-size.md`](batch-size.md). Five arms at a fixed 39,321,600-token
    budget, which is the only way to trade optimizer updates against batch
    without changing anything else. At the inherited rate the response has an
    interior optimum — halving the batch costs 0.63%, doubling it costs 1.71%
    — but the inherited rate was tuned at the centre batch, so both outer arms
    were re-run at `3.0e-3 × √(B/8192)`. Three results: the square-root rule
    is worth about 0.75% and transferred to within 0.04 percentage points
    across a 4× range; the optimum is **asymmetric** (below 8,192 is a tie,
    above it is +0.98% and no rate tested fixes that); and the "optimizer
    updates are the binding constraint" reading of items 11 and 12 is
    **false** — twice the updates on identical tokens can at best draw. Item 6
    is answered: clip rate tracks the gradient norm rather than the learning
    rate, and is a symptom to investigate rather than a quantity to optimize.
14. **The corpus is the only direction left, and it is now specified.**
    Every hyperparameter in the standings table has been tested and is flat
    or harmful; three of the four inherited defaults survived. The
    per-document evaluation puts 50.6% of spread between the easiest and
    hardest book against under 8% for any lever, and
    [`corpus-composition.md`](corpus-composition.md) says what that spread is
    made of: not length (r = −0.122), but **form** — the corpus is 91%
    narrative prose by bytes and contains no drama at all, and the four form
    labels explain 45% of the per-document variance. Verse is +27.8% against
    narrative prose, experimental prose +23.4%, non-fiction +19.3%, and each
    holds about 3% of the bytes. Books by an author who appears more than
    once score 5.0% better, which is the evidence that these forms are
    under-trained rather than intrinsically unpredictable. The next
    experiment is a `gutenberg-mixed-broad` corpus that raises the
    non-narrative share to roughly a third, run against a control with a
    byte-identical evaluation split, and reported **per form** rather than in
    aggregate.
15. **Re-establish a baseline under `eval_spread`.** The in-training
    validation metric covered the first 8 of 27 books until it was fixed, so
    every published val number is a number about those eight. Test numbers,
    and therefore every conclusion drawn so far, are unaffected. The spread
    slice is opt-in to keep the committed configs reproducible; making it the
    default needs one re-run of the released configuration to re-anchor the
    val series.

# Evaluation report — `main-gutenberg`

> **Superseded as the released model.** This run was extended to 4,800 steps;
> see [`eval-report-4800.md`](eval-report-4800.md) for the current model
> (test 1.6569 bpb) and the head-to-head. This report stays as written — it is
> the record of the 2,400-step run and the baseline the extension is measured
> against.

Iteration 2 of the Track 1 walkthrough. A 1.77 M-parameter decoder-only GPT
trained from random init on 27 public-domain books, on two CPU cores.

- **Run:** `main-gutenberg`, 2,400 steps, 19,660,800 tokens (3.41 epochs), 28.4 min
- **Checkpoint:** `best.pt` (step 2,400) — byte-identical to `last.pt`, see below
- **Data:** `gutenberg-mixed-27`, byte-level BPE, vocab 2,048
- **Headline:** test **1.7201 bits/byte**, perplexity 31.12

## 1. Held-out test results

Scored over the full test split (323,046 tokens / 931,467 bytes) with
non-overlapping windows. Every row is scored on the *same* tokens with the
*same* byte denominator, so bits per byte is comparable across rows even though
perplexity is not comparable across tokenizers.

| Model | Loss (nats/token) | Perplexity | Bits per byte |
| --- | ---: | ---: | ---: |
| Uniform over vocabulary | 7.6246 | 2048.00 | 3.8149 |
| This architecture, random init | 7.6486 | 2097.66 | 3.8269 |
| Unigram counts (train split) | 6.2828 | 535.27 | 3.1435 |
| Bigram counts, add-0.1 (train split) | 4.3877 | 80.46 | 2.1953 |
| **Trained model** | **3.4378** | **31.12** | **1.7201** |

The trained model is **1.28×** better than the bigram baseline and **2.22×**
better than uniform. Random init scores *worse* than uniform (3.8269 vs
3.8149), which is the expected sanity signal: an untrained network is a
confidently wrong distribution, not a uniform one.

### Against general-purpose compressors

Same decoded test text, so these are directly comparable to the table above.
"Primed" gives the codec a 400 KB warm-up from the end of the training split —
the fair comparison, since the model also saw the training data.

| Compressor | Standalone | Primed |
| --- | ---: | ---: |
| gzip -9 | 3.1187 | 3.1099 |
| xz -9 | 2.6599 | 2.5138 |
| bzip2 -9 | 2.4603 | **2.4497** |
| **Trained model** | — | **1.7201** |

The model beats the best classical compressor by **1.42×**. This is the result
that moved most between iterations: in iteration 1 the model (2.3284) barely
edged out primed bzip2 (2.4638), a 1.06× margin. Seventeen times more training
text, not a better architecture, produced the difference.

*Byte-count footnote:* the compressors were run on 931,385 decoded bytes
against the model's 931,467 — an 82-byte (0.009%) difference from window
truncation at the tail of the split. It does not move any figure above.

### Why only one checkpoint was evaluated

`best.pt` and `last.pt` have **identical weights** (SHA-256 of the sorted state
dict: `8ad7068430c6d866…` for both). Validation loss was still falling at the
final evaluation, so the best checkpoint *is* the last one. Iteration 1 needed
both because that run overfit and the two diverged; here, reporting them
separately would have been two names for one number.

## 2. Training behaviour

Validation bits per byte, measured every 120 steps on a fixed 98,304-token slice:

| step | 120 | 480 | 960 | 1440 | 1920 | 2160 | 2280 | 2400 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| val bpb | 2.4681 | 2.0754 | 1.8785 | 1.7715 | 1.7117 | 1.6909 | 1.6846 | **1.6815** |

Monotone decreasing at every single evaluation — twenty of twenty. See
`loss-curve.svg`, and `pilot-vs-main.svg` for the pilot and the main run on one
axis.

**The run was stopped by its step budget, not by convergence.** The last three
evaluations still improved by 0.0063, 0.0037 and 0.0031 bpb. This model is
compute-limited, not data-limited — the opposite of iteration 1, which was
data-limited and overfit after 11.9 epochs on a 1.1 MB corpus.

| | iteration 1 (tiny-shakespeare) | iteration 2 (gutenberg-mixed-27) |
| --- | --- | --- |
| Corpus | 1.1 MB | 18.8 MB |
| Epochs | 11.9 | 3.41 |
| Best val bpb | 2.1066 @ step 750 | 1.6815 @ step 2400 (final) |
| End of run | degraded to 2.4147 | still improving |
| Diagnosis | overfit, data-limited | underfit, compute-limited |

### Generalization gap

`train(eval) − val` loss, measured on equal-sized batches from each split:

| step | 120 | 600 | 1080 | 1560 | 2040 | 2400 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gap | +0.1372 | +0.1923 | +0.1967 | +0.1479 | +0.1377 | +0.1180 |

The gap **shrinks** over the second half of training. That is the signature of
a model still absorbing signal rather than memorizing — had it been overfitting,
this column would widen.

The sign is also worth noting: the gap is *positive*, meaning the training
split scores **worse** than validation. That is not a bug. Validation is the
90–95% slice of each book, and the back of a novel is easier to predict than
its whole body: the characters, places and idiom have all been established.
The same inversion appeared in all four pilots.

### Stability

- Gradient clipping engaged on **70 of 2,400 steps (2.9%)** — active but not dominant, which is what a correctly set `clip=1.0` looks like.
- **Zero** non-finite losses.
- Peak RSS 1,147 MB of the 3 GB available.
- 11,552 tokens/s sustained, 143.2 GFLOP/s achieved.

## 3. Memorization probe

Four 240-token samples, temperature 0.8, top-k 40, compared against the exact
17 MB of text the model trained on:

- Longest verbatim match: **26 characters** (mean 23.8)

Twenty-six characters is a common phrase, not recall — at 3.41 epochs over
18.8 MB there is little incentive to memorize. The corpus is public-domain
text, so this is a calibration of memorization behaviour, not a privacy
finding.

**This number is new.** The probe silently measured *nothing* on the first
attempt and reported "0 characters", which reads like a perfect result. Two
defects, both fixed and now covered by ten regression tests
(`tests/test_memorization_probe.py`):

1. The probe reconstructed training text from a single `raw/<corpus>.txt` file. The 27-document corpus caches one file per document, so it got an empty string and skipped — while the report still printed a zero. A skipped check must now print **"Not measured"**, never a number.
2. The longest-common-substring search built a set of every corpus n-gram at each binary-search step: roughly 1.7 GB of live strings on a 17 MB corpus. Replaced with a greedy extension search using `str.find`, verified equivalent to brute force on 200 randomized cases plus a 2 M-character scale test.

## 4. Samples

Full set in `samples.txt`. At temperature 0.6, prompted with a newline:

```text
“Almost astonished, Mr. Cruncher,” said the curate, “and you
will have done no religious defects, and it will be the greatest
in it.”

“Aye, to get on,” said Mrs. Jaggers, turning over the huntsman
heartily.
```

What a 1.77 M-parameter model learned in 28 minutes: correct dialogue
punctuation including nested quotes and the `,” said X,` attribution pattern;
paragraph structure; capitalisation; and names lifted from the corpus
(Cruncher from *A Tale of Two Cities*, Jaggers from *Great Expectations*).
Register follows the sampling temperature — a 0.9-temperature sample drifts
into *Huckleberry Finn*'s dialect (`“Well,” says I`, `warn't`, `ain't`).

What it did not learn: semantic coherence beyond a clause. "You will have done
no religious defects" is syntactically well-formed and means nothing. Mrs.
Jaggers and Mr. Jaggers appear in adjacent paragraphs. This is the expected
ceiling at this scale, and it is why the evaluation above leads with bits per
byte rather than with cherry-picked text.

## 5. What this does not establish

- **One seed, one architecture at this budget.** No error bars. The pilot matrix compared four configurations at 8 minutes each, not four main runs.
- **One corpus, one language.** 19th-century public-domain literary English. Nothing here predicts behaviour on code, dialogue, other languages, or modern prose.
- **No downstream evaluation.** Bits per byte measures compression of held-out text from the same distribution. It is not a claim about any task.
- **The budget question is still open.** Compute-optimal size scales as √C, so the 3.7×-larger main budget nominally favoured a ~3.3 M-parameter model over the 1.77 M that won the pilots. The pilots' measured curves disagreed with the scaling argument and the measurement was trusted. Since the chosen model was still improving at the step limit, this remains untested at the longer horizon — the next iteration should re-run the size sweep at the main budget rather than inheriting this answer.

## 6. Reproducing

```bash
cd scripts/tiny-gpt
python src/data.py --corpus gutenberg-mixed-27 --tokenizer byte_bpe --vocab-size 2048 \
  --out-dir data/prepared-gb-bpe2048
python src/train.py --config configs/main-gutenberg.yaml
python src/evaluate.py --checkpoint artifacts/runs/main-gutenberg/best.pt --split test
```

Pinned in this directory: `config-main.yaml` (the exact config), `data-manifest.json`
(corpus SHA-256 `1960eb63…`, per-document split boundaries, tokenizer hash
`6c223a63…`), `metrics/` (all five runs' JSONL), `eval-test-best.json`,
`compressor-baselines.json`. Re-running `src/data.py` after the fact reproduced
`train.bin` **byte-identically**, so the manifest describes exactly the bytes
this model was trained on.

Environment: Python 3.11.2, torch 2.14.1 (CPU), 2 vCPU Xeon, 3 GB RAM, no GPU,
`OMP_NUM_THREADS=2`, seed 1337.

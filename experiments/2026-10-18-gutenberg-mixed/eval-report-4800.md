# Evaluation report — `main-gutenberg-4800`

The iteration-2 roadmap's first item was *more steps on this exact
configuration*, on the grounds that the 2,400-step run was compute-limited
rather than data-limited. This run doubles the horizon and tests that claim.

- **Run:** `main-gutenberg-4800`, 4,800 steps, 39,321,600 tokens (6.81 epochs), 48.8 min
- **Changed from `main-gutenberg`:** `max_steps` 2400 → 4800, and the time guard. Nothing else — same architecture, data, seed, and recipe.
- **Headline:** test **1.6569 bits/byte**, perplexity 27.43, from 1.7201 / 31.12

## 1. Why this is a new run and not a resume

Resuming the finished run with a larger `max_steps` would not have given
"the same training, continued". The cosine schedule is a function of the
horizon, so extending it rewrites the schedule underneath the optimizer: a
learning rate that had annealed to its 3e-4 floor jumps back toward peak, and
the result is neither the old run continued nor a clean 4,800-step run.

That is the hazard this project's own resume test documents, and why
`src/train.py` fixes the horizon and interrupts with `stop_after_steps`.
Resume exists to recover an interrupted run, not to extend a finished one.

So the horizon was doubled and the schedule run once over the whole thing.
Both runs stay committed and comparable. The cost of this choice is that the
first 2,400 steps were recomputed rather than reused — about 24 minutes — in
exchange for a clean, single-schedule curve.

Determinism check: step 1 loss is 7.6457 in both runs, and the step-120
evaluation is 2.4681 in both. The runs are bit-identical until their learning
rate schedules diverge.

## 2. Head to head

| | `main-gutenberg` | `main-gutenberg-4800` | change |
| --- | ---: | ---: | ---: |
| Steps | 2,400 | 4,800 | 2× |
| Tokens | 19,660,800 | 39,321,600 | 2× |
| Epochs over the train split | 3.41 | 6.81 | 2× |
| Wall clock | 28.4 min | 48.8 min | |
| **Validation bits/byte** | 1.6815 | **1.6158** | −3.9% |
| **Test bits/byte** | 1.7201 | **1.6569** | −3.7% |
| Test perplexity | 31.12 | 27.43 | −11.9% |
| vs. bigram counts | 1.28× | 1.32× | |
| vs. best primed compressor | 1.42× | **1.48×** | |
| Train/val gap at the end | +0.1180 | **+0.0399** | −66% |
| Longest verbatim match | 26 chars | 24 chars | |
| Gradient clips | 70 (2.9%) | 63 (1.3%) | |
| Non-finite losses | 0 | 0 | |
| Peak RSS | 1,147 MB | 1,166 MB | |

Full test table for this run:

| Model | Loss (nats/token) | Perplexity | Bits per byte |
| --- | ---: | ---: | ---: |
| Uniform over vocabulary | 7.6246 | 2048.00 | 3.8149 |
| This architecture, random init | 7.6486 | 2097.66 | 3.8269 |
| Unigram counts (train split) | 6.2828 | 535.27 | 3.1435 |
| Bigram counts, add-0.1 (train split) | 4.3877 | 80.46 | 2.1953 |
| **Trained model** | **3.3116** | **27.43** | **1.6569** |

Every baseline row is identical to the 2,400-step report to four decimal
places, which is the check that both models were scored on exactly the same
bytes. The compressor baselines are unchanged for the same reason: gzip
3.1187, xz 2.6599, bzip2 2.4603 standalone; 2.4497 for primed bzip2, still the
strongest classical competitor.

## 3. What doubling the compute actually bought

**3.7% on test bits per byte.** That is a real improvement and a modest one,
and the shape of the curve explains why.

Validation bits per byte at matched steps:

| step | 120 | 600 | 1200 | 1800 | 2400 | 3000 | 3600 | 4200 | 4800 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| val bpb | 2.4681 | 2.0245 | 1.8361 | 1.7610 | 1.7161 | 1.6786 | 1.6479 | 1.6260 | **1.6158** |

Note the 2,400 column: **1.7161 here against 1.6815 in the shorter run**. At
the same step the longer run is *behind*, because its cosine is only half
finished and its learning rate is still near 1.6e-3 while the shorter run had
already annealed to 3e-4. The annealing is worth roughly 0.035 bpb, and it is
only collected at the end of whatever horizon you chose. This is the concrete
reason you cannot read "how good will this get" off the middle of a cosine
run.

Improvement over the final four evaluations:

- 2,400-step run: **0.0302 bpb** still arriving in the last 480 steps
- 4,800-step run: **0.0098 bpb** — a 3.1× slowdown

So the second doubling is still paying, but at a third of the rate. The run
remains compute-limited rather than data-limited; it is simply deeper into
diminishing returns.

## 4. Still no overfitting at 6.81 epochs

The `train(eval) − val` gap did not widen. It **narrowed by two thirds**:

| step | 600 | 1200 | 1800 | 2400 | 3000 | 3600 | 4200 | 4800 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gap | +0.1935 | +0.1817 | +0.1671 | +0.1304 | +0.0930 | +0.0694 | +0.0515 | **+0.0399** |

Iteration 1 turned over at 11.9 epochs on a 1.1 MB corpus. Twice the epochs
here produced no turn and no widening, and the memorization probe went *down*
(26 → 24 characters) rather than up. An 18.8 MB corpus is simply not something
a 1.77 M-parameter model can memorize, so repeated passes keep buying signal
instead of recall.

This answers the open question from the previous report in the direction the
roadmap predicted, and it means **more data is still not the binding
constraint** at this model size.

## 5. Scoring the pilot projection a second time

The pilots fitted `bpb = 3.2949 − 0.2195·ln(t)` to an eight-minute window and
recorded it so it could be scored rather than forgotten. It has now been
scored at two distances:

| Checked at | Projected | Actual | Error |
| --- | ---: | ---: | ---: |
| 1,702 s (2,400 steps) | 1.6620 | 1.6815 | +1.2% |
| 2,926 s (4,800 steps) | 1.5430 | 1.6158 | **+4.5%** |

The error nearly quadruples as the extrapolation stretches. Measured between
the two annealed endpoints, the true slope is **−0.1213 per ln(t)**, against
the pilot fit's −0.2195: the short-window fit **overstated long-horizon gains
by 1.8×**.

The lesson is specific and reusable. A log-linear fit to a short window is
good enough to rank configurations and to size the next run by a factor of
two. It is not good enough to promise an absolute loss several doublings out,
and it will always err optimistic, because the steep early part of the curve
contaminates the fit no matter how carefully the tail is selected.

*Caveat on wall-clock:* this run averaged 13,437 tok/s against the earlier
run's 11,552 on the same hardware, because the machine was quieter. Comparisons
between runs should be made on tokens, not seconds; the projection above is
scored in seconds only because that is the unit the prediction was recorded in.

## 6. Samples

Full set in `samples-4800.txt`. Temperature 0.6, prompted with a newline:

```text
“Almost astonishing,” said I, “Is it not the best place in the Bay, and
you are to be bound to keep my witch?”

“Why, I should be a bad thing, if you get it?”

“I am not a word, sir.”
```

And at the end of training, the step-4800 sample opens:

```text
"One thing I never did," said Jo.
```

Dialogue mechanics are now essentially correct — nested quotes, attribution,
paragraph breaks per speaker, and character names used consistently within a
passage (Jo, from *Little Women*). Semantic coherence beyond a clause is
still absent, and 3.7% on bits per byte does not change that. The failure mode
is unchanged from the previous report; it is simply rarer.

## 7. What this does not establish

- **One seed, one architecture.** Same caveat as before; this run changes only the horizon.
- **Not a converged model.** The curve was still descending at step 4,800, just slowly. The 1.6569 figure remains a floor, not a ceiling.
- **The size question is now more open, not less.** The pilots chose 1.77 M parameters at an eight-minute budget. This run spent 6× that budget on the same model. Compute-optimal size grows as √C, and the evidence that this model is deep into diminishing returns while its generalization gap keeps shrinking is exactly the signature of a model that is too small for its compute. The next experiment should be a size sweep at this budget, not a third doubling of steps.

## 8. Reproducing

```bash
cd scripts/tiny-gpt
python src/data.py --corpus gutenberg-mixed-27 --tokenizer byte_bpe --vocab-size 2048 \
  --out-dir data/prepared-gb-bpe2048
python src/train.py --config configs/main-gutenberg-4800.yaml
python src/evaluate.py --checkpoint artifacts/runs/main-gutenberg-4800/best.pt --split test
```

The data step was re-run from scratch for this experiment, on a machine where
the corpus cache had been wiped. It re-downloaded all 27 books and reproduced
the corpus SHA-256 `1960eb63…` and every split size exactly — train 5,771,560,
val 317,418, test 323,284 tokens — so the two runs are comparable on the data
axis by construction, not by assumption.

Pinned here: `config-main-4800.yaml`, `metrics/main-gutenberg-4800.jsonl`,
`metrics/main-gutenberg-4800-summary.json`, `eval-test-4800.json`,
`compressor-baselines-4800.json`, `horizon-comparison.svg`,
`loss-curve-4800.svg`, `samples-4800.txt`.

Environment: Python 3.11.2, torch 2.14.1 (CPU), 2 vCPU Xeon, 3 GB RAM, no GPU,
`OMP_NUM_THREADS=2`, seed 1337.

# Batch size, and the death of the "more steps" hypothesis

The last untouched lever in [`plan.md`](plan.md), and the one that could
falsify the story the previous two experiments had built.

## What the previous results had implied, and why it needed checking

Two results pointed the same way:

* **Capacity.** At matched wall clock, a 2.11× larger model is **1.35% worse**
  on test bits per byte, because it pays for its capacity with 46% fewer
  optimizer updates.
* **Horizon.** Doubling the step count at a fixed batch bought **3.00%**.

The tempting reading is *optimizer updates are the binding constraint*. It is
tempting because it explains both, and it is dangerous because in both
experiments the number of updates moved together with something else — with
parameters in the first, with tokens in the second. Neither isolates updates.

Batch size does. It is the only lever that trades updates against batch at a
**fixed token budget**, so it separates "more updates" from "more tokens".
Three arms, every one of them seeing exactly 39,321,600 tokens:

| arm | tokens/update | updates |
| --- | ---: | ---: |
| `batch-4096` | 4,096 | 9,600 |
| `data71-full` (centre) | 8,192 | 4,800 |
| `batch-16384` | 16,384 | 2,400 |

Warmup is 5% of each arm's own horizon and each arm evaluates exactly 40
times, so the schedule has the same shape in normalised time and the *k*-th
evaluation of every arm lands on the same cumulative token count. The curves
are compared token-for-token and never step-for-step.
[`tests/test_batch_bracket.py`](../../scripts/tiny-gpt/tests/test_batch_bracket.py)
asserts all of that, so a later edit to one config fails CI rather than
quietly weakening the comparison.

## Result: an interior optimum, and the hypothesis fails

| arm | tokens/update | updates | elapsed | clipped | val | test | vs centre |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `batch-4096` | 4,096 | 9,600 | 3,127 s | 936 | 1.5984 | 1.6249 | +0.63% |
| `data71-full` | 8,192 | 4,800 | 3,130 s | 34 | 1.5883 | **1.6147** | — |
| `batch-16384` | 16,384 | 2,400 | 2,974 s | 44 | 1.6175 | 1.6423 | +1.71% |

The prediction was a monotone win for the smallest batch. What happened is the
opposite shape: **the centre wins, and 2× the updates at the same tokens is
0.63% worse.** The response is not monotone, so "updates are what matter" is
false as stated. What the horizon run bought was more tokens *and* more
updates together; neither alone reproduces it.

The wall clock is an accident worth noting: the two smaller arms landed within
0.1% of each other, so this is a *token-matched and clock-matched* comparison
on the lower half of the bracket. The largest arm is 5.0% cheaper, because
fewer, larger updates carry less optimizer and Python-loop overhead per token
— a real saving, and not nearly enough to pay for 1.71%.

Across the 40 token-matched evaluation points the small batch leads the first 7
and is behind for the remaining 33; the largest batch never leads. Extra updates do help early, when any descent direction is a good one.
They stop helping once the gradient's direction matters more than its count.

## The confound, and what happened when it was removed

The bracket above has a hole in it, flagged in the config headers before the
runs started: **3.0e-3 was the rate the LR bracket selected at the centre
batch.** The centre was the only arm running at a learning rate chosen for it,
which biases the experiment toward exactly the result it produced.

So both outer arms were re-run at the square-root-scaled rate,
`lr(B) = 3.0e-3 × √(B / 8192)`, the standard rule for transporting a tuned
rate across batch sizes:

| arm | tokens/update | lr | clipped | test | vs centre | vs its own twin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `batch-4096` | 4,096 | 3.00e-3 | 936 | 1.6249 | +0.63% | — |
| `batch-4096-lr2p12e3` | 4,096 | 2.12e-3 | 1,262 | **1.6125** | **−0.13%** | −0.76% |
| `data71-full` | 8,192 | 3.00e-3 | 34 | 1.6147 | — | — |
| `batch-16384-lr4p24e3` | 16,384 | 4.24e-3 | 26 | 1.6304 | +0.98% | −0.72% |
| `batch-16384` | 16,384 | 3.00e-3 | 44 | 1.6423 | +1.71% | — |

Three things fall out, and they are the real results of this experiment.

**1. The square-root rule works, and it is worth about 0.75%.** Moved to its
scaled rate, the small arm improved 0.76% and the large arm 0.72%. Two
independent runs, 4× apart in batch size and moving their rate in *opposite*
directions, agreeing to within 0.04 percentage points. That is a much cleaner
validation of √B scaling than this project had any right to expect from two
runs, and it is the single most transferable thing here.

**2. Going below the centre is free; going above it is not.** At its own
rate the small batch scores −0.13% against the centre, which is inside the
0.37% seed bar and therefore a tie. The large batch recovers only to +0.98%,
still 2.6× the bar. So the interior optimum is real but **asymmetric**: 8,192
tokens per update is at the top of a plateau, not the peak of a hill. You can
halve it for free if you retune the rate; you cannot double it at any rate
tested.

**3. The steps hypothesis is dead either way.** The best the small batch can
do with twice as many updates on identical tokens is *draw*. Updates are not
the resource; they were a proxy for tokens in the horizon run and a proxy for
model size in the capacity run.

## Mechanism, measured rather than asserted

The clip counts in the first table look like noise — 936, 34, 44 — and they
are the key to the whole thing. The standard story is that a larger batch
averages more samples per update, so its gradients are less noisy and it
clips less. That predicts a monotone decrease, and the raw counts are not
monotone.

Raw gradient norms confound two things: how noisy the gradient is, and how
far along the run is. Binning by **training loss** removes the second — two
samples at the same loss are at comparable points in the optimization,
whatever step produced them. Median pre-clip gradient norm:

| loss band | 4096 @3.0e-3 | 4096 @2.12e-3 | 8192 @3.0e-3 | 16384 @3.0e-3 | 16384 @4.24e-3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| [3.6, 3.9) | 0.618 | 0.723 | 0.555 | 0.611 | 0.470 |
| [3.3, 3.6) | 0.665 | 0.816 | 0.584 | 0.598 | 0.472 |
| [3.0, 3.3) | 0.864 | 0.943 | 0.642 | 0.664 | 0.475 |

Read down the columns and the pattern is two-dimensional:

* **Smaller batch, larger gradient norm at matched loss.** 4,096 sits 34%
  above the centre in the tightest band. That is the noise term, and it is why
  the small arm clips on 9.75% of its updates against the centre's 0.71% —
  roughly one update in ten was truncated, so the extra updates were not full
  updates.
* **Higher learning rate, *smaller* gradient norm at matched loss.** The
  largest batch at 4.24e-3 sits 26% *below* the centre and clips least of all
  (26 events). The same batch at 3.0e-3 sits 3% above it.

That second axis explains the one number in this experiment that looks
backwards: the rescued small arm clips *more* than its unrescued twin (1,262
against 936) while scoring better. Lowering its rate did not calm its
gradients, it raised them — and it still won, because clipping is a symptom
of where in the landscape the optimizer is sitting, not a cost in itself.
**Clip frequency tracks the gradient norm, not the learning rate**, and
neither of them predicts the final loss on its own. A high clip rate is worth
investigating and is not worth optimizing.

## Where this leaves the project

| lever | tested range | effect at full budget |
| --- | --- | --- |
| **training data** | 5.77M → 18.9M tokens | **−7.30%** on matching evaluation |
| **training steps** | 2,400 → 4,800 → 9,600 | **−3.67%**, then **−3.00%** |
| model size | 1.77M → 5.64M params | tie on 27 books, +1.35% worse on 71 |
| context length | 128 → 1024 | worse in both directions |
| peak learning rate | 1.5e-3 → 6e-3 | tie above, worse below |
| **batch size** | 4,096 → 16,384 tok/update | **tie below 8,192, +0.98% above** |
| seed | 1337–1339 | 0.37% noise floor |
| checkpoint selection | best vs final | 0.10% |
| *corpus mixture* | *which 71 books* | *50.6% spread between easiest and hardest* |

Batch size joins context length and peak learning rate in a growing list:
**the inherited default was already right.** Three of the four
hyperparameters this project inherited have now survived a direct test, and
the only one that moved — the step budget — moved because the runs were too
short, not because the setting was wrong.

That is the useful negative result. Every lever in the top half of this table
except data and steps has now been tested and found flat or harmful, which
means there is no tuning left to do at this scale. The corpus-mixture row is
an order of magnitude larger than anything above it and is the only direction
with room in it.

## Artifacts

| file | what |
| --- | --- |
| `metrics/batch-{4096,16384}.jsonl`, `-summary.json` | the fixed-rate bracket |
| `metrics/batch-4096-lr2p12e3.jsonl`, `metrics/batch-16384-lr4p24e3.jsonl`, `-summary.json` | the square-root-scaled follow-up |
| `eval-test-batch-*.json` | held-out test evaluations, full 1,038,419-token split |
| `config-batch-*.yaml` | exact configs; each outer arm differs from its twin in `learning_rate` alone |
| `batch-summary.json` | every number in this document |
| `analyze-batch.py` | the analysis, including the gradient-norm-at-matched-loss table |
| `tests/test_batch_bracket.py` | the design as executable invariants, 13 tests |

The centre arm is `data71-full`, reused from
[`clean-data-scaling.md`](clean-data-scaling.md) rather than re-run: it is
already the exact configuration this bracket needs, and re-running it would
have cost an hour to reproduce a number the repository already has.

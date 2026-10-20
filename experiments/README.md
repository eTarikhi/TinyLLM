# Experiments

Evidence trail for runs of [`scripts/tiny-gpt`](../scripts/tiny-gpt/README.md).
One directory per run family, each containing the filled-in copy of
[`templates/experiment-plan.md`](../templates/experiment-plan.md), the metrics
the run actually produced, the evaluation report, samples, and a model card.

This directory is **not** part of the packaged skill. Strip it when staging a
`.skill` (the README's packaging section lists the exclusions, and
`.github/scripts/check_package.py` warns if it is present).

## Runs

| Date | Directory | What it answers | Outcome |
| --- | --- | --- | --- |
| 2026-10-17/18 | [`2026-10-17-tiny-shakespeare/`](2026-10-17-tiny-shakespeare/plan.md) | Does the bundled implementation pass every gate, and what does a ~26-minute CPU pretraining run actually produce? | All gates passed. 3.3M-parameter GPT, **2.3284 bits/byte** on held-out text vs 2.8233 for a bigram counter and 2.5583 for `xz -9` primed with the training split. Validation turned at 11.9 epochs, so the selected checkpoint is step 750 of 1500 — see the [evaluation report](2026-10-17-tiny-shakespeare/eval-report.md) and [model card](2026-10-17-tiny-shakespeare/model-card.md) |
| 2026-10-18 | [`2026-10-18-gutenberg-mixed/`](2026-10-18-gutenberg-mixed/plan.md) | Iteration 1 ran out of data, not compute. What changes when the corpus grows 17× and the model is sized by measurement rather than by the scaling rule? | 1.77M-parameter GPT on 27 Project Gutenberg books, **1.7201 bits/byte** on held-out text vs 2.1953 for a bigram counter and 2.4497 for primed `bzip2 -9`. Validation was still falling at the step limit: compute-limited, not data-limited. See the [evaluation report](2026-10-18-gutenberg-mixed/eval-report.md) |
| 2026-10-18 | [`2026-10-18-gutenberg-mixed/`](2026-10-18-gutenberg-mixed/eval-report-4800.md) | The previous run ended compute-limited, so its roadmap's first item was more steps. What does doubling the horizon actually buy? | Same model, same data, same seed, `max_steps` 2400 → 4800. **1.6569 bits/byte**, a 1.48× margin over the best classical compressor. Doubling compute bought **3.7%**, and the last four evaluations bought a third of what the shorter run's did — still compute-limited, but deep into diminishing returns. No overfitting at 6.81 epochs: the train/val gap *narrowed* by two thirds. See the [evaluation report](2026-10-18-gutenberg-mixed/eval-report-4800.md) and [model card](2026-10-18-gutenberg-mixed/model-card.md) |
| 2026-10-19 | [`2026-10-18-gutenberg-mixed/`](2026-10-18-gutenberg-mixed/size-sweep.md) | Both earlier runs were chosen by eight-minute pilots whose extrapolation was later caught 1.8× optimistic. Does the model-size verdict survive at the real budget? | Three sizes at a matched 2,926 s wall clock, step counts set from measured throughput so each finishes its cosine schedule: 1.77M → **1.6569** test bpb, 3.74M → 1.6605, 5.64M → 1.6920. The pilots' 11% margin for the small model collapsed to 0.2% — a tie at single-seed precision, not a win — and 5.64M is clearly too big. See the [size sweep](2026-10-18-gutenberg-mixed/size-sweep.md) |
| 2026-10-19 | [`2026-10-18-gutenberg-mixed/`](2026-10-18-gutenberg-mixed/context-sweep.md) | Context length had been fixed at 256 since the first pilot and never varied — does prose reward more of it, or less? | Neither: **256 is an interior optimum**, bracketed on both sides at a matched budget with tokens-per-step held constant. T=128 → 1.6676 test bpb, T=256 → **1.6569**, T=512 → 1.6828, T=1024 → 1.8276. The two sides fail differently — above 256 the penalty survives token-matching and is intrinsic; below it, T=128 ties T=256 at matched tokens and loses only because halving the context made it 9.3% *slower*, the one place in this directory where a measured prediction came out with the wrong sign. See the [context sweep](2026-10-18-gutenberg-mixed/context-sweep.md) |
| 2026-10-19 | [`2026-10-18-gutenberg-mixed/`](2026-10-18-gutenberg-mixed/lr-bracket.md) | Every result so far was measured at a learning rate chosen on iteration 1's corpus, with a different tokenizer and a 2× larger model. Does it survive? | Yes, with a caveat worth more than the result. Three runs differing only in peak LR: 1.5e-3 → 1.6707 test bpb, 3e-3 → **1.6569**, 6e-3 → 1.6583. The response is sharply asymmetric — halving costs 0.83%, doubling costs 0.08% — so the inherited value sits on the lower edge of a plateau rather than at a peak. At 1.5e-3 the fixed `grad_clip: 1.0` binds on 54.8% of steps, making the low arm partly a clipping experiment. See the [learning-rate bracket](2026-10-18-gutenberg-mixed/lr-bracket.md) |

Seven pilots preceded the first run ([`pilots.md`](2026-10-17-tiny-shakespeare/pilots.md)):
the tokenizer choice was worth 19% in bits per byte, the learning rate 11%,
and turning dropout off bought 40% throughput on CPU — and cost a validation
minimum twelve epochs in.

Four preceded the second ([`pilots.md`](2026-10-18-gutenberg-mixed/pilots.md)).
They chose a model **half the size** the 6ND budget rule recommends — measured
curves beat the rule by 11.7% — and they carry a recorded prediction that the
later runs then scored twice: the log-linear extrapolation came in **1.2%
optimistic** at 3.5× the pilot's wall-clock and **4.5% optimistic** at 6×.
Short-window fits flatter the future, and the error grows with distance.

Four levers have now been swept at the full budget — steps, model size,
context length and learning rate — and none of them binds. The configuration
the eight-minute pilots picked is a local optimum that roughly six hours of
subsequent compute has not improved on by more than noise. The open item is
the corpus itself.

## How to read a run directory

- `plan.md` — the experiment plan, filled in before and during the run:
  goal, data card, tokenizer, model configuration, pilot matrix, run manifest,
  evaluation and release gate.
- `pilots.md` — the pilot matrix with measured numbers and the decision each
  pilot drove.
- `eval-report.md` / `eval-test.json` — held-out metrics against baselines.
- `samples.txt` — generations with the exact sampling settings recorded.
- `loss-curve.svg`, `pilot-curves.svg` — plotted from `metrics.jsonl`.
- `metrics/` — the raw `metrics.jsonl` from each run, which is the primary
  record; everything else here is derived from it.
- `model-card.md` — what the model is, what it is not, and what was measured.

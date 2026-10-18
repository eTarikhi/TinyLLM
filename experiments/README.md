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

| 2026-10-18 | [`2026-10-18-gutenberg-mixed/`](2026-10-18-gutenberg-mixed/plan.md) | Iteration 1 ran out of data, not compute. What changes when the corpus grows 17× and the model is sized by measurement rather than by the scaling rule? | 1.77M-parameter GPT on 27 Project Gutenberg books, **1.7201 bits/byte** on held-out text vs 2.1953 for a bigram counter and 2.4497 for primed `bzip2 -9` — a 1.42× margin over the best classical compressor, up from 1.06× in iteration 1. Validation was still falling at the step limit: compute-limited, not data-limited. See the [evaluation report](2026-10-18-gutenberg-mixed/eval-report.md) and [model card](2026-10-18-gutenberg-mixed/model-card.md) |

Seven pilots preceded the first run ([`pilots.md`](2026-10-17-tiny-shakespeare/pilots.md)):
the tokenizer choice was worth 19% in bits per byte, the learning rate 11%,
and turning dropout off bought 40% throughput on CPU — and cost a validation
minimum twelve epochs in.

Four preceded the second ([`pilots.md`](2026-10-18-gutenberg-mixed/pilots.md)).
They chose a model **half the size** the 6ND budget rule recommends — measured
curves beat the rule by 11.7% — and they carry a recorded prediction that the
main run then scored: the log-linear extrapolation to 3.5× the pilot's
wall-clock came in **1.2% optimistic**.

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

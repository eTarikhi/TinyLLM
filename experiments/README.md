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
| 2026-10-17 | [`2026-10-17-tiny-shakespeare/`](2026-10-17-tiny-shakespeare/plan.md) | Does the bundled implementation pass every gate, and what does a ~30-minute CPU pretraining run actually produce? | See [the evaluation report](2026-10-17-tiny-shakespeare/eval-report.md) and [model card](2026-10-17-tiny-shakespeare/model-card.md) |

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

# Project scaffold: layout, tests, and the artifacts a run must leave behind

`SKILL.md` names the deliverables; this file is the concrete shape they take on disk. [`scripts/tiny-gpt/`](../scripts/tiny-gpt/README.md) is this layout filled in and runnable, and the test filenames below are the ones it ships.

## Directory layout

Prefer a structure similar to this, adapting only where justified:

```text
my-llm/
  README.md
  pyproject.toml              # or pinned requirements.txt
  configs/
    tiny.yaml
    pilot.yaml
  src/
    tokenizer.py
    data.py
    model.py
    train.py
    generate.py
    evaluate.py
  tests/
    test_shapes.py            # logits [batch, seq, vocab], finite scalar loss
    test_causal_mask.py       # future tokens cannot change earlier logits
    test_overfit.py           # drives loss down on one tiny batch
    test_optim.py             # a step changes params, no non-finite values
    test_generation.py        # valid IDs only, stops at limit or EOS
    test_checkpoint.py        # save/load reproduces logits
    test_resume.py            # restores model, optim, sched, step, RNG, data position
  data/                       # normally ignored by version control
    README.md                 # provenance; do not commit restricted raw data
  artifacts/
    tokenizer/                # vocabulary/tokenizer files
    checkpoints/              # ignored or external storage
  experiments/
    README.md                 # run manifests and decisions
  model-card.md
```

Keep secrets out of source control. Add `.gitignore` entries for corpora, checkpoints, caches, environment files, logs, and API keys. The README should show exact environment setup, small smoke-test commands, training and evaluation commands, hardware assumptions, and a reproducibility recipe.

## The nine artifacts

Produce artifacts proportionate to the user's request; do not stop at advice when implementation is requested. Artifacts 6 and 8 have a fill-in structure in [`../templates/experiment-plan.md`](../templates/experiment-plan.md) — copy it into the user's project once per run (for example `experiments/<run-id>/plan.md`) and fill it in, rather than editing the template in place.

1. **Project brief:** goal, assumptions, hardware, data, constraints, and success metrics.
2. **Architecture note:** diagram or text flow, tensor shapes, parameter count, and why the selected architecture fits the goal.
3. **Data plan:** source/license manifest, cleaning and deduplication approach, split policy, tokenizer design, and storage estimate.
4. **Runnable starter implementation:** pinned dependencies, training, sampling, validation, and checkpoint resume. Start from [`../scripts/tiny-gpt/`](../scripts/tiny-gpt/README.md) rather than from a blank file, then cut what the project does not need.
5. **Tests:** model shapes, causal masking, finite loss, tiny-batch overfit, generation, and checkpoint round-trip — the files listed in the layout above. Test the *evaluation* code too, not only the model: a metric or probe that silently measures nothing still prints a number, and a zero reads like a pass. Assert that a check which could not run says so instead of reporting a value.
6. **Experiment plan:** pilot matrix changing one or a few variables at a time; token budget; stop/go gates; compute estimate and uncertainty.
7. **Evaluation report:** held-out loss, domain tests, baselines, sample generations, limitations, and precise interpretation.
8. **Reproducibility record:** code revision, configs, dependencies, data/tokenizer versions, seed, checkpoint ID, metrics, and environment details.
9. **Next-step roadmap:** only introduce distributed training, larger context, advanced kernels, SFT, or alignment after the previous gate passes.

When asked for code, implement the smallest complete end-to-end version that meets the goal, then add tests. Never imply that unexecuted code has been tested; state what was or was not run.

## A worked instance

[`../experiments/`](../experiments/README.md) holds filled-in examples of artifacts 1, 3, 6, 7, 8 and 9 from two real CPU-only runs of `scripts/tiny-gpt`: the plan with its success criteria and how each resolved, the pilot matrix, the evaluation report with baselines, the model card, and the roadmap that the next iteration picked up.

The two runs together show the loop working, which one run cannot. The first was data-limited — it overfit a 1.1 MB corpus at 11.9 epochs and its best checkpoint came at the halfway mark. Its roadmap said *get more data*. The second used a 17× larger corpus, ran 3.41 epochs, never turned up, and ended still improving: now compute-limited, and its own roadmap says *more steps before anything else*. Bits per byte went 2.3284 → 1.7201, and the margin over the best classical compressor went 1.06× → 1.42×. Neither number is interesting alone; the diagnosis that changed between them is the point.

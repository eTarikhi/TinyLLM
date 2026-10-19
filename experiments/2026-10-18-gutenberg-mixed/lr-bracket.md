# Learning-rate bracket at the main budget

Peak learning rate has been 3e-3 since pilot p5 of iteration 1. That pilot
chose it on a 1.1 MB Shakespeare corpus, with a character-level tokenizer, a
model twice this one's size, and a 600-step horizon. None of those four things
is still true. Every result measured since — the tokenizer, the model size,
the step count, the context length — was measured *conditional on* a learning
rate inherited from a setting that no longer exists, and held fixed by design
in both sweeps.

This is the control that should have been run first.

**3e-3 survives. But the response is strongly asymmetric: halving the learning
rate costs 0.83% and doubling it costs 0.08%, so the inherited value sits on
the lower edge of a plateau rather than at a peak.**

## Method

The narrowest design available. `lr-1p5e3.yaml` and `lr-6e3.yaml` are
identical to `main-gutenberg-4800.yaml` in every effective setting except the
peak learning rate — same 4,800-step horizon, same 240-step warmup, same
cosine decay to 10% of peak, same seed 1337, same 8,192 tokens per optimizer
step, same evaluation cadence of 120 steps. Diffing the three configs with
comments stripped produces exactly two differing lines, `run_id` and
`learning_rate`.

Because the horizons are identical, all three runs see the same 39,321,600
tokens and complete the same cosine shape. **This comparison is matched on
steps and tokens, not on wall clock**, which also makes it immune to the
13% host-speed difference between the session that ran the reference and the
session that ran the two arms.

A half-decade either side was chosen deliberately: a bracket narrow enough
that all three points stay trainable, wide enough that a genuine optimum
inside it would show up as a clear interior minimum.

## Results

| peak LR | steps | tokens | **val bpb** | **test bpb** | clipped steps | median grad norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.5e-3 | 4,800 | 39,321,600 | 1.6285 | 1.6707 | 2,630 | 1.009 |
| **3.0e-3** | 4,800 | 39,321,600 | **1.6158** | **1.6569** | 63 | 0.637 |
| 6.0e-3 | 4,800 | 39,321,600 | 1.6210 | 1.6583 | 23 | 0.416 |

Against the incumbent: 1.5e-3 is **+0.83%** test bpb,
and 6.0e-3 is **+0.08%** test bpb.
On validation the same ordering is clearer — +0.79% and +0.32% — because
validation is the quantity the runs were selected on. Curves in
`lr-bracket.svg`, zoomed on the final half where the three separate.

A log-quadratic through the three points puts the vertex at **3.5e-3** on
validation and 4.0e-3 on test. Three points determine a parabola exactly, so
this is a direction to look rather than a measurement — but it says the
inherited value is within a factor of 1.3 of the best in the bracket, and
that the better direction is *up*.

## The ranking is stable, which single-seed results usually are not

The three runs were evaluated at 40 common steps. The lead changes exactly
once, at step 240 — 5% of the way in — after which **3e-3 is in front at
every single one of the remaining 39 evaluations**. 6e-3 leads only the first
evaluation, at step 120, while the higher rate is still cashing in its faster
start.

That matters because 0.08% on test is well inside single-seed noise and would
normally be unreportable. The step-wise record turns it from one coin flip
into a consistently held ordering: 3e-3 and 6e-3 are close, but they are not
close in a way that suggests the measurement is noise.

## 1.5e-3 is partly a gradient-clipping experiment

The most striking number in the table is not a bits-per-byte figure. It is
that the low arm clipped on **2,630 of 4,800 steps — 54.8%** — against 63
steps (1.3%) at 3e-3 and 23 (0.5%) at 6e-3. Median pre-clip gradient norm runs
1.009 / 0.637 / 0.416 across the bracket, so **gradient norm falls as the
learning rate rises**, monotonically and by a factor of 2.4 across a 4× range.

The mechanism is ordinary: a larger step moves the model out of the steep
early region of the loss surface sooner, and it spends the rest of the run
somewhere flatter. The slow arm stays in steep territory for most of its
horizon.

The consequence is not ordinary, and it is a caveat on this experiment:
`grad_clip: 1.0` was tuned alongside 3e-3 and left fixed, so at 1.5e-3 the
clip is binding more than half the time. Its *effective* learning rate is
lower than its nominal one, by a margin nothing here measures. **The honest
reading of the low arm is "1.5e-3 with a clip calibrated for 3e-3 is 0.83%
worse", not "1.5e-3 is 0.83% worse".** The high arm carries no such
contamination — clipping is essentially inactive at 6e-3 — so the asymmetry
in the headline result is, if anything, understated.

## Generalization gap

The end-of-run gap between the training-split and validation evaluations
narrows monotonically as the learning rate rises:
+0.0663 at 1.5e-3, +0.0399 at 3e-3, +0.0376 at 6e-3. Consistent with the standard reading of a high
learning rate as an implicit regularizer, though at these margins it is a
hint and not evidence. No arm shows any sign of overfitting at 6.81 epochs.

## Limitations

- **One seed per arm**, as everywhere in this directory. The 3e-3/6e-3 difference (0.08% test) is below the noise floor; only the stability of the step-wise ordering argues it is real.
- **Three points cannot locate an optimum**, only bracket one. The useful claim is "the optimum is inside [1.5e-3, 6e-3] and not near the low end", not "the optimum is 3.5e-3".
- **Only peak LR moved.** Warmup length, the 10% cosine floor, β₂ = 0.99, weight decay 0.1 and `grad_clip` 1.0 were all held at values tuned with 3e-3. The clip interaction above shows that holding a co-tuned hyperparameter fixed is not a neutral act.
- **Batch size was not swept.** Learning rate and batch size trade off against each other; this measures one slice through a two-dimensional surface at 8,192 tokens per step.
- **No divergence was observed anywhere in the bracket**, so the upper edge of the stable band is untested. 6e-3 was the largest rate tried and it trained cleanly, with 23 clipped steps and no non-finite gradients.

## Consequence for the roadmap

The inherited recipe is vindicated, which retroactively strengthens the size
and context sweeps: both were run at a learning rate that is, as far as this
bracket can tell, within 1.3× of the best available. Their conclusions were
not artefacts of a stale hyperparameter.

It also closes the last cheap lever. Steps, parameters, context and learning
rate have all now been measured at this budget and none of them binds — the
eight-minute pilots found a local optimum that roughly six hours of
subsequent compute has failed to improve on by more than noise. What is left
is the corpus itself: 18.8 MB, already traversed 6.8 times by the released
run. The next real gain is more data, not a better setting.

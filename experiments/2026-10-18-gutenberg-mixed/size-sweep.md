# Size sweep at the main budget

The pilots chose 1.77 M parameters on eight-minute curves. That model has
since been given 6× that budget and ended deep into diminishing returns — the
last four evaluations bought 0.0098 bpb against the previous run's 0.0302 —
while its train/val gap kept *shrinking*. That combination is the signature of
a model too small for its compute, and compute-optimal size grows as √C, so 6×
the budget nominally argues for ~2.4× the parameters.

The eight-minute verdict had also already been caught overstating long-horizon
gains by 1.8×, so re-testing it by extrapolation was not an option. This sweep
measures it directly, at the real budget.

## Method

**Matched on wall clock, not on steps or tokens.** The budget is 2,926 s, the
elapsed time of the incumbent's 4,800-step run. That is the decision a person
actually faces: *given this long on this machine, what size should I train?*

**Every config must finish its cosine schedule inside the budget.** This is
the detail that makes or breaks the comparison. The 4,800-step run read 1.7161
at step 2,400 against the shorter run's 1.6815 at the identical step, purely
because its learning rate had not annealed yet — annealing is worth about
0.035 bpb, which is the same order as the size effect being measured. A config
that ran out of time mid-schedule would lose for the wrong reason.

So step counts come from measured throughput rather than guesswork.
`src/size_probe.py` times 20 optimizer steps per candidate, then scales by the
one number we have ground truth for — the incumbent's end-to-end rate, which
includes validation overhead that the probe does not see.

The probe validates itself on the config whose answer is already known, and
holds up on the two it had never been calibrated against:

| config | predicted | actual | error |
| --- | ---: | ---: | ---: |
| 3L/192d/3H | 13,437 tok/s → 4,799 steps | 13,437 tok/s → 4,800 steps | −0.0% |
| 4L/256d/4H | 7,200 tok/s → 2,571 steps | 7,101 tok/s | −1.4% |
| 5L/288d/6H | 4,937 tok/s → 1,763 steps | 4,980 tok/s | +0.9% |

All three runs ended on `max_steps`, none on the time guard.

**Warmup is 5% of each horizon** rather than a fixed step count. Holding the
*shape* of the schedule constant is what makes two different-length runs
comparable, and 5% is exactly what the incumbent used (240/4800).

**Everything else is held fixed:** vocab 2,048, block 256, batch 16 × accum 2,
lr 3e-3 cosine to 3e-4, AdamW β=(0.9, 0.99), wd 0.1, clip 1.0, dropout 0.0,
tied embeddings, seed 1337, same prepared data.

## Results

| config | params | steps | elapsed | tokens | epochs | **val bpb** | **test bpb** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **3L/192d/3H** | **1,770,816** | 4,800 | 2,926 s | 39,321,600 | 6.81 | **1.6158** | **1.6569** |
| 4L/256d/4H | 3,737,856 | 2,571 | 2,966 s | 21,061,632 | 3.65 | 1.6208 | 1.6605 |
| 5L/288d/6H | 5,643,360 | 1,763 | 2,900 s | 14,442,496 | 2.50 | 1.6561 | 1.6920 |

Relative to the incumbent: 4L/256d is **+0.22%** test bpb at 2.11× the
parameters; 5L/288d is **+2.11%** test bpb at 3.19× the parameters.

Curves for all three in `size-sweep.svg`.

## What this says

**The pilots' verdict survives, and its margin has almost completely
collapsed.**

| budget | margin of 1.77 M over 3.74 M |
| --- | ---: |
| 480 s (pilots, contention-corrected) | ~11.0% |
| 2,926 s (this sweep) | **0.31%** validation, 0.22% test |

A 0.2–0.3% difference on a single seed is **not a win; it is a tie this
experiment cannot resolve.** The honest statement is that at this budget the
two sizes are equivalent, where at the pilot budget the smaller one was
decisively better. The crossover is at or very near this budget, and the next
doubling of compute should favour the larger model — but a two-point
extrapolation of a difference that has shrunk into the noise is exactly the
kind of claim this project has already been burned by once, so it is recorded
as a prediction to test, not a result.

What *is* resolved: **5.64 M is too big for this budget**, by a margin (2.1%)
comfortably larger than single-seed noise. The optimum sits in the 1.8–3.7 M
band, not above it.

**The small model is winning while doing 2.7× more passes over the data.** It
sees 6.81 epochs against 4L/256d's 3.65 and 5L/288d's 2.50. Re-reading this
corpus is evidently still cheaper than adding capacity, which is consistent
with the earlier finding that 6.81 epochs produced no overfitting at all.

**Capacity shows up in the generalization gap, as it should:**

| config | train(eval) − val at the end | longest verbatim match | peak RSS |
| --- | ---: | ---: | ---: |
| 3L/192d/3H | +0.0399 | 24 chars | 1,166 MB |
| 4L/256d/4H | +0.0539 | 26 chars | 1,366 MB |
| 5L/288d/6H | +0.0972 | 27 chars | 1,579 MB |

The gap and the memorization probe both grow monotonically with parameter
count. None of the three is overfitting — every one of them was still
improving at its final evaluation — but the ordering is the expected one and
is a useful sanity check that the sweep is measuring capacity rather than
noise.

## Limitations

- **Single seed per point.** The 1.77 M vs 3.74 M difference is smaller than the variance I would expect from reseeding, which is precisely why it is reported as a tie rather than a ranking.
- **Learning rate was not retuned per size.** 3e-3 was chosen for the 1.77 M model. Larger models often prefer a lower peak, so the two bigger points may be slightly handicapped. Changing two variables at once would have confounded the result, but this means the sweep measures *size at a fixed recipe*, which is a weaker claim than *size, each tuned*.
- **Width and depth move together.** 4L/256d and 5L/288d each change both. This sweep cannot say whether depth or width is the binding dimension.
- **Wall-clock matching is hardware-specific.** On a machine with different arithmetic intensity — any GPU — the throughput ratios that set these step counts would differ, and so could the winner.
- **4L/256d received 1.4% more wall clock** than the incumbent (2,966 s vs 2,926 s) because the probe slightly under-predicted its throughput, and still finished behind. If anything this understates the incumbent's edge, which does not change the "tie" reading.

## Consequence for the roadmap

Two of the three obvious levers are now measured and closed off at this scale:
more steps buys little (0.0098 bpb over the final four evaluations), and more
parameters buys nothing yet (a tie at 2.11×, a loss at 3.19×). Neither is the
binding constraint.

That points the next experiment at the dimensions this project has held fixed
through every run so far — **context length, fixed at 256 tokens since the
first pilot, and the learning-rate/batch recipe, carried over unexamined from
iteration 1** — rather than at another step or size doubling.

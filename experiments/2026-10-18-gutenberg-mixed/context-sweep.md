# Context-length sweep at the main budget

Context length had been fixed at 256 tokens since the very first pilot and
never once varied, while the two levers that *were* swept — steps and model
size — have both now been measured and found not to be binding. More steps
buys 0.0098 bpb over the final four evaluations; 2.11× the parameters buys a
tie. So this was the remaining unexamined axis, and the expectation going in
was that prose would reward more context.

**It does not. Longer context is monotonically worse here, and the penalty is
large.**

## Method

Same discipline as the size sweep. Matched wall clock (2,926 s), step counts
from measured throughput so every config completes its cosine schedule,
warmup at 5% of each horizon.

**Tokens per optimizer step held at 8,192** by scaling batch inversely with
context — 32×128, 16×256, 8×512, 4×1024, all with grad-accum 2. That keeps
the optimization recipe identical and leaves context as the only deliberate
variable.

One thing that cannot be held fixed: learned positional embeddings mean
parameter count moves with context, from 1,770,816 at T=256 to 1,918,272 at
T=1024 (+8.3%). That is inherent to the axis rather than a removable confound,
and it works *in favour* of the longer-context models, which have more
parameters.

### The FLOP estimate overstates the price of context on this machine

The attention-share rule of thumb `T/(6·d_model)` implies T=1024 should cost
roughly twice T=256. Measured pure training throughput fell only 18%:

| context | params | pure train tok/s | vs T=256 |
| --- | ---: | ---: | ---: |
| T=128 | 1,746,240 | 14,730 | +5.8% |
| T=256 | 1,770,816 | 13,925 | — |
| T=512 | 1,819,968 | 12,149 | −12.8% |
| T=1024 | 1,918,272 | 11,437 | −17.9% |

Attention at this width is memory-bound rather than FLOP-bound, so arithmetic
badly overestimates what context costs in wall clock. Worth knowing before
budgeting a run off a FLOP count — but, as the results show, cheap is not the
same as free.

## Results

| context | params | steps | elapsed | tokens | **val bpb** | **test bpb** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **T=256** | **1,770,816** | 4,800 | 2,926 s | 39,321,600 | **1.6158** | **1.6569** |
| T=512 | 1,819,968 | 4,187 | 2,883 s | 34,299,904 | 1.6442 | 1.6828 |
| T=1024 | 1,918,272 | 3,941 | 3,167 s | 32,284,672 | 1.7812 | 1.8276 |

T=512 is **+1.56%** test bpb; T=1024 is **+10.30%**. Curves in
`context-sweep.svg`.

## Is it just the lost tokens?

A longer context costs throughput, so these runs saw 39.3 M, 34.3 M and 32.3 M
tokens respectively. Some of the deficit is simply less text. Comparing every
run at the smallest run's token count separates the two effects:

| context | val bpb at 32,284,672 tokens | vs T=256 |
| --- | ---: | ---: |
| T=256 | 1.6325 | — |
| T=512 | 1.6484 | +0.97% |
| T=1024 | 1.7812 | +9.11% |

For T=512, roughly half the penalty is lost tokens (1.56% → 0.97%). For
T=1024, almost none of it is (10.30% → 9.11%) — long context is intrinsically
worse here, not merely slower.

**And this comparison is biased in favour of the longer contexts.** At
32.28 M tokens T=1024 is at the end of its schedule and fully annealed, while
T=256 is only 82% through its cosine and T=512 is 94% through. Annealing is
worth roughly 0.035 bpb. The longer-context runs are being compared at their
best against the shorter one mid-decay, and they still lose.

## Why a bigger context loses at this scale

Three effects point the same way, and this experiment cannot fully separate
them:

1. **Fewer tokens per unit time** — worth about half of T=512's deficit and almost none of T=1024's.
2. **More positional embeddings, each seen less often.** T=1024 has 4× the position slots of T=256 and trained on 18% less text, so each position is updated far less.
3. **A 1.77 M-parameter, 3-layer model has little capacity to exploit long-range structure.** Capacity spent representing distant dependencies is capacity not spent on local ones, and local structure is where nearly all the predictable signal in bits-per-byte terms lives.

The generalization gap supports the third reading: +0.0399 at T=256, +0.0777
at T=512, +0.1567 at T=1024. Longer context behaves like added capacity — it
widens the gap — without delivering the accompanying improvement.

## An anomaly worth recording

T=1024 overran its budget: 3,167 s against 2,926 s (+8.2%), because its actual
throughput came in 7.6% under the probe's prediction. The probe was accurate
to 1.5% on the other two.

The cause is visible in the per-quarter throughput:

| run | Q1 | Q2 | Q3 | Q4 | drift |
| --- | ---: | ---: | ---: | ---: | ---: |
| T=256 | 14,379 | 14,260 | 14,449 | 14,206 | −1.2% |
| T=512 | 12,272 | 12,295 | 12,297 | 12,536 | +2.1% |
| T=1024 | 11,421 | 11,552 | 11,417 | 9,161 | **−19.8%** |

T=1024 ran at its predicted rate for three quarters and then lost a fifth of
its throughput. The other two runs were stable to within 2% across the same
machine, so this is specific to the long-context run rather than general
machine noise — most likely cache or allocator behaviour on the larger
attention buffers. **The probe measures steady-state throughput over 20 steps
at the start of training and cannot see drift like this**, which is a real
limitation of the method, not just of this run.

The overrun gave T=1024 8.2% more wall clock than its competitors, and it
still finished 10% behind. The conclusion is unaffected.

## Limitations

- **Single seed per point**, as everywhere in this directory. Unlike the size sweep's 0.2% tie, these margins (1.6% and 10.3%) are far outside plausible seed noise, so the ranking is safe even if the exact figures are not.
- **T=128 was not run.** The sweep brackets the incumbent only from above. Since both longer settings lose, 256 may itself be past the optimum, and a shorter context might be better still — untested, and now the cheapest open question in this directory.
- **Learning rate was not retuned per context**, the same fixed-recipe caveat as the size sweep.
- **The result is scale-specific.** It says long context does not pay *for a 1.77 M-parameter model on 19th-century prose at a 49-minute budget*. It says nothing about context length at scales where models can actually use it, which is where the industry's interest in long context comes from.
- **T=1024's test evaluation covers 322,534 tokens / 930,002 bytes** against 323,046 / 931,467 for the others, because non-overlapping 1024-token windows truncate more at the tail — a 0.16% difference in the denominator, far too small to affect the ranking.

## Consequence for the roadmap

Three levers have now been measured at this budget and none is binding: more
steps buys almost nothing, more parameters buys a tie, more context actively
hurts. The configuration the eight-minute pilots picked is at a local optimum
that an hour of compute per experiment cannot escape.

That makes the honest next move **down, not up**: T=128 is the one direction
the sweep has not probed, it is cheaper than everything tried so far, and the
monotone trend predicts it should win. The other open item is the
learning-rate and batch recipe, carried unexamined from iteration 1 through
every run in this directory and held fixed by design in both sweeps — which
means every size and context conclusion here is conditional on a recipe that
was tuned for neither.

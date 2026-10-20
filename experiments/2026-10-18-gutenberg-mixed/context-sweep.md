# Context-length sweep at the main budget

Context length had been fixed at 256 tokens since the very first pilot and
never once varied, while the two levers that *were* swept — steps and model
size — have both now been measured and found not to be binding. More steps
buys 0.0098 bpb over the final four evaluations; 2.11× the parameters buys a
tie. So this was the remaining unexamined axis, and the expectation going in
was that prose would reward more context.

**It does not. 256 is an interior optimum: every other setting tested, longer
or shorter, is worse.** The two sides lose for different reasons, which is the
more useful half of the result.

## Method

Same discipline as the size sweep. Matched wall clock (2,926 s), step counts
from measured throughput so every config completes its cosine schedule,
warmup at 5% of each horizon.

**Tokens per optimizer step held at 8,192** by scaling batch inversely with
context — 32×128, 16×256, 8×512, 4×1024, all with grad-accum 2. That keeps
the optimization recipe identical and leaves context as the only deliberate
variable.

One thing that cannot be held fixed: learned positional embeddings mean
parameter count moves with context, from 1,746,240 at T=128 to 1,918,272 at
T=1024 (+9.9%). That is inherent to the axis rather than a removable confound,
and it works *in favour* of the longer-context models, which have more
parameters.

### A machine-speed caveat, and the control that handles it

T=256, T=512 and T=1024 ran in one session. T=128 ran in a later one, on a
host that turned out to be measurably slower. Two runs in that later session
(the [learning-rate bracket](lr-bracket.md)) use the T=256 shape, horizon and
evaluation cadence exactly, so they measure the host rather than the
configuration: they averaged 11,683 tok/s end-to-end against the reference
run's 13,437, **13.1% slower** (12.8% on evaluation-free windows).

Wall clock is therefore not comparable across the two sessions, and every
cross-session statement below is made on tokens, or on throughput ratios
against a T=256 measured on the same host.

### Measured throughput, and a prediction that was simply wrong

The attention-share rule of thumb `T/(6·d_model)` implies T=1024 should cost
roughly twice T=256. It costs a fifth more. But the more interesting error is
at the other end — the throughput probe predicted a short context would be
*cheaper*, and it is not:

| context | params | measured tok/s | vs same-host T=256 | probe predicted |
| --- | ---: | ---: | ---: | ---: |
| T=128 | 1,746,240 | 11,314 | **−9.3%** | +5.8% |
| T=256 | 1,770,816 | 14,307 | — | — |
| T=512 | 1,819,968 | 12,386 | −13.4% | −12.8% |
| T=1024 | 1,918,272 | 11,389 | −20.4% | −17.9% |

(Throughput measured over training windows containing no evaluation, so the
differing evaluation cadences do not contaminate it. T=128 is compared against
the same-host T=256 figure of 12,477 tok/s, the other rows against 14,307.)

Halving the context did not make the model faster; it made it **9.3% slower**.
At 3 layers and 192 dimensions attention is not the bottleneck, so halving it
saves almost nothing, while doubling the batch to 32 sequences adds per-sequence
overhead that costs more than the saving. Attention here is memory-bound and
the model as a whole is overhead-bound — arithmetic cannot predict either, and
on this axis even a short empirical probe got the *sign* wrong.

## Results

| context | params | steps | elapsed | tokens | **val bpb** | **test bpb** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| T=128 | 1,746,240 | 4,100 | 3,061 s | 33,587,200 | 1.6310 | 1.6676 |
| **T=256** | **1,770,816** | 4,800 | 2,926 s | 39,321,600 | **1.6158** | **1.6569** |
| T=512 | 1,819,968 | 4,187 | 2,883 s | 34,299,904 | 1.6442 | 1.6828 |
| T=1024 | 1,918,272 | 3,941 | 3,167 s | 32,284,672 | 1.7812 | 1.8276 |

T=128 is **+0.64%** test bpb; T=512 is **+1.56%**; T=1024 is **+10.30%**.
Curves in `context-sweep.svg`.

## Is it just the lost tokens?

Context changes throughput, so these runs saw 33.6 M, 39.3 M, 34.3 M and
32.3 M tokens respectively. Comparing every run at the smallest run's token
count separates "saw less text" from "models text worse":

| context | val bpb at 32,284,672 tokens | vs T=256 | fraction of its own cosine |
| --- | ---: | ---: | ---: |
| T=128 | 1.6331 | +0.03% | 96.1% |
| T=256 | 1.6325 | — | 82.1% |
| T=512 | 1.6484 | +0.97% | 94.1% |
| T=1024 | 1.7812 | +9.11% | 100.0% |

This splits the sweep cleanly in two:

- **Below 256 the penalty is entirely throughput.** T=128 loses 0.64% on test
  at matched wall clock, but at matched tokens it is level with T=256 to
  +0.03% — far inside single-seed noise. A 128-token window models this prose
  exactly as well as a 256-token one; it just gets through less of it per
  second. Nothing is lost by *looking* at less context, which is itself worth
  knowing.
- **Above 256 the penalty is mostly intrinsic.** T=512 keeps 0.97 of its
  1.56%, and T=1024 keeps 9.11 of its 10.30%.

**And the matched-token comparison is biased against T=256.** At 32.28 M
tokens the other three runs are 94–100% through their cosine schedules while
T=256 is only 82% through and still mid-anneal, worth roughly 0.035 bpb. Every
competitor is being shown at its best against T=256 mid-decay — and the best
any of them manages is a tie.

## Why a bigger context loses at this scale

Three effects point the same way, and this experiment cannot fully separate
them:

1. **Fewer tokens per unit time** — worth about half of T=512's deficit and almost none of T=1024's.
2. **More positional embeddings, each seen less often.** T=1024 has 4× the position slots of T=256 and trained on 18% less text, so each position is updated far less.
3. **A 1.77 M-parameter, 3-layer model has little capacity to exploit long-range structure.** Capacity spent representing distant dependencies is capacity not spent on local ones, and local structure is where nearly all the predictable signal in bits-per-byte terms lives.

The generalization gap supports the third reading, and it now orders perfectly
with context: +0.0323 at T=128, +0.0399 at T=256,
+0.0777 at T=512, +0.1567 at T=1024. Longer context behaves like added
capacity — it widens the gap — without delivering the accompanying
improvement.

## An anomaly worth recording

T=1024 overran its budget: 3,167 s against 2,926 s (+8.2%), because its actual
throughput came in 7.6% under the probe's prediction. The cause is visible in
the per-quarter throughput (tokens divided by elapsed time, so evaluation and
checkpoint pauses are included — this is the rate that determines whether a
run finishes inside its budget):

| run | Q1 | Q2 | Q3 | Q4 | drift |
| --- | ---: | ---: | ---: | ---: | ---: |
| T=128 | 11,304 | 10,606 | 10,754 | 11,417 | +1.0% |
| T=256 | 13,574 | 13,377 | 13,504 | 13,300 | −2.0% |
| T=512 | 11,974 | 11,914 | 11,942 | 12,169 | +1.6% |
| T=1024 | 11,135 | 11,079 | 10,871 | 8,448 | **−24.1%** |

T=1024 ran at its predicted rate for three quarters and then lost a quarter of
its throughput. The other three runs were stable to within 2% across the same
machines, so this is specific to the long-context run rather than general
noise — most likely cache or allocator behaviour on the larger attention
buffers. **The probe measures steady-state throughput over 20 steps at the
start of training and cannot see drift like this**, which is a real limitation
of the method, not just of this run.

The overrun gave T=1024 8.2% more wall clock than its competitors, and it
still finished 10% behind. The conclusion is unaffected.

### The probe's error has no stable sign

With four points measured, the probe's record is:

| context | host-corrected prediction | measured | error |
| --- | ---: | ---: | ---: |
| T=128 | 12,396 | 10,971 | **−11.5%** |
| T=256 | 13,437 | 13,437 | −0.0% |
| T=512 | 11,723 | 11,898 | +1.5% |
| T=1024 | 11,036 | 10,193 | −7.6% |

It is exact at its own reference point and wrong by up to 11.5% away from it,
in both directions. T=128's budget had to be re-derived mid-flight: the probe
sized it at 5,076 steps, a 300-step trial of the real configuration measured
19% less throughput, and the run was restarted at 4,100 steps so its cosine
would finish inside the time guard. **A short throughput probe is good enough
to rank configurations and not good enough to set a step budget.** Either
calibrate on a few hundred real steps, or keep a wall-clock guard and accept a
truncated schedule.

## Limitations

- **Single seed per point**, as everywhere in this directory. T=512's and T=1024's margins (1.6% and 10.3%) are far outside plausible seed noise. T=128's 0.64% is not: treat "T=128 is worse at matched wall clock" as solid only because its mechanism — a measured 9.3% throughput deficit — is independently confirmed.
- **Learning rate was not retuned per context.** All four runs use 3e-3. The [learning-rate bracket](lr-bracket.md) shows that choice is near-optimal *at T=256*; whether the optimum moves with context is untested.
- **Wall clock is matched within a session, not across sessions.** T=128 ran 3,061 s against the others' 2,926 s on a host 13% slower, so it got more seconds and fewer tokens. Both handicaps are quantified above and the matched-token comparison removes them.
- **The result is scale-specific.** It says context beyond 256 does not pay *for a 1.77 M-parameter model on 19th-century prose at a 49-minute budget*. It says nothing about context length at scales where models can actually use it, which is where the industry's interest in long context comes from.
- **Test evaluations cover slightly different spans**: 323,174 tokens / 931,863 bytes at T=128, 323,046 / 931,467 at T=256 and T=512, 322,534 / 930,002 at T=1024, because non-overlapping windows truncate more of the tail as they get longer. A 0.2% spread in the denominator, far too small to affect the ranking.

## Consequence for the roadmap

Context is now bracketed on both sides and the incumbent wins. Combined with
the two earlier sweeps, four levers have been measured at this budget and none
is binding: more steps buys almost nothing, more parameters buys a tie, more
context hurts, less context is a throughput loss for no modelling gain.

The configuration the eight-minute pilots picked is at a local optimum that an
hour of compute per experiment cannot escape. The remaining open item in this
directory is **data** — the corpus is 18.8 MB and the best run has already
made 6.8 passes over it — and that is a change of kind, not of degree.

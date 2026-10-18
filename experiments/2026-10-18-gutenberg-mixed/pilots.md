# Pilot matrix — 2026-10-18, `gutenberg-mixed-27`

Four pilots, ~8 minutes each, on 2 vCPU with no GPU. They answer the two
questions [iteration 1](../2026-10-17-tiny-shakespeare/pilots.md) could not ask
while it was starved of data: how large should the **vocabulary** be, and how
large should the **model** be, when the budget is wall-clock rather than tokens.

Everything else — byte-level BPE over a character vocabulary, lr 3e-3, dropout
0.0, tied embeddings, β₂ 0.99 — is carried over from the iteration-1 matrix,
which measured each of them on this same hardware. Re-measuring them would have
cost 35 minutes to re-derive known answers.

## Results

| Pilot | Params | MFLOP/token | Steps | Tokens | Wall-clock | tok/s | Best val bpb |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| g1-vocab1024 | 3,475,712 | 24.00 | 400 | 3,276,800 | 490 s | 6,693 | 2.1891 |
| g2-vocab2048 | 3,737,856 | 25.57 | 400 | 3,276,800 | 616 s | 5,316 | **2.1398** |
| **g3-small** | **1,770,816** | **12.39** | **800** | **6,553,600** | **587 s** | **11,173** | **1.8993** |
| g4-wide | 11,506,560 | 76.12 | 130 | 1,064,960 | 482 s | 2,210 | 2.5814 |

Validation is a fixed 65,536-token slice of the val split, evaluated every 50
steps — the same slice every time, so the traces are directly comparable.
`tok/s` is the run mean including evaluation overhead.

### Validation bits/byte against wall-clock

| Elapsed | g1-vocab1024 | g2-vocab2048 | g3-small | g4-wide |
| ---: | ---: | ---: | ---: | ---: |
| ~65 s | 2.7963 | 2.6965 | 2.5478 | — |
| ~190 s | 2.4034 | 2.4475 | 2.1871 | 2.9430 |
| ~310 s | 2.3041 | 2.3213 | 2.0494 | 2.7148 |
| ~480 s | 2.1933 | 2.1970 | **1.9395** | 2.5818 |

Interpolated onto a common 480-second budget, the ordering is unambiguous:
**1.9395 (1.8M params) < 2.1933 (3.5M) ≈ 2.1970 (3.7M) < 2.5818 (11.5M)**.

## Question 1 — vocabulary size

Vocabulary 2048 produces 2.946 bytes/token against 1024's 2.546, so each
optimizer step covers 16% more text. It also makes the output projection twice
as wide, which on a CPU costs throughput.

- **At a matched token budget** (3,276,800 tokens, step 400 for both):
  2.1398 against 2.1891 — **vocab 2048 is 2.3% better**.
- **At matched observed wall-clock** (480 s): 2.1970 against 2.1933 — a tie,
  and nominally a loss for 2048.

Those two statements disagree, so the throughput number deserves scrutiny
rather than a conclusion. **It was contaminated, and by me:** the g2 window
overlapped a unit-test run and two corpus-preparation jobs on the same two
cores. Its per-interval rate swings between 4,345 and 6,607 tok/s, while its
quiet final stretch holds 6,199–6,283. g1's quiet rate is 7,584–7,639. The
honest throughput ratio is therefore ≈0.83, not the 0.79 the run means imply.

Correcting g2 to a clean 6,300 tok/s puts it at ~3.02M tokens in 480 s, i.e.
step ~369, interpolating to **~2.153 bpb** — ahead of g1's 2.1933 by 1.8%.

**Decision: vocabulary 2048**, consistent on both the clean time-matched and
the contention-free token-matched comparison. The contamination is recorded
rather than hidden because it is the kind of detail that silently inverts a
conclusion, and the correction is an estimate, not a measurement.

## Question 2 — model capacity

The compute budget argues the model should be *smaller*, not larger. At
~2.9e14 FLOPs for the planned run, `C = 6ND` with Chinchilla's `D = 20N` gives
`N ≈ 1.5M` parameters — less than half of iteration 1's 3.3M. Since that ratio
was derived at a scale five orders of magnitude away, it was tested in both
directions rather than believed.

Both directions agreed with it:

- **3L/192d (1.77M params)** reached 1.9395 bpb at 480 s — **11.7% better**
  than the 4L/256d baseline, by running 1.7× faster and therefore covering
  twice the text in the same time.
- **6L/384d (11.5M params)** reached only 2.5818 — **24.9% worse**. At 2,210
  tok/s it managed 130 optimizer steps in eight minutes, and a model that has
  taken 130 steps has barely left its initialization, whatever its capacity.

This is the clearest result in the matrix, and it is large enough that CPU
contention cannot explain it: the effect of contention on g2 was worth ~2% bpb,
while the capacity effect is worth 12–25%.

**Decision: 3 layers, `d_model` 192, 3 heads — 1,770,816 parameters.**

### Projecting to the main run's budget

The pilots cost eight minutes; the main run gets thirty. Compute-optimal model
size grows with budget, so the pilot winner is not automatically the right
choice at 3.7× the compute. Fitting `bpb = a + b·ln(t)` to the last 60% of each
pilot's trace and extrapolating to 1800 s:

| Pilot | slope `b` | fitted @480 s | projected @1800 s |
| --- | ---: | ---: | ---: |
| g1-vocab1024 | −0.2539 | 2.1918 | 1.8562 |
| g2-vocab2048 | −0.2322 | 2.1955 | 1.8887 |
| g3-small | −0.2195 | 1.9398 | **1.6497** |
| g4-wide | −0.1540 | 2.5757 | 2.3722 |

The slopes are close enough that no crossover is projected inside this horizon,
so the pilot winner stays the winner at 1800 s. **This is an extrapolation of a
single eight-minute window by a factor of 3.7, not a law** — the main run tests
it, and the projection is recorded here so it can be scored afterwards rather
than quietly forgotten.

### Scoring it, after the fact

The main run stopped on its step budget at 1,702 s rather than running to
1,800 s, so the comparison is made at matched wall-clock using the same fitted
line (`a = 3.2949`, `b = −0.2195`):

| | val bpb |
| --- | ---: |
| Projected for g3-small @ 1,702 s | 1.6620 |
| **Actual, main run @ 1,702 s** | **1.6815** |
| Error | **+0.0195 (+1.2%)** |

The extrapolation was optimistic by about one percent across a 3.5× stretch in
wall-clock. Close enough to be useful for planning a run, and biased in the
direction worth knowing about: **log-linear fits to a short window flatter the
future**, because the early steep part of the curve leaks into the fit no
matter how carefully the tail is selected.

Two things make this agreement weaker than it looks, and both should temper
any reuse of the method:

- The main run was not g3-small continued. It shared the architecture but had
  its own cosine schedule stretched over 2,400 steps, so its learning rate was
  still at 3e-4 when the pilot's had already annealed. A matched-schedule
  continuation would have scored differently.
- One point is not a validation of the technique. It is one draw, on one
  architecture, on one corpus.

## What this matrix does not establish

- **Context length.** All four pilots use 256 tokens. A longer context costs
  attention FLOPs quadratically but may pay for itself on prose; untested.
- **Whether even smaller is better.** The capacity curve is monotone over the
  range tested — 1.77M is the smallest model measured, not a located optimum.
- **Transfer to a GPU.** Every conclusion here is mediated by CPU throughput.
  On hardware where the output projection is cheap, the vocabulary trade-off
  changes, and a larger model's lower throughput matters far less.

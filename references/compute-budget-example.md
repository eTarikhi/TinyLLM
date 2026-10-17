# Worked example: from config to a defensible compute budget

All arithmetic below was computed and checked on 2026-10-17; reproduce it with the snippets shown. The purpose is not these specific numbers but the order of operations: **exact parameter count → token budget → `6ND` as a sanity check on total work → a measured pilot as the only trustworthy source of wall-clock time.**

## Step 1 — Count parameters exactly, and record whether embeddings are tied

Config: `vocab=8192`, `d_model=512`, `layers=6`, `heads=8` (head dim `512/8 = 64`), `block_size=512`.

| Component | Formula | Parameters |
| --- | --- | --- |
| Token embedding | `vocab × d` | 4,194,304 |
| Position embedding | `block × d` | 262,144 |
| Attention, per layer | `4 × d²` (q, k, v, out) | 1,048,576 |
| MLP, per layer | `8 × d²` (4× expansion, two matrices) | 2,097,152 |
| All 6 layers | `6 × (4d² + 8d²)` | 18,874,368 |
| **Total, tied embeddings** | | **23,330,816 ≈ 23.33M** |
| **Total, untied embeddings** | `+ vocab × d` | **27,525,120 ≈ 27.53M** |

Tying the input and output embeddings removes **15.2%** of the parameters at this scale. That is why `templates/experiment-plan.md` asks for the tied/untied choice explicitly: a recorded parameter count is ambiguous without it, and every downstream FLOP and VRAM estimate inherits the ambiguity. The gap shrinks as the model grows — embeddings dominate small models and become negligible in large ones.

Excluded here: the normalization parameters — 13 LayerNorms (two per block plus a final one), each carrying a weight *and* a bias of width `d`, so `13 × 2 × 512 = 13,312` parameters, or **0.057%** of the total. Dropping the norm biases, as nanoGPT does, halves that to `6,656`. Below 0.1% is fine to omit from a planning estimate, but count them when you report a final parameter number, and say whether norms and attention carry biases.

## Step 2 — Pick the token budget before pricing anything

Take `D = 2,000,000,000` (2B) tokens. Choose this from the corpus you actually have rights to and the loss you are trying to reach, not from a ratio you read somewhere. Record the reasoning in the experiment plan.

## Step 3 — `6ND` estimates the work, not the time

```
6 × 23,330,816 × 2e9 = 2.7997e17 FLOPs   (tied)
6 × 27,525,120 × 2e9 = 3.3030e17 FLOPs   (untied)
```

`6ND` covers the parameter matmuls. It omits the attention score and value matmuls, which have no parameters and therefore do not appear in `N`. Their cost per token is `12 · L · T · d_model`: per layer and per token, `QKᵀ` and `A·V` are `2·T·d` FLOPs each in the forward pass, and the backward pass costs about twice the forward, so `(2+2) × 3 × L · T · d`. The `d_model` factor is the part that is easy to drop — **this table previously read `12·L·T·D`, which understated the attention term by a factor of `d_model` (512×) and produced the comforting, wrong conclusion that attention is negligible at short context.**

With `N`, `L`, `d` and `D` held at the values above, so that only the context length varies:

| Context `T` | Attention term `12·L·T·d·D` | Share of `6ND` |
| --- | --- | --- |
| 512 | 3.77e16 | 13.5% |
| 2,048 | 1.51e17 | 53.9% |
| 8,192 | 6.04e17 | 215.7% |
| 32,768 | 2.42e18 | 862.9% |

The rule of thumb behind the table: relative to the body FLOPs the attention term is `T / (6 · d_model)`, so it is small only while the context is short *compared with the model's width*. At `T = d_model` it is already about one sixth of the body. Two refinements, in opposite directions: a causal kernel computes roughly half of the score matrix, so a measured run lands nearer half this figure, while a long-context model would also carry a larger position embedding and a different `N`.

So `6ND` is a usable planning figure only when `T ≪ 6·d_model`, and even in that regime it is an underestimate (here by ~13%), not a safe upper bound. Quote it with the attention term added, or quote it as a lower bound and say so.

## Step 4 — Only a measured pilot gives wall-clock time

Suppose the pilot on the target device sustains **12,000 tokens/s** after warm-up:

```
2e9 / 12,000 = 166,667 s = 46.3 h = 1.93 days
```

At 25,000 tokens/s the same budget is 22.2 h. **Throughput, not FLOPs, is what your schedule depends on** — and throughput is a property of your hardware, kernel, precision, and data loader, which is why it must be measured rather than assumed.

## Step 5 — Back out MFU and read what it tells you

Use total FLOPs, not just `6ND`, or the achieved figure inherits the omission from step 3:

```
total  = (6N + 12·L·T·d) · D
       = (139,984,896 + 18,874,368) × 2e9 = 3.1772e17 FLOPs
achieved = 3.1772e17 / 166,667 = 1.91 TFLOP/s
MFU = 1.91 / 200 = 0.95%      (against a 200 TFLOP/s peak)
    = 1.91 / 312 = 0.61%      (against a 312 TFLOP/s peak)
```

An MFU under 1% is not a bug to chase. A 23M-parameter model is far too small to saturate a modern accelerator: you are bound by launch overhead, memory movement, and the data pipeline, not by matmul throughput. Two consequences worth stating plainly to a user:

- Do not buy a bigger GPU to make a tiny model train faster. It will not help much.
- Do not quote an MFU from a large-model writeup as the expected efficiency of a small educational run. The number does not transfer.

## Step 6 — Cost, with the rate treated as unknown until checked

Wall clock of 46.3 h across placeholder hourly rates:

| Rate | Compute cost |
| --- | --- |
| $0.35/h | $16.20 |
| $1.00/h | $46.30 |
| $2.00/h | $92.59 |
| $4.00/h | $185.19 |

These rates are **illustrative placeholders, not quotes.** Cloud prices change constantly and this file cannot stay current; price the specific instance type from the provider's published rate at planning time. Then add the parts that a naive estimate omits:

- **Contingency.** +25% on 46.3 h is 57.9 h.
- **Failed runs.** Three aborted full runs would burn roughly 139 h. This is the single most underestimated line item, and it is why the gates require a learning proof and a pilot before the main run.
- **Not-compute costs.** Storage for raw text, cleaned text, token IDs and checkpoints; egress; CPU hours for data processing; evaluation runs; idle time while a job waits; spot-interruption risk.

## Step 7 — Record effective tokens per update, because step counts are not comparable

The same 2B tokens is a different number of optimizer updates depending on batching:

| Micro-batch | Accumulation | Tokens/update | Updates for 2B tokens |
| --- | --- | --- | --- |
| 8 | 1 | 4,096 | 488,281 |
| 8 | 8 | 32,768 | 61,035 |
| 32 | 4 | 65,536 | 30,518 |

A run at 488,281 updates and a run at 30,518 updates are not comparable by step count, learning-rate schedule, or checkpoint index. Compare them at a fixed **token** budget. This is also why the checkpoint must store the data position, not just the step number: resuming at the wrong token offset silently corrupts the comparison.

## Reproduce

Every number above, including the attention table that was wrong until it was checked:

```bash
python3 - <<'PY'
V,d,L,T,D = 8192,512,6,512,2_000_000_000
tied = V*d + T*d + L*(4*d*d + 8*d*d)
print(f"tied   {tied:,}"); print(f"untied {tied+V*d:,}")
print(f"6ND    {6*tied*D:.4e}")
for Tc in (512, 2048, 8192, 32768):                    # attention term, step 3
    att = 12*L*Tc*d*D
    print(f"  T={Tc:>6}: {att:.2e} FLOPs, {att/(6*tied*D)*100:6.1f}% of 6ND")
for tps in (12_000, 25_000):                           # wall clock + MFU, steps 4-5
    s = D/tps
    total = (6*tied + 12*L*T*d)*D
    print(f"{tps:,} tok/s -> {s/3600:.1f} h, {total/s/1e12:.2f} TFLOP/s, MFU {total/s/200e12*100:.2f}%")
PY
```

`.github/scripts/check_docs_math.py` runs this arithmetic against the tables above on every push, so a future edit cannot quietly reintroduce a wrong formula.

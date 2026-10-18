# LLM pretraining experiment plan

Filled-in copy of [`templates/experiment-plan.md`](../../templates/experiment-plan.md)
for the run family `2026-10-17-tiny-shakespeare`, executed with
[`scripts/tiny-gpt`](../../scripts/tiny-gpt/README.md). Sections were written in
order: goal through model configuration before any training, pilots during the
pilot matrix, run manifest and evaluation after the main run.

## Goal and constraints

- **Goal / use case:** demonstrate the whole Track 1 pipeline end to end on
  hardware a reader is likely to have, and produce the evidence trail the skill
  demands. The artefact is the *process and its measurements*; the model itself
  has no downstream use.
- **Base model or instruction-following model:** base model only. No SFT, no
  chat template, no preference tuning.
- **Target languages and domains:** English, specifically early modern English
  dramatic verse and dialogue. Nothing else is claimed.
- **Hardware and accelerator memory:** 2 vCPU Intel Xeon @ 2.60 GHz, 3 GB RAM,
  **no GPU**. FP32 only; no AMP, no gradient scaler, no `torch.compile`.
- **Maximum cloud/local budget:** $0. Existing sandbox, no rented hardware.
- **Runtime ceiling:** ~30 minutes of wall clock for the main run, ~30 minutes
  for the whole pilot matrix.
- **Success metrics and stopping criteria:**
  1. All correctness tests pass (Gate 1).
  2. Held-out **bits per byte** beats uniform, unigram, add-k bigram and the
     same architecture at random init (Gate 2). A model that cannot beat a
     bigram counter has not learned anything worth reporting.
  3. Validation loss still falling, or a documented decision to stop, at the
     runtime ceiling.
  4. Every number in the report reproducible from `metrics.jsonl` plus the
     committed config.
- **How they resolved:** (1) 37 tests pass in 2.4 s. (2) Passed on the test
  split — 2.3284 bpb against 2.8233 for the bigram counter and 4.8619 for the
  untrained model. (3) **Not met as written, and that is the interesting
  result**: validation stopped falling at step 750 of 1,500 (11.9 epochs) and
  rose for the rest of the run, so the stopping criterion became "take the
  best checkpoint, report the cost of not doing so" — 14.6% in held-out bits
  per byte. (4) Met; every figure below is recomputable from
  `metrics/main-metrics.jsonl`, the two `eval-test-*.json` files and
  `configs/main.yaml`.

## Data card

- **Sources and versions:** tiny-shakespeare, the corpus introduced by
  `karpathy/char-rnn`, file `data/tinyshakespeare/input.txt` at master
  `6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e`.
- **Source URLs and access dates:** fetched 2026-10-17 from
  `https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt`
  with automatic fallback to
  `https://codeload.github.com/karpathy/char-rnn/tar.gz/refs/heads/master`
  (member `char-rnn-master/data/tinyshakespeare/input.txt`). The fallback is
  what actually served the bytes here, because `raw.githubusercontent.com` is
  not reachable from this sandbox.
- **License / permission / obligations checked by:** the agent, 2026-10-17,
  against the repository README (§License: MIT) and the GitHub API (`license`
  field is null, so the README is the authority — recorded because an automated
  license check would have reported "unknown"). The *text* is public domain:
  Shakespeare died in 1616. The *file* is distributed under char-rnn's MIT
  license. No attribution obligation beyond keeping the MIT notice if the file
  itself is redistributed; this repository does not redistribute it (it is
  fetched and `.gitignore`d).
- **Intended use and known restrictions:** research and teaching. Early modern
  English only, ~1.1 MB, single domain, no contemporary vocabulary. Not
  suitable for evaluating general language ability, factuality or safety.
- **Privacy and sensitive-data filtering:** the pipeline scans the full corpus
  before training and records counts in `manifest.json`. Result: 0 email
  addresses, 0 phone-like strings, 0 IPv4 addresses, 0 URLs, 0 AWS-key-like
  strings, 0 private-key headers. Nothing was removed, because nothing matched.
- **Cleaning and deduplication:** none applied. Duplication *measured* at the
  512-character block level: 7,222 blocks, 7,148 unique, 74 duplicate blocks
  (1.02%, 897 characters). Repeated short dialogue lines are a property of
  play text, not scrape noise, so removing them would distort the corpus. The
  measurement is recorded so the decision is visible rather than implicit.
- **Train/validation/test split method:** contiguous 90 / 5 / 5 in document
  order, boundaries snapped to the next blank line (character offsets 1,003,857
  and 1,059,682). A *shuffled* split of one continuous text leaks overlapping
  context windows across the boundary — the same sentence appears in train and
  validation under a one-character shift — which makes validation loss
  optimistic. Contiguous splitting avoids that and accepts a mild distribution
  shift between splits instead.
- **Estimated documents / bytes / tokens:** 1 document; 1,115,394 bytes
  (= characters, the file is ASCII); 1,115,394 tokens under the char tokenizer
  and 576,698 under byte-level BPE-512. Train split: 1,003,857 bytes →
  1,003,857 char tokens / 517,246 BPE tokens.
- **Data manifest location and hashes:** `scripts/tiny-gpt/data/prepared-char/manifest.json`
  and `.../prepared-bpe512/manifest.json`. Corpus
  `sha256:86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`,
  verified on every run; train-split text
  `sha256:8b1558a3b33e…`; char token stream `sha256:3043b67193…`;
  BPE token stream `sha256:1d57027d4b…`. The manifest also records the code
  revision that produced it.
- **Transformation / preprocessing code reference:**
  [`scripts/tiny-gpt/src/data.py`](../../scripts/tiny-gpt/src/data.py)
  (fetch → checksum → audit → split → train tokenizer on the train split only →
  encode → `uint16` `.bin` + manifest).

## Tokenizer

- **Algorithm and vocabulary size:** two, compared head to head.
  1. `char` — sorted set of characters in the train split, **67** symbols
     (65 characters + BOS + EOS).
  2. `byte_bpe` — byte-level BPE implemented from scratch (no `tokenizers`
     library), **512** merges-inclusive vocabulary.
- **Training corpus/split:** the **train split only**, for both. Fitting a
  vocabulary on validation text is leakage: it lets the model's input
  representation be shaped by text it is then scored on.
- **Special tokens:** BOS and EOS only (char: ids 65/66; BPE: ids 510/511).
  Both are assigned **zero bytes** in the bits-per-byte denominator, so adding
  special tokens cannot flatter the metric.
- **Tokenizer revision/hash:** char `sha256:77a855c9d7f1c924…`,
  BPE-512 `sha256:dd4a633eb519eab1…` (hash of the serialised `tokenizer.json`).
- **Tokens-per-character measurements (val split):** char 1.000 tok/char
  (1.000 bytes/token); BPE-512 **0.5294 tok/char**, **1.8889 bytes/token**.
  Consequence recorded before any run: at an equal *token* budget the BPE model
  sees 1.89× more text, so per-token loss is not comparable between them and
  every comparison below is in **bits per byte**.

## Model configuration

- **Architecture:** pre-norm decoder-only Transformer (GPT-2 shape, `bias=False`
  everywhere), learned absolute position embeddings, GELU MLP with 4× expansion,
  `F.scaled_dot_product_attention(..., is_causal=True)`.
- **Parameter count:** 3,230,720 total / 3,165,184 non-embedding with the char
  vocabulary; 3,344,640 / 3,165,184 with BPE-512. Reported both ways because
  the two tokenizers give the same model different totals.
- **Layers / width / heads / head dimension:** 4 / 256 / 4 / 64.
- **Maximum sequence length:** 256 tokens.
- **Attention / position encoding / norm / MLP:** causal SDPA / learned
  absolute / LayerNorm without bias / `Linear → GELU → Linear`.
- **Tied or untied input/output embeddings:** **tied** by default; untied is
  pilot p3.
- **Initialization and dtype:** normal(0, 0.02), residual projections scaled by
  `1/√(2·n_layer)`, FP32 throughout.
- **Optimizer / learning-rate schedule / gradient clipping:** AdamW
  (β₁ 0.9, β₂ 0.99 — raised because tokens-per-update is small), weight decay
  0.1 on matrices only, never on gains, biases or embeddings-as-vectors;
  100-step linear warmup then cosine decay to 0.1 × peak **over exactly the
  planned run length**; gradient clipping at 1.0 with the pre-clip norm logged.
- **Learning rate per parameter group:** one group of decayed matrices and one
  undecayed group, both at the same LR. (Noted explicitly because the nanochat
  anchor in `references/training-defaults.md` uses four Muon/AdamW groups whose
  LRs differ by 500×; this run does not.)
- **Precision and gradient scaler:** FP32, no scaler. AMP is a GPU feature;
  on CPU it would add casts without the tensor-core payoff.
- **Microbatch / gradient accumulation / effective tokens per update:**
  pilots 16 × 256 × 1 = **4,096** tokens/update; main run 16 × 256 × 2 =
  **8,192** tokens/update.

### Architecture note — shapes and where the gradients flow

```text
idx                      [B=16, T=256]            int64 token ids
 ├─ wte(idx)             [16, 256, 256]           token embedding, 512×256 = 131,072 params (tied)
 └─ wpe(arange(T))       [    256, 256]           learned positions, 256×256 = 65,536
sum → drop               [16, 256, 256]
 4 × Block:
   x + attn(ln_1(x))     ln_1: [256]              pre-norm: the residual stream is never normalised in place
     qkv = Linear(256→768)                        196,608 params → q,k,v each [16, 4, 256, 64]
     SDPA(is_causal=True)                         [16, 4, 256, 64]; the mask is the kernel's job, not a buffer
     proj = Linear(256→256)                        65,536 params, init scaled by 1/sqrt(2·n_layer)
   x + mlp(ln_2(x))      ln_2: [256]
     fc   = Linear(256→1024) → GELU               262,144 params
     proj = Linear(1024→256)                      262,144 params, same residual scaling
ln_f                     [256]
lm_head = wte.T          [16, 256, 512]           tied, so it adds 0 parameters
loss = cross_entropy(logits.view(-1,512), y.view(-1))
```

Per block: 262,144 attention + 524,288 MLP = 786,432 parameters; four blocks
give 3,145,728, plus 131,072 embedding + 65,536 positional + 2,304 norm
parameters = **3,344,640**. Gradients flow back through every residual
addition, so the embedding matrix receives gradient from both the input lookup
and the output projection — which is exactly why tying changes the training
dynamics and not only the parameter count. `flops_per_token()` returns
`6N + 12·L·T·d` = 6 × 3,344,640 + 12 × 4 × 256 × 256 = **23,213,568**, and
`tests/test_shapes.py` re-derives that formula independently so the code and
`references/compute-budget-example.md` cannot drift apart.

## Pilot matrix

One variable per row, identical budget of 400 steps × 4,096 tokens = 1,638,400
tokens, same seed 1337. Filled in from `artifacts/runs/<id>/summary.json`.

| Pilot ID | Variable changed | Value | Tokens | Val loss | val bpb | tok/s | Peak mem | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| p1-char-baseline | — (reference) | char 67, lr 1e-3, dropout 0.1, tied | 1,638,400 | 2.3325 | 3.3651 | 5,080 | 1,553 MB | baseline |
| p2-bpe512 | tokenizer | byte-level BPE-512 | 1,638,400 | 3.5667 | 2.7252 | 5,022 | 1,553 MB | **adopt** |
| p3-untied | embedding tying | untied | 1,638,400 | 2.3194 | 3.3462 | 4,851 | 1,577 MB | reject: +0.6% bpb for −4.5% throughput |
| p4-lr3e-3 | learning rate | 3e-3 | 1,638,400 | 2.0710 | 2.9878 | 5,149 | 1,535 MB | **adopt** |
| p5-nodropout | dropout | 0.0 | 1,638,400 | 2.2382 | 3.2291 | 7,129 | 1,162 MB | **adopt** |
| p6-bpe-lr3e-3 | the three winners combined, at the main-run batch | BPE-512, 3e-3, 0.0, 8,192 tok/update | 1,638,400 | 3.5308 | **2.6979** | 6,992 | 1,229 MB | **main run** |
| p7-bpe-lr4.2e-3 | LR under the √batch rule | 4.2e-3 | 1,638,400 | 3.5600 | 2.7201 | 7,172 | 1,255 MB | reject: 0.8% worse than 3e-3 |

Full reasoning, the equal-wall-clock comparison and the plotted curves:
[`pilots.md`](pilots.md).

- **Gate for advancing to the main run:** ≥ 5,000 tok/s (measured 6,992),
  peak RSS < 2,500 MB (measured 1,229), validation falling monotonically over
  all four evals (it did), no non-finite steps (none), and a lower bpb than the
  previous best configuration at equal tokens (2.6979 < 2.7252). All five met,
  so the main run was authorised.

## Pilot measurement

- **Hardware and software versions:** 2 vCPU Intel Xeon @ 2.60 GHz, 3 GB RAM,
  no GPU. Linux 6.1.158, Python 3.11.2, torch 2.14.1+cu130 (the default PyPI
  wheel; the `+cu130` tag is misleading — `torch.cuda.is_available()` is False
  and every FLOP here is on CPU), numpy 2.4.6, `torch.set_num_threads(2)`.
- **Warm-up and compilation excluded from steady-state throughput?** No
  `torch.compile`, so there is nothing to exclude; step 1 runs ~25% slower than
  steady state and *is* included in the mean, making the reported figure
  slightly conservative.
- **Sustained tokens/second:** 6,992 at the chosen configuration; 7,172 for the
  identically shaped p7, so ~7,000 ± 3% on a shared sandbox.
- **Peak accelerator memory:** n/a (no accelerator). Peak resident memory
  1,229 MB, 41% of the machine.
- **Validation loss trend:** 2.9590 → 2.8670 → 2.7516 → 2.6979 bpb over p6's
  four evals; still falling at the end, so the main run extends the token
  budget rather than the model.
- **Checkpoint size and duration:** 38.4 MB (`last.pt`: FP32 weights plus two
  AdamW moments), written atomically in about 0.4 s.
- **Estimate (tokens, wall time, cost) plus uncertainty:** 1,500 steps ×
  8,192 = 12,288,000 tokens at ~7,000 tok/s ≈ 1,756 s, plus ~150 s for ten
  validation passes and three sample generations ⇒ **≈ 32 min ± 10%**. Cost
  $0 (local CPU). A `max_seconds: 2700` guard stops the run cleanly rather
  than letting an underestimate run away.

## Run manifest

- **Git commit:** `6f1e224` (`scripts/tiny-gpt` as committed; the data manifests
  record the same revision, written by `src/data.py` at prepare time).
- **Config file/hash:** [`scripts/tiny-gpt/configs/main.yaml`](../../scripts/tiny-gpt/configs/main.yaml),
  reproduced verbatim inside the `run_start` record of
  [`metrics/main-metrics.jsonl`](metrics/main-metrics.jsonl) — the config is
  stored *with the metrics*, so a run can never be read against the wrong file.
- **Dataset and tokenizer hash:** corpus `sha256:86c4e6aa9db7c042…`; BPE-512
  tokenizer `sha256:dd4a633eb519eab1…`; train token stream
  `sha256:1d57027d4bdc607b…`. The data was regenerated from scratch immediately
  before the run and reproduced every hash bit for bit, which also verifies
  that the pipeline is deterministic.
- **Random seed(s):** 1337 for data order and initialization; 0 for the
  evaluation batcher; sampling seeds recorded per sample in `samples.txt`.
- **Start time / end time:** 2026-10-18 00:16:36 → 00:42:13 UTC
  (1,535.5 s = 25.6 min).
- **Tokens trained / optimizer steps:** 12,288,000 tokens / 1,500 steps at
  8,192 tokens per update. **Selected checkpoint: step 750, 6,144,000 tokens.**
- **Checkpoint IDs:** `artifacts/runs/main/best.pt` (step 750, lowest val loss
  2.7570) and `artifacts/runs/main/last.pt` (step 1500). 38.4 MB each, not
  committed.
- **Training and validation metrics:** 10 evaluations, every 150 steps, in
  `metrics/main-metrics.jsonl` with per-record loss, LR, pre-clip gradient
  norm, tokens/s, elapsed time and peak RSS. Sustained **8,003 tok/s**
  (185.8 GFLOP/s by the `6N + 12·L·T·d` estimate), peak RSS 1,269 MB, median
  pre-clip gradient norm 0.83, final train loss 1.3002.
- **Clipped, skipped, or failed optimizer steps:** 380 of 1,500 steps clipped
  at 1.0 (25%, rising through the run as the model fit the corpus harder);
  **0** non-finite steps; 0 skipped; no restarts.
- **Exceptions, restarts, and decisions:** none during the run. Two decisions
  recorded after it: (a) the run was *not* stopped early even though validation
  turned at step 750, because the rising half of the curve is the measurement
  the report needed; (b) `best.pt`, not `last.pt`, is the model, and the 14.6%
  held-out difference between them is reported rather than hidden.
- **Estimate vs actual:** estimated 32 min ± 10% at ~7,000 tok/s; actual
  25.6 min at 8,003 tok/s. The pilots were measured while other work shared the
  two cores, so the estimate was conservative by 14% — the right direction to
  be wrong in, and the reason the estimate is recorded here next to the
  outcome.

## Evaluation and release gate

Full report: [`eval-report.md`](eval-report.md). Summary of the gate:

- **Held-out loss/perplexity protocol:** test split, never used for any
  decision, scored with deterministic non-overlapping 256-token windows
  (29,696 tokens / 55,349 bytes, 99.3% of the split; the trailing partial
  window is dropped). `best.pt`: **3.0082 nats/token, perplexity 20.25**.
- **Bits per byte:** **2.3284** for `best.pt`, 2.6685 for `last.pt`. Byte
  lengths come from the tokenizer's own table, with special tokens counted as
  zero bytes so they cannot flatter the metric.
- **Metric comparability:** the two checkpoints share a tokenizer, so their
  per-token losses are comparable. Nothing else in this project is: the char
  and BPE pilots are compared in bits per byte only, which is why the plan
  fixed that metric before the first run.
- **Domain-specific evaluation and baseline:** no downstream benchmark is
  meaningful at 3.3M parameters. Baselines are uniform (4.8298 bpb), unigram
  (4.0068), add-0.1 bigram (2.8233), the same architecture at random init
  (4.8619), and three general-purpose compressors including `xz -9` primed
  with the training text (2.5583). The model beats all of them.
- **Generation prompts/settings and saved outputs:** [`samples.txt`](samples.txt)
  — four prompts across temperature 0.4/0.8/1.2 and top-k 40/disabled, each
  with its seed and checkpoint.
- **Leakage/contamination checks:** contiguous splits on blank-line
  boundaries so no window spans a boundary; tokenizer fitted on train only;
  test split touched twice, after the run.
- **Privacy/memorization and safety tests:** corpus audit (0 hits across six
  sensitive patterns); memorization probe with a longest verbatim match of 27
  characters over 4 × 240 sampled tokens. No safety evaluation — and none
  claimed.
- **Known limitations:** one 1.1 MB single-domain corpus; 3.3M parameters;
  CPU-only measurements; single seed, so sub-1% differences are ties; the
  run overfits after 11.9 epochs.
- **Release decision and rationale:** **do not release the weights.** They are
  38 MB of FP32 Adam state that regenerate in 26 minutes and would imply a
  usefulness the measurements do not support. Released instead: the code, the
  configs, the metric stream, the evaluation report, the pilot matrix and the
  [model card](model-card.md), all of which reproduce from a clean checkout.

## Next-step roadmap

Each step is gated on the previous one, in the order the evidence justifies:

1. **Fix the data budget before anything else.** 24 epochs over 1 MB is why
   this run overfit; no regularizer, optimizer or architecture change
   substitutes for more text. A 50–100 MB public-domain corpus would let the
   same model train for a single epoch and change the shape of the problem.
2. **Then re-tune the regularization** at that budget — dropout, weight decay
   and the token budget interact, and the pilot answer (dropout 0.0) was
   correct *for 1.6M tokens on CPU* and wrong by step 900.
3. **Then scale the model**, not before: with more data, width and depth have
   something to learn from. Re-run the pilot matrix at the new scale rather
   than extrapolating these numbers.
4. **Then consider a GPU and BF16 autocast**, which changes throughput by
   orders of magnitude and invalidates every timing measured here.
5. **Only then** longer context, better attention kernels, or any form of
   post-training. SFT on a model that cannot hold a clause together would be
   measuring the template, not the model.

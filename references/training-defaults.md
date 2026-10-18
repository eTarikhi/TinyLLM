# Training defaults grounded in maintained harnesses

`SKILL.md` deliberately avoids prescribing hyperparameters, because the right values depend on scale, optimizer, tokenizer and hardware. That leaves a gap when someone asks "so what number do I actually type?" This file fills it with **observed values from two maintained harnesses**, read from source on 2026-10-17, not from memory or blog posts:

- **nanoGPT** (`karpathy/nanoGPT`) — `train.py`, `config/train_shakespeare_char.py`, `config/train_gpt2.py`. AdamW, single global learning rate, the classic educational setup. Deprecated upstream but its hyperparameters remain the clearest published AdamW reference for a small GPT.
- **nanochat** (`karpathy/nanochat`) — `scripts/base_train.py`, `nanochat/optim.py`, `nanochat/loss_eval.py`. Modern: a hybrid Muon + AdamW optimizer with per-parameter-group learning rates.

Treat these as evidence of what a tuned run looks like. Re-tune on the target hardware; do not paste them into a different architecture or optimizer and assume they transfer.

## Anchor A — tiny educational GPT (~10.8M parameters)

This is the direct analogue of the 5–20M starting model `SKILL.md` recommends. nanoGPT's character-level Shakespeare config: `n_layer=6`, `n_head=6`, `n_embd=384`, `block_size=256`, `batch_size=64`, `gradient_accumulation_steps=1`, vocab 65, and `bias=False` from `train.py`'s defaults, which the config does not override.

That is **10,745,088 parameters in total, with the input and output embeddings tied** — `model.py` does `self.transformer.wte.weight = self.lm_head.weight`, so the 24,960-parameter embedding matrix is one tensor serving two roles. nanoGPT itself prints `10.65M`, because `get_num_params()` reports the *non-embedding* count and subtracts the 98,304-parameter position embedding. Both figures are inside the recommended range; quote either, but say which one, and do not describe this model as untied. (An earlier version of this file claimed 10,765,056 untied parameters: the right structure, the wrong tying assumption, and silently missing the LayerNorm weights.)

| Component | Count |
| --- | --- |
| Token embedding `65 × 384` (tied with the output head) | 24,960 |
| Position embedding `256 × 384` | 98,304 |
| 6 blocks × (`3d²` + `d²` + `4d²` + `4d²` + two LayerNorm weights) | 10,621,440 |
| Final LayerNorm | 384 |
| **Total** | **10,745,088** |
| Non-embedding, as nanoGPT prints it | 10,646,784 |

| Setting | Value | Upstream rationale (verbatim comment) |
| --- | --- | --- |
| Peak learning rate | `1e-3` | "with baby networks can afford to go a bit higher" |
| Min learning rate | `1e-4` | "learning_rate / 10 usually" |
| `beta2` | `0.99` | "make a bit bigger because number of tokens per iter is small" |
| Warmup | `100` iters | "not super necessary potentially" |
| Total iters | `5000` | — |
| `lr_decay_iters` | `5000` | "make equal to max_iters usually" |
| Dropout | `0.2` | — |
| Eval interval | `250` | "keep frequent because we'll overfit" |
| Tokens per update | `16,384` (2^14) | `64 × 256 × 1` |

Note the dropout: `0.2` here, but `0.0` in the pretraining config below, where upstream comments "for pretraining 0 is good, for finetuning try 0.1+". A tiny model on a tiny corpus overfits, which is why the char-level config regularizes and the large-scale one does not.

## Anchor B — ~124M parameter AdamW pretraining

nanoGPT's GPT-2 reproduction defaults: `n_layer=12`, `n_head=12`, `n_embd=768`, `block_size=1024`, `batch_size=12`, `gradient_accumulation_steps=40`.

| Setting | Value | Upstream rationale |
| --- | --- | --- |
| Peak learning rate | `6e-4` | "max learning rate" |
| Min learning rate | `6e-5` | "should be ~= learning_rate/10 per Chinchilla" |
| `beta1` / `beta2` | `0.9` / `0.95` | — |
| Weight decay | `1e-1` | — |
| Gradient clip | `1.0` | "clip gradients at this value, or disable if == 0.0" |
| Warmup | `2000` iters | — |
| Total iters | `600000` | — |
| `lr_decay_iters` | `600000` | "should be ~= max_iters per Chinchilla" |
| Dropout | `0.0` | "for pretraining 0 is good" |
| Bias in LN/Linear | `False` | — |
| Dtype | `bfloat16` if supported, else `float16` | — |
| Tokens per update | `491,520` | `12 × 1024 × 40` |

The dtype line is upstream code doing exactly what `SKILL.md` advises: prefer BF16 where the hardware supports it, fall back to FP16 otherwise. BF16 needs no gradient scaler; FP16 does.

## Anchor C — modern hybrid optimizer (nanochat)

| Setting | Value | Note |
| --- | --- | --- |
| Matrix-parameter LR | `0.02` | Muon |
| Embedding LR | `0.3` | AdamW |
| Unembedding LR | `0.008` | AdamW |
| Scalar LR | `0.5` | AdamW |
| Weight decay | `0.28` | "cautious weight decay for the Muon optimizer" |
| Warmup | `40` steps | — |
| Warmdown fraction | `0.65` | 65% of the run is spent decaying |
| Final LR fraction | `0.05` | vs nanoGPT's `0.1` |
| Reference batch | `2^19 = 524,288` tokens | "optimal batch size at d12, measured empirically" |
| Batch scaling | `B ∝ D^0.383` | optimal batch grows sublinearly with the token horizon; the exponent is cited in `scripts/base_train.py` to *Power Lines: Scaling Laws for Weight Decay and Batch Size in LLM Pre-training*, [arXiv:2505.13738](https://arxiv.org/abs/2505.13738) |
| LR vs batch | `η ∝ sqrt(B/B_ref)` | Adam-style square-root scaling, not SGD's linear |
| Param:token ratio | `12` | upstream help text notes "Chinchilla=20" |
| Width rule | `d_model = depth × 64` | `--aspect-ratio 64` |
| Head dim | `128` | — |
| Attention pattern | `SSSL` | sliding window, S = half context, L = full |

**Do not copy learning rates across optimizers.** nanochat's embedding LR of `0.3` is roughly 500× nanoGPT's peak AdamW LR of `6e-4`, and that is not a mistake or an improvement you can borrow — it is a different update rule with a different effective scale. Muon orthogonalizes its update, so its natural LR sits in a completely different range. Quoting one harness's LR as a default for another optimizer is a reliable way to diverge.

Upstream also states the boundary explicitly: Muon "should not be used for the embedding layer, the final fully connected layer, or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW)." Hence the split into four parameter groups with four learning rates.

## What transfers, and what does not

Patterns that hold across both harnesses and both scales:

1. **Final LR is a fixed fraction of peak** — `0.1` in nanoGPT at both 10.8M and 124M, `0.05` in nanochat. Order of magnitude is stable; pick one and record it.
2. **The decay horizon equals the training horizon.** Both nanoGPT configs set `lr_decay_iters = max_iters`. A schedule that finishes decaying early trains at the floor for the remainder; one that never finishes leaves the LR high at the end.
3. **Warmup is short and not very sensitive** — 100 steps at 10.8M, 2000 at 124M, 40 in nanochat. It is a small fraction of the run in every case.
4. **Gradient clipping at 1.0** is the published AdamW value. Clip, and log the pre-clip norm so you can see when clipping is actually binding.
5. **Tokens per update grows with scale**, from `2^14` at ~10.8M to roughly `2^19` at 100M+. The two independent sources agree closely at the larger scale: nanoGPT's 124M config computes to 491,520 tokens per update, and nanochat's empirically measured optimum is 524,288 — a ratio of 0.94.
6. **`beta2` rises when tokens per update is small** (0.99 at 2^14 tokens/update vs 0.95 at ~2^19). Small batches give noisier second-moment estimates, so they need more averaging.

What does **not** transfer:

- Learning rates across different optimizers (see above).
- Dropout between pretraining and fine-tuning, or between a data-starved tiny run and a data-rich large one.
- The param:token ratio. nanochat defaults to 12 while noting Chinchilla's 20 — a concrete demonstration of why `SKILL.md` refuses to treat that ratio as universal. The right value depends on whether you are optimizing for validation loss alone or for inference cost as well.
- Batch size and LR independently. They are coupled by `η ∝ sqrt(B/B_ref)`; changing one without the other silently changes the effective step size.

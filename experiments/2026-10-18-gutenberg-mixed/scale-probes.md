# Probes at 3.74M: learning rate, context, vocabulary

Four short-clock arms (2,571 steps, seed 1337, broad corpus), each moving
one variable off `broad-4L256d-full` (test 1.6467 bits/byte). Their job was
to de-risk the configuration of the long release run: three settings that
had only ever been bracketed at 1.77M parameters on smaller corpora.

| probe | change | test bpb | vs baseline |
| --- | --- | ---: | ---: |
| `big-lr15` | peak LR 3e-3 → 1.5e-3 | 1.6838 | +2.26% |
| `big-lr60` | peak LR 3e-3 → 6e-3 | 1.6708 | +1.47% |
| `big-ctx512` | context 256 → 512 (token-matched) | 1.7519 | +6.39% |
| `big-vocab4096` | BPE vocabulary 2048 → 4096 | **1.6236** | **−1.40%** |

**Learning rate: the optimum does not move.** 3e-3, chosen at 1.77M
parameters on 27 books, is still an interior optimum at 3.74M on 151 —
halving costs +2.26%, doubling costs +1.47%. The intuition that a wider
model wants a cooler peak does not bite at this width and horizon; if
anything the short clock punishes caution more than heat.

**Context: 512 is much worse — a larger penalty than the same move
measured at 1.77M.** +6.39% here, at identical tokens per step. More data and more parameters did not teach the model
to use the longer window within 21M tokens — plausibly the opposite:
the broad mixture's shortest documents (poems, scenes) fill a 512 window
with padding-free but unrelated joins less often usefully, and attention
over twice the positions is simply slower to organise. Whatever the
mechanism, the release stays at 256. (Whether a *long* clock closes the
gap is untested; +6.39% is too large to gamble a release on.)

**Vocabulary: the tokenizer lever pays a second time.** 4,096 merges
beat 2,048 by −1.40% in bits/byte at a fixed token budget — about four
times the 0.37% seed bar. Part of the win is arithmetic (more bytes per
token, so more text per step and per 256-token window) and part is real
(the same reason BPE beat characters: frequent words become single
decisions). It is not parameter-matched — the tied embedding grows by
0.52M — and it trains ~15% slower per byte; even discounted for wall
clock the win survives at roughly −0.8%.

The release configuration follows directly: vocabulary 4096, context
256, peak LR 3e-3, and the one lever that has never stopped paying —
double the horizon, to 10,346 steps (`broad-v4096-release`).

## Release v2

`broad-v4096-release` scores **1.4939 bits/byte** on the seen 71 —
−4.16% against the previous generalist (1.5587) and −4.62% against the
released 1.77M model (1.5662); the two wins stack almost exactly as the
probes predicted (−1.40% vocabulary, ~−3% horizon). Every form improves:
narrative 1.4777, non-fiction 1.7518, verse 1.8078, experimental 1.8405.
On the unseen 40 it scores 1.7907, −1.99% better than the v1 generalist
and −9.96% better than the narrow-corpus model of the same size. It is,
on both yardsticks, the best model this project has produced, and it
exports to `tinylm-4.3m-broad-v4096.gguf` with greedy output verified
identical through llama.cpp.

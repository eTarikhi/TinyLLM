# Research notes and reading list

Reviewed 2026-10-17. This is a curated starting list for the `llm-from-scratch` skill; always verify live repository state, API syntax, licenses, and hardware pricing before a run.

## Verification log

Not every link below was re-checked on the review date, and the difference matters when a recommendation is repeated to a user. Status as of 2026-10-17, when **all 15 links were re-checked**, three of them turning out to need a correction:

| Link | Finding | Action |
| --- | --- | --- |
| Hugging Face Tokenizers quicktour | The `/docs/tokenizers/python/latest/quicktour.html` page serves a banner reading "You are viewing legacy docs" and points at a current page | URL updated to `/docs/tokenizers/quicktour` |
| "PyTorch AMP documentation" | `/docs/stable/accelerator/amp.html` exists, but it is the guide for *accelerator backend authors* — registering autocast kernels under `AutocastPrivateUse1` — not the AMP usage docs a training project needs | Replaced with `torch.amp` and the AMP examples note |
| nanoGPT repository | Linked as `github.com/karpathy/nanogpt`; GitHub resolves it case-insensitively, but the repository is `karpathy/nanoGPT` | Case corrected |

**Re-checked live (5 of 15 links)** against `api.github.com` — existence, archive flag, license, and last push:

| Repository | Live | License | Last push |
| --- | --- | --- | --- |
| `karpathy/nanochat` | yes | MIT | 2026-09-07 |
| `karpathy/nanoGPT` | yes, README states deprecated and points to nanochat | MIT | 2025-11-12 |
| `karpathy/llm.c` | yes | MIT | 2025-06-26 |
| `EleutherAI/lm-evaluation-harness` | yes | MIT | 2026-09-14 |
| `EleutherAI/pythia` | yes | Apache-2.0 | 2025-11-15 |

**Re-checked by fetching the page (10 of 15 links)** — `arxiv.org` (3), `huggingface.co` (4), `docs.pytorch.org` (2), `allenai.org` (1). These were outside the network allowlist of the environment where this log was first written; they have since been fetched and compared against what this file claims about them:

- arXiv `1706.03762`, `2203.15556`, `2307.08691` — titles and authors match (*Attention Is All You Need*; Hoffmann et al., *Training Compute-Optimal Large Language Models*; Dao, *FlashAttention-2*).
- Hugging Face course `chapter7/6` — live, titled "Training a causal language model from scratch", and it still makes the point this file cites it for: pretraining from scratch costs materially more compute than fine-tuning.
- **FineWeb license confirmed at the source:** Open Data Commons Attribution License (ODC-By) v1.0, and the dataset card states that use is *also* subject to CommonCrawl's Terms of Use. Two licences, not one — attribute under ODC-By and read the CommonCrawl terms before redistribution.
- Hugging Face Datasets streaming and PyTorch distributed pages — live, unchanged in substance.
- Ai2 OLMo 2 — live; the open-data/open-code/intermediate-checkpoints claim this file makes is still accurate, and the technical report is arXiv `2501.00656`.
- The two corrections above (tokenizers quicktour, AMP docs) came out of this pass.

**Package versions** re-checked against the PyPI JSON API on 2026-10-17:

| Distribution | Latest | Released | `requires-python` |
| --- | --- | --- | --- |
| `torch` | 2.14.1 | 2026-09-30 | >=3.10 |
| `tokenizers` | 0.23.3 | 2026-10-09 | >=3.10 |
| `datasets` | 5.1.0 | 2026-10-05 | >=3.10.0 |
| `lm-eval` | 0.4.13 | 2026-08-31 | >=3.10 |
| `numpy` | 2.5.4 | 2026-10-10 | >=3.12 |

Three findings from that pass worth carrying into a project brief:

1. **The evaluation harness cannot be installed under its repository name.** `pip install lm-evaluation-harness` returns HTTP 404; the distribution is `lm-eval`. Its PyPI metadata points back at the `EleutherAI/lm-evaluation-harness` repository, which is why the two names get conflated.
2. **The Python interpreter silently caps the numerical stack.** The newest numpy declaring support for Python 3.10 is 2.2.6, for 3.11 it is 2.4.6, and only 3.12+ reaches 2.5.4. nanochat targets Python 3.10, so its environment cannot resolve the newest numpy. Choose the interpreter before pinning, and record both in the manifest.
3. **Maintained harnesses lag the framework.** nanochat pins `torch==2.9.1` while the latest release is 2.14.1. A pinned harness is not broken by this — it is deliberate — but do not assume it runs unmodified on a newer framework, and do not upgrade a pinned training stack mid-run.

Reproduce these checks with:

```bash
# repository state
curl -s https://api.github.com/repos/karpathy/nanochat | jq '{archived, pushed_at, license: .license.spdx_id}'
# package versions (and whether an old name still exists)
curl -s https://pypi.org/pypi/lm-eval/json | jq -r '.info.version'
curl -s -o /dev/null -w '%{http_code}\n' https://pypi.org/pypi/lm-evaluation-harness/json   # -> 404
```


## Learn the model

- **Attention Is All You Need** — foundational Transformer architecture paper. [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

- **Hugging Face: Training a causal language model from scratch** — hands-on tokenizer, data, and pretraining workflow; notes that scratch pretraining needs materially more compute than fine-tuning. [https://huggingface.co/docs/course/chapter7/6](https://huggingface.co/docs/course/chapter7/6)

- **Hugging Face Tokenizers quick tour** — training a BPE tokenizer and tokenizer pipeline concepts. The `/docs/tokenizers/python/latest/` path still resolves but serves a "legacy docs" banner; this is the current page. [https://huggingface.co/docs/tokenizers/quicktour](https://huggingface.co/docs/tokenizers/quicktour)

## Code to inspect

- **nanochat** — end-to-end, minimal/hackable harness covering tokenization, pretraining, fine-tuning, evaluation and inference. Its main path targets one GPU node, but `runs/runcpu.sh` is a maintained CPU / Apple-Silicon (MPS) path worth reading when no CUDA GPU is available. It pins `torch==2.9.1` exactly. Treat its cost examples as specific to its setup, not a general quote. [https://github.com/karpathy/nanochat](https://github.com/karpathy/nanochat)

- **nanoGPT** — useful historical small GPT implementation; its README now says it is deprecated and points to nanochat. [https://github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)

- **llm.c** — low-level C/CUDA training and GPT-2 reproduction work; useful after understanding the PyTorch baseline. [https://github.com/karpathy/llm.c](https://github.com/karpathy/llm.c)

- **PyTorch AMP docs** — `torch.amp` API: autocast, `GradScaler`, and which dtypes need scaling. [https://docs.pytorch.org/docs/stable/amp.html](https://docs.pytorch.org/docs/stable/amp.html) — worked patterns, including gradient accumulation and clipping under AMP, are in the examples note: [https://docs.pytorch.org/docs/stable/notes/amp_examples.html](https://docs.pytorch.org/docs/stable/notes/amp_examples.html). Do not cite `/docs/stable/accelerator/amp.html` for this: that page documents how to register autocast kernels for a *new accelerator backend*.

- **PyTorch distributed docs** — scale to multi-GPU only after a single-device run is correct. [https://docs.pytorch.org/docs/stable/distributed.html](https://docs.pytorch.org/docs/stable/distributed.html)

## Data

- **FineWeb dataset card** — examples of corpus scale, preprocessing and stated license; the user must still verify the license and obligations for their use. [https://huggingface.co/datasets/HuggingFaceFW/fineweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb)

- **Hugging Face Datasets streaming** — useful when datasets are much larger than local storage. [https://huggingface.co/docs/datasets/main/stream](https://huggingface.co/docs/datasets/main/stream)

- **Ai2 OLMo 2** — reference for a transparently released model family with accessible training data/code, checkpoints and evaluations. [https://allenai.org/olmo2](https://allenai.org/olmo2)

## Compute and evaluation

- **Chinchilla paper** — research on model/data allocation under a compute budget; useful historical evidence, not a universal sizing formula. [https://arxiv.org/abs/2203.15556](https://arxiv.org/abs/2203.15556)

- **FlashAttention-2** — memory-efficient attention implementation research; adopt only where hardware/software supports it and benchmark it on the target workload. [https://arxiv.org/abs/2307.08691](https://arxiv.org/abs/2307.08691)

- **EleutherAI lm-evaluation-harness** — common framework for standardized benchmark evaluation. Confirm the tasks and settings match before comparing results. The repository is `EleutherAI/lm-evaluation-harness` but the PyPI distribution is **`lm-eval`**; installing by the repository name returns a 404. [https://github.com/EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)

- **Pythia** — public training artifacts and intermediate checkpoints that support reproducibility and study of model learning dynamics. [https://github.com/EleutherAI/pythia](https://github.com/EleutherAI/pythia)

## Evidence-backed project decisions

1. Scratch training is appropriate for learning, domain-specific data, or research questions, but is generally much more expensive than adapting an existing model. (Hugging Face scratch-training course.)

2. Do a tiny implementation and data sanity checks before increasing model size or GPU count.

3. Measure target-hardware throughput before estimating run time or budget; costs quoted by project READMEs depend on exact hardware, pricing, software and run parameters.

4. A transparent model report should describe data provenance, tokenizer, code, checkpoints, evaluation protocol, and limitations—not just weights.

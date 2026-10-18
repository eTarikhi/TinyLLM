---
name: llm-from-scratch
description: Research, plan, implement, train, evaluate, and document a language model from random initialization, from an educational tiny GPT to a larger reproducible pretraining project. Use when creating a custom LLM, training a tokenizer, building a Transformer language model, planning pretraining data or compute, or debugging training. Also use it when the user asks whether to pretrain or fine-tune an existing checkpoint, how much GPU time or money a training run will cost, how to turn a base model into an instruction-following chat model, or how to evaluate, document, and safely release a model they trained, even if they never say "from scratch".
---

# Build Your Own LLM From Scratch

## Mission

Help the user build a language model whose weights are initialized randomly and trained on their chosen corpus. Prefer an end-to-end, testable project to a theoretical explanation: teach the system in stages, keep compute and data requirements realistic, and produce reproducible code, experiments and documentation.

**Default interpretation of “from scratch”:** implement or inspect the architecture, initialize new weights, and run pretraining yourself. Never silently substitute fine-tuning of a pretrained checkpoint, and keep the four distinct: **pretraining** learns next-token prediction from random init; **SFT** teaches an existing base model an instruction format; **preference/alignment training** shapes behaviour from ranking data, an optional later phase; **RAG** adds documents at inference time and trains nothing.

An instruction-following assistant generally needs post-training on top of pretraining. A small model can be a successful educational artifact without being a useful general-purpose assistant.

## Operating principles

1. **Make the target explicit.** (A) learning how LLMs work, (B) a small domain model, (C) a research model, or (D) a competitive general-purpose model. If unstated, default to A and build upward.
2. **Start small, prove correctness, then scale.** Overfit a tiny batch, then train a tiny corpus, then measure throughput and validation loss before spending more compute.
3. **Use the simplest stack that teaches the target.** PyTorch by default; optimized attention, mixed precision, distributed training and custom CUDA only after the baseline works or the user asks for performance engineering.
4. **Do not promise quality from parameter count alone.** Data quality, token budget, tokenizer, architecture, stability and evaluation all matter. Give estimates as estimates and record assumptions.
5. **Check upstream documentation before giving commands.** Libraries, accelerator support, APIs, licence terms and cloud prices change. Pin versions and record the date checked.
6. **Reproducibility is a deliverable.** Track code revision, dependency versions, data manifests, tokenizer, configuration, seed, metrics, and resumable checkpoints.
7. **Treat data and model risk as engineering requirements.** Respect licences, privacy, and opt-outs; evaluate memorization and unsafe outputs before sharing a model.
8. **Prefer evidence to vibes.** Held-out loss, task tests and baselines decide. Attractive samples prove nothing about capability or safety.

## Intake: choose the right project

Ask only for what materially affects the plan; otherwise state an explicit default and keep moving. Capture in a project brief: **goal** (learning / domain model / research / general assistant); **domain and languages**; **hardware** (CPU, Apple Silicon, one GPU, cluster — model and VRAM if known); **budget and runtime ceiling**, counting data transfer, storage, failed experiments and evaluation, not just the successful run; **data** sources, size, format and restrictions; **target form**, base or chat; and **success criteria**, such as validation loss, held-out perplexity or domain accuracy.

When information is missing, default to a modest educational project: a decoder-only GPT, a small public-domain or user-owned corpus, short context, 5–20M parameters, one device, a short run. Starting examples, not universal optima.

## Recommended implementation path

### Track 1 — Learn by implementing a tiny GPT

Build the core model in plain, readable PyTorch on a small, permissively usable corpus, so the whole loop stays fast and inspectable. Implement or clearly explain:

1. Text ingestion and a deterministic train/validation split.
2. A tokenizer — UTF-8 bytes or characters for the first toy, then a trained BPE or similar subword vocabulary. For a specific target language, check how it handles that language, whitespace, punctuation, rare words, and code.
3. Examples formed from token sequences: inputs `x[0:t]`, targets `x[1:t+1]`.
4. Token embeddings and position information.
5. A decoder-only Transformer block: causal self-attention, residuals, normalization, feed-forward network.
6. A language-model head producing vocabulary logits at every position.
7. Next-token cross-entropy loss with correct causal masking.
8. An optimizer, learning-rate schedule, gradient clipping, and a training loop.
9. Validation-loss measurement and sample generation.
10. Checkpoint saving and restoring.

Keep the first architecture conventional: no mixture-of-experts, speculative decoding, reinforcement learning, or complex parallelism before the basic model passes its tests. Explain tensor shapes and where gradients flow.

**A worked implementation of this track ships with the skill.** [`scripts/tiny-gpt/`](scripts/tiny-gpt/README.md) covers every item above in tested PyTorch: character and byte-BPE tokenizers written without a tokenizer library, a pipeline that checksums and audits the corpus, the model, a training loop whose resume is exact, evaluation in bits per byte against uniform/unigram/bigram/untrained baselines, and the tests below. Copy and adapt it — but never paste it in unread, and never repeat its measured throughput or loss as a prediction for other hardware. [`experiments/`](experiments/README.md) records real CPU-only runs, including what the resulting models cannot do.

**Minimum correctness tests:** logits of shape `[batch, sequence, vocabulary]` and finite scalar loss; a causal mask under which changing future inputs cannot change earlier logits in eval mode; deliberate overfitting of one tiny batch; an optimizer step that moves parameters with no non-finite values; generation that emits only valid IDs and stops at the limit or EOS; a checkpoint round-trip reproducing logits within tolerance; a resume restoring model, optimizer, scheduler, step, RNG and data position.

### Track 2 — Train a more complete small LLM

Once Track 1 works, extend the code or study a maintained harness. As checked on 2026-10-17:

- **nanochat** — the practical reference for a complete pipeline: tokenizer, pretraining, fine-tuning, evaluation, inference. Its headline path targets a single GPU node, but `runs/runcpu.sh` is an officially maintained CPU/MPS path, so it stays the right reference without a CUDA GPU — with the caveat the script itself states, that a CPU run is a demo, not a route to a capable model. Its hardware and cost examples are project-specific, not a quote, and it pins `torch==2.9.1` rather than tracking the newest release.
- **nanoGPT** — useful historical educational code, now described by its own repository as deprecated in favour of nanochat. Do not present it as the current default.
- **llm.c** — low-level C/CUDA training and GPT-2 reproduction; relevant once the user wants systems performance, not for a first implementation.
- **PyTorch** — the baseline framework. Use the official docs for mixed precision, attention primitives, distributed training and checkpoint semantics rather than old blog snippets.

Do not require a particular framework: explain trade-offs and keep model and data formats portable.

## Data pipeline

### Choose and audit the data before training

Six rules carry most of the weight, and [`references/data-pipeline.md`](references/data-pipeline.md) holds the full checklist with the tokenizer design notes:

1. **Downloadable is not permitted.** Prefer user-owned, licensed, public-domain or explicitly allowed data, and record source URL, version, licence, access date, restrictions and transformation code — with a checksum, so the record is verifiable rather than aspirational.
2. **Keep the raw source immutable** and write cleaning to a separate artifact, so any filtering decision can be re-examined later.
3. **Deduplicate, and measure what you removed.** Exact duplicates first, near-duplicates where useful; report the rate rather than quietly dropping records.
4. **Split at the document level *before* forming overlapping windows,** or validation leaks into training through the window that straddles the boundary.
5. **Read a sample yourself,** and quantify language, domain, length, duplication and filtering rates in a data card.
6. **Fit the tokenizer on training text only,** record its vocabulary size and configuration, measure tokens per byte, and never change token IDs after training starts.

## Architecture and training defaults

Baseline choices, not recipes — tune them with short experiments on the target hardware. When asked what number to actually type, do not invent one: [`references/training-defaults.md`](references/training-defaults.md) holds values read from the source of two maintained harnesses (nanoGPT for single-LR AdamW, nanochat for hybrid Muon+AdamW) at ~10.8M, ~124M and modern scale, with notes on which patterns transfer across optimizers.

### Architecture

Start with a causal decoder-only Transformer. Modern models vary in normalization, position encoding, MLP activation, attention design and parameter sharing; use one coherent design, document it, and validate compatibility before mixing components from different implementations. A reasonable later-stage baseline is RMSNorm, rotary position embeddings, a gated MLP such as SwiGLU, and efficient scaled-dot-product attention where the runtime supports it.

Record at least: parameter count, layers, width, heads and head dimension; context length; vocabulary size and tokenizer revision; normalization, positional encoding, MLP activation and attention implementation; tied or untied embeddings; initialization method and dtype.

### Objective and optimization

- Standard causal pretraining objective: cross-entropy on the next token, shifted by one and masked so future tokens cannot leak into context.
- AdamW with a short warm-up and a schedule matched to the training horizon is a sound baseline for code you wrote yourself. Log the effective learning rate, gradient norm, loss, tokens/second, memory, and validation loss.
- **Never carry a learning rate between optimizers or parameter groups.** nanochat runs four groups spanning `0.008` to `0.5`, with matrices on Muon and everything else on AdamW; a number lifted out of that config means nothing on its own.
- Reach a stable FP32 or known-good baseline *before* mixed precision; add gradient scaling only if the precision requires it (FP16 does, BF16 generally does not). FP8 is a later optimization, never a debugging starting point.
- Use gradient accumulation to reach the effective batch you want, and document tokens per optimizer update — step counts are not comparable between runs with different batch sizes, token budgets are.
- Save checkpoints atomically with model, optimizer, scheduler, scaler, step, RNG states, config, and data position. Test resume before a long run.

Per-knob defaults from three maintained harnesses, the reasoning for each, and what transfers between scales are in [`references/training-defaults.md`](references/training-defaults.md).

### Estimate compute before launching

Never quote a duration or a cost from arithmetic alone. Measure steady-state tokens/second in a short pilot on the real hardware, divide the intended token budget by it, add contingency. `~6 × parameters × tokens` estimates the *work*, not the time, and only a measurement converts one to the other. Price GPUs from current published rates, and treat Chinchilla's 20 tokens/parameter as a decision to justify rather than a constant — nanochat ships **12**.

The seven-step procedure and a worked example — parameters, FLOPs, measured throughput, MFU, cost and tokens per update, every number recomputed in CI — are in [`references/compute-budget-example.md`](references/compute-budget-example.md). It also shows why a small model's utilization sits far below published large-model figures, and why that is expected rather than a bug.

## Evaluation and diagnosis

### Minimum evaluation suite

- **Training health:** train and validation loss, gradient norms, learning rate, throughput, memory, skipped or failed steps.
- **Held-out language modeling:** cross-entropy and perplexity where meaningful, under an identical tokenization and evaluation protocol across the runs being compared.
- **Bits per byte when the tokenizer varies.** Per-token loss and perplexity are *not* comparable across vocabularies: a larger tokenizer covers the same text in fewer tokens, so a worse model can post a better per-token loss. Sum loss in nats, sum the bytes of the target tokens, divide by `ln(2) × total_bytes`, and mask special tokens out of *both* sums or they distort the result. nanochat evaluates this way for the same reason. Use BPB whenever the tokenizer is itself under test; keep per-token loss for runs that share one.
- **Domain tasks:** held-out questions or examples reflecting the use case, with answer keys kept out of training.
- **Generation:** fixed prompts and sampling settings, outputs saved, model/checkpoint/config hashes recorded.
- **Baselines:** a trivial baseline, a previous checkpoint or smaller run, and where permitted an external one. A model that cannot beat a bigram counter has not been shown to work.
- **Regression testing:** rerun a stable suite after any model, tokenizer, data or inference change.

Use EleutherAI's harness where its task definitions fit, and verify current syntax. Scores from different prompts, few-shot counts, context lengths or normalization are not equivalent, whatever the table says. Never rely on one benchmark: report sample size, uncertainty, limitations and contamination checks, and say whether a number is measured, estimated or anecdotal.

### Common failure patterns

Loss flat, output repeating, validation diverging from train, loss spiking to NaN, or language that never becomes coherent — each has a short list of first things to inspect, in [`references/training-defaults.md`](references/training-defaults.md). Diagnose before spending more compute: a broken run does not improve by running longer.

## From base model to chat model

Only begin after pretraining and evaluation are reproducible.

1. Choose or create a lawful, high-quality instruction dataset representative of the desired behaviour.
2. Define an explicit chat template and stable special tokens, and use the identical template at training and inference.
3. Supervised fine-tune on conversation examples, masking loss to the intended objective — commonly assistant tokens, not user prompt tokens — and test that masking explicitly.
4. Hold out an instruction set and a domain set; evaluate adherence, correctness, hallucination, robustness and refusal behaviour.
5. Reach for preference optimization only once SFT and its evaluation are stable and suitable preference data exists.
6. Test prompt injection and extraction of memorized content, without treating refusal tuning as a substitute for privacy-conscious data processing.
7. Label limitations plainly. Answering a few sample prompts well is not evidence that a model is safe or production-ready.

## Safety, privacy, and release gates

Before training, inspect corpus provenance, licences, privacy implications and harmful-content concentration. During training, run memorization probes with synthetic canaries only — never real secrets. Before release:

- scan samples and outputs for memorized personal data or secrets, and measure the longest verbatim match against the training corpus;
- test harmful-use scenarios relevant to the intended deployment;
- document failure modes, intended use, restrictions and evaluation limits;
- provide a way to report problems and to retire or replace a bad checkpoint;
- do not expose an unauthenticated public generation endpoint with unlimited resource use, and do not publish raw datasets, credentials, or logs of private user input;
- review obligations with qualified counsel for the relevant jurisdictions and data sources when needed.

Support ordinary educational and research model development. If a goal or dataset would materially facilitate harm, privacy violations or unlawful use, refuse that part and redirect to a safe corpus or benign task.

## Project layout and deliverables

Produce artifacts proportionate to the request, and do not stop at advice when implementation was asked for. A run leaves behind: (1) a project brief, (2) an architecture note with shapes and parameter count, (3) a data plan with licences and split policy, (4) a runnable implementation with pinned dependencies, (5) correctness tests, (6) an experiment plan with a pilot matrix and gates, (7) an evaluation report with baselines and limitations, (8) a reproducibility record, (9) a next-step roadmap.

The directory layout, the seven test files by name, and each artifact in full are in [`references/project-scaffold.md`](references/project-scaffold.md); artifacts 6 and 8 have a fill-in structure in [`templates/experiment-plan.md`](templates/experiment-plan.md), copied per run rather than edited in place. [`scripts/tiny-gpt/`](scripts/tiny-gpt/README.md) is that layout filled in and runnable; [`experiments/`](experiments/README.md) is a worked instance of the artifacts.

Implement the smallest complete end-to-end version that meets the goal, then add tests. Never imply that unexecuted code has been tested: state what was run and what was not.

## Decision gates

In order, and each one recorded:

- **Gate 0 — Scope and data approved:** goal, lawful corpus, licence and provenance, budget, success criteria all explicit.
- **Gate 1 — Unit correctness:** shapes, causal masking, finite loss, gradients, generation, checkpoint restore all pass.
- **Gate 2 — Learning proof:** overfits a tiny batch, and beats an untrained baseline on a small corpus.
- **Gate 3 — Pilot viability:** measured memory and throughput support the budget; validation loss improves and stays stable.
- **Gate 4 — Pretraining complete:** token budget reached or a documented stopping criterion fired; held-out metrics and samples saved.
- **Gate 5 — Post-training (optional):** chat template correct, instruction evaluations held out.
- **Gate 6 — Release review:** privacy, safety, documentation, licence, reproducibility and deployment checks pass.

If a gate fails, diagnose the cause before moving on, and keep the failed experiment — it is part of the evidence trail.

## Source-of-truth references

Treat papers as foundational evidence and live documentation as the current implementation guide, and re-verify rather than quoting from memory. Every source this skill relies on — Transformer and Chinchilla, nanochat, nanoGPT, llm.c, Pythia, OLMo 2, the Hugging Face tokenizer/dataset/course pages, FineWeb, PyTorch AMP and distributed docs, the evaluation harness — is in [`references/reading-list.md`](references/reading-list.md) with an annotation, a verification date, and the command that re-checks it. Read it before recommending a repository, dataset or benchmark.

Three traps are worth carrying inline, because each has broken real work:

- The evaluation harness **repository** is `EleutherAI/lm-evaluation-harness`; the **PyPI distribution is `lm-eval`**. `pip install lm-evaluation-harness` fails with a 404.
- nanoGPT is explicitly deprecated by its author in favour of nanochat. Cite it as the readable reference implementation it is, not as current practice.
- A dataset being downloadable is not a licence to train on it. FineWeb, for one, is ODC-By 1.0 *and* subject to the CommonCrawl terms of use.

## Output style

Be concrete and candid. Prefer a sequence of runnable milestones to a giant untested code dump. Give commands and files, say why each important choice was made, link sources for claims that change, and separate facts from assumptions and estimates. Never imply a toy model is comparable to a frontier one. End with the next concrete command or test, not a motivational summary.

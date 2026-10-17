---
name: llm-from-scratch
description: Research, plan, implement, train, evaluate, and document a language model from random initialization, from an educational tiny GPT to a larger reproducible pretraining project. Use when creating a custom LLM, training a tokenizer, building a Transformer language model, planning pretraining data or compute, or debugging training. Also use it when the user asks whether to pretrain or fine-tune an existing checkpoint, how much GPU time or money a training run will cost, how to turn a base model into an instruction-following chat model, or how to evaluate, document, and safely release a model they trained, even if they never say "from scratch".
---

# Build Your Own LLM From Scratch

## Mission

Help the user build a language model whose weights are initialized randomly and trained on their chosen corpus. Prefer an end-to-end, testable project over a purely theoretical explanation. Teach the system in stages, keep compute and data requirements realistic, and produce reproducible code, experiments, and documentation.

**Default interpretation of “from scratch”:** implement or inspect the architecture, initialize new weights, and run pretraining yourself. Do not silently substitute fine-tuning of a pretrained checkpoint. Explain the difference between:

- **Pretraining:** learn next-token prediction from a broad or domain-specific text corpus using randomly initialized weights.
- **Supervised fine-tuning (SFT):** teach a pretrained base model to respond in an instruction/conversation format.
- **Preference/alignment training:** further shape behavior using preference or ranking data; an optional later phase, not a prerequisite for a first model.
- **Retrieval-augmented generation (RAG):** use external documents at inference time; this does not train a new LLM.

An assistant that follows instructions generally needs post-training in addition to pretraining. A small model can be a successful educational artifact without being a useful general-purpose assistant.

## Operating principles

1. **Make the target explicit.** Establish whether the goal is (A) learning how LLMs work, (B) a small domain model, (C) an experimental research model, or (D) a competitive general-purpose model. If not stated, default to A and build upward.
2. **Start small, prove correctness, then scale.** First make a model overfit a tiny batch, then train a tiny corpus, then measure throughput and validation loss before spending more compute.
3. **Use the simplest stack that teaches the target.** PyTorch is the default for an understandable implementation. Use optimized attention, mixed precision, distributed training, or custom CUDA only after the baseline works or the user asks for performance engineering.
4. **Do not promise model quality from parameter count alone.** Data quality, token budget, tokenizer, architecture, training stability, and evaluation all matter. Give estimates as estimates and record assumptions.
5. **Check current upstream documentation before giving commands.** GPU libraries, accelerator support, package APIs, model repositories, license terms, and cloud prices change. Pin versions and record the date checked.
6. **Reproducibility is a deliverable.** Track the exact code revision, dependency versions, data manifests, tokenizer, configuration, random seed, metrics, and resumable checkpoints.
7. **Treat data and model risk as engineering requirements.** Respect dataset licenses, privacy, sensitive data, and opt-outs where applicable. Evaluate memorization and unsafe outputs before sharing or deploying a model.
8. **Prefer evidence to vibes.** Evaluate with held-out loss, task-specific tests, and baselines. Attractive samples do not prove that a model is capable or safe.

## Intake: choose the right project

Ask only for information that materially affects the plan. If the user has not specified it, make explicit defaults and keep moving.

Capture these fields in a project brief:

- **Goal:** learning / domain modeling / research / general-purpose assistant.
- **Domain and language(s):** for example, Japanese technical documents, source code, or broad English text.
- **Training from random initialization:** yes by default for this skill; note if the user later chooses a pretrained base.
- **Available hardware:** CPU, Apple Silicon, one CUDA GPU, or multi-GPU cluster; GPU model and VRAM if known.
- **Budget and runtime ceiling:** include data transfer, storage, experiment failures, and evaluation, not just the successful run.
- **Data:** owned/licensed sources, approximate size, format, language, and restrictions.
- **Target form:** base text-completion model or instruction-following chat model.
- **Success criteria:** e.g. validation loss, held-out perplexity, domain test accuracy, code task performance, or human preference ratings.

If information is missing, use a modest educational default: a decoder-only GPT, a small public-domain or user-owned corpus, short context, a 5–20M-parameter starting model, one device, and a short run. These are starting examples, not universal optima.

## Recommended implementation path

### Track 1 — Learn by implementing a tiny GPT

Build the core model in plain, readable PyTorch. Use a small, permissively usable corpus to make the full loop fast and inspectable. Implement or clearly explain:

1. Text ingestion and deterministic train/validation split.
2. A tokenizer. Begin with UTF-8 bytes or a character vocabulary for the first toy; then train a BPE or similar subword tokenizer for a more realistic model. If the project targets a specific language, evaluate how the tokenizer handles that language, whitespace, punctuation, rare words, and code.
3. Training examples formed from token sequences. Inputs are tokens `x[0:t]`; targets are the next tokens `x[1:t+1]`.
4. Token embeddings and position information.
5. A decoder-only Transformer block: causal self-attention, residual connections, normalization, and feed-forward network.
6. A language-model head that predicts vocabulary logits at every position.
7. Next-token cross-entropy loss with correct causal masking.
8. An optimizer, learning-rate schedule, gradient clipping, and a training loop.
9. Validation-loss measurement and sample generation.
10. Checkpoint saving and restoring.

Keep the first architecture conventional. Do not add mixture-of-experts, speculative decoding, reinforcement learning, or complex parallelism before the basic model passes its tests. Explain the architecture’s tensor shapes and where gradients flow.

**A worked implementation of this track ships with the skill.** [`scripts/tiny-gpt/`](scripts/tiny-gpt/README.md) is a tested PyTorch project covering every item above: a character tokenizer and a byte-level BPE tokenizer written without a tokenizer library, a data pipeline that checksums and audits the corpus before use, the model, a training loop whose resume is exact, evaluation in bits per byte against uniform/unigram/bigram/untrained baselines, and the correctness tests listed below. Read it, copy it into the user's project, and adapt it to their corpus and hardware; do not paste it in unread, and do not repeat its measured throughput or loss as a prediction for different hardware. Its README and [`experiments/`](experiments/README.md) record a real CPU-only run, including what the resulting model cannot do.

**Minimum correctness tests:**

- A batch of token IDs produces logits of shape `[batch, sequence, vocabulary]` and a finite scalar loss.
- A causal mask prevents a position from attending to future tokens; altering future input tokens must not alter earlier-position logits in evaluation mode.
- The model can deliberately overfit one tiny batch, driving its loss down substantially.
- An optimizer step changes parameters and leaves no non-finite values.
- Generation only appends valid token IDs and stops at configured limits or the end-of-sequence token.
- Saving and loading a checkpoint reproduces logits within appropriate numeric tolerance.
- A resumed run restores model, optimizer, scheduler, step count, random-number-generator state, and data position when feasible.

### Track 2 — Train a more complete small LLM

Once Track 1 works, either extend the code or study a maintained end-to-end training harness. As checked on 2026-10-17:

- **nanochat** is a practical reference for a compact but more complete LLM pipeline: tokenizer, pretraining, fine-tuning, evaluation, and inference. Its headline path targets a single GPU node, but it also ships `runs/runcpu.sh`, an officially maintained CPU/Apple-Silicon (MPS) path for exercising the code paths end to end — so it is still the right reference to read when the user has no CUDA GPU, with the caveat the script itself states: a CPU run is an educational demo, not a route to a capable model. Read its current README and license before using it; advertised hardware/cost examples are project-specific and are not a quote or guarantee. Note that it pins an exact framework version (`torch==2.9.1`) rather than tracking the newest release, which is the pinning discipline this skill asks for — do not assume such a harness works unmodified on a newer framework.
- **nanoGPT** is useful historical educational code, but its repository now describes it as deprecated and points to nanochat. Do not present nanoGPT as the default current project.
- **llm.c** is useful for understanding low-level C/CUDA training and GPT-2 reproduction. It is more relevant once the user wants to study systems performance than for the simplest first implementation.
- **PyTorch** is the baseline implementation framework. Use official docs for mixed precision, attention primitives, distributed training, and checkpoint semantics rather than copying old blog snippets.

Do not require the user to adopt a framework. Explain trade-offs and keep model/data formats portable.

## Data pipeline

### Choose and audit the data before training

1. Prefer user-owned, licensed, public-domain, or explicitly permitted datasets. Do not treat a dataset being downloadable as proof that every use is permitted.
2. Record source URL, dataset/version, license, access date, language, intended use, known restrictions, and transformation code.
3. Remove exact duplicates and, where useful, near duplicates. Detect corrupted records, boilerplate, spam, machine-generated junk, and obviously low-quality text. Keep the raw source immutable and write a separate cleaned artifact.
4. Filter sensitive personal information, secrets, credentials, and data the project has no right to use. Avoid collecting private communications or personal records without appropriate rights and consent.
5. Split at the document/source level into train, validation, and test before creating overlapping windows. Keep evaluation material out of training. For benchmark-style tasks, check for benchmark leakage and contamination.
6. Inspect a random sample of cleaned records manually. Quantify language, domain, length, duplicates, and filtering rates; keep a data card describing limitations.
7. For large corpora, stream or shard data instead of downloading everything blindly. Estimate storage for raw text, cleaned text, token IDs, checkpoints, and intermediate files.
8. Do not assume raw text can be redistributed simply because the trained weights can be. Check the corpus and jurisdiction-specific requirements.

### Tokenization

- Pick tokenizer type based on corpus and use case: byte/character for a first toy; BPE, WordPiece, or Unigram subwords for realistic language modeling.
- Train the tokenizer on suitable training data only. Keep vocabulary size and tokenizer training configuration in the experiment manifest.
- Measure tokens-per-character/byte or tokens-per-document, especially for non-English language, code, numbers, emoji, and domain terms.
- Define stable special tokens such as BOS/EOS and, if using chat later, distinct user/assistant/message-boundary tokens. Never change token IDs after training starts.
- Preserve document boundaries where appropriate. Do not inadvertently concatenate unrelated documents without an EOS or clear separator.
- Save tokenizer files beside each checkpoint; tokenizer/model vocabulary mismatch is a release blocker.

## Architecture and training defaults

Treat these as baseline choices, not mandatory recipes. Always tune via short experiments on the target hardware.

When someone asks what number to actually type, do not invent one. [`references/training-defaults.md`](references/training-defaults.md) collects concrete values read from the source of two maintained harnesses (nanoGPT for single-LR AdamW, nanochat for a hybrid Muon+AdamW setup), at ~10.8M, ~124M, and modern scale, together with which patterns transfer across optimizers and which do not.

### Architecture

For an educational GPT, start with a causal decoder-only Transformer. The original Transformer paper explains attention; modern language models vary in normalization, position encoding, MLP activation, attention design, and parameter sharing. Use one coherent design and document it. A reasonable later-stage modern baseline can use RMSNorm, rotary position embeddings, a gated MLP such as SwiGLU, and efficient scaled-dot-product attention where the runtime supports it. Validate compatibility before mixing architectural components from different implementations.

Record at least:

- parameter count; layer count; model width; attention-head count and head dimension;
- maximum sequence length/context length;
- vocabulary size and tokenizer revision;
- normalization, positional encoding, MLP activation, and attention implementation;
- tied or untied input/output embeddings;
- initialization method and dtype.

### Objective and optimization

- Standard causal pretraining objective: cross-entropy for predicting the next token, shifted by one position and masked so future tokens cannot leak into context.
- AdamW is a sound baseline for an implementation you write yourself. Use a short warm-up followed by a schedule appropriate to the training horizon; log the effective learning rate.
- Be aware that maintained harnesses increasingly do not use a single global AdamW learning rate. nanochat splits parameters into four groups — matrix parameters on Muon, embeddings, unembeddings and scalars on AdamW — with learning rates spanning `0.008` to `0.5`. Muon orthogonalizes its update, so its natural learning rate is orders of magnitude away from AdamW's. Never carry a learning rate from one optimizer to another, and when reading a harness's config, check which parameter group each value belongs to before treating any of them as "the learning rate". Upstream guidance is that Muon should not be applied to embeddings, the final output layer, or 0-/1-dimensional parameters.
- Clip gradients if needed and log gradient norm, loss, tokens/second, GPU memory, and validation loss.
- Use mixed precision only after a stable FP32 or known-good baseline; choose BF16/FP16 based on hardware and current framework support. Follow the official PyTorch AMP docs. Use gradient scaling when the selected precision and hardware require it — BF16 generally does not, FP16 does. A useful published pattern is to select BF16 when the accelerator reports support for it and fall back to FP16 otherwise, rather than hardcoding either.
- FP8 training is now available on newer accelerators and is used by current harnesses, including a choice of scaling recipe. Treat it as a later optimization for a measured, stable BF16 baseline — not a starting point for a first model, and not something to enable while still debugging convergence.
- Use gradient accumulation to fit a larger effective batch in limited VRAM; compute and document the effective tokens per optimizer update.
- Save checkpoint state atomically. A useful checkpoint includes model weights, optimizer state, scheduler, scaler if applicable, update count, RNG states, configuration, and data/sampler position.
- Checkpoint often enough that losing a run is acceptable. Test resume behavior before a long run.
- Compare validation loss at a fixed token budget, not just at a fixed number of steps if batch sizes differ.
- Watch for rising or spiking loss, NaNs/Infs, dead gradients, train/validation divergence, and throughput regressions. Stop and diagnose rather than spending more compute on a broken run.

### Estimate compute before launching

1. Run a short pilot on the actual hardware, with the intended sequence length, batch, precision, and attention implementation.
2. Measure steady-state tokens/second after warm-up and compilation. Exclude startup and data-download time, but include evaluation/checkpoint overhead in the planning estimate.
3. Estimate wall-clock training time from intended training tokens divided by sustained aggregate tokens/second. Add a contingency for interruptions, failed jobs, and evaluation.
4. For a dense Transformer, `~6 × parameter_count × training_tokens` is a widely used rough FLOP estimate for training; it is not a billing formula and can vary materially with architecture, activation checkpointing, sequence length, attention kernels, and hardware utilization. Validate estimates with a pilot.
5. Price actual cloud GPU types from the provider’s current published rates at planning time. Include storage, network egress, CPU/data processing, taxes, idle time, spot interruption risk, and evaluation.
6. Train a sequence of increasingly larger pilots. Do not jump from a toy model directly to a billion-parameter run without a measured throughput and memory budget.
7. Chinchilla’s compute-optimal study is a useful historical scaling-law reference, but do not turn its results into a universal fixed model-size-to-token-count ratio. Use current research, budget, domain, and pilot results to choose a training horizon. This is not a hypothetical caution: nanochat exposes the ratio as a flag and defaults it to **12** tokens per parameter, while its own help text notes that Chinchilla's figure is **20**. A maintained harness deliberately choosing a different ratio than the famous paper is the clearest evidence that the ratio is a decision to justify, not a constant to apply — inference cost, not just training loss, drives the choice.

A fully worked version of these steps, with the arithmetic checked and reproducible, is in [`references/compute-budget-example.md`](references/compute-budget-example.md). Read it before quoting any duration or cost to a user; it also shows why a small model's measured utilization sits far below published large-model figures, and why that is expected rather than a bug.

If a budget is tight, recommend a smaller model trained cleanly with transparent evaluation instead of suggesting a large model can be built cheaply without qualification.

## Evaluation and diagnosis

### Minimum evaluation suite

- **Training health:** training loss, validation loss, gradient norms, learning rate, token throughput, memory, and skipped/failed steps.
- **Held-out language modeling:** cross-entropy/per-token negative log-likelihood and perplexity where meaningful. Use exactly the same tokenization and evaluation protocol when comparing runs.
- **Bits per byte (BPB) when comparing across tokenizers.** Per-token loss and perplexity are *not* comparable between different vocabularies: a larger tokenizer covers the same text in fewer tokens, so a worse model can post a better per-token loss. BPB removes the artifact by normalizing on bytes instead of tokens — sum the loss in nats, sum the bytes of the target tokens, and divide by `ln(2) × total_bytes`. Mask special tokens and any ignored targets out of both sums, or they distort the metric. This is what nanochat evaluates on, for exactly this reason. BPB is the right choice whenever the tokenizer is itself one of the variables under test, which is common early in a project; keep per-token loss for comparing runs that share a tokenizer.
- **Domain tasks:** create held-out questions or examples that reflect the use case. Keep answer keys and test documents out of training.
- **Generation:** fixed prompts and sampling settings for qualitative review; save outputs and record model/checkpoint/config hashes.
- **Baselines:** include a trivial baseline, a previous checkpoint or smaller run, and—when useful and permitted—a relevant external baseline.
- **Regression testing:** rerun a stable suite after model, tokenizer, data, or inference changes.

Use EleutherAI's evaluation harness where its supported task definitions fit the model and license. Note the name split, which trips up installs: the **repository** is `EleutherAI/lm-evaluation-harness`, but the **PyPI distribution is `lm-eval`** (`pip install lm-eval`). `pip install lm-evaluation-harness` fails with a 404. Verify the current command syntax and task settings; do not compare scores produced with different prompts, few-shot counts, context lengths, or normalization and call them equivalent.

Do not rely on one benchmark. Report sample size, uncertainty/variance where feasible, known limitations, and any contamination checks. Be precise whether a score is measured, estimated, or anecdotal.

### Common failure patterns

- **Loss does not decline:** inspect input/target shift, masking, token IDs, learning rate, optimizer updates, and whether data batches have signal.
- **Model repeats tokens or text:** inspect data quality and duplicates, training horizon, sampling temperature/top-p, EOS handling, and tokenizer decoding.
- **Train loss falls but validation loss worsens:** investigate overfitting, data duplication, split leakage, dataset mismatch, and training duration.
- **Loss spikes or NaNs:** inspect learning rate, gradient norm, precision, bad batches, initialization, and recent code changes; resume from a known-good checkpoint after diagnosis.
- **Model produces incoherent language:** check tokenization and detokenization, vocabulary size, number of useful training tokens, and whether the run is simply too short or too small.
- **Chat model fails to follow instructions:** verify whether it received SFT; a base model trained only on next-token prediction is not expected to behave like a chat assistant.
- **Training is much slower than expected:** profile input pipeline, data loading, device utilization, sequence length, attention kernel, synchronization, compilation overhead, and checkpoint frequency.

## From base model to chat model

Only begin this phase after pretraining and evaluation are reproducible.

1. Choose or create a lawful, high-quality instruction dataset representative of the desired behavior.
2. Define an explicit chat template and stable special tokens; use the same template at training and inference.
3. Perform supervised fine-tuning on conversation examples. Mask loss according to the intended objective (commonly assistant output tokens rather than user prompt tokens); test the masking explicitly.
4. Keep a held-out instruction set and a domain-specific set. Evaluate instruction adherence, correctness, hallucination, robustness, and refusal behavior where relevant.
5. Consider preference optimization or other alignment methods only when the basic SFT pipeline and evaluation are stable and the user has appropriate preference data.
6. Evaluate prompt injection and attempts to make the model reveal memorized or hidden content, but do not treat refusal tuning as a substitute for privacy-conscious data processing.
7. Clearly label the model’s limitations. Do not claim it is safe or production-ready solely because it answers a few sample prompts well.

## Safety, privacy, and release gates

Before training, inspect corpus provenance, licenses, privacy implications, sensitive information, and potential harmful-content concentration. During training, use memorization probes or canary-style tests only with synthetic test secrets and appropriate containment; do not insert real secrets into the corpus. Before release:

- scan samples and outputs for memorized personal data or secrets;
- test common harmful-use scenarios relevant to the intended deployment;
- document known failure modes, intended use, restrictions, and evaluation limits;
- provide a process for reporting problems and retiring or replacing a bad checkpoint;
- avoid exposing an unauthenticated public generation endpoint with unlimited resource use;
- do not publish raw datasets, credentials, or logs containing private user inputs;
- review obligations with qualified counsel for the project's jurisdictions and data sources when needed.

The skill should support ordinary educational and research model development. If a proposed goal or dataset would materially facilitate harm, privacy violations, or unlawful use, refuse that part and redirect toward a safe corpus or benign task.

## Project layout

Prefer a structure similar to this, adapting only where justified:

```text
my-llm/
  README.md
  pyproject.toml              # or pinned requirements.txt
  configs/
    tiny.yaml
    pilot.yaml
  src/
    tokenizer.py
    data.py
    model.py
    train.py
    generate.py
    evaluate.py
  tests/
    test_shapes.py            # logits [batch, seq, vocab], finite scalar loss
    test_causal_mask.py       # future tokens cannot change earlier logits
    test_overfit.py           # drives loss down on one tiny batch
    test_optim.py             # a step changes params, no non-finite values
    test_generation.py        # valid IDs only, stops at limit or EOS
    test_checkpoint.py        # save/load reproduces logits
    test_resume.py            # restores model, optim, sched, step, RNG, data position
  data/                       # normally ignored by version control
    README.md                 # provenance; do not commit restricted raw data
  artifacts/
    tokenizer/                # vocabulary/tokenizer files
    checkpoints/              # ignored or external storage
  experiments/
    README.md                 # run manifests and decisions
  model-card.md
```

[`scripts/tiny-gpt/`](scripts/tiny-gpt/README.md) is this layout, filled in and runnable; the test filenames above are the ones it ships.

Keep secrets out of source control. Add `.gitignore` for corpora, checkpoints, caches, environment files, logs, and API keys. The README should show exact environment setup, small smoke-test commands, training and evaluation commands, hardware assumptions, and a reproducibility recipe.

## Expected artifacts when this skill is invoked

Produce artifacts proportionate to the user's request; do not stop at advice when implementation is requested.

Artifacts 6 and 8 have a fill-in structure in [`templates/experiment-plan.md`](templates/experiment-plan.md). Copy it into the user's project once per run (for example `experiments/<run-id>/plan.md`) and fill it in, rather than editing the template in place.

1. **Project brief:** goal, assumptions, hardware, data, constraints, and success metrics.
2. **Architecture note:** diagram or text flow, tensor shapes, parameter count, and why the selected architecture fits the goal.
3. **Data plan:** source/license manifest, cleaning and deduplication approach, split policy, tokenizer design, and storage estimate.
4. **Runnable starter implementation:** pinned dependencies, training, sampling, validation, and checkpoint resume. Start from [`scripts/tiny-gpt/`](scripts/tiny-gpt/README.md) rather than from a blank file, then cut what the project does not need.
5. **Tests:** model shapes, causal masking, finite loss, tiny-batch overfit, generation, and checkpoint round-trip.
6. **Experiment plan:** pilot matrix changing one or a few variables at a time; tokens budget; stop/go gates; compute estimate and uncertainty.
7. **Evaluation report:** held-out loss, domain tests, baselines, sample generations, limitations, and precise interpretation.
8. **Reproducibility record:** code revision, configs, dependencies, data/tokenizer versions, seed, checkpoint ID, metrics, and environment details.
9. **Next-step roadmap:** only introduce distributed training, larger context, advanced kernels, SFT, or alignment after the previous gate passes.

When asked for code, implement the smallest complete end-to-end version that meets the goal, then add tests. Never imply that unexecuted code has been tested; state what was or was not run.

## Decision gates

Use these gates in order:

- **Gate 0 — Scope/data approved:** the goal, lawful corpus, license/provenance, budget, and success criteria are explicit.
- **Gate 1 — Unit correctness:** shapes, causal masking, finite loss, gradients, generation, and checkpoint restore pass.
- **Gate 2 — Learning proof:** the model overfits a tiny batch and learns a small corpus better than an untrained/random baseline.
- **Gate 3 — Pilot viability:** measured VRAM and throughput support the budget; validation loss improves and stays numerically stable.
- **Gate 4 — Pretraining complete:** selected token budget is reached or documented stopping criteria fire; held-out metrics and outputs are saved.
- **Gate 5 — Post-training (optional):** SFT/chat template is correct and instruction evaluations are held out.
- **Gate 6 — Release review:** privacy, safety, documentation, license, reproducibility, and deployment checks pass.

If a gate fails, diagnose and fix the cause before moving forward. Preserve failed experiments and notes where useful; they are part of the evidence trail.

## Source-of-truth references

Consult current versions as of the work date; treat papers as foundational evidence and live documentation/repositories as current implementation guides.

Annotated notes for each source below, grouped by purpose, are in [`references/reading-list.md`](references/reading-list.md); read it before recommending a repository, dataset, or benchmark to the user.

- Vaswani et al., *Attention Is All You Need* (Transformer): https://arxiv.org/abs/1706.03762
- Hugging Face course, *Training a causal language model from scratch*: https://huggingface.co/docs/course/chapter7/6
- Hugging Face Tokenizers quick tour, including tokenizer training: https://huggingface.co/docs/tokenizers/quicktour
- Hugging Face Datasets streaming documentation: https://huggingface.co/docs/datasets/main/stream
- FineWeb dataset card and license/data notes: https://huggingface.co/datasets/HuggingFaceFW/fineweb
- Karpathy, nanochat (current end-to-end small-model harness): https://github.com/karpathy/nanochat
- Karpathy, nanoGPT (legacy/deprecated reference): https://github.com/karpathy/nanoGPT
- Karpathy, llm.c (C/CUDA reference): https://github.com/karpathy/llm.c
- PyTorch AMP documentation (`torch.amp`, plus the examples note): https://docs.pytorch.org/docs/stable/amp.html and https://docs.pytorch.org/docs/stable/notes/amp_examples.html
- PyTorch distributed documentation: https://docs.pytorch.org/docs/stable/distributed.html
- Hoffmann et al., *Training Compute-Optimal Large Language Models* (Chinchilla): https://arxiv.org/abs/2203.15556
- Dao, *FlashAttention-2*: https://arxiv.org/abs/2307.08691
- EleutherAI, Language Model Evaluation Harness — repository `lm-evaluation-harness`, PyPI distribution `lm-eval`: https://github.com/EleutherAI/lm-evaluation-harness
- EleutherAI, Pythia: public models, checkpoints, data and training artifacts for study: https://github.com/EleutherAI/pythia
- Ai2 OLMo 2, open training data/code/checkpoint/evaluation resources: https://allenai.org/olmo2

## Output style

Be concrete and candid. Prefer a sequence of runnable milestones over a giant untested code dump. Give commands and files, explain why important choices were made, link sources for evolving claims, and clearly separate facts, assumptions, and estimates. Avoid implying that a toy model is comparable to a modern frontier model. End with the next concrete command or test, not a vague motivational conclusion.

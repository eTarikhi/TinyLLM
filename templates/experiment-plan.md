# LLM pretraining experiment plan

## Goal and constraints

- Goal / use case:

- Base model or instruction-following model:

- Target languages and domains:

- Hardware and accelerator memory:

- Maximum cloud/local budget:

- Runtime ceiling:

- Success metrics and stopping criteria:

## Data card

- Sources and versions:

- Source URLs and access dates:

- License / permission / obligations checked by:

- Intended use and known restrictions:

- Privacy and sensitive-data filtering:

- Cleaning and deduplication:

- Train/validation/test split method:

- Estimated documents / bytes / tokens:

- Data manifest location and hashes:

- Transformation / preprocessing code reference:

## Tokenizer

- Algorithm and vocabulary size:

- Training corpus/split:

- Special tokens:

- Tokenizer revision/hash:

- Tokens-per-character or tokens-per-document measurements:

## Model configuration

- Architecture:

- Parameter count:

- Layers / width / heads / head dimension:

- Maximum sequence length:

- Attention / position encoding / norm / MLP:

- Tied or untied input/output embeddings:

- Initialization and dtype:

- Optimizer / learning-rate schedule / gradient clipping:

- Learning rate per parameter group (if the optimizer uses more than one):

- Precision (FP32 / BF16 / FP16 / FP8) and gradient scaler used:

- Microbatch / gradient accumulation / effective tokens per update:

## Pilot matrix

One or a few variables per row. Record failed pilots too — they are the evidence behind the go/no-go decision, and `references/compute-budget-example.md` shows how to turn these measurements into a wall-clock and cost estimate.

| Pilot ID | Variable changed | Value | Tokens | Val loss | tok/s | Peak mem | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |

- Gate for advancing to the main run (throughput, memory, loss trend, stability):

## Pilot measurement

- Hardware and software versions:

- Warm-up and compilation excluded from steady-state throughput?:

- Sustained tokens/second:

- Peak accelerator memory:

- Validation loss trend:

- Checkpoint size and duration:

- Estimate (tokens, wall time, cost) plus uncertainty:

## Run manifest

- Git commit:

- Config file/hash:

- Dataset and tokenizer hash:

- Random seed(s):

- Start time / end time:

- Tokens trained / optimizer steps:

- Checkpoint IDs:

- Training and validation metrics (loss, gradient norm, effective LR, tokens/s, memory):

- Clipped, skipped, or failed optimizer steps:

- Exceptions, restarts, and decisions:

## Evaluation and release gate

- Held-out loss/perplexity protocol:

- Bits per byte (BPB), if comparing across tokenizers or vocabulary sizes:

- Metric comparability: do all runs being compared share a tokenizer?:

- Domain-specific evaluation and baseline:

- Generation prompts/settings and saved outputs:

- Leakage/contamination checks:

- Privacy/memorization and safety tests:

- Known limitations:

- Release decision and rationale:

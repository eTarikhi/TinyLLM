# Data pipeline: auditing a corpus and designing a tokenizer

The data decisions are made once and constrain everything downstream, so they are written down before a single training step runs. `SKILL.md` carries the headline rules; this file is the full checklist, and [`../scripts/tiny-gpt/src/data.py`](../scripts/tiny-gpt/src/data.py) is an implementation of it — it checksums each source, strips boilerplate, scans for sensitive patterns, measures duplication, splits per document, and writes a manifest recording all of it.

## Choose and audit the data before training

1. Prefer user-owned, licensed, public-domain or explicitly permitted data. Downloadable is not the same as permitted, and raw text is often not redistributable even when trained weights are.
2. Record source URL, dataset version, licence, access date, language, intended use, known restrictions, and the transformation code — a checksum per source makes the record verifiable.
3. Remove exact and, where useful, near duplicates; detect corrupted records, boilerplate, spam and machine-generated junk. Keep the raw source immutable and write a separate cleaned artifact.
4. Filter personal information, secrets and credentials, and anything the project has no right to use.
5. Split at the document or source level into train, validation and test *before* forming overlapping windows, so no window straddles a boundary. Check benchmark leakage for task-style evaluation.
6. Read a random sample of cleaned records yourself. Quantify language, domain, length, duplicate and filtering rates; keep a data card describing limitations.
7. For large corpora stream or shard rather than downloading blindly, and estimate storage for raw text, cleaned text, token IDs, checkpoints and intermediates.

## Tokenization

- Choose the type from the corpus and use case: byte or character for a first toy, BPE/WordPiece/Unigram for realistic language modeling.
- Fit the tokenizer on training data only — never on validation or test text — and record vocabulary size and training configuration in the manifest.
- Measure tokens per character or byte, especially for non-English text, code, numbers, emoji and domain terms; it sets how much text a token budget buys.
- Define stable special tokens (BOS/EOS, and later distinct chat-role tokens) and never change token IDs after training starts.
- Preserve document boundaries: concatenating unrelated documents without an EOS teaches the model transitions that do not exist.
- Save tokenizer files beside each checkpoint; tokenizer/model vocabulary mismatch is a release blocker.

## What the implementation records

Every run of the bundled pipeline writes a `manifest.json` next to the token files containing: corpus name, per-document source URL, licence, access date and SHA-256; the split policy and the exact boundaries used; token and byte counts per split; the tokenizer kind, vocabulary size, file hash and measured tokens-per-character on validation; and the sensitive-pattern and duplicate-block audit. A training run that cannot point at such a record is not reproducible, whatever its loss curve says.

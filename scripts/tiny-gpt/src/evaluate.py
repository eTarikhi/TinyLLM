"""Held-out evaluation with baselines, bits per byte, and a memorization probe.

"Prefer evidence to vibes" (SKILL.md). A loss number alone says nothing unless
it is next to the trivial baselines any model must beat:

* uniform    -- log2(V) bits per token, the no-knowledge model;
* unigram    -- token frequencies from the training split;
* bigram     -- add-k smoothed counts from the training split;
* untrained  -- the same architecture at random initialisation.

All of them are reported in bits per byte as well as per-token loss, because
per-token numbers cannot be compared across tokenizers with different
vocabularies, and comparing tokenizers is one of the experiments here.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import TokenBatcher, load_manifest, load_tokenizer  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402

LN2 = math.log(2.0)


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------


def count_baselines(train_ids: np.ndarray, vocab_size: int, add_k: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Unigram log-probabilities and bigram conditional log-probabilities."""
    unigram = np.bincount(train_ids, minlength=vocab_size).astype(np.float64) + add_k
    unigram /= unigram.sum()
    log_unigram = np.log(unigram)

    pairs = np.zeros((vocab_size, vocab_size), dtype=np.float64)
    np.add.at(pairs, (train_ids[:-1], train_ids[1:]), 1.0)
    pairs += add_k
    log_bigram = np.log(pairs / pairs.sum(axis=1, keepdims=True))
    return log_unigram, log_bigram


def score_ngram_baselines(
    eval_ids: np.ndarray,
    log_unigram: np.ndarray,
    log_bigram: np.ndarray,
    byte_lens: np.ndarray,
    vocab_size: int,
) -> dict[str, dict[str, float]]:
    targets = eval_ids[1:]
    context = eval_ids[:-1]
    blen = byte_lens[targets]
    mask = blen > 0
    n_tokens = int(mask.sum())
    n_bytes = int(blen.sum())

    def pack(total_nats: float) -> dict[str, float]:
        loss = total_nats / n_tokens
        return {
            "loss": loss,
            "perplexity": math.exp(min(loss, 20.0)),
            "bpb": total_nats / (LN2 * n_bytes),
            "tokens": n_tokens,
            "bytes": n_bytes,
        }

    uniform_nats = float(n_tokens * math.log(vocab_size))
    unigram_nats = float(-(log_unigram[targets] * mask).sum())
    bigram_nats = float(-(log_bigram[context, targets] * mask).sum())
    return {
        "uniform": pack(uniform_nats),
        "unigram": pack(unigram_nats),
        "bigram": pack(bigram_nats),
    }


# --------------------------------------------------------------------------
# Model evaluation
# --------------------------------------------------------------------------


@torch.no_grad()
def evaluate_model(model: GPT, batcher: TokenBatcher, byte_lens: torch.Tensor, max_tokens: int | None = None) -> dict:
    model.eval()
    total_nats = 0.0
    total_tokens = 0
    total_bytes = 0
    for x, y in batcher.iter_eval_batches(max_tokens=max_tokens):
        nll = model.token_nll(x, y)
        blen = byte_lens[y]
        mask = blen > 0
        total_nats += float((nll * mask).sum())
        total_bytes += int(blen.sum())
        total_tokens += int(mask.sum())
    loss = total_nats / total_tokens
    return {
        "loss": loss,
        "perplexity": math.exp(min(loss, 20.0)),
        "bpb": total_nats / (LN2 * total_bytes),
        "tokens": total_tokens,
        "bytes": total_bytes,
    }


# --------------------------------------------------------------------------
# Per-document breakdown
# --------------------------------------------------------------------------


def evaluate_per_document(
    model: GPT,
    split_path: Path,
    eos_id: int,
    byte_lens: torch.Tensor,
    block_size: int,
    doc_records: list[dict] | None = None,
    batch_size: int = 16,
) -> list[dict]:
    """Bits per byte for each document in a multi-document split.

    A single aggregate number averages over whatever mixture the corpus
    happens to be, and that average can hide a large effect in either
    direction. Measured here: adding 44 books to a 27-book corpus moved the
    aggregate test figure by 0.35% while moving the added books' own figure by
    10.47%. Nothing in the aggregate said so.

    The split is the per-document pieces joined by a single EOS, and EOS is
    only ever emitted as that separator, so splitting on it recovers the
    documents exactly and in corpus order. Pieces shorter than one block are
    reported with ``"scored": false`` rather than silently dropped or padded.
    """
    ids = np.fromfile(split_path, dtype=np.uint16)
    cuts = np.flatnonzero(ids == eos_id)
    pieces, prev = [], 0
    for c in cuts:
        pieces.append(ids[prev:c])
        prev = int(c) + 1
    pieces.append(ids[prev:])

    out: list[dict] = []
    for i, piece in enumerate(pieces):
        rec = (doc_records[i] if doc_records and i < len(doc_records) else {})
        entry = {
            "index": i,
            "doc_id": rec.get("doc_id", f"doc{i}"),
            "title": rec.get("title", ""),
            "author": rec.get("author", ""),
            "tokens": int(piece.size),
        }
        if piece.size < block_size + 1:
            entry.update({"scored": False, "bpb": None, "bytes": 0,
                          "reason": f"piece is shorter than block_size+1={block_size + 1}"})
            out.append(entry)
            continue
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as fh:
            tmp = Path(fh.name)
        try:
            piece.tofile(tmp)
            res = evaluate_model(model, TokenBatcher(tmp, block_size, batch_size, seed=0), byte_lens)
        finally:
            tmp.unlink(missing_ok=True)
        entry.update({"scored": True, "bpb": res["bpb"], "bytes": res["bytes"],
                      "loss": res["loss"]})
        out.append(entry)
    return out


def summarise_per_document(rows: list[dict]) -> dict:
    scored = [r for r in rows if r["scored"]]
    if not scored:
        return {"documents": len(rows), "scored": 0}
    bpbs = [r["bpb"] for r in scored]
    best = min(scored, key=lambda r: r["bpb"])
    worst = max(scored, key=lambda r: r["bpb"])
    mean = sum(bpbs) / len(bpbs)
    return {
        "documents": len(rows),
        "scored": len(scored),
        "unscored": len(rows) - len(scored),
        "mean_bpb": mean,
        "min_bpb": best["bpb"],
        "max_bpb": worst["bpb"],
        "range_bpb": worst["bpb"] - best["bpb"],
        "range_pct_of_mean": 100 * (worst["bpb"] - best["bpb"]) / mean,
        "stdev_bpb": statistics.stdev(bpbs) if len(bpbs) > 1 else 0.0,
        "easiest": {"doc_id": best["doc_id"], "title": best["title"], "bpb": best["bpb"]},
        "hardest": {"doc_id": worst["doc_id"], "title": worst["title"], "bpb": worst["bpb"]},
    }


# --------------------------------------------------------------------------
# Memorization probe
# --------------------------------------------------------------------------


def longest_common_substring_length(sample: str, corpus: str, max_len: int = 400) -> int:
    """Longest substring of ``sample`` that appears verbatim in ``corpus``.

    Greedy extension with ``str.find``: walk the sample, and from each position
    only try to beat the current record. The obvious implementation -- a set of
    every corpus n-gram, binary-searched on length -- costs O(len(corpus)) live
    strings per probe step, which is ~1.7 GB on a 17 MB corpus and dies on a
    small box. This version allocates nothing beyond the slices it tests and
    leans on CPython's C substring search.

    This is a memorization probe, not a privacy guarantee: a tiny model trained
    for several epochs on one corpus *will* reproduce short phrases, and the
    point is to measure how long they get.
    """
    best = 0
    limit = min(len(sample), max_len)
    for i in range(len(sample)):
        if i + best >= len(sample):
            break  # no room left to beat the record
        length = best + 1
        while i + length <= len(sample) and length <= limit:
            if corpus.find(sample[i : i + length]) == -1:
                break
            best = length
            length += 1
    return best


def load_train_text(data_dir: Path, manifest: dict) -> str:
    """Reconstruct exactly the text the model trained on, for the probe.

    Handles both corpus layouts. The single-document cache is one file; a
    multi-document corpus caches one file per document, and the probe has to
    apply the same per-document split prepare() applied -- otherwise it would
    compare generations against held-out text the model never saw, and report
    a reassuring zero.
    """
    raw = data_dir.parent / "raw"
    name = manifest["corpus"]["name"]
    bounds = manifest["split_policy"]["boundaries_chars"]
    records = manifest["corpus"].get("document_records") or []

    single = raw / f"{name}.txt"
    if single.exists():
        cut = bounds[0] if isinstance(bounds[0], int) else bounds[0][0]
        return single.read_text(encoding="utf-8")[:cut]

    pieces = []
    for i, rec in enumerate(records):
        path = raw / name / f"{rec['doc_id']}.txt"
        if not path.exists():
            return ""
        doc = path.read_text(encoding="utf-8")
        cut = bounds[i][0] if i < len(bounds) and isinstance(bounds[i], list) else int(len(doc) * 0.9)
        pieces.append(doc[:cut])
    return "\n\n".join(pieces)


def memorization_probe(model: GPT, tokenizer, corpus_text: str, n_samples: int = 4, tokens: int = 240, seed: int = 0) -> dict:
    lengths = []
    samples = []
    for i in range(n_samples):
        g = torch.Generator().manual_seed(seed + i)
        start = torch.tensor([[tokenizer.encode("\n")[0]]], dtype=torch.long)
        out = model.generate(start, max_new_tokens=tokens, temperature=0.8, top_k=40, generator=g)
        text = tokenizer.decode(out[0].tolist())
        samples.append(text)
        lengths.append(longest_common_substring_length(text, corpus_text))
    return {
        "samples": n_samples,
        "tokens_per_sample": tokens,
        "longest_verbatim_match_chars": lengths,
        "max_longest_verbatim_match_chars": max(lengths),
        "mean_longest_verbatim_match_chars": sum(lengths) / len(lengths),
        "example_sample": samples[0],
    }


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def build_report(results: dict) -> str:
    lines = [
        f"# Evaluation report — {results['run_id']}",
        "",
        f"Checkpoint: `{results['checkpoint']}` (step {results['step']}, {results['tokens_trained']:,} training tokens)  ",
        f"Data: `{results['data_dir']}` — {results['tokenizer']['kind']} tokenizer, vocab {results['tokenizer']['vocab_size']}  ",
        f"Split evaluated: **{results['split']}** ({results['model']['tokens']:,} tokens / {results['model']['bytes']:,} bytes, full split, non-overlapping windows)",
        "",
        "## Held-out language modelling",
        "",
        "| Model | Loss (nats/token) | Perplexity | Bits per byte |",
        "| --- | --- | --- | --- |",
    ]
    order = [("uniform", "Uniform over vocabulary"), ("unigram", "Unigram counts (train split)"),
             ("bigram", "Bigram counts, add-0.1 (train split)"), ("untrained", "This architecture, random init"),
             ("model", "**Trained model**")]
    for key, label in order:
        r = results.get(key)
        if not r:
            continue
        lines.append(f"| {label} | {r['loss']:.4f} | {r['perplexity']:.2f} | {r['bpb']:.4f} |")
    m, b = results["model"], results["bigram"]
    lines += [
        "",
        f"The trained model is **{b['bpb'] / m['bpb']:.2f}×** better than the bigram baseline in bits per byte "
        f"and **{results['uniform']['bpb'] / m['bpb']:.2f}×** better than uniform.",
        "",
        "## Memorization probe",
        "",
    ]
    pd = results.get("per_document_summary")
    if pd and pd.get("scored"):
        rows = [r for r in results["per_document"] if r["scored"]]
        rows.sort(key=lambda r: r["bpb"])
        lines += [
            "",
            "## Per-document breakdown",
            "",
            f"{pd['scored']} of {pd['documents']} documents scored. "
            f"Mean {pd['mean_bpb']:.4f} bits per byte, spread "
            f"{pd['range_bpb']:.4f} ({pd['range_pct_of_mean']:.1f}% of the mean), "
            f"sd {pd['stdev_bpb']:.4f}.",
            "",
            "An aggregate figure is a weighted average over whatever mixture the corpus is; "
            "the spread below is how much that average is hiding.",
            "",
            "| | Document | Author | Tokens | Bits per byte |",
            "| --- | --- | --- | ---: | ---: |",
        ]
        shown = rows[:3] + ([{"doc_id": "...", "title": "…", "author": "", "tokens": 0, "bpb": None}]
                            if len(rows) > 6 else []) + rows[-3:] if len(rows) > 6 else rows
        for r in shown:
            if r["bpb"] is None:
                lines.append("| | … | | | |")
                continue
            tag = "easiest" if r is rows[0] else ("hardest" if r is rows[-1] else "")
            lines.append(f"| {tag} | {r['title'] or r['doc_id']} | {r['author']} | "
                         f"{r['tokens']:,} | {r['bpb']:.4f} |")

    mem = results["memorization"]
    if not mem.get("samples"):
        # A probe that silently measured nothing must not print "0 characters",
        # which reads like a clean result instead of a missing measurement.
        lines += [
            "**Not measured.** " + mem.get("reason", "no training text was available")
            + " — this is a missing measurement, not a clean result.",
        ]
    else:
        lines += [
            f"- Samples: {mem['samples']} × {mem['tokens_per_sample']} tokens, temperature 0.8, top-k 40",
            f"- Longest verbatim match against the training split: **{mem['max_longest_verbatim_match_chars']} characters** "
            f"(mean {mem['mean_longest_verbatim_match_chars']:.1f})",
            "",
            "A short verbatim match is expected for a small model trained for multiple epochs on one corpus; this corpus is public-domain text,",
            "so the probe is a calibration of memorization behaviour rather than a privacy finding.",
            "",
            "## Sample",
            "",
            "```text",
            mem["example_sample"][:800].rstrip(),
            "```",
        ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a checkpoint against baselines.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-dir", default=None, help="defaults to the data dir recorded in the checkpoint")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--per-document", action="store_true",
                    help="also score every document in the split separately; an aggregate "
                         "bits-per-byte figure averages over the corpus mixture and can hide "
                         "a large effect on one part of it")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    data_dir = Path(args.data_dir or ckpt.get("data_dir", "data/prepared-char"))
    tokenizer = load_tokenizer(data_dir)
    manifest = load_manifest(data_dir)
    byte_lens_np = np.array(tokenizer.token_byte_lengths(), dtype=np.int64)
    byte_lens = torch.tensor(byte_lens_np)

    model_cfg = GPTConfig(**ckpt["model_config"])
    model = GPT(model_cfg)
    model.load_state_dict(ckpt["model_state"])

    batcher = TokenBatcher(data_dir / f"{args.split}.bin", model_cfg.block_size, args.batch_size, seed=0)
    model_result = evaluate_model(model, batcher, byte_lens, max_tokens=args.max_tokens)

    torch.manual_seed(0)
    untrained = GPT(model_cfg)
    untrained_result = evaluate_model(untrained, batcher, byte_lens, max_tokens=args.max_tokens)

    train_ids = np.fromfile(data_dir / "train.bin", dtype=np.uint16).astype(np.int64)
    eval_ids = np.fromfile(data_dir / f"{args.split}.bin", dtype=np.uint16).astype(np.int64)
    if args.max_tokens:
        eval_ids = eval_ids[: args.max_tokens + 1]
    log_unigram, log_bigram = count_baselines(train_ids, model_cfg.vocab_size)
    baselines = score_ngram_baselines(eval_ids, log_unigram, log_bigram, byte_lens_np, model_cfg.vocab_size)

    train_text = load_train_text(Path(data_dir), manifest)
    probe = memorization_probe(model, tokenizer, train_text) if train_text else {
        "samples": 0, "tokens_per_sample": 0, "longest_verbatim_match_chars": [],
        "max_longest_verbatim_match_chars": None, "mean_longest_verbatim_match_chars": None,
        "example_sample": "", "measured": False,
        "reason": "the training text could not be reconstructed from data/raw",
    }

    results = {
        "run_id": ckpt.get("train_config", {}).get("run_id", "unknown"),
        "checkpoint": str(args.checkpoint),
        "step": ckpt.get("step"),
        "tokens_trained": ckpt.get("tokens", 0),
        "data_dir": str(data_dir),
        "split": args.split,
        "tokenizer": {"kind": manifest["tokenizer"]["kind"], "vocab_size": manifest["tokenizer"]["vocab_size"]},
        "model_config": ckpt["model_config"],
        "params_total": model.num_params(),
        "model": model_result,
        "untrained": untrained_result,
        **baselines,
        "memorization": probe,
    }

    if args.per_document:
        doc_records = manifest.get("corpus", {}).get("document_records") or None
        rows = evaluate_per_document(
            model, data_dir / f"{args.split}.bin", tokenizer.eos_id, byte_lens,
            model_cfg.block_size, doc_records, batch_size=args.batch_size,
        )
        results["per_document"] = rows
        results["per_document_summary"] = summarise_per_document(rows)

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.checkpoint).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"eval-{args.split}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    report = build_report(results)
    (out_dir / f"eval-{args.split}.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()

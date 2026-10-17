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
import sys
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
# Memorization probe
# --------------------------------------------------------------------------


def longest_common_substring_length(sample: str, corpus: str, max_len: int = 400) -> int:
    """Longest substring of ``sample`` that appears verbatim in ``corpus``.

    Binary search on length with a hash set of corpus n-grams. This is a
    memorization probe, not a privacy guarantee: a tiny model trained for
    several epochs on one book *will* reproduce short phrases, and the point is
    to measure how long they get.
    """
    lo, hi, best = 1, min(len(sample), max_len), 0
    while lo <= hi:
        mid = (lo + hi) // 2
        grams = {corpus[i : i + mid] for i in range(0, len(corpus) - mid + 1)}
        if any(sample[i : i + mid] in grams for i in range(0, len(sample) - mid + 1)):
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


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
        f"- Samples: {results['memorization']['samples']} × {results['memorization']['tokens_per_sample']} tokens, temperature 0.8, top-k 40",
        f"- Longest verbatim match against the training split: **{results['memorization']['max_longest_verbatim_match_chars']} characters** "
        f"(mean {results['memorization']['mean_longest_verbatim_match_chars']:.1f})",
        "",
        "A short verbatim match is expected for a small model trained for multiple epochs on one corpus; this corpus is public-domain text,",
        "so the probe is a calibration of memorization behaviour rather than a privacy finding.",
        "",
        "## Sample",
        "",
        "```text",
        results["memorization"]["example_sample"][:800].rstrip(),
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

    corpus_text = (Path(data_dir).parent / "raw" / f"{manifest['corpus']['name']}.txt")
    train_text = corpus_text.read_text(encoding="utf-8")[: manifest["split_policy"]["boundaries_chars"][0]] if corpus_text.exists() else ""
    probe = memorization_probe(model, tokenizer, train_text) if train_text else {
        "samples": 0, "tokens_per_sample": 0, "longest_verbatim_match_chars": [],
        "max_longest_verbatim_match_chars": 0, "mean_longest_verbatim_match_chars": 0.0,
        "example_sample": "",
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

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.checkpoint).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"eval-{args.split}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    report = build_report(results)
    (out_dir / f"eval-{args.split}.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()

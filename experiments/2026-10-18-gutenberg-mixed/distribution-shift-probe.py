#!/usr/bin/env python3
"""Quantify how far the 44 added books sit from the frozen evaluation distribution.

The data-scaling run (``data-gb71``) trains on 3.53x more unique text and ends
up *worse* on the held-out test split.  There are two competing explanations:

1. a 1.77M-parameter model is capacity-limited, not data-limited, so extra
   text cannot help; or
2. the extra text is drawn from a different distribution than val/test, so it
   actively competes for capacity with the material being scored.

Those are distinguishable without training anything.  Any model trained on the 27-book corpus never saw the 44 additions, so its
bits per byte on a sample of them is a clean out-of-distribution reading,
directly comparable to its bits per byte on the 27-book test split -- same
model, same tokenizer, same units.  ``seed-1338`` and ``seed-1339`` are used
for that: they are the released configuration with only the seed changed, and
using two of them puts a seed bar on the probe itself.  (The released
``main-gutenberg-4800`` checkpoint is not retained in the repository -- weights
are gitignored -- and those two runs are the same configuration.)

A large gap means the additions really are a different distribution, and
explanation 2 carries weight.  A small gap means they are "more of the same"
and explanation 1 carries weight.

The same measurement for ``data-gb71`` is also reported, but it is *not* clean:
that run trained on all 44 books, so its number on the new-book probe is a
train-set fit, an optimistic bound, and is labelled as such.

Usage (from scripts/tiny-gpt):
    python ../../experiments/2026-10-18-gutenberg-mixed/distribution-shift-probe.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PKG = HERE.parent.parent / "scripts" / "tiny-gpt"
sys.path.insert(0, str(PKG / "src"))

from data import CORPORA, TokenBatcher, load_tokenizer  # noqa: E402
from evaluate import evaluate_model  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402

REF_DIR = PKG / "data" / "prepared-gb-bpe2048"
RAW_71 = PKG / "data" / "raw" / "gutenberg-mixed-71"
OUT = HERE / "distribution-shift.json"

# Match the test split's size so the two readings cover comparable amounts of
# text.  Spread evenly over the 44 books rather than concatenating whole books,
# so no single long novel dominates the number.
TARGET_CHARS_PER_DOC = 24_000


def new_book_sample() -> tuple[str, list[str]]:
    """A deterministic slice from the middle of every train-only document."""
    corpus = CORPORA["gutenberg-mixed-71"]
    new_docs = [d for d in corpus.documents if d.train_only]
    pieces, used = [], []
    for doc in new_docs:
        body = (RAW_71 / f"{doc.doc_id}.txt").read_text(encoding="utf-8")
        if len(body) < TARGET_CHARS_PER_DOC:
            chunk = body
        else:
            # Middle of the book: past the front matter, before the appendices.
            start = (len(body) - TARGET_CHARS_PER_DOC) // 2
            chunk = body[start : start + TARGET_CHARS_PER_DOC]
        pieces.append(chunk)
        used.append(doc.doc_id)
    return "\n\n".join(pieces), used


def bpb_on(model: GPT, cfg: GPTConfig, path: Path, byte_lens: torch.Tensor) -> dict:
    batcher = TokenBatcher(path, cfg.block_size, 16, seed=0)
    return evaluate_model(model, batcher, byte_lens)


def load(run_id: str) -> tuple[GPT, GPTConfig]:
    ckpt = torch.load(PKG / "artifacts" / "runs" / run_id / "best.pt", map_location="cpu", weights_only=False)
    cfg = GPTConfig(**ckpt["model_config"])
    model = GPT(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg


def main() -> None:
    torch.set_num_threads(2)
    tokenizer = load_tokenizer(REF_DIR)
    byte_lens_np = np.array(tokenizer.token_byte_lengths(), dtype=np.int64)
    byte_lens = torch.tensor(byte_lens_np)

    text, used = new_book_sample()
    ids = np.asarray(tokenizer.encode(text), dtype=np.uint16)
    probe_path = PKG / "data" / "probe-new-books.bin"
    ids.tofile(probe_path)
    print(f"new-book probe: {len(used)} docs, {len(text):,} chars, {len(ids):,} tokens -> {probe_path}")

    test_path = REF_DIR / "test.bin"
    print(f"old test split: {test_path.stat().st_size // 2:,} tokens")

    results: dict[str, dict] = {}
    for run_id, clean_on_new in (("seed-1338", True), ("seed-1339", True), ("data-gb71", False)):
        model, cfg = load(run_id)
        old = bpb_on(model, cfg, test_path, byte_lens)
        new = bpb_on(model, cfg, probe_path, byte_lens)
        results[run_id] = {
            "test_27_books_bpb": old["bpb"],
            "new_44_books_bpb": new["bpb"],
            "shift_pct": (new["bpb"] - old["bpb"]) / old["bpb"] * 100.0,
            "new_books_held_out_for_this_model": clean_on_new,
            "test_bytes": old["bytes"],
            "new_bytes": new["bytes"],
        }
        tag = "held out" if clean_on_new else "TRAINED ON (optimistic)"
        print(
            f"{run_id:20s} test(27) {old['bpb']:.4f}  new(44) {new['bpb']:.4f} [{tag}]  "
            f"shift {results[run_id]['shift_pct']:+.2f}%"
        )

    clean = [r for r in results.values() if r["new_books_held_out_for_this_model"]]
    ref = {
        "shift_pct": sum(r["shift_pct"] for r in clean) / len(clean),
        "new_44_books_bpb": sum(r["new_44_books_bpb"] for r in clean) / len(clean),
        "test_27_books_bpb": sum(r["test_27_books_bpb"] for r in clean) / len(clean),
    }
    payload_ref = ref
    payload = {
        "probe": {
            "docs": len(used),
            "doc_ids": used,
            "chars_per_doc_target": TARGET_CHARS_PER_DOC,
            "chars": len(text),
            "tokens": int(len(ids)),
            "tokenizer": "shared byte-BPE 2048 (byte-identical across both runs)",
        },
        "results": results,
        "clean_mean": ref,
        "reading": (
            "The two 27-book models never saw the 44 additions. Their mean bits per byte on them "
            f"is {ref['shift_pct']:+.2f}% relative to their own score on the 27-book test split, "
            "which is the size of the distribution gap the extra data asks the model to cover."
        ),
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT}")
    print(payload["reading"])


if __name__ == "__main__":
    main()

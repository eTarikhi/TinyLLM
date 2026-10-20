"""Shared fixtures for the test suite: a tiny model and a synthetic corpus.

Every test builds something small enough that the whole suite runs on a laptop
CPU in under a minute. A test suite nobody runs is not a gate.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from model import GPT, GPTConfig  # noqa: E402
from tokenizer import CharTokenizer  # noqa: E402

TINY_TEXT = (
    "to be or not to be that is the question\n"
    "whether tis nobler in the mind to suffer\n"
    "the slings and arrows of outrageous fortune\n"
) * 40


def tiny_config(vocab_size: int = 32, **kwargs) -> GPTConfig:
    base = dict(vocab_size=vocab_size, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0)
    base.update(kwargs)
    return GPTConfig(**base)


def tiny_model(seed: int = 0, **kwargs) -> GPT:
    torch.manual_seed(seed)
    return GPT(tiny_config(**kwargs))


def random_batch(cfg: GPTConfig, batch_size: int = 4, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, cfg.vocab_size, (batch_size, cfg.block_size), generator=g)
    y = torch.randint(0, cfg.vocab_size, (batch_size, cfg.block_size), generator=g)
    return x, y


def make_prepared_dir(tmpdir: str | Path, block_size: int = 16) -> Path:
    """Write a miniature prepared-data directory (tokenizer + .bin splits)."""
    tmpdir = Path(tmpdir)
    tmpdir.mkdir(parents=True, exist_ok=True)
    tok = CharTokenizer.train(TINY_TEXT)
    tok.save(tmpdir / "tokenizer.json")
    ids = np.array(tok.encode(TINY_TEXT), dtype=np.uint16)
    n = len(ids)
    splits = {"train": ids[: int(n * 0.8)], "val": ids[int(n * 0.8) : int(n * 0.9)], "test": ids[int(n * 0.9) :]}
    manifest = {
        "corpus": {"name": "synthetic"},
        "tokenizer": {"kind": "char", "vocab_size": tok.vocab_size, "sha256": "n/a"},
        "splits": {k: {"tokens": int(v.size)} for k, v in splits.items()},
        "split_policy": {"boundaries_chars": [0, 0]},
    }
    for name, arr in splits.items():
        arr.tofile(tmpdir / f"{name}.bin")
    (tmpdir / "manifest.json").write_text(__import__("json").dumps(manifest), encoding="utf-8")
    return tmpdir


class TempDir:
    def __enter__(self) -> Path:
        self._tmp = tempfile.TemporaryDirectory()
        return Path(self._tmp.name)

    def __exit__(self, *exc) -> None:
        self._tmp.cleanup()

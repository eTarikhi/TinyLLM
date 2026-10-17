"""Corpus acquisition, auditing, splitting, tokenization and batching.

Everything here exists to satisfy one rule from SKILL.md: the data pipeline is
part of the deliverable, and provenance is recorded before a single gradient
step. ``prepare`` writes ``manifest.json`` next to the token files containing
the source URL, checksum, license, access date, split boundaries, duplicate and
sensitive-data scan results, and the tokenizer measurement -- everything the
experiment-plan template asks for under "Data card".

Split policy: contiguous, document-order. The corpus here is a single
concatenated text, so a shuffled split would put adjacent, overlapping context
windows on both sides of the train/validation boundary and leak. Contiguous
splitting trades that leak for a mild distribution shift between splits, which
is the right trade and is recorded in the manifest.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import subprocess
import tarfile
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from tokenizer import BaseTokenizer, build_tokenizer

# --------------------------------------------------------------------------
# Corpus registry. Provenance is data, not a comment.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Corpus:
    name: str
    description: str
    sha256: str
    urls: Sequence[str]
    member: str | None  # path inside the archive, if the URL is an archive
    text_license: str
    file_license: str
    restrictions: str
    access_date: str


TINY_SHAKESPEARE = Corpus(
    name="tiny-shakespeare",
    description=(
        "~1.1 MB of concatenated Shakespeare play text, the standard character-level "
        "language-modelling toy corpus introduced by karpathy/char-rnn."
    ),
    sha256="86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed",
    urls=(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        "https://codeload.github.com/karpathy/char-rnn/tar.gz/refs/heads/master",
    ),
    member="char-rnn-master/data/tinyshakespeare/input.txt",
    text_license="Public domain (William Shakespeare, d. 1616); no copyright subsists in the text.",
    file_license="Distributed in karpathy/char-rnn, which is MIT licensed.",
    restrictions=(
        "None known for research or educational use. Early modern English only; "
        "not representative of contemporary text, and not suitable for evaluating "
        "general language ability."
    ),
    access_date="2026-10-17",
)

CORPORA = {TINY_SHAKESPEARE.name: TINY_SHAKESPEARE}


# --------------------------------------------------------------------------
# Acquisition
# --------------------------------------------------------------------------


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def fetch_corpus(corpus: Corpus, cache_dir: str | Path, local_file: str | Path | None = None) -> Path:
    """Download (or copy) the corpus and verify its checksum before use."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"{corpus.name}.txt"

    if target.exists() and sha256_bytes(target.read_bytes()) == corpus.sha256:
        return target

    payload: bytes | None = None
    if local_file is not None:
        payload = Path(local_file).read_bytes()
    else:
        errors = []
        for url in corpus.urls:
            try:
                with urllib.request.urlopen(url, timeout=60) as resp:
                    blob = resp.read()
                if url.endswith((".tar.gz", "tar.gz/refs/heads/master")) or "codeload" in url:
                    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
                        member = tar.extractfile(corpus.member)  # type: ignore[arg-type]
                        if member is None:
                            raise RuntimeError(f"{corpus.member} missing from archive at {url}")
                        blob = member.read()
                payload = blob
                break
            except Exception as exc:  # pragma: no cover - network dependent
                errors.append(f"{url}: {exc}")
        if payload is None:
            raise RuntimeError("could not fetch corpus:\n  " + "\n  ".join(errors))

    digest = sha256_bytes(payload)
    if digest != corpus.sha256:
        raise RuntimeError(
            f"checksum mismatch for {corpus.name}: expected {corpus.sha256}, got {digest}. "
            "Refusing to train on an unverified corpus."
        )
    target.write_bytes(payload)
    return target


# --------------------------------------------------------------------------
# Auditing
# --------------------------------------------------------------------------

PII_PATTERNS = {
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    "phone_like": re.compile(r"(?<!\d)(?:\+\d{1,3}[ -]?)?(?:\(\d{2,4}\)[ -]?)?\d{3}[ -]\d{4}(?!\d)"),
    "ipv4": re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)"),
    "url": re.compile(r"https?://\S+"),
    "aws_key_like": re.compile(r"AKIA[0-9A-Z]{16}"),
    "private_key_header": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
}


def scan_sensitive(text: str) -> dict[str, int]:
    """Count obvious personal data / secret patterns. Cheap, and it is a gate.

    A clean result is not proof of safety; it is proof that the easy cases were
    checked, which is the claim the model card is allowed to make.
    """
    return {name: len(pattern.findall(text)) for name, pattern in PII_PATTERNS.items()}


def duplicate_stats(text: str) -> dict[str, float | int]:
    """Exact-duplicate rate over blank-line separated blocks."""
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    unique = set(blocks)
    dup_chars = sum(len(b) for b in blocks) - sum(len(b) for b in unique)
    return {
        "blocks": len(blocks),
        "unique_blocks": len(unique),
        "duplicate_blocks": len(blocks) - len(unique),
        "duplicate_block_fraction": (len(blocks) - len(unique)) / max(len(blocks), 1),
        "duplicate_chars": dup_chars,
    }


def git_revision(repo_root: str | Path) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except Exception:  # pragma: no cover
        return "unknown"


# --------------------------------------------------------------------------
# Preparation
# --------------------------------------------------------------------------


@dataclass
class PreparedData:
    out_dir: Path
    manifest: dict
    tokenizer_path: Path
    split_paths: dict[str, Path] = field(default_factory=dict)


def prepare(
    corpus_name: str = "tiny-shakespeare",
    tokenizer_kind: str = "char",
    vocab_size: int = 512,
    out_dir: str | Path = "data/prepared",
    cache_dir: str | Path = "data/raw",
    splits: tuple[float, float, float] = (0.90, 0.05, 0.05),
    local_file: str | Path | None = None,
    repo_root: str | Path = ".",
) -> PreparedData:
    if abs(sum(splits) - 1.0) > 1e-9:
        raise ValueError(f"splits must sum to 1.0, got {splits}")
    corpus = CORPORA[corpus_name]
    raw_path = fetch_corpus(corpus, cache_dir, local_file=local_file)
    text = raw_path.read_text(encoding="utf-8")

    n = len(text)
    n_train = int(n * splits[0])
    n_val = int(n * splits[1])
    # Snap boundaries to the next blank line so a split never starts mid-speech.
    def snap(i: int) -> int:
        j = text.find("\n\n", i)
        return (j + 2) if j != -1 else i

    i_train, i_val = snap(n_train), snap(n_train + n_val)
    parts = {
        "train": text[:i_train],
        "val": text[i_train:i_val],
        "test": text[i_val:],
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Tokenizer is trained on the training split only.
    tokenizer = build_tokenizer(tokenizer_kind, parts["train"], vocab_size=vocab_size)
    tokenizer_path = out_dir / "tokenizer.json"
    tokenizer.save(tokenizer_path)

    dtype = np.uint16 if tokenizer.vocab_size <= np.iinfo(np.uint16).max else np.uint32
    split_paths: dict[str, Path] = {}
    split_stats: dict[str, dict] = {}
    for name, part in parts.items():
        ids = np.array(tokenizer.encode(part), dtype=dtype)
        path = out_dir / f"{name}.bin"
        ids.tofile(path)
        split_paths[name] = path
        split_stats[name] = {
            "chars": len(part),
            "bytes": len(part.encode("utf-8")),
            "tokens": int(ids.size),
            "tokens_per_char": float(ids.size / max(len(part), 1)),
            "sha256_text": sha256_bytes(part.encode("utf-8")),
            "sha256_tokens": sha256_bytes(ids.tobytes()),
            "file": str(path.name),
            "dtype": np.dtype(dtype).name,
        }

    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "code_revision": git_revision(repo_root),
        "corpus": {
            "name": corpus.name,
            "description": corpus.description,
            "source_urls": list(corpus.urls),
            "member": corpus.member,
            "access_date": corpus.access_date,
            "sha256": corpus.sha256,
            "text_license": corpus.text_license,
            "file_license": corpus.file_license,
            "restrictions": corpus.restrictions,
            "chars": n,
            "bytes": len(text.encode("utf-8")),
        },
        "split_policy": {
            "method": "contiguous, document order, boundaries snapped to a blank line",
            "fractions": list(splits),
            "rationale": (
                "A shuffled split of one continuous text leaks overlapping context windows "
                "across the boundary. Contiguous splitting avoids the leak and accepts a mild "
                "distribution shift between splits."
            ),
            "boundaries_chars": [i_train, i_val],
        },
        "splits": split_stats,
        "tokenizer": {
            "kind": tokenizer_kind,
            "vocab_size": tokenizer.vocab_size,
            "trained_on": "train split only",
            "requested_vocab_size": vocab_size if tokenizer_kind != "char" else None,
            "bos_id": tokenizer.bos_id,
            "eos_id": tokenizer.eos_id,
            "sha256": sha256_bytes(tokenizer_path.read_bytes()),
            "measurement_on_val": tokenizer.measure(parts["val"]),
        },
        "audit": {
            "sensitive_patterns_full_corpus": scan_sensitive(text),
            "duplicates_full_corpus": duplicate_stats(text),
            "note": (
                "No records were removed. The duplicate rate is reported so the decision is "
                "visible; repeated short lines in dialogue are a property of the text, not "
                "scrape noise."
            ),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return PreparedData(out_dir=out_dir, manifest=manifest, tokenizer_path=tokenizer_path, split_paths=split_paths)


# --------------------------------------------------------------------------
# Batching
# --------------------------------------------------------------------------


class TokenBatcher:
    """Deterministic batches of ``[batch_size, block_size]`` token windows.

    Batch contents are a pure function of ``(seed, step)``. That makes a resumed
    run see exactly the data an uninterrupted run would have seen at that step,
    which is the property ``tests/test_resume.py`` checks, and it means the data
    position in a checkpoint is just the step number.
    """

    def __init__(self, path: str | Path, block_size: int, batch_size: int, seed: int = 1337) -> None:
        self.path = Path(path)
        self.block_size = block_size
        self.batch_size = batch_size
        self.seed = seed
        self.dtype = np.uint16
        self.data = np.memmap(self.path, dtype=self.dtype, mode="r")
        if len(self.data) < block_size + 1:
            raise ValueError(
                f"{self.path} holds {len(self.data)} tokens, need at least block_size+1={block_size + 1}"
            )

    def __len__(self) -> int:
        return len(self.data)

    @property
    def tokens_per_batch(self) -> int:
        return self.batch_size * self.block_size

    def batch(self, step: int, device: str = "cpu"):
        import torch

        rng = np.random.default_rng([self.seed, step])
        hi = len(self.data) - self.block_size - 1
        starts = rng.integers(0, hi, size=self.batch_size)
        x = np.stack([self.data[s : s + self.block_size] for s in starts]).astype(np.int64)
        y = np.stack([self.data[s + 1 : s + 1 + self.block_size] for s in starts]).astype(np.int64)
        return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)

    def iter_eval_batches(self, max_tokens: int | None = None, device: str = "cpu") -> Iterator:
        """Non-overlapping windows covering the split, in order.

        Deterministic and exhaustive, which is what an evaluation number needs
        to be comparable between runs.
        """
        import torch

        T = self.block_size
        n = len(self.data) - 1
        if max_tokens is not None:
            n = min(n, max_tokens)
        n_windows = n // T
        for i in range(0, n_windows, self.batch_size):
            starts = [j * T for j in range(i, min(i + self.batch_size, n_windows))]
            x = np.stack([self.data[s : s + T] for s in starts]).astype(np.int64)
            y = np.stack([self.data[s + 1 : s + 1 + T] for s in starts]).astype(np.int64)
            yield torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


def load_tokenizer(prepared_dir: str | Path) -> BaseTokenizer:
    return BaseTokenizer.load(Path(prepared_dir) / "tokenizer.json")


def load_manifest(prepared_dir: str | Path) -> dict:
    return json.loads((Path(prepared_dir) / "manifest.json").read_text(encoding="utf-8"))


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Prepare a corpus: fetch, audit, split, tokenize.")
    ap.add_argument("--corpus", default="tiny-shakespeare", choices=sorted(CORPORA))
    ap.add_argument("--tokenizer", default="char", choices=["char", "byte_bpe"])
    ap.add_argument("--vocab-size", type=int, default=512, help="byte_bpe only (includes 256 bytes + 2 specials)")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--cache-dir", default="data/raw")
    ap.add_argument("--local-file", default=None, help="use a local copy instead of downloading")
    args = ap.parse_args()

    out_dir = args.out_dir or f"data/prepared-{args.tokenizer}{'' if args.tokenizer == 'char' else '-' + str(args.vocab_size)}"
    prepared = prepare(
        corpus_name=args.corpus,
        tokenizer_kind=args.tokenizer,
        vocab_size=args.vocab_size,
        out_dir=out_dir,
        cache_dir=args.cache_dir,
        local_file=args.local_file,
        repo_root=Path(__file__).resolve().parents[3],
    )
    m = prepared.manifest
    print(f"corpus      {m['corpus']['name']}  {m['corpus']['chars']:,} chars  sha256 {m['corpus']['sha256'][:12]}")
    print(f"tokenizer   {m['tokenizer']['kind']}  vocab {m['tokenizer']['vocab_size']}  "
          f"{m['tokenizer']['measurement_on_val']['tokens_per_char']:.3f} tokens/char on val")
    for name, s in m["splits"].items():
        print(f"  {name:<5} {s['tokens']:>9,} tokens  {s['bytes']:>9,} bytes -> {s['file']}")
    print(f"audit       sensitive={m['audit']['sensitive_patterns_full_corpus']}")
    print(f"            duplicate blocks={m['audit']['duplicates_full_corpus']['duplicate_blocks']}"
          f" / {m['audit']['duplicates_full_corpus']['blocks']}")
    print(f"written to  {prepared.out_dir}")


if __name__ == "__main__":
    main()

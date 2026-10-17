"""Tokenizers for the tiny-gpt reference project.

Two implementations, both written from scratch with no tokenizer library:

* ``CharTokenizer``  -- one token per character seen in the training split.
* ``ByteBPETokenizer`` -- byte-level BPE trained on the training split only.

Both expose the same interface so the rest of the project never needs to know
which one it is using, and both record the byte length of every token, which is
what ``bits per byte`` evaluation needs (see SKILL.md, "Minimum evaluation
suite"): per-token loss is not comparable across vocabularies, bits per byte is.

Special tokens get fixed IDs at the bottom of the vocabulary so that adding
merges later never renumbers them. SKILL.md: "Never change token IDs after
training starts."
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

# Pre-tokenization pattern, simplified from the GPT-2 regex. It keeps merges
# from crossing word boundaries, which is what makes BPE produce word-like
# tokens instead of tokens that straddle whitespace.
PRETOKEN_RE = re.compile(r"'(?:[sdmt]|ll|ve|re)| ?\w+| ?[^\s\w]+|\s+(?!\S)|\s+", re.UNICODE)

SPECIAL_TOKENS = ("<|bos|>", "<|eos|>")


class TokenizerError(RuntimeError):
    pass


@dataclass
class _Special:
    bos_id: int
    eos_id: int


class BaseTokenizer:
    """Shared interface. Subclasses fill ``_id_to_bytes``."""

    kind = "base"

    def __init__(self) -> None:
        self._id_to_bytes: list[bytes] = []
        self.bos_id = -1
        self.eos_id = -1

    # -- size ---------------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return len(self._id_to_bytes)

    @property
    def special_ids(self) -> tuple[int, ...]:
        return (self.bos_id, self.eos_id)

    def token_byte_lengths(self) -> list[int]:
        """Bytes represented by each token; special tokens count as 0 bytes.

        Used by the bits-per-byte metric. Counting a special token as some
        number of bytes would silently inflate the denominator and make the
        metric look better than it is.
        """
        lengths = [len(b) for b in self._id_to_bytes]
        for sid in self.special_ids:
            if sid >= 0:
                lengths[sid] = 0
        return lengths

    # -- coding -------------------------------------------------------------
    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        raise NotImplementedError

    def decode(self, ids: Sequence[int]) -> str:
        out = bytearray()
        for i in ids:
            if i in self.special_ids:
                continue
            if not 0 <= i < self.vocab_size:
                raise TokenizerError(f"token id {i} outside vocabulary of size {self.vocab_size}")
            out += self._id_to_bytes[i]
        return out.decode("utf-8", errors="replace")

    # -- persistence --------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.state_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)
        return path

    def state_dict(self) -> dict:
        raise NotImplementedError

    @staticmethod
    def load(path: str | Path) -> "BaseTokenizer":
        state = json.loads(Path(path).read_text(encoding="utf-8"))
        kind = state.get("kind")
        if kind == "char":
            return CharTokenizer.from_state(state)
        if kind == "byte_bpe":
            return ByteBPETokenizer.from_state(state)
        raise TokenizerError(f"unknown tokenizer kind {kind!r} in {path}")

    # -- reporting ----------------------------------------------------------
    def measure(self, text: str) -> dict:
        """Tokens per character / per byte, as SKILL.md's tokenizer section asks."""
        ids = self.encode(text)
        n_bytes = len(text.encode("utf-8"))
        return {
            "kind": self.kind,
            "vocab_size": self.vocab_size,
            "chars": len(text),
            "bytes": n_bytes,
            "tokens": len(ids),
            "tokens_per_char": len(ids) / max(len(text), 1),
            "tokens_per_byte": len(ids) / max(n_bytes, 1),
            "bytes_per_token": n_bytes / max(len(ids), 1),
        }


class CharTokenizer(BaseTokenizer):
    """Character-level vocabulary: the simplest thing that can possibly work."""

    kind = "char"

    def __init__(self, chars: Sequence[str]) -> None:
        super().__init__()
        chars = sorted(set(chars))
        self._chars = chars
        self._id_to_bytes = [c.encode("utf-8") for c in chars]
        self._stoi = {c: i for i, c in enumerate(chars)}
        self.bos_id = len(chars)
        self.eos_id = len(chars) + 1
        self._id_to_bytes += [b"", b""]  # specials contribute no bytes

    @classmethod
    def train(cls, text: str) -> "CharTokenizer":
        return cls(sorted(set(text)))

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        try:
            ids = [self._stoi[c] for c in text]
        except KeyError as exc:  # pragma: no cover - guarded by caller
            raise TokenizerError(f"character {exc.args[0]!r} is not in the vocabulary") from exc
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def state_dict(self) -> dict:
        return {"kind": self.kind, "chars": self._chars, "bos_id": self.bos_id, "eos_id": self.eos_id}

    @classmethod
    def from_state(cls, state: dict) -> "CharTokenizer":
        tok = cls(state["chars"])
        if tok.bos_id != state["bos_id"] or tok.eos_id != state["eos_id"]:
            raise TokenizerError("special token ids in the saved tokenizer do not match")
        return tok


class ByteBPETokenizer(BaseTokenizer):
    """Byte-level BPE, trained from scratch.

    Base vocabulary is the 256 byte values, so there is no unknown token and any
    byte string round-trips. Merges are learned on whitespace/punctuation
    pre-tokenized chunks.
    """

    kind = "byte_bpe"

    def __init__(self, merges: Sequence[tuple[int, int]]) -> None:
        super().__init__()
        self.merges: list[tuple[int, int]] = [tuple(m) for m in merges]  # type: ignore[misc]
        self._id_to_bytes = [bytes([i]) for i in range(256)]
        for a, b in self.merges:
            self._id_to_bytes.append(self._id_to_bytes[a] + self._id_to_bytes[b])
        self.bos_id = len(self._id_to_bytes)
        self.eos_id = len(self._id_to_bytes) + 1
        self._id_to_bytes += [b"", b""]
        self._ranks = {pair: i for i, pair in enumerate(self.merges)}
        self._merge_to_id = {pair: 256 + i for i, pair in enumerate(self.merges)}
        self._cache: dict[bytes, list[int]] = {}

    # -- training -----------------------------------------------------------
    @classmethod
    def train(cls, text: str, vocab_size: int = 512, verbose: bool = False) -> "ByteBPETokenizer":
        """Learn merges until the vocabulary reaches ``vocab_size``.

        ``vocab_size`` counts the 256 byte tokens, the learned merges, and the
        two special tokens, so the number of merges is ``vocab_size - 258``.
        """
        n_merges = vocab_size - 256 - len(SPECIAL_TOKENS)
        if n_merges < 0:
            raise TokenizerError("vocab_size must be at least 258 for a byte-level BPE tokenizer")

        words: Counter[bytes] = Counter()
        for chunk in PRETOKEN_RE.findall(text):
            words[chunk.encode("utf-8")] += 1

        # word id -> (list of token ids, frequency)
        sequences: list[list[int]] = [list(w) for w in words]
        freqs: list[int] = list(words.values())

        pair_counts: Counter[tuple[int, int]] = Counter()
        pair_where: dict[tuple[int, int], set[int]] = defaultdict(set)
        for wi, seq in enumerate(sequences):
            f = freqs[wi]
            for pair in zip(seq, seq[1:]):
                pair_counts[pair] += f
                pair_where[pair].add(wi)

        merges: list[tuple[int, int]] = []
        next_id = 256
        for step in range(n_merges):
            if not pair_counts:
                break
            best, best_count = max(pair_counts.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))
            if best_count < 2:
                break
            merges.append(best)
            affected = list(pair_where[best])
            for wi in affected:
                seq = sequences[wi]
                f = freqs[wi]
                # remove this word's contribution, rewrite it, add it back
                for pair in zip(seq, seq[1:]):
                    pair_counts[pair] -= f
                    if pair_counts[pair] <= 0:
                        del pair_counts[pair]
                        pair_where.pop(pair, None)
                    else:
                        pair_where[pair].discard(wi)
                new_seq: list[int] = []
                i = 0
                while i < len(seq):
                    if i < len(seq) - 1 and (seq[i], seq[i + 1]) == best:
                        new_seq.append(next_id)
                        i += 2
                    else:
                        new_seq.append(seq[i])
                        i += 1
                sequences[wi] = new_seq
                for pair in zip(new_seq, new_seq[1:]):
                    pair_counts[pair] += f
                    pair_where[pair].add(wi)
            pair_counts.pop(best, None)
            pair_where.pop(best, None)
            next_id += 1
            if verbose and (step + 1) % 100 == 0:
                print(f"  merge {step + 1}/{n_merges}: {best} -> {next_id - 1} (count {best_count})")
        return cls(merges)

    # -- coding -------------------------------------------------------------
    def _encode_chunk(self, chunk: bytes) -> list[int]:
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        ids = list(chunk)
        while len(ids) >= 2:
            best_rank = None
            best_i = -1
            for i, pair in enumerate(zip(ids, ids[1:])):
                rank = self._ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank, best_i = rank, i
            if best_rank is None:
                break
            pair = (ids[best_i], ids[best_i + 1])
            ids[best_i : best_i + 2] = [self._merge_to_id[pair]]
        self._cache[chunk] = ids
        return ids

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids: list[int] = [self.bos_id] if add_bos else []
        for chunk in PRETOKEN_RE.findall(text):
            ids.extend(self._encode_chunk(chunk.encode("utf-8")))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def state_dict(self) -> dict:
        return {
            "kind": self.kind,
            "merges": [list(m) for m in self.merges],
            "bos_id": self.bos_id,
            "eos_id": self.eos_id,
        }

    @classmethod
    def from_state(cls, state: dict) -> "ByteBPETokenizer":
        tok = cls([tuple(m) for m in state["merges"]])
        if tok.bos_id != state["bos_id"] or tok.eos_id != state["eos_id"]:
            raise TokenizerError("special token ids in the saved tokenizer do not match")
        return tok


def build_tokenizer(kind: str, text: str, vocab_size: int = 512) -> BaseTokenizer:
    if kind == "char":
        return CharTokenizer.train(text)
    if kind == "byte_bpe":
        return ByteBPETokenizer.train(text, vocab_size=vocab_size)
    raise TokenizerError(f"unknown tokenizer kind {kind!r}")


def encode_stream(tokenizer: BaseTokenizer, texts: Iterable[str]) -> list[int]:
    out: list[int] = []
    for t in texts:
        out.extend(tokenizer.encode(t))
    return out

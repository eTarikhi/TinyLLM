"""Tokenizer correctness: round-trip, byte accounting, stable ids, persistence.

The byte-length table is tested because bits per byte depends on it. If a
special token were counted as bytes, every BPB number in the project would be
quietly optimistic.
"""

import sys
import unittest
from pathlib import Path

from _helpers import TINY_TEXT, TempDir

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tokenizer import BaseTokenizer, ByteBPETokenizer, CharTokenizer  # noqa: E402


class TestCharTokenizer(unittest.TestCase):
    def test_round_trip(self):
        tok = CharTokenizer.train(TINY_TEXT)
        self.assertEqual(tok.decode(tok.encode(TINY_TEXT)), TINY_TEXT)

    def test_special_ids_are_outside_the_character_range(self):
        tok = CharTokenizer.train(TINY_TEXT)
        self.assertEqual(tok.bos_id, tok.vocab_size - 2)
        self.assertEqual(tok.eos_id, tok.vocab_size - 1)
        self.assertEqual(tok.token_byte_lengths()[tok.eos_id], 0)

    def test_save_load_preserves_ids(self):
        tok = CharTokenizer.train(TINY_TEXT)
        with TempDir() as tmp:
            path = tok.save(Path(tmp) / "tok.json")
            again = BaseTokenizer.load(path)
        self.assertEqual(tok.encode("to be"), again.encode("to be"))
        self.assertEqual(tok.vocab_size, again.vocab_size)


class TestByteBPETokenizer(unittest.TestCase):
    def test_round_trip_including_unseen_text(self):
        tok = ByteBPETokenizer.train(TINY_TEXT, vocab_size=320)
        self.assertEqual(tok.decode(tok.encode(TINY_TEXT)), TINY_TEXT)
        # Byte-level BPE has no unknown token: text it never saw still round-trips.
        exotic = "Zürich 2026 — ünïcödé ✓"
        self.assertEqual(tok.decode(tok.encode(exotic)), exotic)

    def test_merges_compress(self):
        tok = ByteBPETokenizer.train(TINY_TEXT, vocab_size=320)
        self.assertLess(len(tok.encode(TINY_TEXT)), len(TINY_TEXT.encode("utf-8")))

    def test_byte_lengths_match_decoded_bytes(self):
        tok = ByteBPETokenizer.train(TINY_TEXT, vocab_size=300)
        lengths = tok.token_byte_lengths()
        ids = tok.encode("to be or not to be")
        self.assertEqual(sum(lengths[i] for i in ids), len("to be or not to be".encode("utf-8")))

    def test_vocab_size_is_exact(self):
        tok = ByteBPETokenizer.train(TINY_TEXT, vocab_size=300)
        self.assertEqual(tok.vocab_size, 300)

    def test_save_load_preserves_ids(self):
        tok = ByteBPETokenizer.train(TINY_TEXT, vocab_size=300)
        with TempDir() as tmp:
            path = tok.save(Path(tmp) / "tok.json")
            again = BaseTokenizer.load(path)
        self.assertEqual(tok.encode("the slings and arrows"), again.encode("the slings and arrows"))

    def test_measure_reports_compression(self):
        tok = ByteBPETokenizer.train(TINY_TEXT, vocab_size=320)
        m = tok.measure(TINY_TEXT)
        self.assertLess(m["tokens_per_byte"], 1.0)
        self.assertGreater(m["bytes_per_token"], 1.0)


if __name__ == "__main__":
    unittest.main()

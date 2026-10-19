"""A single aggregate bits-per-byte figure averages over the corpus mixture.

That average can hide a large effect on one part of the corpus. Measured on
this project's own corpus: adding 44 books moved the aggregate test figure by
0.35% while moving the added books' own figure by 10.47%. Nothing in the
aggregate said so, and the first experiment drew the wrong conclusion as a
result.

`evaluate_per_document` cuts a multi-document split back into its documents and
scores each one. These tests check that the cut is exact, that the labels line
up with the corpus order, and that short pieces are reported rather than
silently dropped.

Everything runs on synthetic tensors and a tiny model; nothing is downloaded.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluate import evaluate_per_document, summarise_per_document  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402

EOS = 3
BLOCK = 8
VOCAB = 32


def _model() -> GPT:
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=VOCAB, block_size=BLOCK, n_layer=1, n_head=1, n_embd=16))


def _byte_lens() -> torch.Tensor:
    lens = np.ones(VOCAB, dtype=np.int64)
    lens[EOS] = 0  # EOS encodes no bytes, as in the real tokenizers
    return torch.tensor(lens)


def _split_file(pieces: list[np.ndarray]) -> Path:
    joined: list[np.ndarray] = []
    for i, p in enumerate(pieces):
        joined.append(p)
        if i < len(pieces) - 1:
            joined.append(np.array([EOS], dtype=np.uint16))
    tmp = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
    np.concatenate(joined).tofile(tmp.name)
    tmp.close()
    return Path(tmp.name)


class PerDocumentTest(unittest.TestCase):
    def setUp(self):
        self.model = _model()
        self.byte_lens = _byte_lens()
        self.rng = np.random.default_rng(0)

    def _piece(self, n: int) -> np.ndarray:
        # Never emit EOS inside a piece: it is a separator, not content.
        return self.rng.integers(4, VOCAB, size=n, dtype=np.uint16)

    def test_documents_are_recovered_exactly(self):
        sizes = [100, 60, 80]
        path = _split_file([self._piece(n) for n in sizes])
        self.addCleanup(path.unlink)
        rows = evaluate_per_document(self.model, path, EOS, self.byte_lens, BLOCK)
        self.assertEqual([r["tokens"] for r in rows], sizes,
                         "splitting on EOS did not recover the original pieces")

    def test_labels_follow_corpus_order(self):
        path = _split_file([self._piece(50), self._piece(50)])
        self.addCleanup(path.unlink)
        records = [{"doc_id": "a", "title": "Alpha", "author": "X"},
                   {"doc_id": "b", "title": "Beta", "author": "Y"}]
        rows = evaluate_per_document(self.model, path, EOS, self.byte_lens, BLOCK, records)
        self.assertEqual([r["title"] for r in rows], ["Alpha", "Beta"])
        self.assertEqual([r["doc_id"] for r in rows], ["a", "b"])

    def test_short_pieces_are_reported_not_dropped(self):
        """A piece shorter than one block cannot be scored; say so."""
        path = _split_file([self._piece(50), self._piece(BLOCK - 1), self._piece(50)])
        self.addCleanup(path.unlink)
        rows = evaluate_per_document(self.model, path, EOS, self.byte_lens, BLOCK)
        self.assertEqual(len(rows), 3, "a short document disappeared from the breakdown")
        self.assertTrue(rows[0]["scored"] and rows[2]["scored"])
        self.assertFalse(rows[1]["scored"])
        self.assertIsNone(rows[1]["bpb"])
        self.assertIn("shorter than", rows[1]["reason"])

    def test_a_document_the_model_cannot_predict_stands_out(self):
        """The whole point: one odd document must be visible in the breakdown."""
        normal = np.full(400, 7, dtype=np.uint16)       # trivially predictable
        odd = self.rng.integers(4, VOCAB, size=400, dtype=np.uint16)  # high entropy
        path = _split_file([normal, odd])
        self.addCleanup(path.unlink)
        rows = evaluate_per_document(self.model, path, EOS, self.byte_lens, BLOCK)
        self.assertLess(rows[0]["bpb"], rows[1]["bpb"],
                        "the high-entropy document did not score worse")
        summary = summarise_per_document(rows)
        self.assertEqual(summary["hardest"]["bpb"], rows[1]["bpb"])
        self.assertEqual(summary["easiest"]["bpb"], rows[0]["bpb"])
        self.assertGreater(summary["range_pct_of_mean"], 0.0)

    def test_summary_counts_unscored_documents(self):
        path = _split_file([self._piece(50), self._piece(BLOCK - 1)])
        self.addCleanup(path.unlink)
        rows = evaluate_per_document(self.model, path, EOS, self.byte_lens, BLOCK)
        s = summarise_per_document(rows)
        self.assertEqual(s["documents"], 2)
        self.assertEqual(s["scored"], 1)
        self.assertEqual(s["unscored"], 1)

    def test_single_document_split_is_handled(self):
        path = _split_file([self._piece(200)])
        self.addCleanup(path.unlink)
        rows = evaluate_per_document(self.model, path, EOS, self.byte_lens, BLOCK)
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["scored"])
        self.assertEqual(summarise_per_document(rows)["stdev_bpb"], 0.0)

    def test_byte_weighted_mean_matches_the_whole_split(self):
        """The per-document numbers must reconcile with the aggregate.

        Bits per byte is a ratio of sums, so the documents' figures recombine
        by weighting on bytes, not by averaging. Checking that keeps the
        breakdown honest against the headline number.
        """
        from data import TokenBatcher
        from evaluate import evaluate_model

        pieces = [self._piece(240), self._piece(200), self._piece(160)]
        path = _split_file(pieces)
        self.addCleanup(path.unlink)
        rows = evaluate_per_document(self.model, path, EOS, self.byte_lens, BLOCK)

        # Bits per byte is total_nats / (ln2 * total_bytes), so the documents
        # recombine by weighting each figure on its own byte count.
        num = sum(r["bpb"] * r["bytes"] for r in rows if r["scored"])
        den = sum(r["bytes"] for r in rows if r["scored"])
        recombined = num / den

        whole = evaluate_model(self.model, TokenBatcher(path, BLOCK, 16, seed=0), self.byte_lens)
        # The aggregate pass tiles windows across EOS boundaries, so it scores
        # a handful of tokens the per-document pass does not. They should still
        # land within a few percent of each other.
        self.assertAlmostEqual(recombined, whole["bpb"], delta=0.1 * whole["bpb"])


if __name__ == "__main__":
    unittest.main()

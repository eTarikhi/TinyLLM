"""`eval_only` documents and a reused vocabulary, the two halves of a control arm.

`train_only` grows the training split while holding evaluation fixed. That
answers "does more training text help on *this* evaluation" and nothing else —
and when it was measured the answer came out negative, because the added books
turned out to be a different distribution from the evaluation.

Asking the general question needs the additions in val and test too, plus a
control corpus with the *same* evaluation splits and without the extra training
text. Two things have to hold for that comparison to mean anything:

1. the two corpora must produce byte-identical val and test splits, and
2. they must produce byte-identical tokenizers — a model's embedding table is
   indexed by token id, so two models can only be scored on the same bytes if
   those bytes were tokenized the same way.

(1) is `eval_only`; (2) is `tokenizer_from`. Both are checked here. Everything
runs on synthetic local text; nothing is downloaded.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import data as data_mod  # noqa: E402
from tokenizer import TokenizerError  # noqa: E402


def _doc(doc_id: str, train_only: bool = False, eval_only: bool = False) -> data_mod.Document:
    return data_mod.Document(
        doc_id=doc_id, title=doc_id, author="a", year="1900",
        repo=f"x/{doc_id}", member=f"{doc_id}.txt",
        train_only=train_only, eval_only=eval_only,
    )


def _body(tag: str, paragraphs: int = 40) -> str:
    return "\n\n".join(f"{tag} paragraph {i} with enough words to snap on." for i in range(paragraphs))


class _Prepares(unittest.TestCase):
    def _prepare(self, docs, bodies, out_dir, kind="byte_bpe", vocab=300, tokenizer_from=None):
        corpus = data_mod.MultiDocCorpus(
            name="synthetic", description="d", sha256="", documents=tuple(docs),
            text_license="l", file_license="l", restrictions="r", access_date="2026-10-19",
        )
        data_mod.CORPORA["synthetic"] = corpus
        cache = Path(out_dir) / "raw" / "synthetic"
        cache.mkdir(parents=True, exist_ok=True)
        for doc, body in zip(docs, bodies):
            (cache / f"{doc.doc_id}.txt").write_text(body, encoding="utf-8")
            (cache / f"{doc.doc_id}.stats.json").write_text('{"original_chars": 0}')
        try:
            return data_mod.prepare(
                corpus_name="synthetic", tokenizer_kind=kind, vocab_size=vocab,
                out_dir=Path(out_dir) / "prepared", cache_dir=Path(out_dir) / "raw",
                tokenizer_sample_chars=0, tokenizer_from=tokenizer_from,
            )
        finally:
            data_mod.CORPORA.pop("synthetic", None)


class EvalOnlyTest(_Prepares):
    def test_document_cannot_be_both(self):
        with self.assertRaises(ValueError) as cm:
            _doc("x", train_only=True, eval_only=True)
        self.assertIn("both train_only and eval_only", str(cm.exception))

    def test_eval_only_document_reaches_evaluation_but_not_training(self):
        docs = [_doc("a"), _doc("b", eval_only=True)]
        bodies = [_body("alpha"), _body("beta")]
        with tempfile.TemporaryDirectory() as t:
            prep = self._prepare(docs, bodies, t)
            splits = prep.manifest["splits"]
            # Both documents contribute an evaluation piece...
            self.assertEqual(splits["val"]["documents"], 2)
            self.assertEqual(splits["test"]["documents"], 2)
            # ...but only the ordinary one contributes to training.
            self.assertEqual(splits["train"]["documents"], 1)
            train_text = (Path(t) / "prepared" / "train.bin").read_bytes()
            self.assertGreater(len(train_text), 0)
            tok = data_mod.load_tokenizer(Path(t) / "prepared")
            import numpy as np
            ids = np.fromfile(Path(t) / "prepared" / "train.bin", dtype=np.uint16)
            decoded = tok.decode([int(i) for i in ids if int(i) != tok.eos_id])
            self.assertIn("alpha", decoded)
            self.assertNotIn("beta", decoded,
                             "an eval-only document leaked into the training split")

    def test_control_arm_shares_evaluation_splits_with_the_full_arm(self):
        """The whole experiment rests on this: same val/test, different train."""
        full_docs = [_doc("a"), _doc("b"), _doc("c")]
        ctrl_docs = [_doc("a"), _doc("b", eval_only=True), _doc("c", eval_only=True)]
        bodies = [_body("alpha"), _body("beta", 60), _body("gamma", 60)]

        with tempfile.TemporaryDirectory() as t1, tempfile.TemporaryDirectory() as t2:
            full = self._prepare(full_docs, bodies, t1)
            # The control arm must reuse the full arm's vocabulary, otherwise it
            # fits a new one to its smaller training split and the token ids stop
            # meaning the same thing.
            ctrl = self._prepare(ctrl_docs, bodies, t2,
                                 tokenizer_from=Path(t1) / "prepared")

            f, c = full.manifest["splits"], ctrl.manifest["splits"]
            for split in ("val", "test"):
                self.assertEqual(f[split]["sha256_text"], c[split]["sha256_text"],
                                 f"{split} text differs between the two arms")
                self.assertEqual(f[split]["sha256_tokens"], c[split]["sha256_tokens"],
                                 f"{split} tokens differ between the two arms")
                self.assertEqual(f[split]["eos_separators"], c[split]["eos_separators"])
            self.assertEqual(
                (Path(t1) / "prepared" / "test.bin").read_bytes(),
                (Path(t2) / "prepared" / "test.bin").read_bytes(),
                "test.bin is not byte-identical across the arms",
            )
            self.assertGreater(f["train"]["tokens"], c["train"]["tokens"],
                               "the full arm should train on strictly more text")

    def test_tokenizer_sample_stays_aligned_when_both_flags_appear(self):
        """A corpus may hold train-only and eval-only documents at once.

        The tokenizer sample is built by zipping the training pieces against a
        per-piece flag list. Before this was tracked per *piece*, the zip ran
        against the per-*document* flags, and one eval-only document shifted
        every later flag by one — silently fitting the vocabulary on text that
        was supposed to be excluded from it. Nothing downstream reports that.

        The invariant, stated without reference to internals: adding a
        train-only document must not move the vocabulary.
        """
        with_extra = [_doc("a"), _doc("b", eval_only=True), _doc("c", train_only=True)]
        without = [_doc("a"), _doc("b", eval_only=True)]
        bodies = [_body("alpha"), _body("beta", 60), _body("ZZZZ", 60)]

        with tempfile.TemporaryDirectory() as t1, tempfile.TemporaryDirectory() as t2:
            self._prepare(with_extra, bodies, t1)
            self._prepare(without, bodies[:2], t2)
            self.assertEqual(
                (Path(t1) / "prepared" / "tokenizer.json").read_bytes(),
                (Path(t2) / "prepared" / "tokenizer.json").read_bytes(),
                "a train-only document moved the vocabulary -- the per-piece flag "
                "list is misaligned with the training pieces",
            )


class TokenizerReuseTest(_Prepares):
    def test_reused_tokenizer_is_byte_identical(self):
        docs = [_doc("a"), _doc("b")]
        bodies = [_body("alpha"), _body("beta")]
        with tempfile.TemporaryDirectory() as t1, tempfile.TemporaryDirectory() as t2:
            self._prepare(docs, bodies, t1)
            src = Path(t1) / "prepared" / "tokenizer.json"
            self._prepare(docs, bodies, t2, tokenizer_from=src)
            self.assertEqual(src.read_bytes(),
                             (Path(t2) / "prepared" / "tokenizer.json").read_bytes(),
                             "a reused tokenizer did not round-trip byte-identically")

    def test_manifest_records_the_provenance(self):
        docs = [_doc("a"), _doc("b")]
        bodies = [_body("alpha"), _body("beta")]
        with tempfile.TemporaryDirectory() as t1, tempfile.TemporaryDirectory() as t2:
            self._prepare(docs, bodies, t1)
            prep = self._prepare(docs, bodies, t2,
                                 tokenizer_from=Path(t1) / "prepared")
            tok = prep.manifest["tokenizer"]
            self.assertIsNotNone(tok["reused_from"], "reuse was not recorded in the manifest")
            self.assertIn("reused from", tok["trained_on"])
            # A reused vocabulary was not fitted here, so claiming a sample size
            # would be a lie about where it came from.
            self.assertEqual(tok["trained_on_chars"], 0)
            json.dumps(prep.manifest)  # must stay serialisable

    def test_missing_source_is_rejected(self):
        docs = [_doc("a"), _doc("b")]
        bodies = [_body("alpha"), _body("beta")]
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(FileNotFoundError):
                self._prepare(docs, bodies, t, tokenizer_from=Path(t) / "nope")

    def test_mismatched_kind_is_rejected(self):
        docs = [_doc("a"), _doc("b")]
        bodies = [_body("alpha"), _body("beta")]
        with tempfile.TemporaryDirectory() as t1, tempfile.TemporaryDirectory() as t2:
            self._prepare(docs, bodies, t1, kind="char")
            with self.assertRaises(ValueError) as cm:
                self._prepare(docs, bodies, t2, kind="byte_bpe",
                              tokenizer_from=Path(t1) / "prepared")
            self.assertIn("same kind of vocabulary", str(cm.exception))

    def test_mismatched_vocab_size_is_rejected(self):
        docs = [_doc("a"), _doc("b")]
        bodies = [_body("alpha"), _body("beta")]
        with tempfile.TemporaryDirectory() as t1, tempfile.TemporaryDirectory() as t2:
            self._prepare(docs, bodies, t1, vocab=300)
            with self.assertRaises(ValueError) as cm:
                self._prepare(docs, bodies, t2, vocab=320,
                              tokenizer_from=Path(t1) / "prepared")
            self.assertIn("vocabulary of 300", str(cm.exception))


if __name__ == "__main__":
    unittest.main()


class EvalSliceTest(unittest.TestCase):
    """`iter_eval_batches(max_tokens=...)` takes a prefix, which is not a sample.

    The evaluation splits of a multi-document corpus are the per-document
    pieces concatenated in document order. A capped prefix therefore covers
    the first few documents and none of the rest — so a "validation" number
    computed that way is a number about the first few books. `spread=True`
    keeps the cost and covers the whole split.
    """

    def _batcher(self, n_tokens: int, block: int = 8, batch: int = 4):
        import numpy as np
        from data import TokenBatcher
        tmp = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
        # First half all 1s, second half all 2s: a prefix slice can only ever
        # see the 1s, which is exactly the failure mode being tested.
        ids = np.concatenate([
            np.ones(n_tokens // 2, dtype=np.uint16),
            np.full(n_tokens - n_tokens // 2, 2, dtype=np.uint16),
        ])
        ids.tofile(tmp.name)
        tmp.close()
        self.addCleanup(lambda: Path(tmp.name).unlink(missing_ok=True))
        return TokenBatcher(tmp.name, block, batch, seed=0)

    def test_prefix_slice_misses_the_tail_and_spread_does_not(self):
        b = self._batcher(800)
        cap = 200

        seen_prefix = set()
        for x, _ in b.iter_eval_batches(max_tokens=cap):
            seen_prefix.update(int(v) for v in x.flatten())
        seen_spread = set()
        for x, _ in b.iter_eval_batches(max_tokens=cap, spread=True):
            seen_spread.update(int(v) for v in x.flatten())

        self.assertEqual(seen_prefix, {1}, "the prefix slice should only reach the first half")
        self.assertEqual(seen_spread, {1, 2}, "the spread slice should reach both halves")

    def test_spread_costs_the_same(self):
        b = self._batcher(800)
        cap = 200
        n_prefix = sum(x.numel() for x, _ in b.iter_eval_batches(max_tokens=cap))
        n_spread = sum(x.numel() for x, _ in b.iter_eval_batches(max_tokens=cap, spread=True))
        self.assertEqual(n_prefix, n_spread, "spreading must not change the evaluation budget")

    def test_prefix_behaviour_is_unchanged(self):
        """Published numbers were produced with the prefix slice; lock it down."""
        b = self._batcher(800)
        starts = []
        for x, _ in b.iter_eval_batches(max_tokens=200):
            starts.append(x.shape[0])
        # 200 // 8 = 25 windows, in batches of 4 -> 6 full batches and one of 1.
        self.assertEqual(starts, [4, 4, 4, 4, 4, 4, 1])

    def test_both_modes_are_deterministic(self):
        b = self._batcher(800)
        for spread in (False, True):
            first = [x.tolist() for x, _ in b.iter_eval_batches(max_tokens=200, spread=spread)]
            second = [x.tolist() for x, _ in b.iter_eval_batches(max_tokens=200, spread=spread)]
            self.assertEqual(first, second, f"spread={spread} is not deterministic")

    def test_uncapped_evaluation_is_exhaustive_in_both_modes(self):
        b = self._batcher(800)
        n_all = sum(x.numel() for x, _ in b.iter_eval_batches())
        n_all_spread = sum(x.numel() for x, _ in b.iter_eval_batches(spread=True))
        self.assertEqual(n_all, n_all_spread)
        self.assertEqual(n_all, (800 - 1) // 8 * 8)

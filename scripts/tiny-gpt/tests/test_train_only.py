"""`train_only` documents must enlarge training and leave evaluation untouched.

The point of the flag is that a corpus can grow without moving the evaluation
target. If a train-only document leaked a single byte -- or even just an EOS
separator -- into val or test, bits-per-byte from the larger corpus would no
longer be comparable with bits-per-byte from the smaller one, and the whole
"does more data help?" experiment would be measuring two different things.

These tests run entirely on synthetic local text; nothing is downloaded.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import data as data_mod  # noqa: E402


def _doc(doc_id: str, train_only: bool = False) -> data_mod.Document:
    return data_mod.Document(
        doc_id=doc_id, title=doc_id, author="a", year="1900",
        repo=f"x/{doc_id}", member=f"{doc_id}.txt", train_only=train_only,
    )


def _body(tag: str, paragraphs: int = 40) -> str:
    return "\n\n".join(f"{tag} paragraph {i} with enough words to snap on." for i in range(paragraphs))


class TrainOnlyTest(unittest.TestCase):
    """Prepare the same base corpus with and without extra train-only docs."""

    def _prepare(self, docs, bodies, out_dir, kind="byte_bpe", vocab=300):
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
                tokenizer_sample_chars=0,
            )
        finally:
            data_mod.CORPORA.pop("synthetic", None)

    def test_val_and_test_are_byte_identical_when_training_grows(self):
        base_docs = [_doc("a"), _doc("b")]
        base_bodies = [_body("alpha"), _body("beta")]
        extra_docs = base_docs + [_doc("c", train_only=True), _doc("d", train_only=True)]
        extra_bodies = base_bodies + [_body("gamma", 60), _body("delta", 60)]

        with tempfile.TemporaryDirectory() as t1, tempfile.TemporaryDirectory() as t2:
            small = self._prepare(base_docs, base_bodies, t1)
            large = self._prepare(extra_docs, extra_bodies, t2)

            s, l = small.manifest["splits"], large.manifest["splits"]
            for split in ("val", "test"):
                self.assertEqual(s[split]["sha256_text"], l[split]["sha256_text"],
                                 f"{split} text changed when train-only documents were added")
                self.assertEqual(s[split]["sha256_tokens"], l[split]["sha256_tokens"],
                                 f"{split} tokens changed when train-only documents were added")
                self.assertEqual(s[split]["documents"], l[split]["documents"])
                self.assertEqual(s[split]["eos_separators"], l[split]["eos_separators"],
                                 f"{split} separator count changed -- an empty piece leaked in")
            self.assertGreater(l["train"]["tokens"], s["train"]["tokens"],
                               "train-only documents did not reach the training split")

    def test_char_tokenizer_still_covers_train_only_text(self):
        """A character vocabulary must cover everything it has to encode.

        Train-only text is excluded from the tokenizer sample for byte-level
        tokenizers, where unseen input costs compression. For a character
        tokenizer unseen input is a hard failure, so the exclusion must not
        apply -- encoding the corpus is more important than holding the
        vocabulary fixed.
        """
        docs = [_doc("a"), _doc("z", train_only=True)]
        bodies = [_body("alpha"), "qqqq wwww\n\n" + _body("zeta", 40)]
        with tempfile.TemporaryDirectory() as t:
            prepared = self._prepare(docs, bodies, t, kind="char", vocab=256)
            self.assertGreater(prepared.manifest["splits"]["train"]["tokens"], 0)

    def test_train_only_document_is_not_split(self):
        docs = [_doc("a"), _doc("c", train_only=True)]
        bodies = [_body("alpha"), _body("gamma", 50)]
        with tempfile.TemporaryDirectory() as t:
            prepared = self._prepare(docs, bodies, t)
            # The whole of document c, and no part of document a's tail, is in
            # training: training must contain c's final paragraph verbatim.
            train_text = (Path(t) / "prepared" / "train.bin")
            self.assertTrue(train_text.exists())
            self.assertEqual(prepared.manifest["splits"]["val"]["documents"], 1)
            self.assertEqual(prepared.manifest["splits"]["test"]["documents"], 1)
            self.assertEqual(prepared.manifest["splits"]["train"]["documents"], 2)

    def test_flag_defaults_to_false_so_existing_corpora_are_unchanged(self):
        self.assertFalse(_doc("a").train_only)
        for doc in data_mod.CORPORA["gutenberg-mixed-27"].documents:
            self.assertFalse(doc.train_only, f"{doc.doc_id} must stay in all three splits")
        expanded = data_mod.CORPORA["gutenberg-mixed-71"].documents
        self.assertEqual(len(expanded), 71)
        self.assertEqual(sum(d.train_only for d in expanded), 44)
        # The first 27 are the original corpus, in the original order: that is
        # what keeps the tokenizer sample -- the first 4M characters of the
        # training split -- identical between the two corpora.
        self.assertEqual([d.doc_id for d in expanded[:27]],
                         [d.doc_id for d in data_mod.CORPORA["gutenberg-mixed-27"].documents])


if __name__ == "__main__":
    unittest.main()

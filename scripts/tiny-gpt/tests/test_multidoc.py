"""Gate 1 tests for the multi-document corpus path.

Two things in `data.py` can corrupt a corpus silently, which is the worst kind
of bug in a training pipeline: boilerplate stripping that matches the wrong
place, and a per-document split that loses or duplicates text. Both are cheap
to test exactly, so they are tested exactly rather than eyeballed in a manifest.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import data  # noqa: E402

HEADER = (
    "The Project Gutenberg eBook of Example Book, by A. Writer\n\n"
    "This eBook is for the use of anyone anywhere at no cost and with almost\n"
    "no restrictions whatsoever.\n\n"
    "*** START OF THE PROJECT GUTENBERG EBOOK EXAMPLE BOOK ***\n"
)
FOOTER = (
    "\n*** END OF THE PROJECT GUTENBERG EBOOK EXAMPLE BOOK ***\n\n"
    "Updated editions will replace the previous one.\n"
)
BODY = "CHAPTER I.\n\nIt was a bright cold day in April.\n\nThe clocks were striking thirteen.\n"


class TestBoilerplateStripping(unittest.TestCase):
    def test_header_and_footer_are_removed(self):
        body, stats = data.strip_gutenberg_boilerplate(HEADER + BODY + FOOTER)
        self.assertNotIn("Project Gutenberg", body)
        self.assertNotIn("START OF THE PROJECT", body)
        self.assertNotIn("Updated editions", body)
        self.assertIn("bright cold day", body)
        self.assertEqual(stats["original_chars"], len(HEADER + BODY + FOOTER))
        self.assertEqual(stats["body_chars"], len(body))
        self.assertEqual(stats["removed_chars"], stats["original_chars"] - stats["body_chars"])

    def test_produced_by_credit_is_removed(self):
        credit = "Produced by Jane Volunteer, Someone Else and the Online\nDistributed Proofreading Team\n\n"
        body, _ = data.strip_gutenberg_boilerplate(HEADER + credit + BODY + FOOTER)
        self.assertNotIn("Produced by", body)
        self.assertIn("bright cold day", body)

    def test_text_without_markers_survives_unchanged(self):
        # A file that is not a Gutenberg text must pass through, not be emptied.
        body, stats = data.strip_gutenberg_boilerplate(BODY)
        self.assertEqual(body.strip("\n"), BODY.strip("\n"))
        self.assertEqual(stats["removed_chars"], len(BODY) - len(body))

    def test_body_always_ends_with_exactly_one_newline(self):
        for raw in (HEADER + BODY + FOOTER, BODY, BODY + "\n\n\n\n"):
            body, _ = data.strip_gutenberg_boilerplate(raw)
            self.assertTrue(body.endswith("\n"))
            self.assertFalse(body.endswith("\n\n"))


class TestPerDocumentSplit(unittest.TestCase):
    """The split logic from `prepare()`, exercised directly on synthetic docs.

    `prepare()` itself needs a network fetch, so the arithmetic it performs is
    reproduced here against documents whose content makes loss or duplication
    detectable character by character.
    """

    @staticmethod
    def split(documents: list[str], fracs=(0.90, 0.05, 0.05)):
        text = "\n\n".join(documents)
        lengths = [len(d) for d in documents]

        def snap(s: str, i: int) -> int:
            j = s.find("\n\n", i)
            return (j + 2) if j != -1 else len(s)

        parts = {"train": [], "val": [], "test": []}
        offset = 0
        for length in lengths:
            doc = text[offset : offset + length]
            offset += length + 2
            i_train = snap(doc, int(len(doc) * fracs[0]))
            i_val = snap(doc, int(len(doc) * (fracs[0] + fracs[1])))
            if not 0 < i_train < i_val < len(doc):
                i_train = int(len(doc) * fracs[0])
                i_val = int(len(doc) * (fracs[0] + fracs[1]))
            parts["train"].append(doc[:i_train])
            parts["val"].append(doc[i_train:i_val])
            parts["test"].append(doc[i_val:])
        return text, parts

    def setUp(self):
        # Paragraph-structured documents of deliberately different lengths.
        self.docs = [
            "".join(f"doc{d} paragraph {p} of text.\n\n" for p in range(n))
            for d, n in enumerate((60, 80, 140))
        ]

    def test_every_character_lands_in_exactly_one_split(self):
        text, parts = self.split(self.docs)
        offset = 0
        for i, doc in enumerate(self.docs):
            rebuilt = parts["train"][i] + parts["val"][i] + parts["test"][i]
            self.assertEqual(rebuilt, text[offset : offset + len(doc)])
            offset += len(doc) + 2

    def test_no_window_straddles_a_split_boundary(self):
        # Leakage check: the end of train must not reappear at the start of val.
        _, parts = self.split(self.docs)
        for i in range(len(self.docs)):
            tail = parts["train"][i][-64:]
            self.assertNotIn(tail, parts["val"][i])

    def test_every_document_appears_in_every_split(self):
        # The point of splitting within documents rather than between them:
        # validation must measure language modelling, not domain transfer.
        _, parts = self.split(self.docs)
        for name in ("train", "val", "test"):
            for i, piece in enumerate(parts[name]):
                self.assertTrue(piece.strip(), f"document {i} is empty in {name}")
                self.assertIn(f"doc{i} ", piece)

    def test_boundaries_sit_on_blank_lines(self):
        _, parts = self.split(self.docs)
        for name in ("train", "val"):
            for piece in parts[name]:
                self.assertTrue(piece.endswith("\n\n"), f"{name} piece ends mid-paragraph")

    def test_short_document_still_reaches_every_split(self):
        # Regression: `snap` only moves forward, so on a document with few
        # paragraph breaks both boundaries landed on the last one and the test
        # piece came out empty -- a document trained on but never evaluated on.
        short = "".join(f"tiny paragraph {p}.\n\n" for p in range(20))
        _, parts = self.split([short])
        for name in ("train", "val", "test"):
            self.assertTrue(parts[name][0].strip(), f"{name} piece is empty")
        rebuilt = parts["train"][0] + parts["val"][0] + parts["test"][0]
        self.assertEqual(rebuilt, short)

    def test_split_proportions_are_approximately_respected(self):
        _, parts = self.split(self.docs)
        for i, doc in enumerate(self.docs):
            share = len(parts["train"][i]) / len(doc)
            self.assertGreater(share, 0.85)
            self.assertLess(share, 0.97)


class TestCorpusRegistry(unittest.TestCase):
    def test_gutenberg_corpus_is_registered_and_pinned(self):
        corpus = data.CORPORA["gutenberg-mixed-27"]
        self.assertIsInstance(corpus, data.MultiDocCorpus)
        self.assertEqual(len(corpus.documents), 27)
        self.assertEqual(len(corpus.sha256), 64, "corpus hash must be pinned")
        ids = [d.doc_id for d in corpus.documents]
        self.assertEqual(len(set(ids)), len(ids), "duplicate document ids")
        for d in corpus.documents:
            self.assertTrue(d.repo.startswith("GITenberg/"))
            self.assertTrue(d.member.endswith(".txt"))
            self.assertTrue(d.title and d.author and d.year)


if __name__ == "__main__":
    unittest.main()

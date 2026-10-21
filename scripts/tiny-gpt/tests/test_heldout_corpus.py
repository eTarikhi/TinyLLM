"""Pin the held-out probe corpus.

Its whole value is that no model has trained on it. That is a property a
future corpus edit could silently destroy -- adding one of these titles to a
training corpus would turn every probe number into a training-fit number
with nothing failing loudly. So the disjointness is a test, not a comment.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import data  # noqa: E402

HELDOUT = data.CORPORA["gutenberg-heldout-forms"]
SELECTION = (ROOT.parent.parent / "experiments" / "2026-10-18-gutenberg-mixed"
             / "heldout-selection.json")


class HeldoutCorpusTest(unittest.TestCase):
    def test_counts(self):
        self.assertEqual(len(HELDOUT.documents), 40)
        ids = [d.doc_id for d in HELDOUT.documents]
        self.assertEqual(len(ids), len(set(ids)))

    def test_selection_matches_the_registry(self):
        selection = json.loads(SELECTION.read_text())
        self.assertEqual({f"pg{s['pg_id']}" for s in selection},
                         {d.doc_id for d in HELDOUT.documents})

    def test_ten_titles_per_form(self):
        counts = {}
        for s in json.loads(SELECTION.read_text()):
            counts[s["form"]] = counts.get(s["form"], 0) + 1
        self.assertEqual(counts, {"drama": 10, "verse": 10,
                                   "non-fiction": 10, "narrative": 10})

    def test_no_document_is_held_back_from_evaluation(self):
        # The probe needs a test piece from every title.
        self.assertFalse(any(d.train_only for d in HELDOUT.documents))

    def test_disjoint_from_every_training_corpus(self):
        heldout_ids = {d.doc_id for d in HELDOUT.documents}
        for name, corpus in data.CORPORA.items():
            if name == HELDOUT.name:
                continue
            trained_on = {d.doc_id for d in corpus.documents if not d.eval_only}
            self.assertFalse(heldout_ids & trained_on,
                             f"{name} trains on held-out titles: "
                             f"{sorted(heldout_ids & trained_on)}")


if __name__ == "__main__":
    unittest.main()

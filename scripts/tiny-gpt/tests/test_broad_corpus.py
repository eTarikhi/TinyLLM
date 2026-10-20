"""Pin the corpus-composition arms.

The experiment's validity rests on three things that are easy to break by
accident and silent when broken: the two arms must evaluate on exactly the
same documents, the additions must never reach val or test, and the mixture
must actually be the mixture the write-up claims. Each gets a test.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import data  # noqa: E402

FULL = data.CORPORA["gutenberg-mixed-broad-full"]
CTRL = data.CORPORA["gutenberg-mixed-broad-ctrl"]
ADDED = data.GUTENBERG_BROAD_DOCS
ORIGINAL_71 = data.GUTENBERG_DOCS + data.GUTENBERG_EXTRA_DOCS


class BroadCorpusShapeTest(unittest.TestCase):
    def test_counts(self):
        self.assertEqual(len(ORIGINAL_71), 71)
        self.assertEqual(len(ADDED), 80)
        self.assertEqual(len(FULL.documents), 151)
        self.assertEqual(len(CTRL.documents), 71)

    def test_document_ids_are_unique(self):
        for corpus in (FULL, CTRL):
            ids = [d.doc_id for d in corpus.documents]
            self.assertEqual(len(ids), len(set(ids)), f"{corpus.name} repeats a doc_id")

    def test_additions_do_not_collide_with_the_original_71(self):
        self.assertFalse({d.doc_id for d in ADDED} & {d.doc_id for d in ORIGINAL_71})


class EvaluationIsHeldFixedTest(unittest.TestCase):
    """The additions are training mass. They must not move the measurement."""

    def test_every_addition_is_train_only_in_the_treatment(self):
        added = {d.doc_id for d in ADDED}
        for doc in FULL.documents:
            if doc.doc_id in added:
                self.assertTrue(doc.train_only, f"{doc.doc_id} would enter val/test")
                self.assertFalse(doc.eval_only)

    def test_the_control_omits_the_additions_rather_than_evaluating_them(self):
        # eval_only would put them *into* val and test, which is the opposite
        # of what this control is for. Regression test for exactly that bug.
        added = {d.doc_id for d in ADDED}
        self.assertFalse(added & {d.doc_id for d in CTRL.documents})
        self.assertFalse(any(d.eval_only for d in CTRL.documents))

    def test_both_arms_evaluate_on_the_same_71_documents(self):
        def evaluated(corpus):
            return [d.doc_id for d in corpus.documents if not d.train_only]
        self.assertEqual(evaluated(FULL), evaluated(CTRL))
        self.assertEqual(len(evaluated(FULL)), 71)

    def test_the_control_is_the_published_71_book_text(self):
        self.assertEqual(CTRL.sha256, data.GUTENBERG_EXPANDED.sha256)


class MixtureTest(unittest.TestCase):
    """Guard the shares the write-up and the config headers quote."""

    EXPECTED_FORMS = {"drama": 26, "verse": 27, "non-fiction": 27}

    def _forms(self):
        import json
        path = (ROOT.parent.parent / "experiments" / "2026-10-18-gutenberg-mixed"
                / "broad-selection.json")
        return json.loads(path.read_text())

    def test_selection_matches_the_registry(self):
        selection = self._forms()
        self.assertEqual(len(selection), len(ADDED))
        self.assertEqual({f"pg{s['pg_id']}" for s in selection},
                         {d.doc_id for d in ADDED})

    def test_form_counts(self):
        counts = {}
        for s in self._forms():
            counts[s["form"]] = counts.get(s["form"], 0) + 1
        self.assertEqual(counts, self.EXPECTED_FORMS)

    def test_narrative_share_lands_near_two_thirds(self):
        existing = 60_467_324          # gutenberg-mixed-71, characters
        narrative = existing * 0.9066   # from document-summary.json
        added = sum(s["raw_bytes"] for s in self._forms())
        share = narrative / (existing + added)
        # The design target was two thirds; assert the band, not the digits,
        # so re-selecting the mixture does not have to be a doc edit.
        self.assertGreater(share, 0.63)
        self.assertLess(share, 0.72)

    def test_no_single_addition_dominates_a_form(self):
        by_form = {}
        for s in self._forms():
            by_form.setdefault(s["form"], []).append(s["raw_bytes"])
        for form, sizes in by_form.items():
            self.assertLess(max(sizes) / sum(sizes), 0.5,
                            f"one title is more than half of {form}")


class MemberFileTest(unittest.TestCase):
    def test_member_paths_are_well_formed(self):
        for doc in ADDED:
            prefix, _, name = doc.member.partition("/")
            self.assertTrue(prefix.endswith(doc.doc_id[2:] + "-master"), doc.member)
            pg = doc.doc_id[2:]
            self.assertIn(name, {f"{pg}-0.txt", f"{pg}.txt", f"{pg}-8.txt"}, doc.member)

    def test_the_utf8_member_is_preferred_where_it_exists(self):
        # Verification tries <id>-0.txt first, so anything pinned to the
        # legacy name is a repository that genuinely lacks the UTF-8 one.
        canonical = sum(1 for d in ADDED if d.member.endswith(f"{d.doc_id[2:]}-0.txt"))
        self.assertGreater(canonical, 0)
        self.assertLess(canonical, len(ADDED))


if __name__ == "__main__":
    unittest.main()

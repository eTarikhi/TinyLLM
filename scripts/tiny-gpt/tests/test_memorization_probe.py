"""Tests for the memorization probe.

Two defects found after the second training run, both of which produced a
*reassuring* report rather than an obviously broken one:

1. ``load_train_text`` only understood the single-file raw cache. On the
   27-document corpus it returned "" and the probe was skipped, yet the report
   still printed "longest verbatim match: 0 characters" -- indistinguishable
   from a model that had memorized nothing.
2. ``longest_common_substring_length`` materialised a set of every n-gram in
   the corpus at each binary-search step. At 17 MB that is gigabytes of live
   strings; it only survived review because the first corpus was 1 MB.
"""

import json
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluate import build_report, load_train_text, longest_common_substring_length  # noqa: E402


def brute_force_lcs(sample: str, corpus: str, max_len: int = 400) -> int:
    """Obvious O(n^3) reference implementation, used as the oracle."""
    for length in range(min(len(sample), max_len), 0, -1):
        for i in range(len(sample) - length + 1):
            if sample[i : i + length] in corpus:
                return length
    return 0


class TestLongestCommonSubstring(unittest.TestCase):
    def test_matches_brute_force_on_random_inputs(self):
        rng = random.Random(0)
        for _ in range(200):
            corpus = "".join(rng.choice("abc ") for _ in range(200))
            sample = "".join(rng.choice("abc ") for _ in range(40))
            if rng.random() < 0.5:  # plant a verbatim span half the time
                j, k = rng.randrange(150), rng.randrange(3, 20)
                sample = sample[:10] + corpus[j : j + k] + sample[10:]
            self.assertEqual(
                longest_common_substring_length(sample, corpus),
                brute_force_lcs(sample, corpus),
            )

    def test_finds_planted_span_exactly(self):
        corpus = "the quick brown fox jumps over the lazy dog" * 20
        self.assertEqual(longest_common_substring_length("zzz quick brown fox zzz", corpus), 17)

    def test_no_match_returns_zero(self):
        self.assertEqual(longest_common_substring_length("qqqq", "abcabcabc"), 0)

    def test_respects_max_len_cap(self):
        shared = "abcdefghij" * 10
        self.assertEqual(longest_common_substring_length(shared, shared, max_len=25), 25)

    def test_scales_to_a_large_corpus(self):
        """Regression: the old n-gram-set implementation allocated O(corpus)
        strings per step and could not run this at all on a small box."""
        rng = random.Random(1)
        corpus = "".join(rng.choice("abcdefg ") for _ in range(2_000_000))
        sample = corpus[1_234_567 : 1_234_567 + 55] + "zzzz"
        self.assertEqual(longest_common_substring_length(sample, corpus), 55)


class TestLoadTrainText(unittest.TestCase):
    def _manifest(self, name, boundaries, records=None):
        return {
            "corpus": {"name": name, "document_records": records or []},
            "split_policy": {"boundaries_chars": boundaries},
        }

    def test_single_document_cache(self):
        with _tmpdir() as root:
            raw = root / "raw"
            raw.mkdir(parents=True)
            (raw / "solo.txt").write_text("A" * 100 + "B" * 50, encoding="utf-8")
            text = load_train_text(root / "prepared", self._manifest("solo", [100, 125]))
            self.assertEqual(text, "A" * 100)

    def test_multi_document_cache_applies_per_document_split(self):
        with _tmpdir() as root:
            raw = root / "raw" / "multi"
            raw.mkdir(parents=True)
            (raw / "d1.txt").write_text("x" * 10 + "HELD_OUT", encoding="utf-8")
            (raw / "d2.txt").write_text("y" * 20 + "HELD_OUT", encoding="utf-8")
            manifest = self._manifest(
                "multi", [[10, 14], [20, 24]], [{"doc_id": "d1"}, {"doc_id": "d2"}]
            )
            text = load_train_text(root / "prepared", manifest)
            self.assertEqual(text, "x" * 10 + "\n\n" + "y" * 20)
            # The probe must never see held-out text, or it would report a
            # verbatim "match" against data the model was never trained on.
            self.assertNotIn("HELD_OUT", text)

    def test_missing_cache_returns_empty_rather_than_guessing(self):
        with _tmpdir() as root:
            (root / "raw" / "multi").mkdir(parents=True)
            manifest = self._manifest("multi", [[10, 14]], [{"doc_id": "absent"}])
            self.assertEqual(load_train_text(root / "prepared", manifest), "")


class TestUnmeasuredProbeIsReportedHonestly(unittest.TestCase):
    """The failure mode that hid the bug: a skipped probe that reads as a pass."""

    def _results(self, memorization):
        stub = {"loss": 1.0, "perplexity": 2.0, "bpb": 1.0, "tokens": 10, "bytes": 10}
        return {
            "run_id": "unit-test",
            "split": "test",
            "checkpoint": "x.pt",
            "step": 1,
            "tokens_trained": 1,
            "data_dir": "d",
            "tokenizer": {"kind": "byte_bpe", "vocab_size": 2048},
            "model": stub,
            "untrained": stub,
            "uniform": stub,
            "unigram": stub,
            "bigram": stub,
            "memorization": memorization,
        }

    def test_skipped_probe_says_not_measured(self):
        report = build_report(
            self._results(
                {
                    "samples": 0,
                    "tokens_per_sample": 0,
                    "max_longest_verbatim_match_chars": None,
                    "mean_longest_verbatim_match_chars": None,
                    "example_sample": "",
                    "measured": False,
                    "reason": "the training text could not be reconstructed from data/raw",
                }
            )
        )
        self.assertIn("Not measured", report)
        self.assertNotIn("**0 characters**", report)

    def test_real_probe_reports_the_number(self):
        report = build_report(
            self._results(
                {
                    "samples": 4,
                    "tokens_per_sample": 240,
                    "max_longest_verbatim_match_chars": 26,
                    "mean_longest_verbatim_match_chars": 23.75,
                    "example_sample": "hello",
                }
            )
        )
        self.assertIn("**26 characters**", report)
        self.assertNotIn("Not measured", report)


class _tmpdir:
    def __enter__(self):
        import tempfile

        self._t = tempfile.TemporaryDirectory()
        return Path(self._t.name)

    def __exit__(self, *exc):
        self._t.cleanup()
        return False


if __name__ == "__main__":
    unittest.main()

"""Gate 1: generation emits valid ids, respects limits, and stops at EOS."""

import unittest

import torch

from _helpers import TINY_TEXT, tiny_model

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tokenizer import CharTokenizer  # noqa: E402


class TestGeneration(unittest.TestCase):
    def test_only_valid_ids_and_correct_length(self):
        model = tiny_model()
        start = torch.zeros((2, 1), dtype=torch.long)
        g = torch.Generator().manual_seed(0)
        out = model.generate(start, max_new_tokens=20, temperature=1.0, top_k=5, generator=g)
        self.assertEqual(out.shape, (2, 21))
        self.assertTrue(int(out.min()) >= 0)
        self.assertTrue(int(out.max()) < model.cfg.vocab_size)

    def test_stops_at_eos(self):
        model = tiny_model()
        eos = 7
        # Force the model to always pick `eos` by zeroing the head and biasing it.
        with torch.no_grad():
            model.lm_head.weight.zero_()
            bias = torch.full((model.cfg.vocab_size,), -10.0)
            bias[eos] = 10.0
            model.lm_head.bias = torch.nn.Parameter(bias)
        start = torch.zeros((1, 1), dtype=torch.long)
        out = model.generate(start, max_new_tokens=25, temperature=0.0, eos_id=eos)
        self.assertEqual(out.shape[1], 2, "generation should stop immediately after EOS")
        self.assertEqual(int(out[0, -1]), eos)

    def test_never_exceeds_block_size_context(self):
        model = tiny_model()
        long_start = torch.zeros((1, model.cfg.block_size + 5), dtype=torch.long)
        out = model.generate(long_start, max_new_tokens=3, temperature=0.0)
        self.assertEqual(out.shape[1], model.cfg.block_size + 8)

    def test_same_seed_same_sample(self):
        model = tiny_model()
        start = torch.zeros((1, 1), dtype=torch.long)
        a = model.generate(start, 15, temperature=0.9, top_k=5, generator=torch.Generator().manual_seed(42))
        b = model.generate(start, 15, temperature=0.9, top_k=5, generator=torch.Generator().manual_seed(42))
        self.assertTrue(torch.equal(a, b))

    def test_greedy_is_deterministic(self):
        model = tiny_model()
        start = torch.zeros((1, 1), dtype=torch.long)
        a = model.generate(start, 10, temperature=0.0)
        b = model.generate(start, 10, temperature=0.0)
        self.assertTrue(torch.equal(a, b))

    def test_decode_round_trips_and_drops_specials(self):
        tok = CharTokenizer.train(TINY_TEXT)
        ids = tok.encode("to be or not to be", add_bos=True, add_eos=True)
        self.assertEqual(ids[0], tok.bos_id)
        self.assertEqual(ids[-1], tok.eos_id)
        self.assertEqual(tok.decode(ids), "to be or not to be")

    def test_generation_leaves_model_in_training_mode_as_found(self):
        model = tiny_model()
        model.train()
        model.generate(torch.zeros((1, 1), dtype=torch.long), 3, temperature=0.0)
        self.assertTrue(model.training)


if __name__ == "__main__":
    unittest.main()

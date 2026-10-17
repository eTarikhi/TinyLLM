"""Gate 1: logits are [batch, sequence, vocab] and the loss is a finite scalar."""

import unittest

import torch

from _helpers import random_batch, tiny_config, tiny_model


class TestShapes(unittest.TestCase):
    def test_logits_and_loss_shapes(self):
        model = tiny_model()
        x, y = random_batch(model.cfg, batch_size=3)
        logits, loss = model(x, y)
        self.assertEqual(tuple(logits.shape), (3, model.cfg.block_size, model.cfg.vocab_size))
        self.assertEqual(loss.shape, torch.Size([]))
        self.assertTrue(torch.isfinite(loss))

    def test_loss_is_none_without_targets(self):
        model = tiny_model()
        x, _ = random_batch(model.cfg)
        logits, loss = model(x)
        self.assertIsNone(loss)
        self.assertTrue(torch.isfinite(logits).all())

    def test_initial_loss_is_near_uniform(self):
        """A randomly initialised model should score about ln(V) nats/token.

        If it does not, something is wrong with initialisation or the head long
        before any training question matters.
        """
        import math

        model = tiny_model()
        x, y = random_batch(model.cfg, batch_size=8)
        _, loss = model(x, y)
        self.assertAlmostEqual(float(loss), math.log(model.cfg.vocab_size), delta=0.35)

    def test_rejects_sequence_longer_than_block_size(self):
        model = tiny_model()
        x = torch.zeros((1, model.cfg.block_size + 1), dtype=torch.long)
        with self.assertRaises(ValueError):
            model(x)

    def test_tying_changes_the_parameter_count_by_exactly_the_embedding(self):
        """The tied/untied choice is not cosmetic; the experiment plan records it."""
        tied = tiny_model(tie_embeddings=True)
        untied = tiny_model(tie_embeddings=False)
        cfg = tiny_config()
        self.assertEqual(
            untied.num_params() - tied.num_params(), cfg.vocab_size * cfg.n_embd
        )
        self.assertIs(tied.transformer.wte.weight, tied.lm_head.weight)
        self.assertIsNot(untied.transformer.wte.weight, untied.lm_head.weight)

    def test_param_breakdown_sums_to_total(self):
        model = tiny_model()
        breakdown = model.param_breakdown()
        self.assertEqual(breakdown["total"], model.num_params())


if __name__ == "__main__":
    unittest.main()

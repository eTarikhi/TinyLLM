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

    def test_flops_per_token_matches_the_documented_formula(self):
        """`6N + 12·L·T·d`, derived here independently of model.py.

        Per layer and per sequence of T tokens, attention does two matmuls that
        no parameter count sees: `q @ k^T` and `attn @ v`, each `2·T²·d`
        multiply-accumulate FLOPs forward. Backward costs about twice forward,
        so `3 × 2 × (2·T²·d) = 12·T²·d` per layer per sequence, which is
        `12·T·d` per layer per *token*.

        references/compute-budget-example.md once dropped the `d` factor, which
        understated the term by three orders of magnitude at realistic widths.
        This test fails if either the code or that derivation drifts.
        """
        cfg = tiny_config()
        model = tiny_model()
        attention = 12.0 * cfg.n_layer * cfg.block_size * cfg.n_embd
        self.assertAlmostEqual(
            model.flops_per_token(), 6.0 * model.num_params() + attention, places=3
        )
        self.assertAlmostEqual(
            model.flops_per_token(include_attention=False), 6.0 * model.num_params(), places=3
        )
        # The term is linear in width: a model twice as wide pays twice as much
        # per token for attention, which is exactly what dropping `d` hid.
        wide = tiny_model(n_embd=cfg.n_embd * 2, n_head=cfg.n_head)
        self.assertAlmostEqual(
            wide.flops_per_token() - 6.0 * wide.num_params(), 2.0 * attention, places=3
        )


if __name__ == "__main__":
    unittest.main()

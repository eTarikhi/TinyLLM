"""Gate 1: the future cannot reach the past.

The cheapest way to ship a language model that quietly cheats is to get the
mask wrong. The test is behavioural, not structural: change a token at position
t and assert that logits at positions < t do not move.
"""

import unittest

import torch

from _helpers import tiny_model


class TestCausalMask(unittest.TestCase):
    def test_future_tokens_do_not_change_earlier_logits(self):
        model = tiny_model()
        model.eval()
        T = model.cfg.block_size
        g = torch.Generator().manual_seed(7)
        x = torch.randint(0, model.cfg.vocab_size, (2, T), generator=g)

        with torch.no_grad():
            base, _ = model(x)

        cut = T // 2
        x2 = x.clone()
        # Replace everything from `cut` onwards with different tokens.
        x2[:, cut:] = (x2[:, cut:] + 3) % model.cfg.vocab_size
        self.assertFalse(torch.equal(x, x2))

        with torch.no_grad():
            changed, _ = model(x2)

        torch.testing.assert_close(base[:, :cut], changed[:, :cut], rtol=0, atol=1e-6)
        self.assertFalse(torch.allclose(base[:, cut:], changed[:, cut:], atol=1e-6))

    def test_prefix_logits_match_full_sequence_logits(self):
        """Running a prefix alone must give the same logits as the full sequence.

        This is the property that makes incremental generation sound.
        """
        model = tiny_model()
        model.eval()
        T = model.cfg.block_size
        g = torch.Generator().manual_seed(11)
        x = torch.randint(0, model.cfg.vocab_size, (1, T), generator=g)
        with torch.no_grad():
            full, _ = model(x)
            prefix, _ = model(x[:, : T // 2])
        torch.testing.assert_close(full[:, : T // 2], prefix, rtol=0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()

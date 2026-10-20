"""Gate 2: the model can deliberately overfit one tiny batch.

If a model cannot memorise four sequences it will never learn a corpus, and the
bug is in the code, not the hyperparameters. This is the fastest learning proof
available and it runs in a few seconds on a CPU.
"""

import math
import unittest

import torch

from _helpers import random_batch, tiny_model


class TestOverfit(unittest.TestCase):
    def test_overfits_single_batch(self):
        model = tiny_model(seed=3)
        x, y = random_batch(model.cfg, batch_size=4, seed=3)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.99))

        _, first = model(x, y)
        start = float(first.detach())
        self.assertGreater(start, math.log(model.cfg.vocab_size) - 0.5)

        for _ in range(300):
            _, loss = model(x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        end = float(loss)
        self.assertLess(end, 0.05, f"loss only fell from {start:.3f} to {end:.3f}")
        self.assertLess(end, start / 20)

    def test_predicts_the_memorised_batch(self):
        """After overfitting, argmax predictions should match the targets."""
        model = tiny_model(seed=5)
        x, y = random_batch(model.cfg, batch_size=2, seed=5)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.99))
        for _ in range(300):
            _, loss = model(x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            logits, _ = model(x)
        accuracy = (logits.argmax(-1) == y).float().mean()
        self.assertGreater(float(accuracy), 0.99)


if __name__ == "__main__":
    unittest.main()

"""Gate 1: an optimizer step changes parameters and leaves nothing non-finite.

Also checks the learning-rate schedule, because a schedule that silently never
reaches its peak (or never decays) is a bug that looks like a bad model.
"""

import math
import sys
import unittest
from pathlib import Path

import torch

from _helpers import random_batch, tiny_model

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from train import TrainConfig, build_optimizer, lr_at  # noqa: E402


class TestOptimizerStep(unittest.TestCase):
    def test_step_changes_parameters_and_stays_finite(self):
        model = tiny_model(seed=1)
        x, y = random_batch(model.cfg, seed=1)
        before = [p.detach().clone() for p in model.parameters()]
        cfg = TrainConfig(learning_rate=1e-3)
        opt = build_optimizer(model, cfg)

        _, loss = model(x, y)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        self.assertTrue(math.isfinite(float(grad_norm)))
        changed = sum(
            1 for b, p in zip(before, model.parameters()) if not torch.equal(b, p.detach())
        )
        self.assertEqual(changed, len(before), "every parameter should have moved")
        for name, p in model.named_parameters():
            self.assertTrue(torch.isfinite(p).all(), f"{name} went non-finite")

    def test_weight_decay_groups(self):
        """Matrices decay, norms and biases do not."""
        model = tiny_model(bias=True)
        cfg = TrainConfig(weight_decay=0.1)
        opt = build_optimizer(model, cfg)
        self.assertEqual(len(opt.param_groups), 2)
        self.assertAlmostEqual(opt.param_groups[0]["weight_decay"], 0.1)
        self.assertAlmostEqual(opt.param_groups[1]["weight_decay"], 0.0)
        self.assertTrue(all(p.dim() >= 2 for p in opt.param_groups[0]["params"]))
        self.assertTrue(all(p.dim() < 2 for p in opt.param_groups[1]["params"]))
        n_unique = len({id(p) for g in opt.param_groups for p in g["params"]})
        n_total = sum(len(g["params"]) for g in opt.param_groups)
        self.assertEqual(n_unique, n_total, "a tied parameter must not be optimised twice")

    def test_schedule_warms_up_then_decays_to_the_floor(self):
        cfg = TrainConfig(learning_rate=1e-3, min_lr_frac=0.1, warmup_steps=10, max_steps=100)
        self.assertLess(lr_at(0, cfg), cfg.learning_rate)
        self.assertAlmostEqual(lr_at(cfg.warmup_steps - 1, cfg), cfg.learning_rate, places=9)
        self.assertAlmostEqual(lr_at(cfg.max_steps - 1, cfg), cfg.learning_rate * cfg.min_lr_frac, delta=2e-5)
        # monotone decay after warmup
        values = [lr_at(s, cfg) for s in range(cfg.warmup_steps, cfg.max_steps)]
        self.assertTrue(all(a >= b for a, b in zip(values, values[1:])))

    def test_decay_horizon_equals_training_horizon(self):
        """references/training-defaults.md: lr_decay_iters == max_iters in both anchors."""
        cfg = TrainConfig(learning_rate=1e-3, min_lr_frac=0.1, warmup_steps=5, max_steps=50)
        self.assertGreater(lr_at(cfg.max_steps - 2, cfg), lr_at(cfg.max_steps - 1, cfg))


if __name__ == "__main__":
    unittest.main()

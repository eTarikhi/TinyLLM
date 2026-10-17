"""Gate 1: a checkpoint round-trip reproduces logits, and saving is atomic."""

import sys
import unittest
from pathlib import Path

import torch

from _helpers import TempDir, random_batch, tiny_model

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model import GPT, GPTConfig  # noqa: E402
from train import TrainConfig, build_optimizer, load_checkpoint, save_checkpoint  # noqa: E402


class TestCheckpoint(unittest.TestCase):
    def test_round_trip_reproduces_logits(self):
        model = tiny_model(seed=2)
        model.eval()
        x, y = random_batch(model.cfg, seed=2)
        with torch.no_grad():
            before, _ = model(x)

        cfg = TrainConfig()
        opt = build_optimizer(model, cfg)
        with TempDir() as tmp:
            path = Path(tmp) / "ckpt.pt"
            save_checkpoint(path, model, opt, cfg, step=7, tokens=1234, best_val=1.5, extra={})
            self.assertTrue(path.exists())
            self.assertFalse(path.with_suffix(".pt.tmp").exists(), "temp file left behind")

            ckpt = load_checkpoint(path)
            restored = GPT(GPTConfig(**ckpt["model_config"]))
            restored.load_state_dict(ckpt["model_state"])
            restored.eval()
            with torch.no_grad():
                after, _ = restored(x)

        torch.testing.assert_close(before, after, rtol=0, atol=0)
        self.assertEqual(ckpt["step"], 7)
        self.assertEqual(ckpt["tokens"], 1234)

    def test_checkpoint_carries_everything_a_resume_needs(self):
        model = tiny_model()
        cfg = TrainConfig()
        opt = build_optimizer(model, cfg)
        with TempDir() as tmp:
            path = Path(tmp) / "ckpt.pt"
            save_checkpoint(path, model, opt, cfg, step=3, tokens=99, best_val=2.0, extra={})
            ckpt = load_checkpoint(path)
        for key in ("model_state", "optimizer_state", "train_config", "model_config",
                    "step", "tokens", "best_val", "data_position", "rng", "env"):
            self.assertIn(key, ckpt, f"checkpoint is missing {key}")
        self.assertIn("torch", ckpt["rng"])
        self.assertIn("numpy", ckpt["rng"])
        self.assertEqual(ckpt["data_position"]["next_step"], 3)

    def test_optimizer_state_survives(self):
        model = tiny_model()
        x, y = random_batch(model.cfg)
        cfg = TrainConfig()
        opt = build_optimizer(model, cfg)
        _, loss = model(x, y)
        loss.backward()
        opt.step()
        with TempDir() as tmp:
            path = Path(tmp) / "ckpt.pt"
            save_checkpoint(path, model, opt, cfg, step=1, tokens=1, best_val=9.9, extra={})
            ckpt = load_checkpoint(path)
            fresh = GPT(GPTConfig(**ckpt["model_config"]))
            fresh_opt = build_optimizer(fresh, cfg)
            fresh_opt.load_state_dict(ckpt["optimizer_state"])
        state = next(iter(fresh_opt.state.values()))
        self.assertIn("exp_avg", state)
        self.assertEqual(int(state["step"]), 1)


if __name__ == "__main__":
    unittest.main()

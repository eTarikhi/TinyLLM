"""Gate 1: an interrupted run that resumes is the same run.

Restores model, optimizer, schedule position, step, RNG state and data
position. The data position is the strict part: a resumed run must see the
batches the uninterrupted run would have seen, otherwise the comparison between
two runs at a fixed token budget is quietly wrong.
"""

import sys
import unittest
from pathlib import Path

import torch

from _helpers import TempDir, make_prepared_dir

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data import TokenBatcher  # noqa: E402
from train import TrainConfig, train  # noqa: E402


def _config(tmp: Path, data_dir: Path, run_id: str, max_steps: int, stop_after_steps: int = 0) -> TrainConfig:
    return TrainConfig(
        run_id=run_id,
        data_dir=str(data_dir),
        out_dir=str(tmp / "runs"),
        stop_after_steps=stop_after_steps,
        block_size=16,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.0,
        batch_size=4,
        learning_rate=1e-3,
        warmup_steps=2,
        max_steps=max_steps,
        eval_every=0,
        log_every=1000,
        save_every=0,
        keep_best=False,
        eval_tokens=256,
        seed=99,
    )


class TestResume(unittest.TestCase):
    def test_batches_are_a_pure_function_of_seed_and_step(self):
        with TempDir() as tmp:
            data_dir = make_prepared_dir(tmp / "data")
            a = TokenBatcher(data_dir / "train.bin", 16, 4, seed=5)
            b = TokenBatcher(data_dir / "train.bin", 16, 4, seed=5)
            c = TokenBatcher(data_dir / "train.bin", 16, 4, seed=6)
            xa, ya = a.batch(12)
            xb, yb = b.batch(12)
            xc, _ = c.batch(12)
            self.assertTrue(torch.equal(xa, xb) and torch.equal(ya, yb))
            self.assertFalse(torch.equal(xa, xc))
            # targets are inputs shifted by one
            self.assertTrue(torch.equal(xa[:, 1:], ya[:, :-1]))

    def test_resumed_run_matches_uninterrupted_run(self):
        """Interrupt at step 4 of an 8-step run, resume, and land in the same place.

        Note what the interrupted run must *not* change: its horizon. Training
        with ``max_steps=4`` and then continuing with ``max_steps=8`` is a
        different learning-rate schedule, so it is a different run. The
        interruption here is ``stop_after_steps``, which leaves the schedule
        alone -- the trainer prints a warning if a resume changes it.
        """
        with TempDir() as tmp:
            data_dir = make_prepared_dir(tmp / "data")

            full = train(_config(tmp, data_dir, "full", max_steps=8))
            first_half = train(_config(tmp, data_dir, "half", max_steps=8, stop_after_steps=4))
            self.assertEqual(first_half["steps"], 4)
            resumed_cfg = _config(tmp, data_dir, "half", max_steps=8)
            resumed = train(resumed_cfg, resume_from=str(Path(first_half["checkpoints"]["last"])))

            self.assertEqual(full["steps"], resumed["steps"])
            self.assertEqual(full["tokens"], resumed["tokens"])
            self.assertAlmostEqual(full["final_val_loss"], resumed["final_val_loss"], places=5)

            a = torch.load(full["checkpoints"]["last"], map_location="cpu", weights_only=False)
            b = torch.load(resumed["checkpoints"]["last"], map_location="cpu", weights_only=False)
            for key in a["model_state"]:
                torch.testing.assert_close(
                    a["model_state"][key], b["model_state"][key], rtol=1e-5, atol=1e-6,
                    msg=f"parameter {key} differs after resume",
                )


if __name__ == "__main__":
    unittest.main()

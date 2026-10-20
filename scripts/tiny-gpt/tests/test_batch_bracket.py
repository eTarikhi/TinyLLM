"""The batch-size bracket is only meaningful if its three arms stay matched.

A controlled experiment lives in the *differences* between its configs, and
those differences are four numbers spread across three YAML files that nothing
in the repository previously checked. Editing `max_steps` in one arm, or
changing `block_size` for everyone, silently turns the comparison into a
different and much weaker one -- and the runs still complete, still write
plausible metrics, and still get written up.

So the design is encoded here as executable invariants:

    batch-4096     4,096 tok/step   9,600 steps
    data71-full    8,192 tok/step   4,800 steps   (the centre)
    batch-16384   16,384 tok/step   2,400 steps

  * every arm sees exactly 39,321,600 tokens
  * warmup is 5% of each arm's own step budget, so the schedule has the same
    shape in normalised time
  * every arm performs exactly 40 evaluations, and the k-th evaluation of each
    arm lands on the same cumulative token count -- this is what lets the
    curves be compared token-for-token with no interpolation
  * the arms differ in `run_id`, `grad_accum` and the four schedule counts
    derived from the step budget, and in nothing else

The last one is the real guard: it fails if someone "fixes" the learning rate
or the seed in one arm only.

These tests read YAML and do arithmetic. Nothing is trained or downloaded.
"""
from __future__ import annotations

import unittest
from pathlib import Path

import yaml

CONFIGS = Path(__file__).resolve().parents[1] / "configs"

TOKEN_BUDGET = 39_321_600
EVALUATIONS = 40
WARMUP_FRACTION = 0.05

# tokens per optimizer update -> config stem, ordered small to large
ARMS = ((4_096, "batch-4096"), (8_192, "data71-full"), (16_384, "batch-16384"))

# Keys that are *expected* to differ between arms. Everything else must match.
EXPECTED_DIFFERENCES = {
    "run_id", "grad_accum", "max_steps", "warmup_steps", "eval_every",
    "log_every", "sample_every", "save_every", "max_seconds",
}


def load(stem: str) -> dict:
    return yaml.safe_load((CONFIGS / f"{stem}.yaml").read_text())


class BatchBracketTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cfgs = {stem: load(stem) for _, stem in ARMS}

    def test_every_arm_exists(self) -> None:
        for _, stem in ARMS:
            self.assertTrue((CONFIGS / f"{stem}.yaml").is_file(), f"{stem}.yaml is missing")

    def test_tokens_per_step_is_what_the_arm_claims(self) -> None:
        """The arm names are a claim about tokens per update; check the claim."""
        for tokens_per_step, stem in ARMS:
            c = self.cfgs[stem]
            computed = c["batch_size"] * c["grad_accum"] * c["block_size"]
            self.assertEqual(computed, tokens_per_step,
                             f"{stem}: {computed} tokens/step, not {tokens_per_step}")

    def test_token_budget_is_identical(self) -> None:
        """The whole experiment is 'same tokens, different number of updates'."""
        for tokens_per_step, stem in ARMS:
            c = self.cfgs[stem]
            self.assertEqual(tokens_per_step * c["max_steps"], TOKEN_BUDGET,
                             f"{stem} does not see the shared {TOKEN_BUDGET:,}-token budget")

    def test_warmup_is_the_same_fraction_of_every_run(self) -> None:
        for _, stem in ARMS:
            c = self.cfgs[stem]
            self.assertEqual(c["warmup_steps"], round(WARMUP_FRACTION * c["max_steps"]),
                             f"{stem}: warmup is not {WARMUP_FRACTION:.0%} of its step budget")

    def test_each_arm_evaluates_forty_times(self) -> None:
        for _, stem in ARMS:
            c = self.cfgs[stem]
            self.assertEqual(c["max_steps"] % c["eval_every"], 0,
                             f"{stem}: eval_every does not divide max_steps")
            self.assertEqual(c["max_steps"] // c["eval_every"], EVALUATIONS,
                             f"{stem} does not perform exactly {EVALUATIONS} evaluations")

    def test_evaluation_grids_coincide_in_tokens(self) -> None:
        """The comparison is token-for-token, so the grids must actually align.

        Equal counts are not enough: 40 evaluations each could still land on 40
        different token counts. This builds the grids and intersects them.
        """
        grids = []
        for tokens_per_step, stem in ARMS:
            c = self.cfgs[stem]
            grids.append({k * c["eval_every"] * tokens_per_step
                          for k in range(1, EVALUATIONS + 1)})
        common = set.intersection(*grids)
        self.assertEqual(len(common), EVALUATIONS,
                         "the arms' evaluation grids do not coincide in tokens")
        self.assertEqual(max(common), TOKEN_BUDGET,
                         "the last common evaluation is not the end of training")

    def test_arms_differ_only_where_intended(self) -> None:
        centre = self.cfgs["data71-full"]
        for _, stem in ARMS:
            if stem == "data71-full":
                continue
            c = self.cfgs[stem]
            self.assertEqual(set(c), set(centre), f"{stem} has different keys from the centre")
            differing = {k for k in centre if centre[k] != c[k]}
            unexpected = differing - EXPECTED_DIFFERENCES
            self.assertEqual(unexpected, set(),
                             f"{stem} differs from the centre in {sorted(unexpected)}, "
                             "which would confound the batch-size comparison")

    def test_the_confounding_knobs_are_held_fixed(self) -> None:
        """Named explicitly, because these are the ones a future edit will touch."""
        for key in ("learning_rate", "min_lr_frac", "weight_decay", "beta1", "beta2",
                    "grad_clip", "seed", "block_size", "n_layer", "n_head", "n_embd",
                    "dropout", "tie_embeddings", "batch_size", "data_dir",
                    "eval_tokens", "eval_spread"):
            values = {stem: self.cfgs[stem][key] for _, stem in ARMS}
            self.assertEqual(len(set(map(repr, values.values()))), 1,
                             f"{key} differs across the arms: {values}")

    def test_arms_are_distinguishable(self) -> None:
        """A bracket with two identical arms is a bug, not an experiment."""
        accums = [self.cfgs[stem]["grad_accum"] for _, stem in ARMS]
        self.assertEqual(len(set(accums)), len(ARMS), "two arms share a batch size")
        self.assertEqual(accums, sorted(accums), "ARMS is not ordered small to large")
        run_ids = {self.cfgs[stem]["run_id"] for _, stem in ARMS}
        self.assertEqual(len(run_ids), len(ARMS), "two arms would write to the same run dir")


if __name__ == "__main__":
    unittest.main()

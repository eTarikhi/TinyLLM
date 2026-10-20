#!/usr/bin/env python3
"""Batch size at a fixed token budget: the sharpest test of the steps hypothesis.

Two results in `clean-data-scaling.md` say the binding constraint on this
corpus is optimization steps rather than parameters or tokens:

    capacity   2.11x the parameters at matched wall clock is 1.35% WORSE,
               because the bigger model pays for capacity with 46% fewer steps
    horizon    2x the steps at fixed batch bought 3.00%

Batch size is the only lever that trades steps against batch at a FIXED token
budget, so it separates "steps" from "tokens" in a way neither of the other
two can. All three arms see exactly 39,321,600 tokens:

    batch-4096     4,096 tok/step   9,600 steps
    data71-full    8,192 tok/step   4,800 steps   (the centre, already run)
    batch-16384   16,384 tok/step   2,400 steps

The schedule shape is identical in normalised time -- warmup is 5% of every
run and each has exactly 40 evaluations -- so the k-th evaluation of every arm
lands on the same cumulative token count and the curves can be compared
token-for-token with no interpolation. They cannot be compared step-for-step,
and this script never does.

What the script reports:

  * the endpoint table, val and test, with elapsed seconds and throughput
  * the margin between arms in seed-bar units, since 0.37% is the noise floor
  * the token-matched curve comparison across the common 40-point grid
  * clip counts, which test the gradient-noise story behind any batch effect:
    a larger batch averages more samples per update, so it should clip less
  * cost per unit of accuracy -- tokens are matched but seconds are not, and
    an arm that wins on accuracy while also being cheaper wins twice

Usage (from the repository root):
    python experiments/2026-10-18-gutenberg-mixed/analyze-batch.py
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Measured in data-seed-summary.json on the 27-book test split; the best
# estimate available for run-to-run noise. Indicative here, not exact.
SEED_BAR_PCT = 0.3690475821746498

# tokens per optimizer update -> run id. Ordered small to large.
ARMS = ((4096, "batch-4096"), (8192, "data71-full"), (16384, "batch-16384"))
TOKEN_BUDGET = 39_321_600


def line(ch: str = "=") -> None:
    print(ch * 96)


def summary(run: str) -> dict:
    return json.loads((HERE / "metrics" / f"{run}-summary.json").read_text())


def evals(run: str) -> list[dict]:
    path = HERE / "metrics" / f"{run}.jsonl"
    return [json.loads(l) for l in path.read_text().splitlines() if '"eval"' in l]


def test_eval(run: str) -> dict:
    return json.loads((HERE / f"eval-test-{run}.json").read_text())


def main() -> int:
    present = [(tps, run) for tps, run in ARMS
               if (HERE / "metrics" / f"{run}-summary.json").is_file()
               and (HERE / f"eval-test-{run}.json").is_file()]
    missing = [run for tps, run in ARMS if (tps, run) not in present]
    if missing:
        print(f"missing arms, nothing to compare: {', '.join(missing)}")
        return 1

    out: dict = {"token_budget": TOKEN_BUDGET, "seed_bar_pct": SEED_BAR_PCT, "arms": {}}

    line()
    print("BATCH SIZE AT A FIXED TOKEN BUDGET")
    print("  every arm sees 39,321,600 tokens; they differ only in how many updates that buys")
    line()
    print(f"{'run':>14} {'tok/step':>9} {'steps':>7} {'tokens':>12} {'elapsed':>8} "
          f"{'tok/s':>7} {'clip':>5} {'val':>8} {'test':>8}")
    for tps, run in present:
        s, t = summary(run), test_eval(run)
        assert s["tokens"] == TOKEN_BUDGET, f"{run} saw {s['tokens']} tokens, not the budget"
        out["arms"][run] = {
            "tokens_per_step": tps, "steps": s["steps"], "tokens": s["tokens"],
            "elapsed_s": s["elapsed_s"], "tokens_per_s": s["mean_tokens_per_s"],
            "clipped_steps": s["clipped_steps"],
            "clipped_pct": 100 * s["clipped_steps"] / s["steps"],
            "val_bpb": s["final_val_bpb"], "test_bpb": t["model"]["bpb"],
        }
        a = out["arms"][run]
        print(f"{run:>14} {tps:>9,} {s['steps']:>7,} {s['tokens']:>12,} "
              f"{round(s['elapsed_s']):>8,} {round(a['tokens_per_s']):>7,} "
              f"{s['clipped_steps']:>5} {a['val_bpb']:>8.4f} {a['test_bpb']:>8.4f}")

    # ---- the verdict, against the centre and against the noise floor
    centre = "data71-full"
    base = out["arms"][centre]["test_bpb"]
    print(f"\n  against the centre arm ({centre}, {base:.4f} test bpb):")
    best, best_bpb = centre, base
    for tps, run in present:
        if run == centre:
            continue
        a = out["arms"][run]
        d = 100 * (a["test_bpb"] / base - 1)
        a["delta_vs_centre_pct"] = d
        a["delta_in_bar_units"] = abs(d) / SEED_BAR_PCT
        verdict = "INSIDE the seed bar" if abs(d) < SEED_BAR_PCT else f"{abs(d)/SEED_BAR_PCT:.1f}x the bar"
        print(f"    {run:>14}  {d:+.2f}%  ({verdict})")
        if a["test_bpb"] < best_bpb:
            best, best_bpb = run, a["test_bpb"]
    out["best_arm"] = best
    out["best_test_bpb"] = best_bpb

    spread = max(a["test_bpb"] for a in out["arms"].values()) / \
        min(a["test_bpb"] for a in out["arms"].values()) - 1
    out["spread_pct"] = 100 * spread
    out["spread_in_bar_units"] = 100 * spread / SEED_BAR_PCT
    print(f"\n  full spread across a 4x range of batch size: {100*spread:.2f}% "
          f"({100*spread/SEED_BAR_PCT:.1f}x the seed bar)")
    print(f"  best arm: {best} at {best_bpb:.4f}")

    # ---- the prediction this experiment was built to test
    small, large = present[0][1], present[-1][1]
    d_small = out["arms"][small].get("delta_vs_centre_pct", 0.0)
    d_large = out["arms"][large].get("delta_vs_centre_pct", 0.0)
    out["steps_hypothesis"] = {
        "statement": "at a fixed token budget, more optimizer updates should win",
        "small_batch_delta_pct": d_small,
        "large_batch_delta_pct": d_large,
        "monotone_in_steps": d_small < 0 < d_large,
        "supported": d_small < -SEED_BAR_PCT,
    }
    print("\n  the steps hypothesis predicted a monotone win for the smallest batch:")
    print(f"    4x fewer tokens per update {d_small:+.2f}%   4x more {d_large:+.2f}%   "
          f"monotone: {out['steps_hypothesis']['monotone_in_steps']}")

    # ---- clipping: the gradient-noise story, measured rather than asserted
    print("\n  gradient clipping (a larger batch averages more samples, so should clip less):")
    for tps, run in present:
        a = out["arms"][run]
        print(f"    {run:>14}  {a['clipped_steps']:>5} of {a['steps']:>5,} steps "
              f"({a['clipped_pct']:.2f}%)")
    cl = [out["arms"][run]["clipped_pct"] for _, run in present]
    out["clip_monotone_decreasing"] = all(x >= y for x, y in zip(cl, cl[1:]))
    print(f"    monotone decreasing in batch size: {out['clip_monotone_decreasing']}")

    # ---- token-matched curves across the common 40-point grid
    series = {}
    for tps, run in present:
        e = evals(run)
        # Key on cumulative tokens, not step: the arms share a token grid by
        # construction and share no step grid at all.
        series[run] = {r["tokens"]: r["val_bpb"] for r in e}
        for r in e:
            assert r["tokens"] == r["step"] * tps, (
                f"{run}: eval at step {r['step']} logged {r['tokens']} tokens, "
                f"expected {r['step'] * tps} -- the arms are not token-matched")
    grids = [set(v) for v in series.values()]
    common = sorted(set.intersection(*grids))
    out["common_token_points"] = len(common)
    print(f"\n  common token-matched evaluation points: {len(common)}")
    if common:
        leaders = [min(series, key=lambda r: series[r][t]) for t in common]
        held = {run: leaders.count(run) for _, run in present}
        out["lead_counts"] = held
        print("  evaluations led, by arm:", ", ".join(f"{k} {v}" for k, v in held.items()))
        changes = sum(1 for a, b in zip(leaders, leaders[1:]) if a != b)
        out["lead_changes"] = changes
        print(f"  lead changes: {changes}")
        tail = common[-1]
        out["final_common_point"] = {"tokens": tail,
                                     **{run: series[run][tail] for _, run in present}}
        print(f"  at the final common point ({tail:,} tokens): " +
              ", ".join(f"{run} {series[run][tail]:.4f}" for _, run in present))

    # ---- cost: tokens are matched, seconds are not
    print("\n  wall clock was not held fixed -- only tokens were:")
    ref = out["arms"][centre]["elapsed_s"]
    for tps, run in present:
        a = out["arms"][run]
        a["elapsed_vs_centre_pct"] = 100 * (a["elapsed_s"] / ref - 1)
        print(f"    {run:>14}  {round(a['elapsed_s']):>5,} s  "
              f"({a['elapsed_vs_centre_pct']:+.1f}% vs the centre)")
    b = out["arms"][best]
    if best != centre:
        print(f"\n  the winning arm is {b['elapsed_vs_centre_pct']:+.1f}% on wall clock "
              f"for {b['delta_vs_centre_pct']:+.2f}% on test bits per byte")

    # ---- gradient norm at matched training loss
    #
    # Comparing raw gradient norms across the arms confounds two things: a
    # noisier gradient, and simply being less far along. Binning by training
    # loss removes the second -- two samples at the same loss are at
    # comparable points in the optimization, whatever step produced them.
    print("\n  median pre-clip gradient norm, binned by training loss:")
    # Gradient noise is a property of the batch, not of the learning rate, so
    # the LR-matched arms belong in this table too: if the small batch is
    # noisy at matched loss whatever its rate, the noise story holds and the
    # clip counts have to be explained some other way.
    gnorm_runs = [run for _, run in present] + [
        r for r in ("batch-4096-lr2p12e3", "batch-16384-lr4p24e3")
        if (HERE / "metrics" / f"{r}.jsonl").is_file()]
    print(f"    {'loss band':>12} " + "".join(f"{run[-13:]:>15}" for run in gnorm_runs))
    bands = ((3.6, 3.9), (3.3, 3.6), (3.0, 3.3))
    pts = {run: [(r["loss"], r["grad_norm_preclip"])
                 for r in (json.loads(l) for l in
                           (HERE / "metrics" / f"{run}.jsonl").read_text().splitlines())
                 if r.get("event") == "train"]
           for run in gnorm_runs}
    gn: dict[str, dict] = {}
    for lo_, hi_ in bands:
        row, label = [], f"[{lo_}, {hi_})"
        for run in gnorm_runs:
            g = sorted(v for l_, v in pts[run] if lo_ <= l_ < hi_)
            med = statistics.median(g) if g else float("nan")
            gn.setdefault(run, {})[label] = {"n": len(g), "median_gnorm": med}
            row.append(f"{med:>15.3f}")
        print(f"    {label:>12} " + "".join(row))
    out["grad_norm_by_loss_band"] = gn
    tight = bands[-1]
    lbl = f"[{tight[0]}, {tight[1]})"
    base_g = gn[centre][lbl]["median_gnorm"]
    for run in gnorm_runs:
        if run == centre:
            continue
        gn[run][lbl]["vs_centre_pct"] = 100 * (gn[run][lbl]["median_gnorm"] / base_g - 1)
        print(f"    {run} in {lbl}: {gn[run][lbl]['vs_centre_pct']:+.0f}% vs the centre")
    print("    (binning by loss rules out 'it is simply less converged' as the explanation)")

    # ---- the learning-rate follow-up, present only once those runs exist
    PAIRS = (("batch-4096", "batch-4096-lr2p12e3"), ("batch-16384", "batch-16384-lr4p24e3"))
    pairs = [(f, s) for f, s in PAIRS
             if (HERE / "metrics" / f"{s}-summary.json").is_file()
             and (HERE / f"eval-test-{s}.json").is_file()]
    if pairs:
        line()
        print("LEARNING RATE, SQUARE-ROOT SCALED — does the confound explain the bracket?")
        line()
        print(f"{'run':>22} {'lr':>9} {'clip':>6} {'val':>8} {'test':>8} {'vs centre':>10} "
              f"{'vs twin':>9}")
        print(f"{centre:>22} {3.0e-3:>9.2e} {out['arms'][centre]['clipped_steps']:>6} "
              f"{out['arms'][centre]['val_bpb']:>8.4f} {base:>8.4f} {'--':>10} {'--':>9}")
        lr_out: dict = {}
        for fixed, scaled in pairs:
            s_, t_ = summary(scaled), test_eval(scaled)
            tb = t_["model"]["bpb"]
            d_centre = 100 * (tb / base - 1)
            d_twin = 100 * (tb / out["arms"][fixed]["test_bpb"] - 1)
            lr_out[scaled] = {
                "twin": fixed, "steps": s_["steps"], "tokens": s_["tokens"],
                "elapsed_s": s_["elapsed_s"], "clipped_steps": s_["clipped_steps"],
                "clipped_pct": 100 * s_["clipped_steps"] / s_["steps"],
                "val_bpb": s_["final_val_bpb"], "test_bpb": tb,
                "delta_vs_centre_pct": d_centre, "delta_vs_twin_pct": d_twin,
                "closes_the_gap": abs(d_centre) < SEED_BAR_PCT,
            }
            assert s_["tokens"] == TOKEN_BUDGET, f"{scaled} is off the token budget"
            lr = 2.12e-3 if "4096" in scaled else 4.24e-3
            print(f"{scaled:>22} {lr:>9.2e} {s_['clipped_steps']:>6} "
                  f"{s_['final_val_bpb']:>8.4f} {tb:>8.4f} {d_centre:>+9.2f}% {d_twin:>+8.2f}%")
        out["lr_matched"] = lr_out
        closed = [k for k, v in lr_out.items() if v["closes_the_gap"]]
        out["lr_confound_explains_bracket"] = len(closed) == len(lr_out) and len(lr_out) == 2
        print()
        if out["lr_confound_explains_bracket"]:
            print("  both outer arms reach the centre once their rate is scaled:")
            print("  the first bracket measured a learning-rate mismatch, not a batch-size effect")
        elif closed:
            print(f"  scaling the rate rescues {', '.join(closed)} but not the rest:")
            print("  part of the bracket was an LR mismatch and part is a real batch-size effect")
        else:
            print("  neither outer arm reaches the centre at its own scaled rate:")
            print("  the interior optimum survives the confound -- 8,192 tok/step is")
            print("  genuinely near the critical batch size for this model and corpus")

    path = HERE / "batch-summary.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(f"\nwritten to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

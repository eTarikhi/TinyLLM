#!/usr/bin/env python3
"""Derive the data-scaling and seed-bar tables from raw metrics.

Two experiments that only mean something together: a 3.53x larger training
split, and three draws of the released configuration that say how large a
difference has to be before it means anything at all.

Run from the repository root:

    python experiments/2026-10-18-gutenberg-mixed/analyze-data-seed.py
"""
from __future__ import annotations

import json
import pathlib
import statistics
import sys

E = pathlib.Path("experiments/2026-10-18-gutenberg-mixed")
M = E / "metrics"

REFERENCE = "main-gutenberg-4800"
SEEDS = [(REFERENCE, 1337, "eval-test-4800.json"),
         ("seed-1338", 1338, "eval-test-seed-1338.json"),
         ("seed-1339", 1339, "eval-test-seed-1339.json")]


def rows(run: str) -> list[dict]:
    return [json.loads(l) for l in (M / f"{run}.jsonl").read_text().splitlines() if l.strip()]


def summary(run: str) -> dict:
    return json.loads((M / f"{run}-summary.json").read_text())


def evals(run: str) -> list[dict]:
    return [r for r in rows(run) if r["event"] == "eval"]


def load(name: str) -> dict:
    return json.loads((E / name).read_text())


def spread(values: list[float]) -> dict:
    lo, hi = min(values), max(values)
    mean = statistics.mean(values)
    return {
        "values": values, "mean": mean, "min": lo, "max": hi,
        "range": hi - lo, "range_pct": 100 * (hi - lo) / mean,
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "stdev_pct": 100 * statistics.stdev(values) / mean if len(values) > 1 else 0.0,
    }


def main() -> int:
    out: dict = {}

    # ---------------------------------------------------------------- seeds
    print("=" * 96)
    print("SEED BAR — the released configuration, three draws, nothing but the seed changed")
    print("=" * 96)
    print(f"{'seed':>6} {'steps':>6} {'val bpb':>9} {'test bpb':>9} {'gap':>8} "
          f"{'clipped':>8} {'verbatim':>9} {'elapsed':>8}")
    seed_rows = []
    for run, seed, evf in SEEDS:
        su, ev, e = summary(run), load(evf), evals(run)
        rec = {
            "run": run, "seed": seed, "steps": su["steps"],
            "val_bpb": su["final_val_bpb"], "test_bpb": ev["model"]["bpb"],
            "end_gap": e[-1]["train_eval_loss"] - e[-1]["val_loss"],
            "clipped_steps": su["clipped_steps"],
            "longest_verbatim": ev["memorization"]["max_longest_verbatim_match_chars"],
            "elapsed_s": su["elapsed_s"], "tokens": su["tokens"],
        }
        seed_rows.append(rec)
        print(f"{seed:>6} {rec['steps']:>6,} {rec['val_bpb']:>9.4f} {rec['test_bpb']:>9.4f} "
              f"{rec['end_gap']:>+8.4f} {rec['clipped_steps']:>8,} {rec['longest_verbatim']:>9} "
              f"{rec['elapsed_s']:>8.0f}")

    val_spread = spread([r["val_bpb"] for r in seed_rows])
    test_spread = spread([r["test_bpb"] for r in seed_rows])
    print(f"\n  val  mean {val_spread['mean']:.4f}  range {val_spread['range']:.4f} "
          f"({val_spread['range_pct']:.2f}%)  sd {val_spread['stdev']:.4f} ({val_spread['stdev_pct']:.2f}%)")
    print(f"  test mean {test_spread['mean']:.4f}  range {test_spread['range']:.4f} "
          f"({test_spread['range_pct']:.2f}%)  sd {test_spread['stdev']:.4f} ({test_spread['stdev_pct']:.2f}%)")

    # Every margin previously reported in this directory, re-read against the
    # seed range. This is the point of the exercise.
    base_test = load("eval-test-4800.json")["model"]["bpb"]
    verdicts = []
    for label, evf in (("size 3.74M", "eval-test-size-4L256d.json"),
                       ("size 5.64M", "eval-test-size-5L288d.json"),
                       ("context T=128", "eval-test-ctx-128.json"),
                       ("context T=512", "eval-test-ctx-512.json"),
                       ("context T=1024", "eval-test-ctx-1024.json"),
                       ("lr 1.5e-3", "eval-test-lr-1p5e3.json"),
                       ("lr 6.0e-3", "eval-test-lr-6e3.json")):
        if not (E / evf).exists():
            continue
        margin = 100 * (load(evf)["model"]["bpb"] / base_test - 1)
        inside = margin <= test_spread["range_pct"]
        verdicts.append({"label": label, "margin_pct": margin,
                         "inside_seed_range": bool(inside)})
    print(f"\n  margins previously reported, against a seed range of {test_spread['range_pct']:.2f}%:")
    for v in verdicts:
        mark = "INSIDE  — not distinguishable" if v["inside_seed_range"] else "outside — a real difference"
        print(f"    {v['label']:<16} {v['margin_pct']:+6.2f}%   {mark}")

    out["seeds"] = seed_rows
    out["val_spread"] = val_spread
    out["test_spread"] = test_spread
    out["margin_verdicts"] = verdicts

    # ----------------------------------------------------------------- data
    print("\n" + "=" * 96)
    print("DATA SCALING — 3.53x the training split, identical everything else")
    print("=" * 96)
    manifests = {}
    for key, path in (("gb27", "data-manifest.json"), ("gb71", "data-manifest-gb71.json")):
        if (E / path).exists():
            manifests[key] = load(path)

    data_rows = []
    for run, evf, label in ((REFERENCE, "eval-test-4800.json", "27 books"),
                            ("data-gb71", "eval-test-data-gb71.json", "71 books")):
        su, ev, e = summary(run), load(evf), evals(run)
        key = "gb27" if run == REFERENCE else "gb71"
        train_tokens = manifests[key]["splits"]["train"]["tokens"] if key in manifests else None
        rec = {
            "run": run, "label": label, "train_tokens": train_tokens,
            "epochs": su["tokens"] / train_tokens if train_tokens else None,
            "tokens_trained": su["tokens"],
            "val_bpb": su["final_val_bpb"], "test_bpb": ev["model"]["bpb"],
            "end_gap": e[-1]["train_eval_loss"] - e[-1]["val_loss"],
            "longest_verbatim": ev["memorization"]["max_longest_verbatim_match_chars"],
            "mean_longest_verbatim": ev["memorization"]["mean_longest_verbatim_match_chars"],
            "clipped_steps": su["clipped_steps"], "elapsed_s": su["elapsed_s"],
            "peak_rss_mb": su["peak_rss_mb"],
        }
        data_rows.append(rec)
        print(f"  {label:<10} train {rec['train_tokens']:>12,} tok  {rec['epochs']:>5.2f} ep  "
              f"val {rec['val_bpb']:.4f}  test {rec['test_bpb']:.4f}  gap {rec['end_gap']:+.4f}  "
              f"verbatim {rec['longest_verbatim']:>3}")

    small, large = data_rows
    delta_test = 100 * (large["test_bpb"] / small["test_bpb"] - 1)
    delta_val = 100 * (large["val_bpb"] / small["val_bpb"] - 1)
    print(f"\n  71 books vs 27: {delta_test:+.2f}% test bpb, {delta_val:+.2f}% val bpb")
    print(f"  seed range on the same configuration: {test_spread['range_pct']:.2f}% test")
    sig = abs(delta_test) > test_spread["range_pct"]
    print(f"  => {'OUTSIDE the seed range: a real effect' if sig else 'INSIDE the seed range: not distinguishable'}")

    # Where in training does the difference appear? Both runs share the grid.
    grid = sorted({r["step"] for r in evals(REFERENCE)} & {r["step"] for r in evals("data-gb71")})
    s_series = {r["step"]: r["val_bpb"] for r in evals(REFERENCE)}
    l_series = {r["step"]: r["val_bpb"] for r in evals("data-gb71")}
    crossings = []
    prev = None
    for st in grid:
        leader = "71 books" if l_series[st] < s_series[st] else "27 books"
        if leader != prev:
            crossings.append({"step": st, "leader": leader})
            prev = leader
    print(f"\n  {len(grid)} common evaluation steps; lead changes: " +
          ", ".join(f"step {c['step']} -> {c['leader']}" for c in crossings))
    trail = [{"step": st, "gb27": s_series[st], "gb71": l_series[st],
              "delta_pct": 100 * (l_series[st] / s_series[st] - 1)} for st in grid[-6:]]
    print("  last six evaluations:")
    for t in trail:
        print(f"    step {t['step']:>5}  27 books {t['gb27']:.4f}  71 books {t['gb71']:.4f}  {t['delta_pct']:+.2f}%")

    out["data"] = data_rows
    out["data_delta_test_pct"] = delta_test
    out["data_delta_val_pct"] = delta_val
    out["data_outside_seed_range"] = bool(sig)
    out["data_lead_changes"] = crossings
    out["data_common_eval_steps"] = len(grid)
    out["data_tail"] = trail

    # How many seed standard deviations is the data effect? A percentage is only
    # meaningful next to the noise floor measured on the same configuration.
    out["data_sigma_val"] = (large["val_bpb"] - val_spread["mean"]) / val_spread["stdev"]
    out["data_sigma_test"] = (large["test_bpb"] - test_spread["mean"]) / test_spread["stdev"]
    print(f"\n  in units of the seed sd: val {out['data_sigma_val']:+.1f} sd, "
          f"test {out['data_sigma_test']:+.1f} sd  (n=3 seeds -- suggestive, not a p-value)")

    # --------------------------------------------------------------
    # Why: are the 44 additions the same distribution, or a different one?
    # --------------------------------------------------------------
    shift_path = E / "distribution-shift.json"
    if shift_path.exists():
        shift = json.loads(shift_path.read_text())
        res = shift["results"]
        clean = {k: v for k, v in res.items() if v["new_books_held_out_for_this_model"]}
        clean_test = sum(v["test_27_books_bpb"] for v in clean.values()) / len(clean)
        clean_new = sum(v["new_44_books_bpb"] for v in clean.values()) / len(clean)
        gb71 = res["data-gb71"]
        print("\n" + "=" * 96)
        print("DISTRIBUTION SHIFT — how far the 44 additions sit from the frozen evaluation set")
        print("=" * 96)
        print(f"{'model':>22} {'sees new books?':>16} {'test(27 bk)':>12} {'new(44 bk)':>11} {'shift':>8}")
        for k, v in res.items():
            seen = "no (held out)" if v["new_books_held_out_for_this_model"] else "yes (trained on)"
            print(f"{k:>22} {seen:>16} {v['test_27_books_bpb']:>12.4f} "
                  f"{v['new_44_books_bpb']:>11.4f} {v['shift_pct']:>+7.2f}%")
        gain = 100 * (gb71["new_44_books_bpb"] / clean_new - 1)
        cost = 100 * (gb71["test_27_books_bpb"] / clean_test - 1)
        print(f"\n  a clean 27-book model finds the additions {100 * (clean_new / clean_test - 1):+.2f}% harder "
              f"than its own test split")
        print(f"  training on them moves the new-book score {gain:+.2f}% and the test score {cost:+.2f}%")
        print(f"  ratio: {abs(gain / cost):.0f}x more movement on the new distribution than on the old")
        print("  (the 71-book run trained on the probe, so its new-book number is a fit, not a generalisation)")
        out["shift"] = {
            "clean_test_bpb": clean_test, "clean_new_bpb": clean_new,
            "clean_shift_pct": 100 * (clean_new / clean_test - 1),
            "gb71_new_bpb": gb71["new_44_books_bpb"], "gb71_test_bpb": gb71["test_27_books_bpb"],
            "gb71_new_gain_pct": gain, "gb71_test_cost_pct": cost,
            "gain_cost_ratio": abs(gain / cost),
            "probe_tokens": shift["probe"]["tokens"], "probe_docs": shift["probe"]["docs"],
        }

    (E / "data-seed-summary.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwritten to {E / 'data-seed-summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

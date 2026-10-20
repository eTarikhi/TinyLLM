#!/usr/bin/env python3
"""Derive the context- and learning-rate-sweep tables from raw metrics.

Everything the reports quote is computed here from `metrics/*.jsonl`,
`metrics/*-summary.json` and `eval-test-*.json` so no number is transcribed by
hand. Run from the repository root:

    python experiments/2026-10-18-gutenberg-mixed/analyze-sweep.py
"""
from __future__ import annotations

import json
import pathlib
import statistics
import sys

E = pathlib.Path("experiments/2026-10-18-gutenberg-mixed")
M = E / "metrics"


def rows(run: str) -> list[dict]:
    return [json.loads(line) for line in (M / f"{run}.jsonl").read_text().splitlines() if line.strip()]


def summary(run: str) -> dict:
    return json.loads((M / f"{run}-summary.json").read_text())


def evaltest(name: str) -> dict:
    return json.loads((E / f"eval-test-{name}.json").read_text())


def evals(run: str) -> list[dict]:
    return [r for r in rows(run) if r["event"] == "eval"]


def trains(run: str) -> list[dict]:
    return [r for r in rows(run) if r["event"] == "train"]


def bpb_at_tokens(run: str, target: int) -> float | None:
    """Linear interpolation of val bpb against tokens seen."""
    pts = [(r["tokens"], r["val_bpb"]) for r in evals(run)]
    if not pts or target < pts[0][0] or target > pts[-1][0]:
        return None
    for (t0, b0), (t1, b1) in zip(pts, pts[1:]):
        if t0 <= target <= t1:
            if t1 == t0:
                return b1
            return b0 + (b1 - b0) * (target - t0) / (t1 - t0)
    return None


def clean_rate(run: str) -> float:
    """Median throughput over train->train intervals that contain no eval.

    The per-record `tokens_per_s` window is reset before the evaluation block,
    so any window spanning an evaluation also carries its cost. Runs in this
    directory use different evaluation cadences, so comparing them needs a rate
    that excludes evaluation entirely.
    """
    rs = rows(run)
    eval_steps = {r["step"] for r in rs if r["event"] == "eval"}
    tr = trains(run)
    out = []
    for a, b in zip(tr, tr[1:]):
        if any(a["step"] < s <= b["step"] for s in eval_steps):
            continue
        dt = b["elapsed_s"] - a["elapsed_s"]
        if dt > 0:
            out.append((b["tokens"] - a["tokens"]) / dt)
    return statistics.median(out)


def quarters(run: str) -> list[float]:
    tr = trains(run)
    n = len(tr)
    out = []
    for q in range(4):
        chunk = tr[q * n // 4: (q + 1) * n // 4]
        if len(chunk) < 2:
            out.append(float("nan"))
            continue
        dt = chunk[-1]["elapsed_s"] - chunk[0]["elapsed_s"]
        out.append((chunk[-1]["tokens"] - chunk[0]["tokens"]) / dt)
    return out


def gnorms(run: str) -> dict:
    g = [r["grad_norm_preclip"] for r in trains(run)]
    return {"median": statistics.median(g), "max": max(g), "p90": sorted(g)[int(0.9 * len(g))]}


CONTEXT = [("ctx-128", 128), ("main-gutenberg-4800", 256), ("ctx-512", 512), ("ctx-1024", 1024)]
CONTEXT_EVAL = {"ctx-128": "ctx-128", "main-gutenberg-4800": "4800", "ctx-512": "ctx-512", "ctx-1024": "ctx-1024"}
LR = [("lr-1p5e3", 1.5e-3), ("main-gutenberg-4800", 3.0e-3), ("lr-6e3", 6.0e-3)]
LR_EVAL = {"lr-1p5e3": "lr-1p5e3", "main-gutenberg-4800": "4800", "lr-6e3": "lr-6e3"}


def main() -> int:
    out: dict = {}

    print("=" * 100)
    print("CONTEXT SWEEP (four runs, tokens/step held at 8,192)")
    print("=" * 100)
    print(f"{'ctx':>6} {'params':>10} {'steps':>6} {'elapsed':>8} {'tokens':>12} {'val bpb':>8} "
          f"{'test bpb':>9} {'gap':>8} {'tok/s':>7} {'clean':>7} {'RSS':>6} {'clip':>5}")
    ctx_rows = []
    for run, T in CONTEXT:
        s, e = summary(run), evals(run)
        et = evaltest(CONTEXT_EVAL[run])
        # Sign convention matches src/train.py and the earlier reports:
        # positive means the training-split evaluation loss sits *above* the
        # validation loss, which is what every run in this directory shows.
        gap = e[-1]["train_eval_loss"] - e[-1]["val_loss"]
        rec = {
            "run": run, "block_size": T, "params": s["params_total"], "steps": s["steps"],
            "elapsed_s": s["elapsed_s"], "tokens": s["tokens"], "val_bpb": s["final_val_bpb"],
            "test_bpb": et["model"]["bpb"], "test_tokens": et["model"]["tokens"],
            "test_bytes": et["model"]["bytes"], "end_gap": gap,
            "mean_tokens_per_s": s["mean_tokens_per_s"], "clean_tokens_per_s": clean_rate(run),
            "peak_rss_mb": s["peak_rss_mb"], "clipped_steps": s["clipped_steps"],
            "quarter_tokens_per_s": quarters(run),
            "longest_verbatim": et["memorization"]["max_longest_verbatim_match_chars"],
            "epochs": s["tokens"] / 5771560,
        }
        ctx_rows.append(rec)
        print(f"{T:>6} {rec['params']:>10,} {rec['steps']:>6,} {rec['elapsed_s']:>8.0f} "
              f"{rec['tokens']:>12,} {rec['val_bpb']:>8.4f} {rec['test_bpb']:>9.4f} "
              f"{gap:>+8.4f} {rec['mean_tokens_per_s']:>7.0f} {rec['clean_tokens_per_s']:>7.0f} "
              f"{rec['peak_rss_mb']:>6.0f} {rec['clipped_steps']:>5}")
    base = next(r for r in ctx_rows if r["block_size"] == 256)
    for r in ctx_rows:
        r["test_delta_pct"] = 100 * (r["test_bpb"] - base["test_bpb"]) / base["test_bpb"]
        r["val_delta_pct"] = 100 * (r["val_bpb"] - base["val_bpb"]) / base["val_bpb"]
    print("\nversus T=256:  " + "   ".join(
        f"T={r['block_size']} {r['test_delta_pct']:+.2f}% test / {r['val_delta_pct']:+.2f}% val" for r in ctx_rows))

    floor = min(r["tokens"] for r in ctx_rows)
    print(f"\ntoken-matched at the common floor of {floor:,} tokens (val bpb):")
    for r in ctx_rows:
        b = bpb_at_tokens(r["run"], floor)
        r["val_bpb_at_floor"] = b
        frac = floor / r["tokens"]
        r["cosine_fraction_at_floor"] = frac
        print(f"  T={r['block_size']:>4}  {b:.4f}   ({frac:.1%} of its own cosine)")
    base_floor = next(r for r in ctx_rows if r["block_size"] == 256)["val_bpb_at_floor"]
    for r in ctx_rows:
        r["val_delta_pct_at_floor"] = 100 * (r["val_bpb_at_floor"] - base_floor) / base_floor
    print("  deltas: " + "   ".join(f"T={r['block_size']} {r['val_delta_pct_at_floor']:+.2f}%" for r in ctx_rows))

    print("\nper-quarter throughput (tok/s, including evaluation pauses):")
    for r in ctx_rows:
        q = r["quarter_tokens_per_s"]
        r["quarter_drift_pct"] = 100 * (q[-1] - q[0]) / q[0]
        print(f"  T={r['block_size']:>4}  " + "  ".join(f"{v:>7.0f}" for v in q) + f"   drift {r['quarter_drift_pct']:+.1f}%")

    print("\n" + "=" * 100)
    print("LEARNING-RATE BRACKET (three runs, identical except peak LR)")
    print("=" * 100)
    print(f"{'lr':>8} {'steps':>6} {'elapsed':>8} {'val bpb':>8} {'test bpb':>9} {'gap':>8} "
          f"{'clipped':>8} {'gnorm med':>10} {'gnorm max':>10} {'tok/s':>7} {'RSS':>6}")
    lr_rows = []
    for run, lr in LR:
        s, e = summary(run), evals(run)
        et = evaltest(LR_EVAL[run])
        g = gnorms(run)
        rec = {
            "run": run, "learning_rate": lr, "steps": s["steps"], "elapsed_s": s["elapsed_s"],
            "tokens": s["tokens"], "val_bpb": s["final_val_bpb"], "test_bpb": et["model"]["bpb"],
            "end_gap": e[-1]["train_eval_loss"] - e[-1]["val_loss"],
            "clipped_steps": s["clipped_steps"], "clipped_pct": 100 * s["clipped_steps"] / s["steps"],
            "gnorm_median": g["median"], "gnorm_p90": g["p90"], "gnorm_max": g["max"],
            "mean_tokens_per_s": s["mean_tokens_per_s"], "clean_tokens_per_s": clean_rate(run),
            "peak_rss_mb": s["peak_rss_mb"],
            "longest_verbatim": et["memorization"]["max_longest_verbatim_match_chars"],
        }
        lr_rows.append(rec)
        print(f"{lr:>8.1e} {rec['steps']:>6,} {rec['elapsed_s']:>8.0f} {rec['val_bpb']:>8.4f} "
              f"{rec['test_bpb']:>9.4f} {rec['end_gap']:>+8.4f} {rec['clipped_steps']:>8,} "
              f"{g['median']:>10.3f} {g['max']:>10.2f} {rec['mean_tokens_per_s']:>7.0f} {rec['peak_rss_mb']:>6.0f}")
    lrbase = next(r for r in lr_rows if r["learning_rate"] == 3.0e-3)
    for r in lr_rows:
        r["test_delta_pct"] = 100 * (r["test_bpb"] - lrbase["test_bpb"]) / lrbase["test_bpb"]
        r["val_delta_pct"] = 100 * (r["val_bpb"] - lrbase["val_bpb"]) / lrbase["val_bpb"]
    print("\nversus 3e-3:  " + "   ".join(
        f"{r['learning_rate']:.1e} {r['test_delta_pct']:+.2f}% test / {r['val_delta_pct']:+.2f}% val" for r in lr_rows))

    # Three points is enough for a vertex but not for a confidence interval:
    # the fit below is reported as an indication of where to look next, not as
    # a measurement.
    import math
    xs = [math.log10(r["learning_rate"]) for r in lr_rows]
    for key in ("val_bpb", "test_bpb"):
        ys = [r[key] for r in lr_rows]
        (x0, y0), (x1, y1), (x2, y2) = zip(xs, ys)
        d0 = (x0 - x1) * (x0 - x2)
        d1 = (x1 - x0) * (x1 - x2)
        d2 = (x2 - x0) * (x2 - x1)
        a = y0 / d0 + y1 / d1 + y2 / d2
        b = -(y0 * (x1 + x2) / d0 + y1 * (x0 + x2) / d1 + y2 * (x0 + x1) / d2)
        vertex = 10 ** (-b / (2 * a)) if a else float("nan")
        out[f"log_quadratic_vertex_{key}"] = vertex
        print(f"  log-quadratic vertex through the three {key} points: {vertex:.2e}")

    # Where does the ranking settle? Compare the three curves step for step.
    common = sorted(set.intersection(*[{r["step"] for r in evals(run)} for run, _ in LR]))
    series = {run: {r["step"]: r["val_bpb"] for r in evals(run)} for run, _ in LR}
    lead_changes = []
    prev = None
    for st in common:
        best = min(LR, key=lambda rl: series[rl[0]][st])[1]
        if best != prev:
            lead_changes.append({"step": st, "leader_lr": best})
            prev = best
    print(f"\n{len(common)} common evaluation steps; lead changes: " +
          ", ".join(f"step {c['step']} -> {c['leader_lr']:.1e}" for c in lead_changes))

    # Host-speed control: the two LR runs share shape, horizon and evaluation
    # cadence with the reference run, so their throughput measures the host.
    ref = summary("main-gutenberg-4800")["mean_tokens_per_s"]
    today = [r["mean_tokens_per_s"] for r in lr_rows if r["run"] != "main-gutenberg-4800"]
    host = {
        "reference_tokens_per_s": ref,
        "same_shape_today": today,
        "today_mean": statistics.mean(today),
        "slowdown_pct": 100 * (statistics.mean(today) - ref) / ref,
        "reference_clean": clean_rate("main-gutenberg-4800"),
        "today_clean_mean": statistics.mean(
            [r["clean_tokens_per_s"] for r in lr_rows if r["run"] != "main-gutenberg-4800"]),
    }
    host["clean_slowdown_pct"] = 100 * (host["today_clean_mean"] - host["reference_clean"]) / host["reference_clean"]
    print("\n" + "=" * 100)
    print("HOST-SPEED CONTROL (identical shape, horizon and eval cadence as the reference run)")
    print("=" * 100)
    print(f"  reference main-gutenberg-4800 : {ref:>8.0f} tok/s end-to-end, {host['reference_clean']:>8.0f} clean")
    for r in lr_rows:
        if r["run"] == "main-gutenberg-4800":
            continue
        print(f"  {r['run']:<30}: {r['mean_tokens_per_s']:>8.0f} tok/s end-to-end, {r['clean_tokens_per_s']:>8.0f} clean")
    print(f"  host is {host['slowdown_pct']:+.1f}% end-to-end / {host['clean_slowdown_pct']:+.1f}% clean "
          f"versus the session that ran the context sweep")

    # Throughput versus T=256, each point compared against a T=256 measured on
    # the *same* host. T=512 and T=1024 ran in the same session as the
    # reference; T=128 ran in the session the two LR runs measure.
    same_host_t256 = statistics.mean(
        [r["clean_tokens_per_s"] for r in lr_rows if r["run"] != "main-gutenberg-4800"])
    ref_t256 = clean_rate("main-gutenberg-4800")
    for r in ctx_rows:
        ref = same_host_t256 if r["block_size"] == 128 else ref_t256
        r["same_host_t256_clean"] = ref
        r["throughput_vs_t256_pct"] = 100 * (r["clean_tokens_per_s"] - ref) / ref
    print("\nmeasured throughput versus a same-host T=256 (clean, evaluation excluded):")
    for r in ctx_rows:
        print(f"  T={r['block_size']:>4}  {r['clean_tokens_per_s']:>7.0f} tok/s   "
              f"vs {r['same_host_t256_clean']:>7.0f}   {r['throughput_vs_t256_pct']:+6.1f}%")

    # T=128 against T=256 on the same host, same day: clean rates only.
    c128 = next(r for r in ctx_rows if r["block_size"] == 128)["clean_tokens_per_s"]
    print(f"\n  same-host throughput, T=128 {c128:.0f} clean tok/s vs T=256 {host['today_clean_mean']:.0f} "
          f"({100 * (c128 - host['today_clean_mean']) / host['today_clean_mean']:+.1f}%)")
    host["ctx128_clean"] = c128
    host["ctx128_vs_ctx256_pct"] = 100 * (c128 - host["today_clean_mean"]) / host["today_clean_mean"]

    # Probe accuracy, now with the fourth point measured.
    probe = json.loads((E / "context-probe.json").read_text())
    print("\nprobe accuracy (predicted end-to-end tok/s vs measured, host-corrected for T=128):")
    probe_rows = []
    for p in probe["results"]:
        T = p["block_size"]
        r = next((x for x in ctx_rows if x["block_size"] == T), None)
        if r is None:
            continue
        corr = 1.0 if T != 128 else (1 + host["clean_slowdown_pct"] / 100)
        pred = p["predicted_end_to_end_tok_s"] * corr
        err = 100 * (r["mean_tokens_per_s"] - pred) / pred
        probe_rows.append({"block_size": T, "predicted": p["predicted_end_to_end_tok_s"],
                           "host_corrected_prediction": pred, "measured": r["mean_tokens_per_s"],
                           "error_pct": err})
        print(f"  T={T:>4}  predicted {p['predicted_end_to_end_tok_s']:>7.0f}"
              f"  host-corrected {pred:>7.0f}  measured {r['mean_tokens_per_s']:>7.0f}  {err:+6.1f}%")

    out.update({"context": ctx_rows, "token_matched_floor": floor, "learning_rate": lr_rows,
           "lr_lead_changes": lead_changes, "lr_common_eval_steps": len(common),
                "host_control": host, "probe_accuracy": probe_rows})
    (E / "roadmap-sweep-summary.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwritten to {E / 'roadmap-sweep-summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

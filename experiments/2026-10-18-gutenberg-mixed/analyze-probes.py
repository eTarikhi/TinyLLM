"""Collect the four short-clock probe arms at 3.74M into broad-summary.json.

Each arm moves exactly one variable off broad-4L256d-full (2,571 steps,
seed 1337, broad corpus): peak LR (1.5e-3, 6e-3), context (512,
token-matched), and BPE vocabulary (4096, compared in bits/byte).

    python analyze-probes.py
"""
from __future__ import annotations

import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent

ARMS = {
    "lr_1.5e-3": "eval-test-big-lr15.json",
    "lr_6e-3": "eval-test-big-lr60.json",
    "ctx512": "eval-test-big-ctx512.json",
    "vocab4096": "eval-test-big-vocab4096.json",
}


def main() -> None:
    summary_path = HERE / "broad-summary.json"
    summary = json.loads(summary_path.read_text())
    baseline = summary["two_by_two"]["3.74M"]["full"]

    probes = {"baseline": baseline}
    for key, fname in ARMS.items():
        v = json.loads((HERE / fname).read_text())["model"]["bpb"]
        probes[key] = v
        probes[f"{key}_delta_pct"] = 100 * (v - baseline) / baseline

    summary["probes"] = probes
    summary_path.write_text(json.dumps(summary, indent=1) + "\n")

    print(f"baseline broad-4L256d-full (LR 3e-3, ctx 256, V2048): {baseline:.4f}")
    for key in ARMS:
        print(f"  {key:<12} {probes[key]:.4f}  {probes[f'{key}_delta_pct']:+.2f}%")
    print(f"\nwritten to {summary_path}")


if __name__ == "__main__":
    main()

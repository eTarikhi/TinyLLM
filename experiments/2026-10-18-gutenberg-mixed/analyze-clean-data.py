#!/usr/bin/env python3
"""The clean data-scaling experiment: same evaluation, different training sets.

`data-gb71` asked "does more training text help?" with the evaluation pinned to
the original 27 books, and the answer was no (+0.53% test bits per byte). The
probe in `distribution-shift.json` showed why: the 44 additions are 6.93%
harder for a 27-book model than its own test split, so they are a different
distribution rather than more of the same.

This asks the question properly. Both arms are evaluated on val and test slices
drawn from all 71 books, byte-identical between the two prepared corpora and
tokenized with a shared vocabulary:

    data71-full   18,939,699 train tokens   2.08 epochs   all 71 books train
    data71-ctrl    5,771,560 train tokens   6.81 epochs   27 of 71 books train

and then the test split is decomposed into the part that both arms' training
sets cover (the original 27 books) and the part only `full` covers (the 44
additions). That decomposition reproduces the earlier experiment and the new
one inside a single pair of runs, which is the point: whether more data helps
is a question about the evaluation, not only about the data.

Usage (from the repository root):
    python experiments/2026-10-18-gutenberg-mixed/analyze-clean-data.py
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
PKG = ROOT / "scripts" / "tiny-gpt"
sys.path.insert(0, str(PKG / "src"))

from data import TokenBatcher, load_tokenizer  # noqa: E402
from evaluate import evaluate_model  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402

ARMS = ("data71-full", "data71-ctrl")
FULL_DIR = PKG / "data" / "prepared-71-full"
N_ORIGINAL = 27  # the first 27 documents are the original corpus, in order

# Measured in data-seed-summary.json on the 27-book test split. It is the best
# estimate available for run-to-run noise; this experiment's test split is a
# different (3.2x larger) set of bytes, so the bar is indicative, not exact.
SEED_BAR_PCT = 0.3690475821746498


def evals(run: str) -> list[dict]:
    path = HERE / "metrics" / f"{run}.jsonl"
    return [json.loads(l) for l in path.read_text().splitlines() if '"eval"' in l]


def summary(run: str) -> dict:
    return json.loads((HERE / "metrics" / f"{run}-summary.json").read_text())


def test_eval(run: str) -> dict:
    return json.loads((HERE / f"eval-test-{run}.json").read_text())


def manifest(name: str) -> dict:
    return json.loads((PKG / "data" / name / "manifest.json").read_text())


def load_model(run: str) -> tuple[GPT, GPTConfig]:
    ckpt = torch.load(PKG / "artifacts" / "runs" / run / "best.pt",
                      map_location="cpu", weights_only=False)
    cfg = GPTConfig(**ckpt["model_config"])
    model = GPT(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg


def split_test_by_document_group(eos_id: int) -> tuple[Path, Path, dict]:
    """Cut test.bin into the original-27 part and the added-44 part.

    The split is the per-document test pieces joined by a single EOS, and EOS
    is only ever emitted as that separator, so splitting on it recovers the
    documents exactly. Each group is rewritten with the same internal EOS
    structure so neither side gains or loses a boundary token.
    """
    ids = np.fromfile(FULL_DIR / "test.bin", dtype=np.uint16)
    cuts = np.flatnonzero(ids == eos_id)
    pieces, prev = [], 0
    for c in cuts:
        pieces.append(ids[prev:c])
        prev = c + 1
    pieces.append(ids[prev:])
    if len(pieces) != 71:
        raise SystemExit(f"expected 71 test pieces, found {len(pieces)}")

    def join(group: list[np.ndarray]) -> np.ndarray:
        out: list[np.ndarray] = []
        for i, p in enumerate(group):
            out.append(p)
            if i < len(group) - 1:
                out.append(np.array([eos_id], dtype=np.uint16))
        return np.concatenate(out)

    old = join(pieces[:N_ORIGINAL])
    new = join(pieces[N_ORIGINAL:])
    p_old = PKG / "data" / "test-group-original27.bin"
    p_new = PKG / "data" / "test-group-added44.bin"
    old.tofile(p_old)
    new.tofile(p_new)
    stats = {
        "pieces_total": len(pieces),
        "original27_tokens": int(old.size),
        "added44_tokens": int(new.size),
    }
    return p_old, p_new, stats


def main() -> int:
    torch.set_num_threads(2)
    out: dict = {}

    print("=" * 96)
    print("CLEAN DATA SCALING — val and test drawn from all 71 books, identical across arms")
    print("=" * 96)

    mf, mc = manifest("prepared-71-full"), manifest("prepared-71-ctrl")
    for key in ("val", "test"):
        assert mf["splits"][key]["sha256_tokens"] == mc["splits"][key]["sha256_tokens"], key
    assert mf["tokenizer"]["sha256"] == mc["tokenizer"]["sha256"]
    print(f"  evaluation splits verified identical across arms "
          f"(val {mf['splits']['val']['tokens']:,} tok, test {mf['splits']['test']['tokens']:,} tok, "
          f"shared vocabulary {mf['tokenizer']['sha256'][:12]})")
    out["evaluation_identical"] = True
    out["val_tokens"] = mf["splits"]["val"]["tokens"]
    out["test_tokens"] = mf["splits"]["test"]["tokens"]

    rows = []
    print(f"\n{'arm':>12} {'train tok':>12} {'epochs':>7} {'val bpb':>9} {'test bpb':>9} "
          f"{'gap':>8} {'clip':>6} {'verb':>5} {'elapsed':>8}")
    for arm, train_name in zip(ARMS, ("prepared-71-full", "prepared-71-ctrl")):
        s, e, t = summary(arm), evals(arm), test_eval(arm)
        tr = manifest(train_name)["splits"]["train"]["tokens"]
        rec = {
            "run": arm,
            "train_tokens": tr,
            "epochs": s["tokens"] / tr,
            "val_bpb": s["final_val_bpb"],
            "test_bpb": t["model"]["bpb"],
            "end_gap": e[-1]["train_eval_loss"] - e[-1]["val_loss"],
            "clipped_steps": s["clipped_steps"],
            "longest_verbatim": t["memorization"]["max_longest_verbatim_match_chars"],
            "elapsed_s": s["elapsed_s"],
            "test_bytes": t["model"]["bytes"],
        }
        rows.append(rec)
        print(f"{arm:>12} {tr:>12,} {rec['epochs']:>7.2f} {rec['val_bpb']:>9.4f} "
              f"{rec['test_bpb']:>9.4f} {rec['end_gap']:>+8.4f} {rec['clipped_steps']:>6,} "
              f"{rec['longest_verbatim']:>5} {rec['elapsed_s']:>8.0f}")
    out["arms"] = rows

    full, ctrl = rows[0], rows[1]
    d_test = 100 * (full["test_bpb"] / ctrl["test_bpb"] - 1)
    d_val = 100 * (full["val_bpb"] / ctrl["val_bpb"] - 1)
    out["full_vs_ctrl_test_pct"] = d_test
    out["full_vs_ctrl_val_pct"] = d_val
    out["data_multiple"] = full["train_tokens"] / ctrl["train_tokens"]
    verdict = "HELPS" if d_test < 0 else "HURTS"
    print(f"\n  {out['data_multiple']:.2f}x the training text: {d_test:+.2f}% test bpb, "
          f"{d_val:+.2f}% val bpb  => more data {verdict}")
    print(f"  seed bar (measured on the 27-book test split): {SEED_BAR_PCT:.2f}%  "
          f"=> {abs(d_test) / SEED_BAR_PCT:.1f}x the bar")
    out["effect_in_bar_units"] = abs(d_test) / SEED_BAR_PCT

    # ---------------------------------------------------------------- grid
    grid = sorted({r["step"] for r in evals(ARMS[0])} & {r["step"] for r in evals(ARMS[1])})
    f_s = {r["step"]: r["val_bpb"] for r in evals(ARMS[0])}
    c_s = {r["step"]: r["val_bpb"] for r in evals(ARMS[1])}
    crossings, prev = [], None
    for st in grid:
        leader = "full" if f_s[st] < c_s[st] else "ctrl"
        if leader != prev:
            crossings.append({"step": st, "leader": leader})
            prev = leader
    print(f"\n  {len(grid)} common evaluation steps; lead changes: " +
          ", ".join(f"step {c['step']} -> {c['leader']}" for c in crossings))
    out["common_eval_steps"] = len(grid)
    out["lead_changes"] = crossings
    out["tail"] = [{"step": st, "full": f_s[st], "ctrl": c_s[st],
                    "delta_pct": 100 * (f_s[st] / c_s[st] - 1)} for st in grid[-5:]]
    print("  last five evaluations:")
    for t in out["tail"]:
        print(f"    step {t['step']:>5}  full {t['full']:.4f}  ctrl {t['ctrl']:.4f}  {t['delta_pct']:+.2f}%")

    # ------------------------------------------------- decomposition by group
    print("\n" + "=" * 96)
    print("DECOMPOSITION — the same test split, cut into the 27 both arms train on")
    print("                and the 44 only the full arm trains on")
    print("=" * 96)
    tok = load_tokenizer(FULL_DIR)
    p_old, p_new, stats = split_test_by_document_group(tok.eos_id)
    byte_lens = torch.tensor(np.array(tok.token_byte_lengths(), dtype=np.int64))
    print(f"  original 27: {stats['original27_tokens']:,} tokens   "
          f"added 44: {stats['added44_tokens']:,} tokens")

    groups: dict[str, dict] = {}
    for arm in ARMS:
        model, cfg = load_model(arm)
        res = {}
        for label, path in (("original27", p_old), ("added44", p_new)):
            b = TokenBatcher(path, cfg.block_size, 16, seed=0)
            res[label] = evaluate_model(model, b, byte_lens)["bpb"]
        groups[arm] = res
        print(f"  {arm:>12}  original27 {res['original27']:.4f}   added44 {res['added44']:.4f}")

    d_old = 100 * (groups["data71-full"]["original27"] / groups["data71-ctrl"]["original27"] - 1)
    d_new = 100 * (groups["data71-full"]["added44"] / groups["data71-ctrl"]["added44"] - 1)
    out["groups"] = groups
    out["group_stats"] = stats
    out["delta_original27_pct"] = d_old
    out["delta_added44_pct"] = d_new
    print(f"\n  full vs ctrl on the original 27: {d_old:+.2f}%")
    print(f"  full vs ctrl on the added   44: {d_new:+.2f}%")
    print("  the first number is the earlier experiment; the second is what it could not see")

    # ------------------------------------------------- capacity, re-asked
    cap_run = "size-4L256d-gb71"
    if (HERE / f"eval-test-{cap_run}.json").exists():
        print("\n" + "=" * 96)
        print("CAPACITY, RE-ASKED — does the size tie break now the model is not data-limited?")
        print("=" * 96)
        small_s, big_s = summary("data71-full"), summary(cap_run)
        small_t, big_t = test_eval("data71-full"), test_eval(cap_run)
        wall = 100 * (big_s["elapsed_s"] / small_s["elapsed_s"] - 1)
        delta = 100 * (big_t["model"]["bpb"] / small_t["model"]["bpb"] - 1)
        print(f"{'run':>18} {'params':>10} {'steps':>7} {'tokens':>12} {'elapsed':>8} "
              f"{'val':>8} {'test':>8}")
        for run, s, t in (("data71-full", small_s, small_t), (cap_run, big_s, big_t)):
            print(f"{run:>18} {s['params_total']:>10,} {s['steps']:>7,} {s['tokens']:>12,} "
                  f"{s['elapsed_s']:>8.0f} {s['final_val_bpb']:>8.4f} {t['model']['bpb']:>8.4f}")
        print(f"\n  wall clock matched to {wall:+.1f}%")
        print(f"  {big_s['params_total'] / small_s['params_total']:.2f}x the parameters: "
              f"{delta:+.2f}% test bpb")
        print(f"  the same comparison on the 27-book corpus was +0.22% -- inside the seed bar")
        print(f"  => the tie did not break; it widened to {delta / SEED_BAR_PCT:.1f}x the bar")
        out["capacity"] = {
            "small": {"run": "data71-full", "params": small_s["params_total"],
                      "steps": small_s["steps"], "tokens": small_s["tokens"],
                      "elapsed_s": small_s["elapsed_s"], "val_bpb": small_s["final_val_bpb"],
                      "test_bpb": small_t["model"]["bpb"]},
            "big": {"run": cap_run, "params": big_s["params_total"],
                    "steps": big_s["steps"], "tokens": big_s["tokens"],
                    "elapsed_s": big_s["elapsed_s"], "val_bpb": big_s["final_val_bpb"],
                    "test_bpb": big_t["model"]["bpb"], "clipped_steps": big_s["clipped_steps"]},
            "wall_clock_delta_pct": wall,
            "param_multiple": big_s["params_total"] / small_s["params_total"],
            "delta_test_pct": delta,
            "delta_in_bar_units": delta / SEED_BAR_PCT,
            "delta_on_27_book_corpus_pct": 0.21648672334646513,
        }

    (HERE / "clean-data-summary.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwritten to {HERE / 'clean-data-summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Measure parameters and throughput for candidate model sizes.

Why this exists: a compute-matched size sweep is only fair if every config
*finishes* its cosine schedule inside the shared wall-clock budget. The
4,800-step run showed what happens otherwise -- at step 2,400 it read 1.7161
against the shorter run's 1.6815 at the identical step, purely because its
learning rate had not annealed yet. Annealing is worth about 0.035 bpb, which
is the same order as the size effect being measured, so a config that runs out
of time mid-schedule would lose for the wrong reason.

So: measure tokens/s per config, then solve for the step count that lands each
one at the end of its schedule at the same wall clock.

The probe measures pure training throughput. The budget has to cover periodic
validation too, so the caller scales by the ratio between a known run's
end-to-end rate and this probe's rate for the same config.

Usage:
    python src/size_probe.py --budget-s 2926 --reference-tok-s 13437
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from model import GPT, GPTConfig

CANDIDATES = [
    # (label, n_layer, n_embd, n_head)
    ("3L/192d/3H", 3, 192, 3),   # the incumbent, already measured at this budget
    ("4L/256d/4H", 4, 256, 4),
    ("5L/288d/6H", 5, 288, 6),
    ("5L/320d/5H", 5, 320, 5),
    ("6L/384d/6H", 6, 384, 6),
]


def probe(n_layer: int, n_embd: int, n_head: int, vocab_size: int, block_size: int,
          batch_size: int, grad_accum: int, warmup: int, measured: int, threads: int) -> dict:
    torch.set_num_threads(threads)
    torch.manual_seed(1337)
    cfg = GPTConfig(vocab_size=vocab_size, block_size=block_size, n_layer=n_layer,
                    n_head=n_head, n_embd=n_embd, dropout=0.0, tie_embeddings=True)
    model = GPT(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.99), weight_decay=0.1)
    x = torch.randint(0, vocab_size, (batch_size, block_size))
    y = torch.randint(0, vocab_size, (batch_size, block_size))

    tokens_per_step = batch_size * block_size * grad_accum
    elapsed = 0.0
    for step in range(warmup + measured):
        if step == warmup:
            t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        for _ in range(grad_accum):
            _, loss = model(x, y)
            (loss / grad_accum).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    elapsed = time.perf_counter() - t0

    total, non_emb = model.num_params(), model.num_params(non_embedding=True)
    return {
        "params_total": total,
        "params_non_embedding": non_emb,
        "tokens_per_s": measured * tokens_per_step / elapsed,
        "s_per_step": elapsed / measured,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-s", type=float, required=True,
                    help="shared wall-clock budget every config must finish inside")
    ap.add_argument("--reference-tok-s", type=float, required=True,
                    help="end-to-end tokens/s of a completed run of the reference config")
    ap.add_argument("--reference-label", default="3L/192d/3H")
    ap.add_argument("--vocab-size", type=int, default=2048)
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--measured", type=int, default=20)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tokens_per_step = args.batch_size * args.block_size * args.grad_accum
    results = []
    for label, n_layer, n_embd, n_head in CANDIDATES:
        r = probe(n_layer, n_embd, n_head, args.vocab_size, args.block_size,
                  args.batch_size, args.grad_accum, args.warmup, args.measured, args.threads)
        r.update(label=label, n_layer=n_layer, n_embd=n_embd, n_head=n_head)
        results.append(r)
        print(f"  {label:12} {r['params_total']:>10,} params  "
              f"{r['tokens_per_s']:>8,.0f} tok/s (pure train)", flush=True)

    ref = next(r for r in results if r["label"] == args.reference_label)
    # The reference run's end-to-end rate includes validation; the probe does
    # not. One factor, measured on the config we have ground truth for.
    factor = args.reference_tok_s / ref["tokens_per_s"]
    print(f"\n  eval/overhead factor from {args.reference_label}: "
          f"{args.reference_tok_s:,.0f} / {ref['tokens_per_s']:,.0f} = {factor:.3f}")

    print(f"\n  step budgets for a {args.budget_s:,.0f}s matched budget:")
    for r in results:
        predicted = r["tokens_per_s"] * factor
        steps = int(args.budget_s * predicted / tokens_per_step)
        r["predicted_end_to_end_tok_s"] = predicted
        r["max_steps_for_budget"] = steps
        r["tokens_for_budget"] = steps * tokens_per_step
        r["epochs"] = steps * tokens_per_step / 5_771_560
        print(f"  {r['label']:12} {predicted:>8,.0f} tok/s -> {steps:>5,} steps "
              f"({r['tokens_for_budget']:>12,} tokens, {r['epochs']:.2f} epochs)")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"budget_s": args.budget_s, "factor": factor, "results": results}, f, indent=2)
        print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()

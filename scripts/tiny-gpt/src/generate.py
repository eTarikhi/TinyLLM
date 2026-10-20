"""Sample text from a checkpoint.

Sampling settings are part of the result: the same checkpoint at temperature
0.4 and at 1.2 looks like two different models. Every sample this script writes
records the checkpoint, prompt, seed and settings alongside the text.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import load_tokenizer  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402


def load_for_inference(checkpoint: str | Path, data_dir: str | Path | None = None):
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    data_dir = Path(data_dir or ckpt.get("data_dir", "data/prepared-char"))
    tokenizer = load_tokenizer(data_dir)
    cfg = GPTConfig(**ckpt["model_config"])
    if cfg.vocab_size != tokenizer.vocab_size:
        raise SystemExit(
            f"tokenizer/model vocabulary mismatch: model {cfg.vocab_size}, tokenizer {tokenizer.vocab_size}. "
            "This is a release blocker, not a warning."
        )
    model = GPT(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, tokenizer, ckpt


def generate(
    model: GPT,
    tokenizer,
    prompt: str = "\n",
    max_new_tokens: int = 300,
    temperature: float = 0.8,
    top_k: int | None = 40,
    top_p: float | None = None,
    seed: int = 0,
) -> str:
    g = torch.Generator().manual_seed(seed)
    ids = tokenizer.encode(prompt)
    if not ids:
        ids = [tokenizer.eos_id]
    x = torch.tensor([ids], dtype=torch.long)
    out = model.generate(
        x,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_id=tokenizer.eos_id,
        generator=g,
    )
    return tokenizer.decode(out[0].tolist())


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate text from a trained checkpoint.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--prompt", default="\n")
    ap.add_argument("--max-new-tokens", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-samples", type=int, default=1)
    ap.add_argument("--out", default=None, help="write samples to this file as well as stdout")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)

    model, tokenizer, ckpt = load_for_inference(args.checkpoint, args.data_dir)
    records = []
    for i in range(args.num_samples):
        text = generate(
            model, tokenizer, args.prompt, args.max_new_tokens, args.temperature,
            args.top_k, args.top_p, seed=args.seed + i,
        )
        records.append({
            "checkpoint": args.checkpoint,
            "step": ckpt.get("step"),
            "prompt": args.prompt,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "seed": args.seed + i,
            "text": text,
        })
        print(f"--- sample {i + 1}/{args.num_samples} (seed {args.seed + i}, T={args.temperature}, top_k={args.top_k}) ---")
        print(text)
        print()
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".json":
            path.write_text(json.dumps(records, indent=2), encoding="utf-8")
        else:
            path.write_text(
                "\n\n".join(
                    f"--- sample {i + 1} (seed {r['seed']}, T={r['temperature']}, top_k={r['top_k']}, "
                    f"checkpoint step {r['step']}) ---\n{r['text']}"
                    for i, r in enumerate(records)
                ),
                encoding="utf-8",
            )
        print(f"wrote {path}")


if __name__ == "__main__":
    main()

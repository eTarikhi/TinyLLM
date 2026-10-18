"""Pretraining loop: AdamW, warmup + cosine decay, clipping, resume, metrics.

Design rules taken from SKILL.md and references/training-defaults.md:

* the learning-rate decay horizon equals the training horizon;
* the final learning rate is a fixed fraction of the peak (0.1 here, nanoGPT's);
* gradients are clipped at 1.0 and the *pre-clip* norm is logged, so it is
  visible when clipping is actually binding;
* every checkpoint carries model, optimizer, schedule position, step, token
  count, RNG state and data position, so a resumed run is the same run;
* validation is reported as both per-token loss and bits per byte, because
  per-token loss is not comparable across tokenizers.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import resource
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import TokenBatcher, load_manifest, load_tokenizer  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402

LN2 = math.log(2.0)


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------


@dataclass
class TrainConfig:
    run_id: str = "run"
    data_dir: str = "data/prepared-char"
    out_dir: str = "artifacts/runs"
    seed: int = 1337

    # model
    block_size: int = 256
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.1
    bias: bool = False
    tie_embeddings: bool = True

    # optimisation
    batch_size: int = 16
    grad_accum: int = 1
    learning_rate: float = 1e-3
    min_lr_frac: float = 0.1
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.99
    grad_clip: float = 1.0
    warmup_steps: int = 100
    max_steps: int = 2000
    max_seconds: float = 0.0  # 0 disables the wall-clock stop
    stop_after_steps: int = 0  # stop this many steps into *this* process (0 = disabled)

    # evaluation / logging / checkpointing
    eval_every: int = 100
    eval_tokens: int = 131072
    log_every: int = 20
    sample_every: int = 0
    sample_tokens: int = 200
    sample_prompt: str = "\n"
    save_every: int = 0
    keep_best: bool = True
    compile: bool = False
    threads: int = 0  # 0 = leave torch's default

    @staticmethod
    def from_files(config_path: str | None, overrides: dict) -> "TrainConfig":
        values: dict = {}
        if config_path:
            import yaml

            values.update(yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {})
        values.update({k: v for k, v in overrides.items() if v is not None})
        known = {f for f in TrainConfig.__dataclass_fields__}
        unknown = set(values) - known
        if unknown:
            raise SystemExit(f"unknown config keys: {sorted(unknown)}")
        return TrainConfig(**values)

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


# --------------------------------------------------------------------------
# Schedule and optimiser
# --------------------------------------------------------------------------


def lr_at(step: int, cfg: TrainConfig) -> float:
    """Linear warmup, then cosine decay to ``min_lr_frac * peak`` at max_steps."""
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / max(cfg.warmup_steps, 1)
    progress = (step - cfg.warmup_steps) / max(cfg.max_steps - cfg.warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    min_lr = cfg.learning_rate * cfg.min_lr_frac
    return min_lr + coeff * (cfg.learning_rate - min_lr)


def build_optimizer(model: GPT, cfg: TrainConfig) -> torch.optim.AdamW:
    """Weight decay on matrices, none on norms, biases and embeddings-as-vectors.

    Parameter groups matter enough that the experiment plan asks for the
    learning rate *per group*; this project uses one LR but two decay groups.
    """
    decay, no_decay = [], []
    seen: set[int] = set()
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2), eps=1e-8)


# --------------------------------------------------------------------------
# Evaluation helpers
# --------------------------------------------------------------------------


@torch.no_grad()
def evaluate_split(
    model: GPT,
    batcher: TokenBatcher,
    byte_lens: torch.Tensor,
    max_tokens: int | None = None,
) -> dict:
    """Mean per-token loss, perplexity and bits per byte over a split.

    BPB sums loss in nats and divides by ``ln(2) * bytes``; tokens that encode
    no bytes (specials) are excluded from both sums so they cannot distort it.
    """
    was_training = model.training
    model.eval()
    total_nats = 0.0
    total_tokens = 0
    total_bytes = 0
    for x, y in batcher.iter_eval_batches(max_tokens=max_tokens):
        nll = model.token_nll(x, y)
        blen = byte_lens[y]
        mask = blen > 0
        total_nats += float((nll * mask).sum())
        total_bytes += int(blen.sum())
        total_tokens += int(mask.sum())
    model.train(was_training)
    if total_tokens == 0:
        raise RuntimeError("evaluation saw no tokens")
    loss = total_nats / total_tokens
    return {
        "loss": loss,
        "perplexity": math.exp(min(loss, 20.0)),
        "bpb": total_nats / (LN2 * max(total_bytes, 1)),
        "tokens": total_tokens,
        "bytes": total_bytes,
    }


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def environment_record() -> dict:
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
    }


# --------------------------------------------------------------------------
# Checkpointing
# --------------------------------------------------------------------------


def save_checkpoint(path: Path, model: GPT, optimizer, cfg: TrainConfig, step: int, tokens: int, best_val: float, extra: dict) -> None:
    """Atomic: write to a temporary file, then rename over the target."""
    payload = {
        "format": 1,
        "model_state": model.state_dict(),
        "model_config": model.cfg.to_dict(),
        "optimizer_state": optimizer.state_dict(),
        "train_config": cfg.to_dict(),
        "step": step,
        "tokens": tokens,
        "best_val": best_val,
        "data_position": {"next_step": step, "seed": cfg.seed},
        "rng": {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        },
        "env": environment_record(),
        **extra,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------


def train(cfg: TrainConfig, resume_from: str | None = None) -> dict:
    if cfg.threads:
        torch.set_num_threads(cfg.threads)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    data_dir = Path(cfg.data_dir)
    manifest = load_manifest(data_dir)
    tokenizer = load_tokenizer(data_dir)
    byte_lens = torch.tensor(tokenizer.token_byte_lengths(), dtype=torch.long)

    train_batcher = TokenBatcher(data_dir / "train.bin", cfg.block_size, cfg.batch_size, seed=cfg.seed)
    val_batcher = TokenBatcher(data_dir / "val.bin", cfg.block_size, cfg.batch_size, seed=cfg.seed + 1)

    model_cfg = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=cfg.block_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        dropout=cfg.dropout,
        bias=cfg.bias,
        tie_embeddings=cfg.tie_embeddings,
    )
    model = GPT(model_cfg)
    optimizer = build_optimizer(model, cfg)

    start_step = 0
    tokens_seen = 0
    best_val = float("inf")
    if resume_from:
        ckpt = load_checkpoint(resume_from)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_step = ckpt["step"]
        tokens_seen = ckpt["tokens"]
        best_val = ckpt.get("best_val", float("inf"))
        torch.set_rng_state(ckpt["rng"]["torch"])
        np.random.set_state(ckpt["rng"]["numpy"])
        old = ckpt.get("train_config", {})
        # Resuming with a different horizon silently rewrites the learning-rate
        # schedule: the same step number now maps to a different LR, and the
        # resumed run is no longer the run that was interrupted. Say so.
        for key in ("max_steps", "warmup_steps", "learning_rate", "min_lr_frac", "seed", "batch_size", "grad_accum"):
            if key in old and old[key] != getattr(cfg, key):
                print(
                    f"  WARNING: {key} changed on resume ({old[key]} -> {getattr(cfg, key)}); "
                    "this is a different run, not a continuation"
                )
        print(f"resumed from {resume_from} at step {start_step} ({tokens_seen:,} tokens)")

    run_dir = Path(cfg.out_dir) / cfg.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    metrics_file = metrics_path.open("a", encoding="utf-8")

    tokens_per_step = train_batcher.tokens_per_batch * cfg.grad_accum
    flops_per_token = model.flops_per_token()
    header = {
        "event": "run_start",
        "run_id": cfg.run_id,
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": cfg.to_dict(),
        "model_config": model_cfg.to_dict(),
        "params_total": model.num_params(),
        "params_non_embedding": model.num_params(non_embedding=True),
        "param_breakdown": model.param_breakdown(),
        "flops_per_token_estimate": flops_per_token,
        "tokens_per_step": tokens_per_step,
        "planned_tokens": tokens_per_step * cfg.max_steps,
        "data_manifest": {
            "corpus": manifest["corpus"]["name"],
            "tokenizer": manifest["tokenizer"]["kind"],
            "vocab_size": manifest["tokenizer"]["vocab_size"],
            "tokenizer_sha256": manifest["tokenizer"]["sha256"],
            "train_tokens": manifest["splits"]["train"]["tokens"],
        },
        "env": environment_record(),
    }
    metrics_file.write(json.dumps(header) + "\n")
    metrics_file.flush()

    print(
        f"[{cfg.run_id}] params {model.num_params():,} "
        f"({model.num_params(non_embedding=True):,} non-embedding) | "
        f"vocab {tokenizer.vocab_size} | tokens/step {tokens_per_step:,} | "
        f"planned {tokens_per_step * cfg.max_steps:,} tokens"
    )

    model.train()
    t_start = time.time()
    t_window = time.time()
    window_tokens = 0
    stop_reason = "max_steps"
    clipped_steps = 0
    nonfinite_steps = 0
    last_eval: dict = {}

    for step in range(start_step, cfg.max_steps):
        lr = lr_at(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for micro in range(cfg.grad_accum):
            x, y = train_batcher.batch(step * cfg.grad_accum + micro)
            _, loss = model(x, y)
            (loss / cfg.grad_accum).backward()
            # detach before converting: float() on a graph-attached tensor is
            # numerically fine but PyTorch warns, and warnings in a reference
            # implementation train people to ignore warnings.
            loss_sum += float(loss.detach())
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip))
        if not math.isfinite(grad_norm):
            nonfinite_steps += 1
            optimizer.zero_grad(set_to_none=True)
            print(f"  step {step}: non-finite gradient norm, step skipped")
            continue
        if grad_norm > cfg.grad_clip:
            clipped_steps += 1
        optimizer.step()

        tokens_seen += tokens_per_step
        window_tokens += tokens_per_step
        train_loss = loss_sum / cfg.grad_accum

        if (step + 1) % cfg.log_every == 0 or step == start_step:
            dt = max(time.time() - t_window, 1e-9)
            tok_s = window_tokens / dt
            record = {
                "event": "train",
                "step": step + 1,
                "tokens": tokens_seen,
                "loss": train_loss,
                "lr": lr,
                "grad_norm_preclip": grad_norm,
                "tokens_per_s": tok_s,
                "elapsed_s": time.time() - t_start,
                "peak_rss_mb": peak_rss_mb(),
            }
            metrics_file.write(json.dumps(record) + "\n")
            metrics_file.flush()
            print(
                f"  step {step + 1:>5}/{cfg.max_steps} | loss {train_loss:.4f} | lr {lr:.2e} | "
                f"gnorm {grad_norm:5.2f} | {tok_s:7.0f} tok/s | {tokens_seen:,} tokens"
            )
            t_window = time.time()
            window_tokens = 0

        is_last = step + 1 == cfg.max_steps
        if cfg.eval_every and ((step + 1) % cfg.eval_every == 0 or is_last):
            val = evaluate_split(model, val_batcher, byte_lens, max_tokens=cfg.eval_tokens)
            train_eval = evaluate_split(model, train_batcher, byte_lens, max_tokens=cfg.eval_tokens)
            last_eval = val
            record = {
                "event": "eval",
                "step": step + 1,
                "tokens": tokens_seen,
                "val_loss": val["loss"],
                "val_bpb": val["bpb"],
                "val_perplexity": val["perplexity"],
                "train_eval_loss": train_eval["loss"],
                "train_eval_bpb": train_eval["bpb"],
                "elapsed_s": time.time() - t_start,
                "peak_rss_mb": peak_rss_mb(),
            }
            metrics_file.write(json.dumps(record) + "\n")
            metrics_file.flush()
            print(
                f"  eval  {step + 1:>5} | val loss {val['loss']:.4f} | val bpb {val['bpb']:.4f} | "
                f"train(eval) loss {train_eval['loss']:.4f} | gap {train_eval['loss'] - val['loss']:+.4f}"
            )
            if val["loss"] < best_val:
                best_val = val["loss"]
                if cfg.keep_best:
                    save_checkpoint(
                        run_dir / "best.pt", model, optimizer, cfg, step + 1, tokens_seen, best_val,
                        {"val": val, "data_dir": str(data_dir)},
                    )
            t_window = time.time()
            window_tokens = 0

        if cfg.sample_every and (step + 1) % cfg.sample_every == 0:
            sample = sample_text(model, tokenizer, cfg.sample_prompt, cfg.sample_tokens, seed=cfg.seed + step)
            metrics_file.write(json.dumps({"event": "sample", "step": step + 1, "text": sample}) + "\n")
            metrics_file.flush()
            print("  sample: " + sample[:160].replace("\n", "\\n"))

        if cfg.save_every and (step + 1) % cfg.save_every == 0:
            save_checkpoint(run_dir / "last.pt", model, optimizer, cfg, step + 1, tokens_seen, best_val,
                            {"data_dir": str(data_dir)})

        if cfg.max_seconds and (time.time() - t_start) > cfg.max_seconds:
            stop_reason = "max_seconds"
            print(f"  stopping at step {step + 1}: wall-clock budget {cfg.max_seconds}s reached")
            break

        if cfg.stop_after_steps and (step + 1 - start_step) >= cfg.stop_after_steps:
            stop_reason = "stop_after_steps"
            print(f"  stopping at step {step + 1}: stop_after_steps={cfg.stop_after_steps} reached")
            break

    elapsed = time.time() - t_start
    final_step = min(step + 1, cfg.max_steps)
    save_checkpoint(run_dir / "last.pt", model, optimizer, cfg, final_step, tokens_seen, best_val,
                    {"data_dir": str(data_dir)})

    val = evaluate_split(model, val_batcher, byte_lens, max_tokens=cfg.eval_tokens)
    summary = {
        "event": "run_end",
        "run_id": cfg.run_id,
        "stop_reason": stop_reason,
        "steps": final_step,
        "tokens": tokens_seen,
        "elapsed_s": elapsed,
        "mean_tokens_per_s": tokens_seen / max(elapsed, 1e-9),
        "final_val_loss": val["loss"],
        "final_val_bpb": val["bpb"],
        "best_val_loss": best_val,
        "clipped_steps": clipped_steps,
        "nonfinite_steps": nonfinite_steps,
        "peak_rss_mb": peak_rss_mb(),
        "params_total": model.num_params(),
        "achieved_flops_per_s": flops_per_token * tokens_seen / max(elapsed, 1e-9),
        "checkpoints": {"last": str(run_dir / "last.pt"), "best": str(run_dir / "best.pt")},
    }
    metrics_file.write(json.dumps(summary) + "\n")
    metrics_file.close()
    print(
        f"[{cfg.run_id}] done: {final_step} steps, {tokens_seen:,} tokens in {elapsed / 60:.1f} min "
        f"({summary['mean_tokens_per_s']:.0f} tok/s) | val loss {val['loss']:.4f} | val bpb {val['bpb']:.4f} | "
        f"peak RSS {summary['peak_rss_mb']:.0f} MB"
    )
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def sample_text(model: GPT, tokenizer, prompt: str, max_new_tokens: int, seed: int = 0, temperature: float = 0.8, top_k: int = 40) -> str:
    g = torch.Generator().manual_seed(seed)
    ids = tokenizer.encode(prompt) or [tokenizer.eos_id]
    x = torch.tensor([ids], dtype=torch.long)
    out = model.generate(x, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, generator=g)
    return tokenizer.decode(out[0].tolist())


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a tiny GPT from random initialisation.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--resume-from", default=None)
    for field_name, f in TrainConfig.__dataclass_fields__.items():
        kind = f.type
        if kind in ("int", int):
            ap.add_argument(f"--{field_name.replace('_', '-')}", type=int, default=None)
        elif kind in ("float", float):
            ap.add_argument(f"--{field_name.replace('_', '-')}", type=float, default=None)
        elif kind in ("bool", bool):
            ap.add_argument(f"--{field_name.replace('_', '-')}", type=lambda s: s.lower() in {"1", "true", "yes"}, default=None)
        else:
            ap.add_argument(f"--{field_name.replace('_', '-')}", type=str, default=None)
    args = ap.parse_args()
    overrides = {k: v for k, v in vars(args).items() if k not in {"config", "resume_from"}}
    cfg = TrainConfig.from_files(args.config, overrides)
    train(cfg, resume_from=args.resume_from)


if __name__ == "__main__":
    main()

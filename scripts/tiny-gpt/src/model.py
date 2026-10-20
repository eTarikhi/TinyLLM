"""A small decoder-only Transformer, written to be read.

Deliberately conventional, as SKILL.md's Track 1 asks: pre-norm blocks, learned
position embeddings, GELU MLP, optional weight tying, causal self-attention via
``scaled_dot_product_attention``. No mixture of experts, no rotary embeddings,
no fused kernels -- those come after the baseline is correct.

Shapes, once, because they are the thing people get wrong:

    idx      [B, T]            int64 token ids
    tok_emb  [B, T, C]         C = n_embd
    q,k,v    [B, n_head, T, C/n_head]
    att_out  [B, T, C]
    logits   [B, T, V]         V = vocab_size
    loss     []                scalar, mean cross-entropy over non-ignored targets
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

IGNORE_INDEX = -100


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 256
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.0
    bias: bool = False
    tie_embeddings: bool = True
    init_std: float = 0.02

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd={self.n_embd} must be divisible by n_head={self.n_head}")

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    def to_dict(self) -> dict:
        return asdict(self)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.dropout = cfg.dropout
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        # [B, T, C] -> [B, n_head, T, head_dim]
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x), approximate="tanh")))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(cfg.vocab_size, cfg.n_embd),
                wpe=nn.Embedding(cfg.block_size, cfg.n_embd),
                drop=nn.Dropout(cfg.dropout),
                h=nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer)),
                ln_f=nn.LayerNorm(cfg.n_embd, bias=cfg.bias),
            )
        )
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            # One tensor, two uses. Record this: a parameter count is ambiguous
            # without it (references/compute-budget-example.md, step 1).
            self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        # GPT-2 style scaled init on the residual output projections: keeps the
        # variance of the residual stream from growing with depth.
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=cfg.init_std / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)

    # -- bookkeeping --------------------------------------------------------
    def num_params(self, non_embedding: bool = False) -> int:
        """Total parameters. ``non_embedding`` subtracts position embeddings.

        With tied embeddings the token embedding matrix is counted once, which
        is what it is: one tensor.
        """
        seen: dict[int, int] = {}
        for p in self.parameters():
            seen[id(p)] = p.numel()
        total = sum(seen.values())
        if non_embedding:
            total -= self.transformer.wpe.weight.numel()
        return total

    def param_breakdown(self) -> dict[str, int]:
        groups: dict[str, int] = {}
        counted: set[int] = set()
        for name, p in self.named_parameters():
            if id(p) in counted:
                continue
            counted.add(id(p))
            if name.startswith("transformer.wte"):
                key = "token_embedding"
            elif name.startswith("transformer.wpe"):
                key = "position_embedding"
            elif name.startswith("lm_head"):
                key = "lm_head" + (" (tied)" if self.cfg.tie_embeddings else "")
            elif ".attn." in name:
                key = "attention"
            elif ".mlp." in name:
                key = "mlp"
            else:
                key = "norms_and_scalars"
            groups[key] = groups.get(key, 0) + p.numel()
        groups["total"] = sum(v for k, v in groups.items() if k != "total")
        return groups

    def flops_per_token(self, include_attention: bool = True) -> float:
        """Forward+backward FLOPs per token.

        ``6 * N`` for the parameter matmuls, plus ``12 * n_layer * T * d_model``
        for the attention score and value matmuls, which ``6ND`` omits. The
        attention term carries a ``d_model`` factor -- dropping it (a mistake
        this repository shipped once) understates it by three orders of
        magnitude at realistic widths. See references/compute-budget-example.md.
        """
        n = self.num_params()
        flops = 6.0 * n
        if include_attention:
            flops += 12.0 * self.cfg.n_layer * self.cfg.block_size * self.cfg.n_embd
        return flops

    # -- forward ------------------------------------------------------------
    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(f"sequence length {T} exceeds block_size {self.cfg.block_size}")
        pos = torch.arange(T, dtype=torch.long, device=idx.device)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=IGNORE_INDEX,
            )
        return logits, loss

    @torch.no_grad()
    def token_nll(self, idx: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Per-token negative log-likelihood in nats, shape [B, T].

        Used by the bits-per-byte evaluation, which needs the unreduced loss so
        it can weight by the byte length of each target token.
        """
        logits, _ = self(idx)
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=IGNORE_INDEX,
            reduction="none",
        ).view(targets.shape)

    # -- sampling -----------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        eos_id: int | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Append at most ``max_new_tokens`` ids; stop early on ``eos_id``.

        Greedy when ``temperature == 0``.
        """
        was_training = self.training
        self.eval()
        try:
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -self.cfg.block_size :]
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :]
                if temperature == 0:
                    next_id = logits.argmax(dim=-1, keepdim=True)
                else:
                    logits = logits / temperature
                    # top_k <= 0 means "no truncation", the convention every
                    # sampling CLI uses. Passing it through to torch.topk asks
                    # for zero values and raises IndexError, which is how this
                    # was found: `--top-k 0` crashed instead of sampling from
                    # the full distribution.
                    if top_k is not None and top_k > 0:
                        k = min(top_k, logits.size(-1))
                        kth = torch.topk(logits, k, dim=-1).values[:, [-1]]
                        logits = logits.masked_fill(logits < kth, float("-inf"))
                    if top_p is not None and 0 < top_p < 1:
                        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                        probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                        remove = probs - F.softmax(sorted_logits, dim=-1) > top_p
                        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                        logits = torch.full_like(logits, float("-inf")).scatter(
                            1, sorted_idx, sorted_logits
                        )
                    probs = F.softmax(logits, dim=-1)
                    next_id = torch.multinomial(probs, num_samples=1, generator=generator)
                idx = torch.cat([idx, next_id], dim=1)
                if eos_id is not None and bool((next_id == eos_id).all()):
                    break
        finally:
            self.train(was_training)
        return idx


def build_model(cfg: GPTConfig, seed: int | None = None) -> GPT:
    if seed is not None:
        torch.manual_seed(seed)
    return GPT(cfg)

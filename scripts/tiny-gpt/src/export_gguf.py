"""Export a tiny-gpt checkpoint to GGUF, for llama.cpp / LM Studio.

LM Studio (and every other llama.cpp front-end) loads GGUF files whose
architecture matches one of llama.cpp's implementations. Our model is
GPT-2-shaped -- learned position embeddings, pre-norm blocks, fused QKV,
GELU (tanh approximation, which is also what ggml's ``ggml_gelu`` computes),
tied embeddings -- so it maps onto llama.cpp's ``gpt2`` architecture without
any change to the maths. The differences are bookkeeping, and each one is
handled explicitly below:

* llama.cpp's GPT-2 loader **requires** bias tensors. This project trains
  with ``bias=False``, so the exporter writes zeros, which is the same
  function.
* ``output.weight`` is optional in llama.cpp: when absent it reuses the
  token embedding, which is exactly our weight tying. Untied checkpoints
  get their ``lm_head`` written out instead.
* The tokenizer is byte-level BPE with a GPT-2-style regex pretokenizer, so
  it exports as GGUF's ``gpt2`` tokenizer model with the ``gpt-2``
  pretokenizer. The two regexes are not identical (ours keeps letters and
  digits in one class where GPT-2 splits them), so prompt tokenization in
  llama.cpp can differ around letter/digit boundaries. Detokenization is
  byte-exact either way; measure the difference with ``llama-tokenize``
  before trusting prompt-heavy use.
* Token strings are written in the GPT-2 byte-to-unicode alphabet, which is
  what llama.cpp's BPE expects.
* Training never prepends BOS (its embedding is essentially untrained), so
  the file says ``add_bos_token = false``. No chat template is written, as
  with upstream GPT-2 conversions: a pass-through template was tried first
  and llama.cpp's non-jinja format detection rejects it outright
  ("this custom template is not supported"), which is worse than none.
  Treat the model as text completion in the UI.

Weights are written F32: at 1.77M parameters the file is ~7 MB and there is
nothing to gain from quantizing a teaching model.

Usage (from scripts/tiny-gpt/):
    python src/export_gguf.py --checkpoint artifacts/runs/<run>/best.pt \
        --data-dir data/prepared-71-full --out tinylm.gguf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gguf  # noqa: E402

from tokenizer import BaseTokenizer, ByteBPETokenizer  # noqa: E402

# llama_token_type values (llama.h): normal text vs. control tokens.
TOKEN_TYPE_NORMAL = 1
TOKEN_TYPE_CONTROL = 3


def bytes_to_unicode() -> dict[int, str]:
    """The GPT-2 byte alphabet: every byte as a printable unicode character.

    Bytes that are already printable map to themselves; the remaining 68 map
    to code points 256..323 in order. llama.cpp's GPT-2 BPE works in this
    alphabet, so token strings and merges must be written in it.
    """
    direct = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) \
        + list(range(ord("®"), ord("ÿ") + 1))
    mapping = {b: chr(b) for b in direct}
    n = 0
    for b in range(256):
        if b not in mapping:
            mapping[b] = chr(256 + n)
            n += 1
    return mapping


def token_strings(tokenizer: ByteBPETokenizer) -> list[str]:
    """GGUF token list: byte strings in the GPT-2 alphabet, specials by name."""
    enc = bytes_to_unicode()
    names = {tokenizer.bos_id: "<|bos|>", tokenizer.eos_id: "<|eos|>"}
    out = []
    for i in range(tokenizer.vocab_size):
        if i in names:
            out.append(names[i])
        else:
            out.append("".join(enc[b] for b in tokenizer._id_to_bytes[i]))
    return out


def merge_strings(tokenizer: ByteBPETokenizer) -> list[str]:
    """GGUF merges: ``"left right"`` token-string pairs, in rank order."""
    toks = token_strings(tokenizer)
    return [f"{toks[a]} {toks[b]}" for a, b in tokenizer.merges]


def convert(checkpoint: Path, data_dir: Path, out: Path, name: str) -> dict:
    """Write the GGUF file; return a small report of what was written."""
    # Our own checkpoint format (train.py) -- it bundles optimizer and RNG
    # state, so weights_only loading does not apply.
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["model_config"]
    state: dict[str, torch.Tensor] = ckpt["model_state"]

    tokenizer = BaseTokenizer.load(data_dir / "tokenizer.json")
    if not isinstance(tokenizer, ByteBPETokenizer):
        raise SystemExit(f"expected a byte_bpe tokenizer in {data_dir}, got {tokenizer.kind}")
    if tokenizer.vocab_size != cfg["vocab_size"]:
        raise SystemExit(
            f"tokenizer vocab {tokenizer.vocab_size} != model vocab {cfg['vocab_size']}")

    n_layer, n_embd = cfg["n_layer"], cfg["n_embd"]
    writer = gguf.GGUFWriter(str(out), "gpt2")

    writer.add_name(name)
    writer.add_description(
        "1.77M-parameter GPT trained from scratch on 71 public-domain books "
        "(literary English, 1320-1922). A teaching/research model: text "
        "completion in period style, not a chat assistant.")
    writer.add_block_count(n_layer)
    writer.add_context_length(cfg["block_size"])
    writer.add_embedding_length(n_embd)
    writer.add_feed_forward_length(4 * n_embd)
    writer.add_head_count(cfg["n_head"])
    writer.add_layer_norm_eps(1e-5)  # nn.LayerNorm default, as trained
    writer.add_file_type(gguf.LlamaFileType.ALL_F32)

    toks = token_strings(tokenizer)
    types = [TOKEN_TYPE_NORMAL] * tokenizer.vocab_size
    types[tokenizer.bos_id] = TOKEN_TYPE_CONTROL
    types[tokenizer.eos_id] = TOKEN_TYPE_CONTROL
    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("gpt-2")
    writer.add_token_list(toks)
    writer.add_token_types(types)
    writer.add_token_merges(merge_strings(tokenizer))
    writer.add_bos_token_id(tokenizer.bos_id)
    writer.add_eos_token_id(tokenizer.eos_id)
    writer.add_add_bos_token(False)
    writer.add_add_eos_token(False)

    written: list[str] = []

    def add(tensor: gguf.MODEL_TENSOR, bid: int | None, suffix: str,
            arr: np.ndarray) -> None:
        tname = gguf.TENSOR_NAMES[tensor]
        if bid is not None:
            tname = tname.format(bid=bid)
        tname += suffix
        writer.add_tensor(tname, np.ascontiguousarray(arr, dtype=np.float32))
        written.append(tname)

    def w(key: str) -> np.ndarray:
        return state[key].detach().cpu().to(torch.float32).numpy()

    def bias(key: str, n: int) -> np.ndarray:
        # llama.cpp requires GPT-2 biases; an absent one is a zero one.
        return w(key) if key in state else np.zeros(n, dtype=np.float32)

    add(gguf.MODEL_TENSOR.TOKEN_EMBD, None, ".weight", w("transformer.wte.weight"))
    add(gguf.MODEL_TENSOR.POS_EMBD, None, ".weight", w("transformer.wpe.weight"))
    add(gguf.MODEL_TENSOR.OUTPUT_NORM, None, ".weight", w("transformer.ln_f.weight"))
    add(gguf.MODEL_TENSOR.OUTPUT_NORM, None, ".bias", bias("transformer.ln_f.bias", n_embd))
    if not cfg["tie_embeddings"]:
        add(gguf.MODEL_TENSOR.OUTPUT, None, ".weight", w("lm_head.weight"))

    for i in range(n_layer):
        p = f"transformer.h.{i}"
        add(gguf.MODEL_TENSOR.ATTN_NORM, i, ".weight", w(f"{p}.ln_1.weight"))
        add(gguf.MODEL_TENSOR.ATTN_NORM, i, ".bias", bias(f"{p}.ln_1.bias", n_embd))
        add(gguf.MODEL_TENSOR.ATTN_QKV, i, ".weight", w(f"{p}.attn.c_attn.weight"))
        add(gguf.MODEL_TENSOR.ATTN_QKV, i, ".bias", bias(f"{p}.attn.c_attn.bias", 3 * n_embd))
        add(gguf.MODEL_TENSOR.ATTN_OUT, i, ".weight", w(f"{p}.attn.c_proj.weight"))
        add(gguf.MODEL_TENSOR.ATTN_OUT, i, ".bias", bias(f"{p}.attn.c_proj.bias", n_embd))
        add(gguf.MODEL_TENSOR.FFN_NORM, i, ".weight", w(f"{p}.ln_2.weight"))
        add(gguf.MODEL_TENSOR.FFN_NORM, i, ".bias", bias(f"{p}.ln_2.bias", n_embd))
        add(gguf.MODEL_TENSOR.FFN_UP, i, ".weight", w(f"{p}.mlp.c_fc.weight"))
        add(gguf.MODEL_TENSOR.FFN_UP, i, ".bias", bias(f"{p}.mlp.c_fc.bias", 4 * n_embd))
        add(gguf.MODEL_TENSOR.FFN_DOWN, i, ".weight", w(f"{p}.mlp.c_proj.weight"))
        add(gguf.MODEL_TENSOR.FFN_DOWN, i, ".bias", bias(f"{p}.mlp.c_proj.bias", n_embd))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return {"tensors": written, "vocab": tokenizer.vocab_size,
            "merges": len(tokenizer.merges), "bytes": out.stat().st_size}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-dir", required=True, help="prepared dir holding tokenizer.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="TinyLM 1.8M (Gutenberg-71)")
    args = ap.parse_args()

    out = Path(args.out)
    report = convert(Path(args.checkpoint), Path(args.data_dir), out, args.name)
    print(f"wrote {out}: {report['bytes'] / 1e6:.1f} MB, {len(report['tensors'])} tensors, "
          f"vocab {report['vocab']}, {report['merges']} merges")
    return 0


if __name__ == "__main__":
    sys.exit(main())

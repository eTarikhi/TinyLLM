"""Pin the GGUF exporter.

A GGUF file that loads but computes something else is worse than one that
fails, so the checks here are about identity, not plausibility: every weight
must come back byte-for-byte, the tokenizer must rebuild the same byte
strings from its exported alphabet, and the tensor set must be exactly what
llama.cpp's GPT-2 loader requires -- including the zero biases it insists on
and the absent output projection it replaces with the tied embedding.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import gguf  # noqa: E402

import export_gguf  # noqa: E402
from model import GPTConfig, build_model  # noqa: E402
from tokenizer import ByteBPETokenizer  # noqa: E402

TEXT = ("It is a truth universally acknowledged, that a single man in "
        "possession of a good fortune, must be in want of a wife. " * 40)


def field_str(reader, key):
    f = reader.fields[key]
    return bytes(f.parts[f.data[0]]).decode("utf-8")


def field_int(reader, key):
    f = reader.fields[key]
    return int(np.asarray(f.parts[f.data[0]]).reshape(-1)[0])


class ExportGgufTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls.tmp.name)
        cls.tokenizer = ByteBPETokenizer.train(TEXT, vocab_size=300)
        cls.data_dir = tmp / "data"
        cls.tokenizer.save(cls.data_dir / "tokenizer.json")
        cfg = GPTConfig(vocab_size=cls.tokenizer.vocab_size, block_size=16,
                        n_layer=2, n_head=2, n_embd=32, bias=False,
                        tie_embeddings=True)
        cls.model = build_model(cfg, seed=0)
        cls.ckpt = tmp / "best.pt"
        torch.save({"format": 1, "model_state": cls.model.state_dict(),
                    "model_config": cfg.to_dict()}, cls.ckpt)
        cls.out = tmp / "tinylm.gguf"
        export_gguf.convert(cls.ckpt, cls.data_dir, cls.out, "test")
        cls.reader = gguf.GGUFReader(str(cls.out))
        cls.tensors = {t.name: t for t in cls.reader.tensors}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_metadata(self):
        r = self.reader
        self.assertEqual(field_str(r, "general.architecture"), "gpt2")
        self.assertEqual(field_int(r, "gpt2.context_length"), 16)
        self.assertEqual(field_int(r, "gpt2.block_count"), 2)
        self.assertEqual(field_str(r, "tokenizer.ggml.model"), "gpt2")
        self.assertEqual(field_str(r, "tokenizer.ggml.pre"), "gpt-2")
        self.assertEqual(field_int(r, "tokenizer.ggml.bos_token_id"),
                         self.tokenizer.bos_id)
        self.assertEqual(field_int(r, "tokenizer.ggml.eos_token_id"),
                         self.tokenizer.eos_id)

    def test_tensor_set_is_exactly_what_gpt2_requires(self):
        expected = {"token_embd.weight", "position_embd.weight",
                    "output_norm.weight", "output_norm.bias"}
        for i in range(2):
            for base in ("attn_norm", "attn_qkv", "attn_output",
                         "ffn_norm", "ffn_up", "ffn_down"):
                expected.add(f"blk.{i}.{base}.weight")
                expected.add(f"blk.{i}.{base}.bias")
        self.assertEqual(set(self.tensors), expected)
        self.assertNotIn("output.weight", self.tensors)  # tied: must be absent

    def test_weights_round_trip_exactly(self):
        state = self.model.state_dict()
        pairs = {
            "token_embd.weight": "transformer.wte.weight",
            "position_embd.weight": "transformer.wpe.weight",
            "output_norm.weight": "transformer.ln_f.weight",
            "blk.0.attn_qkv.weight": "transformer.h.0.attn.c_attn.weight",
            "blk.1.attn_output.weight": "transformer.h.1.attn.c_proj.weight",
            "blk.0.ffn_up.weight": "transformer.h.0.mlp.c_fc.weight",
            "blk.1.ffn_down.weight": "transformer.h.1.mlp.c_proj.weight",
            "blk.0.attn_norm.weight": "transformer.h.0.ln_1.weight",
            "blk.1.ffn_norm.weight": "transformer.h.1.ln_2.weight",
        }
        for gguf_name, torch_name in pairs.items():
            got = np.asarray(self.tensors[gguf_name].data).astype(np.float32).ravel()
            want = state[torch_name].numpy().ravel()
            np.testing.assert_array_equal(got, want, err_msg=gguf_name)

    def test_absent_biases_are_written_as_zeros(self):
        for name, t in self.tensors.items():
            if name.endswith(".bias"):
                self.assertTrue(np.all(np.asarray(t.data) == 0), name)

    def test_tokenizer_rebuilds_the_same_bytes(self):
        f = self.reader.fields["tokenizer.ggml.tokens"]
        toks = [bytes(f.parts[i]).decode("utf-8") for i in f.data]
        self.assertEqual(len(toks), self.tokenizer.vocab_size)
        inv = {ch: b for b, ch in export_gguf.bytes_to_unicode().items()}
        for tid, s in enumerate(toks):
            if tid in (self.tokenizer.bos_id, self.tokenizer.eos_id):
                continue
            self.assertEqual(bytes(inv[c] for c in s),
                             self.tokenizer._id_to_bytes[tid], f"token {tid}")

        fm = self.reader.fields["tokenizer.ggml.merges"]
        merges = [bytes(fm.parts[i]).decode("utf-8") for i in fm.data]
        self.assertEqual(len(merges), len(self.tokenizer.merges))
        id_of = {s: i for i, s in enumerate(toks)}
        for (a, b), m in zip(self.tokenizer.merges, merges):
            left, right = m.split(" ")
            self.assertEqual((id_of[left], id_of[right]), (a, b))


if __name__ == "__main__":
    unittest.main()

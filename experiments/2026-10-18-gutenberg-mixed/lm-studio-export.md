# Using TinyLM in LM Studio

The released model (1.77M parameters, test **1.5662 bits/byte**, trained on
71 public-domain books) now exports to GGUF as
**`tinylm-1.8m-gutenberg71.gguf`** — 7.2 MB, F32, 40 tensors — and runs in
LM Studio or any llama.cpp front-end.

## Recommended: the 3.7M generalist

**`tinylm-3.7m-broad68.gguf`** (15.1 MB, F32, 52 tensors) is the better
model to actually use. It was trained on the broadened corpus — the same 71
books plus 80 works of drama, verse and non-fiction — at 3.74M parameters
on the long clock. On the original 71 books it scores 1.5587 bits/byte
(slightly better than the 1.8M model), and on 40 books it never trained on
it is **8.13% better** than a same-size model trained only on the 71 —
which is what matters when you type prompts it has never seen, and why it
handles verse- or dialogue-shaped prompts the smaller model fumbles. Same
import steps and settings as below; greedy output verified identical to
the PyTorch checkpoint through llama.cpp. Rebuild with, after retraining
(`configs/broad-4L256d-full-long.yaml`):

```bash
python src/export_gguf.py --checkpoint artifacts/runs/broad-4L256d-full-long/best.pt \
    --data-dir data/prepared-broad-full --out tinylm-3.7m-broad68.gguf
```

## Import

Either drag the `.gguf` file into the LM Studio window, run

```bash
lms import tinylm-1.8m-gutenberg71.gguf
```

or place it by hand at
`~/.lmstudio/models/tinylm/tinylm-1.8m-gutenberg71/tinylm-1.8m-gutenberg71.gguf`
(LM Studio only detects the two-level `publisher/model/` layout). It appears
under **My Models**.

## Settings

| setting | value | why |
| --- | --- | --- |
| Context length | 256 | the model's trained window; longer contexts extrapolate badly |
| Temperature | 0.7–0.9, top-k 40 | greedy (0) loops, as small base models do |
| System prompt | **empty** | it is a base model; a system prompt is just more prose to continue |
| Chat template | none / raw completion | the file deliberately carries no template (see below) |

Type the opening of a paragraph — `"Elizabeth turned to her sister and said,"` —
and let it continue. It has no concept of questions, instructions or turns.

## What to expect

Locally fluent 19th-century prose that drifts within a paragraph and
occasionally invents a word. Sample, temperature 0.8, generated through
llama.cpp from the exported file:

> It was a bright cold day in April, in the street, the crowd of a young
> man on the top of the hillside, and in the dark shells of a lamp, and the
> little rumor of the fence, which was drawn to the street door and to the
> walls of the hill.
>
> "Is that my room, Michael?" said Michael.
>
> "It is a very garden," replied Michael.

It is a teaching and research artifact — the smallest GPT-2 is 70× its
size — not an assistant. Against its own baselines it is well trained:
1.5662 bpb vs 3.79 untrained and 2.16 for a bigram model.

## How the file was made, and how it was verified

```bash
cd scripts/tiny-gpt
python src/data.py --corpus gutenberg-mixed-71-full --tokenizer byte_bpe \
    --vocab-size 2048 --out-dir data/prepared-71-full --cache-dir data/raw/71full
python src/train.py --config configs/data71-full-9600.yaml   # 109 min on 2 vCPU
python src/export_gguf.py \
    --checkpoint artifacts/runs/data71-full-9600/best.pt \
    --data-dir data/prepared-71-full --out ../../tinylm-1.8m-gutenberg71.gguf
```

The retrain reproduced the released model exactly before export was even
attempted: same corpus hash, same tokenizer hash, val 1.5417, test 1.5662,
same per-book extremes. The export itself is verified three ways:

1. **Weights round-trip byte-for-byte** through the GGUF file
   (`tests/test_export_gguf.py`), including the zero biases llama.cpp's
   GPT-2 loader requires and the absent `output.weight` it replaces with
   the tied embedding.
2. **Greedy generation is token-for-token identical** between PyTorch
   (`generate.py`, temperature 0) and llama.cpp (`llama-completion`), on
   two prompts, 64 tokens each — 128 consecutive argmax decisions agreeing
   across two independent implementations.
3. **Tokenization agrees**: `llama-tokenize` and our tokenizer produced
   identical id sequences on every probe tried, including digit and
   underscore cases. The one structural difference — GPT-2's pretokenizer
   splits letters from digits where ours keeps `\w+` — is provably inert
   for this vocabulary: **0 of the 1,790 merges** cross a boundary GPT-2
   would split on, so every merge remains reachable.

Speed on the 2-vCPU sandbox (llama.cpp, F32): 6,838 tok/s prompt
processing, 2,116 tok/s generation. Any machine that runs LM Studio will
run this at interactive speed many times over.

## Caveats worth knowing

* **No chat template in the file.** A pass-through template was tried
  first; llama.cpp's non-jinja format detection aborts on it
  ("this custom template is not supported"), so none is written — the
  same choice upstream GPT-2 conversions make. If your UI wraps input in
  chat markup, the model will continue the markup as if it were prose.
* **256 tokens of context** (~190 words). Prompts longer than that lose
  their beginning.
* **Period text, period prejudices.** The corpus is literary English of
  1320–1922; several source works contain slurs and period bias that the
  model can reproduce (see `model-card-71.md`).
* The `.gguf` is a build artifact, like a checkpoint: it is not committed.
  The exporter (`src/export_gguf.py`) is, and the commands above rebuild
  the file exactly.

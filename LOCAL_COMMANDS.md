# Local commands for TinyLLM

This is the practical command list for running this repository on a local
machine: clone the branch, create the Python environment, validate the
checkout, prepare data, train, monitor, evaluate, export a GGUF for
LM Studio, and push changes back to GitHub.

The commands below are written for Bash or Zsh on macOS, Linux, or Windows
Subsystem for Linux (WSL2). Use WSL2 rather than native Windows Python for
training: the queue scripts are Bash scripts, and `src/train.py` uses the
Unix-only `resource` module.

## Contents

- [1. Get this branch](#1-get-this-branch)
- [2. Check the local prerequisites](#2-check-the-local-prerequisites)
- [3. Create the Python environment](#3-create-the-python-environment)
- [4. Run the fast local checks](#4-run-the-fast-local-checks)
- [5. Run the smoke test](#5-run-the-smoke-test)
- [6. Prepare datasets](#6-prepare-datasets)
- [7. Train models](#7-train-models)
- [8. Monitor a run](#8-monitor-a-run)
- [9. Evaluate, generate, and plot](#9-evaluate-generate-and-plot)
- [10. Export a GGUF and import it into LM Studio](#10-export-a-gguf-and-import-it-into-lm-studio)
- [11. Commit, push, and check GitHub CI](#11-commit-push-and-check-github-ci)
- [12. Helpful discovery commands](#12-helpful-discovery-commands)
- [13. Troubleshooting](#13-troubleshooting)

## Before you start

- Run the setup sections in order the first time. Afterwards, you can jump
  directly to the section you need.
- Unless a section says otherwise, commands start from the repository root:
  the folder containing `README.md`, `SKILL.md`, and `.git`.
- Keep using the same terminal while following a sequence, so the virtual
  environment stays activated and the current directory stays where the
  previous command left it. If a repeated `cd` line says “no such file or
  directory” because you are already in that folder, skip that line and
  continue with the rest of the block.
- Do not run two training queues at the same time. They compete for the same
  CPU and can write to the same data/artifact locations.
- Prepared data, checkpoints, run artifacts, and `.gguf` files are ignored by
  Git. A fresh clone therefore contains the code and configs, but not the
  generated datasets or model weights.

## 1. Get this branch

### Fresh clone

```bash
git clone --branch arena/dfc533fd-tinyllm https://github.com/eTarikhi/TinyLLM.git
cd TinyLLM
git status --short --branch
git log --oneline -5
```

### Existing clone

```bash
cd /path/to/TinyLLM
git status --short
git fetch origin
git switch arena/dfc533fd-tinyllm
git pull --ff-only
git status --short --branch
```

If the existing checkout has unfinished changes, commit or stash them before
switching branches.

For normal training and development, the checkout folder may be named
`TinyLLM`. The stricter folder name `llm-from-scratch` matters when packaging
or validating the folder as an installable skill; section 4 shows how to stage
that form without renaming your working clone.

## 2. Check the local prerequisites

Use Python 3.11 or 3.12. Python 3.11 is the known-good interpreter for the
pinned local requirements, and the GitHub reference-implementation job uses
Python 3.12.

```bash
git --version
python3.11 --version || python3.12 --version || python3 --version
df -h .
```

Check how many CPU cores are available:

```bash
# Linux / WSL2
nproc
```

```bash
# macOS
sysctl -n hw.ncpu
```

Set the interpreter name once for the setup commands. Change `python3.11` to
`python3.12` if that is the version installed on your machine.

```bash
PYTHON_BIN=python3.11
"$PYTHON_BIN" --version
```

## 3. Create the Python environment

Create `.venv` at the **repository root**, not inside `scripts/tiny-gpt`.
Most queue scripts call `../../.venv/bin/python` after changing into
`scripts/tiny-gpt`, so that root location is part of the project layout.

### Linux or WSL2: CPU-only PyTorch

On Linux, install the CPU wheel first so pip does not download the much larger
CUDA-bundled wheel when it reads the pinned requirements file.

```bash
"$PYTHON_BIN" -m venv .venv
source .venv/bin/activate
python --version
python -m pip install --upgrade pip
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r scripts/tiny-gpt/requirements.txt
```

### macOS

```bash
"$PYTHON_BIN" -m venv .venv
source .venv/bin/activate
python --version
python -m pip install --upgrade pip
python -m pip install -r scripts/tiny-gpt/requirements.txt
```

### Optional: install the skill validator used by CI

This is only needed for the full local skill-validation commands in section 4.
It is not needed for training.

```bash
python -m pip install skills-ref==0.1.1
```

On Windows, pip may warn that `agentskills.exe` was installed into a
`...\Python313\Scripts` folder that is not on `PATH`. That warning is harmless
for this project: the published `skills-ref` wheel registers its console
script as `agentskills`, not `skills-ref`, so always invoke the validator as a
module with the same Python interpreter:

```bash
python -m skills_ref.cli --help
python -m skills_ref.cli validate .
```

Do not add that Scripts folder to `PATH` just for this repository, and do not
call `skills-ref` directly. If the warning path shows a user-site
`Python313` folder, you installed the optional validator outside the project
`.venv`; that is fine for validation only. For training, still use WSL2 with
the root-level `.venv` and Python 3.11 or 3.12 from the setup above.

### Verify the environment

```bash
python -c "import sys, torch, numpy, yaml, gguf; print(sys.version); print('torch', torch.__version__); print('torch threads', torch.get_num_threads()); print('numpy', numpy.__version__)"
python -m pip check
```

When you open a new terminal later, return to the repository root and
reactivate the environment:

```bash
cd /path/to/TinyLLM
source .venv/bin/activate
```

## 4. Run the fast local checks

Run these before the first long training job and before pushing changes.

From the repository root:

```bash
python .github/scripts/check_package.py .
python .github/scripts/check_docs_math.py .
cd scripts/tiny-gpt
python -m unittest discover -s tests -t tests -v
cd ../..
```

`check_package.py` may warn that the checkout directory is named `TinyLLM`
rather than `llm-from-scratch`. That warning is expected for a normal Git
checkout. It becomes a hard requirement only for the staged, installable
skill folder.

### Optional: run the official frontmatter validator

This mirrors the pinned validator download used by GitHub CI. Run it from the
repository root:

```bash
mkdir -p .ci
curl -fsSL -o .ci/quick_validate.py "https://raw.githubusercontent.com/anthropics/skills/dbd4588f9e1033efb41dad4bef2f7947c8993d44/skills/skill-creator/scripts/quick_validate.py"
python .ci/quick_validate.py .
```

### Optional: validate a staged skill folder with the correct name

Commit your changes first if you want this staged copy to include them,
because `git archive HEAD` reads the committed tree.

```bash
rm -rf /tmp/llm-from-scratch-stage
mkdir -p /tmp/llm-from-scratch-stage
git archive --format=tar --prefix=llm-from-scratch/ HEAD SKILL.md README.md LICENSE references templates scripts | tar -x -C /tmp/llm-from-scratch-stage
python -m skills_ref.cli validate /tmp/llm-from-scratch-stage/llm-from-scratch
```

## 5. Run the smoke test

This is the quick end-to-end check: prepare the tiny char-level dataset, train
for 20 steps, evaluate the checkpoint, generate text, and plot the validation
curve. The first line moves from the repository root into the implementation
folder; copy the fenced block as plain text, not as rendered Markdown links.

```bash
cd scripts/tiny-gpt
python src/data.py --tokenizer char --out-dir data/prepared-char
python src/train.py --config configs/tiny.yaml --run-id ci-smoke --max-steps 20 --eval-every 20 --log-every 10
python src/evaluate.py --checkpoint artifacts/runs/ci-smoke/best.pt --split test
python src/generate.py --checkpoint artifacts/runs/ci-smoke/best.pt --max-new-tokens 100 --temperature 0.8 --top-k 40
python src/plot_metrics.py artifacts/runs/ci-smoke/metrics.jsonl --metric val_bpb --out artifacts/runs/ci-smoke/val-bpb.svg
ls -lh artifacts/runs/ci-smoke
```

If this sequence works, the local environment, data path, training loop,
checkpointing, evaluator, generator, and metrics plotter are all wired up.

## 6. Prepare datasets

Run all commands in this section from `scripts/tiny-gpt`. The smoke test
leaves you in that directory; if you are at the repository root instead, run
`cd scripts/tiny-gpt` first.

Data preparation downloads the public-domain source texts on the first run,
trains or reuses a tokenizer, and writes each prepared dataset with a
`manifest.json` describing its provenance. Generated data stays under
`scripts/tiny-gpt/data/` and is not committed.

### Tiny Shakespeare datasets

These are the prerequisites for `run_pilots.sh` and the small pilot configs.

```bash
python src/data.py --tokenizer char --out-dir data/prepared-char
python src/data.py --tokenizer byte_bpe --vocab-size 512 --out-dir data/prepared-bpe512
```

### Gutenberg 27-book BPE dataset

```bash
python src/data.py --corpus gutenberg-mixed-27 --tokenizer byte_bpe \
    --vocab-size 2048 --out-dir data/prepared-gb-bpe2048 \
    --tokenizer-sample-chars 4000000
```

### Gutenberg 71-book full dataset

This is the dataset used by the released 1.8M-parameter model.

```bash
python src/data.py --corpus gutenberg-mixed-71-full --tokenizer byte_bpe \
    --vocab-size 2048 --out-dir data/prepared-71-full \
    --cache-dir data/raw/71full
```

### Broad corpus at vocab 2048

Prepare the broad training corpus first. The composition-control datasets
below reuse its tokenizer so their validation/test token streams stay directly
comparable.

```bash
python src/data.py --corpus gutenberg-mixed-broad-full --tokenizer byte_bpe \
    --vocab-size 2048 --out-dir data/prepared-broad-full \
    --cache-dir data/raw/broad
```

Optional composition-control and held-out datasets at the same tokenizer:

```bash
python src/data.py --corpus gutenberg-mixed-broad-ctrl --tokenizer byte_bpe \
    --vocab-size 2048 --out-dir data/prepared-broad-ctrl \
    --cache-dir data/raw/broad --tokenizer-from data/prepared-broad-full

python src/data.py --corpus gutenberg-mixed-narrative --tokenizer byte_bpe \
    --vocab-size 2048 --out-dir data/prepared-broad-narrative \
    --cache-dir data/raw/broad --tokenizer-from data/prepared-broad-full

python src/data.py --corpus gutenberg-mixed-broad50 --tokenizer byte_bpe \
    --vocab-size 2048 --out-dir data/prepared-broad50 \
    --cache-dir data/raw/broad --tokenizer-from data/prepared-broad-full

python src/data.py --corpus gutenberg-heldout-forms --tokenizer byte_bpe \
    --vocab-size 2048 --out-dir data/prepared-heldout \
    --cache-dir data/raw/broad --tokenizer-from data/prepared-broad-full
```

Do not omit `--tokenizer-from` for those comparison datasets. Training a fresh
tokenizer for each arm would change the measurement as well as the training
data.

### Broad corpus at vocab 4096, 8192, and 16384

Each vocabulary size deliberately trains its own tokenizer. These are the
datasets used by the vocab-scaling probes.

```bash
for V in 4096 8192 16384; do
  python -u src/data.py --corpus gutenberg-mixed-broad-full --tokenizer byte_bpe \
      --vocab-size "$V" --out-dir "data/prepared-broad-full-v$V" \
      --cache-dir data/raw/broad
done
```

### Held-out dataset for the vocab-4096 release

```bash
python src/data.py --corpus gutenberg-heldout-forms --tokenizer byte_bpe \
    --vocab-size 4096 --out-dir data/prepared-heldout-v4096 \
    --cache-dir data/raw/broad \
    --tokenizer-from data/prepared-broad-full-v4096
```

### Check what was prepared

```bash
find data -maxdepth 2 -name manifest.json | sort
du -sh data
```

## 7. Train models

Run training commands from `scripts/tiny-gpt` with the root virtual
environment activated. Training is CPU-based in this implementation, and the
larger runs take hours. Exact wall-clock time depends on your CPU and thread
count.

### Pilot runs

Prerequisites: `data/prepared-char` and `data/prepared-bpe512` from section 6.

```bash
bash run_pilots.sh
```

Or run the main tiny-Shakespeare config directly:

```bash
python -u src/train.py --config configs/main.yaml
```

### Gutenberg baseline

Prerequisite: `data/prepared-gb-bpe2048`.

```bash
python -u src/train.py --config configs/main-gutenberg.yaml
```

### Released 1.8M Gutenberg-71 model

Prerequisite: `data/prepared-71-full`.

```bash
python -u src/train.py --config configs/data71-full-9600.yaml
```

### Broad-corpus generalist at vocab 2048

Prerequisite: `data/prepared-broad-full`.

```bash
python -u src/train.py --config configs/broad-4L256d-full-long.yaml
```

### Recommended release: broad corpus at vocab 4096

Prerequisite: `data/prepared-broad-full-v4096`.

```bash
python -u src/train.py --config configs/broad-v4096-release.yaml
```

This is currently the recommended model to reproduce. Its committed result is
test bits-per-byte 1.4939 on the seen corpus family and 1.7907 on the held-out
forms corpus.

### Current unfinished probes: vocab 16384 and 8-layer shape

The queue script prepares the vocab-4096/8192/16384 datasets, then trains and
test-evaluates `big-vocab16384` and `big-shape-8L192d`:

```bash
bash run_queue8.sh
```

To run only one of those probes after preparing its dataset:

```bash
python -u src/train.py --config configs/big-vocab16384.yaml
python -u src/evaluate.py --checkpoint artifacts/runs/big-vocab16384/best.pt \
    --split test --per-document --threads 2
```

```bash
python -u src/train.py --config configs/big-shape-8L192d.yaml
python -u src/evaluate.py --checkpoint artifacts/runs/big-shape-8L192d/best.pt \
    --split test --per-document --threads 2
```

### Resume an interrupted run

Use the same config and the run's `last.pt`, not `best.pt`. `last.pt` is the
latest checkpoint; `best.pt` is the best-validation checkpoint and may be
earlier.

```bash
python -u src/train.py --config configs/broad-v4096-release.yaml \
    --resume-from artifacts/runs/broad-v4096-release/last.pt
```

For an exact continuation, do not change schedule-defining settings such as
`max_steps`, `warmup_steps`, `learning_rate`, `min_lr_frac`, `seed`,
`batch_size`, or `grad_accum`. The trainer warns if those changed, because the
resumed job is then a different run rather than a continuation.

### Use more CPU threads

The established large configs set `threads: 2`. On a larger local machine you
can override that value, for example:

```bash
python -u src/train.py --config configs/broad-v4096-release.yaml --threads 4
```

Keep `threads: 2` when trying to reproduce a committed run as closely as
possible. More threads are not automatically faster once memory bandwidth or
thermal limits become the bottleneck, so compare the reported tokens/second.

## 8. Monitor a run

Open a second terminal, activate the same environment, and go to
`scripts/tiny-gpt`:

```bash
cd /path/to/TinyLLM
source .venv/bin/activate
cd scripts/tiny-gpt
RUN_ID=broad-v4096-release
```

Follow the live metrics:

```bash
tail -f "artifacts/runs/$RUN_ID/metrics.jsonl"
```

Show only the latest evaluation records:

```bash
grep '"event": "eval"' "artifacts/runs/$RUN_ID/metrics.jsonl" | tail -5
```

Check the run files and disk use:

```bash
ls -lh "artifacts/runs/$RUN_ID"
du -sh data artifacts
```

After the run finishes, read its summary:

```bash
python -m json.tool "artifacts/runs/$RUN_ID/summary.json"
```

Check whether a training process is still running:

```bash
ps aux | grep '[s]rc/train.py'
```

Important files in each run directory:

- `metrics.jsonl` — append-only training/evaluation event log.
- `summary.json` — final run summary, written when training exits normally.
- `best.pt` — checkpoint with the best validation loss.
- `last.pt` — latest checkpoint, used for exact resume.
- `eval-test.json` and `eval-test.md` — written by `src/evaluate.py`.

## 9. Evaluate, generate, and plot

From `scripts/tiny-gpt`, set the run you want to inspect:

```bash
RUN_ID=broad-v4096-release
CKPT="artifacts/runs/$RUN_ID/best.pt"
```

### Test-set evaluation

```bash
python src/evaluate.py --checkpoint "$CKPT" \
    --split test --per-document --threads 2
cat "artifacts/runs/$RUN_ID/eval-test.md"
```

Use `--split val` instead of `--split test` for validation-split evaluation.
Reserve test evaluation for finished checkpoints and pre-declared comparison
points; do not tune against the test split.

### Held-out evaluation for the vocab-4096 release

Prerequisite: `data/prepared-heldout-v4096` from section 6.

```bash
python src/evaluate.py \
    --checkpoint artifacts/runs/broad-v4096-release/best.pt \
    --data-dir data/prepared-heldout-v4096 \
    --split test --per-document --threads 2 \
    --out-dir artifacts/runs/broad-v4096-release/heldout
cat artifacts/runs/broad-v4096-release/heldout/eval-test.md
```

The separate `--out-dir` prevents the held-out result from overwriting the
normal test-set result in the run directory.

### Generate text

```bash
python src/generate.py --checkpoint "$CKPT" \
    --prompt "The old lighthouse stood" \
    --max-new-tokens 300 --temperature 0.8 --top-k 40 \
    --num-samples 3 --seed 0 \
    --out "artifacts/runs/$RUN_ID/generated.txt"
```

These are small base language models, not instruction-tuned chat models. Judge
them by held-out bits-per-byte and continuation quality rather than expecting
reliable question answering.

### Plot a training curve

```bash
python src/plot_metrics.py "artifacts/runs/$RUN_ID/metrics.jsonl" \
    --metric val_bpb \
    --out "artifacts/runs/$RUN_ID/val-bpb.svg"
```

## 10. Export a GGUF and import it into LM Studio

LM Studio loads GGUF files. It does not load this project's PyTorch `.pt`
checkpoints directly.

### Export the recommended release

Prerequisites: the trained `broad-v4096-release` checkpoint and
`data/prepared-broad-full-v4096`.

Run from `scripts/tiny-gpt`:

```bash
python src/export_gguf.py \
    --checkpoint artifacts/runs/broad-v4096-release/best.pt \
    --data-dir data/prepared-broad-full-v4096 \
    --out ../../tinylm-4.3m-broad-v4096.gguf
ls -lh ../../tinylm-4.3m-broad-v4096.gguf
cd ../..
```

### Import with the LM Studio CLI

If the `lms` command is enabled:

```bash
lms import "$PWD/tinylm-4.3m-broad-v4096.gguf"
lms ls
```

### Or copy it into LM Studio's model folder manually

On macOS or Linux, the file must be inside a publisher/model subfolder. A
`.gguf` placed directly in the `models` root is ignored.

```bash
mkdir -p "$HOME/.lmstudio/models/tinylm/tinylm-4.3m-broad-v4096"
cp tinylm-4.3m-broad-v4096.gguf \
   "$HOME/.lmstudio/models/tinylm/tinylm-4.3m-broad-v4096/"
```

If you trained inside WSL2 but run LM Studio on Windows, use LM Studio's
import dialog or copy the GGUF to the Windows-side model folder under
`%USERPROFILE%\.lmstudio\models\<publisher>\<model>\`.

For the complete export notes and the other released checkpoints, see
[`experiments/2026-10-18-gutenberg-mixed/lm-studio-export.md`](experiments/2026-10-18-gutenberg-mixed/lm-studio-export.md).

## 11. Commit, push, and check GitHub CI

Run from the repository root.

### Check before committing

```bash
git status --short
git diff --check
python .github/scripts/check_package.py .
python .github/scripts/check_docs_math.py .
cd scripts/tiny-gpt
python -m unittest discover -s tests -t tests -v
cd ../..
git diff --stat
```

### Commit and push this branch

Stage only the files you intended to change. Generated data, checkpoints, run
artifacts, and GGUF files are ignored and should not be force-added.

```bash
git add path/to/changed-file
git status --short
git commit -m "Describe the change"
git push origin arena/dfc533fd-tinyllm
```

### Manually run the GitHub workflow

The workflow also runs automatically for pushes to `main` and for pull
requests. To run it manually on this branch with GitHub CLI:

```bash
gh auth status
gh workflow run validate-skill.yml --ref arena/dfc533fd-tinyllm
gh run list --workflow validate-skill.yml --branch arena/dfc533fd-tinyllm --limit 5
```

Then watch the newest run by replacing `RUN_ID` with its numeric ID:

```bash
gh run watch RUN_ID --exit-status
```

### Optional: open a pull request

```bash
gh pr create --base main --head arena/dfc533fd-tinyllm \
    --title "Describe the change" \
    --body "Summary of the change and how it was tested."
gh pr checks --watch
```

## 12. Helpful discovery commands

Go to `scripts/tiny-gpt`, then list the available configs and queue scripts:

```bash
cd scripts/tiny-gpt
find configs -maxdepth 1 -type f -name '*.yaml' | sort
grep -H '^data_dir:' configs/*.yaml
ls -1 run_*.sh
```

Show each command-line tool's options:

```bash
python src/data.py --help
python src/train.py --help
python src/evaluate.py --help
python src/generate.py --help
python src/export_gguf.py --help
python src/plot_metrics.py --help
python src/size_probe.py --help
```

Inspect a queue script before running it:

```bash
sed -n '1,120p' run_queue8.sh
```

The older `run_queue*.sh` files are reproducibility records for specific
experiment batches. Several assume particular prepared datasets already exist,
so inspect the script and prepare its inputs before rerunning it.

## 13. Troubleshooting

### `can't open file '...\src\[data.py](http://data.py)'`

This usually means two copy/paste problems happened at once: rendered Markdown
link text such as `src/[data.py](http://data.py)` was pasted into the terminal,
and the command was run from the repository root. The implementation does not
live in a root-level `src/` folder; it lives in `scripts/tiny-gpt/src/`.

Copy commands from the fenced code blocks as plain text. In WSL2, from the
repository root, the prompt and command should look like this:

```bash
cd scripts/tiny-gpt
python src/data.py --tokenizer char --out-dir data/prepared-char
```

If you are only checking the path in Windows PowerShell, the prompt must end
in `...\TinyLLM\scripts\tiny-gpt>` and the equivalent path is
`python .\src\data.py --tokenizer char --out-dir data\prepared-char`. Use
WSL2 for the full smoke test and training commands.

### `../../.venv/bin/python: No such file or directory`

The virtual environment was not created at the repository root, or the queue
script is being run from the wrong checkout. Return to the folder containing
`README.md` and create it there:

```bash
cd /path/to/TinyLLM
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r scripts/tiny-gpt/requirements.txt
```

On Linux, use the CPU-wheel install order from section 3.

### `No module named torch`, `numpy`, `yaml`, or `gguf`

The virtual environment is probably not activated in this terminal:

```bash
cd /path/to/TinyLLM
source .venv/bin/activate
which python
python -m pip install -r scripts/tiny-gpt/requirements.txt
```

### A config says its `data_dir` does not exist

Prepared datasets are generated locally and are not in Git. Find the required
directory and run the matching preparation command from section 6:

```bash
cd scripts/tiny-gpt
grep '^data_dir:' configs/broad-v4096-release.yaml
ls data
```

### Tokenizer or vocabulary-size mismatch during evaluation/export

Evaluate or export with the same prepared dataset/tokenizer family used for
training. For cross-corpus comparisons, prepare the comparison corpus with
`--tokenizer-from` pointing at the training dataset, as shown for the held-out
commands in section 6.

### Training was interrupted

Resume from the latest checkpoint with the same config:

```bash
cd scripts/tiny-gpt
ls -lh artifacts/runs/broad-v4096-release/last.pt
python -u src/train.py --config configs/broad-v4096-release.yaml \
    --resume-from artifacts/runs/broad-v4096-release/last.pt
```

`last.pt` is refreshed at the config's `save_every` interval and at normal
process exit. An abrupt stop can therefore lose the work since the most recent
checkpoint save, but it should not lose the whole run.

### Training looks stuck

Check both the process and the newest metrics before deciding it is stuck.
Data preparation and evaluation can also take a while without printing a new
training line.

```bash
ps aux | grep '[s]rc/train.py'
tail -5 artifacts/runs/broad-v4096-release/metrics.jsonl
ls -lh artifacts/runs/broad-v4096-release
```

### LM Studio does not show the GGUF

Use `lms import`, or place the file under
`~/.lmstudio/models/<publisher>/<model>/<file>.gguf`. A `.gguf` sitting
directly in `~/.lmstudio/models/` is ignored. Also confirm that you exported a
`.gguf`; LM Studio cannot load the PyTorch `best.pt` checkpoint.

### The skill validator complains about the folder name

That is expected when validating a normal `TinyLLM` checkout directly. Use the
staged `llm-from-scratch` copy from section 4 for strict package-identity
validation.

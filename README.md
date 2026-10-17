# llm-from-scratch skill

Reusable skill for an AI coding agent helping you create and train a language model from random initialization.

## Contents

- `SKILL.md` — core instructions, workflow, engineering gates, sources and deliverables.
- `references/reading-list.md` — curated research and implementation links with notes, plus a dated verification log and package-version table.
- `references/compute-budget-example.md` — worked config → parameter count → `6ND` → pilot → wall-clock → cost example, with every figure reproducible.
- `references/training-defaults.md` — concrete hyperparameter values read from nanoGPT and nanochat source, and which of them transfer across optimizers.
- `templates/experiment-plan.md` — copyable template for tracking training runs.
- `scripts/tiny-gpt/README.md` — the bundled reference implementation: a tested, CPU-runnable pretraining project (tokenizers, data pipeline, model, training loop with exact resume, baseline-relative evaluation, seven correctness tests).
- `experiments/README.md` — evidence trail from running that implementation end to end: filled-in experiment plan, pilot matrix, evaluation report, samples, model card. Not part of the packaged skill.

## Use

This repository is one Agent Skill: a folder whose `SKILL.md` entry point carries YAML frontmatter plus instructions, with `references/`, `templates/` and `scripts/` bundled as progressively disclosed resources.

**The folder must be named `llm-from-scratch` when you install or package it.** This repository is named `TinyLLM`, but the packaging convention derives the skill's identity from the *directory* name, not from the frontmatter — so packaging the checkout as-is produces `TinyLLM.skill` containing a `TinyLLM/` folder whose `SKILL.md` declares `name: llm-from-scratch`. Nothing warns you: the official validator passes either way, because it checks frontmatter only. Rename the folder on the way in:

```bash
git clone https://github.com/eTarikhi/TinyLLM.git llm-from-scratch
```

Where the folder goes depends on the agent product (personal vs. project scope, Claude Code vs. Claude.ai vs. the Skills API), and those locations change — confirm the current path in the official docs rather than copying one from a blog post:

- [How to create custom skills](https://support.claude.com/en/articles/12512198-creating-custom-skills)
- [Using skills in Claude](https://support.claude.com/en/articles/12512180-using-skills-in-claude)
- [Skills API quickstart](https://docs.claude.com/en/api/skills-guide#creating-a-skill)
- Agent Skills specification: <https://agentskills.io/specification>

To produce a distributable `.skill` file (a zip), use the packager from Anthropic's [`skill-creator`](https://github.com/anthropics/skills/tree/main/skills/skill-creator). It validates before packaging and excludes `evals/`, `__pycache__/`, and `node_modules/`. It must be run from the `skill-creator` directory, because it imports its validator as `scripts.quick_validate`:

```bash
git clone --depth 1 https://github.com/anthropics/skills.git
cd skills/skills/skill-creator
python -m scripts.package_skill /path/to/llm-from-scratch ./dist   # -> dist/llm-from-scratch.skill
```

Running that module from any other directory fails with `ModuleNotFoundError: No module named 'scripts'`.

**Never package a working checkout in place.** The packager excludes `evals/`, `__pycache__/`, `node_modules/`, `*.pyc` and `.DS_Store` — but *not* dotfile directories, and not the directories a training run creates. Measured here on 2026-10-17, with `.venv/` deleted first: packaging the checkout as-is produced an **107 MB** `TinyLLM.skill` of 83 entries and 119.7 MB uncompressed, almost all of it the prepared token files and checkpoints under `scripts/tiny-gpt/data/` and `scripts/tiny-gpt/artifacts/`, plus 32 `.git` entries (61.2 KB). The `.git` part is also a disclosure risk: `.git/config` can carry remote URLs with embedded credentials, and packed history can carry content that was later deleted.

Stage from git instead of copying the directory. `git archive` emits exactly the committed files you name, so no untracked artifact, no `.git`, and no half-finished edit can leak into the archive:

```bash
mkdir -p /tmp/stage && git -C /path/to/checkout archive --format=tar \
  --prefix=llm-from-scratch/ HEAD SKILL.md README.md LICENSE references templates scripts \
  | tar -x -C /tmp/stage
python -m scripts.package_skill /tmp/stage/llm-from-scratch ./dist
python -c "import zipfile;print('\n'.join(zipfile.ZipFile('./dist/llm-from-scratch.skill').namelist()))"
```

That last line lists the archive contents — check it before distributing. A correct package holds 30 entries and weighs about 72 KB: `SKILL.md`, `README.md`, `LICENSE`, the three `references/` files, the one `templates/` file, and the 24 files of `scripts/tiny-gpt/` (its `README.md`, `requirements.txt`, `run_pilots.sh`, three `configs/`, seven `src/` modules and nine `tests/`). It does **not** hold `experiments/`, `data/`, `artifacts/`, `.git/` or `.github/`. `python .github/scripts/check_package.py .` warns when any of those are present in the directory you are about to package.

## Validation

`SKILL.md` passes the official `quick_validate.py` from `anthropics/skills` (checked 2026-10-17). The rules it enforces: frontmatter parses as a YAML mapping; only `name`, `description`, `license`, `allowed-tools`, `metadata`, and `compatibility` are permitted; `name` is kebab-case and at most 64 characters; `description` is at most 1024 characters and contains no angle brackets.

Unlike the packager, the validator has no package-relative imports, so it can be invoked directly by path from anywhere. It does need PyYAML:

```bash
pip install pyyaml
python /path/to/skill-creator/scripts/quick_validate.py /path/to/llm-from-scratch   # -> "Skill is valid!"
```

The `SKILL.md` body stays comfortably under the recommended 500-line progressive-disclosure ceiling, so it does not need another layer of hierarchy, and no bundled reference file is long enough to require a table of contents (the guideline is one above 300 lines). Re-check after any edit rather than trusting a number recorded here:

```bash
awk 'BEGIN{n=0} /^---$/{n++; next} n>=2' SKILL.md | wc -l   # SKILL.md body lines, ceiling 500
wc -l references/*.md templates/*.md                        # ceiling 300 each before a TOC is needed
```

Before substantial training, read current upstream docs, check licenses for the selected data, and run the single-device pilot and tests.

## Continuous checks

`.github/workflows/validate-skill.yml` runs on every push to `main` and on every pull request, so drift cannot ship silently. It fetches Anthropic's `quick_validate.py` at a **pinned revision** rather than tracking `main`, so an upstream change cannot alter what this repository enforces without a deliberate commit. Two jobs:

- **validate** — official frontmatter validation, then the two repository checkers below, then a packaging-identity step that actually performs the staged rename from the section above and fails if the result would disagree with its own frontmatter or would leak `.git/`, `.github/` or `experiments/`.
- **reference-implementation** — installs CPU-only PyTorch, runs the bundled test suite, then smoke-tests the data pipeline and 20 training steps. Code shipped inside a skill is advice an agent will follow literally, so it is tested like code, not like documentation.

`.github/scripts/check_package.py` covers what the official validator cannot see, since that one validates frontmatter only. It asserts that every file README advertises exists, that every relative link in the READMEs, `SKILL.md` and the references resolves, that `SKILL.md` points at each bundled resource, that the documented line budgets hold, that the experiment-plan template still carries the fields `SKILL.md` demands be recorded, and that every file of the bundled implementation — including each test file named in `SKILL.md`'s project layout — exists with an exactly pinned torch version.

`.github/scripts/check_docs_math.py` re-derives **49 published numbers** in `references/` from the configs they claim to describe: parameter counts tied and untied, the component breakdown, the `6ND` and `12·L·T·d·D` terms and every percentage built on them, wall-clock, cost and tokens-per-update rows, and the nanoGPT and nanochat anchor figures. A number that no longer matches *or whose sentence was reworded past the checker's pattern* is a failure, which is what caught an attention-FLOP term that had been wrong by a factor of `d_model`. Both are stdlib only:

```bash
python3 .github/scripts/check_package.py .     # exits 1 on any failure
python3 .github/scripts/check_docs_math.py .   # exits 1 on any mismatch
```

Warnings about `.git/`, `.github/`, `.venv/`, `experiments/`, `data/` and `artifacts/` being present are expected in a working checkout; they are the packaging reminder above, not failures.

## License

MIT — see [LICENSE](LICENSE). This matches the reference implementations the skill points to (`nanochat`, `nanoGPT`, `llm.c`, and `lm-evaluation-harness` are all MIT; `pythia` is Apache-2.0), so copying this skill folder into another project imposes no copyleft obligation.


# llm-from-scratch skill

Reusable skill for an AI coding agent helping you create and train a language model from random initialization.

## Contents

- `SKILL.md` — core instructions, workflow, engineering gates, sources and deliverables.
- `references/reading-list.md` — curated research and implementation links with notes, plus a dated verification log and package-version table.
- `references/compute-budget-example.md` — worked config → parameter count → `6ND` → pilot → wall-clock → cost example, with every figure reproducible.
- `references/training-defaults.md` — concrete hyperparameter values read from nanoGPT and nanochat source, and which of them transfer across optimizers.
- `templates/experiment-plan.md` — copyable template for tracking training runs.

## Use

This repository is one Agent Skill: a folder whose `SKILL.md` entry point carries YAML frontmatter plus instructions, with `references/` and `templates/` bundled as progressively disclosed resources.

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

**Strip `.git` and `.github` before packaging.** The packager excludes `evals/`, `__pycache__/`, `node_modules/`, `*.pyc`, and `.DS_Store` — but *not* dotfile directories. Measured on this repository: packaging a working checkout as-is pulled in 31 `.git` entries totalling 49.7 KB, which is **35% of the archive's uncompressed content** and grew the `.skill` file from 38 KB to 77 KB. Beyond bloat, this is a disclosure risk: `.git/config` can contain remote URLs with embedded credentials, and packed history can contain content that was later deleted. Stage a clean copy first:

```bash
rsync -a --exclude='.git' --exclude='.github' /path/to/llm-from-scratch/ /tmp/stage/llm-from-scratch/
python -m scripts.package_skill /tmp/stage/llm-from-scratch ./dist
python -c "import zipfile;print('\n'.join(zipfile.ZipFile('./dist/llm-from-scratch.skill').namelist()))"
```

That last line lists the archive contents — check it before distributing. A correct package holds exactly `SKILL.md`, `README.md`, `LICENSE`, the three `references/` files, and the one `templates/` file.

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

`.github/workflows/validate-skill.yml` runs both checks above on every push to `main` and on every pull request, so package drift cannot ship silently. It fetches Anthropic's `quick_validate.py` at a **pinned revision** rather than tracking `main`, so an upstream change cannot alter what this repository enforces without a deliberate commit.

`.github/scripts/check_package.py` is the repository-level half, and it covers what the official validator cannot see — it validates frontmatter only. It asserts that every file README advertises exists, that every relative link resolves, that `SKILL.md` points at each bundled resource, that the documented line budgets hold, and that the experiment-plan template still carries the fields `SKILL.md` demands be recorded. It also warns when the directory name would make a packaged `.skill` disagree with its own frontmatter. Stdlib only:

```bash
python3 .github/scripts/check_package.py .     # exits 1 on any failure
```

Warnings about `.git/` and `.github/` being present are expected in a checkout; they are the packaging reminder above, not failures.

## License

MIT — see [LICENSE](LICENSE). This matches the reference implementations the skill points to (`nanochat`, `nanoGPT`, `llm.c`, and `lm-evaluation-harness` are all MIT; `pythia` is Apache-2.0), so copying this skill folder into another project imposes no copyleft obligation.


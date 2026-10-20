#!/usr/bin/env python3
"""Repository consistency checks for the llm-from-scratch skill package.

Anthropic's quick_validate.py checks SKILL.md frontmatter only. It cannot see
whether the files README advertises exist, whether relative links resolve, or
whether SKILL.md actually points at the resources bundled beside it. Every one
of those has been broken in this repository's history, silently.

This script checks them, plus one more class since the skill began shipping a
runnable reference implementation: the test files SKILL.md names in its project
layout must actually exist in that implementation, so the layout cannot drift
away from the code an agent is told to copy.

Arithmetic inside references/ is checked separately by check_docs_math.py.

Stdlib only, no dependencies. Exits 1 on any failure.

Usage:  python3 .github/scripts/check_package.py [repo_root]
"""

import os
import re
import sys

# Budgets from the Agent Skills progressive-disclosure guidance
# (https://agentskills.io/specification#progressive-disclosure, read 2026-10-18):
# the SKILL.md body should stay under 500 lines *and* under ~5,000 tokens, and
# bundled files over 300 lines need a table of contents.
#
# The token ceiling is the one that actually bites, and it was being ignored
# here: at 305 lines this body was well inside the line limit while carrying
# roughly 7,800 tokens, over the budget by half again.
#
# Nothing in this repository can tokenize the way a client will, so the budget
# is enforced against the conservative of two estimators -- characters/4, which
# is the usual rule of thumb for English prose and slightly pessimistic for
# Markdown full of short lines and backticks. Being pessimistic is the point: a
# body that passes this check passes under any reasonable tokenizer.
BODY_LIMIT = 500
BODY_TOKEN_BUDGET = 5000
REFERENCE_LIMIT = 300

# Fields the template must keep. Each was missing at some point while SKILL.md
# explicitly demanded it be recorded, which made the template silently lossy.
REQUIRED_TEMPLATE_TERMS = [
    "tied or untied",
    "bits per byte",
    "per parameter group",
    "precision",
    "gradient norm",
    "skipped",
    "source urls and access dates",
    "intended use and known restrictions",
    "preprocessing code reference",
]

# The runnable reference implementation bundled with the skill.
IMPL_DIR = os.path.join("scripts", "tiny-gpt")
IMPL_REQUIRED = [
    "README.md",
    "requirements.txt",
    os.path.join("src", "tokenizer.py"),
    os.path.join("src", "data.py"),
    os.path.join("src", "model.py"),
    os.path.join("src", "train.py"),
    os.path.join("src", "generate.py"),
    os.path.join("src", "evaluate.py"),
    os.path.join("configs", "tiny.yaml"),
]

# Directories that must never end up inside a distributed .skill.
NEVER_PACKAGE = [".git", ".github", "experiments", ".venv",
                 os.path.join(IMPL_DIR, "data"), os.path.join(IMPL_DIR, "artifacts")]

failures = []
warnings = []


def check(ok, message):
    if ok:
        print(message)
    else:
        failures.append(message)
    return bool(ok)


def warn(message):
    warnings.append(message)
    print("WARN: " + message)


def read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def main(root):
    root = os.path.abspath(root)
    os.chdir(root)

    for required in ("SKILL.md", "README.md", "LICENSE"):
        if not check(os.path.isfile(required), f"{required} exists"):
            return finish()

    skill_md = read("SKILL.md")
    readme = read("README.md")

    # --- 1. frontmatter name -------------------------------------------------
    m = re.match(r"^---\n(.*?)\n---\n", skill_md, re.S)
    if not check(m is not None, "SKILL.md opens with YAML frontmatter"):
        return finish()
    name_match = re.search(r"^name:\s*(.+)$", m.group(1), re.M)
    if not check(name_match is not None, "SKILL.md frontmatter declares a name"):
        return finish()
    name = name_match.group(1).strip()
    check(name == "llm-from-scratch", f"frontmatter name is 'llm-from-scratch' (got {name!r})")

    # The specification is explicit that `name` "must match the parent directory
    # name", and the reference library enforces it: skills-ref (PyPI `skills-ref`,
    # Apache-2.0, published by the spec's author) fails validation with
    # "Directory name 'X' must match skill name 'Y'". Anthropic's quick_validate.py
    # does *not* check this, which is why the mismatch used to ship unnoticed.
    #
    # A git checkout is always named after the repository, so this stays a warning
    # here and a hard failure in CI, which validates a staged copy renamed to the
    # frontmatter name -- the form a user actually installs.
    dirname = os.path.basename(root)
    if dirname != name:
        warn(
            f"directory is '{dirname}' but the skill name is '{name}'. The Agent "
            f"Skills spec requires them to match, and `skills-ref validate` fails "
            f"on this. Packaging here produces {dirname}.skill wrapping a folder "
            f"that declares name: {name}. Stage a copy renamed to '{name}' first."
        )

    # --- 2. bundled resources ------------------------------------------------
    bundled = []
    for directory in ("references", "templates"):
        if os.path.isdir(directory):
            bundled += [
                os.path.join(directory, f)
                for f in sorted(os.listdir(directory))
                if f.endswith(".md")
            ]
    check(bool(bundled), "at least one bundled resource file exists")
    for path in bundled:
        check(path in skill_md, f"SKILL.md points to {path}")
        check(path in readme, f"README lists {path}")

    # --- 3. README Contents matches disk ------------------------------------
    # Scan only the Contents section: the rest of the README legitimately uses
    # backticked bullets for commands and filenames that are not repo paths.
    contents = re.search(r"^## Contents\n(.*?)(?=\n## )", readme, re.S | re.M)
    check(contents is not None, "README has a Contents section")
    listed = re.findall(r"^- `([^`]+)`", contents.group(1), re.M) if contents else []
    on_disk = ["SKILL.md"] + bundled + [os.path.join(IMPL_DIR, "README.md")]
    for entry in listed:
        check(os.path.exists(entry.rstrip("/")), f"README-listed {entry} exists")
    for path in on_disk:
        check(any(entry.rstrip("/") == path for entry in listed), f"{path} is listed in README Contents")

    # --- 4. relative links resolve ------------------------------------------
    scan = ["README.md", "SKILL.md"] + bundled
    impl_readme = os.path.join(IMPL_DIR, "README.md")
    if os.path.isfile(impl_readme):
        scan.append(impl_readme)
    for path in scan:
        base = os.path.dirname(path)
        for _label, target in re.findall(r"\[([^\]]*)\]\(([^)]+)\)", read(path)):
            if target.startswith(("http://", "https://", "#", "mailto:")):
                continue
            resolved = os.path.normpath(os.path.join(base, target.split("#")[0]))
            check(os.path.exists(resolved), f"{path} -> {target} resolves")

    # --- 5. line budgets ----------------------------------------------------
    body = skill_md.split("---\n", 2)[2]
    n_body = body.count("\n")
    check(n_body < BODY_LIMIT, f"SKILL.md body is {n_body} lines (limit {BODY_LIMIT})")
    tokens_chars = len(body) / 4
    tokens_words = len(body.split()) * 1.33
    check(
        tokens_chars < BODY_TOKEN_BUDGET,
        f"SKILL.md body is ~{tokens_chars:,.0f} tokens by chars/4 and "
        f"~{tokens_words:,.0f} by words*1.33 (budget {BODY_TOKEN_BUDGET:,}); "
        f"move detail into references/ rather than trimming meaning out of it",
    )
    for path in bundled:
        text = read(path)
        n = text.count("\n")
        if n > REFERENCE_LIMIT:
            head = text[:2000].lower()
            has_toc = "## contents" in head or "## table of contents" in head
            check(has_toc, f"{path} is {n} lines, over {REFERENCE_LIMIT}, so it needs a table of contents")
        else:
            check(True, f"{path} is {n} lines (limit {REFERENCE_LIMIT})")

    # --- 6. template still records what SKILL.md demands ---------------------
    template = os.path.join("templates", "experiment-plan.md")
    if os.path.isfile(template):
        text = read(template).lower()
        for term in REQUIRED_TEMPLATE_TERMS:
            check(term in text, f"experiment-plan template records '{term}'")

    # --- 7. the bundled implementation matches what SKILL.md promises --------
    if check(os.path.isdir(IMPL_DIR), f"{IMPL_DIR}/ exists"):
        check(IMPL_DIR in skill_md, f"SKILL.md points to {IMPL_DIR}")
        for rel in IMPL_REQUIRED:
            check(os.path.isfile(os.path.join(IMPL_DIR, rel)), f"{IMPL_DIR}/{rel} exists")
        # Every test file named in the skill's project layout must exist here, or
        # the layout is advice the shipped code does not follow. The layout lives
        # in references/project-scaffold.md, so scan the bundled resources too --
        # scanning SKILL.md alone silently checked nothing once the tree moved.
        layout_text = skill_md + "".join(read(b) for b in bundled)
        named_tests = sorted(set(re.findall(r"(test_[a-z_]+\.py)", layout_text)))
        check(len(named_tests) >= 7, f"the project layout names {len(named_tests)} test files")
        for test_name in named_tests:
            check(
                os.path.isfile(os.path.join(IMPL_DIR, "tests", test_name)),
                f"{IMPL_DIR}/tests/{test_name} exists (named in SKILL.md)",
            )
        reqs = read(os.path.join(IMPL_DIR, "requirements.txt"))
        check(bool(re.search(r"^torch==", reqs, re.M)), "requirements.txt pins an exact torch version")

    # --- 8. packaging hygiene ------------------------------------------------
    for leak in NEVER_PACKAGE:
        if os.path.exists(leak):
            warn(
                f"{leak}/ is present. package_skill.py does not exclude it, so packaging "
                f"this checkout in place would bundle it into the distributed .skill. "
                f"Strip it from the staged copy."
            )

    return finish()


def finish():
    print()
    for w in warnings:
        print("note: " + w.split(". ")[0] + ".")
    if failures:
        print(f"\n{len(failures)} check(s) failed:")
        for f in failures:
            print("  FAIL: " + f)
        return 1
    print("All package consistency checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))

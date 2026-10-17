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

# Line budgets from the Agent Skills progressive-disclosure guidance:
# SKILL.md body under 500 lines; bundled files over 300 lines need a TOC.
BODY_LIMIT = 500
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

    # The packager derives the .skill filename and internal folder from the
    # directory name, not the frontmatter, and the validator does not compare
    # them. Surface the mismatch rather than letting it ship.
    dirname = os.path.basename(root)
    if dirname != name:
        warn(
            f"directory is '{dirname}' but the skill name is '{name}'. "
            f"Packaging here produces {dirname}.skill wrapping a folder that "
            f"declares name: {name}. Stage a copy renamed to '{name}' first."
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
        # Every test file named in SKILL.md's project layout must exist here,
        # or the layout is advice the shipped code does not follow.
        for test_name in sorted(set(re.findall(r"(test_[a-z_]+\.py)", skill_md))):
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

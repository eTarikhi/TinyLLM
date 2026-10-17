#!/usr/bin/env python3
"""Repository consistency checks for the llm-from-scratch skill package.

Anthropic's quick_validate.py checks SKILL.md frontmatter only. It cannot see
whether the files README advertises exist, whether relative links resolve, or
whether SKILL.md actually points at the resources bundled beside it. Every one
of those has been broken in this repository's history, silently.

This script checks them. Stdlib only, no dependencies. Exits 1 on any failure.

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

failures = []
warnings = []


def check(ok, message):
    (print if ok else failures.append)(message if ok else "FAIL: " + message)
    return ok


def warn(message):
    warnings.append(message)
    print("WARN: " + message)


def main(root):
    root = os.path.abspath(root)
    os.chdir(root)

    for required in ("SKILL.md", "README.md", "LICENSE"):
        check(os.path.isfile(required), f"{required} exists")

    skill_md = open("SKILL.md", encoding="utf-8").read()
    readme = open("README.md", encoding="utf-8").read()

    # --- 1. frontmatter name -------------------------------------------------
    m = re.match(r"^---\n(.*?)\n---\n", skill_md, re.S)
    check(m is not None, "SKILL.md opens with YAML frontmatter")
    name = re.search(r"^name:\s*(.+)$", m.group(1), re.M).group(1).strip() if m else ""
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
    listed = re.findall(r"^- `([^`]+)`", readme, re.M)
    on_disk = ["SKILL.md"] + bundled
    for entry in listed:
        check(os.path.isfile(entry), f"README-listed {entry} exists")
    for path in on_disk:
        check(path in listed, f"{path} is listed in README Contents")

    # --- 4. relative links resolve ------------------------------------------
    scan = ["README.md", "SKILL.md"] + bundled
    for path in scan:
        base = os.path.dirname(path)
        for label, target in re.findall(r"\[([^\]]*)\]\(([^)]+)\)", open(path, encoding="utf-8").read()):
            if target.startswith(("http://", "https://", "#", "mailto:")):
                continue
            resolved = os.path.normpath(os.path.join(base, target.split("#")[0]))
            check(os.path.exists(resolved), f"{path} -> {target} resolves")

    # --- 5. line budgets ----------------------------------------------------
    body = skill_md.split("---\n", 2)[2]
    n_body = body.count("\n")
    check(n_body < BODY_LIMIT, f"SKILL.md body is {n_body} lines (limit {BODY_LIMIT})")
    for path in bundled:
        n = open(path, encoding="utf-8").read().count("\n")
        if n > REFERENCE_LIMIT:
            head = open(path, encoding="utf-8").read()[:2000].lower()
            has_toc = "## contents" in head or "## table of contents" in head
            check(has_toc, f"{path} is {n} lines, over {REFERENCE_LIMIT}, so it needs a table of contents")
        else:
            check(True, f"{path} is {n} lines (limit {REFERENCE_LIMIT})")

    # --- 6. template still records what SKILL.md demands ---------------------
    template = "templates/experiment-plan.md"
    if os.path.isfile(template):
        text = open(template, encoding="utf-8").read().lower()
        for term in REQUIRED_TEMPLATE_TERMS:
            check(term in text, f"experiment-plan template records '{term}'")

    # --- 7. packaging hygiene ------------------------------------------------
    for leak in (".git", ".github"):
        if os.path.exists(leak):
            warn(
                f"{leak}/ is present. package_skill.py does not exclude dotfile "
                f"directories, so packaging this checkout in place would bundle "
                f"it into the distributed .skill. Strip it from the staged copy."
            )

    print()
    for w in warnings:
        print("note: " + w.split(". ")[0] + ".")
    if failures:
        print(f"\n{len(failures)} check(s) failed:")
        for f in failures:
            print("  " + f)
        return 1
    print("All package consistency checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))

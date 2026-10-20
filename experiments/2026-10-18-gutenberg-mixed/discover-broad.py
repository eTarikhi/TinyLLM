#!/usr/bin/env python3
"""Find non-narrative GITenberg repositories by searching, not by guessing ids.

`resolve-broad.py` starts from a hand-written list of Project Gutenberg ids
and verifies each one. That works, but the ids come from memory and the yield
showed it: 17 of 79 survived, and one of the survivors was Locke's *Essay*
wearing a label that said *The Analects of Confucius*.

This inverts the direction. It asks the GitHub search API which repositories
actually exist under the GITenberg organisation whose names match a form
keyword -- `plays`, `poems`, `tragedy`, `essays`, `treatise` -- takes the
Project Gutenberg id from the `_<id>` suffix the organisation uses, and
verifies the archive the same way. The title is then read off the repository
rather than asserted against it, so the mislabelling failure mode cannot
occur at all.

The form label still comes from the keyword that found the repository, which
is a weaker signal than a hand label and is treated as such: a repository
found by more than one keyword takes the first match in KEYWORDS order, and
the labels are written to JSON for inspection before anything is committed.

Usage:
    python experiments/2026-10-18-gutenberg-mixed/discover-broad.py [--limit N]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

_spec = importlib.util.spec_from_file_location("resolve_broad", HERE / "resolve-broad.py")
_rb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rb)
verify = _rb.verify

# Keyword -> form. Order matters: the first keyword that finds a repository
# assigns its form. Narrative keywords are deliberately absent; the corpus has
# 91% narrative prose already and this is a search for the other 9%.
KEYWORDS: list[tuple[str, str]] = [
    ("Sonnets", "verse"),
    ("Poems", "verse"),
    ("Poetical", "verse"),
    ("Poetry", "verse"),
    ("Ballads", "verse"),
    ("Odes", "verse"),
    ("Tragedy", "drama"),
    ("Tragedie", "drama"),
    ("Comedy", "drama"),
    ("Plays", "drama"),
    ("Play", "drama"),
    ("Essays", "non-fiction"),
    ("Essay", "non-fiction"),
    ("Treatise", "non-fiction"),
    ("Discourse", "non-fiction"),
    ("Philosophy", "non-fiction"),
    ("Lectures", "non-fiction"),
    ("Autobiography", "non-fiction"),
    ("Memoirs", "non-fiction"),
    ("Speeches", "non-fiction"),
    ("Orations", "non-fiction"),
    ("Principles", "non-fiction"),
]

# Repositories to skip regardless of keyword: ambiguous, duplicated, or so
# large that one document would dominate a form group on its own.
MAX_BYTES = 6_000_000


def search(keyword: str) -> list[str]:
    out = subprocess.run(
        ["gh", "api", "--paginate",
         f"search/repositories?q={keyword}+in:name+user:GITenberg&per_page=100",
         "--jq", ".items[].name"],
        capture_output=True, text=True, timeout=180,
    )
    if out.returncode != 0:
        print(f"  search failed for {keyword}: {out.stderr.strip()[:140]}", file=sys.stderr)
        return []
    return [n for n in out.stdout.split() if re.search(r"_\d+$", n)]


def title_from_slug(slug: str) -> str:
    t = slug.replace("-", " ").replace("_", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t[:120]


def already_present() -> set[int]:
    text = (ROOT / "scripts" / "tiny-gpt" / "src" / "data.py").read_text()
    return {int(m) for m in re.findall(r"_gb\((\d+),", text)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=90,
                    help="stop after this many verified repositories")
    ap.add_argument("--per-keyword", type=int, default=6,
                    help="verify at most this many hits per keyword, to keep "
                         "one prolific form from swamping the others")
    args = ap.parse_args()

    seen = already_present()
    known = {c[0] for c in _rb.CANDIDATES}
    found: dict[int, tuple[str, str]] = {}  # pg_id -> (repo name, form)

    for keyword, form in KEYWORDS:
        names = search(keyword)
        time.sleep(2.2)
        added = 0
        for name in names:
            pg_id = int(name.rsplit("_", 1)[1])
            if pg_id in seen or pg_id in known or pg_id in found:
                continue
            found[pg_id] = (name, form)
            added += 1
            if added >= args.per_keyword * 3:  # over-collect; verification prunes
                break
        print(f"  {keyword:>14} -> {len(names):>4} repos, {added} new candidates")

    print(f"\n{len(found)} distinct new candidates to verify")
    good, bad, total = [], [], 0
    per_form: dict[str, int] = {}
    for pg_id, (name, form) in sorted(found.items()):
        if len(good) >= args.limit:
            break
        if per_form.get(form, 0) >= args.per_keyword * 8:
            continue
        slug = name[: -(len(str(pg_id)) + 1)]
        ok, size, member_file = verify(name, pg_id)
        if not ok:
            bad.append((pg_id, name, "no strictly-UTF-8 member"))
            continue
        if size > MAX_BYTES:
            bad.append((pg_id, name, f"too large ({size/1e6:.1f} MB)"))
            continue
        if size < 20_000:
            bad.append((pg_id, name, f"too small ({size/1024:.0f} KB)"))
            continue
        total += size
        per_form[form] = per_form.get(form, 0) + 1
        good.append({"pg_id": pg_id, "slug": slug, "title": title_from_slug(slug),
                     "author": "", "year": "", "form": form, "raw_bytes": size,
                     "member_file": member_file})
        print(f"  ok  {pg_id:>6}  {size/1024:>7.0f} KB  {form:>11}  {name}")

    print(f"\n{len(good)} verified, {len(bad)} rejected, {total/1e6:.1f} MB raw")
    by_form: dict[str, int] = {}
    for g in good:
        by_form[g["form"]] = by_form.get(g["form"], 0) + g["raw_bytes"]
    for form, size in sorted(by_form.items(), key=lambda kv: -kv[1]):
        n = sum(1 for g in good if g["form"] == form)
        print(f"  {form:>12} {n:>3} titles {size/1e6:>6.1f} MB")

    out = HERE / "discovered-broad.json"
    out.write_text(json.dumps(good, indent=1) + "\n")
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

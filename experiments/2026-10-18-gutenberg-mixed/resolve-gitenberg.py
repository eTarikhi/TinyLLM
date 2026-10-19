#!/usr/bin/env python3
"""Resolve Project Gutenberg ids to GITenberg repositories, and verify each one.

A hand-written repo slug is a guess. This asks the GitHub search API for the
repository whose name ends in `_<pg_id>` under the GITenberg organisation,
then downloads the master tarball and confirms the UTF-8 member
`<slug>_<id>-master/<id>-0.txt` is really inside it. Only verified entries are
printed as `_gb(...)` lines ready to paste into `src/data.py`.

Usage:  python experiments/2026-10-18-gutenberg-mixed/resolve-gitenberg.py
"""
from __future__ import annotations

import io
import json
import subprocess
import sys
import tarfile
import time
import urllib.request

# Candidate Project Gutenberg ids, with the title expected back so a wrong
# resolution is visible rather than silent. None of these is in the 27-book
# corpus already in src/data.py.
CANDIDATES = [
    (2600, "War and Peace", "Leo Tolstoy (tr. Maude)", "1869"),
    (1399, "Anna Karenina", "Leo Tolstoy (tr. Garnett)", "1878"),
    (766, "David Copperfield", "Charles Dickens", "1850"),
    (730, "Oliver Twist", "Charles Dickens", "1838"),
    (580, "The Pickwick Papers", "Charles Dickens", "1837"),
    (1023, "Bleak House", "Charles Dickens", "1853"),
    (963, "Little Dorrit", "Charles Dickens", "1857"),
    (883, "Our Mutual Friend", "Charles Dickens", "1865"),
    (786, "Hard Times", "Charles Dickens", "1854"),
    (46, "A Christmas Carol", "Charles Dickens", "1843"),
    (105, "Persuasion", "Jane Austen", "1817"),
    (121, "Northanger Abbey", "Jane Austen", "1817"),
    (141, "Mansfield Park", "Jane Austen", "1814"),
    (2852, "The Hound of the Baskervilles", "Arthur Conan Doyle", "1902"),
    (244, "A Study in Scarlet", "Arthur Conan Doyle", "1887"),
    (834, "The Memoirs of Sherlock Holmes", "Arthur Conan Doyle", "1894"),
    (108, "The Return of Sherlock Holmes", "Arthur Conan Doyle", "1905"),
    (2097, "The Sign of the Four", "Arthur Conan Doyle", "1890"),
    (1184, "The Count of Monte Cristo", "Alexandre Dumas (tr.)", "1844"),
    (1257, "The Three Musketeers", "Alexandre Dumas (tr.)", "1844"),
    (2413, "Madame Bovary", "Gustave Flaubert (tr. Marx-Aveling)", "1856"),
    (135, "Les Miserables", "Victor Hugo (tr. Hapgood)", "1862"),
    (6130, "The Iliad", "Homer (tr. Pope)", "-750"),
    (1727, "The Odyssey", "Homer (tr. Butcher and Lang)", "-725"),
    (4300, "Ulysses", "James Joyce", "1922"),
    (4217, "A Portrait of the Artist as a Young Man", "James Joyce", "1916"),
    (2814, "Dubliners", "James Joyce", "1914"),
    (160, "The Awakening", "Kate Chopin", "1899"),
    (205, "Walden", "Henry David Thoreau", "1854"),
    (215, "The Call of the Wild", "Jack London", "1903"),
    (910, "White Fang", "Jack London", "1906"),
    (43, "The Strange Case of Dr. Jekyll and Mr. Hyde", "Robert Louis Stevenson", "1886"),
    (521, "Robinson Crusoe", "Daniel Defoe", "1719"),
    (829, "Gulliver's Travels", "Jonathan Swift", "1726"),
    (103, "Around the World in Eighty Days", "Jules Verne (tr.)", "1872"),
    (164, "Twenty Thousand Leagues under the Sea", "Jules Verne (tr.)", "1870"),
    (209, "The Turn of the Screw", "Henry James", "1898"),
    (2641, "A Room with a View", "E. M. Forster", "1908"),
    (113, "The Secret Garden", "Frances Hodgson Burnett", "1911"),
    (146, "A Little Princess", "Frances Hodgson Burnett", "1905"),
    (271, "Black Beauty", "Anna Sewell", "1877"),
    (236, "The Jungle Book", "Rudyard Kipling", "1894"),
    (25344, "The Scarlet Letter", "Nathaniel Hawthorne", "1850"),
    (77, "The House of the Seven Gables", "Nathaniel Hawthorne", "1851"),
    (2383, "The Canterbury Tales", "Geoffrey Chaucer", "1400"),
    (1998, "Thus Spake Zarathustra", "Friedrich Nietzsche (tr. Common)", "1883"),
    (1497, "The Republic", "Plato (tr. Jowett)", "-375"),
    (3300, "The Wealth of Nations", "Adam Smith", "1776"),
    (2680, "Meditations", "Marcus Aurelius (tr. Long)", "180"),
    (3207, "Leviathan", "Thomas Hobbes", "1651"),
    (19942, "Candide", "Voltaire (tr.)", "1759"),
    (6593, "The History of Tom Jones", "Henry Fielding", "1749"),
    (5200, "Metamorphosis", "Franz Kafka (tr. Johnston)", "1915"),
    (28054, "The Brothers Karamazov", "Fyodor Dostoyevsky (tr. Garnett)", "1880"),
    (2638, "The Idiot", "Fyodor Dostoyevsky (tr. Martin)", "1869"),
    (600, "Notes from the Underground", "Fyodor Dostoyevsky (tr. Garnett)", "1864"),
    (203, "Uncle Tom's Cabin", "Harriet Beecher Stowe", "1852"),
    (16, "Peter Pan", "J. M. Barrie", "1911"),
    (289, "The Wind in the Willows", "Kenneth Grahame", "1908"),
    (8800, "The Divine Comedy", "Dante Alighieri (tr. Longfellow)", "1320"),
    (3176, "The Innocents Abroad", "Mark Twain", "1869"),
    (1837, "The Prince and the Pauper", "Mark Twain", "1881"),
    (86, "A Connecticut Yankee in King Arthur's Court", "Mark Twain", "1889"),
    (1952, "The Yellow Wallpaper", "Charlotte Perkins Gilman", "1892"),
    (2500, "Siddhartha", "Hermann Hesse (tr. Rosner)", "1922"),
]

UA = {"User-Agent": "tiny-gpt-corpus-builder"}


def resolve(pg_id: int) -> str | None:
    """Exact repository name for a Project Gutenberg id, via the search API."""
    out = subprocess.run(
        ["gh", "api", f"search/repositories?q=_{pg_id}+in:name+user:GITenberg&per_page=100",
         "--jq", ".items[].name"],
        capture_output=True, text=True, timeout=60,
    )
    if out.returncode != 0:
        print(f"  search failed for {pg_id}: {out.stderr.strip()[:120]}", file=sys.stderr)
        return None
    for name in out.stdout.split():
        if name.endswith(f"_{pg_id}"):
            return name
    return None


def verify(slug_with_id: str, pg_id: int) -> tuple[bool, int]:
    """Download the master tarball and confirm the UTF-8 member is present."""
    url = f"https://codeload.github.com/GITenberg/{slug_with_id}/tar.gz/refs/heads/master"
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=120) as r:
            blob = r.read()
    except Exception as exc:
        print(f"  download failed {slug_with_id}: {exc}", file=sys.stderr)
        return False, 0
    member = f"{slug_with_id}-master/{pg_id}-0.txt"
    try:
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tf:
            info = tf.getmember(member)
            return True, info.size
    except KeyError:
        return False, 0
    except Exception as exc:
        print(f"  tar failed {slug_with_id}: {exc}", file=sys.stderr)
        return False, 0


def main() -> int:
    good, bad, total_bytes = [], [], 0
    for pg_id, title, author, year in CANDIDATES:
        name = resolve(pg_id)
        time.sleep(2.2)  # search API allows 30 requests per minute
        if not name:
            bad.append((pg_id, title, "no repo"))
            continue
        slug = name[: -(len(str(pg_id)) + 1)]
        ok, size = verify(name, pg_id)
        if not ok:
            bad.append((pg_id, title, "no UTF-8 member"))
            continue
        total_bytes += size
        good.append((pg_id, slug, title, author, year, size))
        print(f"  ok  {pg_id:>6}  {size/1024:>7.0f} KB  {name}")

    print(f"\n{len(good)} verified, {len(bad)} rejected, {total_bytes/1e6:.1f} MB raw")
    for pg_id, title, why in bad:
        print(f"  reject {pg_id} {title}: {why}")
    print()
    for pg_id, slug, title, author, year, _ in good:
        esc = title.replace('"', '\\"')
        print(f'    _gb({pg_id}, "{slug}", "{esc}", "{author}", "{year}"),')
    json.dump([{"pg_id": i, "slug": s, "title": t, "author": a, "year": y, "raw_bytes": z}
               for i, s, t, a, y, z in good],
              open("/tmp/resolved.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())

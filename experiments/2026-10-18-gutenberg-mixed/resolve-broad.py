#!/usr/bin/env python3
"""Resolve the non-narrative candidates for `gutenberg-mixed-broad`.

`corpus-composition.md` measured what the 50.6% per-document spread is made
of: not length (r = -0.122) but form. The corpus is 91% narrative prose by
bytes and contains no drama at all, and the four form labels explain 45% of
the variance -- verse +27.8% against narrative prose, experimental +23.4%,
non-fiction +19.3%, each holding about 3% of the bytes. Books by an author
who appears more than once score 5.0% better, which is the evidence that
those forms are under-trained rather than intrinsically unpredictable.

This resolves the titles that would test that, reusing the verification from
`resolve-gitenberg.py` rather than reimplementing it: ask the GitHub search
API for the repository whose name ends in `_<pg_id>` under the GITenberg
organisation, download the master tarball, and confirm the canonical UTF-8
member `<id>-0.txt` is really inside. A hand-written slug is a guess; only
verified entries are printed.

Candidates are deliberately over-listed. Resolution is cheap compared with
training, several of these ids are from memory rather than from a catalogue,
and the verifier rejects the wrong ones loudly. Anything already in
`src/data.py` is filtered out before the network is touched, so the same
book cannot enter the corpus twice.

Usage:  python experiments/2026-10-18-gutenberg-mixed/resolve-broad.py
"""
from __future__ import annotations

import importlib.util
import io
import json
import re
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

_spec = importlib.util.spec_from_file_location("resolve_gitenberg",
                                               HERE / "resolve-gitenberg.py")
_rg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rg)
resolve = _rg.resolve


STOPWORDS = {"the", "a", "an", "of", "and", "or", "in", "on", "to", "by", "for",
             "complete", "volume", "vol", "book", "books", "part", "selected",
             "works", "poetical", "plays", "essays", "first", "second", "series"}


def words(text: str) -> set[str]:
    return {w for w in re.split(r"[^a-z0-9]+", text.lower()) if w and w not in STOPWORDS}


def title_agrees(slug: str, title: str) -> tuple[bool, float]:
    """Does the resolved repository look like the book that was asked for?

    `resolve-gitenberg.py` prints the expected title next to the resolved slug
    and calls that "visible rather than silent", but nothing compared them, so
    a search hit on the right id with the wrong book passed straight through.
    It happened: PG 10615 resolves to Locke's Essay Concerning Humane
    Understanding and was about to be committed labelled "The Analects of
    Confucius". This makes the comparison an actual gate.

    Scored as the share of the title's significant words that appear in the
    slug. A collection can legitimately be named for its series rather than
    for the work inside it, so a failure is reported for a human to resolve,
    never silently repaired.
    """
    tw, sw = words(title), words(slug)
    if not tw:
        return True, 1.0
    hit = len(tw & sw) / len(tw)
    return hit >= 0.5, hit


def verify(slug_with_id: str, pg_id: int) -> tuple[bool, int, str]:
    """Confirm a strictly-UTF-8 member exists, preferring the canonical name.

    Returns (ok, size, member_file). `<id>-0.txt` is the UTF-8 member by
    GITenberg convention; where it is absent, `<id>.txt` is accepted only if
    its bytes decode as strict UTF-8. Decoding is the gate rather than the
    file name, so this cannot pull Latin-1 into the corpus.
    """
    url = f"https://codeload.github.com/GITenberg/{slug_with_id}/tar.gz/refs/heads/master"
    try:
        req = urllib.request.Request(url, headers=_rg.UA)
        with urllib.request.urlopen(req, timeout=180) as r:
            blob = r.read()
    except Exception as exc:
        print(f"  download failed {slug_with_id}: {exc}", file=sys.stderr)
        return False, 0, ""
    try:
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tf:
            for candidate in (f"{pg_id}-0.txt", f"{pg_id}.txt", f"{pg_id}-8.txt"):
                member = f"{slug_with_id}-master/{candidate}"
                try:
                    info = tf.getmember(member)
                except KeyError:
                    continue
                handle = tf.extractfile(info)
                if handle is None:
                    continue
                payload = handle.read()
                try:
                    payload.decode("utf-8")
                except UnicodeDecodeError:
                    print(f"    {candidate} is not UTF-8, skipping", file=sys.stderr)
                    continue
                return True, info.size, candidate
    except Exception as exc:
        print(f"  tar failed {slug_with_id}: {exc}", file=sys.stderr)
    return False, 0, ""


# (pg_id, title, author, year, form)
CANDIDATES: list[tuple[int, str, str, str, str]] = [
    # ---- drama: the corpus currently contains none at all
    (1524, "Hamlet", "William Shakespeare", "1603", "drama"),
    (1533, "Macbeth", "William Shakespeare", "1606", "drama"),
    (1513, "Romeo and Juliet", "William Shakespeare", "1597", "drama"),
    (1514, "A Midsummer Night's Dream", "William Shakespeare", "1600", "drama"),
    (1532, "King Lear", "William Shakespeare", "1606", "drama"),
    (1531, "Othello", "William Shakespeare", "1604", "drama"),
    (1522, "Julius Caesar", "William Shakespeare", "1599", "drama"),
    (1526, "Twelfth Night", "William Shakespeare", "1602", "drama"),
    (1515, "The Merchant of Venice", "William Shakespeare", "1600", "drama"),
    (1519, "Much Ado About Nothing", "William Shakespeare", "1600", "drama"),
    (1121, "As You Like It", "William Shakespeare", "1600", "drama"),
    (2235, "The Tempest", "William Shakespeare", "1611", "drama"),
    (844, "The Importance of Being Earnest", "Oscar Wilde", "1895", "drama"),
    (790, "Lady Windermere's Fan", "Oscar Wilde", "1892", "drama"),
    (885, "An Ideal Husband", "Oscar Wilde", "1895", "drama"),
    (854, "A Woman of No Importance", "Oscar Wilde", "1893", "drama"),
    (2542, "A Doll's House", "Henrik Ibsen (tr.)", "1879", "drama"),
    (4093, "Hedda Gabler", "Henrik Ibsen (tr.)", "1891", "drama"),
    (8121, "Ghosts", "Henrik Ibsen (tr.)", "1881", "drama"),
    (2446, "An Enemy of the People", "Henrik Ibsen (tr.)", "1882", "drama"),
    (3825, "Pygmalion", "George Bernard Shaw", "1913", "drama"),
    (3328, "Man and Superman", "George Bernard Shaw", "1903", "drama"),
    (3618, "Arms and the Man", "George Bernard Shaw", "1894", "drama"),
    (7986, "The Cherry Orchard", "Anton Chekhov (tr.)", "1904", "drama"),
    (779, "Doctor Faustus", "Christopher Marlowe", "1604", "drama"),
    (1929, "The School for Scandal", "Richard Brinsley Sheridan", "1777", "drama"),
    (14591, "Faust", "Johann Wolfgang von Goethe (tr.)", "1808", "drama"),
    (31, "Oedipus King of Thebes", "Sophocles (tr. Murray)", "-429", "drama"),
    (1240, "The Playboy of the Western World", "J. M. Synge", "1907", "drama"),

    # ---- verse: currently two books, 3.1% of bytes
    (26, "Paradise Lost", "John Milton", "1667", "verse"),
    (58, "Paradise Regained", "John Milton", "1671", "verse"),
    (8800, "The Divine Comedy", "Dante Alighieri (tr. Longfellow)", "1320", "verse"),
    (2383, "The Canterbury Tales", "Geoffrey Chaucer", "1400", "verse"),
    (1727, "The Odyssey", "Homer (tr. Butcher and Lang)", "-725", "verse"),
    (228, "The Aeneid", "Virgil (tr. Dryden)", "-19", "verse"),
    (21765, "Metamorphoses", "Ovid (tr. More)", "8", "verse"),
    (16328, "Beowulf", "Anonymous (tr. Hall)", "1000", "verse"),
    (15272, "The Faerie Queene", "Edmund Spenser", "1590", "verse"),
    (21700, "Don Juan", "Lord Byron", "1819", "verse"),
    (246, "The Rubaiyat of Omar Khayyam", "Omar Khayyam (tr. FitzGerald)", "1859", "verse"),
    (12242, "Poems by Emily Dickinson", "Emily Dickinson", "1890", "verse"),
    (610, "Idylls of the King", "Alfred Tennyson", "1859", "verse"),
    (1934, "Songs of Innocence and of Experience", "William Blake", "1794", "verse"),
    (19, "The Song of Hiawatha", "Henry Wadsworth Longfellow", "1855", "verse"),
    (1280, "Spoon River Anthology", "Edgar Lee Masters", "1915", "verse"),
    (5131, "The Poetical Works of John Keats", "John Keats", "1820", "verse"),
    (4800, "The Complete Poetical Works of Percy Bysshe Shelley",
     "Percy Bysshe Shelley", "1820", "verse"),
    (12145, "The Poetical Works of Samuel Taylor Coleridge",
     "Samuel Taylor Coleridge", "1834", "verse"),
    (9622, "The Waste Land", "T. S. Eliot", "1922", "verse"),

    # ---- non-fiction: currently three books, 2.9% of bytes
    (3300, "An Inquiry into the Nature and Causes of the Wealth of Nations",
     "Adam Smith", "1776", "non-fiction"),
    (1404, "The Federalist Papers", "Hamilton, Madison and Jay", "1788", "non-fiction"),
    (3207, "Leviathan", "Thomas Hobbes", "1651", "non-fiction"),
    (1497, "The Republic", "Plato (tr. Jowett)", "-375", "non-fiction"),
    (2680, "Meditations", "Marcus Aurelius (tr. Long)", "180", "non-fiction"),
    (1232, "The Prince", "Niccolo Machiavelli (tr. Marriott)", "1532", "non-fiction"),
    (34901, "On Liberty", "John Stuart Mill", "1859", "non-fiction"),
    (815, "Democracy in America, Volume 1",
     "Alexis de Tocqueville (tr. Reeve)", "1835", "non-fiction"),
    (20203, "The Autobiography of Benjamin Franklin",
     "Benjamin Franklin", "1791", "non-fiction"),
    (16643, "Essays of Ralph Waldo Emerson", "Ralph Waldo Emerson", "1841", "non-fiction"),
    (408, "The Souls of Black Folk", "W. E. B. Du Bois", "1903", "non-fiction"),
    (7370, "Second Treatise of Government", "John Locke", "1689", "non-fiction"),
    (2300, "The Descent of Man", "Charles Darwin", "1871", "non-fiction"),
    (944, "The Voyage of the Beagle", "Charles Darwin", "1839", "non-fiction"),
    (5001, "Relativity: The Special and General Theory",
     "Albert Einstein (tr. Lawson)", "1916", "non-fiction"),
    (25717, "The History of the Decline and Fall of the Roman Empire, Volume 1",
     "Edward Gibbon", "1776", "non-fiction"),
    (9662, "An Enquiry Concerning Human Understanding", "David Hume", "1748", "non-fiction"),
    (3420, "A Vindication of the Rights of Woman",
     "Mary Wollstonecraft", "1792", "non-fiction"),
    (61, "The Communist Manifesto", "Marx and Engels (tr. Moore)", "1848", "non-fiction"),
    (147, "Common Sense", "Thomas Paine", "1776", "non-fiction"),
    (132, "The Art of War", "Sun Tzu (tr. Giles)", "-500", "non-fiction"),
    (4280, "The Critique of Pure Reason", "Immanuel Kant (tr. Meiklejohn)", "1781", "non-fiction"),
    (833, "The Theory of the Leisure Class", "Thorstein Veblen", "1899", "non-fiction"),
    (2397, "The Story of My Life", "Helen Keller", "1903", "non-fiction"),
    (2376, "Up from Slavery", "Booker T. Washington", "1901", "non-fiction"),
    (11030, "Incidents in the Life of a Slave Girl", "Harriet Jacobs", "1861", "non-fiction"),
    (10615, "The Analects of Confucius", "Confucius (tr. Legge)", "-400", "non-fiction"),
    (1656, "Apology", "Plato (tr. Jowett)", "-399", "non-fiction"),
    (3600, "Essays of Michel de Montaigne", "Michel de Montaigne (tr. Cotton)",
     "1580", "non-fiction"),
    (6762, "The Advancement of Learning", "Francis Bacon", "1605", "non-fiction"),
]


def already_present() -> set[int]:
    """Project Gutenberg ids already registered in src/data.py."""
    text = (ROOT / "scripts" / "tiny-gpt" / "src" / "data.py").read_text()
    return {int(m) for m in re.findall(r"_gb\((\d+),", text)}


def main() -> int:
    seen = already_present()
    dupes = [c for c in CANDIDATES if c[0] in seen]
    todo = [c for c in CANDIDATES if c[0] not in seen]
    ids = [c[0] for c in CANDIDATES]
    assert len(ids) == len(set(ids)), "the candidate list repeats an id"
    print(f"{len(CANDIDATES)} candidates, {len(dupes)} already in the corpus, "
          f"{len(todo)} to resolve")
    for pg_id, title, *_ in dupes:
        print(f"  skip {pg_id} {title}: already registered")

    good, bad, total = [], [], 0
    for pg_id, title, author, year, form in todo:
        name = resolve(pg_id)
        time.sleep(2.2)  # the search API allows 30 requests per minute
        if not name:
            bad.append((pg_id, title, "no repo"))
            continue
        slug = name[: -(len(str(pg_id)) + 1)]
        agrees, score = title_agrees(slug, title)
        if not agrees:
            bad.append((pg_id, title, f"title mismatch ({score:.0%}) -- repo is {slug}"))
            continue
        ok, size, member_file = verify(name, pg_id)
        if not ok:
            bad.append((pg_id, title, "no strictly-UTF-8 member"))
            continue
        total += size
        good.append((pg_id, slug, title, author, year, form, size, member_file))
        flag = "" if member_file == f"{pg_id}-0.txt" else f"  [{member_file}]"
        print(f"  ok  {pg_id:>6}  {size/1024:>7.0f} KB  {form:>11}  {name}{flag}")

    print(f"\n{len(good)} verified, {len(bad)} rejected, {total/1e6:.1f} MB raw")
    for pg_id, title, why in bad:
        print(f"  reject {pg_id} {title}: {why}")

    by_form: dict[str, int] = {}
    for *_, form, size, _m in good:
        by_form[form] = by_form.get(form, 0) + size
    print("\nraw bytes by form:")
    for form, size in sorted(by_form.items(), key=lambda kv: -kv[1]):
        print(f"  {form:>12} {size/1e6:>6.1f} MB")

    print()
    for pg_id, slug, title, author, year, form, _sz, member_file in good:
        esc = title.replace('"', '\\"')
        extra = "" if member_file == f"{pg_id}-0.txt" else f', member_file="{member_file}"'
        print(f'    _gb({pg_id}, "{slug}", "{esc}", "{author}", "{year}"{extra}),  # {form}')

    out = HERE / "broad-candidates.json"
    out.write_text(json.dumps(
        [{"pg_id": i, "slug": s, "title": t, "author": a, "year": y, "form": f,
          "raw_bytes": z, "member_file": m}
         for i, s, t, a, y, f, z, m in good], indent=1) + "\n")
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

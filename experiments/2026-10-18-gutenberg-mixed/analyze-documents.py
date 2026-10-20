#!/usr/bin/env python3
"""What the per-document scores say about which corpus to build next.

Every hyperparameter this project has tested is now flat or harmful, and the
largest single number anywhere in the results is not a lever at all: the 71
test books span 50.6% of the mean in bits per byte. Before spending hours
fetching more text, it is worth asking what that spread is *made of*, because
"add more books" and "add more books of a kind the model is bad at" are
different plans with different costs.

The spread could be three things, and they imply different actions:

  1. length        -- short documents score worse because there is less of
                      them to amortise the model's warm-up on an unfamiliar
                      style. Action: nothing; it is an artifact.
  2. form          -- verse and experimental prose are genuinely harder than
                      narrative prose. Action: broaden the corpus by form.
  3. training mass -- books whose authors contribute more training text score
                      better. Action: balance the corpus, not just grow it.

This script tests all three against the released model's per-document
evaluation. The form labels are hand-assigned and marked as such: they are
the one judgement call here, so they are written out in full in LABELS below
rather than hidden in a lookup, and the script reports how much of the
variance they explain so a reader can discount them appropriately.

Usage (from the repository root):
    python experiments/2026-10-18-gutenberg-mixed/analyze-documents.py
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RELEASED = "data71-full-9600"

# Hand-assigned form labels. "narrative" is the default and covers the
# 19th-century novel this corpus is mostly made of; the others are the
# exceptions worth separating. Assigned from the title and author, before
# looking at any score.
LABELS: dict[str, str] = {
    "Leaves of Grass": "verse",
    "The Iliad": "verse",
    "The Odyssey": "verse",
    "The Divine Comedy": "verse",
    "Paradise Lost": "verse",
    "The Rime of the Ancient Mariner": "verse",
    "Ulysses": "experimental",
    "A Portrait of the Artist as a Young Man": "experimental",
    "Tristram Shandy": "experimental",
    "The Life and Opinions of Tristram Shandy, Gentleman": "experimental",
    "On the Origin of Species": "non-fiction",
    "The Souls of Black Folk": "non-fiction",
    "Walden": "non-fiction",
    "Narrative of the Life of Frederick Douglass": "non-fiction",
    "Narrative of the Life of Frederick Douglass, an American Slave": "non-fiction",
    "Common Sense": "non-fiction",
    "The Interesting Narrative of the Life of Olaudah Equiano": "non-fiction",
    "Romeo and Juliet": "drama",
    "Hamlet": "drama",
    "Macbeth": "drama",
    "The Importance of Being Earnest": "drama",
    "An Ideal Husband": "drama",
    "A Doll's House": "drama",
}


def label_for(title: str) -> str:
    if title in LABELS:
        return LABELS[title]
    for key, value in LABELS.items():
        if title.startswith(key[:28]):
            return value
    return "narrative"


def line(ch: str = "=") -> None:
    print(ch * 96)


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    return num / (dx * dy) if dx and dy else float("nan")


def main() -> int:
    path = HERE / f"eval-test-{RELEASED}.json"
    if not path.is_file():
        print(f"missing {path.name}")
        return 1
    docs = [d for d in json.load(path.open())["per_document"] if d.get("scored")]
    out: dict = {"model": RELEASED, "documents": len(docs)}

    line()
    print(f"PER-DOCUMENT SPREAD — what the {len(docs)} test books disagree about")
    line()
    bpbs = [d["bpb"] for d in docs]
    print(f"  mean {statistics.mean(bpbs):.4f}  sd {statistics.stdev(bpbs):.4f}  "
          f"min {min(bpbs):.4f}  max {max(bpbs):.4f}")

    # ---- hypothesis 1: it is just length
    sizes = [d["bytes"] for d in docs]
    r_len = pearson([float(s) for s in sizes], bpbs)
    out["corr_bpb_vs_bytes"] = r_len
    print(f"\n  1. length: correlation between document size and bits per byte "
          f"r = {r_len:+.3f}")
    print(f"     {'a real effect' if abs(r_len) > 0.5 else 'weak -- length does not explain the spread'}")

    # ---- hypothesis 2: it is form
    groups: dict[str, list[dict]] = {}
    for d in docs:
        groups.setdefault(label_for(d["title"]), []).append(d)
    print("\n  2. form (hand-assigned labels):")
    print(f"     {'label':>14} {'n':>3} {'bytes':>10} {'mean bpb':>9} {'vs narrative':>13}")
    narrative = statistics.mean([d["bpb"] for d in groups["narrative"]])
    form: dict[str, dict] = {}
    for label in sorted(groups, key=lambda k: statistics.mean(d["bpb"] for d in groups[k])):
        g = groups[label]
        m = statistics.mean(d["bpb"] for d in g)
        form[label] = {"n": len(g), "bytes": sum(d["bytes"] for d in g), "mean_bpb": m,
                       "vs_narrative_pct": 100 * (m / narrative - 1)}
        print(f"     {label:>14} {len(g):>3} {sum(d['bytes'] for d in g):>10,} {m:>9.4f} "
              f"{form[label]['vs_narrative_pct']:>+12.1f}%")
    out["form"] = form

    # Between-group variance as a share of the total: how much of the spread
    # do these labels actually account for?
    grand = statistics.mean(bpbs)
    ss_total = sum((b - grand) ** 2 for b in bpbs)
    ss_between = sum(len(g) * (statistics.mean(d["bpb"] for d in g) - grand) ** 2
                     for g in groups.values())
    out["form_variance_explained"] = ss_between / ss_total
    print(f"     these labels explain {100 * ss_between / ss_total:.0f}% of the variance "
          f"in bits per byte")

    # ---- hypothesis 3: training mass per author
    by_author: dict[str, list[dict]] = {}
    for d in docs:
        by_author.setdefault((d.get("author") or "?").split(" (")[0], []).append(d)
    multi = {a: g for a, g in by_author.items() if len(g) > 1}
    print(f"\n  3. training mass: {len(multi)} authors contribute more than one book")
    if multi:
        multi_bpb = statistics.mean(d["bpb"] for g in multi.values() for d in g)
        solo_bpb = statistics.mean(d["bpb"] for a, g in by_author.items()
                                   if len(g) == 1 for d in g)
        out["multi_book_authors"] = len(multi)
        out["mean_bpb_multi_book_author"] = multi_bpb
        out["mean_bpb_single_book_author"] = solo_bpb
        out["multi_book_advantage_pct"] = 100 * (1 - multi_bpb / solo_bpb)
        print(f"     books by a repeated author  {multi_bpb:.4f}")
        print(f"     books by a one-off author   {solo_bpb:.4f}")
        print(f"     advantage of being repeated {100 * (1 - multi_bpb / solo_bpb):+.1f}%")

    # ---- what to build next
    line()
    print("IMPLICATION FOR THE NEXT CORPUS")
    line()
    worst = sorted(form, key=lambda k: -form[k]["vs_narrative_pct"])
    narrative_share = 100 * form["narrative"]["bytes"] / sum(f["bytes"] for f in form.values())
    out["narrative_share_pct"] = narrative_share
    print(f"  the corpus is {narrative_share:.0f}% narrative prose by bytes")
    for label in worst:
        if label == "narrative":
            continue
        f = form[label]
        print(f"  {label:>14} is {f['vs_narrative_pct']:+.1f}% and holds only "
              f"{100 * f['bytes'] / sum(x['bytes'] for x in form.values()):.1f}% of the bytes")
    print("\n  the hard categories are hard AND rare, which is the combination that")
    print("  says the model has not seen enough of them rather than that they are")
    print("  intrinsically unpredictable -- the testable claim for the next corpus")

    p = HERE / "document-summary.json"
    p.write_text(json.dumps(out, indent=1) + "\n")
    print(f"\nwritten to {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

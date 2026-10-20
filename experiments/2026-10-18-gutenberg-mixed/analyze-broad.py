#!/usr/bin/env python3
"""Did broadening the corpus by form pay, and for which forms?

The per-document analysis of the released model found that form explains 45%
of the spread in test bits per byte -- verse cost +27.8% against narrative
prose, experimental prose +23.4%, non-fiction +19.3% -- while those three
forms were only 9.3% of the bytes. Two readings fit that. Either the forms
are intrinsically harder at this scale, or the model simply never saw enough
of them. They differ in what to do next, so this is the experiment that
separates them.

Three arms, 4,800 steps each, seed 1337, identical schedule and evaluation
grid, byte-identical val.bin / test.bin / tokenizer.json (verified with cmp):

    broad-ctrl       18,939,699 train tokens   90.7% narrative
    broad-narrative  26,057,637 train tokens   ~93%  narrative
    broad-full       25,952,894 train tokens   67.8% narrative

broad-full against broad-ctrl is the headline, but it moves volume and
mixture together. The narrative arm pins that down: it adds the same bytes
(0.02% apart) of ordinary narrative prose, so

    full vs narrative  -- composition, volume held fixed
    narrative vs ctrl  -- volume, composition held roughly fixed

The additions are train_only in every arm, so all three are scored on the
same 71 books, the ones whose form labels were hand-assigned in
analyze-documents.py and whose numbers are already published.

Everything is read from the committed eval-test.json files. Nothing here
depends on a checkpoint, so it survives a wiped workspace.

Usage (from the repository root):
    python experiments/2026-10-18-gutenberg-mixed/analyze-broad.py
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
RUNS = ROOT / "scripts" / "tiny-gpt" / "artifacts" / "runs"

ARMS = ["broad-ctrl", "broad-narrative", "broad-full"]
BASELINE = "broad-ctrl"
SEED_BAR_PCT = 0.3690475821746498   # the 3-seed range; anything under it is noise


def load_labels() -> dict[str, str]:
    """Reuse analyze-documents.py's hand-assigned labels rather than restate them."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("ad", HERE / "analyze-documents.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.LABELS


def label_of(title: str, labels: dict[str, str]) -> str:
    return labels.get(title, "narrative")


def read_arm(run: str) -> list[dict]:
    path = RUNS / run / "eval-test.json"
    if not path.exists():
        print(f"missing {path}; run scripts/tiny-gpt/run_broad.sh first", file=sys.stderr)
        sys.exit(1)
    blob = json.loads(path.read_text())
    return [r for r in blob["per_document"] if r["scored"]], blob


def pct(new: float, old: float) -> float:
    return 100.0 * (new - old) / old


def verdict(delta: float) -> str:
    if abs(delta) < SEED_BAR_PCT:
        return "inside the seed bar"
    return "better" if delta < 0 else "worse"


def main() -> int:
    labels = load_labels()
    docs: dict[str, list[dict]] = {}
    totals: dict[str, float] = {}
    for arm in ARMS:
        rows, blob = read_arm(arm)
        docs[arm] = rows
        totals[arm] = blob["model"]["bpb"]

    titles = [r["title"] for r in docs[ARMS[0]]]
    for arm in ARMS[1:]:
        if [r["title"] for r in docs[arm]] != titles:
            print(f"{arm} scored a different document set", file=sys.stderr)
            return 1
    print(f"{len(titles)} documents scored identically in all {len(ARMS)} arms\n")

    out: dict = {"arms": {}, "seed_bar_pct": SEED_BAR_PCT}

    # ---- aggregate -------------------------------------------------------
    print("aggregate test bits per byte")
    for arm in ARMS:
        d = pct(totals[arm], totals[BASELINE])
        print(f"  {arm:<16} {totals[arm]:.4f}  {d:>+7.2f}%  {verdict(d)}")
        out["arms"][arm] = {"test_bpb": totals[arm], "vs_ctrl_pct": d}

    # ---- per form --------------------------------------------------------
    # The aggregate averages over a mixture that is 91% narrative, so it is
    # almost entirely a narrative-prose number. Per form is the measurement.
    print("\nper form (mean bits per byte over the documents of that form)")
    forms = sorted({label_of(t, labels) for t in titles},
                   key=lambda f: (f != "narrative", f))
    by_form: dict[str, dict] = {}
    header = f"  {'form':<13}{'n':>3}" + "".join(f"{a.split('-')[1]:>12}" for a in ARMS)
    print(header + f"{'full-ctrl':>12}{'full-narr':>12}")
    for form in forms:
        idx = [i for i, t in enumerate(titles) if label_of(t, labels) == form]
        means = {arm: statistics.fmean(docs[arm][i]["bpb"] for i in idx)
                 for arm in ARMS}
        d_ctrl = pct(means["broad-full"], means["broad-ctrl"])
        d_narr = pct(means["broad-full"], means["broad-narrative"])
        row = f"  {form:<13}{len(idx):>3}" + "".join(f"{means[a]:>12.4f}" for a in ARMS)
        print(row + f"{d_ctrl:>+11.2f}%{d_narr:>+11.2f}%")
        by_form[form] = {
            "n": len(idx),
            **{arm: means[arm] for arm in ARMS},
            "full_vs_ctrl_pct": d_ctrl,
            "full_vs_narrative_pct": d_narr,
            "verdict_vs_ctrl": verdict(d_ctrl),
            "verdict_vs_narrative": verdict(d_narr),
        }
    out["by_form"] = by_form

    # ---- the decomposition the three arms were built for -----------------
    print("\ndecomposition, aggregate")
    vol = pct(totals["broad-narrative"], totals["broad-ctrl"])
    comp = pct(totals["broad-full"], totals["broad-narrative"])
    both = pct(totals["broad-full"], totals["broad-ctrl"])
    print(f"  volume     (narrative vs ctrl)  {vol:>+7.2f}%  {verdict(vol)}")
    print(f"  composition(full vs narrative)  {comp:>+7.2f}%  {verdict(comp)}")
    print(f"  both       (full vs ctrl)       {both:>+7.2f}%  {verdict(both)}")
    out["decomposition"] = {"volume_pct": vol, "composition_pct": comp, "combined_pct": both}

    # ---- does broadening close the gap it was aimed at? ------------------
    # The claim under test was that the per-form penalties shrink. Measure the
    # penalty itself -- each form against narrative prose, within an arm --
    # rather than the raw score, which moves for every form at once.
    print("\npenalty against narrative prose, within each arm")
    print(f"  {'form':<13}" + "".join(f"{a.split('-')[1]:>12}" for a in ARMS))
    penalties: dict[str, dict] = {}
    for form in forms:
        if form == "narrative":
            continue
        row = {}
        for arm in ARMS:
            row[arm] = pct(by_form[form][arm], by_form["narrative"][arm])
        print(f"  {form:<13}" + "".join(f"{row[a]:>+11.1f}%" for a in ARMS))
        penalties[form] = row
    out["penalty_vs_narrative"] = penalties

    closed = {f: penalties[f]["broad-full"] - penalties[f]["broad-ctrl"] for f in penalties}
    print("\n  change in penalty, full vs ctrl (negative = the gap closed)")
    for form, delta in sorted(closed.items(), key=lambda kv: kv[1]):
        print(f"    {form:<13}{delta:>+8.1f} percentage points")
    out["penalty_change_pp"] = closed

    # ---- the stated falsifier -------------------------------------------
    hard = [f for f in penalties if f != "narrative"]
    survived = [f for f in hard if closed[f] >= 0]
    out["falsifier"] = {
        "statement": ("if the verse and non-fiction penalties survive this increase "
                      "in training mass, the forms are intrinsically harder at 1.77M "
                      "parameters and 256 tokens of context"),
        "forms_whose_penalty_survived": survived,
        "forms_whose_penalty_closed": [f for f in hard if closed[f] < 0],
    }
    print("\nfalsifier")
    if survived:
        print(f"  penalty survived for: {', '.join(survived)}")
    if len(survived) < len(hard):
        print(f"  penalty closed for:   {', '.join(f for f in hard if closed[f] < 0)}")

    (HERE / "broad-summary.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"\nwritten to {HERE / 'broad-summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

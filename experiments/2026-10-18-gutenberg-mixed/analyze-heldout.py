#!/usr/bin/env python3
"""Does broad training transfer to books neither model has seen?

Every corpus-composition number so far was measured on the original 71
books -- the same books both arms trained on. That leaves the most
interesting version of the question unanswered: broad-full got better at
verse it trained on (the Iliad, Leaves of Grass), but is it better at
verse *in general*?

The obvious probe is contaminated. In the broad corpus the 80 additions
are train_only, and a train_only document contributes its entire text to
training, so their would-be test pieces were trained on. Scoring there
would measure fit. `gutenberg-heldout-forms` is the clean version: 40
titles, ten per form, verified disjoint from every training corpus
(tests/test_heldout_corpus.py keeps them that way).

Both checkpoints of the capacity x horizon pair were scored per-document
on its test split (71,397 tokens). Narrative prose is the within-probe
control: both arms trained on the same 71 books' narrative mass, so any
narrative delta here is the volume/capacity competition measured on
unseen books, against which the drama/verse/non-fiction deltas are the
transfer of having trained on those forms at all.

Form labels come from the search keyword that found each repository --
weaker than the hand labels used for the original 71, and reported as such.

Usage (from the repository root):
    python experiments/2026-10-18-gutenberg-mixed/analyze-heldout.py
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
RUNS = ROOT / "scripts" / "tiny-gpt" / "artifacts" / "runs"

ARMS = {"ctrl": "broad-4L256d-ctrl-long", "full": "broad-4L256d-full-long"}
SEED_BAR_PCT = 0.3690475821746498


def load(run: str) -> tuple[float, list[dict]]:
    path = RUNS / run / "heldout" / "eval-test.json"
    if not path.exists():
        print(f"missing {path}; run scripts/tiny-gpt/run_queue2.sh first", file=sys.stderr)
        sys.exit(1)
    blob = json.loads(path.read_text())
    return blob["model"]["bpb"], [r for r in blob["per_document"] if r["scored"]]


def pct(new: float, old: float) -> float:
    return 100.0 * (new - old) / old


def verdict(delta: float) -> str:
    if abs(delta) < SEED_BAR_PCT:
        return "inside the seed bar"
    return "better" if delta < 0 else "worse"


def main() -> int:
    selection = {f"pg{s['pg_id']}": s["form"]
                 for s in json.loads((HERE / "heldout-selection.json").read_text())}
    data = {arm: load(run) for arm, run in ARMS.items()}

    docs = [r["doc_id"] for r in data["ctrl"][1]]
    if docs != [r["doc_id"] for r in data["full"][1]]:
        print("arms scored different document sets", file=sys.stderr)
        return 1
    missing = [d for d in docs if d not in selection]
    if missing:
        print(f"no form label for {missing}", file=sys.stderr)
        return 1
    print(f"{len(docs)} unseen books scored identically in both arms\n")

    out: dict = {"arms": {a: data[a][0] for a in ARMS}, "by_form": {}}
    d_all = pct(data["full"][0], data["ctrl"][0])
    print(f"aggregate held-out test bpb   ctrl {data['ctrl'][0]:.4f}   "
          f"full {data['full'][0]:.4f}   {d_all:+.2f}%  {verdict(d_all)}")
    out["aggregate_delta_pct"] = d_all

    print(f"\n{'form':<13}{'n':>3}{'ctrl':>10}{'full':>10}{'delta':>10}")
    forms = sorted({selection[d] for d in docs}, key=lambda f: (f != "narrative", f))
    for form in forms:
        idx = [i for i, d in enumerate(docs) if selection[d] == form]
        means = {a: statistics.fmean(data[a][1][i]["bpb"] for i in idx) for a in ARMS}
        d = pct(means["full"], means["ctrl"])
        print(f"{form:<13}{len(idx):>3}{means['ctrl']:>10.4f}{means['full']:>10.4f}"
              f"{d:>+9.2f}%  {verdict(d)}")
        out["by_form"][form] = {"n": len(idx), **means, "delta_pct": d}

    targeted = [f for f in forms if f != "narrative"]
    transfer = {f: out["by_form"][f]["delta_pct"] - out["by_form"]["narrative"]["delta_pct"]
                for f in targeted}
    print("\ntransfer beyond the narrative control (negative = the trained form gained more):")
    for f, v in sorted(transfer.items(), key=lambda kv: kv[1]):
        print(f"  {f:<13}{v:>+8.2f} percentage points")
    out["transfer_vs_narrative_pp"] = transfer

    (HERE / "heldout-summary.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"\nwritten to {HERE / 'heldout-summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

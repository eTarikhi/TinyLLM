#!/usr/bin/env python3
"""Recompute every number the reference files assert, and fail if one drifts.

Why this exists: `references/compute-budget-example.md` shipped an attention
FLOP term of `12·L·T·D` for two months. The correct term is `12·L·T·d_model·D`.
Nothing caught it -- the frontmatter validator only reads YAML, the package
checker only checks that files and links exist, and the markdown table looked
entirely plausible. The error made attention look like 0.026% of training FLOPs
at a 512-token context when it is 13.5%, and it would have propagated into any
budget an agent quoted from the file.

`references/training-defaults.md` shipped a parameter count for nanoGPT's
character-level config that assumed untied embeddings. nanoGPT ties them.

Both classes of defect are arithmetic, so arithmetic can catch them. Each check
below recomputes a value from first principles and then greps the document for
the number it publishes. A missing match is also a failure: if the wording
changed, the claim needs re-verifying, not silently skipping.

Stdlib only. Usage:  python3 .github/scripts/check_docs_math.py [repo_root]
"""

import os
import re
import sys

failures: list[str] = []
checked = 0


def claim(doc_text: str, pattern: str, expected: float, label: str, tol: float = 0.0, doc: str = "") -> None:
    """Assert the document publishes `expected` where `pattern` matches."""
    global checked
    checked += 1
    m = re.search(pattern, doc_text)
    if not m:
        failures.append(f"{doc}: claim not found for {label!r} (pattern {pattern!r}). "
                        "If the wording changed, re-verify the number and update this check.")
        return
    published = float(m.group(1).replace(",", "").rstrip("."))
    if tol:
        ok = abs(published - expected) <= tol
    else:
        ok = published == expected
    if ok:
        print(f"  ok   {label}: document says {m.group(1)}, computed {expected:,.6g}")
    else:
        failures.append(f"{doc}: {label}: document says {m.group(1)}, computed {expected:,.6g}")


def check_compute_budget(root: str) -> None:
    path = os.path.join(root, "references", "compute-budget-example.md")
    doc = open(path, encoding="utf-8").read()
    print("references/compute-budget-example.md")

    # Config stated in step 1.
    V, d, L, T, D = 8192, 512, 6, 512, 2_000_000_000
    attn_per_layer = 4 * d * d          # q, k, v, out projections
    mlp_per_layer = 8 * d * d           # 4x expansion, two matrices
    body = L * (attn_per_layer + mlp_per_layer)
    tied = V * d + T * d + body
    untied = tied + V * d

    claim(doc, r"\*\*Total, tied embeddings\*\* \| \| \*\*([\d,]+)", tied, "tied parameter count", doc="compute-budget")
    claim(doc, r"\*\*Total, untied embeddings\*\*.*?\*\*([\d,]+)", untied, "untied parameter count", doc="compute-budget")
    claim(doc, r"Token embedding \| `vocab × d` \| ([\d,]+)", V * d, "token embedding", doc="compute-budget")
    claim(doc, r"Position embedding \| `block × d` \| ([\d,]+)", T * d, "position embedding", doc="compute-budget")
    claim(doc, r"All 6 layers \| `6 × \(4d² \+ 8d²\)` \| ([\d,]+)", body, "all layers", doc="compute-budget")
    claim(doc, r"removes \*\*([\d.]+)%\*\* of the parameters", 100 * (V * d) / untied,
          "tying saving (% of untied)", tol=0.05, doc="compute-budget")

    # 13 LayerNorms (2 per block + 1 final), each with a weight and a bias of
    # width d. Whether the norms carry biases changes this number by 2x, which
    # is why the document states the composition rather than just the total.
    norms = (2 * L + 1) * 2 * d
    claim(doc, r"= ([\d,]+)` parameters", norms, "excluded norm parameters", doc="compute-budget")
    claim(doc, r"parameters, or \*\*([\d.]+)%\*\* of the total", 100 * norms / tied,
          "excluded norm share", tol=0.001, doc="compute-budget")
    claim(doc, r"halves that to `([\d,]+)`", norms // 2, "norm parameters without biases", doc="compute-budget")

    # Step 3: 6ND and the attention term.
    claim(doc, r"6 × 23,330,816 × 2e9 = ([\d.]+)e17", 6 * tied * D / 1e17, "6ND tied", tol=0.0005, doc="compute-budget")
    claim(doc, r"6 × 27,525,120 × 2e9 = ([\d.]+)e17", 6 * untied * D / 1e17, "6ND untied", tol=0.0005, doc="compute-budget")

    for context, exponent in ((512, 16), (2048, 17), (8192, 17), (32768, 18)):
        attention = 12 * L * context * d * D
        share = 100 * attention / (6 * tied * D)
        row = rf"\| {context:,} \| ([\d.]+)e{exponent} \| [\d.]+% \|"
        claim(doc, row, attention / 10 ** exponent, f"attention FLOPs at T={context}", tol=0.006, doc="compute-budget")
        share_row = rf"\| {context:,} \| [\d.]+e{exponent} \| ([\d.]+)% \|"
        claim(doc, share_row, share, f"attention share at T={context}", tol=0.06, doc="compute-budget")

    # Step 4: wall clock.
    for tps, exp_hours in ((12_000, D / 12_000 / 3600), (25_000, D / 25_000 / 3600)):
        if tps == 12_000:
            claim(doc, r"= 166,667 s = ([\d.]+) h", exp_hours, "wall clock at 12k tok/s", tol=0.05, doc="compute-budget")
            claim(doc, r"h = ([\d.]+) days", exp_hours / 24, "wall clock in days", tol=0.01, doc="compute-budget")
        else:
            claim(doc, r"the same budget is ([\d.]+) h", exp_hours, "wall clock at 25k tok/s", tol=0.05, doc="compute-budget")

    # Step 5: achieved FLOP/s and MFU, with the attention term included.
    seconds = D / 12_000
    total_flops = (6 * tied + 12 * L * T * d) * D
    achieved = total_flops / seconds
    claim(doc, r"achieved = 3\.1772e17 / 166,667 = ([\d.]+) TFLOP/s", achieved / 1e12, "achieved TFLOP/s", tol=0.005, doc="compute-budget")
    claim(doc, r"MFU = [\d.]+ / 200 = ([\d.]+)%", 100 * achieved / 200e12, "MFU vs 200 TFLOP/s", tol=0.005, doc="compute-budget")
    claim(doc, r"/ 312 = ([\d.]+)%", 100 * achieved / 312e12, "MFU vs 312 TFLOP/s", tol=0.005, doc="compute-budget")

    # Step 6: cost table and contingency.
    hours = seconds / 3600
    for rate in (0.35, 1.00, 2.00, 4.00):
        claim(doc, rf"\| \${rate:.2f}/h \| \$([\d.]+) \|", rate * hours, f"cost at ${rate:.2f}/h", tol=0.02, doc="compute-budget")
    claim(doc, r"\+25% on 46\.3 h is ([\d.]+) h", hours * 1.25, "contingency hours", tol=0.1, doc="compute-budget")
    claim(doc, r"burn roughly ([\d]+) h", hours * 3, "three failed runs", tol=0.6, doc="compute-budget")

    # Step 7: tokens per update.
    for micro, accum in ((8, 1), (8, 8), (32, 4)):
        per_update = micro * T * accum
        updates = D // per_update
        claim(doc, rf"\| {micro} \| {accum} \| ([\d,]+) \|", per_update, f"tokens/update {micro}x{accum}", doc="compute-budget")
        claim(doc, rf"\| {micro} \| {accum} \| [\d,]+ \| ([\d,]+) \|", updates, f"updates {micro}x{accum}", tol=1, doc="compute-budget")


def check_training_defaults(root: str) -> None:
    path = os.path.join(root, "references", "training-defaults.md")
    doc = open(path, encoding="utf-8").read()
    print("references/training-defaults.md")

    # Anchor A: nanoGPT config/train_shakespeare_char.py on top of train.py's
    # defaults (bias=False), with tied input/output embeddings (model.py).
    V, d, L, T = 65, 384, 6, 256
    per_block = 3 * d * d + d * d + 4 * d * d + 4 * d * d + 2 * d   # attn qkv+proj, mlp fc+proj, 2 LayerNorm weights
    total = V * d + T * d + L * per_block + d                        # + final LayerNorm
    non_embedding = total - T * d

    claim(doc, r"\*\*([\d,]+) parameters in total", total, "nanoGPT char total params", doc="training-defaults")
    claim(doc, r"Non-embedding, as nanoGPT prints it \| ([\d,]+)", non_embedding, "nanoGPT char non-embedding", doc="training-defaults")
    claim(doc, r"Token embedding `65 × 384` \(tied with the output head\) \| ([\d,]+)", V * d, "token embedding", doc="training-defaults")
    claim(doc, r"Position embedding `256 × 384` \| ([\d,]+)", T * d, "position embedding", doc="training-defaults")
    claim(doc, r"two LayerNorm weights\) \| ([\d,]+)", L * per_block, "six blocks", doc="training-defaults")

    # Tokens per update at both anchors.
    claim(doc, r"Tokens per update \| `([\d,]+)` \(2\^14\)", 64 * 256 * 1, "anchor A tokens/update", doc="training-defaults")
    claim(doc, r"Tokens per update \| `([\d,]+)` \| `12 × 1024 × 40`", 12 * 1024 * 40, "anchor B tokens/update", doc="training-defaults")
    claim(doc, r"nanoGPT's 124M config computes to ([\d,]+) tokens per update", 12 * 1024 * 40,
          "anchor B restated", doc="training-defaults")
    claim(doc, r"empirically measured optimum is ([\d,]+)", 2 ** 19, "nanochat reference batch", doc="training-defaults")
    claim(doc, r"a ratio of ([\d.]+)", (12 * 1024 * 40) / 2 ** 19, "ratio of the two batch sizes", tol=0.005, doc="training-defaults")
    claim(doc, r"Reference batch \| `2\^19 = ([\d,]+)` tokens", 2 ** 19, "2^19", doc="training-defaults")

    # The cross-optimizer warning quotes a ratio; check it too.
    claim(doc, r"embedding LR of `0\.3` is roughly ([\d]+)× nanoGPT's peak AdamW LR", 0.3 / 6e-4,
          "Muon/AdamW LR ratio", tol=20, doc="training-defaults")


def check_experiment_reports(root: str) -> None:
    """Tie the committed write-ups back to the raw run artifacts.

    A report is a transcription of JSON a human did by hand, which is exactly
    where a digit gets dropped. Every headline number in the iteration-2
    reports and model card is recomputed here from the committed evaluation
    JSON, compressor baselines and run summaries, so the prose cannot drift
    away from the measurements it describes.

    Two runs are covered: `main-gutenberg` (2,400 steps) and the doubled
    horizon `main-gutenberg-4800`, which is the released model and therefore
    the one the model card must agree with.
    """
    import json

    exp = os.path.join(root, "experiments", "2026-10-18-gutenberg-mixed")
    if not os.path.isdir(exp):
        return

    def load(name):
        return json.load(open(os.path.join(exp, name), encoding="utf-8"))

    def text(name):
        return open(os.path.join(exp, name), encoding="utf-8").read()

    TRAIN_TOKENS = 5_771_560  # train split, from data-manifest.json

    runs = [
        {
            "label": "main-gutenberg",
            "doc": "eval-report.md",
            "eval": load("eval-test-best.json"),
            "comp": load("compressor-baselines.json"),
            "summ": load(os.path.join("metrics", "main-gutenberg-summary.json")),
            "bpb_row": r"\*\*Trained model\*\* \| \*\*3\.4378\*\* \| \*\*31\.12\*\* \| \*\*([\d.]+)\*\*",
            "steps": 2400,
        },
        {
            "label": "main-gutenberg-4800",
            "doc": "eval-report-4800.md",
            "eval": load("eval-test-4800.json"),
            "comp": load("compressor-baselines-4800.json"),
            "summ": load(os.path.join("metrics", "main-gutenberg-4800-summary.json")),
            "bpb_row": r"\*\*Trained model\*\* \| \*\*3\.3116\*\* \| \*\*27\.43\*\* \| \*\*([\d.]+)\*\*",
            "steps": 4800,
        },
    ]

    for run in runs:
        print(f"experiments/2026-10-18-gutenberg-mixed/{run['doc']} — report vs. raw artifacts")
        doc, ev, comp, summ = text(run["doc"]), run["eval"], run["comp"], run["summ"]

        # Held-out table: every row comes from the evaluation JSON.
        claim(doc, run["bpb_row"], round(ev["model"]["bpb"], 4), "test bpb, model",
              tol=5e-5, doc=run["doc"])
        for key, pattern in (
            ("bigram", r"Bigram counts, add-0\.1 \(train split\) \| 4\.3877 \| 80\.46 \| ([\d.]+)"),
            ("unigram", r"Unigram counts \(train split\) \| 6\.2828 \| 535\.27 \| ([\d.]+)"),
            ("uniform", r"Uniform over vocabulary \| 7\.6246 \| 2048\.00 \| ([\d.]+)"),
            ("untrained", r"random init \| 7\.6486 \| 2097\.66 \| ([\d.]+)"),
        ):
            claim(doc, pattern, round(ev[key]["bpb"], 4), f"test bpb, {key}",
                  tol=5e-5, doc=run["doc"])

        # Ratios the prose draws from that table.
        claim(doc, r"vs\. bigram counts \| 1\.28× \| ([\d.]+)×" if run["steps"] == 4800
              else r"\*\*([\d.]+)×\*\* better than the bigram baseline",
              round(ev["bigram"]["bpb"] / ev["model"]["bpb"], 2), "model vs bigram ratio",
              tol=5e-3, doc=run["doc"])

        # Compressor baselines. Both runs are scored on the same bytes, so the
        # two files must agree -- that is the check that they really are
        # comparable, not an assumption.
        by_codec = {c["codec"]: c for c in comp["codecs"]}
        base_comp = {c["codec"]: c for c in runs[0]["comp"]["codecs"]}
        for codec in ("gzip -9", "xz -9", "bzip2 -9"):
            global checked
            checked += 1
            if abs(by_codec[codec]["bpb_standalone"] - base_comp[codec]["bpb_standalone"]) > 5e-9:
                failures.append(f"{run['doc']}: {codec} baseline differs between runs "
                                f"({by_codec[codec]['bpb_standalone']} vs "
                                f"{base_comp[codec]['bpb_standalone']}) -- the two models were "
                                "not scored on the same bytes")
        best_primed = min(c["bpb_primed"] for c in comp["codecs"])
        claim(doc, r"vs\. best primed compressor \| 1\.42× \| \*\*([\d.]+)×\*\*" if run["steps"] == 4800
              else r"best classical compressor by \*\*([\d.]+)×\*\*",
              round(best_primed / ev["model"]["bpb"], 2), "model vs best primed compressor",
              tol=5e-3, doc=run["doc"])

        # Run facts.
        epochs = round(summ["tokens"] / TRAIN_TOKENS, 2)
        claim(doc, rf"([\d,]+) tokens \({epochs} epochs\)", summ["tokens"],
              "tokens trained", doc=run["doc"])
        claim(doc, r"\(([\d.]+) epochs\)", epochs, "epochs", tol=5e-3, doc=run["doc"])

    # The model card describes the released model: the 4,800-step run.
    print("experiments/2026-10-18-gutenberg-mixed/model-card.md — card vs. raw artifacts")
    card = text("model-card.md")
    ev4, summ4 = runs[1]["eval"], runs[1]["summ"]
    ev24 = runs[0]["eval"]
    epochs4 = round(summ4["tokens"] / TRAIN_TOKENS, 2)
    claim(card, rf"([\d,]+) \({epochs4} epochs\)", summ4["tokens"], "tokens trained", doc="model-card.md")
    claim(card, r"\(([\d.]+) epochs\)", epochs4, "epochs", tol=5e-3, doc="model-card.md")
    claim(card, r"\*\*([\d,]+)\*\* total", summ4["params_total"], "parameter count", doc="model-card.md")
    claim(card, r"at ([\d,]+) tokens/s", round(summ4["mean_tokens_per_s"]), "mean throughput",
          tol=1, doc="model-card.md")
    claim(card, r"engaged on (\d+) steps", summ4["clipped_steps"], "clipped steps", doc="model-card.md")
    claim(card, r"engaged on \d+ steps, (\d\.\d)%",
          round(100 * summ4["clipped_steps"] / summ4["steps"], 1), "clipped share",
          tol=5e-2, doc="model-card.md")
    claim(card, r"\*\*This model\*\* \| \*\*([\d.]+)\*\*", round(ev4["model"]["bpb"], 4),
          "released model test bpb", tol=5e-5, doc="model-card.md")
    claim(card, r"Validation bits per byte ([\d.]+)", round(summ4["final_val_bpb"], 4),
          "final val bpb", tol=5e-5, doc="model-card.md")
    claim(card, r"predecessor scored ([\d.]+) test bits per byte", round(ev24["model"]["bpb"], 4),
          "predecessor test bpb", tol=5e-5, doc="model-card.md")
    claim(card, r"bought ([\d.]+)%", round(100 * (1 - ev4["model"]["bpb"] / ev24["model"]["bpb"]), 1),
          "improvement from doubling compute", tol=5e-2, doc="model-card.md")
    claim(card, r"match against the training text \*\*(\d+)\s*\n?characters\*\*",
          ev4["memorization"]["max_longest_verbatim_match_chars"], "probe chars", doc="model-card.md")


def main(root: str = ".") -> int:
    check_compute_budget(root)
    check_training_defaults(root)
    check_experiment_reports(root)
    print()
    if failures:
        print(f"{len(failures)} of {checked} arithmetic check(s) failed:")
        for f in failures:
            print("  FAIL: " + f)
        return 1
    print(f"All {checked} documented numbers recomputed and matched.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))

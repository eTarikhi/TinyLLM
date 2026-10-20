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

    # The size sweep: three runs at a matched wall-clock budget. Each row of
    # its results table is recomputed from that run's own summary and
    # evaluation JSON, and the two relative margins from the pair they
    # compare -- those margins are the whole point of the sweep.
    sweep_doc = "size-sweep.md"
    if os.path.exists(os.path.join(exp, sweep_doc)):
        print(f"experiments/2026-10-18-gutenberg-mixed/{sweep_doc} — sweep vs. raw artifacts")
        sweep = text(sweep_doc)

        # Results-table row: | label | params | steps | elapsed s | tokens | epochs | val | test |
        FIELDS = [r"\*{0,2}([\d,]+)\*{0,2}", r"([\d,]+)", r"([\d,]+) s", r"([\d,]+)",
                  r"([\d.]+)", r"\*{0,2}([\d.]+)\*{0,2}", r"\*{0,2}([\d.]+)\*{0,2}"]

        def row_pattern(label, index):
            """The table row, with a capture group on field `index` only."""
            cells = [f if i == index else f.replace("(", "(?:") for i, f in enumerate(FIELDS)]
            return r"\| \*{0,2}" + re.escape(label) + r"\*{0,2} \| " + r" \| ".join(cells) + r" \|"

        points = [
            ("3L/192d/3H", "main-gutenberg-4800", "eval-test-4800.json"),
            ("4L/256d/4H", "size-4L256d", "eval-test-size-4L256d.json"),
            ("5L/288d/6H", "size-5L288d", "eval-test-size-5L288d.json"),
        ]
        loaded = []
        for label, run, evf in points:
            su, evj = load(os.path.join("metrics", f"{run}-summary.json")), load(evf)
            loaded.append((label, su, evj))
            for index, value, what in ((0, su["params_total"], "params"),
                                       (1, su["steps"], "steps"),
                                       (3, su["tokens"], "tokens")):
                claim(sweep, row_pattern(label, index), value, f"{what}, {label}", doc=sweep_doc)
            claim(sweep, row_pattern(label, 5), round(su["final_val_bpb"], 4),
                  f"val bpb, {label}", tol=5e-5, doc=sweep_doc)
            claim(sweep, row_pattern(label, 6), round(evj["model"]["bpb"], 4),
                  f"test bpb, {label}", tol=5e-5, doc=sweep_doc)

        # The margins, recomputed from the pair each one compares.
        base_su, base_ev = loaded[0][1], loaded[0][2]
        for label, su, evj in loaded[1:]:
            short = label.rsplit("/", 1)[0]  # the prose drops the head count
            claim(sweep, re.escape(short) + r" is \*\*\+([\d.]+)%\*\* test bpb",
                  round((evj["model"]["bpb"] / base_ev["model"]["bpb"] - 1) * 100, 2),
                  f"test margin vs incumbent, {short}", tol=5e-3, doc=sweep_doc)
            claim(sweep, re.escape(short) + r" is \*\*\+[\d.]+%\*\* test bpb at ([\d.]+)×",
                  round(su["params_total"] / base_su["params_total"], 2),
                  f"param ratio, {short}", tol=5e-3, doc=sweep_doc)
        claim(sweep, r"\*\*([\d.]+)%\*\* validation, [\d.]+% test",
              round((loaded[1][1]["final_val_bpb"] / base_su["final_val_bpb"] - 1) * 100, 2),
              "val margin, 1.77M over 3.74M", tol=5e-3, doc=sweep_doc)

    # The context sweep, checked the same way as the size sweep -- now four
    # points, because T=128 turned the monotone trend into an interior
    # optimum and the document's headline changed with it.
    def jsonl(run):
        path = os.path.join(exp, "metrics", f"{run}.jsonl")
        return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]

    def last_eval(run):
        return [r for r in jsonl(run) if r["event"] == "eval"][-1]

    def bpb_at_tokens(run, target):
        """Linear interpolation of val bpb against tokens, as the report does."""
        pts = [(r["tokens"], r["val_bpb"]) for r in jsonl(run) if r["event"] == "eval"]
        for (t0, b0), (t1, b1) in zip(pts, pts[1:]):
            if t0 <= target <= t1:
                return b1 if t1 == t0 else b0 + (b1 - b0) * (target - t0) / (t1 - t0)
        return pts[-1][1]

    def clean_rate(run):
        """Median throughput over train windows that contain no evaluation."""
        rs = jsonl(run)
        eval_steps = {r["step"] for r in rs if r["event"] == "eval"}
        tr = [r for r in rs if r["event"] == "train"]
        rates = []
        for a, b in zip(tr, tr[1:]):
            if any(a["step"] < st <= b["step"] for st in eval_steps):
                continue
            dt = b["elapsed_s"] - a["elapsed_s"]
            if dt > 0:
                rates.append((b["tokens"] - a["tokens"]) / dt)
        rates.sort()
        n = len(rates)
        return rates[n // 2] if n % 2 else (rates[n // 2 - 1] + rates[n // 2]) / 2

    def quarter_rates(run):
        tr = [r for r in jsonl(run) if r["event"] == "train"]
        n = len(tr)
        out = []
        for q in range(4):
            chunk = tr[q * n // 4:(q + 1) * n // 4]
            dt = chunk[-1]["elapsed_s"] - chunk[0]["elapsed_s"]
            out.append((chunk[-1]["tokens"] - chunk[0]["tokens"]) / dt)
        return out

    ctx_doc = "context-sweep.md"
    if os.path.exists(os.path.join(exp, ctx_doc)):
        print(f"experiments/2026-10-18-gutenberg-mixed/{ctx_doc} — sweep vs. raw artifacts")
        ctx = text(ctx_doc)
        CFIELDS = [r"\*{0,2}([\d,]+)\*{0,2}", r"([\d,]+)", r"([\d,]+) s", r"([\d,]+)",
                   r"\*{0,2}([\d.]+)\*{0,2}", r"\*{0,2}([\d.]+)\*{0,2}"]

        def ctx_row(label, index):
            cells = [f if i == index else f.replace("(", "(?:") for i, f in enumerate(CFIELDS)]
            return r"\| \*{0,2}" + re.escape(label) + r"\*{0,2} \| " + r" \| ".join(cells) + r" \|"

        cpoints = [("T=128", "ctx-128", "eval-test-ctx-128.json"),
                   ("T=256", "main-gutenberg-4800", "eval-test-4800.json"),
                   ("T=512", "ctx-512", "eval-test-ctx-512.json"),
                   ("T=1024", "ctx-1024", "eval-test-ctx-1024.json")]
        cloaded = {}
        for label, run, evf in cpoints:
            su, evj = load(os.path.join("metrics", f"{run}-summary.json")), load(evf)
            cloaded[label] = (run, su, evj)
            for index, value, what in ((0, su["params_total"], "params"),
                                       (1, su["steps"], "steps"),
                                       (2, round(su["elapsed_s"]), "elapsed"),
                                       (3, su["tokens"], "tokens")):
                claim(ctx, ctx_row(label, index), value, f"{what}, {label}", doc=ctx_doc)
            claim(ctx, ctx_row(label, 4), round(su["final_val_bpb"], 4),
                  f"val bpb, {label}", tol=5e-5, doc=ctx_doc)
            claim(ctx, ctx_row(label, 5), round(evj["model"]["bpb"], 4),
                  f"test bpb, {label}", tol=5e-5, doc=ctx_doc)

        cbase = cloaded["T=256"][2]
        for label in ("T=128", "T=512", "T=1024"):
            evj = cloaded[label][2]
            claim(ctx, re.escape(label) + r" is \*\*\+([\d.]+)%\*\*",
                  round((evj["model"]["bpb"] / cbase["model"]["bpb"] - 1) * 100, 2),
                  f"test margin vs T=256, {label}", tol=5e-3, doc=ctx_doc)

        # Token-matched table: interpolate every run at the smallest run's
        # token count, which is the whole basis of the "is it just the lost
        # tokens?" argument.
        floor = min(su["tokens"] for _, su, _ in cloaded.values())
        claim(ctx, r"val bpb at ([\d,]+) tokens", floor, "token-matched floor", doc=ctx_doc)
        TFIELDS = [r"([\d.]+)", r"(?:—|\+[\d.]+%)", r"([\d.]+)%"]
        for label in ("T=128", "T=256", "T=512", "T=1024"):
            run, su, _ = cloaded[label]
            matched = bpb_at_tokens(run, floor)
            claim(ctx, r"\| " + re.escape(label) + r" \| ([\d.]+) \| (?:—|\+[\d.]+%) \| (?:[\d.]+)% \|",
                  round(matched, 4), f"token-matched val bpb, {label}", tol=5e-5, doc=ctx_doc)
            claim(ctx, r"\| " + re.escape(label) + r" \| (?:[\d.]+) \| (?:—|\+[\d.]+%) \| ([\d.]+)% \|",
                  round(100 * floor / su["tokens"], 1), f"cosine fraction, {label}",
                  tol=5e-2, doc=ctx_doc)

        # The generalization-gap ordering, read off the final evaluation.
        for label in ("T=128", "T=256", "T=512", "T=1024"):
            run = cloaded[label][0]
            e = last_eval(run)
            claim(ctx, r"\+([\d.]+) at " + re.escape(label),
                  round(e["train_eval_loss"] - e["val_loss"], 4), f"end gap, {label}",
                  tol=5e-5, doc=ctx_doc)

        # Throughput: the probe predicted a short context would be cheaper and
        # it is not, which is the only sign error in the whole directory.
        same_host = [clean_rate(r) for r in ("lr-1p5e3", "lr-6e3")]
        same_host_t256 = sum(same_host) / len(same_host)
        ref_t256 = clean_rate("main-gutenberg-4800")
        for label in ("T=128", "T=256", "T=512", "T=1024"):
            run = cloaded[label][0]
            rate = clean_rate(run)
            claim(ctx, r"\| " + re.escape(label) + r" \| (?:[\d,]+) \| ([\d,]+) \|",
                  round(rate), f"measured tok/s, {label}", tol=1, doc=ctx_doc)
            if label == "T=256":
                continue
            ref = same_host_t256 if label == "T=128" else ref_t256
            claim(ctx, r"\| " + re.escape(label) + r" \| (?:[\d,]+) \| (?:[\d,]+) \| \*{0,2}−([\d.]+)%",
                  round(abs(100 * (rate - ref) / ref), 1), f"throughput vs T=256, {label}",
                  tol=5e-2, doc=ctx_doc)
        claim(ctx, r"same-host T=256 figure of ([\d,]+) tok/s", round(same_host_t256),
              "same-host T=256 throughput", tol=1, doc=ctx_doc)

        # Host-speed control.
        ref_mean = load(os.path.join("metrics", "main-gutenberg-4800-summary.json"))["mean_tokens_per_s"]
        today = [load(os.path.join("metrics", f"{r}-summary.json"))["mean_tokens_per_s"]
                 for r in ("lr-1p5e3", "lr-6e3")]
        today_mean = sum(today) / len(today)
        claim(ctx, r"averaged ([\d,]+) tok/s end-to-end", round(today_mean),
              "same-shape throughput today", tol=1, doc=ctx_doc)
        claim(ctx, r"\*\*([\d.]+)% slower\*\*", round(abs(100 * (today_mean - ref_mean) / ref_mean), 1),
              "host slowdown", tol=5e-2, doc=ctx_doc)

        # Per-quarter throughput, including the T=1024 collapse the table exists for.
        for label in ("T=128", "T=256", "T=512", "T=1024"):
            run = cloaded[label][0]
            q = quarter_rates(run)
            for i in range(4):
                cells = [r"(?:[\d,]+)"] * 4
                cells[i] = r"([\d,]+)"
                claim(ctx, r"\| " + re.escape(label) + r" \| " + r" \| ".join(cells) +
                      r" \| \*{0,2}[+−][\d.]+%", round(q[i]), f"Q{i + 1} tok/s, {label}",
                      tol=1, doc=ctx_doc)
            claim(ctx, r"\| " + re.escape(label) + r" \| (?:[\d,]+) \| (?:[\d,]+) \| (?:[\d,]+) \| "
                  r"(?:[\d,]+) \| \*{0,2}[+−]([\d.]+)%",
                  round(abs(100 * (q[3] - q[0]) / q[0]), 1), f"quarter drift, {label}",
                  tol=5e-2, doc=ctx_doc)

    # The learning-rate bracket: three runs identical except peak LR, so the
    # comparison is matched on steps and tokens and every margin is exact.
    lr_doc = "lr-bracket.md"
    if os.path.exists(os.path.join(exp, lr_doc)):
        print(f"experiments/2026-10-18-gutenberg-mixed/{lr_doc} — bracket vs. raw artifacts")
        lrt = text(lr_doc)
        LFIELDS = [r"([\d,]+)", r"([\d,]+)", r"\*{0,2}([\d.]+)\*{0,2}",
                   r"\*{0,2}([\d.]+)\*{0,2}", r"([\d,]+)", r"([\d.]+)"]

        def lr_row(label, index):
            cells = [f if i == index else f.replace("(", "(?:") for i, f in enumerate(LFIELDS)]
            return r"\| \*{0,2}" + re.escape(label) + r"\*{0,2} \| " + r" \| ".join(cells) + r" \|"

        lpoints = [("1.5e-3", "lr-1p5e3", "eval-test-lr-1p5e3.json"),
                   ("3.0e-3", "main-gutenberg-4800", "eval-test-4800.json"),
                   ("6.0e-3", "lr-6e3", "eval-test-lr-6e3.json")]
        lloaded = {}
        for label, run, evf in lpoints:
            su, evj = load(os.path.join("metrics", f"{run}-summary.json")), load(evf)
            lloaded[label] = (run, su, evj)
            gnorms = sorted(r["grad_norm_preclip"] for r in jsonl(run) if r["event"] == "train")
            n = len(gnorms)
            median = gnorms[n // 2] if n % 2 else (gnorms[n // 2 - 1] + gnorms[n // 2]) / 2
            for index, value, what in ((0, su["steps"], "steps"),
                                       (1, su["tokens"], "tokens"),
                                       (4, su["clipped_steps"], "clipped steps")):
                claim(lrt, lr_row(label, index), value, f"{what}, lr {label}", doc=lr_doc)
            claim(lrt, lr_row(label, 2), round(su["final_val_bpb"], 4),
                  f"val bpb, lr {label}", tol=5e-5, doc=lr_doc)
            claim(lrt, lr_row(label, 3), round(evj["model"]["bpb"], 4),
                  f"test bpb, lr {label}", tol=5e-5, doc=lr_doc)
            claim(lrt, lr_row(label, 5), round(median, 3),
                  f"median grad norm, lr {label}", tol=5e-4, doc=lr_doc)

        lbase_su, lbase_ev = lloaded["3.0e-3"][1], lloaded["3.0e-3"][2]
        for label in ("1.5e-3", "6.0e-3"):
            _, su, evj = lloaded[label]
            claim(lrt, re.escape(label) + r" is \*\*\+([\d.]+)%\*\* test bpb",
                  round((evj["model"]["bpb"] / lbase_ev["model"]["bpb"] - 1) * 100, 2),
                  f"test margin vs 3e-3, lr {label}", tol=5e-3, doc=lr_doc)
        claim(lrt, r"same ordering is clearer — \+([\d.]+)%",
              round((lloaded["1.5e-3"][1]["final_val_bpb"] / lbase_su["final_val_bpb"] - 1) * 100, 2),
              "val margin, lr 1.5e-3", tol=5e-3, doc=lr_doc)
        claim(lrt, r"same ordering is clearer — \+[\d.]+% and \+([\d.]+)%",
              round((lloaded["6.0e-3"][1]["final_val_bpb"] / lbase_su["final_val_bpb"] - 1) * 100, 2),
              "val margin, lr 6e-3", tol=5e-3, doc=lr_doc)

        low_su = lloaded["1.5e-3"][1]
        claim(lrt, r"\*\*([\d,]+) of 4,800 steps", low_su["clipped_steps"],
              "clipped steps, lr 1.5e-3", doc=lr_doc)
        claim(lrt, r"\*\*[\d,]+ of 4,800 steps — ([\d.]+)%\*\*",
              round(100 * low_su["clipped_steps"] / low_su["steps"], 1),
              "clipped share, lr 1.5e-3", tol=5e-2, doc=lr_doc)

        for label, phrase in (("1.5e-3", r"\+([\d.]+) at 1\.5e-3"),
                              ("3.0e-3", r"\+([\d.]+) at 3e-3"),
                              ("6.0e-3", r"\+([\d.]+) at 6e-3")):
            run = lloaded[label][0]
            e = [r for r in jsonl(run) if r["event"] == "eval"][-1]
            claim(lrt, phrase, round(e["train_eval_loss"] - e["val_loss"], 4),
                  f"end gap, lr {label}", tol=5e-5, doc=lr_doc)

        # Common evaluation grid and the single lead change.
        grids = [{r["step"] for r in jsonl(run) if r["event"] == "eval"} for run, _, _ in lloaded.values()]
        common = sorted(set.intersection(*grids))
        claim(lrt, r"evaluated at (\d+) common steps", len(common), "common eval steps", doc=lr_doc)
        series = {label: {r["step"]: r["val_bpb"] for r in jsonl(run) if r["event"] == "eval"}
                  for label, (run, _, _) in lloaded.items()}
        leaders = [min(series, key=lambda lb: series[lb][st]) for st in common]
        trailing = len(common) - 1 - leaders[1:].index(leaders[-1]) if leaders[-1] in leaders[1:] else 0
        held = sum(1 for lb in leaders[1:] if lb == "3.0e-3")
        claim(lrt, r"remaining (\d+) evaluations", held, "evaluations led by 3e-3", doc=lr_doc)

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


def check_data_seed_report(root: str) -> None:
    """Tie `data-scaling-and-seed-bar.md` back to `data-seed-summary.json`.

    This report carries the seed bar, which every other margin in the directory
    is now judged against, so a transcription slip here would silently re-grade
    all of them. Each number is re-read from the JSON the analysis script emits
    rather than from the prose.
    """
    import json

    exp = os.path.join(root, "experiments", "2026-10-18-gutenberg-mixed")
    summary_path = os.path.join(exp, "data-seed-summary.json")
    doc_path = os.path.join(exp, "data-scaling-and-seed-bar.md")
    if not (os.path.isfile(summary_path) and os.path.isfile(doc_path)):
        return

    s = json.load(open(summary_path, encoding="utf-8"))
    d = open(doc_path, encoding="utf-8").read()
    name = "data-scaling-and-seed-bar.md"
    print("experiments/2026-10-18-gutenberg-mixed/data-scaling-and-seed-bar.md — report vs. raw artifacts")

    # ---- per-seed table. One pattern per column; `.` never crosses a newline.
    rows = {1337: r"\| 1337 \(released\) \|", 1338: r"\| 1338 \|", 1339: r"\| 1339 \|"}
    for rec in s["seeds"]:
        head = rows[rec["seed"]]
        tag = f"seed {rec['seed']}"
        claim(d, head + r" ([\d.]+) \|", round(rec["val_bpb"], 4), f"{tag} val bpb", tol=5e-5, doc=name)
        claim(d, head + r" [\d.]+ \| ([\d.]+) \|", round(rec["test_bpb"], 4), f"{tag} test bpb",
              tol=5e-5, doc=name)
        claim(d, head + r" [\d.]+ \| [\d.]+ \| \+([\d.]+) \|", round(rec["end_gap"], 4),
              f"{tag} gap", tol=5e-5, doc=name)
        claim(d, head + r"(?:[^|\n]*\|){3} (\d+) \|", rec["clipped_steps"], f"{tag} clipped steps", doc=name)
        claim(d, head + r"(?:[^|\n]*\|){4} (\d+) \|", rec["longest_verbatim"], f"{tag} longest verbatim",
              doc=name)
        claim(d, head + r"(?:[^|\n]*\|){5} ([\d,]+) s \|", round(rec["elapsed_s"]), f"{tag} elapsed",
              tol=1, doc=name)

    # ---- spread table.
    for key, head in (("val_spread", r"\| val bpb \|"), ("test_spread", r"\| test bpb \|")):
        sp = s[key]
        lab = key.split("_")[0]
        claim(d, head + r" ([\d.]+) \|", round(sp["mean"], 4), f"{lab} mean", tol=5e-5, doc=name)
        claim(d, head + r" [\d.]+ \| ([\d.]+) \|", round(sp["min"], 4), f"{lab} min", tol=5e-5, doc=name)
        claim(d, head + r" [\d.]+ \| [\d.]+ \| ([\d.]+) \|", round(sp["max"], 4), f"{lab} max",
              tol=5e-5, doc=name)
        claim(d, head + r"(?:[^|\n]*\|){3} ([\d.]+) \(", round(sp["range"], 4), f"{lab} range",
              tol=5e-5, doc=name)
        claim(d, head + r"(?:[^|\n]*\|){3} [\d.]+ \(\*\*([\d.]+)%", round(sp["range_pct"], 2),
              f"{lab} range pct", tol=5e-3, doc=name)
        claim(d, head + r"(?:[^|\n]*\|){4} ([\d.]+) \(", round(sp["stdev"], 4), f"{lab} sd",
              tol=5e-5, doc=name)
        claim(d, head + r"(?:[^|\n]*\|){4} [\d.]+ \(([\d.]+)%", round(sp["stdev_pct"], 2),
              f"{lab} sd pct", tol=5e-3, doc=name)

    # ---- re-grading table. Labels must match the analysis script's own labels.
    heads = {
        "size 3.74M": r"\| size 3\.74M params \| \+([\d.]+)%",
        "size 5.64M": r"\| size 5\.64M params \| \+([\d.]+)%",
        "context T=128": r"\| context T=128 \| \+([\d.]+)%",
        "context T=512": r"\| context T=512 \| \+([\d.]+)%",
        "context T=1024": r"\| context T=1024 \| \+([\d.]+)%",
        "lr 1.5e-3": r"\| peak LR 1\.5e-3 \| \+([\d.]+)%",
        "lr 6.0e-3": r"\| peak LR 6\.0e-3 \| \+([\d.]+)%",
    }
    for v in s["margin_verdicts"]:
        claim(d, heads[v["label"]], round(v["margin_pct"], 2), f"re-graded margin {v['label']}",
              tol=5e-3, doc=name)

    bar = s["test_spread"]["range_pct"]
    claim(d, r"T=128 is only ([\d.]+)× the bar",
          round(s["margin_verdicts"][2]["margin_pct"] / bar, 1), "T=128 in bar units", tol=5e-2, doc=name)
    claim(d, r"1\.5e-3 is ([\d.]+)× the bar", round(s["margin_verdicts"][5]["margin_pct"] / bar, 1),
          "lr 1.5e-3 in bar units", tol=5e-2, doc=name)

    # ---- corpus table.
    small, large = s["data"][0], s["data"][1]
    claim(d, r"\| train tokens \| ([\d,]+) \|", small["train_tokens"], "27-book train tokens", doc=name)
    claim(d, r"\| train tokens \| [\d,]+ \| \*\*([\d,]+)\*\*", large["train_tokens"],
          "71-book train tokens", doc=name)
    claim(d, r"\*\*[\d,]+\*\* \(([\d.]+)×\)", round(large["train_tokens"] / small["train_tokens"], 2),
          "data multiple", tol=5e-3, doc=name)
    claim(d, r"\| epochs over train at 39\.3M tokens \| ([\d.]+) \|", round(small["epochs"], 2),
          "27-book epochs", tol=5e-3, doc=name)
    claim(d, r"\| epochs over train at 39\.3M tokens \| [\d.]+ \| ([\d.]+) \|", round(large["epochs"], 2),
          "71-book epochs", tol=5e-3, doc=name)

    # ---- result table.
    claim(d, r"\| 71 books \| ([\d,]+) \|", large["train_tokens"], "result row train tokens", doc=name)
    claim(d, r"\| 71 books \| [\d,]+ \| ([\d.]+) \|", round(large["epochs"], 2), "result row epochs",
          tol=5e-3, doc=name)
    claim(d, r"\| 71 books \|(?:[^|\n]*\|){2} ([\d.]+) \|", round(large["val_bpb"], 4),
          "71-book val bpb", tol=5e-5, doc=name)
    claim(d, r"\| 71 books \|(?:[^|\n]*\|){3} ([\d.]+) \|", round(large["test_bpb"], 4),
          "71-book test bpb", tol=5e-5, doc=name)
    claim(d, r"\| 71 books \|(?:[^|\n]*\|){4} \+([\d.]+) \|", round(large["end_gap"], 4),
          "71-book gap", tol=5e-5, doc=name)
    claim(d, r"\| 71 books \|(?:[^|\n]*\|){5} (\d+) \|", large["clipped_steps"],
          "71-book clipped steps", doc=name)
    claim(d, r"\| 71 books \|(?:[^|\n]*\|){6} (\d+) \|", large["longest_verbatim"],
          "71-book longest verbatim", doc=name)
    claim(d, r"\| Δ \| [\d.]+× \| \| \*\*\+([\d.]+)%\*\*", round(s["data_delta_val_pct"], 2),
          "data val delta", tol=5e-3, doc=name)
    claim(d, r"\| Δ \|(?:[^|\n]*\|){3} \*\*\+([\d.]+)%\*\*", round(s["data_delta_test_pct"], 2),
          "data test delta", tol=5e-3, doc=name)

    # ---- significance prose.
    claim(d, r"is ([\d.]+)× the seed range", round(s["data_delta_test_pct"] / bar, 1),
          "data effect in bar units", tol=5e-2, doc=name)
    claim(d, r"seed range and\s+\+([\d.]+) seed standard deviations", round(s["data_sigma_test"], 1),
          "data test sigma", tol=5e-2, doc=name)
    claim(d, r"on val is \+([\d.]+) seed standard deviations", round(s["data_sigma_val"], 1),
          "data val sigma", tol=5e-2, doc=name)
    claim(d, r"takes the lead at \*\*step (\d+)\*\*", s["data_lead_changes"][0]["step"],
          "lead change step", doc=name)
    claim(d, r"across all \*\*(\d+)\*\* common evaluation points", s["data_common_eval_steps"],
          "common eval points", doc=name)

    # ---- distribution-shift probe.
    if "shift" in s:
        sh = s["shift"]
        shift_doc = json.load(open(os.path.join(exp, "distribution-shift.json"), encoding="utf-8"))
        res = shift_doc["results"]
        for run in ("seed-1338", "seed-1339"):
            r = res[run]
            claim(d, rf"\| {run} \| no — held out \| ([\d.]+) \|", round(r["test_27_books_bpb"], 4),
                  f"{run} probe test bpb", tol=5e-5, doc=name)
            claim(d, rf"\| {run} \| no — held out \| [\d.]+ \| ([\d.]+) \|",
                  round(r["new_44_books_bpb"], 4), f"{run} probe new bpb", tol=5e-5, doc=name)
            claim(d, rf"\| {run} \| no — held out \|(?:[^|\n]*\|){{2}} \+([\d.]+)%",
                  round(r["shift_pct"], 2), f"{run} probe shift", tol=5e-3, doc=name)
        g = res["data-gb71"]
        claim(d, r"\| data-gb71 \| \*\*yes — trained on\*\* \| ([\d.]+) \|",
              round(g["test_27_books_bpb"], 4), "gb71 probe test bpb", tol=5e-5, doc=name)
        claim(d, r"\| data-gb71 \| \*\*yes — trained on\*\* \| [\d.]+ \| ([\d.]+) \|",
              round(g["new_44_books_bpb"], 4), "gb71 probe new bpb", tol=5e-5, doc=name)
        claim(d, r"\| data-gb71 \| \*\*yes — trained on\*\* \|(?:[^|\n]*\|){2} −([\d.]+)%",
              round(abs(g["shift_pct"]), 2), "gb71 probe shift", tol=5e-3, doc=name)
        claim(d, r"additions ([\d.]+)% harder than its own test", round(sh["clean_shift_pct"], 2),
              "clean out-of-distribution shift", tol=5e-3, doc=name)
        claim(d, r"nearly (\d+)× the seed bar", round(sh["clean_shift_pct"] / bar),
              "shift in bar units", doc=name)
        claim(d, r"by \*\*−([\d.]+)%\*\* and the test score by", round(abs(sh["gb71_new_gain_pct"]), 2),
              "gb71 new-book gain", tol=5e-3, doc=name)
        claim(d, r"the test score by \*\*\+([\d.]+)%\*\*", round(sh["gb71_test_cost_pct"], 2),
              "gb71 test cost", tol=5e-3, doc=name)
        claim(d, r"\*\*(\d+)× more movement on the new", round(sh["gain_cost_ratio"]),
              "gain/cost ratio", doc=name)
        claim(d, r"\(\*\*([\d,]+)\*\* tokens, comparable", sh["probe_tokens"], "probe tokens", doc=name)

    # ---- the one cross-report number in the summary table.
    ev24 = json.load(open(os.path.join(exp, "eval-test-best.json"), encoding="utf-8"))["model"]["bpb"]
    ev48 = json.load(open(os.path.join(exp, "eval-test-4800.json"), encoding="utf-8"))["model"]["bpb"]
    claim(d, r"−([\d.]+) bpb \(−[\d.]+%\) — the only lever", round(ev24 - ev48, 4),
          "steps lever absolute gain", tol=5e-5, doc=name)
    claim(d, r"−[\d.]+ bpb \(−([\d.]+)%\) — the only lever", round(100 * (1 - ev48 / ev24), 2),
          "steps lever relative gain", tol=5e-3, doc=name)

    # ---- the model card quotes the seed bar and the data result; check those too.
    card_path = os.path.join(exp, "model-card.md")
    if os.path.isfile(card_path):
        card = open(card_path, encoding="utf-8").read()
        cn = "model-card.md"
        claim(card, r"seed span\s+\*\*([\d.]+)%\*\* of test bits per byte", round(bar, 2),
              "card: seed bar", tol=5e-3, doc=cn)
        claim(card, r"[Aa]gainst a ([\d.]+)× larger corpus",
              round(large["train_tokens"] / small["train_tokens"], 2), "card: data multiple",
              tol=5e-3, doc=cn)
        claim(card, r"appeared to win by ([\d.]+)% of test bits per byte",
              round(s["data_delta_test_pct"], 2), "card: narrow-evaluation margin", tol=5e-3, doc=cn)
        claim(card, r"inside the ([\d.]+)% seed bar", round(bar, 2), "card: seed bar restated",
              tol=5e-3, doc=cn)
        clean_path = os.path.join(exp, "clean-data-summary.json")
        if os.path.isfile(clean_path):
            cd = json.load(open(clean_path, encoding="utf-8"))
            claim(card, r"the larger corpus wins by ([\d.]+)%",
                  round(abs(cd["full_vs_ctrl_test_pct"]), 2), "card: clean data margin",
                  tol=5e-3, doc=cn)
            claim(card, r"costs \+([\d.]+)% on the original 27 books",
                  round(cd["delta_original27_pct"], 2), "card: original27 cost", tol=5e-3, doc=cn)
            claim(card, r"gains ([\d.]+)% on the rest", round(abs(cd["delta_added44_pct"]), 2),
                  "card: added44 gain", tol=5e-3, doc=cn)



def check_clean_data_report(root: str) -> None:
    """Tie `clean-data-scaling.md` back to `clean-data-summary.json`.

    This report reverses the sign of the project's headline data result, so a
    transcription slip here would be expensive. Every number is re-read from
    the JSON the analysis script emits.
    """
    import json

    exp = os.path.join(root, "experiments", "2026-10-18-gutenberg-mixed")
    sp = os.path.join(exp, "clean-data-summary.json")
    dp = os.path.join(exp, "clean-data-scaling.md")
    if not (os.path.isfile(sp) and os.path.isfile(dp)):
        return
    s = json.load(open(sp, encoding="utf-8"))
    d = open(dp, encoding="utf-8").read()
    name = "clean-data-scaling.md"
    print("experiments/2026-10-18-gutenberg-mixed/clean-data-scaling.md — report vs. raw artifacts")

    full = next(a for a in s["arms"] if a["run"] == "data71-full")
    ctrl = next(a for a in s["arms"] if a["run"] == "data71-ctrl")

    # ---- setup table
    claim(d, r"\| train tokens \| \*\*([\d,]+)\*\*", full["train_tokens"], "full train tokens", doc=name)
    claim(d, r"\| train tokens \| \*\*[\d,]+\*\* \| \*\*([\d,]+)\*\*", ctrl["train_tokens"],
          "ctrl train tokens", doc=name)
    claim(d, r"\| epochs at 39\.3M tokens \| ([\d.]+) \|", round(full["epochs"], 2),
          "full epochs", tol=5e-3, doc=name)
    claim(d, r"\| epochs at 39\.3M tokens \| [\d.]+ \| ([\d.]+) \|", round(ctrl["epochs"], 2),
          "ctrl epochs", tol=5e-3, doc=name)
    claim(d, r"\| val tokens \| ([\d,]+) \|", s["val_tokens"], "val tokens", doc=name)
    claim(d, r"\| test tokens \| ([\d,]+) \|", s["test_tokens"], "test tokens", doc=name)

    # ---- result table
    claim(d, r"\*\*full\*\* \(71 books\) \| ([\d,]+) \|", full["train_tokens"],
          "result row full tokens", doc=name)
    claim(d, r"\*\*full\*\* \(71 books\) \|(?:[^|\n]*\|){2} \*\*([\d.]+)\*\*",
          round(full["val_bpb"], 4), "full val bpb", tol=5e-5, doc=name)
    claim(d, r"\*\*full\*\* \(71 books\) \|(?:[^|\n]*\|){3} \*\*([\d.]+)\*\*",
          round(full["test_bpb"], 4), "full test bpb", tol=5e-5, doc=name)
    claim(d, r"\*\*full\*\* \(71 books\) \|(?:[^|\n]*\|){4} −([\d.]+) \|",
          round(abs(full["end_gap"]), 4), "full gap", tol=5e-5, doc=name)
    claim(d, r"\*\*full\*\* \(71 books\) \|(?:[^|\n]*\|){5} (\d+) \|", full["clipped_steps"],
          "full clipped", doc=name)
    claim(d, r"\*\*full\*\* \(71 books\) \|(?:[^|\n]*\|){6} (\d+) \|", full["longest_verbatim"],
          "full verbatim", doc=name)
    claim(d, r"\| control \(27 books\) \|(?:[^|\n]*\|){2} ([\d.]+) \|", round(ctrl["val_bpb"], 4),
          "ctrl val bpb", tol=5e-5, doc=name)
    claim(d, r"\| control \(27 books\) \|(?:[^|\n]*\|){3} ([\d.]+) \|", round(ctrl["test_bpb"], 4),
          "ctrl test bpb", tol=5e-5, doc=name)
    claim(d, r"\| control \(27 books\) \|(?:[^|\n]*\|){4} −([\d.]+) \|", round(abs(ctrl["end_gap"]), 4),
          "ctrl gap", tol=5e-5, doc=name)
    claim(d, r"\| control \(27 books\) \|(?:[^|\n]*\|){5} (\d+) \|", ctrl["clipped_steps"],
          "ctrl clipped", doc=name)
    claim(d, r"\| Δ \| ([\d.]+)× \|", round(s["data_multiple"], 2), "data multiple", tol=5e-3, doc=name)
    claim(d, r"\| Δ \|(?:[^|\n]*\|){2} \*\*−([\d.]+)%\*\*", round(abs(s["full_vs_ctrl_val_pct"]), 2),
          "val delta", tol=5e-3, doc=name)
    claim(d, r"\| Δ \|(?:[^|\n]*\|){3} \*\*−([\d.]+)%\*\*", round(abs(s["full_vs_ctrl_test_pct"]), 2),
          "test delta", tol=5e-3, doc=name)

    # ---- headline prose
    claim(d, r"helps, by ([\d.]+)% of test bits per byte", round(abs(s["full_vs_ctrl_test_pct"]), 2),
          "headline margin", tol=5e-3, doc=name)
    claim(d, r"— ([\d.]+)× the measured seed\s+bar", round(s["effect_in_bar_units"], 1),
          "margin in bar units", tol=5e-2, doc=name)
    claim(d, r"leads from \*\*step (\d+)\*\*", s["lead_changes"][0]["step"], "first lead step", doc=name)
    claim(d, r"at all \*\*(\d+)\*\* common evaluation", s["common_eval_steps"],
          "common eval points", doc=name)

    # ---- decomposition
    g = s["groups"]
    claim(d, r"the original 27 books \| ([\d,]+) \|", s["group_stats"]["original27_tokens"],
          "original27 tokens", doc=name)
    claim(d, r"the original 27 books \| [\d,]+ \| ([\d.]+) \|", round(g["data71-full"]["original27"], 4),
          "full on original27", tol=5e-5, doc=name)
    claim(d, r"the original 27 books \|(?:[^|\n]*\|){2} ([\d.]+) \|", round(g["data71-ctrl"]["original27"], 4),
          "ctrl on original27", tol=5e-5, doc=name)
    claim(d, r"the original 27 books \|(?:[^|\n]*\|){3} \*\*\+([\d.]+)%\*\*",
          round(s["delta_original27_pct"], 2), "original27 delta", tol=5e-3, doc=name)
    claim(d, r"the added 44 books \| ([\d,]+) \|", s["group_stats"]["added44_tokens"],
          "added44 tokens", doc=name)
    claim(d, r"the added 44 books \| [\d,]+ \| ([\d.]+) \|", round(g["data71-full"]["added44"], 4),
          "full on added44", tol=5e-5, doc=name)
    claim(d, r"the added 44 books \|(?:[^|\n]*\|){2} ([\d.]+) \|", round(g["data71-ctrl"]["added44"], 4),
          "ctrl on added44", tol=5e-5, doc=name)
    claim(d, r"the added 44 books \|(?:[^|\n]*\|){3} \*\*−([\d.]+)%\*\*",
          round(abs(s["delta_added44_pct"]), 2), "added44 delta", tol=5e-3, doc=name)
    claim(d, r"costs \*\*\+([\d.]+)%\*\* on the original", round(s["delta_original27_pct"], 2),
          "restated original27 cost", tol=5e-3, doc=name)
    claim(d, r"model is \*\*([\d.]+)% better\*\*", round(abs(s["delta_added44_pct"]), 2),
          "restated added44 gain", tol=5e-3, doc=name)
    claim(d, r"buys ([\d.]+)% on the rest", round(abs(s["delta_added44_pct"]), 2),
          "blockquote gain", tol=5e-3, doc=name)
    claim(d, r"training on ([\d.]+)× more text costs", round(s["data_multiple"], 2),
          "blockquote multiple", tol=5e-3, doc=name)

    # ---- the reproducibility floor
    ref = json.load(open(os.path.join(exp, "eval-test-4800.json"), encoding="utf-8"))["model"]["bpb"]
    ctrl27 = g["data71-ctrl"]["original27"]
    claim(d, r"scores \*\*([\d.]+)\*\* against the released", round(ctrl27, 4),
          "ctrl on the released test split", tol=5e-5, doc=name)
    claim(d, r"against the released run's \*\*([\d.]+)\*\*", round(ref, 4),
          "released test bpb", tol=5e-5, doc=name)
    claim(d, r"\*\*\+([\d.]+)%\*\*\.\s*\n*The cause", round(100 * (ctrl27 / ref - 1), 3),
          "checkpoint-selection cost", tol=5e-4, doc=name)

    # ---- the standings table
    claim(d, r"\| \*\*training data\*\* \|(?:[^|\n]*\|){1} \*\*−([\d.]+)%\*\*",
          round(abs(s["full_vs_ctrl_test_pct"]), 2), "standings data row", tol=5e-3, doc=name)

    # ---- capacity, re-asked
    if "capacity" in s:
        c = s["capacity"]
        claim(d, r"`data71-full` \| ([\d,]+) \|", c["small"]["params"], "capacity: small params", doc=name)
        claim(d, r"`data71-full` \| [\d,]+ \| ([\d,]+) \|", c["small"]["steps"],
              "capacity: small steps", doc=name)
        claim(d, r"`data71-full` \|(?:[^|\n]*\|){2} ([\d,]+) \|", c["small"]["tokens"],
              "capacity: small tokens", doc=name)
        claim(d, r"`data71-full` \|(?:[^|\n]*\|){3} ([\d,]+) s \|", round(c["small"]["elapsed_s"]),
              "capacity: small elapsed", tol=1, doc=name)
        claim(d, r"`data71-full` \|(?:[^|\n]*\|){4} \*\*([\d.]+)\*\*", round(c["small"]["val_bpb"], 4),
              "capacity: small val", tol=5e-5, doc=name)
        claim(d, r"`data71-full` \|(?:[^|\n]*\|){5} \*\*([\d.]+)\*\*", round(c["small"]["test_bpb"], 4),
              "capacity: small test", tol=5e-5, doc=name)
        claim(d, r"`size-4L256d-gb71` \| ([\d,]+) \|", c["big"]["params"], "capacity: big params", doc=name)
        claim(d, r"`size-4L256d-gb71` \| [\d,]+ \| ([\d,]+) \|", c["big"]["steps"],
              "capacity: big steps", doc=name)
        claim(d, r"`size-4L256d-gb71` \|(?:[^|\n]*\|){2} ([\d,]+) \|", c["big"]["tokens"],
              "capacity: big tokens", doc=name)
        claim(d, r"`size-4L256d-gb71` \|(?:[^|\n]*\|){3} ([\d,]+) s \|", round(c["big"]["elapsed_s"]),
              "capacity: big elapsed", tol=1, doc=name)
        claim(d, r"`size-4L256d-gb71` \|(?:[^|\n]*\|){4} ([\d.]+) \|", round(c["big"]["val_bpb"], 4),
              "capacity: big val", tol=5e-5, doc=name)
        claim(d, r"`size-4L256d-gb71` \|(?:[^|\n]*\|){5} ([\d.]+) \|", round(c["big"]["test_bpb"], 4),
              "capacity: big test", tol=5e-5, doc=name)
        claim(d, r"Wall clock matched to \*\*−([\d.]+)%\*\*", round(abs(c["wall_clock_delta_pct"]), 1),
              "capacity: wall clock match", tol=5e-2, doc=name)
        claim(d, r"At ([\d.]+)× the parameters", round(c["param_multiple"], 2),
              "capacity: param multiple", tol=5e-3, doc=name)
        claim(d, r"is\s+\*\*\+([\d.]+)%\*\* worse", round(c["delta_test_pct"], 2),
              "capacity: margin", tol=5e-3, doc=name)
        claim(d, r"against \+([\d.]+)% on the 27-book corpus",
              round(c["delta_on_27_book_corpus_pct"], 2), "capacity: old margin", tol=5e-3, doc=name)
        claim(d, r"it widened to ([\d.]+)× the seed bar", round(c["delta_in_bar_units"], 1),
              "capacity: margin in bar units", tol=5e-2, doc=name)
        claim(d, r"giving up (\d+)% of its steps",
              round(100 * (1 - c["big"]["steps"] / c["small"]["steps"])), "capacity: steps given up",
              tol=1, doc=name)
        claim(d, r"\| model size \|(?:[^|\n]*\|){1} tie on 27 books, \*\*\+([\d.]+)% worse\*\*",
              round(c["delta_test_pct"], 2), "standings size row", tol=5e-3, doc=name)

    # ---- horizon: the same model and corpus, twice the steps
    if "horizon" in s:
        h = s["horizon"]
        sh, lo = h["short"], h["long"]
        # Scope the row patterns to this section. The capacity table above
        # has a `data71-full` row too, with a different column order, and an
        # unscoped pattern silently reads the wrong one.
        m = re.search(r"\n## Horizon\b.*?(?=\n## )", d, re.S)
        assert m, "horizon section not found in clean-data-scaling.md"
        dh = m.group(0)
        for key, run, bold in (("short", "data71-full", False), ("long", "data71-full-9600", True)):
            r = h[key]
            row = rf"\| `{run}` \|"
            claim(dh, row + r" ([\d,]+) \|", r["steps"], f"horizon: {key} steps", doc=name)
            claim(dh, row + r"(?:[^|\n]*\|){1} ([\d,]+) \|", r["tokens"],
                  f"horizon: {key} tokens", doc=name)
            claim(dh, row + r"(?:[^|\n]*\|){2} ([\d.]+) \|", round(r["epochs"], 2),
                  f"horizon: {key} epochs", tol=5e-3, doc=name)
            claim(dh, row + r"(?:[^|\n]*\|){3} ([\d,]+) \|", round(r["elapsed_s"]),
                  f"horizon: {key} elapsed", tol=1, doc=name)
            claim(dh, row + r"(?:[^|\n]*\|){4} ([\d.]+) \|", round(r["val_bpb"], 4),
                  f"horizon: {key} val", tol=5e-5, doc=name)
            star = r"\*\*([\d.]+)\*\*" if bold else r"([\d.]+)"
            claim(dh, row + r"(?:[^|\n]*\|){5} " + star, round(r["test_bpb"], 4),
                  f"horizon: {key} test", tol=5e-5, doc=name)
        claim(dh, r"bought ([\d.]+)% of test bits per byte", round(h["gain_pct"], 2),
              "horizon: gain pct", tol=5e-3, doc=name)
        claim(dh, r"of test bits per byte\*\* \(([\d.]+) absolute", round(h["gain_bpb"], 4),
              "horizon: gain bpb", tol=5e-5, doc=name)
        claim(dh, r"or ([\d.]+)× the seed bar", round(h["gain_in_bar_units"], 1),
              "horizon: gain in bar units", tol=5e-2, doc=name)
        claim(dh, r"still\s+improving at ([\d.]+) bpb", round(sh["last_four_bpb"], 4),
              "horizon: short marginal", tol=5e-5, doc=name)
        claim(dh, r"the 9,600-step run at ([\d.]+) bpb", round(lo["last_four_bpb"], 4),
              "horizon: long marginal", tol=5e-5, doc=name)
        claim(dh, r"([\d.]+)× flatter", round(sh["last_four_bpb"] / lo["last_four_bpb"], 1),
              "horizon: marginal ratio", tol=5e-2, doc=name)
        claim(dh, r"`best\.pt` is step ([\d,]+)", lo["steps"], "horizon: argmin step", doc=name)
        claim(dh, r"\| original 27 \| ([\d.]+) \|", round(h["released_test_bpb"], 4),
              "horizon: released on original 27", tol=5e-5, doc=name)
        claim(dh, r"\| original 27 \|(?:[^|\n]*\|){2} \*\*([\d.]+)\*\*",
              round(h["groups"]["original27"], 4), "horizon: long on original 27", tol=5e-5, doc=name)
        claim(dh, r"\| added 44 \|(?:[^|\n]*\|){2} \*\*([\d.]+)\*\*",
              round(h["groups"]["added44"], 4), "horizon: long on added 44", tol=5e-5, doc=name)
        claim(dh, r"the longer run is \*\*([\d.]+)% better\*\*",
              round(h["vs_released_on_original27_pct"], 2), "horizon: vs released", tol=5e-3, doc=name)
        claim(dh, r"better\*\* — ([\d.]+)× the\s+seed bar",
              round(h["vs_released_in_bar_units"], 1),
              "horizon: vs released in bar units", tol=5e-2, doc=name)
        pd = h["per_document"]
        claim(dh, r"scores all (\d+) test documents", pd["documents"], "horizon: documents", doc=name)
        claim(dh, r"\| easiest \|(?:[^|\n]*\|){1} ([\d.]+) \|", round(pd["min_bpb"], 4),
              "horizon: easiest bpb", tol=5e-5, doc=name)
        claim(dh, r"\| hardest \|(?:[^|\n]*\|){1} ([\d.]+) \|", round(pd["max_bpb"], 4),
              "horizon: hardest bpb", tol=5e-5, doc=name)
        claim(dh, r"Mean ([\d.]+), sd", round(pd["mean_bpb"], 4), "horizon: mean bpb",
              tol=5e-5, doc=name)
        claim(dh, r"sd ([\d.]+), range", round(pd["stdev_bpb"], 4), "horizon: sd bpb",
              tol=5e-5, doc=name)
        claim(dh, r"range ([\d.]+) —", round(pd["range_bpb"], 4), "horizon: range bpb",
              tol=5e-5, doc=name)
        claim(dh, r"\*\*([\d.]+)% of the mean\*\*", round(pd["range_pct_of_mean"], 1),
              "horizon: range pct of mean", tol=5e-2, doc=name)
        claim(d, r"\| \*corpus mixture\* \|(?:[^|\n]*\|){1} \*([\d.]+)% spread",
              round(pd["range_pct_of_mean"], 1), "standings mixture row", tol=5e-2, doc=name)
        claim(d, r"\*\*training steps\*\* \|(?:[^|\n]*\|){1} \*\*−[\d.]+%\*\*, then \*\*−([\d.]+)%\*\*",
              round(h["gain_pct"], 2), "standings steps row", tol=5e-3, doc=name)

        # ---- the card for the model this run released
        if not os.path.isfile(os.path.join(exp, "model-card-71.md")):
            return
        print("experiments/2026-10-18-gutenberg-mixed/model-card-71.md — card vs. raw artifacts")
        run71 = h["long"]["run"]
        n71 = "model-card-71.md"
        c71 = open(os.path.join(exp, n71), encoding="utf-8").read()
        card = open(os.path.join(exp, "model-card.md"), encoding="utf-8").read()
        lt = json.load(open(os.path.join(exp, f"eval-test-{run71}.json"), encoding="utf-8"))
        ls = json.load(open(os.path.join(exp, "metrics", f"{run71}-summary.json"),
                            encoding="utf-8"))
        claim(c71, r"\| Steps \| ([\d,]+) \|", ls["steps"], "card71: steps", doc=n71)
        claim(c71, r"\| Tokens seen \| ([\d,]+) ", ls["tokens"], "card71: tokens", doc=n71)
        claim(c71, r"\(([\d.]+) epochs\)", round(h["long"]["epochs"], 2), "card71: epochs",
              tol=5e-3, doc=n71)
        claim(c71, r"\*\*([\d,]+)\*\* total", ls["params_total"], "card71: params", doc=n71)
        claim(c71, r"engaged on (\d+) steps", ls["clipped_steps"], "card71: clipped", doc=n71)
        claim(c71, r"Non-finite steps \| (\d+)", ls["nonfinite_steps"], "card71: nonfinite", doc=n71)
        claim(c71, r"\| ([\d,]+) s \(", round(ls["elapsed_s"]), "card71: elapsed", tol=1, doc=n71)
        claim(c71, r"s \(([\d.]+) min\)", round(ls["elapsed_s"] / 60, 1), "card71: minutes",
              tol=5e-2, doc=n71)
        claim(c71, r"at ([\d,]+) tokens/s", round(ls["mean_tokens_per_s"]), "card71: throughput",
              tol=1, doc=n71)
        claim(c71, r"Peak RSS \| ([\d,]+) MB", round(ls["peak_rss_mb"]), "card71: peak rss",
              tol=1, doc=n71)
        claim(c71, r"\*\*This model\*\* \| \*\*([\d.]+)\*\*", round(lt["model"]["bpb"], 4),
              "card71: test bpb", tol=5e-5, doc=n71)
        claim(c71, r"Final validation bits per byte: ([\d.]+)", round(ls["final_val_bpb"], 4),
              "card71: val bpb", tol=5e-5, doc=n71)
        claim(c71, r"split, ([\d,]+) tokens", lt["model"]["tokens"], "card71: eval tokens", doc=n71)
        claim(c71, r"tokens / ([\d,]+) bytes", lt["model"]["bytes"], "card71: eval bytes", doc=n71)
        for base in ("bigram", "unigram"):
            claim(c71, rf"\| {base.capitalize()} \| ([\d.]+) \|", round(lt[base]["bpb"], 4),
                  f"card71: {base} bpb", tol=5e-5, doc=n71)
            claim(c71, rf"\| {base.capitalize()} \| [\d.]+ \| ([\d.]+)×",
                  round(lt[base]["bpb"] / lt["model"]["bpb"], 3), f"card71: {base} ratio",
                  tol=5e-4, doc=n71)
        claim(c71, r"was \*\*(\d+) characters\*\*",
              lt["memorization"]["max_longest_verbatim_match_chars"], "card71: probe chars", doc=n71)
        claim(c71, r"characters\*\* \(mean ([\d.]+)\)",
              lt["memorization"]["mean_longest_verbatim_match_chars"], "card71: probe mean",
              tol=5e-2, doc=n71)
        claim(c71, r"this model is \*\*([\d.]+)% better\*\*",
              round(h["vs_released_on_original27_pct"], 2), "card71: vs released", tol=5e-3, doc=n71)
        claim(c71, r"([\d.]+)× the seed bar", round(h["vs_released_in_bar_units"], 1),
              "card71: vs released in bar units", tol=5e-2, doc=n71)
        claim(c71, r"test bytes: ([\d.]+) vs", round(h["groups"]["original27"], 4),
              "card71: long on original 27", tol=5e-5, doc=n71)
        claim(c71, r"split: ([\d.]+) vs", round(lt["model"]["bpb"], 4),
              "card71: vs short run", tol=5e-5, doc=n71)
        pd71 = h["per_document"]
        claim(c71, r"\| easiest \|(?:[^|\n]*\|){1} ([\d.]+) \|", round(pd71["min_bpb"], 4),
              "card71: easiest", tol=5e-5, doc=n71)
        claim(c71, r"\| hardest \|(?:[^|\n]*\|){1} ([\d.]+) \|", round(pd71["max_bpb"], 4),
              "card71: hardest", tol=5e-5, doc=n71)
        claim(c71, r"Mean ([\d.]+), standard deviation", round(pd71["mean_bpb"], 4),
              "card71: mean", tol=5e-5, doc=n71)
        claim(c71, r"standard deviation ([\d.]+), range", round(pd71["stdev_bpb"], 4),
              "card71: sd", tol=5e-5, doc=n71)
        claim(c71, r"range ([\d.]+) —", round(pd71["range_bpb"], 4), "card71: range",
              tol=5e-5, doc=n71)
        claim(c71, r"\*\*([\d.]+)% of the mean\*\*", round(pd71["range_pct_of_mean"], 1),
              "card71: range pct", tol=5e-2, doc=n71)
        claim(card, r"these same test bytes by ([\d.]+)%",
              round(h["vs_released_on_original27_pct"], 2), "superseded banner", tol=5e-3,
              doc="model-card.md")



def check_batch_report(root: str) -> None:
    """Tie `batch-size.md` back to `batch-summary.json`.

    This report retires a hypothesis the two previous reports built up, so a
    transcription slip here would propagate backwards into how those are read.
    Every number is re-read from the JSON the analysis script emits.
    """
    import json

    exp = os.path.join(root, "experiments", "2026-10-18-gutenberg-mixed")
    sp = os.path.join(exp, "batch-summary.json")
    dp = os.path.join(exp, "batch-size.md")
    if not (os.path.isfile(sp) and os.path.isfile(dp)):
        return
    s = json.load(open(sp, encoding="utf-8"))
    d = open(dp, encoding="utf-8").read()
    name = "batch-size.md"
    print("experiments/2026-10-18-gutenberg-mixed/batch-size.md — report vs. raw artifacts")

    claim(d, r"exactly ([\d,]+) tokens", s["token_budget"], "shared token budget", doc=name)

    # Row per arm in the fixed-rate table: tokens/update, updates, elapsed,
    # clipped, val, test. The centre's test figure is bold, the others are not.
    for run, bold in (("batch-4096", False), ("data71-full", True), ("batch-16384", False)):
        a = s["arms"][run]
        row = rf"\| `{run}` \|"
        claim(d, row + r" ([\d,]+) \|", a["tokens_per_step"], f"{run}: tokens/update", doc=name)
        claim(d, row + r"(?:[^|\n]*\|){1} ([\d,]+) \|", a["steps"], f"{run}: updates", doc=name)
        claim(d, row + r"(?:[^|\n]*\|){2} ([\d,]+) s \|", round(a["elapsed_s"]),
              f"{run}: elapsed", tol=1, doc=name)
        claim(d, row + r"(?:[^|\n]*\|){3} ([\d,]+) \|", a["clipped_steps"],
              f"{run}: clipped", doc=name)
        claim(d, row + r"(?:[^|\n]*\|){4} ([\d.]+) \|", round(a["val_bpb"], 4),
              f"{run}: val", tol=5e-5, doc=name)
        star = r"\*\*([\d.]+)\*\*" if bold else r"([\d.]+)"
        claim(d, row + r"(?:[^|\n]*\|){5} " + star, round(a["test_bpb"], 4),
              f"{run}: test", tol=5e-5, doc=name)
        if "delta_vs_centre_pct" in a:
            claim(d, row + r"(?:[^|\n]*\|){6} \+([\d.]+)%", round(a["delta_vs_centre_pct"], 2),
                  f"{run}: delta vs centre", tol=5e-3, doc=name)

    claim(d, r"([\d.]+)% worse\.\*\* The response",
          round(s["arms"]["batch-4096"]["delta_vs_centre_pct"], 2),
          "small-batch penalty in prose", tol=5e-3, doc=name)
    claim(d, r"largest arm is ([\d.]+)% cheaper",
          round(abs(s["arms"]["batch-16384"]["elapsed_vs_centre_pct"]), 1),
          "large-batch saving", tol=5e-2, doc=name)
    claim(d, r"enough to pay for ([\d.]+)%",
          round(s["arms"]["batch-16384"]["delta_vs_centre_pct"], 2),
          "large-batch penalty", tol=5e-3, doc=name)
    claim(d, r"within\s+([\d.]+)% of each other",
          round(abs(s["arms"]["batch-4096"]["elapsed_vs_centre_pct"]), 1),
          "clock match on the lower half", tol=5e-2, doc=name)
    claim(d, r"leads the first (\d+)\s+and is behind", s["lead_counts"]["batch-4096"],
          "small-batch leads", doc=name)
    claim(d, r"behind for the\s+remaining (\d+)", s["lead_counts"]["data71-full"],
          "centre leads", doc=name)
    claim(d, r"([\d,]+) token-matched evaluation points", s["common_token_points"],
          "common grid size", doc=name)

    # The learning-rate follow-up.
    if "lr_matched" in s:
        for run in ("batch-4096-lr2p12e3", "batch-16384-lr4p24e3"):
            a = s["lr_matched"][run]
            row = rf"\| `{run}` \|"
            claim(d, row + r" ([\d,]+) \|", s["arms"][a["twin"]]["tokens_per_step"],
                  f"{run}: tokens/update", doc=name)
            claim(d, row + r"(?:[^|\n]*\|){2} ([\d,]+) \|", a["clipped_steps"],
                  f"{run}: clipped", doc=name)
            bold = r"\*\*([\d.]+)\*\*" if "4096" in run else r"([\d.]+)"
            claim(d, row + r"(?:[^|\n]*\|){3} " + bold, round(a["test_bpb"], 4),
                  f"{run}: test", tol=5e-5, doc=name)
            claim(d, row + r"(?:[^|\n]*\|){5} −([\d.]+)%", round(abs(a["delta_vs_twin_pct"]), 2),
                  f"{run}: delta vs twin", tol=5e-3, doc=name)
        lo = s["lr_matched"]["batch-4096-lr2p12e3"]
        hi = s["lr_matched"]["batch-16384-lr4p24e3"]
        claim(d, r"small arm improved ([\d.]+)%", round(abs(lo["delta_vs_twin_pct"]), 2),
              "small arm gain from scaling", tol=5e-3, doc=name)
        claim(d, r"the large arm ([\d.]+)%", round(abs(hi["delta_vs_twin_pct"]), 2),
              "large arm gain from scaling", tol=5e-3, doc=name)
        claim(d, r"within ([\d.]+) percentage points",
              round(abs(abs(lo["delta_vs_twin_pct"]) - abs(hi["delta_vs_twin_pct"])), 2),
              "agreement between the two scalings", tol=5e-3, doc=name)
        claim(d, r"scores −([\d.]+)% against the centre", round(abs(lo["delta_vs_centre_pct"]), 2),
              "rescued small arm vs centre", tol=5e-3, doc=name)
        claim(d, r"recovers only to \+([\d.]+)%", round(hi["delta_vs_centre_pct"], 2),
              "rescued large arm vs centre", tol=5e-3, doc=name)
        claim(d, r"still ([\d.]+)× the bar",
              round(hi["delta_vs_centre_pct"] / s["seed_bar_pct"], 1),
              "rescued large arm in bar units", tol=5e-2, doc=name)
        claim(d, r"inside the\s+([\d.]+)% seed bar", round(s["seed_bar_pct"], 2),
              "seed bar", tol=5e-3, doc=name)
        claim(d, r"\(1,262\s+against (\d+)\)", s["arms"]["batch-4096"]["clipped_steps"],
              "unrescued small arm clip count", doc=name)

    # Gradient norm at matched loss.
    gn = s["grad_norm_by_loss_band"]
    for band in ("[3.6, 3.9)", "[3.3, 3.6)", "[3.0, 3.3)"):
        lo_, hi_ = band[1:-1].split(", ")
        row = rf"\| \[{re.escape(lo_)}, {re.escape(hi_)}\) \|"
        for i, run in enumerate(("batch-4096", "batch-4096-lr2p12e3", "data71-full",
                                 "batch-16384", "batch-16384-lr4p24e3")):
            if run not in gn:
                continue
            pat = row + (r"(?:[^|\n]*\|){%d} ([\d.]+)" % i if i else r" ([\d.]+)")
            claim(d, pat, round(gn[run][band]["median_gnorm"], 3),
                  f"gnorm {run} in {band}", tol=5e-4, doc=name)
    tight = "[3.0, 3.3)"
    claim(d, r"sits (\d+)%\s+above the centre",
          round(gn["batch-4096"][tight]["vs_centre_pct"]), "small arm gnorm excess",
          tol=1, doc=name)
    claim(d, r"sits (\d+)% \*below\*",
          round(abs(gn["batch-16384-lr4p24e3"][tight]["vs_centre_pct"])),
          "scaled large arm gnorm deficit", tol=1, doc=name)
    claim(d, r"sits (\d+)% above it",
          round(gn["batch-16384"][tight]["vs_centre_pct"]), "large arm gnorm excess",
          tol=1, doc=name)
    claim(d, r"clips on ([\d.]+)% of its updates",
          round(s["arms"]["batch-4096"]["clipped_pct"], 2), "small arm clip rate",
          tol=5e-3, doc=name)
    claim(d, r"centre's ([\d.]+)%", round(s["arms"]["data71-full"]["clipped_pct"], 2),
          "centre clip rate", tol=5e-3, doc=name)
    claim(d, r"clips least of all\s*\((\d+) events\)",
          s["lr_matched"]["batch-16384-lr4p24e3"]["clipped_steps"],
          "scaled large arm clip count", doc=name)
    claim(d, r"\*\*batch size\*\* \|(?:[^|\n]*\|){1} \*\*tie below [\d,]+, \+([\d.]+)% above\*\*",
          round(s["lr_matched"]["batch-16384-lr4p24e3"]["delta_vs_centre_pct"], 2),
          "standings batch row", tol=5e-3, doc=name)


def check_corpus_composition_report(root: str) -> None:
    """Tie `corpus-composition.md` back to `document-summary.json`.

    This report is the argument for what to build next, so its numbers decide
    where the next several hours of compute go.
    """
    import json

    exp = os.path.join(root, "experiments", "2026-10-18-gutenberg-mixed")
    sp = os.path.join(exp, "document-summary.json")
    dp = os.path.join(exp, "corpus-composition.md")
    if not (os.path.isfile(sp) and os.path.isfile(dp)):
        return
    s = json.load(open(sp, encoding="utf-8"))
    d = open(dp, encoding="utf-8").read()
    name = "corpus-composition.md"
    print("experiments/2026-10-18-gutenberg-mixed/corpus-composition.md — report vs. raw artifacts")

    claim(d, r"across the (\d+)\s+test books", s["documents"], "document count", doc=name)
    claim(d, r"r = −([\d.]+)", round(abs(s["corr_bpb_vs_bytes"]), 3),
          "length correlation", tol=5e-4, doc=name)

    rows = (("narrative prose", "narrative"), ("non-fiction", "non-fiction"),
            ("experimental prose", "experimental"), ("verse", "verse"))
    total_bytes = sum(f["bytes"] for f in s["form"].values())
    for label, key in rows:
        f = s["form"][key]
        row = rf"\| {re.escape(label)} \|"
        claim(d, row + r" (\d+) \|", f["n"], f"{key}: book count", doc=name)
        claim(d, row + r"(?:[^|\n]*\|){1} ([\d,]+) \|", f["bytes"], f"{key}: bytes", doc=name)
        claim(d, row + r"(?:[^|\n]*\|){2} ([\d.]+) \|", round(f["mean_bpb"], 4),
              f"{key}: mean bpb", tol=5e-5, doc=name)
        if key != "narrative":
            claim(d, row + r"(?:[^|\n]*\|){3} \+([\d.]+)%", round(f["vs_narrative_pct"], 1),
                  f"{key}: vs narrative", tol=5e-2, doc=name)
            claim(d, rf"{re.escape(key)}\s+(?:prose\s+)?holds ([\d.]+)%",
                  round(100 * f["bytes"] / total_bytes, 1), f"{key}: byte share",
                  tol=5e-2, doc=name)

    claim(d, r"account for \*\*(\d+)% of the variance", round(100 * s["form_variance_explained"]),
          "variance explained", tol=1, doc=name)
    claim(d, r"\*\*(\d+)% narrative prose by bytes\*\*",
          round(100 * s["form"]["narrative"]["bytes"] / total_bytes), "narrative share",
          tol=1, doc=name)
    claim(d, r"(\d+) of the 71 books. authors contribute more than one book",
          s["multi_book_authors"], "multi-book authors", doc=name)
    claim(d, r"score\s+\*\*([\d.]+)\*\* against", round(s["mean_bpb_multi_book_author"], 4),
          "repeated-author bpb", tol=5e-5, doc=name)
    claim(d, r"against \*\*([\d.]+)\*\* for books", round(s["mean_bpb_single_book_author"], 4),
          "one-off-author bpb", tol=5e-5, doc=name)
    claim(d, r"\*\*([\d.]+)%\*\* advantage", round(s["multi_book_advantage_pct"], 1),
          "repeated-author advantage", tol=5e-2, doc=name)


def main(root: str = ".") -> int:
    check_compute_budget(root)
    check_training_defaults(root)
    check_experiment_reports(root)
    check_data_seed_report(root)
    check_clean_data_report(root)
    check_batch_report(root)
    check_corpus_composition_report(root)
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

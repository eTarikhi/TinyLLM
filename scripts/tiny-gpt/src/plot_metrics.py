"""Render metrics.jsonl into a small SVG chart. No plotting dependency.

Usage:
    python src/plot_metrics.py artifacts/runs/*/metrics.jsonl --out curve.svg --metric val_bpb
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

COLORS = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#0891b2"]


def read_series(path: Path, metric: str) -> tuple[str, list[tuple[float, float]]]:
    label = path.parent.name
    points: list[tuple[float, float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("event") == "run_start":
            label = rec.get("run_id", label)
        if metric in rec and "tokens" in rec:
            points.append((float(rec["tokens"]), float(rec[metric])))
    return label, points


def render(series: list[tuple[str, list[tuple[float, float]]]], metric: str, width: int = 760, height: int = 420) -> str:
    pad_l, pad_r, pad_t, pad_b = 70, 150, 40, 55
    xs = [x for _, pts in series for x, _ in pts]
    ys = [y for _, pts in series for _, y in pts]
    if not xs:
        raise SystemExit(f"no points found for metric {metric!r}")
    x0, x1 = 0.0, max(xs)
    y0, y1 = min(ys), max(ys)
    span = max(y1 - y0, 1e-6)
    y0, y1 = y0 - 0.05 * span, y1 + 0.05 * span

    def px(x: float) -> float:
        return pad_l + (x - x0) / max(x1 - x0, 1e-9) * (width - pad_l - pad_r)

    def py(y: float) -> float:
        return height - pad_b - (y - y0) / (y1 - y0) * (height - pad_t - pad_b)

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" font-family="ui-sans-serif, system-ui, sans-serif">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{pad_l}" y="24" font-size="15" font-weight="600" fill="#111827">{metric} vs training tokens</text>',
    ]
    for i in range(5):
        y = y0 + (y1 - y0) * i / 4
        yy = py(y)
        out.append(f'<line x1="{pad_l}" y1="{yy:.1f}" x2="{width - pad_r}" y2="{yy:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
        out.append(f'<text x="{pad_l - 8}" y="{yy + 4:.1f}" font-size="11" fill="#6b7280" text-anchor="end">{y:.3f}</text>')
    for i in range(5):
        x = x0 + (x1 - x0) * i / 4
        xx = px(x)
        out.append(f'<line x1="{xx:.1f}" y1="{pad_t}" x2="{xx:.1f}" y2="{height - pad_b}" stroke="#f3f4f6" stroke-width="1"/>')
        out.append(f'<text x="{xx:.1f}" y="{height - pad_b + 18}" font-size="11" fill="#6b7280" text-anchor="middle">{x / 1e6:.2f}M</text>')
    out.append(f'<text x="{(pad_l + width - pad_r) / 2:.0f}" y="{height - 12}" font-size="12" fill="#374151" text-anchor="middle">training tokens</text>')

    for i, (label, pts) in enumerate(series):
        if not pts:
            continue
        color = COLORS[i % len(COLORS)]
        d = " ".join(f"{'M' if j == 0 else 'L'}{px(x):.1f},{py(y):.1f}" for j, (x, y) in enumerate(pts))
        out.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2"/>')
        last_x, last_y = pts[-1]
        out.append(f'<circle cx="{px(last_x):.1f}" cy="{py(last_y):.1f}" r="3" fill="{color}"/>')
        out.append(
            f'<text x="{width - pad_r + 10}" y="{pad_t + 16 + i * 18}" font-size="12" fill="{color}">'
            f'{label} ({pts[-1][1]:.3f})</text>'
        )
    out.append("</svg>")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot metrics.jsonl files as an SVG.")
    ap.add_argument("metrics", nargs="+")
    ap.add_argument("--metric", default="val_loss")
    ap.add_argument("--out", default="curve.svg")
    args = ap.parse_args()
    series = [read_series(Path(p), args.metric) for p in args.metrics]
    svg = render(series, args.metric)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(svg, encoding="utf-8")
    print(f"wrote {args.out} ({sum(len(p) for _, p in series)} points from {len(series)} run(s))")


if __name__ == "__main__":
    main()

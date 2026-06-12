#!/usr/bin/env python3
"""Generate the site's benchmark SVGs from a bench results.csv.

Usage: python3 docs/gen_charts.py build/results_site.csv docs/charts
Reads rows of (contestant, workload, op, n, ns_per_op) and emits one
horizontal-bar SVG per (workload, op) the site embeds. Lower is better;
artpp variants carry the accent gradient, everything else the cool grays.
Regenerate whenever the numbers are refreshed — the charts ARE the data.
"""
import csv
import sys
from pathlib import Path

ACCENT = {"artpp::map": "url(#hot)", "artpp::map<buckets>": "url(#hot2)"}
GRAY = {
    "artpp::map (std::allocator)": "#3f8f86",
    "libart": "#5e97d0",
    "absl::btree_map": "#7587a3",
    "std::map": "#4a5a73",
}
ORDER = [
    "artpp::map",
    "artpp::map<buckets>",
    "artpp::map (std::allocator)",
    "libart",
    "absl::btree_map",
    "std::map",
]
TITLES = {
    ("dict", "hit"): "Point lookups — dictionary words (236k string keys)",
    ("clustered", "hit"): "Point lookups — clustered strings (885k keys)",
    ("uniform", "hit"): "Point lookups — uniform random uint64 (1M keys)",
    ("sequential", "hit"): "Point lookups — sequential uint64 (1M keys)",
    ("dict", "insert"): "Insert — dictionary words",
    ("clustered", "insert"): "Insert — clustered strings",
    ("uniform", "insert"): "Insert — uniform uint64",
    ("sequential", "insert"): "Insert — sequential uint64",
    ("dict", "scan"): "Full ordered scan — dictionary words",
    ("clustered", "scan"): "Full ordered scan — clustered strings",
    ("uniform", "scan"): "Full ordered scan — uniform uint64",
    ("sequential", "scan"): "Full ordered scan — sequential uint64",
}


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def chart(title: str, rows: list[tuple[str, float]], out: Path) -> None:
    pad_l, pad_r, bar_h, gap, top = 190, 86, 30, 14, 54
    width = 760
    height = top + len(rows) * (bar_h + gap) + 16
    vmax = max(v for _, v in rows)
    span = width - pad_l - pad_r
    p = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'font-family="ui-sans-serif, system-ui, sans-serif" role="img" aria-label="{esc(title)}">',
        "<defs>"
        '<linearGradient id="hot" x1="0" y1="0" x2="1" y2="0">'
        '<stop offset="0" stop-color="#22d3ee"/><stop offset="1" stop-color="#34d399"/></linearGradient>'
        '<linearGradient id="hot2" x1="0" y1="0" x2="1" y2="0">'
        '<stop offset="0" stop-color="#34d399"/><stop offset="1" stop-color="#a3e635"/></linearGradient>'
        "</defs>",
        f'<text x="14" y="28" font-size="17" font-weight="600" fill="#e6edf6">{esc(title)}</text>',
        f'<text x="14" y="46" font-size="12" fill="#8da3bf">ns / operation — lower is better</text>',
    ]
    for i, (name, v) in enumerate(rows):
        y = top + i * (bar_h + gap)
        w = max(3.0, span * v / vmax)
        fill = ACCENT.get(name) or GRAY.get(name, "#4a5a73")
        weight = "600" if name in ACCENT else "400"
        col = "#e6edf6" if name in ACCENT else "#b7c5d8"
        p.append(
            f'<text x="{pad_l - 10}" y="{y + bar_h / 2 + 4}" font-size="13" font-weight="{weight}" '
            f'fill="{col}" text-anchor="end">{esc(name)}</text>'
        )
        p.append(f'<rect x="{pad_l}" y="{y}" width="{w:.1f}" height="{bar_h}" rx="6" fill="{fill}"/>')
        p.append(
            f'<text x="{pad_l + w + 8}" y="{y + bar_h / 2 + 4}" font-size="13" '
            f'fill="#e6edf6">{v:,.1f}</text>'
        )
    p.append("</svg>")
    out.write_text("\n".join(p))


def main() -> None:
    src, outdir = Path(sys.argv[1]), Path(sys.argv[2])
    outdir.mkdir(parents=True, exist_ok=True)
    data: dict[tuple[str, str], dict[str, float]] = {}
    with src.open() as f:
        for r in csv.DictReader(f):
            data.setdefault((r["workload"], r["op"]), {})[r["contestant"]] = float(r["ns_per_op"])
    made = 0
    for key, title in TITLES.items():
        if key not in data:
            continue
        rows = [(c, data[key][c]) for c in ORDER if c in data[key]]
        chart(title, rows, outdir / f"{key[0]}_{key[1]}.svg")
        made += 1
    print(f"wrote {made} charts to {outdir}")


if __name__ == "__main__":
    main()

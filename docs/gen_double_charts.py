#!/usr/bin/env python3
"""Generate the double-keyed scaling SVGs (the 1M→100M story) for the site.

Measured ns/op on an idle M5 (128 GB), min-of-reps, RANDOM insertion order; doubles
keyed through psio's `key` codec via psio::to_key (order-preserving, fixed-width → a
stack buffer, no per-op allocation). Each container built/measured in isolation (one map
resident at a time). libart is the upstream C radix tree (point ops only — it has no
lower_bound and we key it on the same normalized bytes); std::unordered_map has no
ordered scan / lower_bound at any size.

  python3 docs/gen_double_charts.py docs/charts
"""
import sys
from pathlib import Path

# (1M, 100M) ns per lookup
HIT = {
    "artpp::map": (37.8, 85.5),
    "libart": (53.9, 107.0),
    "std::unordered_map": (13.2, 37.3),
    "absl::btree_map": (87.8, 404.0),
    "std::map": (179.0, 614.4),
}
HIT_100 = {k: v[1] for k, v in HIT.items()}
LB_100 = {  # lower_bound at 100M; None → no ordered API
    "artpp::map": 178.5,
    "absl::btree_map": 397.9,
    "std::map": 642.7,
    "libart": None,
    "std::unordered_map": None,
}

LINE = {
    "artpp::map": "#34d399",
    "libart": "#5e97d0",
    "std::unordered_map": "#a78bfa",
    "absl::btree_map": "#7587a3",
    "std::map": "#4a5a73",
}
BARFILL = dict(LINE, **{"artpp::map": "url(#hot)"})
TXT, SUB, GRID = "#e6edf6", "#8da3bf", "#22304a"
DEFS = (
    '<defs><linearGradient id="hot" x1="0" y1="0" x2="1" y2="0">'
    '<stop offset="0" stop-color="#22d3ee"/><stop offset="1" stop-color="#34d399"/>'
    "</linearGradient></defs>"
)


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def slope(out: Path) -> None:
    W, H = 820, 470
    padl, padr, top, padb = 58, 116, 78, 44
    x1, x2 = padl + 120, W - padr - 36
    ymax = 650.0
    yb, yt = H - padb, top

    def Y(v):
        return yb - (v / ymax) * (yb - yt)

    p = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
        f'font-family="ui-sans-serif, system-ui, sans-serif" role="img" '
        f'aria-label="Point-lookup latency from 1M to 100M double keys">',
        DEFS,
        f'<text x="14" y="28" font-size="17" font-weight="600" fill="{TXT}">'
        "Point lookups: latency as the map grows 100×</text>",
        f'<text x="14" y="46" font-size="11.5" fill="{SUB}">double keys · ns per lookup, lower is '
        "better · the radix trees stay flat; the comparison trees fan upward</text>",
    ]
    # legend row
    lx = 14
    for name in HIT:
        p.append(f'<rect x="{lx}" y="58" width="12" height="12" rx="2" fill="{LINE[name]}"/>')
        p.append(f'<text x="{lx+17}" y="68" font-size="11" fill="{SUB}">{esc(name)}</text>')
        lx += 22 + 8.0 * len(name) + 14
    # y grid
    for gv in range(0, 651, 100):
        y = Y(gv)
        p.append(f'<line x1="{x1}" y1="{y:.1f}" x2="{x2}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        p.append(f'<text x="{x1-8}" y="{y+4:.1f}" font-size="10.5" fill="{SUB}" text-anchor="end">{gv}</text>')
    p.append(f'<text x="{x1}" y="{yb+22}" font-size="12.5" fill="{SUB}" text-anchor="middle">1M keys</text>')
    p.append(f'<text x="{x2}" y="{yb+22}" font-size="12.5" fill="{SUB}" text-anchor="middle">100M keys</text>')
    # lines (draw comparison trees first, radix on top)
    for name in ("std::map", "absl::btree_map", "std::unordered_map", "libart", "artpp::map"):
        v1, v2 = HIT[name]
        col, wide = LINE[name], (3.4 if name == "artpp::map" else 2.2)
        p.append(f'<line x1="{x1}" y1="{Y(v1):.1f}" x2="{x2}" y2="{Y(v2):.1f}" stroke="{col}" stroke-width="{wide}"/>')
        p.append(f'<circle cx="{x1}" cy="{Y(v1):.1f}" r="3.6" fill="{col}"/>')
        p.append(f'<circle cx="{x2}" cy="{Y(v2):.1f}" r="4.5" fill="{col}"/>')
    # de-collided value labels at the 100M end
    labels = sorted(((Y(v2), v2, LINE[n], n) for n, (v1, v2) in HIT.items()))
    placed = []
    for y, v, col, n in labels:
        ly = y if not placed else max(y, placed[-1] + 15)
        placed.append(ly)
        bold = ' font-weight="600"' if n == "artpp::map" else ""
        p.append(f'<text x="{x2+10}" y="{ly+4:.1f}" font-size="12" fill="{col}"{bold}>{v:.0f} ns</text>')
    p.append("</svg>")
    out.write_text("\n".join(p))


def bars(title: str, sub: str, rows, out: Path) -> None:
    padl, padr, bar_h, gap, top = 190, 96, 30, 16, 54
    W = 760
    H = top + len(rows) * (bar_h + gap) + 16
    vmax = max(v for _, v in rows if v is not None)
    span = W - padl - padr
    p = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
        f'font-family="ui-sans-serif, system-ui, sans-serif" role="img" aria-label="{esc(title)}">',
        DEFS,
        f'<text x="14" y="28" font-size="17" font-weight="600" fill="{TXT}">{esc(title)}</text>',
        f'<text x="14" y="46" font-size="11.5" fill="{SUB}">{esc(sub)}</text>',
    ]
    for i, (name, v) in enumerate(rows):
        y = top + i * (bar_h + gap)
        accent = name == "artpp::map"
        col = TXT if accent else "#b7c5d8"
        p.append(f'<text x="{padl-10}" y="{y+bar_h/2+4}" font-size="13" font-weight="{"600" if accent else "400"}" '
                 f'fill="{col}" text-anchor="end">{esc(name)}</text>')
        if v is None:
            p.append(f'<rect x="{padl}" y="{y}" width="{span}" height="{bar_h}" rx="6" fill="none" '
                     f'stroke="#2e4f7a" stroke-width="1.5" stroke-dasharray="5 4"/>')
            p.append(f'<text x="{padl+span/2}" y="{y+bar_h/2+4}" font-size="12.5" fill="#7587a3" '
                     f'text-anchor="middle" font-style="italic">no ordered API — unsupported</text>')
            continue
        w = max(3.0, span * v / vmax)
        p.append(f'<rect x="{padl}" y="{y}" width="{w:.1f}" height="{bar_h}" rx="6" fill="{BARFILL[name]}"/>')
        p.append(f'<text x="{padl+w+8:.1f}" y="{y+bar_h/2+4}" font-size="13" fill="{TXT}">{v:,.0f} ns</text>')
    p.append("</svg>")
    out.write_text("\n".join(p))


def main() -> None:
    outdir = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/charts")
    outdir.mkdir(parents=True, exist_ok=True)
    slope(outdir / "double_scale_hit.svg")
    order = ["artpp::map", "libart", "std::unordered_map", "absl::btree_map", "std::map"]
    bars("Point lookups at 100M keys",
         "double keys · ns / lookup, lower is better · artpp leads every ordered map; only the hash is faster",
         [(n, HIT_100[n]) for n in order],
         outdir / "double_hit_100M.svg")
    bars("lower_bound at 100M keys",
         "double keys · ns / op · neither the hash map nor libart offers ordered positioning",
         [(n, LB_100[n]) for n in order],
         outdir / "double_lbound_100M.svg")
    print(f"wrote 3 double-keyed charts to {outdir}")


if __name__ == "__main__":
    main()

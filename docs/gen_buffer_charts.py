#!/usr/bin/env python3
"""Generate the 32-byte-buffer (real-world key) scaling SVGs for the site.

Measured ns/op on an idle M5 (128 GB), min-of-reps, random 32-byte keys (hashes/UUIDs/
composite — the key that doesn't fit a register). Each container built/measured in
isolation. artpp reaches 100M on std::allocator (line_pool caps near 50M for 32-byte keys:
its 4 GB terminal region is the price of 4-byte handles). art_map is ABSENT: its key must
fit one register-width integer, so a 32-byte key fails to compile (static_assert).

  python3 docs/gen_buffer_charts.py docs/charts
"""
import sys
from pathlib import Path

HIT = {  # (1M, 100M) ns per lookup
    "artpp::map": (51.9, 89.5),
    "std::unordered_map": (64.3, 100.0),
    "libart": (66.0, 117.8),
    "unodb": (210.4, 352.4),
    "absl::btree_map": (505.3, 1470.8),
    "std::map": (807.1, 1965.9),
}
HIT_100 = {k: v[1] for k, v in HIT.items()}
LB_100 = {"artpp::map": 158.8, "unodb": 638.2, "absl::btree_map": 1488.7, "std::map": 2169.1,
          "libart": None, "std::unordered_map": None}
# why a contestant has no bar
NOTE = {"libart": "no ordered query API", "std::unordered_map": "no ordered query API",
        "art_map": "key > 16 B — does not compile"}

LINE = {"artpp::map": "#34d399", "libart": "#5e97d0", "std::unordered_map": "#a78bfa",
        "unodb": "#d99a4e", "absl::btree_map": "#7587a3", "std::map": "#4a5a73", "art_map": "#6b6f7a"}
BARFILL = dict(LINE, **{"artpp::map": "url(#hot)"})
TXT, SUB, GRID = "#e6edf6", "#8da3bf", "#22304a"
DEFS = ('<defs><linearGradient id="hot" x1="0" y1="0" x2="1" y2="0">'
        '<stop offset="0" stop-color="#22d3ee"/><stop offset="1" stop-color="#34d399"/>'
        "</linearGradient></defs>")


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def slope(out: Path) -> None:
    W, H = 820, 470
    padl, padr, top, padb = 64, 116, 78, 44
    x1, x2 = padl + 120, W - padr - 36
    ymax, yb, yt = 2100.0, H - padb, top

    def Y(v):
        return yb - (v / ymax) * (yb - yt)

    p = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
         f'font-family="ui-sans-serif, system-ui, sans-serif" role="img" '
         f'aria-label="32-byte-key point-lookup latency from 1M to 100M">',
         DEFS,
         f'<text x="14" y="28" font-size="17" font-weight="600" fill="{TXT}">'
         "32-byte keys: point-lookup latency as the map grows 100×</text>",
         f'<text x="14" y="46" font-size="11.5" fill="{SUB}">ns per lookup, lower is better · '
         "the comparison trees explode; the radix tree stays flat (art_map can&#39;t key 32 B)</text>"]
    lx = 14
    for name in HIT:
        p.append(f'<rect x="{lx}" y="58" width="12" height="12" rx="2" fill="{LINE[name]}"/>')
        p.append(f'<text x="{lx+17}" y="68" font-size="11" fill="{SUB}">{esc(name)}</text>')
        lx += 22 + 8.0 * len(name) + 14
    for gv in range(0, 2101, 500):
        y = Y(gv)
        p.append(f'<line x1="{x1}" y1="{y:.1f}" x2="{x2}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        p.append(f'<text x="{x1-8}" y="{y+4:.1f}" font-size="10.5" fill="{SUB}" text-anchor="end">{gv}</text>')
    p.append(f'<text x="{x1}" y="{yb+22}" font-size="12.5" fill="{SUB}" text-anchor="middle">1M keys</text>')
    p.append(f'<text x="{x2}" y="{yb+22}" font-size="12.5" fill="{SUB}" text-anchor="middle">100M keys</text>')
    for name in ("std::map", "absl::btree_map", "unodb", "libart", "std::unordered_map", "artpp::map"):
        v1, v2 = HIT[name]
        col, wide = LINE[name], (3.4 if name == "artpp::map" else 2.2)
        p.append(f'<line x1="{x1}" y1="{Y(v1):.1f}" x2="{x2}" y2="{Y(v2):.1f}" stroke="{col}" stroke-width="{wide}"/>')
        p.append(f'<circle cx="{x1}" cy="{Y(v1):.1f}" r="3.6" fill="{col}"/>')
        p.append(f'<circle cx="{x2}" cy="{Y(v2):.1f}" r="4.5" fill="{col}"/>')
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
    padl, padr, bar_h, gap, top = 190, 110, 30, 16, 54
    W = 780
    H = top + len(rows) * (bar_h + gap) + 16
    vmax = max(v for _, v in rows if v is not None)
    span = W - padl - padr
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
         f'font-family="ui-sans-serif, system-ui, sans-serif" role="img" aria-label="{esc(title)}">',
         DEFS,
         f'<text x="14" y="28" font-size="17" font-weight="600" fill="{TXT}">{esc(title)}</text>',
         f'<text x="14" y="46" font-size="11.5" fill="{SUB}">{esc(sub)}</text>']
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
                     f'text-anchor="middle" font-style="italic">{esc(NOTE.get(name, "unsupported"))}</text>')
            continue
        w = max(3.0, span * v / vmax)
        p.append(f'<rect x="{padl}" y="{y}" width="{w:.1f}" height="{bar_h}" rx="6" fill="{BARFILL[name]}"/>')
        p.append(f'<text x="{padl+w+8:.1f}" y="{y+bar_h/2+4}" font-size="13" fill="{TXT}">{v:,.0f} ns</text>')
    p.append("</svg>")
    out.write_text("\n".join(p))


def main() -> None:
    outdir = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/charts")
    outdir.mkdir(parents=True, exist_ok=True)
    slope(outdir / "buffer_scale_hit.svg")
    hit_order = ["artpp::map", "std::unordered_map", "libart", "unodb", "absl::btree_map", "std::map", "art_map"]
    bars("Point lookups at 100M keys (32-byte buffers)",
         "ns / lookup, lower is better · artpp leads the C++ ART field and the trees alike",
         [(n, {**HIT_100, "art_map": None}.get(n)) for n in hit_order],
         outdir / "buffer_hit_100M.svg")
    lb_order = ["artpp::map", "unodb", "absl::btree_map", "std::map", "libart", "std::unordered_map", "art_map"]
    bars("lower_bound at 100M keys (32-byte buffers)",
         "ns / op · artpp 4× over unodb (the canonical C++ ART), far over the trees; hash/libart have none",
         [(n, {**LB_100, "art_map": None}.get(n)) for n in lb_order],
         outdir / "buffer_lbound_100M.svg")
    print(f"wrote 3 buffer charts to {outdir}")


if __name__ == "__main__":
    main()

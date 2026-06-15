# artpp v2 — node layout & descent spec (DRAFT for review)

Companion to `alt_ptr_design.md` (the rationale). This is the concrete, reviewable
specification of every node type's byte layout, the handle encoding, the common-prefix
policy, inline packing, the per-tier index kernel, and the structural transitions —
the contract the `v2::map` node layer will be built against.

Validated already (proto, M5 -O3, 2M keys, vs v1 `mode::none`): **find 2.2× faster
(spread) / 1.4× (text), memory 1.9× / 1.1× denser, identical descent node-count.** This
spec is how we turn that into a functional map without regressing v1's inline win.

---

## 0. Units & invariants

- **Line = 128 B** (`cacheline_bytes`) = one M-series hardware cacheline = the alloc unit.
  All router nodes are 128 B-aligned and a whole number of lines.
- **Goal: 1 line loaded per descent level** for the common case (inline/replicated prefix). The
  navigation metadata for *which* line of a child to load rides in the **parent's handle** (in a
  register), so we never load a child header line just to route. **Exception:** a `tail_prefix`
  node (long shared prefix, §2.3) costs `O(plen/128)` extra lines — the long-run case where one
  prefetch can't hide a multi-line cold prefix (M6); the common short-prefix case stays at 1.
- **Branchless line-selection at every tier** (divider compare / `byte>>4` / popcount-rank);
  only the in-line scan branches, and that's the NEON parallel compare.
- **Stems stay sorted ascending, unused lanes padded `0xFF`** — preserves ordered iteration
  / `lower_bound` and lets the SIMD scan run mask-free (v1 invariant, kept).
- **Terminals are prefix-free at the router level**: a key ending at a router lives in that
  router's `term` slot (v1 invariant, kept).

---

## 1. The handle (tagged pointer)

`N`-byte little-endian value, templated so node branch arrays re-tile from it. The **2 high
bytes are navigation metadata**; the low `N-2` bytes are the address field with the tag in its
low alignment bits. The address width is chosen per allocator — and it directly buys
`sparse_full` inline-prefix room (§3.4):

**Tag = the low 5 bits** (`kind`:4 + `tail_prefix`:1); the index/pointer takes the rest, so the
pool size is computed AFTER those 5 bits (fixes B2 — the tag and index must not both claim the
alignment hole):

| `N` | address field | index | addressable | `sparse_full` range (16 slots) |
|---|---|---|---|---|
| 6 | 4-byte offset | 27-bit ×128 B | 16 GB pool | 16·6=96 B → ≤15 B prefix fits |
| **7** | **5-byte offset** | **35-bit ×128 B** | **4 TB pool** | 16·7=112 B → **≤15 B prefix fits** |
| 8 | 6-byte 48-bit ptr | tag in 128 B-align hole | full malloc | 16·8=128 B → no room → **tail prefix** |

**7 is the pool sweet spot**: same 16 branches/range as 8 B, but the smaller offset frees ~15 B/range
for the replicated prefix. Direct/malloc (8) → tail cacheline. (A COW/refcount bit is deferred — in
indexed mode it would cost one more index bit.)

- for `N ≤ 8` the handle IS a `u64`: load once, extract tag/meta/addr by shift+mask (the
  measured fast path) rather than a `u8[]` array of per-field `memcpy`s.

```
[ address field : N-2 bytes ][ metadata : 2 bytes ]
indexed:  field = [tag : low 5 bits][line index];  addr = base + (index << 7)   // ×128 B
direct :  field = 48-bit pointer (128 B-aligned ⇒ low 7 bits free); tag in low 5; addr = ptr & ~0x7f
```
The handle is a `u64` for `N ≤ 8`: load once, extract tag/meta/addr by shift+mask (the measured
fast path). Branch/range arrays are padded so an 8-byte load at the last handle stays in-bounds;
the over-read bytes are masked (N5).

Tag — the low **5** bits of the address field:

| bits | field | meaning |
|---|---|---|
| 0..3 | `kind` | node kind (table below) |
| 4 | `tail_prefix` | 1 ⇒ this node's common prefix is appended at its tail (§2.3) |

`metadata` (2 bytes) — interpreted by `kind`, and overridden when `tail_prefix` is set:

| kind | value | metadata 16 bits |
|---|---|---|
| `value_ptr` | 0 | — (external leaf) |
| `value_inline` | 1 | part of the value — `value_inline` uses the freed address field **+** meta, up to **N-1 bytes** after the tag byte (e.g. ≤7 B at N=8); fixes M7 |
| `setlist` | 2 | — (1 line; inline prefix in the line) |
| `sdiv` | 3 | two divider bytes `[d0][d1]` (§3.3); `nlines` (2–4) + line derived branchlessly |
| `sparse_full` | 4 | 16-bit presence vector of occupied sub-ranges |
| `full` | 5 | — (all sub-ranges present; rank-free) |
| `user_ptr` | 6 | an inline user **pointer** value — the value is a pointer; reconstructed at full resolution from the address field **+** the 2 meta bytes (the meta restores the 5 bits the tag stole). No external leaf. |
| `null` | 0xF | the all-ones handle |

> **When `tail_prefix` is set, `metadata` instead holds the prefix length** (`u16`, ≤4 KB) — so the
> routing byte `key[depth+plen]` is computable from the register *before* any cold load (fixes B4,
> homes the u16 of M10). Such a node's dividers/presence then live in its CL0, read alongside the
> tail prefix; a `tail_prefix` node is the acknowledged exception to "1 line/level" (§0, M6).

Routers are contiguous (`setlist`..`full` = 2..5) so `is_router` is one range check and the
descent jump table stays dense.

> **Open decision A — RESOLVED (no `prefix_node`).** A long shared run is just a prefix *on the
> router it leads into* — inline (≤7), replicated across sdiv lines (≤ `SDIV_PREFIX_CAP`), or
> tail-appended cachelines (§2.3); a lone key's long tail is simply its leaf suffix. There is no
> "prefix + one child" shape left, so the dedicated `prefix_node` type is **dropped entirely**.
> **`bucket` is also dropped** — prefix-free leaves + inline children cover the last mile.

---

## 2. Common-prefix policy (reqs 1–3) — the decision ladder

A router covering branches that share a `plen`-byte prefix (consumed before the routing
byte) places that prefix by the **first rule that fits**:

1. **Inline in the setlist line (req 1).** `plen ≤ PREFIX_CAP`. The single S1 line (or, for an
   sdiv, see rule 2) carries the prefix in its `prefix[]` field. No extra line, no extra hop.
   This is v1's behavior and the common case (dict shared prefixes avg <2 B, 99.9% ≤8).

2. **Replicated across all sdiv lines (req 2).** When the node is an `sdiv` of N=2..4 setlist
   lines and `plen ≤ SDIV_PREFIX_CAP` (≈14), **every** line stores its own copy of the prefix.
   It is redundant, but it means the one line we route to already contains the prefix — so the
   descent still loads exactly ONE line. Storing the prefix in only one line would force a
   second cacheline load whenever the key routes elsewhere; replication trades a few bytes to
   keep 1 line/level. (Replicated copies are kept identical on every structural change.)

3. **Tail prefix + prefetch (req 3).** When `plen` exceeds what rule 1/2 can inline, the node
   sets the `tail_prefix` tag bit and appends the prefix **at the very end of the node**, after
   all branch lines. Crucially the branch lines keep their normal offsets, so divider/presence
   routing is unchanged. Descent then runs:

   ```
   line_off = route(kind, metadata, byte)      // which branch line we WILL need
   prefetch(node + line_off)                    // warm the branch line NOW
   p = node + tail_off                          // the prefix lines, at the tail
   if (key[depth..depth+plen) != p[0..plen))    // cold-loads the tail; compared meanwhile
        return miss
   depth += plen
   child = *(Ptr*)(node + line_off + slot)       // branch line already warm (prefetched)
   ```

   `plen` lives in the handle's `metadata` (register) when `tail_prefix` is set (B4/M10), so the
   routing byte `key[depth+plen]` is known *before* any cold load and the branch-line prefetch can
   issue. For `plen ≲ 112` the single prefetch hides the one cold prefix line → ≈1 effective miss.
   **A longer prefix spans multiple lines (≤4 KB) and costs `O(plen/128)` lines** (M6) — acceptable
   for the rare long-run case; chain-prefetch the tail lines to soften it. A run >4 KB chains
   tail-prefixed routers (§3.5) — no `prefix_node` pointer-chase.

> **Open decision B — RESOLVED:** the tail `plen` is a `u16` carried in `metadata` (not the 6-bit
> `node_hdr.prefix_len`) whenever `tail_prefix` is set; that node's dividers/presence then live in
> its CL0, read alongside the prefix.

---

## 3. Node types — byte layouts (W = sizeof(handle); 6 pool / 8 direct)

### 3.1 Terminals — `value_ptr` (external) / `value_inline` / `user_ptr` (in-handle)
- `value_inline`: a trivially-copyable `T` rides in the handle — the address field **+** the 2 meta
  bytes give `(N-2)*8 − 5 + 16` value bits after the tag (≈51 b at N=7, **59 b at N=8**), so any `T`
  up to ~6 bytes inlines with no allocation. (A full `uint64` value, 64 b, exceeds this → external.)
- `user_ptr`: the value **is a pointer**. A 48-bit pointer won't fit the tag-stolen address field
  alone, so its full resolution is reconstructed from the address field + the 2 meta bytes (the meta
  carries the 5 bits the tag occupies). **Pointer-valued maps (`key → object*`) are then allocation-free
  on the value side** — no external leaf, no indirection.
- `value_ptr` → external leaf, 16 B-granular (terminal region):
  ```
  [ u16 suffix_len ][ suffix bytes ][ T value ]
  ```

### 3.2 `setlist` (S1) — 1 line, ≤ CAP branches  *(v1 layout, kept verbatim)*
```
router_hdr rh        // node_hdr{has_term:1, nbranch:9, prefix_len:6} + term handle (W)
u8         inl       // bit7 = inline mode active; bits0..6 = per-branch inline-child flag
u8         bytes[16] // sorted stems, unused lanes = 0xFF (SIMD lower_bound, mask-free)
u8         prefix[PREFIX_CAP]   // PREFIX_CAP = 7 ; rh.prefix_len bytes used (req 1)
Ptr        branch[CAP]          // LAST — slots [nbranch..CAP) + line tail form the inline region
```
**CAP = min(LANES=16, floor((128 − round_up(2+W,2) − 1[inl] − 16[bytes] − 7[prefix]) / W))** (B3 +
the v1 16-lane SIMD cap I'd dropped). The lane count binds for the smaller handles, so **W4/W5/W6 → 16**
(v1's proven density), **W7 → 13, W8 → 11** — not a flat 16, but 16 across the whole pool-handle range
v1 used. `branch[]` is last so `[&branch[nbranch], line+128)` is one contiguous **inline region** for
inline children (§5). When `tail_prefix` is set the `prefix[7]` is unused (length in `metadata`), CAP unchanged.

### 3.3 `sdiv` (S2/S3/S4) — 2..4 setlist lines under in-handle dividers
A contiguous block of `nlines` (2..4) **setlist lines**, each laid out exactly as §3.2 (so the
in-line scan/insert/inline code is shared verbatim). The routing byte selects the line via the
handle's divider metadata; the chosen line is scanned like an S1.

- `metadata = [d0][d1]` (two divider bytes), encoded so ONE `sdiv` tag covers S2/S3/S4 with
  no per-tier branch. **`0` = inactive divider = the implicit start of the key range**; S4 is
  signalled by storing its two half-dividers in *descending* order:
  - S2 = `(0, AA)`            — one divider at `AA`        → 2 lines `[0,AA) [AA,256)`
  - S3 = `(AA, BB)`, `0<AA<BB` (ascending)                 → 3 lines
  - S4 = `(hi, lo)`, `hi>128>lo≥1` (strict, descending)    → 4 lines, the implicit-128 split
  - (S1 = 1 line keeps its own `setlist` tag and skips the divider eval entirely.)
  - **`hi>128` is strict** (M5): `hi==128` would make line 2 = `[128,128)` a permanently-empty
    wasted line. Operands `d0,d1,byte` are **`uint8_t`** (N1); only these four shapes are ever
    stored — a debug-assert rejects any other `(d0,d1)` (N2).
- **Line selection — ONE branchless formula for all of S2/S3/S4** (no data-dependent branch,
  ~6 independent compares + masks, verified at every boundary):
  ```
  line = (d0 != 0  & byte >= d0)        // first divider (0 ⇒ inactive)
       + (d1 != 0  & byte >= d1)        // second divider
       + (d0 >  d1 & byte >= 128)       // S4 only: implicit-128 split (hi,lo stored hi>lo)
  ```
  `nlines` (alloc/iteration) = `1 + (d0!=0) + (d1!=0) + (d0>d1)` — falls out the same way.
- **Common prefix (`SDIV_PREFIX_CAP = 13`):** a prefix ≤13 B is **replicated into every line's
  `prefix[]`** (req 2) so the one routed line already holds it (still 1 line/level); the copies are
  kept identical on every change (replication invariant, N7 / §7). A prefix >13 B goes to a
  **tail cacheline** (req 3). sdiv lines therefore size `prefix[13]` (not the setlist's `[7]`).
- **Term** (key ending at this router): in **line 0's** `rh.term`; the other lines' `has_term=0`.
- Dividers are chosen at split time to balance branch counts per line (median / terciles),
  and re-checked so no line exceeds its CAP (else promote a tier).

> Budget (M8, recomputed from the real formula `sizeof(rh)=round_up(2+W,2)`, `prefix[13]`):
> `CAP = floor((128 − round_up(2+W,2) − 1 − 16 − 13)/W)` ⇒ **W6→15, W7→13, W8→11** per line.
> Tier totals ≈ `nlines × CAP`: S2 ≤ 30, S3 ≤ 45, S4 ≤ 60 at W6 (less at W7/W8). Promote to
> `sparse_full` once the balanced split can't keep every line ≤ CAP.

### 3.4 `sparse_full` / `full` — contiguous compact direct index (wide nodes, >~56 branches)
**Contiguous and header-free (resolved):** the child address is computed from the parent handle
+ key byte ALONE — never visiting a node header (that's the whole point of contiguity). The 256
byte-values split into **16 sub-ranges of 16** (`sub = byte>>4`, `slot = byte&0xF`). The parent
handle's 2-byte metadata is the **16-bit presence vector** (bit `sub` ⇒ that sub-range exists).
Occupied sub-ranges are stored contiguously in rank order:
```
sub  = byte >> 4
present = (presence >> sub) & 1                 // else miss
rank = popcount(presence & ((1<<sub) - 1))
child = *(Ptr*)(node_base + rank*128 + (byte&0xF)*W + RANGE_HDR)   // no header read
```
Each 128 B **range line** holds 16 child handles (16·W) plus the node's **replicated common
prefix** (req 2) in the space the smaller offset frees:
- **W=7** (pool, 5-byte offset + 2 meta): 16·7 = 112 B → 16 free → `[plen:1][prefix:≤15][16 handles]`
  (true max prefix is **15 B**, not 13, N6). The routed range line carries the prefix → 1 line/level.
- **W=8** (direct, 6-byte ptr + 2 meta): 16·8 = 128 B, no room → prefix → **tail cacheline** (req 3).
- **`full`** = `presence == 0xFFFF` ⇒ `rank == sub` identically → **rank-free** (no popcount).
  **The rank-free path is reachable ONLY via the `full` tag** (N3) — `sparse_full` must always
  popcount (a lone sub-range at `sub=15` under rank-free would read 15 lines out of bounds);
  `assert(presence==0xFFFF)` on the full arm.
- grows by splicing a range line at `rank` (realloc the contiguous block); shrinks by dropping an
  emptied range. `node_size` is therefore per-node (computed from `popcount(presence)`).

> **B1 / G′ — RESOLVED (W-dependent):** range line 0 = `[plen:1][prefix:plen][term:W?][16 handles]`.
> At **W=7** there are 16 free bytes after the 112 B of handles, so a term **fits inline** alongside a
> short prefix: `1 + plen + (has_term?W:0) ≤ 16` ⇒ `plen ≤ 8` with a term (≤15 without). Only at **W=8**
> (16·8 = 128 B, zero free) — or W=7 with `plen>8` — does the **term move to the node tail** (same tail
> region as a tail-prefix; a key ending at a 56+-branch router is rare). Branch count = `Σ` per-range `nset`.

### 3.5 (no dedicated prefix node)
There is no `prefix_node` type. Every shared run is carried by the router it precedes (§2:
inline / replicated / tail-appended cachelines) or, for a lone key, by that key's leaf suffix.
A run longer than the 4 KB tail cap (pathological) is split: the router consumes the first
4 KB as its tail prefix and its single child carries the remainder the same way — a chain of
tail-prefixed routers, never a distinct node kind.

---

## 4. Indexing kernel (shared, branchless) — already specced/measured in `node_addr.hpp`

`child_line(kind, metadata, byte)` → line index; per §3.3/§3.4. Measured cost on the
latency-bound descent: sdiv ≈0–0.7 ns, sparse_full popcount ≈3.9 ns arm64 / ~free x86, full
rank-free. In-line exact-match is the NEON parallel compare (`vceqq`+`shrn`+`ctz`) — beat
scalar lower_bound (re-load), SWAR `find_byte`, and the serializing half-select.

---

## 5. Inline packing (req 4) — DO NOT REGRESS v1's win

**The mechanism — an intra-node branch handle.** A `branch[i]` slot normally holds a *global*
handle (address/index + tag + meta). When the parent's `inl` byte flags bit `i`, that slot is
instead an **offset into the parent's own 128 B line**: descent resolves it as `line_base +
offset` — **not** a global deref — and the child's bytes live in the line's contiguous inline
region `[&branch[nbranch], line+128)`. The slot's tag still gives the child `kind`, so an inline
child can be a leaf OR a small router. This is exactly v1's `inl` (bit 7 = inline mode active;
bits 0..6 = which branches are inline), kept and generalized. **Effect: 2–5 small children
(leaves and/or tiny setlists) live in ONE cacheline, and descending into any of them costs no
additional line** — the parent line is already in registers/L1.

Two cases:

1. **Inline leaf children (v1, kept).** A small leaf (suffix + value) sits in the inline region;
   `branch[i]` is its intra-node offset (`inl` bit i). A key ending under it costs no extra line.

2. **Inline sub-routers (req 4 — the generalization).** When the children are themselves tiny
   setlists (2–3 branches), the parent + those sub-setlists are packed into ONE line as a single
   logical multi-level node: `branch[i]` offsets to a mini-setlist in the inline region, whose own
   `branch[]` slots are in turn inline offsets or global handles. Descent resolves every packed
   level from the single loaded line — no second cacheline. Predicate: prefix-less parent,
   term-less prefix-less ≤3-branch children, total fits one line with headroom; canonical re-pack
   (v1 `sl_repack_`) after every structural change.

The descent's child-resolve is therefore: `addr = (inl & (1<<i)) ? line_base + off(branch[i])
                                                                  : deref(branch[i])` — one
branchless select on the already-loaded `inl` byte.

> **D — RESOLVED:** intra-node offsets **and** inline sizes are **1 byte** — a 128 B line means any
> in-line position or length fits 0..127 in 7 bits. An inline `branch[i]` holds a 1-byte offset
> into the line's inline region; an inline child's own length (leaf `suffix_len`, sub-setlist
> `nbranch`) is likewise a single byte. Reuses v1's tail-offset slot encoding directly.

---

## 6. Descent (1 line / level)

```
load parent line → child handle h (tag+metadata+addr, in register)
switch (h.kind):
  value_ptr/value_inline : terminal — compare suffix / unpack value
  <any router>           : [consume prefix: inline | replicated-in-line | tail (prefetch+check)]
                           if depth == key.size():                         // key ends AT this router (M9)
                               return has_term ? visit(term) : miss        // never read key[depth] here
                           byte = key[depth]
  setlist                : scan line for byte ; depth++
  sdiv                   : li = child_line(sdiv, meta, byte) ; scan line li for byte ; depth++
  sparse_full/full       : sub=byte>>4 ; present? else miss ; child = lines[rank].ptr[byte&0xF] ; depth++
```
**M9: the key-exhausted-at-router arm is mandatory** — before reading `key[depth]` as a routing
byte, check `depth==key.size()` and return the node's `term` (or miss). find / lower_bound / both
iterators depend on it. **Iterator contract:** forward emits a router's `term` *before* its
children; reverse emits it *after* (the node's final reverse step); `rbegin()`/`--end()` cross-checked.
Every router arm otherwise loads exactly one branch line; the tail-prefix arm overlaps the prefix
cold miss with the branch-line prefetch (§2.3). The chosen `branch[i]` resolves through the §5
inline select — `(inl>>i)&1 ? line_base + off(branch[i]) : deref(branch[i])`.

---

## 7. Transitions (insert-side)

- **leaf split** → setlist (S1); a shared run goes inline (≤7), and a *long* shared run makes
  the new setlist a **tail-prefix** node (§2.3) — never a `prefix_node`.
- **setlist overflow** (>CAP) → **sdiv2** (1 median divider, 2 lines), replicating the prefix
  (req 2). Further overflow → sdiv3 → sdiv4 (dividers at quantiles, balanced).
- **sdiv4 full** (≈56) or distribution dense → **sparse_full**; all-present → **full**.
- **prefix grows past inline/replication budget** → set `tail_prefix`, move prefix to the tail.
- **inline fusion** fires after a split when the parent+children fit one line (§5.2).
- **remove** mirrors: drop branch; collapse a 1-branch line; de-sdiv / de-sparse when sparse;
  re-inline when a child shrinks to fit; pull a tail prefix back inline when it fits again.
- **Prefix replication invariant (N7):** every sdiv line / sparse_full range carries an *identical*
  copy of the node's common prefix. Any transition that adds/removes a line MUST re-stamp the prefix
  into the new copy (and may shorten/lengthen all copies together); a debug build asserts all copies
  are equal after every structural change. A partial update is a silent correctness bug.

---

## 8. Build order (each lands green against the v1 behavioral gate on `v2::`)

1. compact `node_full` → `sparse_full`/`full` (§3.4) — the memory win, no new tag.
2. `sdiv` tier (§3.3) replacing the `c2/c4/c8` ladder — the branchless compare-index + req 2.
3. inline sub-setlist fusion (§5.2) — req 4.
4. tail-prefix mechanism (§2.3, §4) — req 3.
5. handle-carried metadata / headerless (the address-field move) — the last find-speed step.

---

## Open decisions for review
All RESOLVED except D. (Verification verdict 2026-06-14: kernels empirically proven; 4 blockers +
6 majors in the spec text fixed inline — B1 term→tail §3.4, B2 5-bit tag/recomputed pools §1, B3
per-W CAP §3.2, B4+M10 plen-in-meta §1/§2.3, M5 `hi>128` §3.3, M6 honest 1-line/level §0/§2.3, M7
value_inline=N-1 §1, M8 recomputed sdiv CAP §3.3, M9 term descent arm §6, N1–N7 noted in place.)
- **A. RESOLVED** — no `prefix_node`, no `bucket` (§1, §3.5).
- **B. RESOLVED** — tail `plen` is a `u16` in `metadata` (register) when `tail_prefix` set (§1, §2.3).
- **C. RESOLVED** — `sparse_full` contiguous + header-free; 16-value sub-ranges (§3.4).
- **E. RESOLVED** — one `sdiv` tag, branchless 4-tier decode, `SDIV_PREFIX_CAP=13` (§3.3).
- **G/G′. RESOLVED** — `sdiv` term in line 0; `sparse_full`/`full` term in the node tail (B1, §3.4).
- **D. RESOLVED** — intra-node offsets + inline sizes are **1 byte** (128 B line ⇒ 7-bit positions);
  reuse v1's tail-offset slot encoding (§5).

**All decisions resolved — spec frozen, build-ready.**

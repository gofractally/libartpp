<p align="center"><img src="docs/logo.svg" width="140" alt="libart++"></p>

# libart++ — `artpp::map`, a header-only adaptive radix tree map for C++23

**[Website & benchmark charts](https://gofractally.github.io/libartpp/)** ·
with the flagship `line_pool` allocator (anonymous mapping, u32-index handles,
two regions): point lookups **1.2–2.0× faster than upstream libart on all four
workloads** (plain radix — 2.0× sequential, 1.37× uniform, 1.32× clustered,
1.18× dictionary), 1.9–4.3× faster than `absl::btree_map`, and 2.1–7.6× faster
than `std::map`. The invariant behind the numbers: **at
most one cold cacheline per level of descent** — headerless tagged-pointer
dispatch; prefixes fused into the node's own header line; small values inline
in the handle; a small leaf living **inside the routing node's cacheline** (the
`inl` byte); and larger leaves in a dedicated **16-byte-aligned terminal
region** (a terminal is pure payload — it doesn't pay for a 128-byte line).

`artpp::map<Key, T>` is an ordered associative container backed by an adaptive
radix tree (ART) with *headerless dispatch*: the node kind rides in the tagged
pointer, so descent never loads a header byte before deciding how to route. Keys
are stored as their encoded bytes along the tree paths; values live inline in the
pointer when small and trivially copyable, otherwise in compact leaf allocations.

It offers the familiar `std::map`-style surface — ordered iteration, bounds,
`find`/`erase`/`at`/`operator[]`, allocator awareness — with radix-tree
complexity: every operation is **O(key length)**, independent of element count,
with no comparisons and no rebalancing.

```c++
#include "artpp/map.hpp"

artpp::map<std::string_view, uint64_t> t;
t.insert("alpha", 1);             // insert or assign; true iff newly inserted
t.emplace("beta", 2);             // insert only if absent (never overwrites)
t.try_emplace("beta", 99);        // present → untouched, nothing constructed

if (auto it = t.find("alpha"); it != t.end())
    use(it->first, it->second);

for (const auto& [key, value] : t)         // ascending key order
    std::println("{} = {}", key, value);

t.erase("beta");
```

String-keyed maps take `std::string_view` arguments regardless of the owning
key type — `artpp::map<std::string, T>` and `artpp::map<std::string_view, T>`
share one parameter signature, so literals, `char*`, views, and strings all
pass with **no temporary `std::string` ever constructed**:

```c++
artpp::map<std::string, int> owning;   // keys stored/decoded as std::string
owning.insert("alice", 1);             // literal: no temporary
std::string_view sv = user_input();
owning.contains(sv);                   // view: no temporary
```

Integral keys work out of the box and iterate in numeric order:

```c++
artpp::map<int64_t, std::string> m;     // big-endian, sign-flipped encoding:
m.insert(-5, "neg");                      // byte order == numeric order
m.insert(7, "pos");                       // iteration yields -5 before 7
```

## Contents

- [Building and integrating](#building-and-integrating)
- [Requirements](#requirements)
- [Template parameters](#template-parameters)
- [Key types and `key_codec`](#key-types-and-key_codec)
- [Arbitrary key types via psio](#arbitrary-key-types-via-psio)
- [API reference](#api-reference)
- [Iterators](#iterators)
- [Element access and inline values](#element-access-and-inline-values)
- [Whole-tree operations](#whole-tree-operations)
- [Allocators](#allocators)
- [Limits and guarantees](#limits-and-guarantees)
- [Performance](#performance)
- [Benchmark suite](#benchmark-suite)
- [Testing](#testing)

## Building and integrating

Header-only — copy `include/artpp` or use CMake:

```sh
cmake -B build && cmake --build build && ctest --test-dir build
```

As a dependency (`FetchContent` or `add_subdirectory`), link `artpp::artpp`:

```cmake
FetchContent_Declare(artpp GIT_REPOSITORY https://github.com/gofractally/libart++.git)
FetchContent_MakeAvailable(artpp)
target_link_libraries(your_target PRIVATE artpp::artpp)
```

| CMake option | Default | Effect |
|---|---|---|
| `ARTPP_TESTS` | ON when top-level | The functional suite, run by `ctest` |
| `ARTPP_BENCHMARKS` | OFF | The benchmark/compare suite: vendors upstream libart, uses `absl::btree_map` (installed or fetched) |
| `ARTPP_WITH_PSIO` | OFF | The [psio](https://github.com/gofractally/psio) key-format codec adapter and its test |
| `ARTPP_PSIO_SOURCE_DIR` | — | Path to an existing psio checkout (skips the fetch) |

## Requirements

- C++23 (`std::byteswap`, `requires`-clauses, `if constexpr`).
- A little-endian target (statically asserted) with 48-bit virtual addresses for
  the default 6-byte handle; an allocator can select 8-byte handles to lift the
  VA assumption ([Allocators](#allocators)).
- NEON (ARM64) or SSE2 (x86-64) for the SIMD router probes; both paths are
  provided and selected automatically.
- No dependencies beyond the standard library. Header-only:
  `#include <artpp/map.hpp>` (and `artpp/pool.hpp` for the optional line pool).

## Template parameters

```c++
template <class Key, class T,
          artpp::mode Mode         = artpp::mode::none,
          class       Allocator    = std::allocator<T>,
          std::size_t ExpectedSize = 0>
class map;
```

| Parameter | Meaning |
|---|---|
| `Key` | Any type with a `key_codec` specialization. Built in: `std::string`, `std::string_view`, `std::vector<uint8_t>`, all integral types. |
| `T` | The mapped type. Move-constructible required; copy/equality/default-constructibility are required only by the operations that use them. |
| `Mode` | Structural policy flags, combinable with `\|` (below). |
| `Allocator` | A standard allocator. Every node and every value is allocated and constructed through it ([Allocators](#allocators)). |
| `ExpectedSize` | Optional capacity hint. When small (`0 < ExpectedSize ≤ ~3M`) it auto-selects the compact cnode density ladder — a perf-per-byte win for small maps (see [`compact_map`](#compact_map-capacity-hint)). `0` (unknown) leaves the fast flat default. |

### Modes

| Flag | Effect |
|---|---|
| `mode::none` | Pure radix: leaves, prefix nodes, and routers that widen `setlist → node_full`. The fastest default for point lookups. |
| `mode::buckets` | Terminals are *buckets* — packed (suffix, value) pages whose index is kept in total suffix order by insertion, collapsing sparse or deep key tails into one node. Routers appear only when a bucket overflows and bursts. The preferred mode for clustered string keys: on the 236k-word dictionary it runs point lookups ~1.16× faster than the default radix (77 vs 89 ns) and ordered scans ~2.9× faster (2.6 vs 7.4 ns/element — contiguous pages out-scan pointer-chased leaves), and its scans even rival `absl::btree_map` on clustered keys. Buckets cost nothing when off: the engine dead-strips from other modes. |
| `mode::adaptive` | Pure radix that watches its own descent: when leaf splits keep landing under deep, narrow paths, those split points collapse to buckets. A middle ground that keeps uniform data bucket-free. |
| `mode::dense_tiers` | Forces the segmented-bitset cnode density ladder on (`setlist → c2 → c4 → node_full`): sparse fan-out routers stay compact cnodes instead of a sparse 14-line `node_full`. On the `line_pool` this is a **perf-per-byte win for small maps** — at ≤~2M keys ~40% less RAM and ~2× faster build — but the gain fades as the map grows (fan-out nodes densify toward `node_full` and leaves come to dominate the footprint), and past ~3M it only adds build churn + pool fragmentation. Usually reached via the `ExpectedSize` hint / [`compact_map`](#compact_map-capacity-hint) rather than set by hand. |
| `mode::flat_full` | Forces the ladder **off** (`setlist → node_full` direct) even when a capacity hint would enable it — the plain-radix behavior, for find-only or large maps. Same as the default `mode::none`. |
| `mode::ladder_c8` | Adds the `c8` rung to the ladder (`… → c4 → c8 → node_full`). Off by default — on bimodal data `c8` is ~never the final tier — and kept for testing adversarial mid-density-clustered fan-outs. Only meaningful combined with the ladder. |
| `mode::wide_stems` | Sparse routers consume **two** key bytes per hop (`setlist_u16`), halving router-hop depth on deep sparse trees; nodes re-stride back to one-byte routers when they densify. |

All modes share the same external semantics; the flags select internal
representations only. Every mode is exercised against `std::map` by the test
suite.

### `compact_map` (capacity hint)

```c++
template <class Key, class T, std::size_t ExpectedN, class Allocator = std::allocator<T>>
using compact_map = artpp::map<Key, T, artpp::mode::none, Allocator, ExpectedN>;

artpp::compact_map<std::string_view, uint64_t, 500'000> small;  // → cnode density ladder
```

Declaring roughly how many keys you expect lets the map pick the better
internal representation automatically. The default `map` widens sparse fan-out
routers straight to a 14-line `node_full`; at small scale those nodes are
genuinely sparse (~30 of 256 branches), so on the `line_pool` a `compact_map`
stores them as 3–5-line cnodes instead — **~40% less RAM and ~2× faster to
build below ~3M keys**, with point/ordered lookups within noise. Above the
measured crossover the hint transparently falls back to the flat default
(the ladder would only add build churn and pool fragmentation once fan-out
nodes saturate toward `node_full` and leaf storage dominates the footprint).
The crossover is a measured constant (`ladder_capacity_max`); for explicit
control use `mode::dense_tiers` (force on) or `mode::flat_full` (force off).

## Key types and `key_codec`

The radix orders keys by their **encoded bytes** (lexicographic, unsigned).
`artpp::key_codec<Key>` is the customization point that maps a typed key to those
bytes and back:

```c++
template <>
struct artpp::key_codec<MyKey>
{
   // EITHER zero-copy (contiguous keys):
   static constexpr bool   zero_copy = true;
   static std::string_view view(const MyKey& k) noexcept;

   // OR serializing (the default protocol):
   static std::string_view encode(const MyKey& k, std::string& scratch);

   static MyKey decode(std::string_view bytes);   // stored bytes -> key
};
```

- Zero-copy codecs never touch a scratch buffer on the lookup path.
- A serializing codec may define its own `scratch` type; the integral codec uses
  a fixed `char[sizeof(K)]` so no `std::string` machinery runs per operation.
- Integral keys encode fixed-width big-endian with the sign bit flipped for
  signed types, so byte order equals numeric order — iteration, `lower_bound`,
  and `upper_bound` follow numeric semantics exactly.
- `decode` is why iteration returns the key **by value**: a non-string key has
  no stored object to reference; it is rebuilt from the path bytes on demand.

## Arbitrary key types via psio

With `ARTPP_WITH_PSIO`, `<artpp/psio_codec.hpp>` adapts
[psio](https://github.com/gofractally/psio)'s `key` format — memcmp-sortable
encodings for bools, integrals, **doubles** (IEEE sign-transform), strings,
optionals, vectors, variants, and any `PSIO_REFLECT`'ed struct — as an artpp
key codec. Opting in is one line per type:

```c++
#include <artpp/psio_codec.hpp>

struct point { double x, y; };
PSIO_REFLECT(point, x, y)
ARTPP_PSIO_KEY(point)

artpp::map<point, std::string> m;   // ordered by (x, y), numerically
ARTPP_PSIO_KEY(double)
artpp::map<double, int> d;          // doubles as keys, numeric order
```

The opt-in is per type so the built-in codecs (zero-copy string views,
single-byteswap integrals) keep their faster scratch-free paths.

## API reference

`map` is **move-only** (copying a large tree is never an accident; an
explicit copy can be built from a range). All operations below are
O(encoded-key-length) unless noted. `key` parameters take the `Key` type;
cheap keys (`string_view`, integers) pass by value.

### Modifiers

| Member | Semantics | Returns |
|---|---|---|
| `insert(key, v)` / `upsert(key, v)` / `insert_or_assign(key, v)` | Insert, or **assign over the existing value** (all three are the same operation; `insert` deliberately follows ART convention rather than `std::map::insert`'s no-overwrite rule — use `emplace` for that). | `bool` — `true` iff newly inserted |
| `emplace(key, args...)` / `try_emplace(key, args...)` | Insert only if absent. The value is constructed from `args` **exactly once, at its final address**; when the key is present, *nothing* is constructed (stronger than `std::map`, which may construct and discard). | `bool` — `true` iff inserted |
| `update(key, v)` | Assign only if present; never inserts, never restructures. | `bool` — `true` iff assigned |
| `remove(key)` | Erase if present. The structure shrinks on the way out (empty routers freed, single-branch routers collapsed, sparse wide nodes de-widened). | `bool` |
| `erase(key)` | `remove` with the STL shape. | `size_type` (0 or 1) |
| `erase(pos)` | Erase the element at a dereferenceable iterator. Two descents (the key is captured, then the successor re-sought). | `const_iterator` to the next element |
| `clear()` | Destroy every value, free every node. `noexcept`. | — |

Overwrites use `T`'s **assignment operator** (`insert_or_assign` semantics), not
destroy-and-reconstruct, so a throwing assignment leaves a valid object behind.

### Lookup

| Member | Semantics | Returns |
|---|---|---|
| `find(key, T& out)` | Copy the value out. The fastest lookup: one descent, no iterator state. (Copy-out, not a pointer, because small values may live inline in the handle bits.) | `bool` |
| `find_opt(key)` | Copy-out as an optional. | `std::optional<T>` |
| `find(key)` | STL shape. | `const_iterator` |
| `contains(key)` | Presence test; never reads the value (cheaper than `find` for allocating `T`). | `bool` |
| `count(key)` | `contains` with the STL shape. | `size_type` (0 or 1) |
| `at(key)` | Reference access; throws `std::out_of_range` if absent. Non-inline `T` only. | `T&` / `const T&` |
| `find_ptr(key)` | Reference access without the throw. Non-inline `T` only. | `T*` / `const T*` (null if absent) |
| `operator[](key)` | Access, default-inserting if absent. Non-inline, default-constructible `T` only. | `T&` |
| `lower_bound(key)` / `upper_bound(key)` | First element `>= key` / `> key` in encoded-byte order. | `const_iterator` |
| `equal_range(key)` | `{lower_bound, upper_bound}` — at most one element apart (keys are unique). | pair of iterators |

### Traversal

| Member | Semantics |
|---|---|
| `begin() / end() / cbegin() / cend()` | Bidirectional const iteration in ascending key order. `begin()` does not allocate. |
| `rbegin() / rend()` (+`cr` forms) | `std::reverse_iterator` over the same. |
| `for_each(f)` | Recursive visitor `f(std::string_view key_bytes, const T&)` in ascending order — roughly 3× the iterator on cache-resident scans (no per-element resumption). |
| `for_each_value(f)` | `f(const T&)` in ascending key order without materializing keys. |

### Capacity, comparison, management

| Member | Semantics |
|---|---|
| `size() / empty()` | Element count, `noexcept`. |
| `max_size()` | Theoretical bound. |
| `get_allocator()` | The allocator. |
| `swap(other)` / ADL `swap` | O(1) handle swap. Allocators swap iff `propagate_on_container_swap`; otherwise they must compare equal (standard rule). |
| `operator==` / `operator!=` | Content equality, compared **structurally** — node against node, prefixes and edge directories `memcmp`'d, no keys materialized. Falls back to an element-wise sweep only when equal-content trees have divergent shapes (possible after removals; see source). `T` must be equality-comparable only if used. |
| `debug_stats()` | Node-type census and depth diagnostics (`dbg_counts`). |

## Iterators

`const_iterator` (and `iterator`, an alias — the container is read-only through
iterators) is a **bidirectional proxy iterator**:

- `iterator_category` is `std::bidirectional_iterator_tag` in the legacy sense;
  like `std::vector<bool>::iterator`, `operator*` returns a proxy
  (`std::pair<Key, const T&>` — the key **by value**, decoded on demand), so it
  does not model the C++20 `std::bidirectional_iterator` concept. Range-`for`,
  `std::reverse_iterator`, and the classic algorithms (`std::distance`,
  `std::find_if`, `std::prev`, …) all work.
- `it.key()` returns the decoded `Key`; `it.key_bytes()` returns the raw encoded
  bytes (zero-copy when the leaf stores the whole key); `it.value()` returns
  `const T&`.
- Key materialization is **adaptive**: until `key()`/`key_bytes()` is first
  read, `++`/`--` do no key bookkeeping at all (value-only scans pay nothing).
  The first read primes the path; from then on the key is maintained
  incrementally.
- `--end()` seats on the last element (`std::map` semantics); on an empty tree
  it stays `end()`. A default-constructed iterator compares equal to `end()` of
  any tree.
- Equality compares positions structurally — no key comparison.

**Invalidation.** Any structurally mutating operation (`insert`, `emplace`,
`upsert`, `remove`, `erase`, `clear`, `operator[]` on a missing key) invalidates
all iterators, and may invalidate references obtained from
`at`/`find_ptr`/`operator[]` whose leaf or bucket was restructured. `update`
(and assignment to a present key) never restructures, so iterators stay valid —
but note an iterator positioned on a small *inline* value holds a copy made when
it was positioned and will not observe later updates; non-inline values are
always read live. The safe rule: re-acquire after mutation.

## Element access and inline values

A value that is trivially copyable and no larger than `inline_cap`
(`sizeof(handle) - 1`; **5 bytes** with the default 6-byte handle) is stored in
the pointer bits of its branch — zero allocation, zero indirection. Such values
have **no addressable object**, so the by-reference API (`at`, `find_ptr`,
`operator[]`) is constrained away at compile time (`requires`-clauses); use the
copy-out forms (`find(key, out)`, `find_opt`) instead. `map<K,T>::inlineable`
reports the decision; `uint32_t` inlines, `uint64_t` does not.

For everything else — strings, 8-byte integers, your structs — values are real
objects constructed through the allocator inside leaf (or bucket) storage, and
the reference API hands out genuine `T&` with the invalidation rule above.

## Whole-tree operations

Whole-tree work follows one principle: **operate on the structure, not the
elements.**

- **Move construction** always steals the root — O(1), `noexcept` (if the
  allocator's move is).
- **Move assignment** picks the cheapest sound transport, in order:
  1. *Steal* — allocator propagates (`POCMA`) or compares equal: O(1).
  2. *Image adoption* — index-based allocators that opt in (`artpp_adopt`, e.g.
     `pool_alloc`) with trivially-copyable `T`: the source pool's used range is
     copied with **one `memcpy`** and every handle, including the root, stays
     valid verbatim, because handles are base-relative line indexes.
  3. *Structural clone* — raw-pointer allocators: the tree is rebuilt **node by
     node** (allocate, `memcpy` the node, re-point child handles at cloned
     children; values move-construct object-by-object). Same shape as the
     source, no descents, no per-key re-insertion.
- **Equality** walks both trees node-against-node as described above.
- **Destruction / `clear`** walks once, destroying values and freeing nodes
  through the allocator — nothing is leaked even after exceptions (verified by
  fault-injection tests).

## Allocators

`map` is allocator-aware in the standard sense: nodes are 128-byte-aligned
block runs taken from a rebound copy of your allocator, values are constructed
and destroyed via `allocator_traits`, and the container honors the propagation
traits on move, swap, and assignment. PMR works as expected.

Beyond the standard protocol, an allocator can customize the tree through three
**opt-in members** (detected, never required):

| Member | Effect |
|---|---|
| `using artpp_handle = artpp::packed_ptr_t<N>;` | Selects the branch-handle width. Default is the 6-byte packed pointer (48-bit VA); `packed_ptr_t<8>` stores full pointers and drops the VA assumption. |
| `using artpp_handle = artpp::detail::line_ptr;` + `artpp_base()` | Index-based handles: a 4-byte handle holding a 28-bit *line index* resolved against `artpp_base()`. Branches cost 4 bytes and the whole store becomes one relocatable range. |
| `artpp_adopt(const A& src)` | Opts into bulk image adoption for cross-pool move assignment (see above). Providing it asserts the pool backs exactly one tree. |

### `artpp::line_pool`

`artpp/pool.hpp` ships the reference index-based allocator: one contiguous
reservation of 128-byte lines (up to 32 GB), addressed by 4-byte handles,
committed in 4 MB steps, with exact-size free lists. It can be **anonymous or
file-backed**:

```c++
artpp::line_pool                pool;                       // anonymous memory
artpp::line_pool                disk("mystore");              // file-backed: a store DIRECTORY
using A = artpp::pool_alloc<uint64_t>;
artpp::map<std::string_view, uint64_t, artpp::mode::none, A> t{A{&pool}};
```

File backing is **two things at once**: out-of-core storage (page eviction,
larger-than-RAM data sets) and a **durable store across a clean close/reopen**.
A file-backed pool is a **directory with one dense file per address region** —
`nodes` (128-byte lines; its first page(s) hold a self-describing superblock:
geometry + carving state + the tree's root handle and count) and `terms`
(16-byte terminal units). One file per region means each grows densely from
offset 0 — **no sparse files, no 32 GB logical size, no dependency on sparse-file
support** (so it's safe to `cp`/`tar`/`rsync`/back up). Reopening restores the
prior image with **no relocation**: handles are byte offsets, so the bytes
already *are* the tree.

```c++
// write, then close
{
   artpp::line_pool pool("mystore");                           // reopen-or-create the directory
   artpp::map<std::string_view, uint64_t, artpp::mode::none, A> t{A{&pool}};
   t.insert("key", 42);
   pool.checkpoint(t.root_handle(), t.size());                 // persist root + carving state
   t.detach();                                                 // tree lives on disk; skip teardown
}
// reopen — same process or a later one
{
   artpp::line_pool pool("mystore");                           // pool.restored() == true
   artpp::map<std::string_view, uint64_t, artpp::mode::none, A>
       t{A{&pool}, artpp::attach, pool.root(), pool.count()};  // O(1) attach, no rebuild
   uint64_t v;
   t.find("key", v);                                           // == 42
}
```

`checkpoint(root, count)` stamps the superblock and flushes it; `attach`
rebinds a map to the reopened image in O(1); `detach()` relinquishes the
in-memory tree without freeing so close stays O(1). This is **clean-close**
durability — crash/reboot durability (a full data sync + write-ahead log) is
the next layer: `checkpoint` flushes the header, and data pages ride the page
cache across a clean reopen.

## Limits and guarantees

- **Key length** — at most `max_key_bytes` = **65535** encoded bytes (the node
  formats use 16-bit length fields). Mutating operations throw
  `std::length_error` *before any allocation or structural change*; lookups of
  longer keys simply miss (no stored key can be that long), keeping the hot
  path free of the check.
- **Exception safety** — mutating operations provide the basic guarantee with
  **zero leaks** (fault-injection-tested: every allocation point swept, leak
  and double-destroy counters asserted). Single-leaf inserts and the
  pre-mutation checks are strong. `remove` is nothrow in practice for POD
  values in radix mode (documented per overload in the header).
- **Thread safety** — that of a standard container: concurrent reads on a
  `const` tree are fine; any write requires exclusive access. No internal
  threads, no locks.
- **Ordering** — strictly ascending encoded-byte order; with the integral
  codecs this is exact numeric order, including negatives.
- **The container is move-only**; copy construction/assignment are deleted by
  design.

## Performance

Measured on Apple M5, clang -O3, 1M keys, flagship `line_pool` allocator,
point-lookup latency (fastest of five reps). How many times **faster** than
each baseline (higher is better):

| workload | vs libart | vs `absl::btree_map` | vs `std::map` |
|---|---|---|---|
| dictionary words | **1.18×** | 1.93× | 2.14× |
| clustered strings | **1.32×** | 2.36× | 3.72× |
| uniform `u64` | **1.37×** | 2.89× | 5.47× |
| sequential `u64` | **2.02×** | 4.32× | 7.63× |

Bucket mode is faster still on the dictionary (1.16× over the default radix)
and its scans rival `absl::btree_map`. The headerless tagged-pointer dispatch
is the structural reason reads win: libart loads a node header on every hop
before it can route; artpp routes from the pointer itself, and a small leaf
often rides in the routing node's own cacheline.

**Inserts** are allocator-bound, so the honest comparison holds the tree fixed
and swaps the allocator: `line_pool` builds the same tree **1.1–1.4× faster
than `std::allocator`** (bump placement, children next to parents). Held to the
same malloc class, artpp and libart build at similar rates — artpp leads on
sequential keys, libart's flatter node growth leads on the deeply-shared
dictionary (a radix splits more there). With the pool, builds beat
`absl::btree_map` 1.7–2.8× and `std::map` 2.4–7.2×. Reproduce with
`-DARTPP_BENCHMARKS=ON` (`bench/`); the per-process regression gate lives in
`bench/perf_gate.cpp`.

`for_each` scans are ~3× iterator scans when cache-resident; value-only
iteration (`for_each_value` or an un-keyed iterator walk) is cheaper still.

## Benchmark suite

`-DARTPP_BENCHMARKS=ON` builds `artpp_bench`: **one templated engine**
(`bench/engine.hpp`) drives every contestant through identical code — the same
key vectors, op sequences, and checksum accounting — so the comparison carries
no per-library benchmark deviations. Contestants are thin adapters
(`bench/contestants.hpp`):

- `artpp::map` (and `mode::buckets` for the string workloads)
- upstream [libart](https://github.com/armon/libart), vendored under
  `external/libart` (bytes-only: integer workloads use the same big-endian
  bytes artpp's integral codec produces)
- `absl::btree_map` (installed abseil or fetched; heterogeneous lookup enabled)
- `std::map` (reference row)

Workloads: `dict` (system word list), `clustered` (synthetic shared-prefix
strings), `uniform` and `sequential` (typed `uint64_t`). Ops: insert, point
hit, point miss, full ordered scan, erase. Results print as a table and land
in `results.csv` (one row per contestant × workload × op) for charting.

```sh
cmake -B build -DARTPP_BENCHMARKS=ON && cmake --build build -j
./build/bin/artpp_bench            # [N] [scan_reps] [csv_path]
```

`artpp_perf_gate` is the machine-calibrated artpp/libart ratio regression gate
(`bench/perf_baseline.tsv`); run it alone on a quiet machine.

## Testing

`ctest` runs the functional suite:

| binary | scope |
|---|---|
| `artpp_smoke` | API basics across all modes, allocator routing (counting, PMR, wide-handle, pool), typed keys, `std::map` API-conformance battery, randomized cross-checks |
| `artpp_conformance` | STL-grade contracts: key-length limit, move/swap branches incl. pool image adoption and structural clone, exactly-once construction, move-only values, compile-time constraints, string_view key parameters, `erase(iterator)`, structural equality incl. shape-divergent trees, iterator edge cases, degenerate shapes, mixed-op differential fuzz vs `std::map` (all modes) |
| `artpp_reverse_iter` | Bidirectional iteration fuzz vs `std::map`: churn, full walks both directions, `--end()`, random `lower_bound` + backward steps, `++`/`--` ping-pong |
| `artpp_exception_safety` | Exception safety under systematic fault injection (throwing values + failing allocator at every allocation point), leak/corruption/double-destroy counters vs a `std::map` oracle |
| `artpp_wide_stems` | `setlist_u16` (wide-stem) cross-checks and depth/latency demonstration |
| `artpp_psio_codec` | (with `ARTPP_WITH_PSIO`) double / reflected-struct / optional keys vs `std::map` ordering oracles |
| `artpp_persistence` | file-backed `line_pool` survives close + reopen: 40k-key round-trip through a full teardown, then further insert/erase; restores carving state + root with no relocation |
| `artpp_fuzz_map` | differential fuzzer vs `std::map` — every op mirrored on both, full forward/reverse/bound/size agreement; standalone seeded RNG runner (also replays corpus files) |

### Fuzzing

`artpp_fuzz_map` is dual-mode. As the ctest binary above it is a deterministic, seeded random
runner over `std::map` as the oracle. Built with `-DARTPP_FUZZER=ON` (needs a libFuzzer-capable
clang — e.g. Homebrew `llvm`; the build probes for the runtime and skips otherwise) it is a
**coverage-guided libFuzzer target** under ASan+UBSan:

```sh
cmake -B build -DARTPP_FUZZER=ON && cmake --build build --target artpp_fuzzer
./build/bin/artpp_fuzzer -max_total_time=60        # or pass a corpus dir
./build/bin/artpp_fuzz_map <corpus-or-crash-file>  # replay a reproducer in the default toolchain
```

The first input byte selects the config so one corpus exercises every mode (flat / dense_tiers /
+ladder_c8 / buckets / adaptive / wide_stems), the `line_pool` allocator, and a `std::string` value
(external leaves). Keys are drawn from a small alphabet and a recently-used key pool (extend /
truncate / mutate a prior key) to manufacture the shared prefixes and wide fan-out that drive the
split / widen-ladder / bucket-burst / de-widen transitions; any divergence aborts with the input
bytes for replay.

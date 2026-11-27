<p align="center"><img src="docs/logo.svg" width="140" alt="libart++"></p>

# libart++ — `artpp::map`, a header-only adaptive radix tree map for C++20

**[Website & benchmark charts](https://gofractally.github.io/libartpp/)**

With the default allocator, point lookups are **1.2–2.0× faster than upstream
libart on all four benchmark workloads**, 2.0–4.4× faster than
`absl::btree_map` and 2.4–7.2× faster than `std::map`, in the run described
under [Performance](#performance) (one machine, Apple silicon). The bundled
`line_pool` allocator is faster again on three of the four and slower on one.
The invariant behind the numbers: **at
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
#include <iostream>

artpp::map<std::string_view, uint64_t> t;
t.insert("alpha", 1);             // insert or assign; true iff newly inserted
t.emplace("beta", 2);             // insert only if absent (never overwrites)
t.try_emplace("beta", 99);        // present → untouched, nothing constructed

if (auto it = t.find("alpha"); it != t.end())
    use(it->first, it->second);

for (const auto& [key, value] : t)         // ascending key order
    std::cout << key << " = " << value << '\n';

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
- [Status and versioning](#status-and-versioning)
- [Contributing](#contributing)
- [License](#license)

## Building and integrating

Header-only — copy `include/artpp` or use CMake:

```sh
cmake -B build && cmake --build build && ctest --test-dir build
```

As a dependency (`FetchContent` or `add_subdirectory`), link `artpp::artpp`:

```cmake
include(FetchContent)
FetchContent_Declare(artpp GIT_REPOSITORY https://github.com/gofractally/libartpp.git
                     GIT_TAG main)
FetchContent_MakeAvailable(artpp)
target_link_libraries(your_target PRIVATE artpp::artpp)
```

Or install it and use `find_package`:

```sh
cmake -B build -DARTPP_TESTS=OFF && cmake --install build --prefix /your/prefix
```

```cmake
find_package(artpp 0.1 REQUIRED)
target_link_libraries(your_target PRIVATE artpp::artpp)
```

`<artpp/version.hpp>` defines `ARTPP_VERSION` (and `_MAJOR`, `_MINOR`, `_PATCH`, `_STRING`).

| CMake option | Default | Effect |
|---|---|---|
| `ARTPP_TESTS` | ON when top-level | The functional suite, run by `ctest` |
| `ARTPP_BENCHMARKS` | OFF | The benchmark/compare suite: vendors upstream libart, uses `absl::btree_map` (installed or fetched) |
| `ARTPP_WITH_PSIO` | OFF | The [psio](https://github.com/gofractally/psio) key-format codec adapter and its test (needs C++23 and the Boost headers, as psio does) |
| `ARTPP_FUZZER` | OFF | The coverage-guided libFuzzer target `artpp_fuzzer` ([Fuzzing](#fuzzing)) |
| `ARTPP_PSIO_SOURCE_DIR` | — | Path to an existing psio checkout (skips the fetch) |

## Requirements

- C++20 (`<bit>`, `requires`-clauses, `if constexpr`; `std::byteswap` when available, else a `__builtin_bswap` shim).
- A 64-bit, little-endian target (both statically asserted) with 48-bit virtual addresses for
  the default 6-byte handle; an allocator can select 8-byte handles to lift the
  VA assumption ([Allocators](#allocators)). The assumption is checked at run
  time, once per allocation: a block above 2^48 (top-byte pointer tagging such
  as HWASan or MTE, 5-level paging, 52-bit ARM addresses) makes the allocating
  operation throw `artpp::bad_allocation_address` instead of being truncated.
- No particular instruction set: the router probes use NEON (ARM64) or SSE2
  (x86-64) when the compiler targets them, selected automatically, with a
  scalar fallback (the two-byte wide-stem probe is NEON or scalar).
- No dependencies beyond the standard library. Header-only:
  `#include <artpp/map.hpp>` (and `artpp/pool.hpp` for the optional line pool).
  The optional psio adapter is the exception: it needs psio, and with it C++23
  and the Boost headers.
- A POSIX system for `artpp::line_pool` (`mmap`); the map itself with
  `std::allocator` or any other allocator does not need it.
- Tested in CI with GCC and Clang on x86-64 Linux and Apple Clang on ARM64
  macOS, each also under AddressSanitizer and UndefinedBehaviorSanitizer, and
  with GCC on ARM64 Linux (no sanitizers). The psio adapter, the benchmarks and
  the libFuzzer target are not built in CI. MSVC is not supported.

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
| `Key` | Any type with a `key_codec` specialization. Built in: `std::string`, `std::string_view`, `std::vector<uint8_t>`, all integral types (including `bool`, which orders `false` before `true`, and the character types). |
| `T` | The mapped type. Move-constructible required; copy/equality/default-constructibility are required only by the operations that use them (default construction: `operator[]` and `find_opt`). `alignof(T)` up to 128 ([Limits and guarantees](#limits-and-guarantees)). |
| `Mode` | Structural policy flags, combinable with `\|` (below). |
| `Allocator` | A standard allocator. Every node and every value is allocated and constructed through it ([Allocators](#allocators)). |
| `ExpectedSize` | Optional capacity hint. When small (`0 < ExpectedSize ≤ 3,000,000`) it auto-selects the compact cnode density ladder — a perf-per-byte win for small maps (see [`compact_map`](#compact_map-capacity-hint)). `0` (unknown) leaves the fast flat default. |

### Modes

| Flag | Effect |
|---|---|
| `mode::none` | Pure radix: leaves, prefix nodes, and routers that widen `setlist → node_full`. The fastest default for point lookups. |
| `mode::buckets` | Terminals are *buckets* — packed (suffix, value) pages whose index is kept in total suffix order by insertion, collapsing sparse or deep key tails into one node. Routers appear only when a bucket overflows and bursts. The preferred mode for clustered string keys: on the 236k-word dictionary (the run described under [Performance](#performance)) it runs point lookups ~1.14× faster than the default radix (78 vs 89 ns) and ordered scans ~2.8× faster (2.6 vs 7.3 ns/element — contiguous pages out-scan pointer-chased leaves), and its scans even rival `absl::btree_map` on clustered keys. Buckets cost nothing when off: the engine dead-strips from other modes. |
| `mode::adaptive` | Pure radix that watches its own descent: when leaf splits keep landing under deep, narrow paths, those split points collapse to buckets. A middle ground that keeps uniform data bucket-free. |
| `mode::dense_tiers` | Forces the segmented-bitset cnode density ladder on (`setlist → c2 → c4 → node_full`): sparse fan-out routers stay compact cnodes instead of a sparse `node_full` (14 lines with the default handle, 10 on the `line_pool`). On the `line_pool` this is a **perf-per-byte win for small maps** — below ~3M keys ~40% less RAM and ~2× faster build, as measured by `artpp_widen_probe` — but the gain fades as the map grows (fan-out nodes densify toward `node_full` and leaves come to dominate the footprint), and past ~3M it only adds build churn + pool fragmentation. Usually reached via the `ExpectedSize` hint / [`compact_map`](#compact_map-capacity-hint) rather than set by hand. |
| `mode::flat_full` | Forces the ladder **off** (`setlist → node_full` direct) even when a capacity hint would enable it — the plain-radix behavior, for find-only or large maps. Same as the default `mode::none`. |
| `mode::ladder_c8` | Adds the `c8` rung to the ladder (`… → c4 → c8 → node_full`). Off by default — on bimodal data `c8` is ~never the final tier — and kept for testing adversarial mid-density-clustered fan-outs. Only meaningful combined with the ladder. |
| `mode::wide_stems` | Sparse routers consume **two** key bytes per hop (`setlist_u16`), halving router-hop depth on deep sparse trees; nodes re-stride back to one-byte routers when they densify. |

All modes share the same external semantics; the flags select internal
representations only. The default, `buckets`, `adaptive`, `dense_tiers` (with
and without `ladder_c8`) and `wide_stems` are each exercised against `std::map`
by the test suite; `flat_full` and the capacity hint only choose between two of
those representations and have shape checks, not differential runs of their
own.

### `compact_map` (capacity hint)

```c++
template <class Key, class T, std::size_t ExpectedN, class Allocator = std::allocator<T>>
using compact_map = artpp::map<Key, T, artpp::mode::none, Allocator, ExpectedN>;

artpp::compact_map<std::string_view, uint64_t, 500'000> small;  // → cnode density ladder
```

Declaring roughly how many keys you expect lets the map pick the better
internal representation automatically. The default `map` widens sparse fan-out
routers straight to a `node_full` (14 lines with the default handle, 10 on the
`line_pool`); at small scale those nodes are genuinely sparse (~30 of 256
branches), so on the `line_pool` a `compact_map` stores them as 2–4-line
cnodes instead (3–5 lines with the default handle) — **~40% less RAM and ~2×
faster to build below ~3M keys**, with point/ordered lookups within noise
(measured with `artpp_widen_probe`, `bench/widen_probe.cpp`). Above the
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
- `decode` is why iteration returns a `Key`, not a `const Key&`: a key has no
  stored object to reference; it is rebuilt from the path bytes on demand. (The
  built-in `std::string_view` codec decodes to a view of those bytes, which is
  why such a key is only valid while its iterator stays put — see
  [Iterators](#iterators).)

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

The adapter needs what psio needs: **C++23 and the Boost headers**
(Boost.Preprocessor, for `PSIO_REFLECT`). The rest of the library stays C++20
with no dependencies. The adapter and its test are not built in CI.

## API reference

`map` is **move-only** (copying a large tree is never an accident; to copy,
iterate and `insert` into a new map — there is no range constructor). All operations below are
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
| `begin() / end() / cbegin() / cend()` | Bidirectional const iteration in ascending key order. `begin()` does not allocate unless the first element is more than 16 nodes deep. |
| `rbegin() / rend()` (+`cr` forms) | The same in descending order, through the container's own `const_reverse_iterator` (see [Iterators](#iterators)). |
| `for_each(f)` | Recursive visitor `f(std::string_view key_bytes, const T&)` in ascending order — faster than the iterator on cache-resident scans (no per-element resumption; not benchmarked in the suite). |
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
  (`std::pair<Key, const T&>` — the key is not stored as an object, it is
  **decoded on demand** from the path bytes), so it does not model the C++20
  `std::bidirectional_iterator` concept. Range-`for` and the classic algorithms
  (`std::distance`, `std::find_if`, `std::prev`, …) work.
- **What an iterator hands out lives as long as the iterator stays put.** An
  owning key (`std::string`, an integer, a struct) is a copy and is yours. A
  `std::string_view` key — from `*it`, `it->first`, `it.key()` or
  `it.key_bytes()` — views the iterator's key buffer, and a reference to a small
  *inline* value ([below](#element-access-and-inline-values)) refers to the
  copy the iterator holds: both are valid **until that iterator is advanced or
  destroyed**. Copy the key (`std::string(it.key())`) to keep it longer.
- `rbegin()` / `rend()` return the container's own `const_reverse_iterator`
  (`reverse_iterator` is an alias), for `map` and for `set`. It holds the
  iterator positioned **on** the element, so the rule above applies to it
  unchanged: `*r`, `r->`, `r.key()`, `r.key_bytes()` and `r.value()` are valid
  until `r` is advanced or destroyed. It is bidirectional, works with the
  classic algorithms, and `r.base()` has the standard meaning (the iterator one
  past the element; `rbegin().base() == end()`). Do **not** wrap the iterators
  in `std::reverse_iterator` yourself: that adaptor dereferences a temporary
  copy, so a `string_view` key or an inline value obtained through it dangles.
- `it.key()` returns the decoded `Key`; `it.key_bytes()` returns the raw encoded
  bytes (zero-copy when the leaf stores the whole key); `it.value()` returns
  `const T&`. The lifetime rule above applies to all three.
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
`at`/`find_ptr`/`operator[]` whose leaf or bucket was restructured. `update`,
and `insert` / `insert_or_assign` / `upsert` / `emplace` on a key that is
**already present**, never restructure — no node is allocated, freed or moved —
so iterators and references stay valid, in every mode;
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
reports the decision; `uint32_t` inlines, `uint64_t` does not. That is with the
default handle: under `line_pool`'s 4-byte handle the cap is 3 bytes, so a
`uint32_t` is a real object there and the reference API is available.

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

The rebound allocator must honour the alignment of the type it is asked for:
128 bytes for nodes and `max(16, alignof(T))` for leaves (`std::allocator` and
PMR do; a `malloc`-backed allocator that ignores `alignof` does not). The tree
checks every block it receives — alignment, and that the address fits the
handle — and throws `artpp::bad_allocation_address` (derived from
`std::bad_alloc`) after returning the block, so a non-conforming allocator is
reported on the first allocation rather than corrupting the tree. The check is
on the allocation path only; lookups are unaffected.

Beyond the standard protocol, an allocator can customize the tree through
**opt-in members** (detected, never required):

| Member | Effect |
|---|---|
| `using artpp_handle = artpp::packed_ptr_t<N>;` | Selects the branch-handle width. Default is the 6-byte packed pointer (48-bit VA); `packed_ptr_t<8>` stores full pointers and drops the VA assumption. |
| `using artpp_handle = artpp::detail::line_ptr;` + `artpp_base()` + `artpp_term_base()` | Index-based handles: a 4-byte handle holding a 28-bit *line index* resolved against `artpp_base()` (nodes) or a 16-byte-aligned offset resolved against `artpp_term_base()` (leaves; the two may be the same address). Branches cost 4 bytes and the whole store becomes one relocatable range. |
| `artpp_adopt(const A& src)` | Opts into bulk image adoption for cross-pool move assignment (see above). Providing it asserts the pool backs exactly one tree. |

### `artpp::line_pool`

`artpp/pool.hpp` ships the reference index-based allocator: two contiguous
reservations — 128-byte node lines (up to 32 GB) and 16-byte terminal units
for leaves (up to 4 GB, one eighth of the node capacity) — addressed by 4-byte
handles, committed in 4 MB steps, with exact-size free lists. It can be **anonymous or
file-backed**:

```c++
artpp::line_pool                pool;                       // anonymous memory
artpp::line_pool                disk("mystore");              // file-backed: a store DIRECTORY
using A = artpp::pool_alloc<uint64_t>;
artpp::map<std::string_view, uint64_t, artpp::mode::none, A> t{A{&pool}};
```

File backing is **two things at once**: out-of-core storage (page eviction,
larger-than-RAM data sets) and a **store that survives a clean close and
reopen**. A file-backed pool is a **directory with one dense file per address
region** — `nodes` (128-byte lines) and `terms` (16-byte terminal units), each
behind a 64 KB header. One file per region means each grows densely from
offset 0 — **no sparse files, no 32 GB logical size, no dependency on
sparse-file support**. Reopening restores the image with **no relocation**:
handles are offsets, so the bytes already *are* the tree.

```c++
using Map = artpp::map<std::string_view, uint64_t, artpp::mode::none, A>;
{
   artpp::line_pool pool("mystore");     // open or create the store directory
   Map t{A{&pool}, artpp::attach};       // the checkpointed tree; empty on a new store
   t.insert("key", 42);
   t.checkpoint();                       // record the tree as the store's content
}                                        // the map and the pool close; the tree stays
{
   artpp::line_pool pool("mystore");     // a later run: pool.restored() == true
   Map t{A{&pool}, artpp::attach};       // O(1), no rebuild
   uint64_t v;
   t.find("key", v);                     // == 42
}
```

#### Lifecycle

**open → attach → use → `checkpoint()` → close.**

1. `line_pool pool(dir)` opens the store in `dir`, or creates one if there is
   none (no files, or two empty ones). It takes an exclusive lock for as long
   as the pool lives. `pool.restored()` says whether a checkpointed tree was
   found.
2. `Map t{alloc, artpp::attach}` binds a map to that tree in O(1) — or to an
   empty tree, on a store that was never checkpointed.
3. `t.checkpoint()` records the tree as the store's content. By default it
   forces nothing to disk; see [Sync level](#sync-level).
   **What a later open sees is what the last `checkpoint()` recorded.**
4. Close by destroying the map and the pool, in that order. **Destroying a map
   of a trivially destructible mapped type never tears down a tree in a
   file-backed pool**: the tree belongs to the store, so the destructor leaves
   it in place (and is O(1)). It does not
   matter whether the map is destroyed before or after the checkpoint.
   `clear()` is how to give a tree's space back to the pool.

Rules that follow from this:

- **Checkpoint last.** If the tree is changed after the last `checkpoint()` and
  the store is closed — or the process dies — the image is neither the old tree
  nor the new one, and the next open **refuses it** (`artpp::store_dirty`).
  There is no way to open such a store; `pool.dirty()` tells you beforehand
  whether you are in that state. A store that was *never* checkpointed is the
  exception: it has nothing to lose and reopens empty.
- **One store, one tree, one map type.** The store remembers which
  instantiation wrote it, and attaching another is refused. Other maps may use
  the same pool as scratch space; `clear()` them before the last checkpoint (a
  scratch map that is destroyed with content leaves its nodes allocated in the
  store, and one destroyed after the checkpoint marks the store as modified).
- `attach` and `checkpoint()` need a trivially copyable mapped type: the image
  stores its bytes. (A map whose mapped type has a non-trivial destructor can
  still *use* a file-backed pool as out-of-core memory; its destructor destroys
  the values and frees the nodes. A mapped type that is trivially destructible
  but not trivially copyable is left in place like any other and cannot be
  attached: `clear()` such a map before destroying it.)
- `line_pool` is neither copyable nor movable; a map on it moves as usual and
  takes the tree, and the right to checkpoint it, along.
- `pool.close()` does what the pool's destructor does and reports a failure,
  which a destructor cannot. It does not checkpoint.

`pool.checkpoint(root, count)` with `map::root_handle()` / `size()`, the
four-argument `attach` constructor and `detach()` are the low-level calls
underneath. Prefer the calls above: `checkpoint(root, count)` records which map
type the tree belongs to only if a map attached in the same session.
`detach()` is needed only to hand a tree over within an anonymous pool.

#### What is checked on open

An existing store is validated before anything is mapped or written; a store
that fails is left byte for byte as it was, and nothing is created beside it.

| Refused with | When |
|---|---|
| `artpp::store_corrupt` | Not a store, or not a whole one: a foreign file, a missing, empty or truncated file (sizes are checked against what the checkpoint committed), a superblock field outside the range the others allow (capacities, committed sizes, frontiers, free-list heads, the root handle), or `nodes` and `terms` that are not from the same store and the same checkpoint. |
| `artpp::store_incompatible` | A store this program cannot use: another format version, byte order, line size, handle layout or superblock layout; a header reservation the system's page size cannot map; or — thrown by `attach` — a different map instantiation: size or alignment of the mapped type, mode, handle width, map versus set, or key encoding. |
| `artpp::store_dirty` | Modified after its last checkpoint (see above). |
| `artpp::store_locked` | Open in another `line_pool`, in this process or another (`flock`). |
| `std::system_error` | A system call failed (`open`, `mmap`, `ftruncate`, `flock`, `msync`, …); it carries the `errno` and names the file. |

The first four derive from `artpp::store_error`, a `std::runtime_error`, and
name the store and the reason in `what()`. While a pool is in use, running out
of space throws `artpp::pool_exhausted` and a failure of the system to grow it
`artpp::pool_commit_failed` (with the `errno`); both are `std::bad_alloc`, so
the tree's exception guarantees apply.

#### Sync level

`checkpoint()` takes an `artpp::sync_level`, on the map and on the low-level
`pool.checkpoint(root, count, level)`:

| Level | What it does | Cost | Survives |
|---|---|---|---|
| `sync_level::none` (default) | Records the image through the shared file mapping; the operating system writes it back when it chooses. | Writing two headers. | The process exiting or being killed. **Not** the machine crashing or losing power before the write-back. A write error (full disk, I/O error) is **not** reported. |
| `sync_level::flush` | Writes the data, then the headers, with `msync(MS_SYNC)`, each step checked. | Grows with what changed since the previous flush. | As above, plus a machine crash after `checkpoint()` returns, as far as the system honours `msync`. A write error is thrown from `checkpoint()` as `std::system_error`. |

```c++
t.checkpoint();                          // default: nothing forced to disk
t.checkpoint(artpp::sync_level::flush);  // written out; write errors reported
```

Neither level is crash consistency: the image is updated in place, so a crash
*during* a checkpoint can leave a store that is refused.

#### What is not guaranteed

- **No crash durability.** This is clean-close persistence: there is no
  write-ahead log and the image is updated in place. A crash of the process
  after a change leaves a store that is refused (`store_dirty`), not one that
  is recovered; a crash of the machine, or power loss, can lose or damage
  anything written since the last *flushed* checkpoint (`sync_level::flush`;
  the default forces nothing to disk), and — because `msync` does not force a
  drive's write cache on every system — possibly that checkpoint as well. Keep
  a copy if the data matters.
- **The on-disk format is specific to the library version.** A store is only
  guaranteed to open with the version of the library that wrote it. The format
  version and the layout parameters are stamped and checked, so an incompatible
  store is refused rather than misread, but there is no migration.
- **Not everything can be checked.** Validation reads the two headers, not the
  data: damage inside the node or terminal data, or inside a free list, is not
  detected. The stamp cannot tell apart two mapped types of the same size and
  alignment (`uint64_t` and `double`), nor two user key codecs that do not name
  their encoding (give a codec a `static constexpr std::uint64_t image_id` and
  it can; the built-in codecs have one). A value overwritten in place after the
  last checkpoint (`insert_or_assign` on a present key, a write through a
  reference) changes no structure and is not noticed: it is simply part of the
  image.
- **The lock is advisory.** It keeps a second `line_pool` out, not other
  programs; copy a store only while no pool has it open.
- A store can be moved between machines of the same byte order with 4, 16 or
  64 KB pages. It has not been run on a 64 KB-page kernel.

## Limits and guarantees

- **Key length** — at most `max_key_bytes` = **65535** encoded bytes (the node
  formats use 16-bit length fields). The inserting operations (`insert`,
  `insert_or_assign`, `upsert`, `emplace`, `try_emplace`, `operator[]`) throw
  `std::length_error` *before any allocation or structural change*; `update`,
  `erase` / `remove` and lookups of a longer key simply miss (no stored key
  can be that long), keeping the hot path free of the check.
- **Value alignment** — `alignof(T)` up to **128** is supported. Up to 16 costs
  nothing; above that, leaves are allocated in `alignof(T)`-sized, `alignof(T)`-aligned
  units through the rebound allocator, which must honour that alignment
  (`std::allocator` and `std::pmr::polymorphic_allocator` do). `alignof(T) > 128`,
  or `alignof(T) > 16` with a `line_pool` allocator (its terminal region is
  16-byte granular), is rejected at compile time.
- **`line_pool` capacity** — 2^28 node lines (32 GB) and 2^28 terminal units
  (4 GB) at most, or what the pool was constructed with (the terminal region
  is one eighth of the node capacity). Every leaf that is not stored inside
  its router's line takes terminal units, so on large maps of long keys or
  large values the 4 GB terminal region is the limit that is reached first;
  the allocating operation then throws `artpp::pool_exhausted`.
- **Stack use** — a tree can be as deep as its longest key: keys `a`, `aa`,
  `aaa`, … build one level per byte, up to 65,535. Depth costs heap, not stack,
  for every operation, in every mode: `find` / `contains` / `insert` /
  `emplace` / `update` / `erase` (loops), iterators and `lower_bound` /
  `upper_bound` (heap-backed descent stack), `clear` and the destructor
  (constant stack, no allocation), move assignment between unequal allocators
  and the diagnostic `debug_stats()` (heap work lists). `for_each`,
  `for_each_value` and `operator==` recurse for the first 64 levels and hand
  anything deeper to the iterator: at most about 13 KB of stack in an
  optimized build, about 100 KB unoptimized.
- **Exception safety** — an operation that throws leaves the container valid
  and leaks nothing: no node, and no value's destructor skipped or run twice.
  In every mode:
  - `insert`, `insert_or_assign`, `upsert`, `emplace`, `try_emplace` and
    `operator[]` are **strong**: if one throws (`bad_alloc`, or an exception
    from `T`'s constructors), the contents are exactly what they were — every
    key and every value. An rvalue passed to `insert` may have been moved from.
  - `erase(key)` / `remove` and `clear` **do not throw**. The steps of an
    erase that allocate (collapsing a router, repacking a bucket) are
    best-effort: if one fails the key is still removed and the tree is merely
    less compact. (`erase(key)` is not declared `noexcept` only because a
    user-supplied key codec may throw while encoding the key. `erase(pos)`
    never fails through the tree's allocator either, but it copies the key
    and builds the iterator it returns, which for long keys and deep trees
    takes memory from the global heap.)
  - Move assignment between unequal allocators gives the **basic** guarantee:
    if the clone fails, the target is empty, and the source keeps all its keys,
    with values that are not trivially copyable possibly moved-from.
  - One documented exception to "strong", in `mode::buckets` and
    `mode::adaptive` only: a mapped type that is **move-only and has a throwing
    move constructor**. Existing values are relocated when a bucket overflows
    or a leaf is collapsed into one; the tree moves them if the move cannot
    throw and copies them otherwise (the `std::vector` rule), so nothing is
    lost. A type that can do neither keeps all its keys if such a move throws,
    but values already moved are left moved-from.

  What is tested (`artpp_exception_safety`): random sequences of insert,
  `insert_or_assign`, `emplace`, `operator[]`, `erase` by key and by iterator,
  `clear` and cross-allocator move over clustered, nested and oversized keys,
  repeated once per fault point — the allocator failing at each allocation in
  turn, and `T`'s constructors throwing at each construction in turn — and,
  on further sequences twice as long, with every 2nd, 3rd, 5th, 7th and 13th
  one failing. After each failure the contents are compared with a `std::map`
  oracle, and at the end blocks and live values must balance. This runs in
  `mode::buckets` and `mode::adaptive` for 8-byte, in-handle (4-byte),
  120-byte, `std::string`, throwing-copy and move-only mapped types, and for a
  sample of those types combined with `dense_tiers`, `dense_tiers | ladder_c8`
  and `wide_stems`; each run injects either allocation or constructor failures
  (both for the move-only type); in the default, `dense_tiers` and
  `wide_stems` modes for a sample of the types; and for `artpp::set`. The paths the audit named (router widening, bucket overflow,
  bucket repacking on erase, leaf-to-bucket collapse) have their own sweeps on
  top. Limits of the testing: wide (`setlist_u16`) nodes hardly occur in
  these sequences; their creation and re-striding are fault-tested by a
  bulk-insert sweep that checks validity and leaks rather than contents, and
  erase and cross-allocator move are not fault-tested on a tree that holds
  one. `line_pool` is not put through the systematic sweep
  (`artpp_pool_safety` checks a map that runs its pool dry); key codecs that
  throw are not fault-tested.
- **Thread safety** — that of a standard container: concurrent reads on a
  `const` tree are fine; any write requires exclusive access. No internal
  threads, no locks.
- **Ordering** — strictly ascending encoded-byte order; with the integral
  codecs this is exact numeric order, including negatives.
- **The container is move-only**; copy construction/assignment are deleted by
  design.

## Performance

One run: Apple M5 Max, Apple clang -O3, June 2026, `artpp_bench` (the data is
[docs/results.csv](docs/results.csv), the charts on the website are drawn from
it). 236k dictionary words, 885k clustered strings, 1M uniform and 1M
sequential `uint64_t`; point-lookup latency, fastest of five repetitions. The
ratios were re-measured on the 0.1.0 code on the same machine under load and
agree to within about 10%; the absolute times did not, so treat the times as
indicative. Nothing here has been measured on x86.

How many times **faster** point lookups are than each baseline (higher is
better; below 1 is slower):

| workload | allocator | vs libart | vs `absl::btree_map` | vs `std::map` |
|---|---|---|---|---|
| dictionary words | `std::allocator` | **1.24×** | 2.04× | 2.39× |
| | `line_pool` | **1.29×** | 2.12× | 2.49× |
| clustered strings | `std::allocator` | **1.22×** | 2.22× | 3.17× |
| | `line_pool` | **1.33×** | 2.42× | 3.47× |
| uniform `u64` | `std::allocator` | **1.23×** | 2.82× | 4.74× |
| | `line_pool` | **0.73×** | 1.68× | 2.82× |
| sequential `u64` | `std::allocator` | **1.98×** | 4.37× | 7.23× |
| | `line_pool` | **2.21×** | 4.87× | 8.05× |

- **`line_pool` is not uniformly a win for lookups.** It is the faster
  allocator on three workloads and clearly slower on uniform random integers
  (52 ns against 31 ns with `std::allocator`, and 38 ns for libart). The likely
  cause, not isolated, is that its 4-byte index handles are not pointers, so the
  processor's pointer prefetcher does not follow them. Measure your own
  workload before choosing it for lookup speed.
- **Bucket mode** is the fastest on the dictionary (78 ns, 1.14× the default
  radix) and its ordered scans are 2–5× faster than the radix modes, close to
  `absl::btree_map` on clustered strings (6.5 against 6.8 ns per element);
  btree still scans the dictionary and the integer workloads faster.
- The structural reason reads win: libart loads a node header on every hop
  before it can route; artpp routes from the pointer itself, and a small leaf
  often rides in the routing node's own cacheline.

**Inserts** depend on the allocator and the workload:

| workload | `line_pool` vs `std::allocator` | `line_pool` vs libart | vs `absl::btree_map` | vs `std::map` |
|---|---|---|---|---|
| dictionary words | 1.00× | 0.71× | 0.63× | 1.14× |
| clustered strings | 1.19× | 1.09× | 1.65× | 2.39× |
| uniform `u64` | 1.32× | 1.16× | 2.32× | 4.60× |
| sequential `u64` | 1.30× | 1.72× | 2.55× | 7.05× |

On the dictionary, libart and `absl::btree_map` build faster than artpp: a
radix splits more on deeply shared prefixes. With `std::allocator`, artpp
builds faster than libart only on sequential keys (1.32×) and is 0.71–0.91×
elsewhere. Reproduce with `-DARTPP_BENCHMARKS=ON` (`bench/`); the per-process
regression gate lives in `bench/perf_gate.cpp`.

`for_each` scans are faster than iterator scans when cache-resident; value-only
iteration (`for_each_value` or an un-keyed iterator walk) is cheaper still.

### `artpp::set` — keys-only ordered set

`artpp::set<Key>` is the map's value-free sibling — a thin facade over
`map<Key, none_t>` that shares the radix core verbatim, elides the value path, and
lets short keys ride entirely inside the routing handle (*suffix-in-handle*, no leaf
allocation). Measured **separately** from the table above — its own methodology:
2M keys, `std::allocator` / `operator new` (**not** `line_pool`), find = min-of-7
over a shuffled all-present probe (`bench/ordered_bench.cpp`):

| 2M keys | `artpp::set` find · B/key | `absl::btree_set` | `std::set` |
|---|---|---|---|
| random 16B | **63 ns** · 94 | 378 ns (6.0×) · 26 | 869 ns (13.7×) · 56 |
| alnum 16B | **69 ns** · 41 | 376 ns (5.5×) · 26 | 884 ns (12.9×) · 56 |
| short 3–7B | **61 ns** · 26 | 418 ns (6.9×) · 26 | 843 ns (13.9×) · 56 |

`artpp::set` finds **5.5–6.9× faster than `absl::btree_set`** and ~13× faster than
`std::set`. The trade is build cost and footprint: btree builds ~1.5× faster and
stays denser on high-entropy keys (26 vs 94 B/key) — but `artpp::set` **ties** btree's
bytes/key on short keys, where suffix-in-handle elides the leaf entirely. These use
`operator new`; the set has not been measured on the `line_pool`.

## Benchmark suite

`-DARTPP_BENCHMARKS=ON` builds `artpp_bench`: **one templated engine**
(`bench/engine.hpp`) drives every contestant through identical code — the same
key vectors, op sequences, and checksum accounting — so the comparison carries
no per-library benchmark deviations. Contestants are thin adapters
(`bench/contestants.hpp`):

- `artpp::map` three ways: on the `line_pool` (the `artpp::map` row), on the
  `line_pool` with `mode::buckets`, and on `std::allocator`
- upstream [libart](https://github.com/armon/libart), vendored under
  `external/libart` (bytes-only: integer workloads use the same big-endian
  bytes artpp's integral codec produces)
- `absl::btree_map` (installed abseil or fetched; heterogeneous lookup enabled)
- `std::map` (reference row)

Workloads: `dict` (system word list), `clustered` (synthetic shared-prefix
strings), `uniform` and `sequential` (typed `uint64_t`). Ops: insert, point
hit, point miss, `lower_bound`, full ordered scan, erase (libart has no
`lower_bound`, so it has no row for it). Results print as a table and land
in `results.csv` (one row per contestant × workload × op) for charting.

```sh
cmake -B build -DARTPP_BENCHMARKS=ON && cmake --build build -j
./build/bin/artpp_bench            # [N] [scan_reps] [csv_path]
```

`artpp_perf_gate` is the machine-calibrated artpp/libart ratio regression gate
(`bench/perf_baseline.tsv`); run it alone on a quiet machine. It is not run in
CI, and it measures a different configuration from the tables above (5M
string keys, `uint32_t` values, `std::allocator`), so its ratios are not
comparable with them.

## Testing

`ctest` runs the functional suite:

| binary | scope |
|---|---|
| `artpp_smoke` | API basics in the default, `dense_tiers`, `buckets` and `adaptive` modes, allocator routing (counting, PMR, wide-handle, pool), typed keys, `std::map` API-conformance battery, randomized cross-checks |
| `artpp_conformance` | STL-grade contracts: key-length limit, move-construction and move-assignment branches incl. pool image adoption and structural clone, exactly-once construction, move-only values, compile-time constraints, string_view key parameters, `erase(iterator)`, structural equality incl. shape-divergent trees, iterator edge cases, degenerate shapes, mixed-op differential fuzz vs `std::map` (default, `buckets`, `adaptive`, `dense_tiers`, `wide_stems`) |
| `artpp_reverse_iter` | Bidirectional iteration fuzz vs `std::map` and `std::set`: churn, full walks both directions, `--end()`, random `lower_bound` + backward steps, `++`/`--` ping-pong; the reverse iterators of map and set dereferenced directly (`string_view` keys, inline values), `base()`, std algorithms over the reverse range |
| `artpp_exception_safety` | Exception safety under systematic fault injection, in the default, `buckets`, `adaptive`, `dense_tiers` and `wide_stems` modes (and combinations) and for `artpp::set`: the allocator failing at every allocation point, and value constructors throwing at every construction, of random operation sequences (insert, assign, emplace, `operator[]`, erase, clear, cross-allocator move) and of targeted scenarios (leaf and prefix splits, router widening, bucket overflow, bucket repacking on erase, leaf-to-bucket collapse, structural clone); contents checked against a `std::map` oracle after every failure (validity and leaks only for the bulk-insert and clone sweeps), with leak and double-destroy counters. `ARTPP_EH_SEEDS` / `ARTPP_EH_OPS` lengthen the run |
| `artpp_regression` | Deterministic guards for fixed bugs: fuzzer-found shapes, and the pre-release audit's (erase cascades past the path window, no restructuring on assignment to a present key, over-aligned values, deep trees on a small stack in each of the five base modes, blocks the handle cannot address, `bool` keys, values without a default constructor) |
| `artpp_wide_stems` | `setlist_u16` (wide-stem) cross-checks and depth/latency demonstration |
| `artpp_psio_codec` | (with `ARTPP_WITH_PSIO`; not built in CI) double and reflected-struct keys vs `std::map` ordering oracles; optional keys (nullopt first) |
| `artpp_persistence` | file-backed `line_pool` survives close + reopen: 40k-key round-trip through a full teardown, then further inserts verified through a second reopen (a final erase is checkpointed but not re-read); restores carving state + root with no relocation |
| `artpp_pool_safety` | the file-backed `line_pool` contract: never-checkpointed stores, map destruction after a checkpoint, repeated attach / mutate / checkpoint / reopen cycles; both sync levels recording the same image (the reporting of a write error by `sync_level::flush` is not tested); refusal of truncated, foreign, mismatched and field-corrupted stores without modifying them, of stores mutated after their last checkpoint (also by a writer that died), of a second opener, and of another map instantiation; page-size arithmetic and stores laid out for 64 KB and 1 MB pages; exhaustion at the capacity and at the 2^28-unit handle limits, size overflow, `errno` reporting, `close()` |
| `artpp_fuzz_map` | differential fuzzer vs `std::map` — every op mirrored on both, full forward/reverse/bound/size agreement; standalone seeded RNG runner (also replays corpus files) |
| `artpp_fuzz_set` | differential fuzzer vs `std::set`, covering key suffixes packed in the handle; same seeded runner |
| `artpp_set_smoke` | `artpp::set` basics: the facade over the map with no mapped value |
| `artpp_set_conformance` | STL contracts for `artpp::set`, and a set attached to a file-backed `line_pool` |

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
(external leaves). Keys are drawn from a small alphabet (or, per input, the full byte range) and a recently-used key pool (extend /
truncate / mutate a prior key) to manufacture the shared prefixes and wide fan-out that drive the
split / widen-ladder / bucket-burst / de-widen transitions; any divergence aborts with the input
bytes for replay.

## Status and versioning

The current version is **0.1.0**, the first release. Before 1.0 the API may
change between minor versions; changes are recorded in
[CHANGELOG.md](CHANGELOG.md). The on-disk format of a file-backed `line_pool`
is **specific to the library version**: a store is only guaranteed to open
with the version that wrote it (see
[What is not guaranteed](#what-is-not-guaranteed)). 0.1.0 writes store format
version 2 (`line_pool::VERSION`); a store of any other format version is
refused with `artpp::store_incompatible`.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Bugs and feature requests go to the
[issue tracker](https://github.com/gofractally/libartpp/issues).

## License

MIT; see [LICENSE](LICENSE).

The benchmark suite (not the library) vendors upstream
[libart](https://github.com/armon/libart) under the BSD 3-Clause license; see
[external/libart/LICENSE](external/libart/LICENSE).

# Changelog

Notable changes to libart++. Versions follow [semantic versioning](https://semver.org);
before 1.0 a minor version may change the API.

## Unreleased

### Added
- MIT licence; the vendored libart's BSD 3-Clause licence text.
- `find_package(artpp)` support: installed package config and version files.
- `<artpp/version.hpp>` with `ARTPP_VERSION`.
- Continuous integration: GCC and Clang on Linux (x86-64, ARM64), Apple Clang on
  macOS, each also under AddressSanitizer and UndefinedBehaviorSanitizer, plus an
  install-and-consume check.
- `map::checkpoint()` / `set::checkpoint()` and the two-argument attach
  constructor `map{alloc, artpp::attach}`: the lifecycle of a tree in a
  file-backed `line_pool` without passing root handles and counts by hand.
- `line_pool::close()`: releases the mappings, files and lock and reports a
  failure to do so, which the destructor cannot.
- `artpp_pool_safety`, the test of the file-backed `line_pool` contract. The
  README section on `line_pool` states the lifecycle, what an open validates
  and what is not guaranteed (crash durability; the on-disk format is specific
  to the library version).
- `artpp::bad_allocation_address` (derived from `std::bad_alloc`), thrown when
  an allocator returns a block the tree's handles cannot address.

### Changed
- `checkpoint()` takes an `artpp::sync_level` and forces nothing to disk by
  default (`sync_level::none`); `sync_level::flush` writes the image out with
  `msync` and reports write errors. Durability cost is opt-in.

### Removed
- `docs/sal2.md`, a design note about another project.

### Fixed
- `rbegin()` / `rend()` were `std::reverse_iterator` over the container's
  iterator, which dereferences a temporary copy: with `std::string_view` keys
  the key returned by `*r` viewed a freed buffer (heap-use-after-free), and a
  reference to a value stored in the handle dangled. `map` and `set` now have
  their own `const_reverse_iterator`, which holds the iterator positioned on
  the element; `base()` keeps the standard meaning. The README states the
  lifetime rule: a `string_view` key and a reference to an inline value
  obtained from an iterator are valid until that iterator is advanced or
  destroyed (#19).
- GCC warnings (`-Wreturn-type`, `-Wmaybe-uninitialized`) in `map.hpp`.
- Undefined behaviour: `memcpy` from a null pointer for an empty key suffix or
  prefix (#2).
- `erase` that freed a chain of more than 64 nested single-child nodes left a
  null child in a surviving node, so `begin()`, `for_each` and `lower_bound`
  crashed afterwards. The cascade now continues by re-descent past the
  64-level path window and is correct at any depth (#3).
- `insert`, `insert_or_assign`, `upsert` and `emplace` on a key that is already
  present no longer restructure the tree, so iterators and references stay
  valid as the README promises. Previously such a call could move the values
  of neighbouring keys, free the key's leaf (`mode::adaptive`) or free a prefix
  node that live iterators still pointed at (#11).
- A value constructor that throws (from `emplace` arguments, or a throwing copy
  or move in `insert`) no longer leaks the leaf allocated for it; in
  `mode::buckets` it no longer leaks the new bucket page or consumes space in
  an existing one (#12).
- Inserting below a small inline value (a trivially copyable `T` of at most 5
  bytes) no longer leaks a 128-byte node when the second allocation of the
  split fails (#13).
- Values with `alignof(T) > 16` were stored misaligned (undefined behaviour).
  Leaves for such types are now allocated in `alignof(T)`-aligned units;
  alignments the storage cannot provide (above 128, or above 16 with
  `line_pool`) fail a `static_assert` (#14).
- Stack overflow on deep trees (long keys that are prefixes of one another):
  the destructor and `clear`, `lower_bound` / `upper_bound`, and move
  assignment between unequal allocators no longer recurse; `for_each`,
  `for_each_value` and `operator==` recurse for at most 64 levels. Inserting in
  `mode::buckets` and `debug_stats()`, which at first still recursed once per
  level, are iterative as well: no operation's stack use grows with the depth
  of the tree (#15).
- A block whose address does not fit the handle (above 2^48 with the default
  6-byte handle) or that is not aligned as required (128 bytes for nodes) was
  only caught by `assert`; a release build silently truncated the address.
  Both are now checked on every allocation and reported by throwing
  `artpp::bad_allocation_address` (#16).
- `map<bool, T>` and `set<bool>` did not compile; `bool` is now a built-in key
  type (one byte, `false` before `true`) (#18).
- A mapped type without a default constructor could not be used if it was
  small and trivially copyable (stored in the handle), nor at all in
  `mode::adaptive`. Only `operator[]` and `find_opt` need a default
  constructor now, as documented (#18).
- A 32-bit target is rejected with a `static_assert` instead of miscompiling
  (#18).
- README: the fault-injection and exception-safety statements now say what is
  tested; the test suite gained throwing-value, erase and cross-allocator-move
  fault sweeps to back them (#18).
- `mode::buckets`: when the insert that overflowed a bucket page then failed to
  allocate, the parent (or the root) was left pointing at the freed page: every
  key of that bucket was lost and the next access was a use-after-free. The
  router that replaces the page is now linked before anything else can fail
  (#5).
- `mode::buckets`: an allocation failure while a full router took one more
  child (a widen) dropped the inline prefix of the router, so every key below
  it changed and could no longer be found; with `mode::dense_tiers` a failure
  after the first widening step left the parent pointing at a freed node (#4).
- `mode::buckets` and `mode::adaptive`: `erase` of a key held in a bucket, for
  a mapped type that is not trivially copyable, destroyed the value and then
  allocated; when that allocation failed, `erase` threw and the destroyed
  value was still in the map (use-after-free, double destruction). `erase`
  no longer throws in any mode: it repacks the bucket before destroying
  anything and, if that cannot be done, removes the entry in place (#6).
- `mode::adaptive`: when a leaf was collapsed into a bucket and the new
  value's constructor threw (`emplace` arguments, a throwing copy or move),
  the tree kept pointing at the leaf it had already freed and leaked the new
  bucket page. The page is now completed before the leaf is touched, so the
  insert is all-or-nothing (#7).
- `mode::buckets` and `mode::adaptive`: an insert that overflowed a bucket and
  then failed (an allocation, or a throwing copy) left the keys of that bucket
  in place but, for a mapped type that is not trivially copyable, with
  moved-from values: their contents were lost. Bursting a bucket is now
  all-or-nothing (#1).
- Exception safety is now tested in `mode::buckets` and `mode::adaptive` (and
  their combinations with `dense_tiers`, `ladder_c8`, `wide_stems`) and for
  `artpp::set` the way it was in the radix modes, and beyond: random operation
  sequences with every allocation and every value construction failed in turn,
  checked against a `std::map` oracle. The README's exception-safety section
  states the guarantee per operation and what the tests cover (#1).
- File-backed `line_pool`: a store that was created but never checkpointed
  reopened as `restored()` with root handle 0, which is not the null handle,
  and the documented attach then crashed on the first lookup. Such a store is
  now not `restored()`: `root()` is the null handle (`line_pool::null_root`)
  and `count()` is 0, as for an anonymous pool, and attaching a count of 0
  always gives an empty map. The image format version is now 2 (#9).
- File-backed `line_pool`: destroying a map after `checkpoint()` without
  calling `detach()` freed every node into the pool's free lists, through the
  mapped file; the store then reopened with the checkpointed root and count and
  no keys. A map's destructor now never tears down a tree that lives in a
  file-backed pool (for a trivially destructible mapped type), so `detach()` is
  no longer part of the lifecycle and the order of checkpoint and destruction
  does not matter. `clear()` still frees (#8).
- File-backed `line_pool`: opening a store validated only its magic number and
  version. A truncated `terms` or `nodes` file was extended with zeros and
  used (tens of thousands of keys missing, no error), a `terms` file from
  another checkpoint or another store was accepted, superblock fields were
  trusted, and a small unrelated file named `nodes` was overwritten with a new
  superblock. An existing store is now validated before anything is mapped or
  written: identity and layout fields, every geometry and free-list field
  against the others, the root handle, the file sizes against what the
  checkpoint committed, and that both files carry the same store identity and
  checkpoint generation. A refused open throws `artpp::store_corrupt` or
  `artpp::store_incompatible` (both `artpp::store_error`, a
  `std::runtime_error`), or `std::system_error` with the `errno` for a failed
  system call, instead of `std::bad_alloc`; it modifies and creates nothing.
  `line_pool::checkpoint` throws on a root handle that cannot belong to the
  pool and on a failed write (it was `noexcept` and ignored `msync`) (#10).
- File-backed `line_pool`: a store that was modified after its last
  checkpoint and then closed reopened with the superblock of the checkpoint
  over the pages of a different tree (half the keys missing, no error). The
  superblock now carries a dirty flag, set through the mapping by the first
  allocation or free after an open or a checkpoint and cleared by
  `checkpoint()`; a map that is destroyed or detached holding a tree other
  than the checkpointed one sets it as well. Reopening such a store throws
  `artpp::store_dirty`, also when the writing process died without closing.
  A store that was never checkpointed is reset to empty instead.
  `line_pool::dirty()` reports the state. The flag costs the allocation path
  no branch and no store (#10).
- File-backed `line_pool`: nothing stopped two pools, in one process or in
  two, from opening the same store and overwriting each other's nodes. The
  pool now holds an advisory exclusive lock (`flock`) on the store while it is
  open; a second opener gets `artpp::store_locked` (#10).
- File-backed `line_pool`: a store could be attached by any map
  instantiation, which then read the nodes of another layout. The store now
  records the `image_stamp` of the map that wrote it (size and alignment of
  the mapped type, effective mode, handle width, whether it is a set, the key
  codec's `image_id`), and attaching a different instantiation throws
  `artpp::store_incompatible`. The built-in key codecs name their encoding;
  a user codec can, with a `static constexpr std::uint64_t image_id` (#10).
- `line_pool`: a request whose size in bytes overflowed (`alloc_lines`,
  `alloc_term`, `pool_alloc::allocate` and `artpp_alloc_term` with an absurd
  count) wrapped around, was taken to fit, and moved the frontier past the
  mapping instead of failing. The limits are now compared in units, without
  the multiplication (#1).
- `line_pool`: running out of space throws `artpp::pool_exhausted` and a
  failure of the operating system to commit memory or grow a file throws
  `artpp::pool_commit_failed` with the `errno`; both derive from
  `std::bad_alloc`, which is what was thrown before, without a reason (#1).
- `line_pool::checkpoint` flushes the data of both regions before the headers
  and checks every `msync`, so a write error surfaces from `checkpoint()`
  rather than being lost after a close that looked clean; it flushed only the
  superblock and ignored the result (#1).
- File-backed `line_pool` could not be opened on a kernel with 64 KB pages
  (ppc64le, some arm64): the superblock reservation, which is the file offset
  of the mapped data, was a constant 32 KB. It is now computed from the page
  size with a floor of 64 KB (`line_pool::header_bytes_for`), recorded in the
  store and checked against the page size of the system that opens it; the
  layout is the same for 4, 16 and 64 KB pages, so a store moves between such
  systems. Not run on a 64 KB-page kernel; the arithmetic is unit-tested for
  page sizes from 4 KB to 4 MB, and the whole path with a simulated page size
  (#17).

## 0.1.0

Initial public code: `artpp::map`, `artpp::set`, `artpp::line_pool`, the key
codecs, the test suite, the benchmark suite and the website.

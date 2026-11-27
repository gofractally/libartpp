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
- `artpp::bad_allocation_address` (derived from `std::bad_alloc`), thrown when
  an allocator returns a block the tree's handles cannot address.

### Fixed
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
  `mode::buckets` and `debug_stats()` still recurse once per level; see
  "Limits and guarantees" (#15).
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

### Known issues
- Open bugs from the pre-release audit: #4 to #7 (exception safety in
  `mode::buckets` and `mode::adaptive`), #8 to #10 and #17 (file-backed
  `line_pool`), and the `mode::buckets` insert recursion left from #15.
  Reproducers are in `tests/known_bugs`.

## 0.1.0

Initial public code: `artpp::map`, `artpp::set`, `artpp::line_pool`, the key
codecs, the test suite, the benchmark suite and the website.

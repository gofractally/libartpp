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

### Known issues
- Open bugs from the pre-release audit: #4 to #18. Reproducers are in
  `tests/known_bugs`.

## 0.1.0

Initial public code: `artpp::map`, `artpp::set`, `artpp::line_pool`, the key
codecs, the test suite, the benchmark suite and the website.

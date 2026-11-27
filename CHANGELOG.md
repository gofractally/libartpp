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

### Known issues
- Open bugs from the pre-release audit: #4 to #18. Reproducers are in
  `tests/known_bugs`.

## 0.1.0

Initial public code: `artpp::map`, `artpp::set`, `artpp::line_pool`, the key
codecs, the test suite, the benchmark suite and the website.

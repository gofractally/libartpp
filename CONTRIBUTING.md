# Contributing to libart++

## Building and testing

```sh
cmake -B build && cmake --build build && ctest --test-dir build
```

Before sending a change, also run the suite under the sanitizers:

```sh
cmake -B build-san -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-sanitize-recover=undefined" \
      -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined"
cmake --build build-san && ctest --test-dir build-san
```

CI runs both on x86-64 Linux (GCC and Clang) and macOS (Apple Clang), and the
plain build with GCC on ARM64 Linux. It does not build `ARTPP_WITH_PSIO`, the
benchmarks or the libFuzzer target: run those yourself if a change touches them.

## What a change needs

- **A bug fix** comes with a test that fails without it. Deterministic guards
  for bugs the fuzzers found live in `tests/regression.cpp`.
- **A new node kind or code path** is reachable from the differential fuzzers
  (`tests/fuzz_map.cpp`, `tests/fuzz_set.cpp`). `tools/coverage.sh` (macOS:
  it needs Homebrew llvm or Xcode's `llvm-cov`) reports header coverage for
  the tests it lists in `TESTS`; that list does not yet include `pool_safety`,
  `regression`, the set tests or `fuzz_set`.
- **A performance claim** comes with numbers from the benchmark suite
  (`-DARTPP_BENCHMARKS=ON`), the machine and compiler they were measured on,
  and the comparison run on the same machine. `artpp_perf_gate` checks the
  artpp/libart ratio against `bench/perf_baseline.tsv` (run it by hand on a
  quiet machine; CI does not).
- Public behaviour changes are noted in `CHANGELOG.md` and, where relevant,
  the README.

## Style

`.clang-format` is not provided; match the surrounding code (three-space
indent). Keep comments about why the code is the way it is.

## Licence

Contributions are accepted under the MIT licence in `LICENSE`.

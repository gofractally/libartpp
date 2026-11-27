# Reproducers for open bugs

These programs demonstrate bugs found by the pre-release audit
([#1](https://github.com/gofractally/libartpp/issues/1)). They are **not** part
of `ctest`: each one fails until its issue is fixed. When a bug is fixed, its
case moves into `tests/regression.cpp` or `tests/exception_safety.cpp` and the
reproducer is deleted.

Build one from the repository root:

```sh
c++ -std=c++20 -O1 -g -fsanitize=address,undefined -fno-sanitize-recover=undefined \
    -Iinclude tests/known_bugs/leak_emplace.cpp -o leak_emplace
```

| File | Issue | Run |
|---|---|---|
| `repro_eh.cpp` | #4, #5, #6, #7, #13 exception paths | `./repro_eh 2 1`, `7 9`, `1 0`, `5 -1`, `4 1` |
| `fault.cpp` | #5, #6, #12 allocation and value fault sweep | `./fault 1 1 1 250` |
| `pool_tests.cpp` | #8, #9, #10 file-backed `line_pool` | `./pool_tests 1` to `6` |
| `leak_emplace.cpp` | #12 throwing value constructor leaks | `./leak_emplace` |
| `overalign.cpp` | #14 over-aligned values | `./overalign` |
| `deep_recursion.cpp` | #15 recursion depth; build without sanitizers | `./deep_recursion 8000 0 t` |
| `boolkey.cpp`, `nodefault.cpp` | #18 README claims; these fail to compile | |

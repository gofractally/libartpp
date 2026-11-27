# Reproducers for open bugs

These programs demonstrate bugs found by the pre-release audit
([#1](https://github.com/gofractally/libartpp/issues/1)). They are **not** part
of `ctest`: each one fails until its issue is fixed. When a bug is fixed, its
case moves into `tests/regression.cpp` or `tests/exception_safety.cpp` and the
reproducer is deleted.

Build one from the repository root:

```sh
c++ -std=c++20 -O1 -g -fsanitize=address,undefined -fno-sanitize-recover=undefined \
    -Iinclude tests/known_bugs/pool_tests.cpp -o pool_tests
```

| File | Issue | Run |
|---|---|---|
| `pool_tests.cpp` | #8, #9, #10 file-backed `line_pool` | `./pool_tests 1` to `6` |

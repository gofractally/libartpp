#!/usr/bin/env bash
# tools/coverage.sh — systematic source-coverage of include/artpp/ by the test suite.
#
# Builds every test TU with clang source-based coverage (-fprofile-instr-generate
# -fcoverage-mapping), runs them (fuzz_map gets a heavy iteration count), merges the
# profiles, and reports REGION / LINE / BRANCH coverage of the headers — the code that
# matters, since the library is header-only. Pass a path/regex to drill in, e.g.
#   tools/coverage.sh                      # summary table for all headers
#   tools/coverage.sh show map.hpp         # annotated source of uncovered regions in map.hpp
#
# Compiler and llvm-cov/llvm-profdata come from ONE toolchain (Homebrew llvm if present,
# else xcrun) so the coverage-mapping format versions match.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -x /opt/homebrew/opt/llvm/bin/clang++ ]; then
   CXX=/opt/homebrew/opt/llvm/bin/clang++
   COV=/opt/homebrew/opt/llvm/bin/llvm-cov
   PROFDATA=/opt/homebrew/opt/llvm/bin/llvm-profdata
else
   CXX="$(xcrun -f clang++)"
   COV="$(xcrun -f llvm-cov)"
   PROFDATA="$(xcrun -f llvm-profdata)"
fi

OUT=${ARTPP_COV_OUT:-/tmp/artpp_cov}
FUZZ_ITERS=${ARTPP_FUZZ_ITERS:-200000}
mkdir -p "$OUT"
rm -f "$OUT"/*.profraw "$OUT"/*.profdata
FLAGS=(-std=c++23 -O1 -g -fprofile-instr-generate -fcoverage-mapping -Iinclude)

# Test TUs that need no external deps (psio_codec needs ARTPP_WITH_PSIO; excluded here).
TESTS=(smoke conformance reverse_iter exception_safety wide_stems persistence fuzz_map)

echo "== building instrumented test binaries (${CXX##*/}) =="
OBJS=()
for t in "${TESTS[@]}"; do
   "$CXX" "${FLAGS[@]}" "tests/$t.cpp" -o "$OUT/$t"
   OBJS+=(-object "$OUT/$t")
done

echo "== running (fuzz_map: $FUZZ_ITERS inputs) =="
for t in "${TESTS[@]}"; do
   if [ "$t" = fuzz_map ]; then
      LLVM_PROFILE_FILE="$OUT/$t.profraw" "$OUT/$t" "$FUZZ_ITERS" >/dev/null 2>&1 \
         && echo "   $t: ok" || echo "   $t: NONZERO EXIT (investigate)"
   else
      LLVM_PROFILE_FILE="$OUT/$t.profraw" "$OUT/$t" >/dev/null 2>&1 \
         && echo "   $t: ok" || echo "   $t: NONZERO EXIT (investigate)"
   fi
done

"$PROFDATA" merge -sparse "$OUT"/*.profraw -o "$OUT/merged.profdata"

# OBJS[0] is positional; the rest stay as -object args.
MAIN="${OBJS[1]}"; REST=("${OBJS[@]:2}")
if [ "${1:-report}" = show ]; then
   "$COV" show "$MAIN" "${REST[@]}" -instr-profile="$OUT/merged.profdata" \
      -show-branches=count -show-line-counts-or-regions \
      -path-equivalence=.,"$PWD" "include/artpp/${2:-map.hpp}" | less -R
else
   "$COV" report "$MAIN" "${REST[@]}" -instr-profile="$OUT/merged.profdata" \
      -show-branch-summary -show-region-summary include/artpp/*.hpp
   echo
   echo "drill into uncovered regions:  tools/coverage.sh show map.hpp   (or pool.hpp, cnode.hpp …)"
fi

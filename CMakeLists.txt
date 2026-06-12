# libart++ — artpp::map, a header-only adaptive-radix-tree map for C++23.
#
# Standalone-buildable: clone + `cmake -B build && cmake --build build` needs a
# C++23 compiler and nothing else. Options:
#   ARTPP_TESTS       (default ON when top-level)  functional suite, `ctest` runs it
#   ARTPP_BENCHMARKS  (default OFF)  benchmark/compare suite — vendors upstream
#                     libart (external/libart, BSD-3) and uses absl::btree via
#                     find_package, falling back to FetchContent of abseil-cpp
#   ARTPP_WITH_PSIO   (default OFF)  psio `key`-format codec adapter + its test;
#                     psio is located via ARTPP_PSIO_SOURCE_DIR, find_package,
#                     or FetchContent from github.com/gofractally/psio
cmake_minimum_required(VERSION 3.20)

# project() only when top-level (the psio dependency pattern): embedded via
# FetchContent, the parent's project() has already set up the toolchain.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
   project(artpp VERSION 0.1.0 LANGUAGES C CXX)  # C: vendored libart (benchmarks)
   set(ARTPP_TOPLEVEL ON)
else()
   set(ARTPP_TOPLEVEL OFF)
endif()

option(ARTPP_TESTS      "Build artpp tests"                          ${ARTPP_TOPLEVEL})
option(ARTPP_BENCHMARKS "Build benchmark/compare suite (libart, absl::btree)" OFF)
option(ARTPP_WITH_PSIO  "Enable the psio key-format codec adapter"   OFF)
set(ARTPP_PSIO_SOURCE_DIR "" CACHE PATH "Existing psio checkout (repo root) to use instead of fetching")

# ── the library: header-only ─────────────────────────────────────────────────
add_library(artpp INTERFACE)
add_library(artpp::artpp ALIAS artpp)
target_include_directories(artpp INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_compile_features(artpp INTERFACE cxx_std_23)

include(GNUInstallDirs)
install(DIRECTORY include/artpp TYPE INCLUDE)
install(TARGETS artpp EXPORT artppTargets)
install(EXPORT artppTargets NAMESPACE artpp:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/artpp)

# ── optional: psio key-format adapter ────────────────────────────────────────
if(ARTPP_WITH_PSIO)
   include(FetchContent)
   if(ARTPP_PSIO_SOURCE_DIR)
      FetchContent_Declare(psio SOURCE_DIR ${ARTPP_PSIO_SOURCE_DIR}/cpp)
   else()
      find_package(psio QUIET)
      if(NOT psio_FOUND)
         FetchContent_Declare(psio
             GIT_REPOSITORY https://github.com/gofractally/psio.git
             GIT_TAG        main
             SOURCE_SUBDIR  cpp)
      endif()
   endif()
   if(NOT psio_FOUND)
      FetchContent_MakeAvailable(psio)
   endif()
endif()

# ── tests ────────────────────────────────────────────────────────────────────
if(ARTPP_TESTS)
   enable_testing()

   function(artpp_test name)               # artpp_test(<name> [EXTRA_OPTS...])
      add_executable(artpp_${name} tests/${name}.cpp)
      target_link_libraries(artpp_${name} PRIVATE artpp)
      target_compile_options(artpp_${name} PRIVATE -Wall -Werror=switch ${ARGN})
      set_target_properties(artpp_${name} PROPERTIES
          RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
      add_test(NAME ${name} COMMAND artpp_${name})
   endfunction()

   artpp_test(smoke            -O2 -g)
   artpp_test(conformance      -O2 -g)
   artpp_test(reverse_iter     -O2 -g)
   # fault injection: -O1 keeps every allocation point distinct for the sweep
   artpp_test(exception_safety -O1 -g)
   artpp_test(wide_stems       -O3 -DNDEBUG)

   if(ARTPP_WITH_PSIO)
      artpp_test(psio_codec -O2 -g)
      target_link_libraries(artpp_psio_codec PRIVATE psio)
   endif()
endif()

# ── benchmark / compare suite ────────────────────────────────────────────────
# A single templated engine instantiated per contestant — identical code drives
# artpp, upstream libart, absl::btree_map and std::map, so the comparison has no
# per-library bench deviations. Results print as a table and land in results.csv.
if(ARTPP_BENCHMARKS)
   # upstream libart, vendored (external/libart, see PROVENANCE.md) — bench-only
   add_library(artpp_vendor_libart STATIC external/libart/art.c)
   target_include_directories(artpp_vendor_libart PUBLIC external/libart)
   target_compile_options(artpp_vendor_libart PRIVATE -O3 -DNDEBUG -w)

   # absl::btree — prefer an installed abseil, else fetch it
   find_package(absl QUIET)
   if(NOT absl_FOUND)
      include(FetchContent)
      set(ABSL_PROPAGATE_CXX_STD ON)
      set(ABSL_ENABLE_INSTALL OFF)
      FetchContent_Declare(abseil
          GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
          GIT_TAG        20240722.0)
      FetchContent_MakeAvailable(abseil)
   endif()

   add_executable(artpp_bench bench/bench_main.cpp)
   target_link_libraries(artpp_bench PRIVATE artpp artpp_vendor_libart absl::btree)
   target_compile_options(artpp_bench PRIVATE -O3 -DNDEBUG -Wall -Werror=switch)
   set_target_properties(artpp_bench PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

   # the artpp/libart ratio regression gate (machine-calibrated baseline; run alone)
   add_executable(artpp_perf_gate bench/perf_gate.cpp)
   target_link_libraries(artpp_perf_gate PRIVATE artpp artpp_vendor_libart)
   target_compile_options(artpp_perf_gate PRIVATE -O3 -DNDEBUG -Wall -Werror=switch)
   target_compile_definitions(artpp_perf_gate PRIVATE
       ARTPP_PERF_BASELINE="${CMAKE_CURRENT_SOURCE_DIR}/bench/perf_baseline.tsv")
   set_target_properties(artpp_perf_gate PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endif()

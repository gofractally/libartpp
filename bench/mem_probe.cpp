// mem_probe — cold hard memory footprint per contestant per distribution.
//
// Each map is built in isolation and its bytes measured at the source that is
// honest for it:
//   * pool-backed artpp:  pool.used_lines()*128 + pool.used_term_units()*16 —
//     the exact bytes the line_pool handed out (its nodes are mmap'd, not malloc'd).
//   * everything else (libart, absl::btree_map, std::map, std::allocator-artpp):
//     the malloc default zone's `size_in_use` delta across the build — the real
//     resident bytes including the allocator's own rounding/headers, which a
//     malloc-per-node structure genuinely pays.
// Both are "bytes the structure occupies"; the methods differ because the systems
// do. Reported as total and bytes/key. Build with -DARTPP_BENCHMARKS=ON.
#include "contestants.hpp"
#include "engine.hpp"  // val_of

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include <malloc/malloc.h>

using namespace artpp_bench;

static size_t zone_in_use()
{
   malloc_statistics_t s{};
   malloc_zone_statistics(malloc_default_zone(), &s);
   return s.size_in_use;
}

// Build contestant C over `keys` in isolation; return its footprint in bytes.
template <class C, class Key>
static size_t footprint(const std::vector<Key>& keys)
{
   const size_t before = zone_in_use();
   C            c;
   for (const Key& k : keys) c.insert(k, val_of(k));
   size_t bytes;
   if constexpr (requires(C& cc) { cc.pool.used_lines(); })
      bytes = c.pool.used_lines() * 128 + c.pool.used_term_units() * 16;  // exact pool bytes
   else
      bytes = zone_in_use() - before;  // real malloc-zone growth (incl. rounding)
   return bytes;  // c destroyed as it goes out of scope, after bytes is read
}

template <class... Cs, class Key>
static void probe_row(const char* workload, const std::vector<Key>& keys)
{
   std::printf("%-11s n=%zu\n", workload, keys.size());
   (
       [&] {
          const size_t b = footprint<Cs>(keys);
          std::printf("    %-28s %10zu B  %6.1f B/key\n", Cs::name(), b,
                      double(b) / double(keys.size()));
       }(),
       ...);
}

int main(int argc, char** argv)
{
   const std::size_t N = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 1000000;
   std::mt19937_64   rng(42);

   {  // dict
      std::ifstream            f("/usr/share/dict/words");
      std::vector<std::string> w;
      for (std::string l; std::getline(f, l);)
         if (!l.empty()) w.push_back(l);
      if (w.size() > 1000)
      {
         std::vector<std::string_view> kv(w.begin(), w.end());
         probe_row<artpp_sv, artpp_buckets_sv, artpp_malloc_sv, libart_sv, absl_btree_sv, std_map_sv>(
             "dict", kv);
      }
   }
   {  // clustered
      std::vector<std::string> keys;
      char                     buf[64];
      for (std::size_t i = 0; i < N; ++i)
      {
         std::snprintf(buf, sizeof buf, "user:%08llu/field%llu",
                       (unsigned long long)(rng() % (N / 4 + 1)), (unsigned long long)(i % 16));
         keys.emplace_back(buf);
      }
      std::sort(keys.begin(), keys.end());
      keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
      std::vector<std::string_view> kv(keys.begin(), keys.end());
      probe_row<artpp_sv, artpp_buckets_sv, artpp_malloc_sv, libart_sv, absl_btree_sv, std_map_sv>(
          "clustered", kv);
   }
   {  // uniform u64
      std::vector<uint64_t> keys;
      for (std::size_t i = 0; i < N; ++i) keys.push_back(rng() | 1);
      probe_row<artpp_u64, artpp_buckets_u64, artpp_malloc_u64, libart_u64, absl_btree_u64, std_map_u64>(
          "uniform", keys);
   }
   {  // sequential u64
      std::vector<uint64_t> keys;
      for (std::size_t i = 0; i < N; ++i) keys.push_back(i * 2);
      probe_row<artpp_u64, artpp_buckets_u64, artpp_malloc_u64, libart_u64, absl_btree_u64, std_map_u64>(
          "sequential", keys);
   }
   return 0;
}

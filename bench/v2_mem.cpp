// v2_mem — honest MEMORY comparison of the REAL artpp::v2::map (compact sparse_full,
// v2 §8.1) vs the REAL artpp::map (v1, mode::none). Both are pool-backed; the footprint
// is the exact bytes the line_pool handed out:  used_lines()*128 + used_term_units()*16
// (its nodes are mmap'd, not malloc'd — see bench/mem_probe.cpp for the same accounting).
//
// Two workloads at 2M keys:
//   spread : fixed 16-byte uniformly-random keys      (wide fan-out, prefix-free)
//   text   : fixed 12-byte lowercase a..z keys         (clustered, prefix-free)
// Both maps are fed the IDENTICAL key set; we report total bytes and bytes/key and the
// v1/v2 ratio (>1.0 means v2 is denser — the compaction win).
#include "artpp/map.hpp"      // v1: artpp::map, artpp::line_pool, artpp::pool_alloc
#include "artpp/pool.hpp"
#include "artpp/v2/map.hpp"   // v2: artpp::v2::map, artpp::v2::line_pool, artpp::v2::pool_alloc
#include "artpp/v2/pool.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <string_view>
#include <vector>

static std::vector<std::string> gen(const char* which, size_t N, std::mt19937_64& rng)
{
   std::vector<std::string> v;
   v.reserve(N);
   if (std::string_view(which) == "spread")  // fixed 16-byte uniform random — prefix-free
      for (size_t i = 0; i < N; ++i) { std::string s(16, 0); for (auto& c : s) c = char(rng()); v.push_back(s); }
   else  // "text": fixed 12-byte lowercase a..z — clustered, prefix-free
      for (size_t i = 0; i < N; ++i) { std::string s(12, 0); for (auto& c : s) c = char('a' + rng() % 26); v.push_back(s); }
   std::sort(v.begin(), v.end());
   v.erase(std::unique(v.begin(), v.end()), v.end());
   return v;
}

template <class Pool>
static size_t pool_bytes(const Pool& p)
{
   return p.used_lines() * 128 + p.used_term_units() * 16;  // exact bytes the pool handed out
}

int main(int argc, char** argv)
{
   const size_t    N = (argc > 1) ? std::strtoull(argv[1], nullptr, 10) : 2'000'000;
   std::mt19937_64 rng(42);

   std::printf("%-7s  %12s  %12s  %9s  %9s  %s\n", "wkld", "keys", "", "v1 B/key", "v2 B/key", "v1/v2");
   for (const char* w : {"spread", "text"})
   {
      auto keys = gen(w, N, rng);
      std::vector<std::string_view> kv(keys.begin(), keys.end());

      // v1: artpp::map<…, mode::none, pool_alloc>, pool-backed
      artpp::line_pool                                                            p1;
      artpp::map<std::string_view, uint64_t, artpp::mode::none, artpp::pool_alloc<uint64_t>> m1{
          artpp::pool_alloc<uint64_t>{&p1}};
      for (auto k : kv) m1.insert(k, 0xABCDu);
      size_t v1_bytes = pool_bytes(p1);

      // v2: artpp::v2::map<…, mode::none, v2::pool_alloc>, pool-backed
      artpp::v2::line_pool                                                                            p2;
      artpp::v2::map<std::string_view, uint64_t, artpp::v2::mode::none, artpp::v2::pool_alloc<uint64_t>> m2{
          artpp::v2::pool_alloc<uint64_t>{&p2}};
      for (auto k : kv) m2.insert(k, 0xABCDu);
      size_t v2_bytes = pool_bytes(p2);

      // correctness: every key present in both, with the value we stored
      for (auto k : kv)
      {
         uint64_t a = 0, b = 0;
         if (!m1.find(k, a) || !m2.find(k, b) || a != 0xABCDu || b != 0xABCDu)
         {
            std::printf("CORRECTNESS FAIL (%s)\n", w);
            return 1;
         }
      }

      std::printf("%-7s  %12zu  %12s  %9.2f  %9.2f  %.3fx\n", w, keys.size(), "",
                  double(v1_bytes) / keys.size(), double(v2_bytes) / keys.size(),
                  double(v1_bytes) / double(v2_bytes));
      std::printf("           v1 total = %zu B (%zu lines, %zu term-units) ; v2 total = %zu B (%zu lines, %zu term-units)\n",
                  v1_bytes, p1.used_lines(), p1.used_term_units(),
                  v2_bytes, p2.used_lines(), p2.used_term_units());

      auto d1 = m1.debug_stats();
      auto d2 = m2.debug_stats();
      auto dump = [](const char* tag, const auto& d) {
         std::printf("           %s census: full=%llu (lt48=%llu 48-96=%llu 96-160=%llu 160-192=%llu >=192=%llu) "
                     "fullp=%llu setlist=%llu sl16=%llu c2=%llu c4=%llu c8=%llu prefix=%llu mean_fanout=%.1f\n",
                     tag, (unsigned long long)d.full, (unsigned long long)d.full_lt48,
                     (unsigned long long)d.full_48_96, (unsigned long long)d.full_96_160,
                     (unsigned long long)d.full_160_192, (unsigned long long)d.full_ge192,
                     (unsigned long long)d.fullp, (unsigned long long)d.setlist,
                     (unsigned long long)d.setlist16, (unsigned long long)d.c2,
                     (unsigned long long)d.c4, (unsigned long long)d.c8, (unsigned long long)d.prefix,
                     (d.full + d.fullp) ? double(d.full_nbr_sum) / double(d.full + d.fullp) : 0.0);
      };
      dump("v1", d1);
      dump("v2", d2);
      // The trees are structurally identical (same census); the whole byte delta lives in the
      // full router: v1 full = fixed 1792 B (1 hdr + 13 direct lines); v2 sparse_full =
      // (1 + popcount(presence)) * 128 B. Measured mean popcount(presence): spread 13.65
      // (uniform scatter lights up ~14/16 sub-ranges => no compaction win), text 2.00
      // (a-z confined to 2 sub-ranges => 384 B/full vs 1792 B => the win).
   }
   return 0;
}

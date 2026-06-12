// bench_main — the benchmark/compare suite: artpp vs upstream libart vs
// absl::btree_map vs std::map, one templated engine for all (see engine.hpp).
//
// Workloads:
//   dict        /usr/share/dict/words (string keys, clustered/deep tails)
//   clustered   1M synthetic "user:NNNNNNNN/fieldK" keys (shared prefixes)
//   uniform     1M uniform-random uint64_t keys (typed; libart gets the BE bytes)
//   sequential  1M sequential uint64_t keys (dense radix ladder / btree append)
//
// Ops per workload: insert (build), hit, miss, scan (full ordered iteration),
// erase (every other key). Output: a table on stdout + results.csv for charts.
//
// Usage: artpp_bench [N] [scan_reps] [csv_path]
#include "contestants.hpp"
#include "engine.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <string>
#include <vector>

using namespace artpp_bench;

static std::vector<std::string> load_dict()
{
   std::vector<std::string> w;
   std::ifstream            f("/usr/share/dict/words");
   for (std::string line; std::getline(f, line);)
      if (!line.empty()) w.push_back(line);
   return w;
}

template <class... Cs>
static void run_sv(const char* workload, const std::vector<std::string>& keys,
                   const std::vector<std::string>& misses, int scan_reps)
{
   std::vector<std::string_view> kv(keys.begin(), keys.end());
   std::vector<std::string_view> mv(misses.begin(), misses.end());
   (run_workload<Cs>(workload, kv, mv, scan_reps), ...);
}

template <class... Cs>
static void run_u64(const char* workload, const std::vector<uint64_t>& keys,
                    const std::vector<uint64_t>& misses, int scan_reps)
{
   (run_workload<Cs>(workload, keys, misses, scan_reps), ...);
}

int main(int argc, char** argv)
{
   const std::size_t N         = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 1000000;
   const int         scan_reps = argc > 2 ? std::atoi(argv[2]) : 5;
   const char*       csv       = argc > 3 ? argv[3] : "results.csv";
   std::mt19937_64   rng(42);

   // dict — natural clustered strings
   if (auto words = load_dict(); words.size() > 1000)
   {
      std::vector<std::string> miss;
      miss.reserve(words.size());
      for (const auto& w : words) miss.push_back(w + "\x01");  // disjoint by construction
      std::printf("dict: %zu words\n", words.size());
      run_sv<artpp_sv, artpp_buckets_sv, libart_sv, absl_btree_sv, std_map_sv>(
          "dict", words, miss, scan_reps);
   }
   else
      std::printf("dict: /usr/share/dict/words not found — skipped\n");

   // clustered — synthetic shared-prefix keys
   {
      std::vector<std::string> keys, miss;
      keys.reserve(N);
      miss.reserve(N);
      char buf[64];
      for (std::size_t i = 0; i < N; ++i)
      {
         std::snprintf(buf, sizeof buf, "user:%08llu/field%llu",
                       (unsigned long long)(rng() % (N / 4 + 1)), (unsigned long long)(i % 16));
         keys.emplace_back(buf);
      }
      std::sort(keys.begin(), keys.end());
      keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
      std::shuffle(keys.begin(), keys.end(), rng);
      for (const auto& k : keys) miss.push_back(k + "\x01");
      std::printf("clustered: %zu keys\n", keys.size());
      run_sv<artpp_sv, artpp_buckets_sv, libart_sv, absl_btree_sv, std_map_sv>(
          "clustered", keys, miss, scan_reps);
   }

   // uniform — random u64, typed interfaces
   {
      std::vector<uint64_t> keys, miss;
      keys.reserve(N);
      miss.reserve(N);
      for (std::size_t i = 0; i < N; ++i) keys.push_back(rng() | 1);  // odd
      for (std::size_t i = 0; i < N; ++i) miss.push_back(rng() & ~1ull);  // even: disjoint
      std::printf("uniform: %zu keys\n", keys.size());
      run_u64<artpp_u64, libart_u64, absl_btree_u64, std_map_u64>("uniform", keys, miss,
                                                                  scan_reps);
   }

   // sequential — dense ascending u64
   {
      std::vector<uint64_t> keys, miss;
      keys.reserve(N);
      miss.reserve(N);
      for (std::size_t i = 0; i < N; ++i) keys.push_back(i * 2);      // even
      for (std::size_t i = 0; i < N; ++i) miss.push_back(i * 2 + 1);  // odd: disjoint
      std::printf("sequential: %zu keys\n", keys.size());
      run_u64<artpp_u64, libart_u64, absl_btree_u64, std_map_u64>("sequential", keys, miss,
                                                                  scan_reps);
   }

   print_table();
   write_csv(csv);
   std::printf("(sink %llu)\n", (unsigned long long)g_sink);
   return 0;
}

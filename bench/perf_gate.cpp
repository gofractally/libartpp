// Performance regression guard. Absolute ns/op are machine- and thermal-dependent, so we
// gate on the artpp/libart RATIO measured in the SAME run (both feel the same machine state).
// Baselines (artpp ns, libart ns, ratio) live in bench/perf_baseline.tsv; the test flags any
// metric whose ratio is worse than baseline by more than TOL.
//
//   ./artpp_perf                 compare against the committed baseline (exit 1 on regression)
//   ./artpp_perf --update        re-measure and rewrite the baseline (run on a quiet machine)
extern "C" {
#include "art.h"  // art_tree / art_insert / art_search
}
#include "artpp/map.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <string>
#include <vector>

static double now_s()
{
   using namespace std::chrono;
   return duration<double>(steady_clock::now().time_since_epoch()).count();
}

static constexpr double TOL = 0.03;  // a metric fails if its artpp/libart ratio is >3% worse

struct result
{
   double artpp = 1e30, lib = 1e30;  // best (min) ns/op across reps
   double ratio() const { return artpp / lib; }
};

// One workload: build libart + artpp<u32> from `keys`, query `order`. Best-of-reps (min) for
// each of insert/query, separately for the two impls (ratio uses the per-impl bests).
struct workload
{
   result insert, query;
};

static workload run(const std::vector<std::string>& keys, const std::vector<uint32_t>& order,
                    int reps)
{
   workload w;
   const size_t N = keys.size(), Q = order.size();
   for (int r = 0; r < reps; ++r)
   {
      {  // libart
         art_tree t;
         art_tree_init(&t);
         double t0 = now_s();
         for (size_t i = 0; i < N; ++i)
            art_insert(&t, (const unsigned char*)keys[i].data(), (int)keys[i].size(),
                       (void*)uintptr_t(i + 1));
         w.insert.lib = std::min(w.insert.lib, (now_s() - t0) * 1e9 / double(N));
         uint64_t sink = 0;
         t0            = now_s();
         for (uint32_t i : order)
            sink ^= (uintptr_t)art_search(&t, (const unsigned char*)keys[i].data(), (int)keys[i].size());
         w.query.lib = std::min(w.query.lib, (now_s() - t0) * 1e9 / double(Q));
         asm volatile("" : : "g"(sink) : "memory");
         art_tree_destroy(&t);
      }
      {  // artpp
         artpp::map<std::string_view, uint32_t> t;
         double                                   t0 = now_s();
         for (size_t i = 0; i < N; ++i) t.insert(keys[i], uint32_t(i + 1));
         w.insert.artpp = std::min(w.insert.artpp, (now_s() - t0) * 1e9 / double(N));
         uint64_t sink = 0;
         t0            = now_s();
         for (uint32_t i : order)
         {
            uint32_t v;
            if (t.find(keys[i], v)) sink ^= v;
         }
         w.query.artpp = std::min(w.query.artpp, (now_s() - t0) * 1e9 / double(Q));
         asm volatile("" : : "g"(sink) : "memory");
      }
   }
   return w;
}

int main(int argc, char** argv)
{
   const bool   update = argc > 1 && std::string_view(argv[1]) == "--update";
   const size_t N      = (argc > 2) ? std::strtoull(argv[2], nullptr, 10) : 5'000'000ULL;
   const size_t Q      = N / 2;
   const int    reps   = 4;
   std::mt19937_64 rng(12345);

   // uniform: 16 random bytes (the hash24-style memory-bound case)
   std::vector<std::string> uni(N);
   for (size_t i = 0; i < N; ++i)
   {
      std::string k(16, '\0');
      for (int j = 0; j < 16; ++j) k[size_t(j)] = char(rng());
      uni[i] = std::move(k);
   }
   // clustered: a few shared-prefix families (the radix prefix/split paths)
   std::vector<std::string> clu(N);
   for (size_t i = 0; i < N; ++i)
   {
      std::string k = "family/" + std::to_string(rng() % 4096) + "/";
      int         n = 4 + int(rng() % 8);
      for (int j = 0; j < n; ++j) k.push_back(char('a' + rng() % 16));
      clu[i] = std::move(k);
   }
   std::vector<uint32_t> order(Q);
   for (size_t i = 0; i < Q; ++i) order[i] = uint32_t(rng() % N);

   std::printf("artpp_perf  N=%zu  reps=%d  (gate = artpp/libart ratio, tol %.0f%%)\n", N, reps,
               TOL * 100);
   workload u = run(uni, order, reps);
   workload c = run(clu, order, reps);

   struct row { const char* name; result r; };
   std::vector<row> rows = {{"uniform_insert", u.insert},
                            {"uniform_query", u.query},
                            {"clustered_insert", c.insert},
                            {"clustered_query", c.query}};

#ifndef ARTPP_PERF_BASELINE
#define ARTPP_PERF_BASELINE "bench/perf_baseline.tsv"
#endif
   const char* path = ARTPP_PERF_BASELINE;
   if (update)
   {
      std::ofstream f(path);
      f << "# metric\tartpp_ns\tlibart_ns\tratio\n";
      for (auto& [name, r] : rows)
         f << name << '\t' << r.artpp << '\t' << r.lib << '\t' << r.ratio() << '\n';
      std::printf("baseline written to %s:\n", path);
      for (auto& [name, r] : rows)
         std::printf("  %-18s artpp=%.1f lib=%.1f ratio=%.3f\n", name, r.artpp, r.lib, r.ratio());
      return 0;
   }

   std::map<std::string, double> base;
   std::ifstream                 f(path);
   if (!f)
   {
      std::printf("no baseline (%s) — run `artpp_perf --update` first\n", path);
      return 0;  // don't gate without a baseline
   }
   std::string line;
   while (std::getline(f, line))
   {
      if (line.empty() || line[0] == '#') continue;
      auto t1 = line.find('\t'), t3 = line.rfind('\t');
      base[line.substr(0, t1)] = std::strtod(line.c_str() + t3 + 1, nullptr);
   }

   int fails = 0;
   for (auto& [name, r] : rows)
   {
      auto it = base.find(name);
      if (it == base.end()) { std::printf("  %-18s no baseline entry\n", name); continue; }
      const double cur = r.ratio(), b = it->second, dev = (cur - b) / b;
      const bool   bad = cur > b * (1.0 + TOL);
      if (bad) ++fails;
      std::printf("  %-18s artpp=%6.1f lib=%6.1f  ratio %.3f  base %.3f  %+5.1f%%  %s\n", name, r.artpp,
                  r.lib, cur, b, dev * 100, bad ? "REGRESSION" : (dev < -TOL ? "improved" : "ok"));
   }
   std::printf(fails ? "\nartpp_perf: %d REGRESSION(S)\n" : "\nartpp_perf: OK\n", fails);
   return fails ? 1 : 0;
}

// bench_double — artpp::map<double> vs std::map<double> vs absl::btree_map<double>.
//
// The point: a radix tree is byte-ordered, so people don't expect it to key on
// floating point. It does — psio's `key` codec emits memcmp-sortable bytes for
// doubles (IEEE sign-transform), so artpp::map<double,V> iterates in true numeric
// order, negatives and all. This first CROSS-CHECKS that order against std::map
// (the proof it works), then times insert / lookup / ordered-scan / lower_bound
// against std::map and absl::btree_map. One process, min-of-reps for point ops;
// a relative read on a warm machine, not a polished absolute benchmark.
#include <artpp/map.hpp>
#include <artpp/psio_codec.hpp>
#include <artpp/pool.hpp>

#include <absl/container/btree_map.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <random>
#include <vector>

ARTPP_PSIO_KEY(double)

using clk = std::chrono::steady_clock;
static double now_ns()
{
   return double(std::chrono::duration_cast<std::chrono::nanoseconds>(clk::now().time_since_epoch()).count());
}
template <class F>
static double best_ns(F&& fn, std::size_t n, int reps)
{
   double best = 1e30;
   for (int r = 0; r < reps; ++r)
   {
      const double t0 = now_ns();
      fn();
      best = std::min(best, (now_ns() - t0) / double(n));
   }
   return best;
}

int main(int argc, char** argv)
{
   const std::size_t N    = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 1000000;
   const int         reps = argc > 2 ? std::atoi(argv[2]) : 7;
   std::mt19937_64   rng(42);

   // Keys: mixed sign and magnitude — the order-preservation stress (negatives,
   // tiny, huge). De-duplicated so all three maps hold the identical key set.
   std::vector<double> keys;
   keys.reserve(N);
   {
      std::map<double, char> uniq;  // also our dedup
      while (uniq.size() < N)
      {
         double d = (double(rng()) / double(rng() | 1)) * (rng() & 1 ? 1.0 : -1.0);
         if (std::isfinite(d)) uniq.emplace(d, 0);
      }
      for (auto& [k, _] : uniq) keys.push_back(k);
   }
   // absent keys for lower_bound / miss probes
   std::vector<double> probes;
   probes.reserve(N);
   for (std::size_t i = 0; i < N; ++i)
   {
      double d = (double(rng()) / double(rng() | 1)) * (rng() & 1 ? 1.0 : -1.0);
      probes.push_back(std::isfinite(d) ? d : 0.0);
   }
   // randomized query order (not the sorted build order)
   std::vector<double> q = keys;
   std::shuffle(q.begin(), q.end(), rng);
   std::shuffle(probes.begin(), probes.end(), rng);

   // ── the three maps ───────────────────────────────────────────────────────
   artpp::line_pool pool;
   artpp::map<double, uint64_t, artpp::mode::none, artpp::pool_alloc<uint64_t>> art{
       artpp::pool_alloc<uint64_t>{&pool}};
   std::map<double, uint64_t>        smap;
   absl::btree_map<double, uint64_t> btree;
   for (std::size_t i = 0; i < keys.size(); ++i)
   {
      art.upsert(keys[i], i);
      smap[keys[i]] = i;
      btree[keys[i]] = i;
   }

   // ── correctness: artpp numeric order == std::map order, keys + values ──────
   {
      bool ok = (art.size() == smap.size());
      auto mi = smap.begin();
      std::size_t checked = 0;
      for (auto it = art.begin(); ok && it != art.end(); ++it, ++mi)
      {
         if (mi == smap.end() || it->first != mi->first || it->second != mi->second) ok = false;
         ++checked;
      }
      // spot-check lower_bound agreement on the absent probes
      for (std::size_t i = 0; ok && i < 10000; ++i)
      {
         auto a = art.lower_bound(probes[i]);
         auto s = smap.lower_bound(probes[i]);
         const bool ae = (a == art.end()), se = (s == smap.end());
         if (ae != se || (!ae && a->first != s->first)) ok = false;
      }
      std::printf("correctness: %s  (artpp double-keyed order == std::map over %zu keys, "
                  "+ lower_bound agreement)\n\n",
                  ok ? "PASS" : "FAIL", checked);
      if (!ok) return 1;
   }

   // ── timings ────────────────────────────────────────────────────────────────
   volatile uint64_t sink = 0;
   std::printf("%-18s %12s %12s %14s %14s\n", "double-keyed", "insert ns", "hit ns", "scan ns/el",
               "lbound ns");

   {  // artpp (rebuild for insert timing, in isolation)
      double ins = best_ns(
          [&] {
             artpp::line_pool p2;
             artpp::map<double, uint64_t, artpp::mode::none, artpp::pool_alloc<uint64_t>> m{
                 artpp::pool_alloc<uint64_t>{&p2}};
             for (std::size_t i = 0; i < keys.size(); ++i) m.upsert(keys[i], i);
             sink += m.size();
          },
          keys.size(), reps);
      double hit = best_ns([&] { uint64_t s = 0, o; for (double k : q) if (art.find(k, o)) s += o; sink += s; },
                           q.size(), reps);
      double scn = best_ns([&] { uint64_t s = 0; art.for_each_value([&](const uint64_t& v) { s += v; }); sink += s; },
                           keys.size(), reps);
      double lb  = best_ns([&] { uint64_t s = 0; for (double k : probes) { auto it = art.lower_bound(k); if (it != art.end()) s += it.value(); } sink += s; },
                           probes.size(), reps);
      std::printf("%-18s %12.1f %12.1f %14.2f %14.1f\n", "artpp::map", ins, hit, scn, lb);
   }
   {  // std::map
      double ins = best_ns([&] { std::map<double, uint64_t> m; for (std::size_t i = 0; i < keys.size(); ++i) m[keys[i]] = i; sink += m.size(); },
                           keys.size(), reps);
      double hit = best_ns([&] { uint64_t s = 0; for (double k : q) { auto it = smap.find(k); if (it != smap.end()) s += it->second; } sink += s; },
                           q.size(), reps);
      double scn = best_ns([&] { uint64_t s = 0; for (auto& [k, v] : smap) s += v; sink += s; }, keys.size(), reps);
      double lb  = best_ns([&] { uint64_t s = 0; for (double k : probes) { auto it = smap.lower_bound(k); if (it != smap.end()) s += it->second; } sink += s; },
                           probes.size(), reps);
      std::printf("%-18s %12.1f %12.1f %14.2f %14.1f\n", "std::map", ins, hit, scn, lb);
   }
   {  // absl::btree_map
      double ins = best_ns([&] { absl::btree_map<double, uint64_t> m; for (std::size_t i = 0; i < keys.size(); ++i) m[keys[i]] = i; sink += m.size(); },
                           keys.size(), reps);
      double hit = best_ns([&] { uint64_t s = 0; for (double k : q) { auto it = btree.find(k); if (it != btree.end()) s += it->second; } sink += s; },
                           q.size(), reps);
      double scn = best_ns([&] { uint64_t s = 0; for (auto& [k, v] : btree) s += v; sink += s; }, keys.size(), reps);
      double lb  = best_ns([&] { uint64_t s = 0; for (double k : probes) { auto it = btree.lower_bound(k); if (it != btree.end()) s += it->second; } sink += s; },
                           probes.size(), reps);
      std::printf("%-18s %12.1f %12.1f %14.2f %14.1f\n", "absl::btree_map", ins, hit, scn, lb);
   }
   return int(sink & 1) & 0;  // keep sink live
}

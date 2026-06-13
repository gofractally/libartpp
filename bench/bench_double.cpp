// bench_double — double-keyed maps: artpp vs std::map vs absl::btree vs std::unordered_map.
//
// Two points. (1) A radix tree is byte-ordered, yet it keys on floating point: psio's
// `key` codec emits memcmp-sortable bytes (IEEE sign-transform) and psio::to_key hands a
// fixed-width key straight into a stack buffer, so artpp::map<double,V> iterates in true
// numeric order with no per-op allocation. We CROSS-CHECK that order against std::map.
// (2) The ordered-vs-hash trade: std::unordered_map is fast on point ops but has NO
// lower_bound and NO ordered scan (shown as "—") — the reason to reach for an ordered map.
//
// Each contestant is built and measured in its own scope (one map alive at a time) so the
// 100M run fits in RAM. Point ops are min-of-reps. Run alone; pass N and reps:
//   artpp_bench_double 100000000 3
#include <artpp/map.hpp>
#include <artpp/psio_codec.hpp>
#include <artpp/pool.hpp>

#include <absl/container/btree_map.h>

#include "art.h"  // vendored upstream libart (point ops only here)

#include <algorithm>
#include <array>
#include <cstring>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <random>
#include <unordered_map>
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

static std::vector<double>    g_keys;   // the stored set
static std::vector<double>    g_query;  // present keys, shuffled (hit)
static std::vector<double>    g_probe;  // random doubles (lower_bound / miss)
static volatile std::uint64_t g_sink = 0;

static void prow(const char* name, double ins, double hit, double scan, double lb)
{
   char sc[16], lc[16];  // negative value → op not offered ("n/a")
   if (scan < 0) std::snprintf(sc, sizeof sc, "%12s", "n/a");
   else          std::snprintf(sc, sizeof sc, "%12.2f", scan);
   if (lb < 0)   std::snprintf(lc, sizeof lc, "%12s", "n/a");
   else          std::snprintf(lc, sizeof lc, "%12.1f", lb);
   std::printf("    %-20s %10.1f %10.1f %s %s\n", name, ins, hit, sc, lc);
}

int main(int argc, char** argv)
{
   const std::size_t N        = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 1000000;
   const int         reps     = argc > 2 ? std::atoi(argv[2]) : 5;
   const int         ins_reps = N >= 10'000'000 ? 1 : std::min(reps, 3);  // a 100M build is costly
   std::mt19937_64   rng(42);

   g_keys.reserve(N);
   for (std::size_t i = 0; i < N; ++i)
   {
      double d = (double(rng()) / double(rng() | 1)) * (rng() & 1 ? 1.0 : -1.0);
      g_keys.push_back(std::isfinite(d) ? d : double(i));  // dups (≈0 at random) just overwrite
   }
   const std::size_t M = std::min<std::size_t>(N, 5'000'000);  // bounded query set
   g_query.assign(g_keys.begin(), g_keys.begin() + M);
   std::shuffle(g_query.begin(), g_query.end(), rng);
   g_probe.reserve(M);
   for (std::size_t i = 0; i < M; ++i)
   {
      double d = (double(rng()) / double(rng() | 1)) * (rng() & 1 ? 1.0 : -1.0);
      g_probe.push_back(std::isfinite(d) ? d : 0.0);
   }

   // ── correctness: artpp double-keyed order == std::map (small, fixed scale) ──
   {
      const std::size_t            nc = std::min<std::size_t>(N, 200000);
      artpp::map<double, uint64_t> a;
      std::map<double, uint64_t>   s;
      for (std::size_t i = 0; i < nc; ++i) { a.upsert(g_keys[i], i); s[g_keys[i]] = i; }
      bool ok = (a.size() == s.size());
      auto si = s.begin();
      for (auto it = a.begin(); ok && it != a.end(); ++it, ++si)
         if (si == s.end() || it->first != si->first) ok = false;
      std::printf("correctness: %s  (artpp double order == std::map over %zu keys)\n\n",
                  ok ? "PASS" : "FAIL", nc);
      if (!ok) return 1;
   }

   std::printf("double-keyed  N=%zu  (ns/op, min of %d reps; ord-scan = ns/elem)\n", N, reps);
   std::printf("    %-20s %10s %10s %12s %12s\n", "", "insert", "hit", "ord-scan", "lower_bnd");

   using A = artpp::pool_alloc<uint64_t>;
   {  // artpp::map — pool-backed flagship
      double ins = best_ns(
          [&] {
             artpp::line_pool                                         pool;
             artpp::map<double, uint64_t, artpp::mode::none, A>       m{A{&pool}};
             for (std::size_t i = 0; i < g_keys.size(); ++i) m.upsert(g_keys[i], i);
             g_sink += m.size();
          },
          g_keys.size(), ins_reps);
      artpp::line_pool                                   pool;
      artpp::map<double, uint64_t, artpp::mode::none, A> m{A{&pool}};
      for (std::size_t i = 0; i < g_keys.size(); ++i) m.upsert(g_keys[i], i);
      double hit = best_ns([&] { uint64_t s = 0, o; for (double k : g_query) if (m.find(k, o)) s += o; g_sink += s; }, g_query.size(), reps);
      double scn = best_ns([&] { uint64_t s = 0; m.for_each_value([&](const uint64_t& v) { s += v; }); g_sink += s; }, g_keys.size(), reps);
      double lb  = best_ns([&] { uint64_t s = 0; for (double k : g_probe) { auto it = m.lower_bound(k); if (it != m.end()) s += it.value(); } g_sink += s; }, g_probe.size(), reps);
      prow("artpp::map", ins, hit, scn, lb);
   }
   {  // std::map
      double ins = best_ns([&] { std::map<double, uint64_t> m; for (std::size_t i = 0; i < g_keys.size(); ++i) m[g_keys[i]] = i; g_sink += m.size(); }, g_keys.size(), ins_reps);
      std::map<double, uint64_t> m; for (std::size_t i = 0; i < g_keys.size(); ++i) m[g_keys[i]] = i;
      double hit = best_ns([&] { uint64_t s = 0; for (double k : g_query) { auto it = m.find(k); if (it != m.end()) s += it->second; } g_sink += s; }, g_query.size(), reps);
      double scn = best_ns([&] { uint64_t s = 0; for (auto& kv : m) s += kv.second; g_sink += s; }, g_keys.size(), reps);
      double lb  = best_ns([&] { uint64_t s = 0; for (double k : g_probe) { auto it = m.lower_bound(k); if (it != m.end()) s += it->second; } g_sink += s; }, g_probe.size(), reps);
      prow("std::map", ins, hit, scn, lb);
   }
   {  // absl::btree_map
      double ins = best_ns([&] { absl::btree_map<double, uint64_t> m; for (std::size_t i = 0; i < g_keys.size(); ++i) m[g_keys[i]] = i; g_sink += m.size(); }, g_keys.size(), ins_reps);
      absl::btree_map<double, uint64_t> m; for (std::size_t i = 0; i < g_keys.size(); ++i) m[g_keys[i]] = i;
      double hit = best_ns([&] { uint64_t s = 0; for (double k : g_query) { auto it = m.find(k); if (it != m.end()) s += it->second; } g_sink += s; }, g_query.size(), reps);
      double scn = best_ns([&] { uint64_t s = 0; for (auto& kv : m) s += kv.second; g_sink += s; }, g_keys.size(), reps);
      double lb  = best_ns([&] { uint64_t s = 0; for (double k : g_probe) { auto it = m.lower_bound(k); if (it != m.end()) s += it->second; } g_sink += s; }, g_probe.size(), reps);
      prow("absl::btree_map", ins, hit, scn, lb);
   }
   {  // libart (upstream) — point ops only; same normalized key bytes (psio::to_key, so
      // -0.0 collapses like everywhere else), but NO ordered scan and NO lower_bound API.
      auto enc = [](double d) { return psio::to_key(d); };  // std::array<char,8> for double
      double ins = best_ns(
          [&] {
             art_tree t; art_tree_init(&t);
             for (std::size_t i = 0; i < g_keys.size(); ++i)
             { auto b = enc(g_keys[i]); art_insert(&t, reinterpret_cast<const unsigned char*>(b.data()), int(b.size()), reinterpret_cast<void*>(uintptr_t(i + 1))); }
             g_sink += art_size(&t);
             art_tree_destroy(&t);
          },
          g_keys.size(), ins_reps);
      art_tree t; art_tree_init(&t);
      for (std::size_t i = 0; i < g_keys.size(); ++i)
      { auto b = enc(g_keys[i]); art_insert(&t, reinterpret_cast<const unsigned char*>(b.data()), int(b.size()), reinterpret_cast<void*>(uintptr_t(i + 1))); }
      double hit = best_ns([&] { uint64_t s = 0; for (double k : g_query) { auto b = enc(k); void* r = art_search(&t, reinterpret_cast<const unsigned char*>(b.data()), int(b.size())); if (r) s += uint64_t(uintptr_t(r)); } g_sink += s; }, g_query.size(), reps);
      art_tree_destroy(&t);
      prow("libart", ins, hit, -1, -1);
   }
   {  // std::unordered_map — hash: fast point ops, NO ordered scan / lower_bound
      double ins = best_ns([&] { std::unordered_map<double, uint64_t> m; m.reserve(g_keys.size()); for (std::size_t i = 0; i < g_keys.size(); ++i) m[g_keys[i]] = i; g_sink += m.size(); }, g_keys.size(), ins_reps);
      std::unordered_map<double, uint64_t> m; m.reserve(g_keys.size()); for (std::size_t i = 0; i < g_keys.size(); ++i) m[g_keys[i]] = i;
      double hit = best_ns([&] { uint64_t s = 0; for (double k : g_query) { auto it = m.find(k); if (it != m.end()) s += it->second; } g_sink += s; }, g_query.size(), reps);
      prow("std::unordered_map", ins, hit, -1, -1);
   }
   return int(g_sink & 1) & 0;
}

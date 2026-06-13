// bench_buffer — ordered maps on RANDOM 32-BYTE BUFFER keys (hashes, UUIDs, composite
// keys — the real-world key that doesn't fit in a register).
//
// The point: a previous double-keyed run flattered art_map, whose key must fit in one
// register-width integer (≤16 B). For a 32-byte key it does not even COMPILE
// (static_assert "Unsupported key type") — so the field here is artpp / libart /
// absl::btree / std::map / std::unordered_map; art_map is structurally out.
//
// Each contestant built/measured in isolation (one map at a time) so 100M fits in RAM.
// Point ops are min-of-reps; identical workload across contestants. Run alone:
//   artpp_bench_buffer 1000000 7
#include <artpp/map.hpp>
#include <artpp/pool.hpp>

#include <absl/container/btree_map.h>

#include "art.h"  // vendored upstream libart

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <map>
#include <random>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

using clk = std::chrono::steady_clock;
static double now_ns()
{
   return double(std::chrono::duration_cast<std::chrono::nanoseconds>(clk::now().time_since_epoch()).count());
}
template <class F>
static double best_ns(F&& fn, std::size_t n, int reps)
{
   double best = 1e30;
   for (int r = 0; r < reps; ++r) { double t0 = now_ns(); fn(); best = std::min(best, (now_ns() - t0) / double(n)); }
   return best;
}
static void prow(const char* name, double ins, double hit, double scan, double lb)
{
   char sc[16], lc[16];
   if (scan < 0) std::snprintf(sc, sizeof sc, "%12s", "n/a"); else std::snprintf(sc, sizeof sc, "%12.2f", scan);
   if (lb < 0)   std::snprintf(lc, sizeof lc, "%12s", "n/a"); else std::snprintf(lc, sizeof lc, "%12.1f", lb);
   std::printf("    %-20s %10.1f %10.1f %s %s\n", name, ins, hit, sc, lc);
}

static std::vector<std::string> g_keys;   // 32-byte random buffers (stored set)
static std::vector<std::string> g_query;  // present, shuffled (hit)
static std::vector<std::string> g_probe;  // absent random buffers (lower_bound)
static volatile std::uint64_t   g_sink = 0;

int main(int argc, char** argv)
{
   const std::size_t N        = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 1000000;
   const int         reps     = argc > 2 ? std::atoi(argv[2]) : 5;
   const int         ins_reps = N >= 10'000'000 ? 1 : std::min(reps, 3);
   const std::string amode    = argc > 3 ? argv[3] : "both";  // "std" skips the pool row
   std::mt19937_64   rng(42);

   auto mkbuf = [&](std::string& s) { s.resize(32); auto* p = reinterpret_cast<std::uint64_t*>(s.data()); for (int i = 0; i < 4; ++i) p[i] = rng(); };
   g_keys.resize(N);
   for (auto& s : g_keys) mkbuf(s);
   const std::size_t M = std::min<std::size_t>(N, 5'000'000);
   g_query.assign(g_keys.begin(), g_keys.begin() + M);
   std::shuffle(g_query.begin(), g_query.end(), rng);
   g_probe.resize(M);
   for (auto& s : g_probe) mkbuf(s);

   std::printf("32-byte buffer keys  N=%zu  (ns/op, min of %d reps; ord-scan = ns/elem)\n", N, reps);
   std::printf("    %-20s %10s %10s %12s %12s\n", "", "insert", "hit", "ord-scan", "lower_bnd");

   using A = artpp::pool_alloc<uint64_t>;
   if (amode != "std")
   {  // artpp::map<string_view> — pool-backed flagship (4-byte handles; terminal region caps at 4 GB)
      double ins = best_ns([&] { artpp::line_pool p; artpp::map<std::string_view, uint64_t, artpp::mode::none, A> m{A{&p}}; for (std::size_t i = 0; i < N; ++i) m.insert(g_keys[i], i); g_sink += m.size(); }, N, ins_reps);
      artpp::line_pool pool;
      artpp::map<std::string_view, uint64_t, artpp::mode::none, A> m{A{&pool}};
      for (std::size_t i = 0; i < N; ++i) m.insert(g_keys[i], i);
      double hit = best_ns([&] { uint64_t s = 0, o; for (auto& k : g_query) if (m.find(k, o)) s += o; g_sink += s; }, M, reps);
      double scn = best_ns([&] { uint64_t s = 0; m.for_each_value([&](const uint64_t& v) { s += v; }); g_sink += s; }, N, reps);
      double lb  = best_ns([&] { uint64_t s = 0; for (auto& k : g_probe) { auto it = m.lower_bound(k); if (it != m.end()) s += it.value(); } g_sink += s; }, M, reps);
      prow("artpp::map (line_pool)", ins, hit, scn, lb);
   }
   {  // artpp::map<string_view> — std::allocator (zero-setup; no 4 GB terminal cap → reaches 100M)
      double ins = best_ns([&] { artpp::map<std::string_view, uint64_t> m; for (std::size_t i = 0; i < N; ++i) m.insert(g_keys[i], i); g_sink += m.size(); }, N, ins_reps);
      artpp::map<std::string_view, uint64_t> m;
      for (std::size_t i = 0; i < N; ++i) m.insert(g_keys[i], i);
      double hit = best_ns([&] { uint64_t s = 0, o; for (auto& k : g_query) if (m.find(k, o)) s += o; g_sink += s; }, M, reps);
      double scn = best_ns([&] { uint64_t s = 0; m.for_each_value([&](const uint64_t& v) { s += v; }); g_sink += s; }, N, reps);
      double lb  = best_ns([&] { uint64_t s = 0; for (auto& k : g_probe) { auto it = m.lower_bound(k); if (it != m.end()) s += it.value(); } g_sink += s; }, M, reps);
      prow("artpp::map (std::alloc)", ins, hit, scn, lb);
   }
   {  // libart — point ops + ordered scan (byte-string radix); no lower_bound API
      auto kp = [](const std::string& s) { return reinterpret_cast<const unsigned char*>(s.data()); };
      double ins = best_ns([&] { art_tree t; art_tree_init(&t); for (std::size_t i = 0; i < N; ++i) art_insert(&t, kp(g_keys[i]), 32, reinterpret_cast<void*>(uintptr_t(i + 1))); g_sink += art_size(&t); art_tree_destroy(&t); }, N, ins_reps);
      art_tree t; art_tree_init(&t);
      for (std::size_t i = 0; i < N; ++i) art_insert(&t, kp(g_keys[i]), 32, reinterpret_cast<void*>(uintptr_t(i + 1)));
      double hit = best_ns([&] { uint64_t s = 0; for (auto& k : g_query) { void* r = art_search(&t, kp(k), 32); if (r) s += uint64_t(uintptr_t(r)); } g_sink += s; }, M, reps);
      double scn = best_ns([&] { uint64_t s = 0; art_iter(&t, [](void* d, const unsigned char*, uint32_t, void* v) { *static_cast<uint64_t*>(d) += uint64_t(uintptr_t(v)); return 0; }, &s); g_sink += s; }, N, reps);
      art_tree_destroy(&t);
      prow("libart", ins, hit, scn, -1);
   }
   {  // absl::btree_map
      double ins = best_ns([&] { absl::btree_map<std::string, uint64_t> m; for (std::size_t i = 0; i < N; ++i) m[g_keys[i]] = i; g_sink += m.size(); }, N, ins_reps);
      absl::btree_map<std::string, uint64_t> m; for (std::size_t i = 0; i < N; ++i) m[g_keys[i]] = i;
      double hit = best_ns([&] { uint64_t s = 0; for (auto& k : g_query) { auto it = m.find(k); if (it != m.end()) s += it->second; } g_sink += s; }, M, reps);
      double scn = best_ns([&] { uint64_t s = 0; for (auto& kv : m) s += kv.second; g_sink += s; }, N, reps);
      double lb  = best_ns([&] { uint64_t s = 0; for (auto& k : g_probe) { auto it = m.lower_bound(k); if (it != m.end()) s += it->second; } g_sink += s; }, M, reps);
      prow("absl::btree_map", ins, hit, scn, lb);
   }
   {  // std::map
      double ins = best_ns([&] { std::map<std::string, uint64_t> m; for (std::size_t i = 0; i < N; ++i) m[g_keys[i]] = i; g_sink += m.size(); }, N, ins_reps);
      std::map<std::string, uint64_t> m; for (std::size_t i = 0; i < N; ++i) m[g_keys[i]] = i;
      double hit = best_ns([&] { uint64_t s = 0; for (auto& k : g_query) { auto it = m.find(k); if (it != m.end()) s += it->second; } g_sink += s; }, M, reps);
      double scn = best_ns([&] { uint64_t s = 0; for (auto& kv : m) s += kv.second; g_sink += s; }, N, reps);
      double lb  = best_ns([&] { uint64_t s = 0; for (auto& k : g_probe) { auto it = m.lower_bound(k); if (it != m.end()) s += it->second; } g_sink += s; }, M, reps);
      prow("std::map", ins, hit, scn, lb);
   }
   {  // std::unordered_map — hash: fast point ops, no ordered scan / lower_bound
      double ins = best_ns([&] { std::unordered_map<std::string, uint64_t> m; m.reserve(N); for (std::size_t i = 0; i < N; ++i) m[g_keys[i]] = i; g_sink += m.size(); }, N, ins_reps);
      std::unordered_map<std::string, uint64_t> m; m.reserve(N); for (std::size_t i = 0; i < N; ++i) m[g_keys[i]] = i;
      double hit = best_ns([&] { uint64_t s = 0; for (auto& k : g_query) { auto it = m.find(k); if (it != m.end()) s += it->second; } g_sink += s; }, M, reps);
      prow("std::unordered_map", ins, hit, -1, -1);
   }
   std::printf("    %-20s %10s %10s %12s %12s   (key > 16 B: does not compile)\n", "art_map", "—", "—", "—", "—");
   return int(g_sink & 1) & 0;
}

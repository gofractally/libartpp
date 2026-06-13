// bench_buffer_array — the fair-to-the-trees variant of bench_buffer: the comparison maps
// key on std::array<unsigned char,32> (a fixed-size VALUE type stored INLINE in the node),
// not std::string (a heap allocation per key). This isolates the data-structure cost from
// std::string's per-key malloc + cache-scattering. unsigned char ⇒ operator< is unsigned
// lexicographic = the same byte order artpp/libart use (so lower_bound answers the same
// question). artpp/libart key on the identical 32 bytes (string_view / raw), so their numbers
// match the std::string run — they're anchors here; only the std/btree/hash maps change.
//   artpp_bench_buffer_array 1000000 7
#include <artpp/map.hpp>
#include <artpp/pool.hpp>

#include <absl/container/btree_map.h>

#include "art.h"  // vendored upstream libart

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <map>
#include <random>
#include <string_view>
#include <unordered_map>
#include <vector>

using Key = std::array<unsigned char, 32>;
struct KeyHash  // FNV-1a over the 4 little-endian words
{
   std::size_t operator()(const Key& a) const noexcept
   {
      std::uint64_t w[4];
      std::memcpy(w, a.data(), 32);
      std::uint64_t h = 1469598103934665603ull;
      for (int i = 0; i < 4; ++i) { h ^= w[i]; h *= 1099511628211ull; }
      return h;
   }
};
static std::string_view sv(const Key& k) { return std::string_view{reinterpret_cast<const char*>(k.data()), k.size()}; }

using clk = std::chrono::steady_clock;
static double now_ns() { return double(std::chrono::duration_cast<std::chrono::nanoseconds>(clk::now().time_since_epoch()).count()); }
template <class F>
static double best_ns(F&& fn, std::size_t n, int reps)
{
   double b = 1e30;
   for (int r = 0; r < reps; ++r) { double t0 = now_ns(); fn(); b = std::min(b, (now_ns() - t0) / double(n)); }
   return b;
}
static void prow(const char* name, double ins, double hit, double scan, double lb)
{
   char sc[16], lc[16];
   if (scan < 0) std::snprintf(sc, sizeof sc, "%12s", "n/a"); else std::snprintf(sc, sizeof sc, "%12.2f", scan);
   if (lb < 0)   std::snprintf(lc, sizeof lc, "%12s", "n/a"); else std::snprintf(lc, sizeof lc, "%12.1f", lb);
   std::printf("    %-22s %10.1f %10.1f %s %s\n", name, ins, hit, sc, lc);
}

static std::vector<Key>      g_keys, g_query, g_probe;
static volatile std::uint64_t g_sink = 0;

int main(int argc, char** argv)
{
   const std::size_t N        = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 1000000;
   const int         reps     = argc > 2 ? std::atoi(argv[2]) : 5;
   const std::string amode    = argc > 3 ? argv[3] : "both";
   const int         ins_reps = N >= 10'000'000 ? 1 : std::min(reps, 3);
   std::mt19937_64   rng(42);
   auto mk = [&](Key& k) { auto* p = reinterpret_cast<std::uint64_t*>(k.data()); for (int i = 0; i < 4; ++i) p[i] = rng(); };
   g_keys.resize(N); for (auto& k : g_keys) mk(k);
   const std::size_t M = std::min<std::size_t>(N, 5'000'000);
   g_query.assign(g_keys.begin(), g_keys.begin() + M); std::shuffle(g_query.begin(), g_query.end(), rng);
   g_probe.resize(M); for (auto& k : g_probe) mk(k);

   std::printf("array<u8,32> keys  N=%zu  (ns/op, min of %d reps; ord-scan = ns/elem)\n", N, reps);
   std::printf("    %-22s %10s %10s %12s %12s\n", "", "insert", "hit", "ord-scan", "lower_bnd");

   using A = artpp::pool_alloc<uint64_t>;
   if (amode != "std")
   {  // artpp anchor (string_view over the 32 bytes) — should match the std::string run
      double ins = best_ns([&] { artpp::line_pool p; artpp::map<std::string_view, uint64_t, artpp::mode::none, A> m{A{&p}}; for (std::size_t i = 0; i < N; ++i) m.insert(sv(g_keys[i]), i); g_sink += m.size(); }, N, ins_reps);
      artpp::line_pool pool; artpp::map<std::string_view, uint64_t, artpp::mode::none, A> m{A{&pool}};
      for (std::size_t i = 0; i < N; ++i) m.insert(sv(g_keys[i]), i);
      double hit = best_ns([&] { uint64_t s = 0, o; for (auto& k : g_query) if (m.find(sv(k), o)) s += o; g_sink += s; }, M, reps);
      double scn = best_ns([&] { uint64_t s = 0; m.for_each_value([&](const uint64_t& v) { s += v; }); g_sink += s; }, N, reps);
      double lb  = best_ns([&] { uint64_t s = 0; for (auto& k : g_probe) { auto it = m.lower_bound(sv(k)); if (it != m.end()) s += it.value(); } g_sink += s; }, M, reps);
      prow("artpp::map (pool)", ins, hit, scn, lb);
   }
   {  // artpp std::allocator anchor
      double ins = best_ns([&] { artpp::map<std::string_view, uint64_t> m; for (std::size_t i = 0; i < N; ++i) m.insert(sv(g_keys[i]), i); g_sink += m.size(); }, N, ins_reps);
      artpp::map<std::string_view, uint64_t> m; for (std::size_t i = 0; i < N; ++i) m.insert(sv(g_keys[i]), i);
      double hit = best_ns([&] { uint64_t s = 0, o; for (auto& k : g_query) if (m.find(sv(k), o)) s += o; g_sink += s; }, M, reps);
      double scn = best_ns([&] { uint64_t s = 0; m.for_each_value([&](const uint64_t& v) { s += v; }); g_sink += s; }, N, reps);
      double lb  = best_ns([&] { uint64_t s = 0; for (auto& k : g_probe) { auto it = m.lower_bound(sv(k)); if (it != m.end()) s += it.value(); } g_sink += s; }, M, reps);
      prow("artpp::map (std::alloc)", ins, hit, scn, lb);
   }
   {  // libart anchor (raw 32 bytes)
      double ins = best_ns([&] { art_tree t; art_tree_init(&t); for (std::size_t i = 0; i < N; ++i) art_insert(&t, g_keys[i].data(), 32, reinterpret_cast<void*>(uintptr_t(i + 1))); g_sink += art_size(&t); art_tree_destroy(&t); }, N, ins_reps);
      art_tree t; art_tree_init(&t);
      for (std::size_t i = 0; i < N; ++i) art_insert(&t, g_keys[i].data(), 32, reinterpret_cast<void*>(uintptr_t(i + 1)));
      double hit = best_ns([&] { uint64_t s = 0; for (auto& k : g_query) { void* r = art_search(&t, k.data(), 32); if (r) s += uint64_t(uintptr_t(r)); } g_sink += s; }, M, reps);
      art_tree_destroy(&t);
      prow("libart", ins, hit, -1, -1);
   }
   {  // absl::btree_map<array> — key inline, contiguous leaves
      double ins = best_ns([&] { absl::btree_map<Key, uint64_t> m; for (std::size_t i = 0; i < N; ++i) m[g_keys[i]] = i; g_sink += m.size(); }, N, ins_reps);
      absl::btree_map<Key, uint64_t> m; for (std::size_t i = 0; i < N; ++i) m[g_keys[i]] = i;
      double hit = best_ns([&] { uint64_t s = 0; for (auto& k : g_query) { auto it = m.find(k); if (it != m.end()) s += it->second; } g_sink += s; }, M, reps);
      double scn = best_ns([&] { uint64_t s = 0; for (auto& kv : m) s += kv.second; g_sink += s; }, N, reps);
      double lb  = best_ns([&] { uint64_t s = 0; for (auto& k : g_probe) { auto it = m.lower_bound(k); if (it != m.end()) s += it->second; } g_sink += s; }, M, reps);
      prow("absl::btree_map<array>", ins, hit, scn, lb);
   }
   {  // std::map<array> — key inline in the RB node
      double ins = best_ns([&] { std::map<Key, uint64_t> m; for (std::size_t i = 0; i < N; ++i) m[g_keys[i]] = i; g_sink += m.size(); }, N, ins_reps);
      std::map<Key, uint64_t> m; for (std::size_t i = 0; i < N; ++i) m[g_keys[i]] = i;
      double hit = best_ns([&] { uint64_t s = 0; for (auto& k : g_query) { auto it = m.find(k); if (it != m.end()) s += it->second; } g_sink += s; }, M, reps);
      double scn = best_ns([&] { uint64_t s = 0; for (auto& kv : m) s += kv.second; g_sink += s; }, N, reps);
      double lb  = best_ns([&] { uint64_t s = 0; for (auto& k : g_probe) { auto it = m.lower_bound(k); if (it != m.end()) s += it->second; } g_sink += s; }, M, reps);
      prow("std::map<array>", ins, hit, scn, lb);
   }
   {  // std::unordered_map<array> — inline key + custom 32-byte hash
      double ins = best_ns([&] { std::unordered_map<Key, uint64_t, KeyHash> m; m.reserve(N); for (std::size_t i = 0; i < N; ++i) m[g_keys[i]] = i; g_sink += m.size(); }, N, ins_reps);
      std::unordered_map<Key, uint64_t, KeyHash> m; m.reserve(N); for (std::size_t i = 0; i < N; ++i) m[g_keys[i]] = i;
      double hit = best_ns([&] { uint64_t s = 0; for (auto& k : g_query) { auto it = m.find(k); if (it != m.end()) s += it->second; } g_sink += s; }, M, reps);
      prow("std::unordered_map<array>", ins, hit, -1, -1);
   }
   return int(g_sink & 1) & 0;
}

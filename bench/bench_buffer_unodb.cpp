// bench_buffer_unodb — unodb (Laurynas Biveinis' canonical C++ ART) on random 32-byte
// keys, the IDENTICAL workload to bench_buffer.cpp, so the numbers drop straight into the
// site's #scaling comparison. unodb is the strongest pure-C++ ART peer: unlike art_map it
// keys on variable-length `key_view`, and it has ordered scan + iterator seek. On these
// 32-byte keys it trails artpp ~4× (its key_view iterator reconstructs the key; it's tuned
// for register-width keys) — but it's a real contestant, not an N/A.
//
// unodb is NOT vendored here (it needs Boost and its own .cpp). Build standalone:
//   git clone https://github.com/laurynas-biveinis/unodb /tmp/unodb
//   clang++ -std=c++20 -O3 -DNDEBUG -I/tmp/unodb -I<boost-include> \
//       bench/bench_buffer_unodb.cpp /tmp/unodb/art_internal.cpp -o unodb_buf_bench
//   ./unodb_buf_bench 100000000 3
//
// unodb's scan callback halts on `return true` (note: inverted vs many APIs) — return
// false to visit all (scan), true to stop after the first match (scan_from = lower_bound).
#include "art.hpp"  // unodb

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

using clk = std::chrono::steady_clock;
static double now_ns() { return double(std::chrono::duration_cast<std::chrono::nanoseconds>(clk::now().time_since_epoch()).count()); }
template <class F>
static double best_ns(F&& fn, std::size_t n, int reps)
{
   double b = 1e30;
   for (int r = 0; r < reps; ++r) { double t0 = now_ns(); fn(); b = std::min(b, (now_ns() - t0) / double(n)); }
   return b;
}
static unodb::key_view kv(const std::string& s) { return unodb::key_view{reinterpret_cast<const std::byte*>(s.data()), s.size()}; }

int main(int argc, char** argv)
{
   const std::size_t N        = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 1000000;
   const int         reps     = argc > 2 ? std::atoi(argv[2]) : 5;
   const int         ins_reps = N >= 10000000 ? 1 : std::min(reps, 3);
   std::mt19937_64   rng(42);
   auto mk = [&](std::string& s) { s.resize(32); auto* p = reinterpret_cast<std::uint64_t*>(s.data()); for (int i = 0; i < 4; ++i) p[i] = rng(); };
   std::vector<std::string> keys(N); for (auto& s : keys) mk(s);
   const std::size_t M = std::min<std::size_t>(N, 5000000);
   std::vector<std::string> query(keys.begin(), keys.begin() + M); std::shuffle(query.begin(), query.end(), rng);
   std::vector<std::string> probe(M); for (auto& s : probe) mk(s);
   volatile std::uint64_t sink = 0;
   double ins = best_ns([&] { unodb::db<unodb::key_view, std::uint64_t> t; for (std::size_t i = 0; i < N; ++i) (void)t.insert(kv(keys[i]), i); sink += t.empty() ? 0 : 1; }, N, ins_reps);
   unodb::db<unodb::key_view, std::uint64_t> t; for (std::size_t i = 0; i < N; ++i) (void)t.insert(kv(keys[i]), i);
   double hit = best_ns([&] { std::uint64_t s = 0; for (auto& k : query) { auto g = t.get(kv(k)); if (g) s += *g; } sink += s; }, M, reps);
   double scn = best_ns([&] { std::uint64_t s = 0; t.scan([&](auto& v) { s += v.get_value(); return false; }); sink += s; }, N, reps);  // false = continue
   double lb  = best_ns([&] { std::uint64_t s = 0; for (auto& k : probe) { t.scan_from(kv(k), [&](auto& v) { s += v.get_value(); return true; }, true); } sink += s; }, M, reps);  // true = stop after first >= key
   std::printf("unodb  N=%zu  insert=%.1f  hit=%.1f  ord-scan=%.2f  lower_bound=%.1f  (sink=%llu)\n",
               N, ins, hit, scn, lb, (unsigned long long)sink);
   return int(sink & 1) & 0;
}

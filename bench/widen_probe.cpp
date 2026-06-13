// widen_probe — the occupancy + perf-per-byte diagnostic behind the capacity-hint policy.
// Builds artpp::map in mode::flat_full (the default: setlist→node_full direct) vs the cnode
// density ladder (mode::dense_tiers: setlist→c2→c4→full) on the four bench workloads, and dumps:
// node-kind census, node_full OCCUPANCY histogram (how sparse are the fulls?), branches held in
// cnodes, find/lower_bound timings, and — over the flagship line_pool — bytes/key + a perf-per-
// byte composite. Findings that set compact_map's crossover: occupancy is bimodal (~30 or ~256,
// little between) and DENSIFIES with N, so the ladder is a perf-per-byte win only for small maps
// (<=~3M); past that it converges to node_full and only adds build churn + pool fragmentation.
//   widen_probe [N]
#include <artpp/map.hpp>
#include <artpp/pool.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
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
static std::string be(std::uint64_t v)  // big-endian u64 bytes ⇒ radix order == numeric order
{
   std::string k(8, '\0');
   for (int i = 0; i < 8; ++i) k[size_t(i)] = char(v >> (56 - 8 * i));
   return k;
}

template <artpp::mode M>
static void run(const char* modename, const char* wl, const std::vector<std::string>& keys,
                const std::vector<std::string>& probe)
{
   double ins = best_ns([&] { artpp::map<std::string_view, std::uint64_t, M> mm;
                               for (std::size_t i = 0; i < keys.size(); ++i) mm.insert(keys[i], i); }, keys.size(), 3);
   artpp::map<std::string_view, std::uint64_t, M> m;
   for (std::size_t i = 0; i < keys.size(); ++i) m.insert(keys[i], i);
   auto                   d    = m.debug_stats();
   const std::size_t      M2   = probe.size();
   volatile std::uint64_t sink = 0;
   double hit = best_ns([&] { std::uint64_t s = 0, o; for (auto& k : probe) if (m.find(k, o)) s += o; sink += s; }, M2, 7);
   double lb  = best_ns([&] { std::uint64_t s = 0; for (auto& k : probe) { auto it = m.lower_bound(k); if (it != m.end()) s += it.value(); } sink += s; }, M2, 7);
   const std::uint64_t nfull = d.full + d.fullp;
   std::printf("%-12s %-10s | sl %llu c2 %llu c4 %llu c8 %llu full %llu(+%llu) | "
               "occ <48 %llu  48-96 %llu  96-160 %llu  160-192 %llu  >=192 %llu  mean %.0f | "
               "cn-br %llu | hit %.1f lb %.1f\n",
               modename, wl, (unsigned long long)d.setlist, (unsigned long long)d.c2, (unsigned long long)d.c4,
               (unsigned long long)d.c8, (unsigned long long)d.full, (unsigned long long)d.fullp,
               (unsigned long long)d.full_lt48, (unsigned long long)d.full_48_96, (unsigned long long)d.full_96_160,
               (unsigned long long)d.full_160_192, (unsigned long long)d.full_ge192,
               nfull ? double(d.full_nbr_sum) / double(nfull) : 0.0, (unsigned long long)d.cn_nbr_sum, hit, lb);
   std::printf("  %*sinsert %.1f\n", 24, "", ins);
   (void)sink;
}

// Typed-u64 path (inline values, the path the stale "+70% insert" note was measured on).
template <artpp::mode M>
static void run_u64(const char* modename, const char* wl, const std::vector<std::uint64_t>& keys,
                    const std::vector<std::uint64_t>& probe)
{
   double ins = best_ns([&] { artpp::map<std::uint64_t, std::uint64_t, M> mm;
                               for (std::size_t i = 0; i < keys.size(); ++i) mm.insert(keys[i], i); }, keys.size(), 3);
   artpp::map<std::uint64_t, std::uint64_t, M> m;
   for (std::size_t i = 0; i < keys.size(); ++i) m.insert(keys[i], i);
   auto                   d    = m.debug_stats();
   const std::size_t      M2   = probe.size();
   volatile std::uint64_t sink = 0;
   double hit = best_ns([&] { std::uint64_t s = 0, o; for (auto k : probe) if (m.find(k, o)) s += o; sink += s; }, M2, 7);
   double lb  = best_ns([&] { std::uint64_t s = 0; for (auto k : probe) { auto it = m.lower_bound(k); if (it != m.end()) s += it.value(); } sink += s; }, M2, 7);
   const std::uint64_t nfull = d.full + d.fullp;
   std::printf("%-12s %-10s | sl %llu c2 %llu c4 %llu c8 %llu full %llu(+%llu) | occ <48 %llu  >=192 %llu  mean %.0f | "
               "cn-br %llu | insert %.1f hit %.1f lb %.1f\n",
               modename, wl, (unsigned long long)d.setlist, (unsigned long long)d.c2, (unsigned long long)d.c4,
               (unsigned long long)d.c8, (unsigned long long)d.full, (unsigned long long)d.fullp,
               (unsigned long long)d.full_lt48, (unsigned long long)d.full_ge192,
               nfull ? double(d.full_nbr_sum) / double(nfull) : 0.0, (unsigned long long)d.cn_nbr_sum, ins, hit, lb);
   (void)sink;
}

// Memory + perf-per-byte: build over a line_pool (the flagship allocator), read its logical
// footprint (node lines * 128 + terminal units * 16) and report bytes/key alongside latency.
// The composite lat*B/key (ns * bytes/key, lower=better) is the time-AND-space cost of an op —
// a smaller structure caches better and holds more keys per RAM byte, which raw latency hides.
// uint64_t VALUES (8 > inline_cap 5) ⇒ external leaves, so the terminal region is exercised.
template <artpp::mode M>
static void run_pool(const char* modename, const char* wl, const std::vector<std::uint64_t>& keys,
                     const std::vector<std::uint64_t>& probe)
{
   using A    = artpp::pool_alloc<std::uint64_t>;
   double ins = best_ns([&] { artpp::line_pool p; artpp::map<std::uint64_t, std::uint64_t, M, A> mm{A{&p}};
                              for (std::size_t i = 0; i < keys.size(); ++i) mm.insert(keys[i], i); }, keys.size(), 3);
   artpp::line_pool                              pool;
   artpp::map<std::uint64_t, std::uint64_t, M, A> m{A{&pool}};
   for (std::size_t i = 0; i < keys.size(); ++i) m.insert(keys[i], i);
   const std::size_t      M2   = probe.size();
   volatile std::uint64_t sink = 0;
   double hit = best_ns([&] { std::uint64_t s = 0, o; for (auto k : probe) if (m.find(k, o)) s += o; sink += s; }, M2, 7);
   double lb  = best_ns([&] { std::uint64_t s = 0; for (auto k : probe) { auto it = m.lower_bound(k); if (it != m.end()) s += it.value(); } sink += s; }, M2, 7);
   const double bytes = double(pool.used_lines()) * 128.0 + double(pool.used_term_units()) * 16.0;
   const double bpk   = bytes / double(keys.size());
   std::printf("%-12s %-10s | bytes/key %5.1f (%.0f MB) | insert %.1f hit %.1f lb %.1f | "
               "perf*byte: ins %.0f hit %.0f lb %.0f\n",
               modename, wl, bpk, bytes / 1e6, ins, hit, lb, ins * bpk, hit * bpk, lb * bpk);
   (void)sink;
}

static std::vector<std::string> probe_of(const std::vector<std::string>& src, std::mt19937_64& rng)
{
   const std::size_t M2 = std::min<std::size_t>(src.size(), 2000000);
   std::vector<std::string> p(src.begin(), src.begin() + M2);
   std::shuffle(p.begin(), p.end(), rng);
   return p;
}

int main(int argc, char** argv)
{
   const std::size_t N = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 2000000;
   std::mt19937_64   rng(42);

   std::vector<std::string> dict;
   { std::ifstream f("/usr/share/dict/words"); std::string w; while (std::getline(f, w)) if (!w.empty()) dict.push_back(w); }

   std::vector<std::string> uni(N), seqk(N), clu(N);
   for (std::size_t i = 0; i < N; ++i) uni[i]  = be(rng());
   for (std::size_t i = 0; i < N; ++i) seqk[i] = be(i);
   for (std::size_t i = 0; i < N; ++i) clu[i]  = be((rng() & 0xFFFFu) | (std::uint64_t(i & 0xFF) << 16));

   auto pu = probe_of(uni, rng), ps = probe_of(seqk, rng), pc = probe_of(clu, rng);

   std::printf("widen_probe N=%zu  dict=%zu  (hit/lb = ns/op min of 7)\n", N, dict.size());
   if (dict.size() > 1000)
   {
      auto pd = probe_of(dict, rng);
      run<artpp::mode::flat_full>("flat", "dict", dict, pd);
      run<artpp::mode::dense_tiers>("ladder", "dict", dict, pd);
   }
   run<artpp::mode::flat_full>("flat", "uniform", uni, pu);
   run<artpp::mode::dense_tiers>("ladder", "uniform", uni, pu);
   run<artpp::mode::flat_full>("flat", "sequential", seqk, ps);
   run<artpp::mode::dense_tiers>("ladder", "sequential", seqk, ps);
   run<artpp::mode::flat_full>("flat", "clustered", clu, pc);
   run<artpp::mode::dense_tiers>("ladder", "clustered", clu, pc);

   std::printf("\n-- typed u64 path (inline values) --\n");
   std::vector<std::uint64_t> u64u(N), u64s(N), u64c(N);
   std::mt19937_64            rng2(42);
   for (std::size_t i = 0; i < N; ++i) u64u[i] = rng2();
   for (std::size_t i = 0; i < N; ++i) u64s[i] = i;
   for (std::size_t i = 0; i < N; ++i) u64c[i] = (rng2() & 0xFFFFu) | (std::uint64_t(i & 0xFF) << 16);
   const std::size_t          M2 = std::min<std::size_t>(N, 2000000);
   std::vector<std::uint64_t> qu(u64u.begin(), u64u.begin() + M2), qs(u64s.begin(), u64s.begin() + M2),
       qc(u64c.begin(), u64c.begin() + M2);
   std::shuffle(qu.begin(), qu.end(), rng2); std::shuffle(qs.begin(), qs.end(), rng2); std::shuffle(qc.begin(), qc.end(), rng2);
   run_u64<artpp::mode::flat_full>("flat", "uniform", u64u, qu);
   run_u64<artpp::mode::dense_tiers>("ladder", "uniform", u64u, qu);
   run_u64<artpp::mode::flat_full>("flat", "sequential", u64s, qs);
   run_u64<artpp::mode::dense_tiers>("ladder", "sequential", u64s, qs);
   run_u64<artpp::mode::flat_full>("flat", "clustered", u64c, qc);
   run_u64<artpp::mode::dense_tiers>("ladder", "clustered", u64c, qc);

   std::printf("\n-- memory + perf-per-byte (line_pool, u64 value ⇒ external leaves) --\n");
   run_pool<artpp::mode::flat_full>("flat", "uniform", u64u, qu);
   run_pool<artpp::mode::dense_tiers>("ladder", "uniform", u64u, qu);
   run_pool<artpp::mode::flat_full>("flat", "sequential", u64s, qs);
   run_pool<artpp::mode::dense_tiers>("ladder", "sequential", u64s, qs);
   run_pool<artpp::mode::flat_full>("flat", "clustered", u64c, qc);
   run_pool<artpp::mode::dense_tiers>("ladder", "clustered", u64c, qc);
   return 0;
}

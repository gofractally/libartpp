// set_bench — artpp::set<std::string_view> vs std::set<std::string> at 2M keys.
//
// Two metrics per workload:
//   * find ns/op   — min-of-N timed passes over a SHUFFLED probe order (all keys present),
//                    so the number is the warm steady-state lookup cost, not a cold-cache fluke.
//   * bytes/key    — a counting allocator wraps BOTH structures: every allocate/deallocate is
//                    summed, peak live bytes recorded, and divided by the key count. For artpp
//                    that's the node_block(128B)/term_block(16B) runs; for std::set the RB-tree
//                    nodes (the std::string keys here are <= SSO, so no separate char heap).
//
// Three workloads (2M keys each):
//   (1) random 16-byte         — full-entropy binary keys (no shared structure)
//   (2) random alphanumeric 16 — [0-9A-Za-z] 16-byte keys (text-like, still mostly distinct)
//   (3) SHORT 3-7 byte         — exercises the suffix-in-handle path (most terminals inline)
//
// Build: /opt/homebrew/opt/llvm/bin/clang++ -O3 -DNDEBUG -std=c++20 -I include
//                                            bench/set_bench.cpp -o set_bench
#include "artpp/map.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <random>
#include <set>
#include <string>
#include <string_view>
#include <vector>

// ── counting allocator: tracks live (and peak) bytes across all rebinds ─────────
struct counter
{
   std::size_t live = 0, peak = 0;
   void        add(std::size_t n) { live += n; if (live > peak) peak = live; }
   void        sub(std::size_t n) { live -= n; }
};
template <class T>
struct count_alloc
{
   using value_type = T;
   counter* c;
   count_alloc(counter* cc) noexcept : c(cc) {}
   template <class U>
   count_alloc(const count_alloc<U>& o) noexcept : c(o.c) {}
   T* allocate(std::size_t n)
   {
      const std::size_t b = n * sizeof(T);
      c->add(b);
      // artpp's node/term blocks are over-aligned (128 / 16); honor the type's alignment.
      return static_cast<T*>(::operator new(b, std::align_val_t(alignof(T) < 16 ? 16 : alignof(T))));
   }
   void deallocate(T* p, std::size_t n) noexcept
   {
      c->sub(n * sizeof(T));
      ::operator delete(p, std::align_val_t(alignof(T) < 16 ? 16 : alignof(T)));
   }
   template <class U>
   bool operator==(const count_alloc<U>& o) const noexcept { return c == o.c; }
   template <class U>
   bool operator!=(const count_alloc<U>& o) const noexcept { return c != o.c; }
};

using clk = std::chrono::steady_clock;
static double ns_per(const std::function<std::size_t()>& f, int passes, std::size_t nprobe)
{
   double best = 1e30;
   for (int p = 0; p < passes; ++p)
   {
      auto         t0 = clk::now();
      std::size_t  acc = f();
      auto         t1  = clk::now();
      asm volatile("" : : "r"(acc) : "memory");  // keep the lookup result live
      const double ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / double(nprobe);
      best            = std::min(best, ns);
   }
   return best;
}

// ── one workload row ────────────────────────────────────────────────────────────
static void row(const char* name, const std::vector<std::string>& keys, std::mt19937_64& rng)
{
   const std::size_t N = keys.size();
   // shuffled probe order (all present) — warm steady-state lookup
   std::vector<std::uint32_t> order(N);
   for (std::size_t i = 0; i < N; ++i) order[i] = std::uint32_t(i);
   std::shuffle(order.begin(), order.end(), rng);

   // ── artpp::set<string_view> over a counting allocator ──
   double aps_ns, sps_ns;
   double aps_bpk, sps_bpk;
   {
      counter ctr;
      artpp::set<std::string_view, artpp::mode::none, count_alloc<std::string_view>> s{
          count_alloc<std::string_view>{&ctr}};
      for (auto& k : keys) s.insert(k);
      aps_bpk = double(ctr.live) / double(N);
      aps_ns  = ns_per(
          [&] { std::size_t hit = 0; for (auto i : order) hit += s.contains(keys[i]); return hit; },
          7, N);
   }
   // ── std::set<string> over a counting allocator ──
   {
      counter ctr;
      std::set<std::string, std::less<>, count_alloc<std::string>> s{
          std::less<>{}, count_alloc<std::string>{&ctr}};
      for (auto& k : keys) s.insert(k);
      sps_bpk = double(ctr.live) / double(N);
      sps_ns  = ns_per(
          [&] { std::size_t hit = 0; for (auto i : order) hit += (s.find(keys[i]) != s.end()); return hit; },
          7, N);
   }
   std::printf("%-22s  artpp: %6.1f ns/op  %6.1f B/key   |   std::set: %6.1f ns/op  %6.1f B/key\n",
               name, aps_ns, aps_bpk, sps_ns, sps_bpk);
}

int main(int argc, char** argv)
{
   const std::size_t N = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 2'000'000;
   std::mt19937_64   rng(0xBEEF);
   std::printf("artpp::set<string_view> vs std::set<string>  (N=%zu, min-of-7 shuffled probe)\n\n", N);

   {  // (1) random 16-byte binary
      std::vector<std::string> keys;
      keys.reserve(N);
      for (std::size_t i = 0; i < N; ++i)
      {
         std::string k(16, '\0');
         for (auto& c : k) c = char(rng());
         keys.push_back(std::move(k));
      }
      std::sort(keys.begin(), keys.end());
      keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
      row("random 16B", keys, rng);
   }
   {  // (2) random alphanumeric 16-byte
      static const char A[] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
      std::vector<std::string> keys;
      keys.reserve(N);
      for (std::size_t i = 0; i < N; ++i)
      {
         std::string k(16, '\0');
         for (auto& c : k) c = A[rng() % 62];
         keys.push_back(std::move(k));
      }
      std::sort(keys.begin(), keys.end());
      keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
      row("random alnum 16B", keys, rng);
   }
   {  // (3) SHORT keys 3-7 bytes (suffix-in-handle path)
      static const char A[] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
      std::vector<std::string> keys;
      keys.reserve(N);
      for (std::size_t i = 0; i < N; ++i)
      {
         const int   len = 3 + int(rng() % 5);  // 3..7
         std::string k(size_t(len), '\0');
         for (auto& c : k) c = A[rng() % 62];
         keys.push_back(std::move(k));
      }
      std::sort(keys.begin(), keys.end());
      keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
      row("short 3-7B", keys, rng);
   }
   return 0;
}

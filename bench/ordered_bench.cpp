// ordered_bench — artpp::set against the ordered standard-library / absl analogues, same
// keys, same counting allocator, at N keys (default 2M).
//
//   artpp::set   vs   absl::btree_set   vs   std::set
//
// Metrics per contestant:
//   * find ns/op  — min-of-7 timed passes over a SHUFFLED all-present probe order (warm
//                   steady-state lookup, not a cold-cache fluke).
//   * insert ns   — wall time to build the structure from empty / key count.
//   * bytes/key   — a counting allocator wraps every contestant; peak live bytes / key count.
//
// The keys vector holds std::strings, so probing is allocation-free for everyone: the
// std-library contestants probe with keys[i] directly, artpp with a string_view over it.
//
// Build (needs absl): CMake target artpp_ordered_bench under -DARTPP_BENCHMARKS=ON.
#include "artpp/map.hpp"

#include <absl/container/btree_set.h>

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

// ── counting allocator ──────────────────────────────────────────────────────────────────
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
      return static_cast<T*>(
          ::operator new(b, std::align_val_t(alignof(T) < 16 ? 16 : alignof(T))));
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
template <class F>
static double ns_per(F&& f, int passes, std::size_t nprobe)
{
   double best = 1e30;
   for (int p = 0; p < passes; ++p)
   {
      auto         t0  = clk::now();
      std::size_t  acc = f();
      auto         t1  = clk::now();
      asm volatile("" : : "r"(acc) : "memory");
      best = std::min(best, std::chrono::duration<double, std::nano>(t1 - t0).count() / double(nprobe));
   }
   return best;
}

struct result { double find_ns, ins_ns, bpk; };

// artpp::set: built with count_alloc<string_view>, probed with string_view over keys[i].
static result run_artpp(const std::vector<std::string>& keys, const std::vector<std::uint32_t>& order)
{
   using sv = std::string_view;
   counter ctr;
   artpp::set<sv, artpp::mode::none, count_alloc<sv>> s{count_alloc<sv>{&ctr}};
   auto t0 = clk::now();
   for (auto& k : keys) s.insert(sv(k));
   auto t1 = clk::now();
   const double ins = std::chrono::duration<double, std::nano>(t1 - t0).count() / double(keys.size());
   const double bpk = double(ctr.live) / double(keys.size());
   const double fns = ns_per(
       [&] { std::size_t h = 0; for (auto i : order) h += s.contains(sv(keys[i])); return h; },
       7, keys.size());
   return {fns, ins, bpk};
}

// std-like (absl btree, std::set): built with count_alloc<string>, probed with keys[i].
template <class S>
static result run_stdlike(const std::vector<std::string>& keys, const std::vector<std::uint32_t>& order)
{
   counter ctr;
   S       s{count_alloc<std::string>{&ctr}};
   auto    t0 = clk::now();
   for (auto& k : keys) s.insert(k);
   auto         t1  = clk::now();
   const double ins = std::chrono::duration<double, std::nano>(t1 - t0).count() / double(keys.size());
   const double bpk = double(ctr.live) / double(keys.size());
   const double fns = ns_per(
       [&] { std::size_t h = 0; for (auto i : order) h += s.count(keys[i]); return h; },
       7, keys.size());
   return {fns, ins, bpk};
}

static void row(const char* name, const std::vector<std::string>& keys, std::mt19937_64& rng)
{
   const std::size_t          N = keys.size();
   std::vector<std::uint32_t> order(N);
   for (std::size_t i = 0; i < N; ++i) order[i] = std::uint32_t(i);
   std::shuffle(order.begin(), order.end(), rng);

   auto a = run_artpp(keys, order);
   auto b = run_stdlike<absl::btree_set<std::string, std::less<>, count_alloc<std::string>>>(keys, order);
   auto s = run_stdlike<std::set<std::string, std::less<>, count_alloc<std::string>>>(keys, order);

   std::printf("\n== %s (N=%zu) ==\n", name, N);
   std::printf("  %-16s find %7.1f ns   insert %6.1f ns   %6.1f B/key\n", "artpp::set", a.find_ns, a.ins_ns, a.bpk);
   std::printf("  %-16s find %7.1f ns   insert %6.1f ns   %6.1f B/key   (find %.1fx)\n",
               "absl::btree_set", b.find_ns, b.ins_ns, b.bpk, b.find_ns / a.find_ns);
   std::printf("  %-16s find %7.1f ns   insert %6.1f ns   %6.1f B/key   (find %.1fx)\n",
               "std::set", s.find_ns, s.ins_ns, s.bpk, s.find_ns / a.find_ns);
}

int main(int argc, char** argv)
{
   const std::size_t N = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 2'000'000;
   std::mt19937_64   rng(0xBEEF);
   std::printf("artpp::set vs absl::btree_set vs std::set  (min-of-7 shuffled probe, counting allocator)\n");

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
         const int   len = 3 + int(rng() % 5);
         std::string k(std::size_t(len), '\0');
         for (auto& c : k) c = A[rng() % 62];
         keys.push_back(std::move(k));
      }
      std::sort(keys.begin(), keys.end());
      keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
      row("short 3-7B", keys, rng);
   }
   return 0;
}

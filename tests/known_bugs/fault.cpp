// Fault-injection sweep: random clustered op sequence, fail the n-th allocation (or n-th value
// construction), then verify the tree against a std::map oracle, keep going, check leaks.
// usage: fault <cfg> <valtype 0=str 1=u64 2=u32> <seed> <nops> [mode: a=alloc v=valuector]
#include "artpp/map.hpp"
#include <cstdio>
#include <cstdlib>
#include <map>
#include <random>
#include <string>
#include <vector>

using artpp::map;
using artpp::mode;

struct fail_counter { long countdown = -1; long allocs = 0, frees = 0; long total = 0; };
static long g_live = 0;
static long g_vcount = -1;  // value-ctor countdown
static bool g_arm = false;
struct counted
{
   std::string s;
   counted() { ++g_live; }
   counted(std::string v) : s(std::move(v))
   {
      if (g_arm && g_vcount >= 0 && g_vcount-- == 0) throw std::runtime_error("ctor");
      ++g_live;
   }
   counted(const counted& o) : s(o.s) { ++g_live; }
   counted(counted&& o) noexcept : s(std::move(o.s)) { ++g_live; }
   counted& operator=(const counted&) = default;
   counted& operator=(counted&&) noexcept = default;
   ~counted() { --g_live; }
   bool operator==(const counted& o) const { return s == o.s; }
};
template <class T>
struct throwing_alloc
{
   using value_type = T;
   fail_counter* fc;
   explicit throwing_alloc(fail_counter* f) noexcept : fc(f) {}
   template <class U> throwing_alloc(const throwing_alloc<U>& o) noexcept : fc(o.fc) {}
   T* allocate(std::size_t n)
   {
      ++fc->total;
      if (fc->countdown >= 0 && fc->countdown-- == 0) throw std::bad_alloc();
      ++fc->allocs;
      return static_cast<T*>(::operator new(n * sizeof(T), std::align_val_t(alignof(T))));
   }
   void deallocate(T* p, std::size_t) noexcept { ++fc->frees; ::operator delete(p, std::align_val_t(alignof(T))); }
   template <class U> bool operator==(const throwing_alloc<U>& o) const noexcept { return fc == o.fc; }
   template <class U> bool operator!=(const throwing_alloc<U>& o) const noexcept { return fc != o.fc; }
};

static std::string vstr(const counted& c) { return c.s; }
static std::string vstr(uint64_t v) { return std::to_string(v); }
static std::string vstr(uint32_t v) { return std::to_string(v); }
template <class V> static V mk(uint64_t x);
template <> counted  mk<counted>(uint64_t x) { return counted(std::string("value_padding_to_heap_") + std::to_string(x)); }
template <> uint64_t mk<uint64_t>(uint64_t x) { return x; }
template <> uint32_t mk<uint32_t>(uint64_t x) { return uint32_t(x); }

struct Op { int kind; std::string key; uint64_t v; };  // 0 insert 1 emplace 2 erase

static std::vector<Op> gen(uint64_t seed, int nops)
{
   std::mt19937_64 rng(seed);
   std::vector<Op> ops;
   std::vector<std::string> seen;
   static const char* pfx[] = {"", "pre", "prefixlong_", "p", "prefixlong_and_longer_still_"};
   for (int i = 0; i < nops; ++i)
   {
      Op o; o.v = rng() % 100000;
      const int r = int(rng() % 10);
      if (r >= 8 && !seen.empty()) { o.kind = 2; o.key = seen[rng() % seen.size()]; }
      else
      {
         o.kind = (r == 7) ? 1 : 0;
         if (r < 2 && !seen.empty())
         {
            o.key = seen[rng() % seen.size()];
            int m = int(rng() % 3);
            if (m == 0 && !o.key.empty()) o.key.pop_back();
            else if (m == 1) o.key.push_back(char('a' + rng() % 3));
         }
         else
         {
            o.key = pfx[rng() % 5];
            o.key.push_back(char('A' + rng() % 40));
            int tl = int(rng() % 4) == 0 ? int(rng() % 60) : int(rng() % 6);
            for (int j = 0; j < tl; ++j) o.key.push_back(char('a' + rng() % 2));
         }
         seen.push_back(o.key);
      }
      ops.push_back(o);
   }
   return ops;
}

template <class Map, class V>
static std::map<std::string, std::string> dump(const Map& m)
{
   std::map<std::string, std::string> d;
   size_t n = 0;
   for (auto it = m.begin(); it != m.end(); ++it, ++n) d[std::string(it.key_bytes())] = vstr(it.value());
   if (n != d.size()) { std::printf("  DUP KEYS in iteration\n"); }
   return d;
}

template <class Map, class V>
static int run(const char* name, uint64_t seed, int nops, bool valmode)
{
   const auto ops = gen(seed, nops);
   long total = 0;
   int  bad = 0;
   for (long n = -1;; ++n)
   {
      fail_counter fc;
      const long   vbase = g_live;
      bool         threw = false;
      {
         Map m{typename Map::allocator_type(&fc)};
         std::map<std::string, std::string> ref;
         if (valmode) g_vcount = n; else fc.countdown = n;
         for (size_t i = 0; i < ops.size(); ++i)
         {
            const Op& o = ops[i];
            auto before = ref;
            auto after  = ref;
            if (o.kind == 0) after[o.key] = vstr(mk<V>(o.v));
            else if (o.kind == 1) after.emplace(o.key, vstr(mk<V>(o.v)));
            else after.erase(o.key);
            bool t = false;
            try
            {
               if (o.kind == 0) m.insert(o.key, mk<V>(o.v));
               else if (o.kind == 1)
               {
                  if constexpr (std::is_same_v<V, counted>)
                  {
                     std::string a = std::string("value_padding_to_heap_") + std::to_string(o.v);
                     struct R { ~R() { g_arm = false; } } r_;
                     g_arm = true;
                     m.emplace(o.key, std::move(a));
                  }
                  else m.emplace(o.key, mk<V>(o.v));
               }
               else m.erase(o.key);
            }
            catch (const std::bad_alloc&) { t = true; }
            catch (const std::runtime_error&) { t = true; }
            if (t)
            {
               threw = true;
               fc.countdown = -1; g_vcount = -1;
               auto d = dump<Map, V>(m);
               const bool sb = d == before, sa = d == after;
               if (!sb && !sa)
               {
                  std::printf("  %s seed=%llu fault#%ld op#%zu kind=%d key='%s': CONTENT DIVERGED (size %zu, expect %zu or %zu, m.size()=%zu)\n",
                              name, (unsigned long long)seed, n, i, o.kind, o.key.c_str(), d.size(), before.size(), after.size(), m.size());
                  ++bad;
                  ref = d;  // resync and continue
               }
               else
               {
                  ref = sb ? before : after;
                  if (m.size() != ref.size())
                  { std::printf("  %s fault#%ld op#%zu: size() %zu != %zu\n", name, n, i, m.size(), ref.size()); ++bad; }
               }
               for (auto& kv : ref) if (!m.contains(kv.first)) { std::printf("  %s fault#%ld op#%zu: iterated key not findable\n", name, n, i); ++bad; break; }
            }
            else ref = after;
         }
         if (n == -1) total = valmode ? 1000000 : fc.total;
         auto d = dump<Map, V>(m);
         if (d != ref) { std::printf("  %s fault#%ld: FINAL CONTENT MISMATCH\n", name, n); ++bad; }
      }
      if (fc.allocs != fc.frees) { std::printf("  %s seed=%llu fault#%ld: MEM LEAK allocs=%ld frees=%ld\n", name, (unsigned long long)seed, n, fc.allocs, fc.frees); ++bad; }
      if (g_live != vbase) { std::printf("  %s seed=%llu fault#%ld: VALUE LIVE-COUNT off by %ld\n", name, (unsigned long long)seed, n, g_live - vbase); ++bad; g_live = vbase; }
      if (n >= 0 && !threw) break;  // past the last fault point
      if (bad > 8) break;
      (void)total;
   }
   std::printf("%s seed=%llu: %s\n", name, (unsigned long long)seed, bad ? "FAIL" : "ok");
   return bad;
}

template <class V>
static int dispatch(int cfg, uint64_t seed, int nops, bool vm)
{
   using A = throwing_alloc<V>;
   switch (cfg)
   {
      case 0: return run<map<std::string_view, V, mode::none, A>, V>("none", seed, nops, vm);
      case 1: return run<map<std::string_view, V, mode::buckets, A>, V>("buckets", seed, nops, vm);
      case 2: return run<map<std::string_view, V, mode::adaptive, A>, V>("adaptive", seed, nops, vm);
      case 3: return run<map<std::string_view, V, mode::dense_tiers, A>, V>("dense", seed, nops, vm);
      case 4: return run<map<std::string_view, V, mode::wide_stems, A>, V>("wide", seed, nops, vm);
      case 5: return run<map<std::string_view, V, mode::buckets | mode::dense_tiers, A>, V>("buckets+dense", seed, nops, vm);
      case 6: return run<map<std::string_view, V, mode::adaptive | mode::dense_tiers | mode::ladder_c8, A>, V>("adaptive+dense+c8", seed, nops, vm);
   }
   return 0;
}
int main(int argc, char** argv)
{
   const int      cfg  = argc > 1 ? atoi(argv[1]) : 0;
   const int      vt   = argc > 2 ? atoi(argv[2]) : 0;
   const uint64_t seed = argc > 3 ? strtoull(argv[3], nullptr, 10) : 1;
   const int      nops = argc > 4 ? atoi(argv[4]) : 250;
   const bool     vm   = argc > 5 && argv[5][0] == 'v';
   if (vt == 0) return dispatch<counted>(cfg, seed, nops, vm) ? 1 : 0;
   if (vt == 1) return dispatch<uint64_t>(cfg, seed, nops, vm) ? 1 : 0;
   return dispatch<uint32_t>(cfg, seed, nops, vm) ? 1 : 0;
}

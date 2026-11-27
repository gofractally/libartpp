// Targeted exception-safety reproducers. usage: repro_eh <scenario> <fault-index>
// Prints one line: what the faulted operation left behind. Run under ASan.
#include "artpp/map.hpp"
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>
using artpp::map;
using artpp::mode;

struct fail_counter { long countdown = -1; long allocs = 0, frees = 0; };
template <class T>
struct FA
{
   using value_type = T;
   fail_counter* fc;
   explicit FA(fail_counter* f) noexcept : fc(f) {}
   template <class U> FA(const FA<U>& o) noexcept : fc(o.fc) {}
   T* allocate(std::size_t n)
   {
      if (fc->countdown >= 0 && fc->countdown-- == 0) throw std::bad_alloc();
      ++fc->allocs;
      return static_cast<T*>(::operator new(n * sizeof(T), std::align_val_t(alignof(T))));
   }
   void deallocate(T* p, std::size_t) noexcept { ++fc->frees; ::operator delete(p, std::align_val_t(alignof(T))); }
   template <class U> bool operator==(const FA<U>& o) const noexcept { return fc == o.fc; }
};
static long g_live = 0, g_vthrow = -1;
struct TV  // value whose converting ctor can throw (emplace path)
{
   std::string s;
   TV() { ++g_live; }
   explicit TV(const char* c) : s(c)
   {
      if (g_vthrow >= 0 && g_vthrow-- == 0) throw std::runtime_error("ctor");
      ++g_live;
   }
   TV(TV&& o) noexcept : s(std::move(o.s)) { ++g_live; }
   TV& operator=(TV&&) noexcept = default;
   ~TV() { --g_live; }
};
static std::string vs(const std::string& s) { return s; }
static std::string vs(uint64_t v) { return std::to_string(v); }
static std::string vs(uint32_t v) { return std::to_string(v); }
static std::string vs(const TV& v) { return v.s; }

template <class M>
static std::map<std::string, std::string> dump(const M& m)
{
   std::map<std::string, std::string> d;
   for (auto it = m.begin(); it != m.end(); ++it) d[std::string(it.key_bytes())] = vs(it.value());
   return d;
}
template <class M, class Build, class OpF>
static void sweep(const char* name, long n, Build build, OpF op)
{
   fail_counter fc;
   const long   lb = g_live;
   {
      M m{typename M::allocator_type(&fc)};
      build(m);
      auto before = dump(m);
      const size_t sz0 = m.size();
      bool threw = false;
      fc.countdown = n;
      try { op(m); }
      catch (const std::bad_alloc&) { threw = true; }
      catch (const std::runtime_error&) { threw = true; }
      fc.countdown = -1; g_vthrow = -1;
      std::printf("%s fault#%ld: %s; size() %zu -> %zu; ", name, n, threw ? "THREW" : "completed", sz0, m.size());
      std::fflush(stdout);
      auto after = dump(m);
      size_t lost = 0, changed = 0;
      for (auto& kv : before)
      {
         auto it = after.find(kv.first);
         if (it == after.end()) ++lost; else if (it->second != kv.second) ++changed;
      }
      size_t unfindable = 0;
      for (auto& kv : before) if (!m.contains(kv.first)) ++unfindable;
      std::printf("iterated %zu keys (before %zu): %zu pre-existing keys LOST from iteration, %zu not findable by contains(); ",
                  after.size(), before.size(), lost, unfindable);
      (void)changed;
      std::fflush(stdout);
   }
   std::printf("allocs-frees=%ld live-values=%ld\n", fc.allocs - fc.frees, g_live - lb);
}

int main(int argc, char** argv)
{
   const int  sc = argc > 1 ? atoi(argv[1]) : 1;
   const long n  = argc > 2 ? atol(argv[2]) : 0;
   const std::string tail(40, 't');
   switch (sc)
   {
      case 1:  // buckets + non-trivial value: erase() destroys the value, THEN allocates the rebuilt bucket
      {
         using M = map<std::string_view, std::string, mode::buckets, FA<std::string>>;
         sweep<M>("S1 buckets/std::string erase", n,
                  [](M& m) { for (const char* k : {"alpha", "beta", "gamma"}) m.insert(k, std::string(60, 'v')); },
                  [](M& m) { m.erase("beta"); });
         break;
      }
      case 2:  // buckets: full 16-way setlist WITH an inline prefix, 17th child -> widen
      {
         using M = map<std::string_view, uint64_t, mode::buckets, FA<uint64_t>>;
         sweep<M>("S2 buckets/u64 widen prefixed setlist", n,
                  [&](M& m) {
                     for (int c = 0; c < 16; ++c)
                        for (int j = 0; j < 2; ++j)
                           m.insert(std::string("pre") + char('A' + c) + char('0' + j) + tail, uint64_t(c * 10 + j));
                  },
                  [&](M& m) { m.insert(std::string("pre") + 'Z' + tail, 999); });
         break;
      }
      case 3:  // buckets: page overflow -> bkt_split, then the re-put of the new key allocates
      {
         using M = map<std::string_view, uint64_t, mode::buckets, FA<uint64_t>>;
         sweep<M>("S3 buckets/u64 bucket burst", n,
                  [&](M& m) { for (int c = 0; c < 9; ++c) m.insert(std::string(1, char('A' + c)) + tail, uint64_t(c)); },
                  [&](M& m) { for (int c = 9; c < 12; ++c) m.insert(std::string(1, char('A' + c)) + tail, uint64_t(c)); });
         break;
      }
      case 4:  // default mode, inlineable value: split of a value_inline terminal has no build_guard
      {
         using M = map<std::string_view, uint32_t, mode::none, FA<uint32_t>>;
         sweep<M>("S4 none/u32 split inline terminal", n,
                  [](M& m) { m.insert("ab", 1); m.insert("ac", 2); },
                  [&](M& m) { m.insert(std::string("ab") + tail + tail + tail + tail, 3); });
         break;
      }
      case 5:  // adaptive: collapse_to_bucket frees the old leaf before the new value is constructed
      {
         using M = map<std::string_view, TV, mode::adaptive, FA<TV>>;
         sweep<M>("S5 adaptive/emplace throwing ctor", n,
                  [](M& m) { for (const char* k : {"a", "ab", "abc"}) m.emplace(k, "old-value-long-enough-to-be-on-the-heap"); },
                  [n](M& m) { g_vthrow = 0; (void)n; m.emplace("abcd", "new"); });
         break;
      }
      case 6:  // buckets + dense tiers: widen() succeeds (old node freed), then make_prefix allocates
      {
         using M = map<std::string_view, uint64_t, mode::buckets | mode::dense_tiers, FA<uint64_t>>;
         sweep<M>("S6 buckets|dense widen prefixed setlist", n,
                  [&](M& m) {
                     for (int c = 0; c < 16; ++c)
                        for (int j = 0; j < 2; ++j)
                           m.insert(std::string("pre") + char('A' + c) + char('0' + j) + tail, uint64_t(c * 10 + j));
                  },
                  [&](M& m) { m.insert(std::string("pre") + 'Z' + tail, 999); });
         break;
      }
      case 7:  // buckets: the 9th 41-byte suffix overflows the page -> bkt_split, then re-put allocates
      {
         using M = map<std::string_view, uint64_t, mode::buckets, FA<uint64_t>>;
         sweep<M>("S7 buckets/u64 burst on one insert", n,
                  [&](M& m) { for (int c = 0; c < 8; ++c) m.insert(std::string(1, char('A' + c)) + tail, uint64_t(c)); },
                  [&](M& m) { m.insert(std::string(1, 'Z') + tail, 99); });
         break;
      }
   }
   return 0;
}

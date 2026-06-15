// fuzz_set — differential fuzzer for artpp::set<Key> against std::set<std::string> as the
// oracle. Modeled on fuzz_map (same adversarial key generator); the value compares are dropped
// (a set has no value). Every op runs on BOTH containers and the results — return value, size,
// the full ordered contents forward AND backward, and operator== against a fresh clone — must
// match; any divergence aborts so libFuzzer / ASan capture the exact input.
//
// Stage-C focus: a TINY alphabet (4 letters) produces short suffixes, so the suffix-in-handle
// (value_inline) path dominates — split_leaf with inline suffixes, the inline-suffix iterator
// copy, and the descend inl comparison are all exercised hard. Stage-D adds the wider configs.
#include "artpp/map.hpp"
#include "artpp/pool.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <set>
#include <string>
#include <type_traits>
#include <vector>

using artpp::set;
using artpp::mode;

struct Reader
{
   const uint8_t* p;
   const uint8_t* end;
   uint8_t  u8() { return p < end ? *p++ : 0; }
   uint32_t u32() { uint32_t v = 0; for (int i = 0; i < 4; ++i) v = (v << 8) | u8(); return v; }
   bool     done() const { return p >= end; }
};

static const uint8_t* g_data = nullptr;
static size_t         g_size = 0;
static int            g_cfg  = -1;
[[noreturn]] static void fail(const char* what, int op)
{
   std::fprintf(stderr, "\nFUZZ DIVERGENCE: %s  (config=%d, op#%d)\n  input[%zu]= ", what, g_cfg, op,
                g_size);
   for (size_t i = 0; i < g_size; ++i) std::fprintf(stderr, "%02x", g_data[i]);
   std::fprintf(stderr, "\n");
   std::abort();
}
#define FZCHECK(cond, op) do { if (!(cond)) fail(#cond, (op)); } while (0)

// TINY alphabet → short keys → suffix-in-handle dominates. Includes 0x00 / 0xff boundary bytes.
static const unsigned char ALPHABET[] = {'a', 'b', 0x00, 0xff};
static std::string gen_key(Reader& r, std::vector<std::string>& pool)
{
   auto byte = [&] { return char(ALPHABET[r.u8() % sizeof ALPHABET]); };
   std::string   k;
   const uint8_t mode_byte = r.u8();
   if (!pool.empty() && (mode_byte & 1))
   {
      k                = pool[r.u32() % pool.size()];
      const uint8_t op = (mode_byte >> 1) & 3;
      if (op == 0 && !k.empty()) k.pop_back();
      else if (op == 1) k.push_back(byte());
      else if (op == 2 && !k.empty()) k[r.u8() % k.size()] = byte();
   }
   else
   {
      int n = (mode_byte >> 1) % 9;             // mostly short (<= cap+ a bit), occasionally longer
      if ((mode_byte & 0x70) == 0x70) n += int(r.u8() % 40);
      for (int j = 0; j < n; ++j) k.push_back(byte());
   }
   if (pool.size() < 256) pool.push_back(k);
   else pool[r.u32() % pool.size()] = k;
   return k;
}

template <class Set>
static void cross_check(const Set& s, const std::set<std::string>& o, int op)
{
   FZCHECK(s.size() == o.size(), op);
   {  // forward
      auto oi = o.begin();
      for (auto it = s.begin(); it != s.end(); ++it, ++oi)
         FZCHECK(oi != o.end() && std::string(it.key()) == *oi, op);
      FZCHECK(oi == o.end(), op);
   }
   {  // backward — operator-- and --end()
      auto it = s.end();
      auto oi = o.end();
      while (oi != o.begin())
      {
         FZCHECK(it != s.begin(), op);
         --it;
         --oi;
         FZCHECK(std::string(it.key()) == *oi, op);
      }
      FZCHECK(it == s.begin(), op);
   }
   {  // for_each — separate walk, same order
      std::vector<std::string> fe;
      s.for_each([&](std::string_view k) { fe.emplace_back(k); });
      FZCHECK(fe.size() == o.size(), op);
      std::size_t idx = 0;
      for (auto& k : o) { FZCHECK(idx < fe.size() && fe[idx] == k, op); ++idx; }
   }
}

template <class Set>
static void run_impl(Set& s, Reader& r)
{
   std::set<std::string>    o;
   std::vector<std::string> pool;
   int                      op = 0;
   for (; !r.done(); ++op)
   {
      const std::string k      = gen_key(r, pool);
      const uint8_t     choice = r.u8() % 8;
      switch (choice)
      {
         case 0: FZCHECK(s.insert(k).second == o.insert(k).second, op); break;
         case 1: FZCHECK(s.emplace(k).second == o.emplace(k).second, op); break;
         case 2: FZCHECK(s.erase(k) == o.erase(k), op); break;
         case 3:  // erase through an iterator at lower_bound(k)
         {
            auto it = s.lower_bound(k);
            auto mi = o.lower_bound(k);
            FZCHECK((it == s.end()) == (mi == o.end()), op);
            if (mi != o.end())
            {
               it = s.erase(it);
               mi = o.erase(mi);
               FZCHECK((it == s.end()) == (mi == o.end()), op);
               if (mi != o.end()) FZCHECK(std::string(it.key()) == *mi, op);
            }
            break;
         }
         case 4:  // membership three ways
         {
            const bool have = s.contains(k);
            const bool om   = o.count(k) != 0;
            FZCHECK(have == om, op);
            FZCHECK(s.count(k) == (om ? 1u : 0u), op);
            FZCHECK((s.find(k) != s.end()) == om, op);
            break;
         }
         case 5:  // ordered queries
         {
            auto lo = s.lower_bound(k);
            auto ml = o.lower_bound(k);
            FZCHECK((lo == s.end()) == (ml == o.end()), op);
            if (ml != o.end()) FZCHECK(std::string(lo.key()) == *ml, op);
            auto hi = s.upper_bound(k);
            auto mu = o.upper_bound(k);
            FZCHECK((hi == s.end()) == (mu == o.end()), op);
            if (mu != o.end()) FZCHECK(std::string(hi.key()) == *mu, op);
            auto er = s.equal_range(k);
            FZCHECK(er.first == lo && er.second == hi, op);
            break;
         }
         default: FZCHECK(s.insert(k).second == o.insert(k).second, op); break;  // 6,7: more inserts
      }
      if ((op & 0x3f) == 0x3f)
      {
         cross_check(s, o, op);
         if constexpr (std::is_same_v<typename Set::allocator_type, std::allocator<std::string_view>>)
         {
            Set clone;
            for (auto& ck : o) clone.insert(ck);
            FZCHECK(clone == s && s == clone, op);
         }
      }
      if (r.u8() == 0) { s.clear(); o.clear(); }
   }
   cross_check(s, o, op);
}

static void run_dispatch(const uint8_t* data, size_t size)
{
   g_data = data; g_size = size;
   Reader    r{data, data + size};
   const int cfg = (g_cfg = r.u8() % 6);
   using sv  = std::string_view;
   switch (cfg)
   {
      case 0: { set<sv> s;                                      run_impl(s, r); break; }
      case 1: { set<sv, mode::dense_tiers> s;                   run_impl(s, r); break; }
      case 2: { set<sv, mode::dense_tiers | mode::ladder_c8> s; run_impl(s, r); break; }
      case 3: { set<sv, mode::wide_stems> s;                    run_impl(s, r); break; }
      case 4: { set<sv, mode::buckets> s;                       run_impl(s, r); break; }
      case 5: { set<sv, mode::adaptive> s;                      run_impl(s, r); break; }
   }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size)
{
   run_dispatch(data, size);
   return 0;
}

#ifndef ARTPP_LIBFUZZER
static std::vector<uint8_t> slurp(const char* path)
{
   std::vector<uint8_t> v;
   if (FILE* f = std::fopen(path, "rb"))
   {
      uint8_t buf[4096];
      size_t  n;
      while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) v.insert(v.end(), buf, buf + n);
      std::fclose(f);
   }
   return v;
}
int main(int argc, char** argv)
{
   if (argc > 1 && std::fopen(argv[1], "rb"))
   {
      for (int i = 1; i < argc; ++i) { auto b = slurp(argv[i]); run_dispatch(b.data(), b.size()); }
      std::printf("artpp_fuzz_set: replayed %d input(s) OK\n", argc - 1);
      return 0;
   }
   const std::uint64_t  seed  = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 0x5E7FABULL;
   const std::size_t    iters = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 20000;
   std::mt19937_64      rng(seed);
   std::vector<uint8_t> buf;
   for (std::size_t i = 0; i < iters; ++i)
   {
      std::size_t len = 16 + rng() % 512;
      if ((rng() & 0x1f) == 0) len += rng() % 4096;
      buf.resize(len);
      for (auto& b : buf) b = uint8_t(rng());
      run_dispatch(buf.data(), buf.size());
   }
   std::printf("artpp_fuzz_set: ALL PASS (%zu inputs, seed %llu)\n", iters,
               (unsigned long long)seed);
   return 0;
}
#endif

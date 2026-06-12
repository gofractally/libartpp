// artpp_wide — correctness + demonstration for mode::wide_stems (the setlist_u16 wide router).
//
// Correctness: for several workloads, cross-check a wide-stems map against std::map as an
// oracle — every key found with the right value, iteration complete and in lexicographic
// order, update + remove behave identically. Run the same checks in DEFAULT mode so a bug in
// the shared paths can't hide. Demonstration: print router-hops/lookup and query ns for wide
// vs default on the sparse workloads, showing the depth (cacheline-touch) collapse.
#include "artpp/map.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <map>
#include <random>
#include <string>
#include <vector>

using artpp::map;
using artpp::mode;

static int g_fail = 0;
#define CHECK(c, msg)                                                              \
   do {                                                                            \
      if (!(c)) { std::printf("  FAIL: %s\n", msg); ++g_fail; }                    \
   } while (0)

static double now_s()
{
   using namespace std::chrono;
   return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// Build the cross-check workloads.
static std::vector<std::string> gen(const char* which, size_t N, std::mt19937_64& rng)
{
   std::vector<std::string> v(N);
   if (std::string_view(which) == "paths")
      for (auto& s : v)
      { s.clear(); for (int lvl = 0; lvl < 4; ++lvl) { s += "/"; s += std::to_string(rng() % 32); } }
   else if (std::string_view(which) == "clustered")
      for (auto& s : v)
      {
         s = "family/" + std::to_string(rng() % 4096) + "/";
         int n = 4 + int(rng() % 8);
         for (int j = 0; j < n; ++j) s.push_back(char('a' + rng() % 16));
      }
   else  // uuid16
      for (auto& s : v) { s.resize(16); for (auto& c : s) c = char(rng()); }
   return v;
}

template <mode M>
static void xcheck(const char* tag, const std::vector<std::string>& keys)
{
   using Tree = map<std::string_view, uint64_t, M>;
   Tree                            t;
   std::map<std::string, uint64_t> oracle;
   for (size_t i = 0; i < keys.size(); ++i)
   {
      bool a = t.insert(keys[i], uint64_t(i + 1));
      bool b = oracle.emplace(keys[i], uint64_t(i + 1)).second;
      CHECK(a == b, "insert newness mismatch vs oracle");
      if (!b) t.update(keys[i], uint64_t(i + 1)), oracle[keys[i]] = uint64_t(i + 1);  // dup key: keep in sync
   }
   CHECK(t.size() == oracle.size(), "size mismatch");

   // every key present with the right value
   for (auto& [k, val] : oracle)
   {
      uint64_t got;
      CHECK(t.find(k, got) && got == val, "find/value mismatch");
   }
   // iteration: complete + lexicographically ordered + matches oracle exactly
   {
      auto   oit = oracle.begin();
      size_t n   = 0;
      std::string prev;
      for (auto it = t.begin(); it != t.end(); ++it, ++oit, ++n)
      {
         CHECK(oit != oracle.end(), "iterator overran oracle");
         if (oit == oracle.end()) break;
         std::string k(it.key());
         CHECK(k == oit->first, "iteration key mismatch");
         CHECK(it.value() == oit->second, "iteration value mismatch");
         CHECK(n == 0 || prev < k, "iteration not strictly ascending");
         prev = k;
      }
      CHECK(n == oracle.size(), "iteration count mismatch");
   }
   // lower_bound / upper_bound vs std::map — probe each key plus boundary variants (prefix,
   // just-after, incremented last byte, before-first, after-last). This is the check that was
   // MISSING: it traverses the wide u16 routers on the lower-bound path.
   {
      auto chk = [&](const std::string& probe, bool upper) {
         auto oit = upper ? oracle.upper_bound(probe) : oracle.lower_bound(probe);
         auto hit = upper ? t.upper_bound(std::string_view(probe)) : t.lower_bound(std::string_view(probe));
         if (oit == oracle.end()) { CHECK(hit == t.end(), "lb/ub: expected end()"); return; }
         CHECK(hit != t.end(), "lb/ub: expected a key, got end()");
         if (hit == t.end()) return;
         CHECK(std::string(hit.key()) == oit->first, "lb/ub key mismatch");
         CHECK(hit.value() == oit->second, "lb/ub value mismatch");
      };
      for (auto& [k, v] : oracle)
      {
         (void)v;
         chk(k, false); chk(k, true);                 // exact key
         chk(k + '\x00', false); chk(k + '\x00', true);  // just after k
         if (!k.empty()) { std::string p = k; p.pop_back(); chk(p, false); chk(p, true); }  // a prefix
         { std::string b = k; b.back()++; chk(b, false); chk(b, true); }  // incremented last byte
      }
      chk("", false); chk("", true);                   // before everything
      chk("\xff\xff\xff", false); chk("\xff\xff\xff", true);  // after everything
   }
   // remove half (every other key in oracle order), then re-verify against the oracle
   {
      std::vector<std::string> dead;
      size_t                   i = 0;
      for (auto& [k, val] : oracle) { (void)val; if (i++ & 1) dead.push_back(k); }
      for (auto& k : dead) { CHECK(t.remove(k), "remove of present key failed"); oracle.erase(k); }
      CHECK(t.size() == oracle.size(), "size mismatch after remove");
      for (auto& [k, val] : oracle)
      { uint64_t got; CHECK(t.find(k, got) && got == val, "survivor find mismatch"); }
      for (auto& k : dead) { uint64_t got; CHECK(!t.find(k, got), "removed key still found"); }
   }
   std::printf("  [%s] %s: ok (%zu keys)\n", tag, "xcheck", oracle.size());
}

template <mode M>
static void measure(const char* tag, const std::vector<std::string>& keys,
                    const std::vector<uint32_t>& order)
{
   using Tree = map<std::string_view, uint64_t, M>;
   Tree t;
   for (size_t i = 0; i < keys.size(); ++i) t.insert(keys[i], uint64_t(i + 1));
   auto         d   = t.debug_stats();
   const double rh  = double(d.router_hops) / double(d.terminals ? d.terminals : 1);
   double       best = 1e30;
   for (int r = 0; r < 4; ++r)
   {
      uint64_t sink = 0;
      double   t0   = now_s();
      for (uint32_t i : order) { uint64_t v; if (t.find(keys[i], v)) sink ^= v; }
      best = std::min(best, (now_s() - t0) * 1e9 / double(order.size()));
      asm volatile("" : : "g"(sink) : "memory");
   }
   std::printf("  [%-7s] rtr-hops/lkup=%.2f  query=%.1f ns  (setlist=%llu setlist16=%llu full=%llu)\n",
               tag, rh, best, (unsigned long long)d.setlist, (unsigned long long)d.setlist16,
               (unsigned long long)d.full);
}

int main(int argc, char** argv)
{
   const size_t N = (argc > 1) ? std::strtoull(argv[1], nullptr, 10) : 1'000'000ULL;
   std::mt19937_64 rng(7);

   std::printf("== correctness (wide vs std::map oracle; default mode too) ==\n");
   for (const char* w : {"paths", "clustered", "uuid16"})
   {
      auto keys = gen(w, N / 4, rng);
      std::printf(" %s:\n", w);
      xcheck<mode::wide_stems>("wide", keys);
      xcheck<mode::none>("dflt", keys);
   }

   std::printf("\n== demonstration: wide vs default depth + query ==\n");
   for (const char* w : {"paths", "clustered", "uuid16"})
   {
      auto                  keys = gen(w, N, rng);
      std::vector<uint32_t> order(N / 2);
      for (auto& x : order) x = uint32_t(rng() % N);
      std::printf(" %s:\n", w);
      measure<mode::none>("default", keys, order);
      measure<mode::wide_stems>("wide", keys, order);
   }

   std::printf(g_fail ? "\nartpp_wide: %d FAIL\n" : "\nartpp_wide: ALL PASS\n", g_fail);
   return g_fail ? 1 : 0;
}

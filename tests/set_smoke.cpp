// Correctness smoke test for artpp::set<Key> — the value-free facade over map<Key, none_t>.
// Differential against std::set<std::string>: insert / contains / count / erase / ordered
// iterate, including long suffixes (exercise the leaf path), the empty key, and 0x00 / 0xFF
// boundary bytes. The set shares the radix core verbatim, so this also guards that the
// value-elision gating didn't break the key set behavior.
#include "artpp/map.hpp"

#include <cstdint>
#include <cstdio>
#include <random>
#include <set>
#include <string>
#include <string_view>
#include <vector>

using artpp::set;
using artpp::mode;

static int g_fail = 0;
#define CHECK(cond)                                                                          \
   do {                                                                                      \
      if (!(cond)) { std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); ++g_fail; } \
   } while (0)

static void test_basic()
{
   set<std::string_view> s;
   CHECK(s.insert("hello").second);
   CHECK(s.insert("help").second);
   CHECK(s.insert("hell").second);
   CHECK(s.insert("world").second);
   CHECK(!s.insert("hello").second);  // duplicate
   CHECK(s.size() == 4);

   CHECK(s.contains("hello"));
   CHECK(s.contains("help"));
   CHECK(s.contains("hell"));
   CHECK(s.contains("world"));
   CHECK(!s.contains("hel"));
   CHECK(!s.contains("helloo"));
   CHECK(!s.contains(""));
   CHECK(s.count("hello") == 1);
   CHECK(s.count("nope") == 0);

   CHECK(s.insert("").second);  // empty key
   CHECK(s.contains(""));
   CHECK(s.size() == 5);

   CHECK(s.erase("hell") == 1);
   CHECK(!s.contains("hell"));
   CHECK(s.erase("hell") == 0);  // already gone
   CHECK(s.size() == 4);
}

static void test_ordered_iterate()
{
   set<std::string_view> s;
   std::set<std::string>  oracle;
   const char*            keys[] = {"banana", "apple", "cherry", "app", "applepie", "", "z"};
   for (auto* k : keys)
   {
      s.insert(k);
      oracle.insert(k);
   }
   // forward
   auto oi = oracle.begin();
   for (auto it = s.begin(); it != s.end(); ++it, ++oi)
   {
      CHECK(oi != oracle.end());
      CHECK(std::string(it.key()) == *oi);
      CHECK(std::string(*it) == *oi);  // operator* yields the key
   }
   CHECK(oi == oracle.end());
   // backward (-- / --end())
   auto it = s.end();
   auto ri = oracle.end();
   while (ri != oracle.begin())
   {
      CHECK(it != s.begin());
      --it;
      --ri;
      CHECK(std::string(it.key()) == *ri);
   }
   CHECK(it == s.begin());
   // for_each
   std::vector<std::string> fe;
   s.for_each([&](std::string_view k) { fe.emplace_back(k); });
   CHECK(fe.size() == oracle.size());
   std::size_t idx = 0;
   for (auto& k : oracle) { CHECK(idx < fe.size() && fe[idx] == k); ++idx; }
}

static void test_bounds()
{
   set<std::string_view> s;
   for (const char* k : {"a", "c", "e", "g"}) s.insert(k);
   CHECK(std::string(*s.lower_bound("c")) == "c");
   CHECK(std::string(*s.lower_bound("b")) == "c");
   CHECK(std::string(*s.upper_bound("c")) == "e");
   CHECK(s.upper_bound("g") == s.end());
   CHECK(s.lower_bound("h") == s.end());
   auto er = s.equal_range("e");
   CHECK(std::string(*er.first) == "e");
   CHECK(std::string(*er.second) == "g");
   CHECK(s.find("c") != s.end());
   CHECK(s.find("d") == s.end());
}

// Long suffixes (>16 bytes) force external leaves rather than any short-key inline path.
static void test_long_suffixes()
{
   set<std::string_view> s;
   std::set<std::string>  oracle;
   const char*            longs[] = {
       "this_is_a_very_long_key_number_one_aaaaaaaaaaaaaaaaaaaa",
       "this_is_a_very_long_key_number_two_bbbbbbbbbbbbbbbbbbbb",
       "this_is_a_very_long_key_number_one_aaaaaaaaaaaaaaaaaaab",
       "short",
   };
   for (auto* k : longs)
   {
      s.insert(k);
      oracle.insert(k);
   }
   CHECK(s.size() == oracle.size());
   for (auto* k : longs) CHECK(s.contains(k));
   auto oi = oracle.begin();
   for (auto it = s.begin(); it != s.end(); ++it, ++oi) CHECK(std::string(it.key()) == *oi);
}

// boundary bytes (0x00, 0xFF) and binary keys
static void test_binary_keys()
{
   set<std::string_view> s;
   std::set<std::string>  oracle;
   std::string            k0("\x00\x00", 2), k1("\xff\xff", 2), k2("\x00\xff", 2), k3("a\x00""b", 3);
   for (auto& k : {k0, k1, k2, k3})
   {
      s.insert(k);
      oracle.insert(k);
   }
   CHECK(s.size() == oracle.size());
   auto oi = oracle.begin();
   for (auto it = s.begin(); it != s.end(); ++it, ++oi) CHECK(std::string(it.key()) == *oi);
}

// randomized differential vs std::set across a few configs
template <class Set>
static void test_random(unsigned seed)
{
   Set                    s;
   std::set<std::string>  oracle;
   std::mt19937           rng(seed);
   const char             alpha[] = {'a', 'b', 'c', 'd', 0x00, char(0xff)};
   auto                   genkey  = [&] {
      std::string k;
      int         n = int(rng() % 10);
      for (int i = 0; i < n; ++i) k.push_back(alpha[rng() % sizeof(alpha)]);
      return k;
   };
   for (int i = 0; i < 4000; ++i)
   {
      std::string k = genkey();
      if (rng() & 1)
         CHECK(s.insert(k).second == oracle.insert(k).second);
      else
         CHECK(s.erase(k) == oracle.erase(k));
   }
   CHECK(s.size() == oracle.size());
   auto oi = oracle.begin();
   for (auto it = s.begin(); it != s.end(); ++it, ++oi)
   {
      CHECK(oi != oracle.end());
      CHECK(std::string(it.key()) == *oi);
   }
   CHECK(oi == oracle.end());
   // contains agreement
   for (int i = 0; i < 1000; ++i)
   {
      std::string k = genkey();
      CHECK(s.contains(k) == (oracle.count(k) != 0));
   }
}

int main()
{
   test_basic();
   test_ordered_iterate();
   test_bounds();
   test_long_suffixes();
   test_binary_keys();
   test_random<set<std::string_view>>(1);
   test_random<set<std::string_view, mode::dense_tiers>>(2);
   test_random<set<std::string_view, mode::buckets>>(3);
   test_random<set<std::string_view, mode::wide_stems>>(4);
   if (g_fail == 0) std::printf("set_smoke: ALL PASS\n");
   return g_fail ? 1 : 0;
}

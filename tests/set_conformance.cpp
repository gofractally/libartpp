// set_conformance — STL-grade contract suite for artpp::set<Key> (the value-free facade over
// map<Key, none_t>). Probes the contracts a standard-library review would: iterator edge cases
// (empty / single element / default-constructed), --end() / rbegin, the max_key_bytes boundary
// + length_error, move construction/assignment, classic-algorithm compatibility over the proxy
// iterator, the suffix-in-handle boundary (keys at / just over the inline cap), and a file-backed
// line_pool round-trip (checkpoint + O(1) attach).
#include "artpp/map.hpp"
#include "artpp/pool.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unistd.h>
#include <vector>

using artpp::compact_set;
using artpp::mode;
using artpp::set;

static int g_fail = 0;
#define CHECK(cond)                                                                          \
   do {                                                                                      \
      if (!(cond)) { std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); ++g_fail; } \
   } while (0)

// ── iterator edge cases ─────────────────────────────────────────────────────────
static void test_empty_iterators()
{
   set<std::string_view> s;
   CHECK(s.begin() == s.end());
   CHECK(s.cbegin() == s.cend());
   CHECK(s.rbegin() == s.rend());
   CHECK(s.empty() && s.size() == 0);
   CHECK(s.find("x") == s.end());
   CHECK(s.lower_bound("x") == s.end());
   CHECK(s.upper_bound("x") == s.end());
   auto er = s.equal_range("x");
   CHECK(er.first == s.end() && er.second == s.end());
   // default-constructed iterators compare equal
   set<std::string_view>::const_iterator a, b;
   CHECK(a == b);
}

static void test_single_element()
{
   set<std::string_view> s;
   s.insert("solo");
   CHECK(s.size() == 1);
   auto it = s.begin();
   CHECK(it != s.end());
   CHECK(std::string(*it) == "solo");
   ++it;
   CHECK(it == s.end());
   // --end() seats at the (only) last element
   auto e = s.end();
   --e;
   CHECK(std::string(*e) == "solo");
   CHECK(e == s.begin());
   // rbegin: the set's own reverse iterator holds the iterator positioned on the element, so
   // *r is valid until r moves; base() keeps the std meaning (one past the element)
   auto r = s.rbegin();
   CHECK(r != s.rend());
   CHECK(std::string(*r) == "solo");
   { auto fwd = r.base(); CHECK(fwd == s.end()); --fwd; CHECK(std::string(fwd.key()) == "solo"); }
   ++r;
   CHECK(r == s.rend());
}

static void test_end_decrement()
{
   set<std::string_view> s;
   std::set<std::string>  o;
   for (const char* k : {"alpha", "beta", "gamma", "delta", "epsilon"})
   {
      s.insert(k);
      o.insert(k);
   }
   // --end() then walk backward to begin() must equal the oracle in reverse
   auto it = s.end();
   auto oi = o.end();
   while (oi != o.begin())
   {
      CHECK(it != s.begin());
      --it;
      --oi;
      CHECK(std::string(it.key()) == *oi);
   }
   CHECK(it == s.begin());
   // reverse iterator agreement, dereferenced directly and through base()
   std::vector<std::string> rev;
   for (auto r = s.rbegin(); r != s.rend(); ++r)
   {
      auto fwd = r.base(); --fwd;
      CHECK(std::string(*r) == std::string(fwd.key()));
      rev.emplace_back(*r);
   }
   std::vector<std::string> orev(o.rbegin(), o.rend());
   CHECK(rev == orev);
}

// ── max_key_bytes boundary ──────────────────────────────────────────────────────
static void test_max_key_bytes()
{
   set<std::string_view> s;
   CHECK(set<std::string_view>::max_key_bytes == 65535);
   std::string at_limit(65535, 'q');
   CHECK(s.insert(at_limit).second);  // exactly at the limit: OK
   CHECK(s.contains(at_limit));
   std::string over(65536, 'q');
   bool threw = false;
   try { s.insert(over); } catch (const std::length_error&) { threw = true; }
   CHECK(threw);
   // an overlong LOOKUP simply misses (no stored key can be that long)
   CHECK(!s.contains(over));
}

// ── move construction / assignment ──────────────────────────────────────────────
static void test_move()
{
   set<std::string_view> a;
   std::set<std::string>  o;
   for (const char* k : {"one", "two", "three", "four", "five"}) { a.insert(k); o.insert(k); }
   set<std::string_view> b(std::move(a));  // move ctor
   CHECK(b.size() == o.size());
   CHECK(a.size() == 0);  // moved-from is valid-empty
   {
      auto oi = o.begin();
      for (auto it = b.begin(); it != b.end(); ++it, ++oi) CHECK(std::string(it.key()) == *oi);
   }
   // moved-from is reusable
   a.insert("reborn");
   CHECK(a.size() == 1 && a.contains("reborn"));
   // move assignment
   set<std::string_view> c;
   c.insert("temp");
   c = std::move(b);
   CHECK(c.size() == o.size());
   {
      auto oi = o.begin();
      for (auto it = c.begin(); it != c.end(); ++it, ++oi) CHECK(std::string(it.key()) == *oi);
   }
}

// ── classic algorithms over the proxy iterator ──────────────────────────────────
static void test_algorithms()
{
   set<std::string_view> s;
   std::set<std::string>  o;
   for (const char* k : {"m", "a", "z", "f", "q", "b"}) { s.insert(k); o.insert(k); }
   // std::distance, std::find_if, std::count_if over the bidirectional proxy iterator
   CHECK(std::distance(s.begin(), s.end()) == (std::ptrdiff_t)o.size());
   auto found = std::find_if(s.begin(), s.end(), [](std::string_view k) { return k == "q"; });
   CHECK(found != s.end() && std::string(*found) == "q");
   const auto n = std::count_if(s.begin(), s.end(), [](std::string_view k) { return k < "m"; });
   CHECK(n == 3);  // a, b, f
   // is_sorted: keys come out ascending
   CHECK(std::is_sorted(s.begin(), s.end(),
                        [](std::string_view x, std::string_view y) { return x < y; }));
}

// ── suffix-in-handle boundary: keys at / just over the inline cap ────────────────
static void test_inline_boundary()
{
   set<std::string_view> s;
   std::set<std::string>  o;
   // keys of length 0..8 over a single shared prefix → suffixes straddle the 5-byte cap
   for (int n = 0; n <= 8; ++n)
   {
      std::string k(size_t(n), 'a');
      s.insert(k);
      o.insert(k);
      std::string k2 = "px" + std::string(size_t(n), 'b');  // shared prefix, varied suffix
      s.insert(k2);
      o.insert(k2);
   }
   CHECK(s.size() == o.size());
   for (auto& k : o) CHECK(s.contains(k));
   auto oi = o.begin();
   for (auto it = s.begin(); it != s.end(); ++it, ++oi) CHECK(std::string(it.key()) == *oi);
   // erase every other and re-verify ordering (exercises inline-suffix shrink)
   int i = 0;
   for (auto k : std::vector<std::string>(o.begin(), o.end()))
      if (i++ & 1) { s.erase(k); o.erase(k); }
   CHECK(s.size() == o.size());
   oi = o.begin();
   for (auto it = s.begin(); it != s.end(); ++it, ++oi) CHECK(std::string(it.key()) == *oi);
}

// ── compact_set + dense ladder + operator== ─────────────────────────────────────
static void test_compact_and_equality()
{
   compact_set<std::string_view, 1000> s;
   set<std::string_view, mode::dense_tiers> a, b;
   std::set<std::string>                    o;
   for (int i = 0; i < 500; ++i)
   {
      std::string k = "key" + std::to_string(i * 7 % 1000);
      s.insert(k);
      a.insert(k);
      o.insert(k);
   }
   s.compact_set();  // no-op accessor, must compile + leave content intact
   CHECK(s.size() == o.size());
   for (auto& k : o) CHECK(s.contains(k) && a.contains(k));
   // operator== over a fresh clone (same config ⇒ same shape)
   for (auto& k : o) b.insert(k);
   CHECK(a == b && b == a);
   b.insert("extra_unique_key");
   CHECK(a != b);
}

// ── file-backed line_pool round-trip (checkpoint + O(1) attach) ──────────────────
using PA  = artpp::pool_alloc<std::string_view>;
using PSet = artpp::set<std::string_view, mode::none, PA>;
static void wipe(const char* dir)
{
   ::unlink((std::string(dir) + "/nodes").c_str());
   ::unlink((std::string(dir) + "/terms").c_str());
   ::rmdir(dir);
}
static void test_line_pool_attach()
{
   char path[] = "/tmp/artpp_set_persist_XXXXXX";  // unique to this run
   CHECK(mkdtemp(path) != nullptr);
   const int               N = 20000;
   std::vector<std::string> keys;
   std::set<std::string>    ref;
   for (int i = 0; i < N; ++i)
   {
      std::string k(3 + (i % 20), char('a' + (i % 26)));  // mix of short (inline) and long keys
      for (size_t j = 0; j < k.size(); ++j) k[j] = char('a' + (i * 31 + int(j)) % 26);
      keys.push_back(k);
      ref.insert(k);
   }
   {  // create, populate, checkpoint, detach
      artpp::line_pool pool(path);
      CHECK(!pool.restored());
      PSet s{PA{&pool}};
      for (auto& k : keys) s.insert(k);
      CHECK(s.size() == ref.size());
      pool.checkpoint(s.root_handle(), s.size());
      s.detach();
   }
   {  // reopen, attach in O(1), verify
      artpp::line_pool pool(path);
      CHECK(pool.restored() && pool.count() == ref.size());
      PSet s{PA{&pool}, artpp::attach, pool.root(), pool.count()};
      CHECK(s.size() == ref.size());
      for (auto& k : ref) CHECK(s.contains(k));
      auto ri = ref.begin();
      for (auto it = s.begin(); it != s.end(); ++it, ++ri) CHECK(std::string(it.key()) == *ri);
      CHECK(ri == ref.end());
      s.detach();
   }
   wipe(path);
}

int main()
{
   test_empty_iterators();
   test_single_element();
   test_end_decrement();
   test_max_key_bytes();
   test_move();
   test_algorithms();
   test_inline_boundary();
   test_compact_and_equality();
   test_line_pool_attach();
   if (g_fail == 0) std::printf("set_conformance: ALL PASS\n");
   return g_fail ? 1 : 0;
}

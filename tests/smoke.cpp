// Correctness tests for artpp::map<T> (copy-out find API):
//   * basic insert / find / update / miss
//   * lazy-expansion leaf splits and path-compression (prefix_node) splits
//   * 256-way fan-out (forces setlist -> node_full growth); int value = INLINE path
//   * long shared prefixes (multi-cacheline prefix_node)
//   * non-POD value (std::string) and 8-byte value (uint64) -> LEAF path
//   * dense c2/c4/c8 tiers; remove; randomized cross-check vs std::map
#include "artpp/pool.hpp"
#include "artpp/map.hpp"

#include <sys/stat.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <map>
#include <memory_resource>
#include <new>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using artpp::map;
using artpp::mode;

static int g_fail = 0;
#define CHECK(cond)                                                                          \
   do {                                                                                      \
      if (!(cond)) { std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); ++g_fail; } \
   } while (0)

// copy-out helpers
template <class Tree, class V>
static bool has(const Tree& t, std::string_view k, const V& exp)
{
   typename Tree::mapped_type v{};
   return t.find(k, v) && v == exp;
}
template <class Tree>
static bool miss(const Tree& t, std::string_view k)
{
   return !t.contains(k);
}

static void test_basic()
{
   map<std::string_view, uint64_t> t;  // 8-byte value -> leaf path
   CHECK(t.insert("hello", 1));
   CHECK(t.insert("help", 2));
   CHECK(t.insert("hell", 3));
   CHECK(t.insert("world", 4));
   CHECK(!t.insert("hello", 11));  // update, not new
   CHECK(t.size() == 4);

   CHECK(has(t, "hello", uint64_t(11)));
   CHECK(has(t, "help", uint64_t(2)));
   CHECK(has(t, "hell", uint64_t(3)));
   CHECK(has(t, "world", uint64_t(4)));
   CHECK(miss(t, "hel"));     // prefix of others, not inserted
   CHECK(miss(t, "helloo"));
   CHECK(miss(t, ""));

   CHECK(t.insert("", 99));    // empty key
   CHECK(has(t, "", uint64_t(99)));
}

static void test_fanout256()
{
   map<std::string_view, int> t;  // 4-byte value -> INLINE path (no leaf for single-byte keys)
   for (int b = 0; b < 256; ++b) { char c = char(b); CHECK(t.insert(std::string_view(&c, 1), b)); }
   CHECK(t.size() == 256);
   for (int b = 0; b < 256; ++b) { char c = char(b); CHECK(has(t, std::string_view(&c, 1), b)); }
   for (int b = 0; b < 256; ++b) { char c = char(b); CHECK(!t.insert(std::string_view(&c, 1), b + 1000)); }
   for (int b = 0; b < 256; ++b) { char c = char(b); CHECK(has(t, std::string_view(&c, 1), b + 1000)); }
}

static void test_long_prefix()
{
   map<std::string_view, int> t;
   std::string  base(500, 'a');  // forces a multi-cacheline prefix_node
   std::string  k1 = base + "X", k2 = base + "Y", k3 = base.substr(0, 250) + "Z";
   CHECK(t.insert(k1, 1));
   CHECK(t.insert(k2, 2));
   CHECK(t.insert(k3, 3));
   CHECK(has(t, k1, 1));
   CHECK(has(t, k2, 2));
   CHECK(has(t, k3, 3));
   CHECK(miss(t, base));
}

static void test_nonpod()
{
   map<std::string_view, std::string> t;  // non-POD -> leaf path, copy/move ctor + dtor
   CHECK(t.insert("alpha", std::string("one")));
   CHECK(t.insert("alpine", std::string("two")));
   CHECK(t.insert("beta", std::string(200, 'z')));
   CHECK(has(t, "alpha", std::string("one")));
   CHECK(has(t, "alpine", std::string("two")));
   std::string v;
   CHECK(t.find("beta", v) && v.size() == 200);
   CHECK(!t.insert("alpha", std::string("updated")));
   CHECK(has(t, "alpha", std::string("updated")));
}

static void test_dense_tiers()
{
   map<std::string_view, int, mode::dense_tiers> t;  // DenseTiers
   for (int b = 0; b < 256; ++b) { char c = char(b); CHECK(t.insert(std::string_view(&c, 1), b)); }
   CHECK(t.size() == 256);
   for (int b = 0; b < 256; ++b) { char c = char(b); CHECK(has(t, std::string_view(&c, 1), b)); }

   map<std::string_view, uint64_t, mode::dense_tiers> t2;  // DenseTiers
   std::map<std::string, uint64_t> ref;
   std::mt19937_64 rng(7);
   for (int i = 0; i < 40000; ++i)
   {
      std::string k;
      int         n = 1 + int(rng() % 6);
      for (int j = 0; j < n; ++j) k.push_back(char(rng() % 70));
      uint64_t val = rng();
      bool     newk = ref.find(k) == ref.end();
      ref[k]        = val;
      CHECK(t2.insert(k, val) == newk);
   }
   CHECK(t2.size() == ref.size());
   for (auto& [k, v] : ref) CHECK(has(t2, k, v));
}

template <class Tree>
static void remove_fanout_body()  // 256-way fanout remove, run per config (radix / dense)
{
   Tree t;
   for (int b = 0; b < 256; ++b) { char c = char(b); t.insert(std::string_view(&c, 1), b); }
   t.insert("", -1);  // term at root
   CHECK(t.size() == 257);
   for (int b = 0; b < 256; b += 2) { char c = char(b); CHECK(t.remove(std::string_view(&c, 1))); }
   CHECK(t.size() == 257 - 128);
   for (int b = 0; b < 256; ++b)
   {
      char c = char(b);
      if (b % 2 == 0) CHECK(miss(t, std::string_view(&c, 1)));
      else            CHECK(has(t, std::string_view(&c, 1), b));
   }
   char zero = 0;
   CHECK(!t.remove(std::string_view(&zero, 1)));  // byte-0 key already gone (idempotent)
   CHECK(has(t, "", -1));                          // term survives
   CHECK(t.remove(""));                            // remove term
   CHECK(miss(t, ""));
}

static void test_remove()
{
   remove_fanout_body<map<std::string_view, int>>();
   remove_fanout_body<map<std::string_view, int, mode::dense_tiers>>();  // DenseTiers

   map<std::string_view, uint64_t>               t;
   std::map<std::string, uint64_t> ref;
   std::mt19937_64                 rng(42);
   for (int i = 0; i < 30000; ++i)
   {
      std::string k;
      int         n = int(rng() % 10);
      for (int j = 0; j < n; ++j) k.push_back(char(rng() % 256));
      ref[k] = rng();
      t.insert(k, ref[k]);
   }
   for (auto it = ref.begin(); it != ref.end();)
   {
      if (rng() & 1) { CHECK(t.remove(it->first)); it = ref.erase(it); }
      else ++it;
   }
   CHECK(t.size() == ref.size());
   for (auto& [k, v] : ref) CHECK(has(t, k, v));
}

static void test_buckets()
{
   // bucket mode with ADVERSARIAL keys: arbitrary bytes incl embedded NUL, varied
   // lengths (the case that breaks naive wide stems). Cross-checked vs std::map.
   map<std::string_view, uint32_t, mode::buckets> t;  // Buckets
   std::map<std::string, uint32_t> ref;
   std::mt19937_64                 rng(2024);
   for (int i = 0; i < 60000; ++i)
   {
      std::string k;
      int         n = int(rng() % 12);
      for (int j = 0; j < n; ++j) k.push_back(char(rng() % 256));  // embedded NUL allowed
      uint32_t v    = uint32_t(rng());
      bool     newk = ref.find(k) == ref.end();
      ref[k]        = v;
      CHECK(t.insert(k, v) == newk);
   }
   CHECK(t.size() == ref.size());
   for (auto& [k, v] : ref) CHECK(has(t, k, v));
   for (auto it = ref.begin(); it != ref.end();)
   {
      if (rng() & 1) { CHECK(t.remove(it->first)); it = ref.erase(it); }
      else ++it;
   }
   CHECK(t.size() == ref.size());
   for (auto& [k, v] : ref) CHECK(has(t, k, v));
   CHECK(miss(t, std::string(30, '\xC3')));
}

static void test_buckets_nonpod()
{
   // NON-POD value (std::string) in bucket mode: exercises in-place construct/destruct,
   // copy on find, rebuild-on-remove (reclaim), and the bucket destructor (no leaks).
   map<std::string_view, std::string, mode::buckets>        t;  // Buckets
   std::map<std::string, std::string> ref;
   std::mt19937_64                    rng(77);
   auto mkstr = [&](int n) { std::string s; for (int j = 0; j < n; ++j) s.push_back(char('A' + rng() % 26)); return s; };
   auto mkkey = [&]() { std::string k = "k"; int n = int(rng() % 6); for (int j = 0; j < n; ++j) k.push_back(char('a' + rng() % 4)); return k; };  // clustered → buckets

   for (int i = 0; i < 40000; ++i)
   {
      std::string k = mkkey(), v = mkstr(int(rng() % 40));
      bool        nk = ref.find(k) == ref.end();
      ref[k]         = v;
      CHECK(t.insert(k, std::string(v)) == nk);
   }
   for (auto& [k, v] : ref) { std::string g; CHECK(t.find(k, g) && g == v); }
   // churn: remove ~half (reclaim + non-POD rebuild), then re-insert
   for (auto it = ref.begin(); it != ref.end();)
   {
      if (rng() & 1) { CHECK(t.remove(it->first)); it = ref.erase(it); }
      else ++it;
   }
   CHECK(t.size() == ref.size());
   for (int i = 0; i < 20000; ++i)
   {
      std::string k = mkkey(), v = mkstr(int(rng() % 40));
      bool        nk = ref.find(k) == ref.end();
      ref[k]         = v;
      CHECK(t.insert(k, std::string(v)) == nk);
   }
   CHECK(t.size() == ref.size());
   for (auto& [k, v] : ref) { std::string g; CHECK(t.find(k, g) && g == v); }
}

static void test_adaptive()
{
   // adaptive mode on MIXED data: clustered (shared-prefix) + uniform (incl NUL).
   // Exercises both the radix path (uniform, never trips) and bucket collapse (clustered).
   map<std::string_view, uint32_t, mode::adaptive> t;  // Adaptive
   std::map<std::string, uint32_t> ref;
   std::mt19937_64                 rng(555);
   for (int i = 0; i < 60000; ++i)
   {
      std::string k;
      if (i & 1)  // clustered: shared "node_" prefix, narrow tail
      {
         k = "node_";
         int n = int(rng() % 6);
         for (int j = 0; j < n; ++j) k.push_back(char('a' + rng() % 5));
      }
      else  // uniform
      {
         int n = int(rng() % 12);
         for (int j = 0; j < n; ++j) k.push_back(char(rng() % 256));
      }
      uint32_t v    = uint32_t(rng());
      bool     newk = ref.find(k) == ref.end();
      ref[k]        = v;
      CHECK(t.insert(k, v) == newk);
   }
   CHECK(t.size() == ref.size());
   for (auto& [k, v] : ref) CHECK(has(t, k, v));
   for (auto it = ref.begin(); it != ref.end();)
   {
      if (rng() & 1) { CHECK(t.remove(it->first)); it = ref.erase(it); }
      else ++it;
   }
   CHECK(t.size() == ref.size());
   for (auto& [k, v] : ref) CHECK(has(t, k, v));
}

static void test_random_vs_map()
{
   std::mt19937_64                    rng(12345);
   std::uniform_int_distribution<int> len(0, 24);
   std::uniform_int_distribution<int> byte(0, 255);

   map<std::string_view, uint64_t>               t;
   std::map<std::string, uint64_t> ref;
   for (int i = 0; i < 50000; ++i)
   {
      std::string k;
      int         n = len(rng);
      for (int j = 0; j < n; ++j) k.push_back(char(byte(rng)));
      uint64_t val  = rng();
      bool     newk = ref.find(k) == ref.end();
      ref[k]        = val;
      CHECK(t.insert(k, val) == newk);
   }
   CHECK(t.size() == ref.size());
   for (auto& [k, v] : ref) CHECK(has(t, k, v));
   CHECK(miss(t, std::string(40, '\xAB')));
}

template <class Tree>
static void uu_inline_body()  // INLINE-value path, run per config (radix / buckets)
{
   {
      Tree ti;
      CHECK(ti.insert("alpha", 1));
      CHECK(ti.insert("alpine", 2));
      CHECK(ti.insert("", 9));        // term at root
      CHECK(ti.update("alpha", 100)); // update present → true, value changes
      CHECK(has(ti, "alpha", 100));
      CHECK(ti.update("", 99));        // update the term
      CHECK(has(ti, "", 99));
      const size_t before = ti.size();
      CHECK(!ti.update("alp", 7));     // update absent → false, NO insert
      CHECK(!ti.update("alphabet", 7));
      CHECK(miss(ti, "alp"));
      CHECK(miss(ti, "alphabet"));
      CHECK(ti.size() == before);
      CHECK(!ti.upsert("alpine", 222));   // present → false (not new) + overwrite
      CHECK(has(ti, "alpine", 222));
      CHECK(ti.upsert("alphabet", 333));  // absent → true (new)
      CHECK(has(ti, "alphabet", 333));
      CHECK(ti.update("alphabet", 444));  // now present
      CHECK(has(ti, "alphabet", 444));
   }
}

static void test_update_upsert()
{
   // update = assign-only-if-present (never inserts); upsert = insert-or-assign.
   // Cover every value location: INLINE (int<=5B), LEAF (uint64), TERM, BUCKET — and both
   // the radix and bucket configs (compile-time policies now).
   uu_inline_body<map<std::string_view, int>>();
   uu_inline_body<map<std::string_view, int, mode::buckets>>();  // Buckets

   // LEAF path (uint64) cross-checked vs std::map over a churn of update/upsert.
   map<std::string_view, uint64_t>               t;
   std::map<std::string, uint64_t> ref;
   std::mt19937_64                 rng(31337);
   auto mkkey = [&] { std::string k; int n = int(rng() % 8); for (int j = 0; j < n; ++j) k.push_back(char('a' + rng() % 6)); return k; };
   for (int i = 0; i < 30000; ++i)
   {
      std::string k   = mkkey();
      uint64_t    v   = rng();
      bool        had = ref.count(k) != 0;
      if (rng() & 1) { CHECK(t.update(k, v) == had); if (had) ref[k] = v; }   // update: only if present
      else           { CHECK(t.upsert(k, v) == !had); ref[k] = v; }           // upsert: always
   }
   CHECK(t.size() == ref.size());
   for (auto& [k, v] : ref) CHECK(has(t, k, v));
}

template <class Tree>
static void iter_xcheck_body(unsigned seed)  // ordered cross-check vs std::map, per config
{
   Tree                            t;
   std::map<std::string, uint64_t> ref;
   std::mt19937_64                 rng(seed);
   for (int i = 0; i < 40000; ++i)
   {
      std::string k;
      int         n = int(rng() % 10);
      for (int j = 0; j < n; ++j) k.push_back(char(rng() % 256));
      uint64_t v = rng();
      ref[k]     = v;
      t.upsert(k, v);
   }
   std::vector<std::pair<std::string, uint64_t>> got;
   for (const auto& [k, v] : t) got.emplace_back(std::string(k), v);
   // for_each (recursive visitor) must yield the exact same sequence as the iterator
   std::vector<std::pair<std::string, uint64_t>> got2;
   t.for_each([&](std::string_view k, const uint64_t& v) { got2.emplace_back(std::string(k), v); });
   CHECK(got2 == got);
   CHECK(got.size() == ref.size());
   bool match = (got.size() == ref.size());
   auto it    = ref.begin();
   for (size_t i = 0; i < got.size() && it != ref.end(); ++i, ++it)
      if (got[i].first != it->first || got[i].second != it->second) match = false;
   for (size_t i = 1; i < got.size(); ++i)
      if (!(got[i - 1].first < got[i].first)) match = false;  // strictly ascending, unsigned-byte
   CHECK(match);
}

static void test_iterator()
{
   // empty → begin()==end()
   {
      map<std::string_view, int> t;
      CHECK(t.begin() == t.end());
      int c = 0;
      for (auto kv : t) { (void)kv; ++c; }
      CHECK(c == 0);
   }
   // single key (root leaf)
   {
      map<std::string_view, int>                            t;
      t.insert("solo", 42);
      std::vector<std::pair<std::string, int>> got;
      for (auto it = t.begin(); it != t.end(); ++it) got.emplace_back(std::string(it.key()), it.value());
      CHECK(got.size() == 1 && got[0].first == "solo" && got[0].second == 42);
   }
   // empty-key term + a few keys → ordered, includes ""
   {
      map<std::string_view, int> t;
      t.insert("", 0);
      t.insert("b", 2);
      t.insert("a", 1);
      t.insert("ab", 3);
      std::vector<std::string> ks;
      for (const auto& [k, v] : t) ks.emplace_back(k);
      CHECK((ks == std::vector<std::string>{"", "a", "ab", "b"}));
   }
   // ordered cross-check vs std::map across all modes (arbitrary bytes incl NUL)
   iter_xcheck_body<map<std::string_view, uint64_t>>(900);                      // radix
   iter_xcheck_body<map<std::string_view, uint64_t, mode::dense_tiers>>(901);  // DenseTiers
   iter_xcheck_body<map<std::string_view, uint64_t, mode::buckets>>(902);                // Buckets
   iter_xcheck_body<map<std::string_view, uint64_t, mode::adaptive>>(903);         // Adaptive
   // non-POD values via iterator (bucket mode) — value() returns a real std::string
   {
      map<std::string_view, std::string, mode::buckets> t;  // Buckets
      t.insert("k1", "one");
      t.insert("k3", "three");
      t.insert("k2", "two");
      std::vector<std::pair<std::string, std::string>> got;
      for (const auto& [k, v] : t) got.emplace_back(std::string(k), v);
      CHECK((got == std::vector<std::pair<std::string, std::string>>{{"k1", "one"}, {"k2", "two"}, {"k3", "three"}}));
   }
}

static void test_element_access()
{
   // find_ptr / at / operator[] return REAL references for non-inline T (here uint64).
   map<std::string_view, uint64_t> t;
   t.insert("a", 1);
   t.insert("b", 2);
   CHECK(t.find_ptr("a") && *t.find_ptr("a") == 1);
   CHECK(t.find_ptr("zzz") == nullptr);
   *t.find_ptr("a") = 111;  // mutate through the reference
   CHECK(has(t, "a", uint64_t(111)));
   CHECK(t.at("b") == 2);
   t.at("b") = 222;
   CHECK(has(t, "b", uint64_t(222)));
   bool threw = false;
   try { (void)t.at("nope"); } catch (const std::out_of_range&) { threw = true; }
   CHECK(threw);
   // operator[] : access existing + auto-insert default
   t["a"] += 1;
   CHECK(has(t, "a", uint64_t(112)));
   CHECK(t.size() == 2);
   uint64_t& r = t["c"];  // auto-insert default-constructed 0
   CHECK(t.size() == 3 && r == 0);
   r = 333;
   CHECK(has(t, "c", uint64_t(333)));
   // a mutation through find_ptr is observed by the iterator (it yields the live value)
   *t.find_ptr("c") = 999;
   bool seen = false;
   for (const auto& [k, v] : t)
      if (k == "c") { CHECK(v == 999); seen = true; }
   CHECK(seen);

   // bucket-backed references (non-inline std::string in bucket mode)
   map<std::string_view, std::string, mode::buckets> s;  // Buckets
   s.insert("kx", std::string("hi"));
   CHECK(s.find_ptr("kx") && *s.find_ptr("kx") == "hi");
   s.find_ptr("kx")->append("!");  // mutate in place
   CHECK(*s.find_ptr("kx") == "hi!");
   s["ky"] = "made";
   CHECK(*s.find_ptr("ky") == "made");
}

template <class Tree>
static void lb_xcheck_body(unsigned seed)  // lower/upper/equal_range vs std::map, per config
{
   Tree                            m;
   std::map<std::string, uint64_t> ref;
   std::mt19937_64                 rng(seed);
   for (int i = 0; i < 6000; ++i)
   {
      std::string k;
      int         n = int(rng() % 6);
      for (int j = 0; j < n; ++j) k.push_back(char('a' + rng() % 5));  // clustered → buckets trip
      uint64_t v = rng();
      ref[k]     = v;
      m.upsert(k, v);
   }
   for (int q = 0; q < 4000; ++q)  // probe present + absent keys
   {
      std::string k;
      int         n = int(rng() % 7);
      for (int j = 0; j < n; ++j) k.push_back(char('a' + rng() % 6));
      auto rlo = ref.lower_bound(k), rhi = ref.upper_bound(k);
      auto hlo = m.lower_bound(k), hhi = m.upper_bound(k);
      if (rlo == ref.end()) CHECK(hlo == m.end());
      else                  CHECK(hlo != m.end() && hlo.key() == rlo->first && hlo.value() == rlo->second);
      if (rhi == ref.end()) CHECK(hhi == m.end());
      else                  CHECK(hhi != m.end() && hhi.key() == rhi->first && hhi.value() == rhi->second);
   }
   for (auto& [k, v] : ref)  // equal_range on a present key → exactly that element, then upper
   {
      auto [lo, hi] = m.equal_range(k);
      CHECK(lo != m.end() && lo.key() == k && lo.value() == v);
      auto nx = lo;
      ++nx;
      CHECK(nx == hi);
   }
}

static void test_common_surface()
{
   map<std::string_view, uint64_t> t;
   CHECK(t.empty() && t.size() == 0 && t.max_size() > 0);
   t.insert("b", 2);
   t.insert("d", 4);
   t.insert("f", 6);
   CHECK(!t.empty() && t.size() == 3);
   // swap
   map<std::string_view, uint64_t> u;
   u.insert("z", 26);
   t.swap(u);
   CHECK(t.size() == 1 && has(t, "z", uint64_t(26)));
   CHECK(u.size() == 3 && has(u, "b", uint64_t(2)) && miss(u, "z"));
   t.swap(u);
   CHECK(t.size() == 3 && u.size() == 1);
   // clear → empty + reusable
   u.clear();
   CHECK(u.empty() && u.size() == 0 && miss(u, "z"));
   u.insert("x", 1);
   CHECK(u.size() == 1 && has(u, "x", uint64_t(1)));
   // ordered positioning vs std::map across configs
   lb_xcheck_body<map<std::string_view, uint64_t>>(2025);                // radix
   lb_xcheck_body<map<std::string_view, uint64_t, mode::buckets>>(2026);          // Buckets
   lb_xcheck_body<map<std::string_view, uint64_t, mode::adaptive>>(2027);   // Adaptive
}

struct alloc_counters
{
   size_t allocs = 0, deallocs = 0, constructs = 0, destroys = 0;
   long   live_bytes = 0;
};
// Stateful, instrumented allocator. All rebinds share one alloc_counters (so block
// allocations AND element constructions are tallied together), proving artpp routes both
// memory and element lifetime through the user's allocator.
template <class T>
struct counting_alloc
{
   using value_type                             = T;
   using propagate_on_container_move_assignment = std::true_type;  // propagating allocator
   using propagate_on_container_swap            = std::true_type;
   alloc_counters* c;
   explicit counting_alloc(alloc_counters* cc) noexcept : c(cc) {}
   template <class U>
   counting_alloc(const counting_alloc<U>& o) noexcept : c(o.c) {}
   T* allocate(std::size_t n)
   {
      c->allocs++;
      c->live_bytes += long(n * sizeof(T));
      return static_cast<T*>(::operator new(n * sizeof(T), std::align_val_t(alignof(T))));
   }
   void deallocate(T* p, std::size_t n) noexcept
   {
      c->deallocs++;
      c->live_bytes -= long(n * sizeof(T));
      ::operator delete(p, std::align_val_t(alignof(T)));
   }
   template <class U, class... A>
   void construct(U* p, A&&... a)
   {
      c->constructs++;
      ::new (static_cast<void*>(p)) U(std::forward<A>(a)...);
   }
   template <class U>
   void destroy(U* p) noexcept
   {
      c->destroys++;
      p->~U();
   }
   template <class U>
   bool operator==(const counting_alloc<U>& o) const noexcept { return c == o.c; }
   template <class U>
   bool operator!=(const counting_alloc<U>& o) const noexcept { return c != o.c; }
};

static void test_allocator()
{
   alloc_counters cnt;
   {
      map<std::string_view, std::string, mode::none, counting_alloc<std::string>> t{
          counting_alloc<std::string>(&cnt)};
      for (int i = 0; i < 3000; ++i) t.insert("key" + std::to_string(i), std::string(20, 'x'));
      CHECK(t.size() == 3000);
      CHECK(cnt.allocs > 0);          // arena blocks came from OUR allocator
      CHECK(cnt.constructs >= 3000);  // elements constructed via OUR allocator (+ split-moves)
      CHECK(t.get_allocator() == counting_alloc<std::string>(&cnt));
      std::string v;
      CHECK(t.find("key1234", v) && v == std::string(20, 'x'));
      t.clear();  // frees arena + destroys elements, but tree stays usable
      CHECK(t.empty() && cnt.live_bytes == 0);
      t.insert("again", std::string("y"));
      CHECK(has(t, "again", std::string("y")));
   }
   // after destruction: every block freed, every element destroyed, no bytes outstanding
   CHECK(cnt.allocs == cnt.deallocs);
   CHECK(cnt.constructs == cnt.destroys);
   CHECK(cnt.live_bytes == 0);

   // swap must carry the (stateful) allocator with the arena, so each side stays consistent.
   alloc_counters ca, cb;
   {
      using Tree = map<std::string_view, std::string, mode::none, counting_alloc<std::string>>;
      Tree a{counting_alloc<std::string>(&ca)};
      Tree b{counting_alloc<std::string>(&cb)};
      a.insert("alpha", std::string("A"));
      b.insert("beta", std::string("B"));
      a.swap(b);
      CHECK(a.get_allocator() == counting_alloc<std::string>(&cb));  // allocator swapped with arena
      CHECK(b.get_allocator() == counting_alloc<std::string>(&ca));
      CHECK(has(a, "beta", std::string("B")) && a.size() == 1);
      CHECK(has(b, "alpha", std::string("A")) && b.size() == 1);
      a.insert("gamma", std::string("G"));  // a now allocates via cb (its swapped-in allocator)
   }
   CHECK(ca.allocs == ca.deallocs && ca.live_bytes == 0);  // each counter balances despite the swap
   CHECK(cb.allocs == cb.deallocs && cb.live_bytes == 0);
}

// ── Stage B: allocator-sourced branch-handle width ───────────────────────────
// An allocator opts into a non-default handle via `using artpp_handle = packed_ptr_t<N>;`.
// 8 bytes = full pointers (drops the 48-bit VA assumption); every node layout re-tiles
// from sizeof(Ptr). Cross-check all router families + erase + iteration at that width.
template <class T>
struct wide_handle_alloc : std::allocator<T>
{
   using artpp_handle    = artpp::packed_ptr_t<8>;
   wide_handle_alloc() = default;
   template <class U>
   wide_handle_alloc(const wide_handle_alloc<U>&) noexcept {}
};

template <class Tree>
static void handle_xcheck_body(Tree& t, unsigned seed)  // tree passed in: allocator-armed
{
   std::mt19937_64                    rng(seed);
   std::uniform_int_distribution<int> len(0, 24);
   std::uniform_int_distribution<int> byte(0, 255);
   std::map<std::string, uint64_t>    ref;
   for (int b = 0; b < 256; ++b)  // single-byte fanout → node_full at the 8-byte tiling
   {
      char c = char(b);
      ref[std::string(&c, 1)] = uint64_t(b);
      CHECK(t.insert(std::string_view(&c, 1), uint64_t(b)));
   }
   for (int i = 0; i < 20000; ++i)
   {
      std::string k;
      int         n = len(rng);
      for (int j = 0; j < n; ++j) k.push_back(char(byte(rng)));
      uint64_t val  = rng();
      bool     newk = ref.find(k) == ref.end();
      ref[k]        = val;
      CHECK(t.insert(k, val) == newk);
   }
   CHECK(t.size() == ref.size());
   for (auto& [k, v] : ref) CHECK(has(t, k, v));
   auto rit = ref.begin();  // ordered iteration matches the reference order
   for (auto it = t.begin(); it != t.end(); ++it, ++rit)
      CHECK(rit != ref.end() && it.key_bytes() == rit->first && it.value() == rit->second);
   CHECK(rit == ref.end());
   size_t n = 0;  // erase every other key, re-verify
   for (auto i = ref.begin(); i != ref.end();)
   {
      if (++n & 1) { CHECK(t.erase(i->first) == 1); i = ref.erase(i); }
      else ++i;
   }
   CHECK(t.size() == ref.size());
   for (auto& [k, v] : ref) CHECK(has(t, k, v));
   for (auto& [k, v] : ref) CHECK(t.erase(k) == 1);  // erase the rest: removal must SHRINK
   CHECK(t.empty());
   auto d = t.debug_stats();  // ...all the way back to an empty tree — zero nodes left
   CHECK(d.terminals == 0 && d.leaf + d.inl + d.prefix + d.setlist + d.setlist16 + d.c2 + d.c4 +
                                     d.c8 + d.full ==
                                 0);
   t.clear();
   CHECK(t.empty());
}

static void test_wide_handle()
{
   using WA = wide_handle_alloc<uint64_t>;
   { map<std::string_view, uint64_t, mode::none, WA> t; handle_xcheck_body(t, 99); }
   { map<std::string_view, uint64_t, mode::adaptive | mode::dense_tiers, WA> t; handle_xcheck_body(t, 7); }
   { map<std::string_view, uint64_t, mode::adaptive | mode::wide_stems, WA> t; handle_xcheck_body(t, 31); }
   // inline-value path at 8 bytes (cap = 7): a 4-byte value packs into the handle
   map<std::string_view, uint32_t, mode::none, wide_handle_alloc<uint32_t>> ti;
   for (uint32_t i = 0; i < 1000; ++i) CHECK(ti.insert("k" + std::to_string(i), i));
   for (uint32_t i = 0; i < 1000; ++i) CHECK(has(ti, "k" + std::to_string(i), i));
}

template <class Tree>
static void shrink_body(unsigned seed)  // removal must shrink: full delete -> zero nodes
{
   std::mt19937_64                    rng(seed);
   std::uniform_int_distribution<int> len(0, 24);
   std::uniform_int_distribution<int> byte(0, 255);
   Tree                               t;
   std::map<std::string, uint64_t>    ref;
   for (int b = 0; b < 256; ++b)  // dense fanout -> node_full (de-widen coverage)
   {
      char c = char(b);
      ref[std::string(&c, 1)] = uint64_t(b);
      t.insert(std::string_view(&c, 1), uint64_t(b));
   }
   for (int i = 0; i < 30'000; ++i)
   {
      std::string k;
      int         m = len(rng);
      for (int j = 0; j < m; ++j) k.push_back(char(byte(rng)));
      ref[k] = uint64_t(i);
      t.insert(k, uint64_t(i));
   }
   // delete 90%: structure must stay compact (collapsed: routers can't outnumber terminals)
   size_t n = 0;
   for (auto i = ref.begin(); i != ref.end();)
   {
      if (++n % 10 != 0) { CHECK(t.erase(i->first) == 1); i = ref.erase(i); }
      else ++i;
   }
   for (auto& [k, v] : ref) CHECK(has(t, k, v));
   auto d       = t.debug_stats();
   auto routers = d.setlist + d.setlist16 + d.c2 + d.c4 + d.c8 + d.full;
   // Strict census invariants only where every entry is a counted terminal — the debug
   // census deliberately doesn't descend buckets, so bucket-capable modes get sanity only.
   if constexpr (!has_mode(Tree::policy, mode::buckets) && !has_mode(Tree::policy, mode::adaptive))
   {
      CHECK(d.terminals == ref.size());
      CHECK(routers < d.terminals + 1);  // collapsed: routers can't outnumber terminals
      CHECK(d.prefix <= d.terminals);
   }
   else
      CHECK(routers <= ref.size());
   for (auto& [k, v] : ref) CHECK(t.erase(k) == 1);  // delete the rest -> empty structure
   d = t.debug_stats();
   CHECK(t.empty() && d.terminals == 0 &&
         d.leaf + d.inl + d.prefix + d.setlist + d.setlist16 + d.c2 + d.c4 + d.c8 + d.full == 0);
}

static void test_remove_shrink()
{
   shrink_body<map<std::string_view, uint64_t>>(11);
   shrink_body<map<std::string_view, uint64_t, mode::adaptive | mode::dense_tiers>>(12);
   shrink_body<map<std::string_view, uint64_t, mode::adaptive | mode::wide_stems>>(13);
   shrink_body<map<std::string_view, uint64_t, mode::buckets>>(14);
   {  // pool round-trip: delete-all then identical rebuild reuses every line
      artpp::line_pool pool;
      using PA = artpp::pool_alloc<uint64_t>;
      map<std::string_view, uint64_t, mode::none, PA> t{PA(&pool)};
      std::mt19937_64          rng(15);
      std::vector<std::string> keys(20'000);
      for (auto& k : keys)
      {
         k.resize(8 + rng() % 9);
         for (auto& c : k) c = char(rng());
      }
      for (size_t i = 0; i < keys.size(); ++i) t.insert(keys[i], i);
      const size_t lines1 = pool.used_lines();
      for (auto& k : keys) CHECK(t.erase(k) == 1);
      CHECK(t.empty());
      for (size_t i = 0; i < keys.size(); ++i) t.insert(keys[i], i);
      CHECK(pool.used_lines() == lines1);  // every node recycled through the free lists
      for (size_t i = 0; i < keys.size(); ++i) CHECK(has(t, keys[i], uint64_t(i)));
   }
}

// Counts every construction — proves emplace builds T exactly once, AT the leaf.
struct probe
{
   int               v = 0;
   static inline int ctors = 0, moves = 0, copies = 0;
   probe() { ++ctors; }
   explicit probe(int x) : v(x) { ++ctors; }
   probe(probe&& o) noexcept : v(o.v) { ++moves; }
   probe(const probe& o) : v(o.v) { ++copies; }
   probe& operator=(probe&&) noexcept = default;
   probe& operator=(const probe&)     = default;
};

static void test_emplace()  // std::map semantics: construct from args, insert iff absent
{
   {  // in-place proof: one ctor at the final address, zero moves/copies
      map<std::string_view, probe> t;
      probe::ctors = probe::moves = probe::copies = 0;
      CHECK(t.emplace("alpha", 41));
      CHECK(probe::ctors == 1 && probe::moves == 0 && probe::copies == 0);
      CHECK(!t.emplace("alpha", 99));  // present: args never even construct a T
      CHECK(probe::ctors == 1 && probe::moves == 0 && probe::copies == 0);
      CHECK(t.find_ptr("alpha") && t.find_ptr("alpha")->v == 41);
      map<std::string_view, probe> t2;  // fresh tree: no split relocations in the count
      probe::ctors = probe::moves = probe::copies = 0;
      CHECK(t2.insert("beta", probe(7)));  // insert path: construct + the one move
      CHECK(probe::ctors == 1 && probe::moves == 1 && probe::copies == 0);
   }
   map<std::string_view, std::string> t;
   CHECK(t.emplace("a", 3, 'x'));  // string(3, 'x')
   CHECK(has(t, "a", std::string("xxx")));
   CHECK(!t.emplace("a", 5, 'y'));  // present: untouched
   CHECK(has(t, "a", std::string("xxx")));
   CHECK(t.try_emplace("b", "hello"));
   CHECK(!t.try_emplace("b"));
   CHECK(has(t, "b", std::string("hello")));

   map<std::string_view, uint32_t> ti;  // inline-value + term paths
   CHECK(ti.emplace("k", 7u));
   CHECK(!ti.emplace("k", 9u));
   CHECK(has(ti, "k", 7u));
   CHECK(ti.emplace("kk", 1u));   // "k" becomes a term under a router
   CHECK(!ti.emplace("k", 2u));   // term present: untouched
   CHECK(has(ti, "k", 7u) && has(ti, "kk", 1u));
   for (int b = 0; b < 256; ++b)  // node_full term path
   {
      char c = char(b);
      ti.emplace(std::string_view(&c, 1), uint32_t(b));
   }
   CHECK(!ti.emplace("k", 3u) && has(ti, "k", 7u));

   map<std::string_view, uint64_t, mode::buckets> tb;  // bucket entry path
   CHECK(tb.emplace("x", uint64_t(1)));
   CHECK(!tb.emplace("x", uint64_t(2)));
   CHECK(has(tb, "x", uint64_t(1)));

   map<std::string_view, uint64_t, mode::adaptive> ta;  // collapse-to-bucket path
   CHECK(ta.emplace("deep/narrow/aaaa", uint64_t(1)));
   CHECK(ta.emplace("deep/narrow/aaab", uint64_t(2)));
   CHECK(!ta.emplace("deep/narrow/aaaa", uint64_t(9)));
   CHECK(has(ta, "deep/narrow/aaaa", uint64_t(1)));
}

// ── Stage C: the 4-byte indexed handle over artpp::line_pool ──────────────────
// Every branch is a 28-bit cacheline index + 4-bit tag resolved as base + (idx << 7);
// the same cross-checks run with anonymous and file-backed pools, every router family,
// erase + ordered iteration. Freelist reuse is asserted by rebuilding identical content.
static void test_line_pool()
{
   using PA = artpp::pool_alloc<uint64_t>;
   {  // anonymous backing, all three router-family configs
      artpp::line_pool pool;
      map<std::string_view, uint64_t, mode::none, PA> t{PA(&pool)};
      handle_xcheck_body(t, 1234);
   }
   {
      artpp::line_pool pool;
      map<std::string_view, uint64_t, mode::adaptive | mode::dense_tiers, PA> t{PA(&pool)};
      handle_xcheck_body(t, 5);
   }
   {
      artpp::line_pool pool;
      map<std::string_view, uint64_t, mode::adaptive | mode::wide_stems, PA> t{PA(&pool)};
      handle_xcheck_body(t, 6);
   }
   {  // file-backed: same engine over an mmap'd file; the file grows with the commits
      char path[] = "/tmp/artpp_pool_smoke_XXXXXX";
      int  fd     = mkstemp(path);
      CHECK(fd >= 0);
      ::close(fd);
      {
         artpp::line_pool pool(path);
         map<std::string_view, uint64_t, mode::none, PA> t{PA(&pool)};
         handle_xcheck_body(t, 777);
         struct stat st{};
         CHECK(::stat(path, &st) == 0 && size_t(st.st_size) == pool.committed());
         CHECK(pool.committed() > 0);
      }
      ::unlink(path);
   }
   {  // freelist reuse: clear() then rebuild the same content — the frontier must not move
      artpp::line_pool pool;
      map<std::string_view, uint64_t, mode::none, PA> t{PA(&pool)};
      std::mt19937_64 rng(42);
      std::vector<std::string> keys(50'000);
      for (auto& k : keys)
      {
         k.resize(8 + rng() % 9);
         for (auto& c : k) c = char(rng());
      }
      for (size_t i = 0; i < keys.size(); ++i) t.insert(keys[i], i);
      const size_t lines1 = pool.used_lines();
      t.clear();
      for (size_t i = 0; i < keys.size(); ++i) t.insert(keys[i], i);
      CHECK(pool.used_lines() == lines1);  // every node came back from the free lists
      for (size_t i = 0; i < keys.size(); ++i) CHECK(has(t, keys[i], uint64_t(i)));
   }
   {  // inline-value path at 4 bytes (cap = 3): a 2-byte value packs into the handle
      artpp::line_pool pool;
      map<std::string_view, uint16_t, mode::none, artpp::pool_alloc<uint16_t>> ti{
          artpp::pool_alloc<uint16_t>(&pool)};
      for (uint16_t i = 0; i < 1000; ++i) CHECK(ti.insert("k" + std::to_string(i), i));
      for (uint16_t i = 0; i < 1000; ++i) CHECK(has(ti, "k" + std::to_string(i), i));
   }
}

// Generic container-interface conformance — written ONLY against the STL map subset,
// so the identical test validates artpp::map AND std::map<std::string, V>. Keys are
// passed as const char*, which converts to either container's key_type.
template <class Map>
static void api_conformance()
{
   Map m;
   CHECK(m.empty() && m.size() == 0);

   m.insert_or_assign("banana", 1);
   m.insert_or_assign("apple", 2);
   m.insert_or_assign("cherry", 3);
   m.insert_or_assign("apple", 22);  // update, not insert
   CHECK(m.size() == 3 && !m.empty());

   CHECK(m.contains("apple") && m.count("apple") == 1 && m.count("zzz") == 0);

   auto it = m.find("apple");
   CHECK(it != m.end() && it->first == "apple" && it->second == 22);
   CHECK(m.find("zzz") == m.end());

   CHECK(m.at("banana") == 1);
   m["date"] = 4;  // insert via operator[]
   CHECK(m.size() == 4 && m.at("date") == 4);

   // ordered iteration (range-for + structured bindings)
   std::vector<std::string> ks;
   for (const auto& [k, v] : m) ks.emplace_back(k);
   CHECK((ks == std::vector<std::string>{"apple", "banana", "cherry", "date"}));

   // ordered positioning
   CHECK(m.lower_bound("apple")->first == "apple");
   CHECK(m.upper_bound("apple")->first == "banana");
   CHECK(m.lower_bound("b")->first == "banana");  // absent key → next greater
   auto [lo, hi] = m.equal_range("cherry");
   CHECK(lo->first == "cherry");
   auto nx = lo;
   ++nx;
   CHECK(nx == hi);

   // post-increment + pure equality
   auto i2 = m.begin();
   auto i3 = i2++;
   CHECK(i3 == m.begin() && i2 != m.begin());

   // erase by key returns the count
   CHECK(m.erase("banana") == 1 && m.erase("zzz") == 0);
   CHECK(m.size() == 3 && !m.contains("banana"));

   m.try_emplace("kiwi", 9);   // same shape + semantics on std::map and map
   m.try_emplace("kiwi", 10);  // present: untouched (both containers)
   CHECK(m.size() == 4 && m.find("kiwi")->second == 9);

   m.clear();
   CHECK(m.empty() && m.size() == 0);
}

static void test_api_conformance()
{
   api_conformance<std::map<std::string, uint64_t>>();                     // the reference: std::map
   api_conformance<map<std::string_view, uint64_t>>();                 // artpp radix, view keys
   api_conformance<map<std::string, uint64_t>>();                      // artpp, OWNING std::string keys
   api_conformance<map<std::string_view, uint64_t, mode::buckets>>();  // artpp buckets
   api_conformance<map<std::string_view, uint64_t, mode::adaptive>>(); // artpp adaptive
}

// Non-string keys via key_codec: cross-check vs std::map<KeyT,V> over random keys —
// proves the big-endian (sign-flipped) encoding makes byte order == numeric order, so
// iteration is in true numeric order and lookups match. it.key() returns KeyT by value.
template <class KeyT>
static void typed_key_body(uint64_t seed)
{
   map<KeyT, uint32_t> h;
   std::map<KeyT, uint32_t> m;
   std::mt19937_64          rng(seed);
   for (int i = 0; i < 5000; ++i)
   {
      KeyT     k = KeyT(rng());
      uint32_t v = uint32_t(rng());
      h.insert_or_assign(k, v);
      m[k] = v;
   }
   CHECK(h.size() == m.size());
   // iterate both: identical (key,value) sequence, strictly ascending in numeric KeyT order
   auto mi    = m.begin();
   bool ok    = true;
   bool first = true;
   KeyT prev{};
   for (auto it = h.begin(); it != h.end(); ++it, ++mi)
   {
      if (mi == m.end()) { ok = false; break; }
      if (it->first != mi->first || it->second != mi->second) ok = false;
      if (!first && !(prev < it->first)) ok = false;  // numeric ascending (incl. signed)
      prev  = it->first;
      first = false;
   }
   CHECK(ok && mi == m.end());
   // point lookups by typed key
   for (int i = 0; i < 1000; ++i)
   {
      KeyT k  = KeyT(rng());
      auto a  = h.find(k);
      auto b  = m.find(k);
      CHECK(h.contains(k) == (b != m.end()));
      if (b == m.end()) CHECK(a == h.end());
      else              CHECK(a != h.end() && a->second == b->second && a->first == k);
   }
}

static void test_typed_keys()
{
   typed_key_body<uint64_t>(11);
   typed_key_body<int64_t>(22);   // signed → negatives must sort before positives
   typed_key_body<uint32_t>(33);
}

static void test_pmr_move()
{
   // PMR polymorphic_allocator: non-propagating + not-always-equal. Move-assigning between
   // two distinct memory_resources can't steal the arena → element-wise move into the
   // target's allocator (move_elements_from_), keeping the target's allocator.
   std::pmr::monotonic_buffer_resource rA, rB;
   using A    = std::pmr::polymorphic_allocator<std::string>;
   using Tree = map<std::string_view, std::string, mode::none, A>;
   Tree a(A{&rA}), b(A{&rB});
   a.insert("keep", std::string("old"));
   for (int i = 0; i < 2000; ++i) b.insert("k" + std::to_string(i), std::string(20, 'z'));

   a = std::move(b);  // unequal, non-POCMA → drain b's elements into a's allocator
   CHECK(a.size() == 2000);
   CHECK(a.get_allocator() == A{&rA});  // a keeps ITS allocator (non-propagating)
   CHECK(!a.contains("keep"));          // a's prior content replaced
   std::string v;
   CHECK(a.find("k1234", v) && v.size() == 20);
   CHECK(b.size() == 0);  // source drained, valid-empty
}

int main()
{
   test_basic();
   test_fanout256();
   test_long_prefix();
   test_nonpod();
   test_dense_tiers();
   test_remove();
   test_buckets();
   test_buckets_nonpod();
   test_adaptive();
   test_update_upsert();
   test_iterator();
   test_element_access();
   test_common_surface();
   test_allocator();
   test_remove_shrink();
   test_emplace();
   test_wide_handle();
   test_line_pool();
   test_pmr_move();
   test_api_conformance();
   test_typed_keys();
   test_random_vs_map();
   if (g_fail == 0) std::printf("artpp_smoke: ALL PASS\n");
   else             std::printf("artpp_smoke: %d FAILURES\n", g_fail);
   return g_fail ? 1 : 0;
}

// Exception-safety conformance via exhaustive fault injection. A throwing allocator
// fails on a countdown; we sweep the countdown over every allocation an insert makes and,
// on each induced throw, check the STRONG guarantee: the container is unchanged. std::map
// is the oracle (single-element insert is strong) — run the identical generic harness on
// it and on map; the deltas are artpp's bugs.
#include "artpp/map.hpp"

#include <cstdint>
#include <cstdio>
#include <map>
#include <new>
#include <string>
#include <utility>
#include <vector>

using artpp::map;
using artpp::mode;

struct fail_counter
{
   long countdown = -1;  // -1 = never fail; else throw bad_alloc when it reaches 0
   long allocs = 0, frees = 0;
};

// Value type that counts live instances, so a SKIPPED destructor on a throwing path shows up
// as g_live != 0 after the container dies. (The allocator-balance check can't see this: a
// value's resources leak even when its node's memory is reclaimed.)
static long g_live = 0;
struct counted
{
   std::string s;
   counted() { ++g_live; }
   counted(std::string v) : s(std::move(v)) { ++g_live; }       // implicit: insert(k, std::string)
   counted(const counted& o) : s(o.s) { ++g_live; }
   counted(counted&& o) noexcept : s(std::move(o.s)) { ++g_live; }
   counted& operator=(const counted&) = default;               // assign: no live-count change
   counted& operator=(counted&&) noexcept = default;
   ~counted() { --g_live; }
   operator std::string() const { return s; }                  // for dump()
};

template <class T>
struct throwing_alloc
{
   using value_type = T;
   fail_counter* fc;
   explicit throwing_alloc(fail_counter* f) noexcept : fc(f) {}
   template <class U>
   throwing_alloc(const throwing_alloc<U>& o) noexcept : fc(o.fc) {}
   T* allocate(std::size_t n)
   {
      if (fc->countdown >= 0 && fc->countdown-- == 0) throw std::bad_alloc();
      ++fc->allocs;
      return static_cast<T*>(::operator new(n * sizeof(T), std::align_val_t(alignof(T))));
   }
   void deallocate(T* p, std::size_t n) noexcept
   {
      ++fc->frees;
      ::operator delete(p, std::align_val_t(alignof(T)));
   }
   template <class U>
   bool operator==(const throwing_alloc<U>& o) const noexcept { return fc == o.fc; }
   template <class U>
   bool operator!=(const throwing_alloc<U>& o) const noexcept { return fc != o.fc; }
};

template <class Map>
static std::vector<std::pair<std::string, std::string>> dump(const Map& m)
{
   std::vector<std::pair<std::string, std::string>> v;
   for (const auto& [k, val] : m) v.emplace_back(std::string(k), std::string(val));
   return v;
}

static int g_fail = 0;

// Sweep every allocation point during inserting `nk` into a tree built from `pre`,
// asserting the strong guarantee on each induced bad_alloc.
template <class Map>
static void eh_insert(const char* cont, const char* scen, const std::vector<std::string>& pre,
                      const std::string& nk)
{
   int    violations = 0, leaks = 0, vleaks = 0, swept = 0;
   for (int n = 0;; ++n)
   {
      fail_counter fc;
      bool         completed = false, threw = false;
      const long   vbase     = g_live;  // live values before this map exists
      {
         typename Map::allocator_type al(&fc);
         Map                          m(al);
         for (size_t i = 0; i < pre.size(); ++i) m.insert_or_assign(pre[i], std::string(8, char('a' + (i & 15))));
         auto before = dump(m);

         fc.countdown = n;  // arm the (n+1)-th allocation to throw
         try
         {
            m.insert_or_assign(nk, std::string("NEWVALUE"));
            completed = true;
         }
         catch (const std::bad_alloc&) { threw = true; }
         fc.countdown = -1;  // disarm before verification reads

         auto after = dump(m);  // also exercises validity (ASan traps a corrupt tree here)
         if (threw && after != before) ++violations;  // STRONG violated: insert had an effect
         if (completed)
         {
            bool found = false;
            for (auto& kv : after)
               if (kv.first == nk) found = true;
            if (!found) ++violations;
         }
      }  // map destructed
      if (fc.allocs != fc.frees) ++leaks;          // node memory leaked
      if (g_live != vbase) ++vleaks;               // a value's destructor was skipped
      ++swept;
      if (completed) break;
   }
   const bool ok = (violations == 0 && leaks == 0 && vleaks == 0);
   if (!ok) ++g_fail;
   std::printf("  %-10s %-14s  %s  (%d pts, %d strong-viol, %d mem-leak, %d dtor-leak)\n", cont,
               scen, ok ? "OK  " : "FAIL", swept, violations, leaks, vleaks);
}

// Basic guarantee under fault during a CLUSTERED bulk insert — this actually exercises
// bkt_split / collapse_to_bucket / widen (which the single-insert harness rarely arms).
// On any induced throw we require no corruption (clean iteration, consistent size, every
// visible key findable) and no leak.
template <class Map>
static void eh_bulk(const char* cont, const char* scen, const std::vector<std::string>& keys)
{
   int violations = 0, leaks = 0, vleaks = 0, swept = 0;
   for (int n = 0;; ++n)
   {
      fail_counter fc;
      bool         completed = false;
      const long   vbase     = g_live;
      {
         typename Map::allocator_type al(&fc);
         Map                          m(al);
         fc.countdown = n;
         size_t done  = 0;
         try
         {
            for (const auto& k : keys) { m.insert_or_assign(k, std::string(6, 'v')); ++done; }
            completed = (done == keys.size());
         }
         catch (const std::bad_alloc&) {}
         fc.countdown = -1;
         auto d = dump(m);  // clean iteration of a possibly-half-built tree (ASan traps corruption)
         if (d.size() != m.size()) ++violations;
         for (auto& kv : d)
            if (!m.contains(kv.first)) ++violations;
      }
      if (fc.allocs != fc.frees) ++leaks;
      if (g_live != vbase) ++vleaks;
      ++swept;
      if (completed) break;
   }
   const bool ok = (violations == 0 && leaks == 0 && vleaks == 0);
   if (!ok) ++g_fail;
   std::printf("  %-10s %-14s  %s  (%d points, %d corruption, %d mem-leak, %d dtor-leak)\n", cont, scen,
               ok ? "OK  " : "FAIL", swept, violations, leaks, vleaks);
}

// ── throwing VALUE constructors (audit #12) ──────────────────────────────────────
// The allocator sweep above never throws from a value. `tvalue` throws from whichever of
// its constructors is the n-th to run; sweeping n over an operation arms every value
// construction it performs: emplace's in-place construction from arguments, the copy
// insert(k, const T&) makes, the move into the leaf, and the move of an OLD value when a
// leaf splits. A leaf allocated for a value whose constructor then throws must be freed.
static long g_vcountdown = -1;  // -1 = never; else the constructor that finds 0 throws
struct value_error {};
struct tvalue
{
   std::string s;
   static void tick() { if (g_vcountdown >= 0 && g_vcountdown-- == 0) throw value_error{}; }
   tvalue() { tick(); ++g_live; }
   tvalue(std::string v) : s(std::move(v)) { tick(); ++g_live; }
   tvalue(const char* a, int n) : s(std::string(a) + std::to_string(n)) { tick(); ++g_live; }  // emplace args
   tvalue(const tvalue& o) : s(o.s) { tick(); ++g_live; }
   tvalue(tvalue&& o)  // throws BEFORE touching the source, so a failed move leaves o intact
   {
      tick();
      s = std::move(o.s);
      ++g_live;
   }
   tvalue& operator=(const tvalue&) = default;
   tvalue& operator=(tvalue&&)      = default;
   ~tvalue() { --g_live; }
   operator std::string() const { return s; }
};

// op 0: emplace(nk, "E", 7)   op 1: insert_or_assign(nk, const tvalue&)   op 2: insert(nk, tvalue&&)
// On every induced throw: the contents are unchanged (strong), the map still iterates, and
// after destruction every block and every value is accounted for.
template <class Map>
static void eh_value(const char* cont, const char* scen, const std::vector<std::string>& pre,
                     const std::string& nk, int opk)
{
   int violations = 0, leaks = 0, vleaks = 0, swept = 0;
   for (int n = 0;; ++n)
   {
      fail_counter fc;
      bool         completed = false, threw = false;
      const long   vbase     = g_live;
      {
         typename Map::allocator_type al(&fc);
         Map                          m(al);
         for (size_t i = 0; i < pre.size(); ++i) m.insert_or_assign(pre[i], std::string(8, char('a' + (i & 15))));
         auto         before = dump(m);
         const tvalue src(std::string("NEWVALUE"));
         tvalue       mv(std::string("NEWVALUE"));
         g_vcountdown = n;  // arm the (n+1)-th value construction to throw
         try
         {
            if constexpr (requires { typename Map::key_compare; })  // the std::map oracle's spellings
            {
               if (opk == 0) m.try_emplace(nk, "E", 7);
               else if (opk == 1) m.insert_or_assign(nk, src);
               else m.insert_or_assign(nk, std::move(mv));
            }
            else
            {
               if (opk == 0) m.emplace(nk, "E", 7);
               else if (opk == 1) m.insert_or_assign(nk, src);
               else m.insert(nk, std::move(mv));
            }
            completed = true;
         }
         catch (const value_error&) { threw = true; }
         g_vcountdown = -1;
         auto after = dump(m);
         if (threw && after != before) ++violations;
         if (completed && !m.contains(nk)) ++violations;
         if (m.size() != after.size()) ++violations;
      }
      if (fc.allocs != fc.frees) ++leaks;
      if (g_live != vbase) ++vleaks;
      ++swept;
      if (completed) break;
   }
   const bool ok = (violations == 0 && leaks == 0 && vleaks == 0);
   if (!ok) ++g_fail;
   static const char* const opn[] = {"emplace", "insert copy", "insert move"};
   std::printf("  %-10s %-14s %-11s  %s  (%d pts, %d strong-viol, %d mem-leak, %d dtor-leak)\n", cont,
               scen, opn[opk], ok ? "OK  " : "FAIL", swept, violations, leaks, vleaks);
}
template <class Map>
static void run_value_scenarios(const char* cont)
{
   std::vector<std::string> fan;
   for (int b = 0; b < 16; ++b) fan.push_back(std::string(1, char('a' + b)));
   for (int opk = 0; opk < 3; ++opk)
   {
      eh_value<Map>(cont, "empty->x", {}, "x", opk);                                  // make_leaf_full
      eh_value<Map>(cont, "long-key", {}, std::string(300, 'k'), opk);                // make_leaf
      eh_value<Map>(cont, "leaf-split", {"apple"}, "apricot", opk);                   // + old value moves
      eh_value<Map>(cont, "key-is-prefix", {"apple"}, "app", opk);                    // new key is the term
      eh_value<Map>(cont, "extends-key", {"app"}, "apple", opk);                      // old key is the term
      eh_value<Map>(cont, "prefix-split", {"abcdefgh1", "abcdefgh2"}, "abcZ", opk);
      eh_value<Map>(cont, "new-branch", {"node_aa", "node_ab", "node_ba"}, "node_c", opk);
      eh_value<Map>(cont, "setlist-widen", fan, "Z", opk);
      eh_value<Map>(cont, "present-key", {"apple", "apricot"}, "apple", opk);
   }
}

// Repeated throwing emplaces must not consume anything: 1,000 of them leak 0 blocks and —
// in bucket mode — leave the page's free space intact, so a key that fits still goes in
// without a single allocation (a constructor that threw after its entry bytes were claimed
// used to eat the page until the next insert burst the bucket).
template <class Map>
static void eh_value_repeat(const char* cont, bool expect_no_alloc_after)
{
   fail_counter fc;
   const long   vbase = g_live;
   bool         ok    = true;
   {
      typename Map::allocator_type al(&fc);
      Map                          m(al);
      m.insert_or_assign("seed", std::string("s"));
      for (int i = 0; i < 1000; ++i)
      {
         g_vcountdown = 0;
         try { m.emplace("some-key", "E", i); ok = false; }
         catch (const value_error&) {}
         g_vcountdown = -1;
      }
      ok = ok && m.size() == 1 && !m.contains("some-key");
      const long a0 = fc.allocs;
      m.emplace("sx", "E", 1);  // shares the bucket page with "seed"
      ok = ok && m.size() == 2 && m.contains("sx");
      if (expect_no_alloc_after) ok = ok && fc.allocs == a0;
   }
   ok = ok && fc.allocs == fc.frees && g_live == vbase;
   if (!ok) ++g_fail;
   std::printf("  %-10s 1000 throwing emplaces  %s  (%ld allocs, %ld frees)\n", cont, ok ? "OK  " : "FAIL",
               fc.allocs, fc.frees);
}

// ── small INLINE values (audit #13) ──────────────────────────────────────────────
// A trivially copyable value of <= 5 bytes lives in the handle itself, so the harness above
// (std::string values) never reaches the code that splits such a terminal: it allocates a
// router and then a leaf for the new key, and a failure of the second allocation used to
// leak the router. Same sweep, uint32_t values: strong guarantee and zero leaks at every
// allocation point.
template <class Map>
static void eh_insert_inline(const char* cont, const char* scen, const std::vector<std::string>& pre,
                             const std::string& nk)
{
   auto dump32 = [](const Map& m) {
      std::vector<std::pair<std::string, uint32_t>> v;
      for (auto it = m.begin(); it != m.end(); ++it) v.emplace_back(std::string(it.key_bytes()), it.value());
      return v;
   };
   int violations = 0, leaks = 0, swept = 0;
   for (int n = 0;; ++n)
   {
      fail_counter fc;
      bool         completed = false, threw = false;
      {
         typename Map::allocator_type al(&fc);
         Map                          m(al);
         for (size_t i = 0; i < pre.size(); ++i) m.insert_or_assign(pre[i], uint32_t(100 + i));
         auto before  = dump32(m);
         fc.countdown = n;
         try
         {
            m.insert_or_assign(nk, uint32_t(7));
            completed = true;
         }
         catch (const std::bad_alloc&) { threw = true; }
         fc.countdown = -1;
         auto after = dump32(m);
         if (threw && after != before) ++violations;
         uint32_t got = 0;
         if (completed && !(m.find(nk, got) && got == 7)) ++violations;
         if (m.size() != after.size()) ++violations;
      }
      if (fc.allocs != fc.frees) ++leaks;
      ++swept;
      if (completed) break;
   }
   const bool ok = (violations == 0 && leaks == 0);
   if (!ok) ++g_fail;
   std::printf("  %-10s %-14s  %s  (%d pts, %d strong-viol, %d mem-leak)\n", cont, scen,
               ok ? "OK  " : "FAIL", swept, violations, leaks);
}
template <class Map>
static void run_inline_scenarios(const char* cont)
{
   eh_insert_inline<Map>(cont, "empty->x", {}, "x");
   eh_insert_inline<Map>(cont, "inline-split", {"ab", "ac"}, "ab" + std::string(40, 'z'));  // the #13 shape
   eh_insert_inline<Map>(cont, "inline-split1", {"ab", "ac"}, "abq");
   eh_insert_inline<Map>(cont, "root-inline", {""}, "q");          // the root itself is an inline value
   eh_insert_inline<Map>(cont, "leaf-split", {"apple"}, "apricot");
   eh_insert_inline<Map>(cont, "key-is-prefix", {"apple"}, "app");
   eh_insert_inline<Map>(cont, "prefix-split", {"abcdefgh1", "abcdefgh2"}, "abcZ");
   eh_insert_inline<Map>(cont, "setlist-widen",
                         [] { std::vector<std::string> v; for (int b = 0; b < 16; ++b) v.push_back(std::string(1, char('a' + b))); return v; }(),
                         "Z");
}

template <class Map>
static void run_scenarios(const char* cont)
{
   eh_insert<Map>(cont, "empty->x", {}, "x");
   eh_insert<Map>(cont, "leaf-split", {"apple"}, "apricot");                      // split_leaf
   eh_insert<Map>(cont, "prefix-split", {"abcdefgh1", "abcdefgh2"}, "abcZ");      // split_prefix
   eh_insert<Map>(cont, "setlist-widen",                                         // setlist -> node_full
                  [] { std::vector<std::string> v; for (int b = 0; b < 16; ++b) v.push_back(std::string(1, char('a' + b))); return v; }(),
                  "Z");
   eh_insert<Map>(cont, "deep-cluster",
                  {"node_aa", "node_ab", "node_ac", "node_ba", "node_bb"}, "node_ad");
   {
      std::vector<std::string> pre;
      for (int a = 0; a < 6; ++a) for (int b = 0; b < 6; ++b) pre.push_back(std::string("k") + char('a'+a) + char('a'+b));
      eh_insert<Map>(cont, "bucket-fill", pre, "kZZ");
   }
}

int main()
{
   using RefAlloc = throwing_alloc<std::pair<const std::string, counted>>;
   using RefMap   = std::map<std::string, counted, std::less<>, RefAlloc>;
   using HkvRadix = map<std::string_view, counted, mode::none, throwing_alloc<counted>>;  // flat (default)
   using HkvBkt   = map<std::string_view, counted, mode::buckets, throwing_alloc<counted>>;
   using HkvAdp   = map<std::string_view, counted, mode::adaptive, throwing_alloc<counted>>;
   using HkvDense = map<std::string_view, counted, mode::adaptive | mode::dense_tiers, throwing_alloc<counted>>;
   using HkvWide  = map<std::string_view, counted, mode::wide_stems, throwing_alloc<counted>>;

   std::printf("exception-safety (strong insert under bad_alloc injection):\n");
   run_scenarios<RefMap>("std::map");
   run_scenarios<HkvRadix>("artpp-radix");
   run_scenarios<HkvBkt>("artpp-bucket");
   run_scenarios<HkvAdp>("artpp-adapt");
   run_scenarios<HkvDense>("artpp-dense");
   run_scenarios<HkvWide>("artpp-wide");

   // Clustered bulk insert under fault — drives bkt_split / collapse / widen (basic guarantee).
   std::vector<std::string> cluster;
   for (int a = 0; a < 8; ++a)
      for (int b = 0; b < 8; ++b)
         for (int c = 0; c < 3; ++c)
            cluster.push_back(std::string("k") + char('a' + a) + char('a' + b) + char('a' + c));
   std::printf("\nbasic guarantee (no corruption) under fault during clustered bulk insert:\n");
   eh_bulk<RefMap>("std::map", "bulk-cluster", cluster);
   eh_bulk<HkvRadix>("artpp-radix", "bulk-cluster", cluster);
   eh_bulk<HkvBkt>("artpp-bucket", "bulk-cluster", cluster);
   eh_bulk<HkvAdp>("artpp-adapt", "bulk-cluster", cluster);
   eh_bulk<HkvDense>("artpp-dense", "bulk-cluster", cluster);
   eh_bulk<HkvWide>("artpp-wide", "bulk-cluster", cluster);

   // Dense fan-out under fault — many distinct bytes at one node drives the cnode widen ladder
   // (setlist→c2→c4→c8→full) and, for wide_stems, u16 fusion + restride back to u8.
   std::vector<std::string> wide;
   for (int a = 0; a < 24; ++a)
      for (int b = 0; b < 24; ++b)
         wide.push_back(std::string("p") + char('A' + a) + char('A' + b));
   std::printf("\nbasic guarantee under fault during dense fan-out (cnode ladder / u16 restride):\n");
   eh_bulk<RefMap>("std::map", "fanout", wide);
   eh_bulk<HkvDense>("artpp-dense", "fanout", wide);
   eh_bulk<HkvWide>("artpp-wide", "fanout", wide);

   // Throwing value constructors (audit #12). The radix modes are swept in full; the bucket
   // and adaptive insert paths have open exception-safety issues of their own (#4 to #7,
   // tests/known_bugs), so there only the paths #12 fixed are asserted.
   using TRef   = std::map<std::string, tvalue, std::less<>, throwing_alloc<std::pair<const std::string, tvalue>>>;
   using TRadix = map<std::string_view, tvalue, mode::none, throwing_alloc<tvalue>>;
   using TDense = map<std::string_view, tvalue, mode::dense_tiers, throwing_alloc<tvalue>>;
   using TWide  = map<std::string_view, tvalue, mode::wide_stems, throwing_alloc<tvalue>>;
   using TBkt   = map<std::string_view, tvalue, mode::buckets, throwing_alloc<tvalue>>;
   using TAdp   = map<std::string_view, tvalue, mode::adaptive, throwing_alloc<tvalue>>;
   std::printf("\nthrowing value constructors (strong, no leak):\n");
   run_value_scenarios<TRef>("std::map");
   run_value_scenarios<TRadix>("artpp-radix");
   run_value_scenarios<TDense>("artpp-dense");
   run_value_scenarios<TWide>("artpp-wide");
   for (int opk = 0; opk < 3; ++opk)
   {
      eh_value<TBkt>("artpp-bucket", "empty->x", {}, "x", opk);                    // make_bucket_branch
      eh_value<TBkt>("artpp-bucket", "long-key", {}, std::string(700, 'k'), opk);  // over a page: a leaf
      eh_value<TBkt>("artpp-bucket", "same-bucket", {"apple"}, "apricot", opk);    // bkt_insert_entry
      eh_value<TAdp>("artpp-adapt", "empty->x", {}, "x", opk);
      eh_value<TAdp>("artpp-adapt", "leaf-split", {"apple"}, "apricot", opk);
   }
   eh_value_repeat<TRadix>("artpp-radix", false);
   eh_value_repeat<TDense>("artpp-dense", false);
   eh_value_repeat<TBkt>("artpp-bucket", true);
   eh_value_repeat<TAdp>("artpp-adapt", false);

   // Inline (in-handle) values under allocation failure (audit #13).
   std::printf("\ninline uint32_t values (strong insert under bad_alloc injection):\n");
   run_inline_scenarios<map<std::string_view, uint32_t, mode::none, throwing_alloc<uint32_t>>>("artpp-radix");
   run_inline_scenarios<map<std::string_view, uint32_t, mode::dense_tiers, throwing_alloc<uint32_t>>>("artpp-dense");
   run_inline_scenarios<map<std::string_view, uint32_t, mode::wide_stems, throwing_alloc<uint32_t>>>("artpp-wide");

   std::printf(g_fail ? "\nartpp_eh: %d containers/scenarios FAILED\n" : "\nartpp_eh: ALL OK\n", g_fail);
   return g_fail ? 1 : 0;
}

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

   std::printf(g_fail ? "\nartpp_eh: %d containers/scenarios FAILED\n" : "\nartpp_eh: ALL OK\n", g_fail);
   return g_fail ? 1 : 0;
}

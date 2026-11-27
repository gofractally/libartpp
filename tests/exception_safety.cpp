// Exception-safety conformance via exhaustive fault injection. A throwing allocator fails
// on a countdown, and value types throw from their constructors on another; each sweep
// moves the countdown over every allocation (or construction) an operation makes and, on
// each induced throw, checks the guarantee the README states: the container is unchanged
// (single-key inserts are strong), erase and clear never throw, nothing leaks, no value is
// destroyed twice or not at all. std::map is the oracle, and the first battery runs the
// identical harness on std::map itself: a difference there is the harness's bug.
//
// Layout: targeted single-insert / bulk / value / clone / erase sweeps (all modes); then
// the paths the pre-release audit named in the bucket and adaptive modes (#4 to #7), one
// faulted operation at a time against the oracle; then random operation sequences with
// every fault point failed in turn, for every mode and for artpp::set.
#include "artpp/map.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
   long period = 0;      // > 0: after failing, fail again every `period` allocations
   // The countdown this allocator obeys: its own, or another counter's, so that two
   // unequal allocators (a cross-allocator move) fail on one shared schedule.
   fail_counter* sched = this;
   bool          due()
   {
      fail_counter& s = *sched;
      if (s.countdown < 0 || s.countdown-- != 0) return false;
      s.countdown = s.period > 0 ? s.period - 1 : -1;
      return true;
   }
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
      if (fc->due()) throw std::bad_alloc();
      ++fc->allocs;
      return static_cast<T*>(::operator new(n * sizeof(T), std::align_val_t(alignof(T))));
   }
   void deallocate(T* p, std::size_t n) noexcept
   {
      ++fc->frees;
      // Scribble over the block before releasing it: a tree that still reads a node it has
      // freed then sees garbage in a plain build too, not only under AddressSanitizer.
      std::memset(static_cast<void*>(p), 0xDD, n * sizeof(T));
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
static long g_vperiod    = 0;   // > 0: after throwing, throw again every g_vperiod constructions
struct value_error {};
static void value_tick()
{
   if (g_vcountdown < 0 || g_vcountdown-- != 0) return;
   g_vcountdown = g_vperiod > 0 ? g_vperiod - 1 : -1;
   throw value_error{};
}
struct tvalue
{
   std::string s;
   static void tick() { value_tick(); }
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

// The same value, 120 bytes wide: a bucket page holds three or four, so pages overflow and
// are repacked constantly, and two entries often cannot share one page at all.
struct tbig : tvalue
{
   char pad[96] = {};
   using tvalue::tvalue;
   tbig()                       = default;
   tbig(const tbig&)            = default;
   tbig(tbig&&)                 = default;
   tbig& operator=(const tbig&) = default;
   tbig& operator=(tbig&&)      = default;
};

// Move-only, and the move can throw: the one kind of type the tree cannot relocate safely
// (nothing to copy, and a move that fails half-way through a batch cannot be undone). The
// tree never relocates such values on erase, and promises only the basic guarantee where an
// insert has to move existing ones. A moved-from tmoveonly holds "".
struct tmoveonly
{
   std::string s;
   tmoveonly() { ++g_live; }
   tmoveonly(std::string v) : s(std::move(v)) { ++g_live; }
   tmoveonly(tmoveonly&& o)  // throws BEFORE touching the source
   {
      value_tick();
      s = std::move(o.s);
      o.s.clear();
      ++g_live;
   }
   tmoveonly& operator=(tmoveonly&& o) noexcept { s = std::move(o.s); return *this; }
   tmoveonly(const tmoveonly&)            = delete;
   tmoveonly& operator=(const tmoveonly&) = delete;
   ~tmoveonly() { --g_live; }
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

// ── cross-allocator move assignment (the structural clone) under fault ────────────
// Unequal, non-propagating allocators force operator=(map&&) to clone the source node by
// node into the destination's allocator. Fail each allocation of the clone in turn (or, for
// tvalue, each value move): the destination must come out valid, the source must stay
// destructible, and neither allocator may be left holding a block or a live value.
template <class Map>
static void eh_clone(const char* cont, const char* scen, const std::vector<std::string>& keys, bool by_value)
{
   int violations = 0, leaks = 0, vleaks = 0, swept = 0;
   for (int n = 0;; ++n)
   {
      fail_counter fa, fb;
      bool         completed = false;
      const long   vbase     = g_live;
      {
         Map src{typename Map::allocator_type(&fa)}, dst{typename Map::allocator_type(&fb)};
         for (size_t i = 0; i < keys.size(); ++i) src.insert_or_assign(keys[i], std::string(8, char('a' + (i & 15))));
         dst.insert_or_assign("previous-content", std::string("p"));
         if (by_value) g_vcountdown = n;
         else          fb.countdown = n;
         try
         {
            dst       = std::move(src);
            completed = true;
         }
         catch (const std::bad_alloc&) {}
         catch (const value_error&) {}
         g_vcountdown = -1;
         fb.countdown = -1;
         auto d = dump(dst);  // both trees must still walk cleanly
         auto s = dump(src);
         if (d.size() != dst.size() || s.size() != src.size()) ++violations;
         if (completed && (d.size() != keys.size() || !src.empty())) ++violations;
         if (!completed && !dst.empty()) ++violations;  // the half-built clone was reclaimed
         for (auto& kv : d)
            if (!dst.contains(kv.first)) ++violations;
      }
      if (fa.allocs != fa.frees || fb.allocs != fb.frees) ++leaks;
      if (g_live != vbase) ++vleaks;
      ++swept;
      if (completed) break;
   }
   const bool ok = (violations == 0 && leaks == 0 && vleaks == 0);
   if (!ok) ++g_fail;
   std::printf("  %-10s %-14s %-5s  %s  (%d points, %d corruption, %d mem-leak, %d dtor-leak)\n", cont, scen,
               by_value ? "value" : "alloc", ok ? "OK  " : "FAIL", swept, violations, leaks, vleaks);
}

// ── erase under allocation failure ───────────────────────────────────────────────
// erase shrinks the structure on the way out, and some of that allocates: re-homing a
// collapsed router's prefix in a prefix node, de-widening a sparse dense node to a setlist,
// repacking a bucket of non-trivial values. Those steps are best-effort — on bad_alloc the
// tree simply stays unshrunk — so erase must never throw, must always remove the key, and
// must leave every other key in place, whichever of its allocations fails.
template <class Map>
static void eh_erase(const char* cont, const char* scen, const std::vector<std::string>& keys,
                     size_t erase_every)
{
   int violations = 0, leaks = 0, vleaks = 0, swept = 0;
   for (int n = 0;; ++n)
   {
      fail_counter fc;
      bool         all_points = false;
      const long   vbase      = g_live;
      {
         typename Map::allocator_type       al(&fc);
         Map                                m(al);
         std::map<std::string, std::string> oracle;
         for (size_t i = 0; i < keys.size(); ++i)
         {
            const std::string v(8, char('a' + (i & 15)));
            m.insert_or_assign(keys[i], v);
            oracle[keys[i]] = v;
         }
         fc.countdown = n;  // the (n+1)-th allocation made by ANY of the erases below fails
         for (size_t i = 0; i < keys.size(); ++i)
         {
            if (i % erase_every == 0) continue;  // survivors
            try
            {
               if (m.erase(keys[i]) != oracle.erase(keys[i])) ++violations;
            }
            catch (...) { ++violations; }
         }
         all_points   = fc.countdown >= 0;  // never fired: every allocation point has been failed
         fc.countdown = -1;
         auto d = dump(m);
         if (d.size() != oracle.size() || m.size() != oracle.size()) ++violations;
         else
         {
            size_t i = 0;
            for (auto& kv : oracle)
            {
               if (d[i].first != kv.first || d[i].second != kv.second) ++violations;
               ++i;
            }
         }
      }
      if (fc.allocs != fc.frees) ++leaks;
      if (g_live != vbase) ++vleaks;
      ++swept;
      if (all_points) break;
   }
   const bool ok = (violations == 0 && leaks == 0 && vleaks == 0);
   if (!ok) ++g_fail;
   std::printf("  %-10s %-14s  %s  (%d points, %d violations, %d mem-leak, %d dtor-leak)\n", cont, scen,
               ok ? "OK  " : "FAIL", swept, violations, leaks, vleaks);
}

// ── bucket and adaptive modes: one faulted operation against a std::map oracle ────
// (audit #4 to #7 and the sweep that followed.) `build` fills the map and the oracle; `op`
// makes one mutating call on the map (or a few) and, only once a call has returned, the same
// change on the oracle. The sweep fails the n-th allocation of `op` (or throws from the n-th value
// construction) for every n until the fault no longer fires. After each run:
//   * an operation that threw left the contents exactly as they were (keys AND values);
//   * one that returned made its change, whatever best-effort allocation failed inside;
//   * iteration, size(), contains() and find() agree with the oracle;
//   * the map is then worked on some more (inserts, assignments, erases) and must still
//     agree, and after destruction every block and every value is accounted for.
// The harness's own values are made with the value fault disarmed: every armed construction
// is then one the MAP performs (the move into a leaf or bucket entry, a copy, a relocation).
struct value_fault_pause
{
   long saved = g_vcountdown;
   value_fault_pause() { g_vcountdown = -1; }
   ~value_fault_pause() { g_vcountdown = saved; }
};
template <class V>
static V mkv(uint64_t x)
{
   value_fault_pause p;
   if constexpr (std::is_arithmetic_v<V>) return V(x);
   else return V(std::string("value-long-enough-to-own-heap-memory-") + std::to_string(x));
}
[[maybe_unused]] static std::string vstr(const std::string& s) { return s; }
[[maybe_unused]] static std::string vstr(uint64_t v) { return std::to_string(v); }
[[maybe_unused]] static std::string vstr(uint32_t v) { return std::to_string(v); }
[[maybe_unused]] static std::string vstr(const tvalue& v) { return v.s; }
[[maybe_unused]] static std::string vstr(const tmoveonly& v) { return v.s; }
template <class V>
static std::string mks(uint64_t x) { return vstr(mkv<V>(x)); }

using oracle_t = std::map<std::string, std::string>;
// Does the map hold exactly `want`? Checked three ways: ordered iteration, size(), lookups.
template <class Map>
static bool same_as(const Map& m, const oracle_t& want)
{
   if (m.size() != want.size()) return false;
   auto w = want.begin();
   for (auto it = m.begin(); it != m.end(); ++it, ++w)
      if (w == want.end() || it.key_bytes() != w->first || vstr(it.value()) != w->second) return false;
   if (w != want.end()) return false;
   for (auto& kv : want)
   {
      auto it = m.find(kv.first);
      if (it == m.end() || vstr(it.value()) != kv.second || !m.contains(kv.first)) return false;
   }
   return true;
}
// The helpers below work on artpp::map and on artpp::set alike (for a set the oracle's values
// are empty), so the same sweeps drive both.
template <class C>
constexpr bool is_map_v = requires { typename C::mapped_type; };
static int g_sweep_viol = 0;  // checks made inside an operation (cross-allocator move)

struct fault_pause  // the harness's own bookkeeping must not trip the fault it has armed
{
   fail_counter& fc;
   long          a, v;
   explicit fault_pause(fail_counter& f) : fc(f), a(f.countdown), v(g_vcountdown) { fc.countdown = g_vcountdown = -1; }
   ~fault_pause() { fc.countdown = a; g_vcountdown = v; }
};
template <class C>
static std::string c_value(uint64_t x)
{
   if constexpr (is_map_v<C>) return mks<typename C::mapped_type>(x);
   else return std::string();
}
template <class C>
static void c_insert(C& m, oracle_t& o, const std::string& k, uint64_t x)  // insert-or-assign by move
{
   if constexpr (is_map_v<C>)
   {
      auto v = mkv<typename C::mapped_type>(x);
      m.insert(k, std::move(v));
   }
   else m.insert(k);
   o[k] = c_value<C>(x);
}
// `weak`: the type is move-only with a throwing move, so a value the tree had to move while
// an insert failed may be left moved-from (empty); its key must still be there.
template <class C>
static bool c_same(const C& m, const oracle_t& want, bool weak = false)
{
   if constexpr (is_map_v<C>)
   {
      if (!weak) return same_as(m, want);
      if (m.size() != want.size()) return false;
      auto w = want.begin();
      for (auto it = m.begin(); it != m.end(); ++it, ++w)
      {
         if (w == want.end() || it.key_bytes() != w->first) return false;
         const std::string got = vstr(it.value());
         if (!got.empty() && got != w->second) return false;
         if (!m.contains(w->first)) return false;
      }
      return w == want.end();
   }
   else
   {
      if (m.size() != want.size()) return false;
      auto w = want.begin();
      for (auto it = m.begin(); it != m.end(); ++it, ++w)
         if (w == want.end() || it.key_bytes() != w->first) return false;
      if (w != want.end()) return false;
      for (auto& kv : want)
         if (!m.contains(kv.first) || m.find(kv.first) == m.end()) return false;
      return true;
   }
}
template <class C>
static void c_restore_values(C& m, const oracle_t& o)  // after a failure that may leave values moved-from
{
   if constexpr (is_map_v<C>)
      for (auto& kv : o)
         if (auto it = m.find(kv.first); it != m.end() && vstr(it.value()) != kv.second)
         {
            using V = typename C::mapped_type;
            if constexpr (!std::is_arithmetic_v<V>) m.insert(kv.first, V(kv.second));
         }
}
// Keep using a container that has been through a fault: it must behave like any other.
template <class C>
static bool c_still_usable(C& m, oracle_t& o)
{
   std::vector<std::string> keys;
   for (auto& kv : o) keys.push_back(kv.first);
   uint64_t x = 5000;
   for (size_t i = 0; i < keys.size(); ++i)
   {
      if (i % 3 == 0) { m.erase(keys[i]); o.erase(keys[i]); }
      else if (i % 3 == 1) c_insert(m, o, keys[i], ++x);
      else c_insert(m, o, keys[i] + "+", ++x);
   }
   for (int i = 0; i < 24; ++i) c_insert(m, o, std::string("~after/") + char('a' + i % 6) + std::string(size_t(i), 'q'), ++x);
   if (!c_same(m, o)) return false;
   m.clear();
   o.clear();
   c_insert(m, o, "again", 1);
   return c_same(m, o);
}

enum class fault { alloc, value };
struct tally
{
   int points = 0, threw = 0, violations = 0, leaks = 0, vleaks = 0;
   void report(const char* cont, const char* scen, fault f) const
   {
      const bool ok = (violations == 0 && leaks == 0 && vleaks == 0);
      if (!ok) ++g_fail;
      std::printf("  %-18s %-24s %-5s  %s  (%d points, %d threw, %d violations, %d mem-leak, %d dtor-leak)\n",
                  cont, scen, f == fault::alloc ? "alloc" : "value", ok ? "OK  " : "FAIL", points, threw,
                  violations, leaks, vleaks);
   }
};
template <class Map, class Build, class Op>
static void eh_point(tally& t, fault f, Build build, Op op, bool may_throw = true)
{
   for (int n = 0;; ++n)
   {
      fail_counter fc;
      bool         fired = false;
      const long   vbase = g_live;
      {
         typename Map::allocator_type al(&fc);
         Map                          m(al);
         oracle_t                     o;
         build(m, o);
         if (!same_as(m, o)) ++t.violations;
         bool threw = false;
         (f == fault::alloc ? fc.countdown : g_vcountdown) = n;
         try { op(m, o); }
         catch (const std::bad_alloc&) { threw = true; }
         catch (const value_error&) { threw = true; }
         fired        = (f == fault::alloc ? fc.countdown : g_vcountdown) < 0;
         fc.countdown = g_vcountdown = -1;
         t.threw += threw;
         if (threw && !may_throw) ++t.violations;
         // The oracle holds exactly the steps of `op` that returned: a call that threw must
         // have changed nothing, a call that returned must have done all of its work.
         if (!same_as(m, o)) ++t.violations;
         if (!c_still_usable(m, o)) ++t.violations;
      }
      if (fc.allocs != fc.frees) ++t.leaks;
      if (g_live != vbase) ++t.vleaks;
      ++t.points;
      if (!fired) break;
   }
}
template <class Map, class Build, class Op>
static void eh_point(const char* cont, const char* scen, fault f, Build build, Op op, bool may_throw = true)
{
   tally t;
   eh_point<Map>(t, f, build, op, may_throw);
   t.report(cont, scen, f);
}

// The leading keys of `cand` that share one bucket page (how many depends on sizeof(T)):
// those that go in on a single allocation. `*next` is the key that then bursts the page.
template <class Map>
static std::vector<std::string> one_page(const std::vector<std::string>& cand, std::string* next)
{
   using V = typename Map::mapped_type;
   fail_counter fc;
   Map          m{typename Map::allocator_type(&fc)};
   size_t       c = 0;
   for (; c < cand.size(); ++c)
   {
      m.insert(cand[c], mkv<V>(0));
      if (fc.allocs > 1) break;
   }
   if (c == cand.size()) { ++g_fail; std::printf("  one_page: the candidates never filled a page\n"); c = cand.size() - 1; }
   *next = cand[c];
   return std::vector<std::string>(cand.begin(), cand.begin() + long(c));
}
template <class Map>
static auto put_keys(std::vector<std::string> keys, uint64_t v0)
{
   using V = typename Map::mapped_type;
   return [keys, v0](Map& m, oracle_t& o) {
      uint64_t x = v0;
      for (auto& k : keys)
      {
         V v = mkv<V>(x);
         m.insert(k, std::move(v));  // throws before the oracle learns of the key
         o[k] = mks<V>(x);
         ++x;
      }
   };
}

// audit #5: the insert that overflows a bucket page bursts it into a router; the router then
// existed only in a local while the new key's own bucket was allocated, so a failure there
// left the parent slot (or the root) pointing at the freed page.
template <class Map>
static void eh_bucket_burst(const char* cont, fault f = fault::alloc)
{
   const std::string        tail(40, 't');
   std::vector<std::string> cand, child;
   std::string              next;
   for (int c = 0; c < 26; ++c) cand.push_back(std::string(1, char('A' + c)) + tail);
   const std::vector<std::string> root = one_page<Map>(cand, &next);
   eh_point<Map>(cont, "burst root bucket", f, put_keys<Map>(root, 0), put_keys<Map>({next}, 99));
   // the same page one level down, reached through a router: the slot is a branch, not the root
   for (int c = 0; c < 30; ++c) child.push_back(std::string(1, char('A' + c)) + tail);
   for (size_t c = 0; c + 1 < root.size(); ++c) child.push_back(std::string("A") + char('a' + c) + tail);
   eh_point<Map>(cont, "burst child bucket", f, put_keys<Map>(child, 0), put_keys<Map>({"Az" + tail}, 99));
   // every part of the burst at once: a shared prefix (short: inline in the router; long: a
   // prefix node), a key that ends at the prefix (the router's term), and as many distinct
   // next bytes as a page holds (for small values more than a setlist: it widens mid-burst)
   for (const std::string& pfx : {std::string("shared/"), std::string(20, 'P')})
   {
      cand = {pfx};
      for (int c = 0; c < 60; ++c) cand.push_back(pfx + char('0' + c) + "x");
      const std::vector<std::string> page = one_page<Map>(cand, &next);
      eh_point<Map>(cont, pfx.size() < 9 ? "burst: term+prefix+fan" : "burst: term+long prefix", f,
                    put_keys<Map>(page, 0), put_keys<Map>({next}, 99));
   }
}

// A bucket directly below a prefix node (erase leaves that shape when a router is down to
// one branch) that bursts into a router too wide for a setlist: the prefix then belongs in
// the new node_full's header, and insert fuses it there and re-descends from the prefix
// node's slot. The fusion frees the prefix node the descent was standing in.
template <class Map>
static void eh_burst_under_prefix(const char* cont)
{
   using V = typename Map::mapped_type;
   const std::string        tail(40, 't');
   std::vector<std::string> first, small;
   for (int c = 0; c < 12; ++c) first.push_back(std::string("xyz") + char('A' + c) + tail);
   for (int c = 0; c < 40; ++c) small.push_back(std::string("xyzA") + char('0' + c));
   auto build = [&](Map& m, oracle_t& o) {
      put_keys<Map>(first, 0)(m, o);  // bursts the root: router "xyz" over one bucket per letter
      for (size_t i = 1; i < first.size(); ++i) { m.erase(first[i]); o.erase(first[i]); }  // → prefix "xyzA" over a bucket
   };
   {
      fail_counter fc;
      Map          m{typename Map::allocator_type(&fc)};
      oracle_t     o;
      build(m, o);
      const auto before = m.debug_stats();
      put_keys<Map>(small, 100)(m, o);
      const auto after = m.debug_stats();
      // before: the prefix node is there; after: it is fused into the node_full's header
      const bool ok = before.prefix == 1 && before.setlist + before.full + before.fullp == 0 && after.prefix == 0 &&
                      after.fullp == 1 && same_as(m, o);
      if (!ok) ++g_fail;
      std::printf("  %-18s %-24s        %s  (prefix nodes %llu -> %llu, fused fulls %llu -> %llu)\n", cont,
                  "burst under prefix: shape", ok ? "OK  " : "FAIL", (unsigned long long)before.prefix,
                  (unsigned long long)after.prefix, (unsigned long long)before.fullp, (unsigned long long)after.fullp);
      (void)sizeof(V);
   }
   eh_point<Map>(cont, "burst under prefix node", fault::alloc, build, put_keys<Map>(small, 100));
}

// audit #4: a router that is full takes one more child by widening, in one step or (dense
// tiers) several. bkt_put widened into a local: a setlist's inline prefix was cleared first
// and re-attached last, so a failed widen dropped the prefix of every key below; and once
// the first widen had freed the old node, any later failure (the next rung, or the prefix
// node) left the parent pointing at freed memory. Every fan-out from 1 to 70 children under
// the shared prefix "pre" is grown by one child here, which crosses each rung of the ladder.
template <class Map>
static void eh_bucket_widen(const char* cont)
{
   const std::string tail(40, 't');
   for (int spread = 0; spread < 2; ++spread)
   {
      auto edge = [spread](int c) { return char(spread ? uint8_t(c * 67 + 5) : uint8_t('0' + c)); };
      tally t;
      for (int n = 1; n <= 70; ++n)
      {
         std::vector<std::string> pre;
         for (int c = 0; c < n; ++c)
            for (int j = 0; j < 2; ++j) pre.push_back(std::string("pre") + edge(c) + char('0' + j) + tail);
         eh_point<Map>(t, fault::alloc, put_keys<Map>(pre, 0),
                       put_keys<Map>({std::string("pre") + edge(n) + tail}, 999));
      }
      t.report(cont, spread ? "widen, spread edges" : "widen, adjacent edges", fault::alloc);
   }
}

// audit #6: removing an entry from a bucket of non-trivial values destroyed the value and
// THEN allocated the page the survivors are repacked into. When that allocation failed the
// destroyed value was still indexed: the next read, or the destructor, used a dead object.
// erase must not throw at all, must remove the key and must keep every other key and value,
// whichever allocation fails and (for a T whose copy can throw) whichever copy throws.
template <class Map>
static void eh_bucket_erase(const char* cont, fault f)
{
   const std::vector<std::vector<std::string>> sets = {
       {"alpha", "beta", "gamma", "delta", "epsilon"},                                   // one page
       {"a", "ab", "abc", "abcd", "abcde", "abcdef", "abcdx", "abcdy"},                 // deep and narrow
       {"k/one", "k/two", "k/three", "k/" + std::string(70, 'w'), "k/four", "k/five"},  // mixed entry sizes
   };
   tally one, all;
   for (auto& keys : sets)
   {
      for (auto& victim : keys)
         eh_point<Map>(one, f, put_keys<Map>(keys, 0),
                       [&](Map& m, oracle_t& o) { if (m.erase(victim) != o.erase(victim)) throw std::bad_alloc(); },
                       false);
      for (int dir = 0; dir < 2; ++dir)  // empty the map key by key, forwards and backwards
         eh_point<Map>(all, f, put_keys<Map>(keys, 0),
                       [&](Map& m, oracle_t& o) {
                          for (size_t i = 0; i < keys.size(); ++i)
                          {
                             const std::string& k = keys[dir ? keys.size() - 1 - i : i];
                             if (m.erase(k) != o.erase(k)) throw std::bad_alloc();
                          }
                       },
                       false);
   }
   one.report(cont, "erase one key", f);
   all.report(cont, "erase every key", f);
}

// audit #7: mode::adaptive turns a leaf at a deep, narrow split point into a bucket holding
// the old entry and the new key. The old leaf was emptied and freed before the new value was
// constructed, so a throwing constructor (emplace arguments, a copy, a move) left the slot
// pointing at the freed leaf and leaked the new page.
template <class Map>
static auto emplace_key(std::string k)
{
   using V = typename Map::mapped_type;
   return [k](Map& m, oracle_t& o) {
      if constexpr (std::is_same_v<V, tvalue>)
      {
         if (m.emplace(k, "E", 7)) o[k] = "E7";  // constructed in place from the arguments
      }
      else
      {
         V v = mkv<V>(7);
         if (m.emplace(k, std::move(v))) o[k] = mks<V>(7);
      }
   };
}
template <class Map>
static void eh_adaptive_collapse(const char* cont, fault f)
{
   using V = typename Map::mapped_type;
   const std::vector<std::pair<std::vector<std::string>, std::string>> shapes = {
       {{"a", "ab", "abc"}, "abcd"},                                  // the new key extends the leaf's
       {{"a", "ab", "abc" + std::string(30, 'x')}, "abcd"},           // old and new diverge, long old suffix
       {{"a", "ab", "abcd"}, "abc"},                                  // the new key ends where the leaf starts
       {{"q", "qr", "qrs", "qrst", "qrstu"}, "qrstu" + std::string(60, 'v')},
   };
   tally t;
   for (auto& [pre, nk] : shapes)
   {
      eh_point<Map>(t, f, put_keys<Map>(pre, 0), emplace_key<Map>(nk));
      eh_point<Map>(t, f, put_keys<Map>(pre, 0), put_keys<Map>({nk}, 7));  // insert(k, T&&)
      eh_point<Map>(t, f, put_keys<Map>(pre, 0), [&](Map& m, oracle_t& o) {
         const V v = mkv<V>(8);
         m.insert_or_assign(nk, v);  // copies
         o[nk] = mks<V>(8);
      });
   }
   t.report(cont, "leaf -> bucket collapse", f);
}

// ── bucket and adaptive modes: random operation sequences under systematic faults ─────
// The scenarios above each aim at one audited path. This sweep does not know the paths: it
// runs a fixed pseudo-random sequence of insert / insert_or_assign / emplace / operator[] /
// erase(key) / erase(iterator) / clear / cross-allocator move over clustered keys (shared
// prefixes, nested keys, keys longer than a bucket page, 0x00 and 0xFF bytes) and
//   * single faults: repeats the whole sequence once per fault point, failing the n-th
//     allocation (or throwing from the n-th value construction the map performs) for every
//     n until a run completes without the fault firing — every point of the sequence;
//   * periodic faults: one run in which every p-th allocation / construction fails, so
//     operations fail back to back on a tree that has already been through failures.
// After every operation that throws: the contents equal the std::map oracle, which only
// learns of a change once the map's call has returned (the strong guarantee); erase and
// clear never throw. After every run: contents equal the oracle, the map survives further
// use (still_usable), and every block and every value is accounted for.
// The same code drives artpp::set (the oracle's values are then empty).
struct sweep_op
{
   int         kind;  // 0 insert(T&&)  1 emplace  2 erase(key)  3 insert_or_assign(const T&)
   std::string key;   // 4 operator[]   5 erase(iterator)  6 clear  7 move to another allocator and back
   uint64_t    v;
};
// Even seeds cluster the keys under six prefixes (deep, narrow trees); odd seeds put most of
// them under one prefix or none, with up to 64 different next bytes, so routers fill up and
// widen (with and without a prefix to carry).
static std::vector<sweep_op> gen_ops(uint64_t seed, int nops)
{
   const bool fan = seed % 2 == 1;
   uint64_t x    = seed * 0x9E3779B97F4A7C15ull + 0x1234567;
   auto     rnd  = [&x]() { x ^= x << 13; x ^= x >> 7; x ^= x << 17; return x; };  // xorshift: same everywhere
   static const char* const pfx[] = {"", "pre", "prefixlong_", "p", "prefixlong_and_longer_still_", "pre\xff"};
   std::vector<sweep_op>    ops;
   std::vector<std::string> seen;
   for (int i = 0; i < nops; ++i)
   {
      sweep_op o;
      o.v         = rnd() % 100000;
      const int r = int(rnd() % 40);
      if (r == 0) o.kind = 6;
      else if (r == 1 || r == 2) o.kind = 7;
      else if (r < 11 && !seen.empty())
      {
         o.kind = (r < 8) ? 2 : 5;
         o.key  = seen[rnd() % seen.size()];
      }
      else
      {
         o.kind = (r < 22) ? 0 : (r < 28) ? 1 : (r < 35) ? 3 : 4;
         const int shape = int(rnd() % 16);
         if (shape < 3 && !seen.empty())  // a neighbour of an existing key: its prefix, or an extension
         {
            o.key = seen[rnd() % seen.size()];
            if (rnd() % 2 && !o.key.empty()) o.key.pop_back();
            else o.key.push_back(char('a' + rnd() % 3));
         }
         else if (shape == 3)  // longer than a bucket page: stored as a plain leaf in bucket mode
            o.key = std::string(pfx[rnd() % 6]) + std::string(520 + rnd() % 40, 'L') + char('a' + rnd() % 4);
         else if (shape == 4)  // a nested chain
            o.key = std::string(pfx[rnd() % 6]) + std::string(1 + rnd() % 40, 'n');
         else if (fan)
         {
            const int w = int(rnd() % 8);
            o.key       = w < 5 ? "pre" : w < 7 ? "" : "prefixlong_and_longer_still_";
            o.key.push_back(char('0' + rnd() % 64));
            const int tl = int(rnd() % 4);
            for (int j = 0; j < tl; ++j) o.key.push_back(char('a' + rnd() % 2));
         }
         else
         {
            o.key = pfx[rnd() % 6];
            o.key.push_back(char(shape == 5 ? rnd() % 256 : 'A' + rnd() % 40));
            const int tl = (rnd() % 4 == 0) ? int(rnd() % 60) : int(rnd() % 6);
            for (int j = 0; j < tl; ++j) o.key.push_back(char('a' + rnd() % 2));
         }
         seen.push_back(o.key);
      }
      ops.push_back(std::move(o));
   }
   return ops;
}
// One operation on the container, then (only if it returned) the same change on the oracle.
template <class C>
static void c_apply(C& m, oracle_t& o, const sweep_op& op, fail_counter& fc, fail_counter& side)
{
   const std::string& k = op.key;
   switch (op.kind)
   {
      case 0: c_insert(m, o, k, op.v); break;
      case 1:
         if constexpr (is_map_v<C>)
         {
            using V = typename C::mapped_type;
            if constexpr (std::is_same_v<V, tvalue>)
            {
               if (m.emplace(k, "E", int(op.v))) o[k] = "E" + std::to_string(op.v);  // built in place
            }
            else
            {
               V v = mkv<V>(op.v);
               if (m.emplace(k, std::move(v))) o[k] = mks<V>(op.v);
            }
         }
         else if (m.emplace(k).second) o[k];
         break;
      case 2:
         if (m.erase(k) != o.erase(k)) ++g_sweep_viol;
         break;
      case 3:
         if constexpr (is_map_v<C>)
         {
            using V = typename C::mapped_type;
            if constexpr (std::is_copy_constructible_v<V>)
            {
               const V v = mkv<V>(op.v);
               m.insert_or_assign(k, v);  // copies inside the map
               o[k] = mks<V>(op.v);
            }
            else c_insert(m, o, k, op.v);
         }
         else c_insert(m, o, k, op.v);
         break;
      case 4:
         if constexpr (requires { m[k]; })  // not offered for a value stored in the handle
         {
            using V = typename C::mapped_type;
            V  v   = mkv<V>(op.v);
            V& ref = m[k];  // may insert a default-constructed value (and throw doing so) …
            ref    = std::move(v);  // … the assignment itself cannot throw
            o[k]   = mks<V>(op.v);
         }
         else c_insert(m, o, k, op.v);
         break;
      case 5:
         if (auto it = m.find(k); it != m.end())
         {
            m.erase(it);
            if (o.erase(k) != 1) ++g_sweep_viol;
         }
         else if (o.count(k)) ++g_sweep_viol;
         break;
      case 6:
         m.clear();
         o.clear();
         break;
      case 7:
      {
         // Unequal allocators: each assignment clones node by node into the target's
         // allocator. A clone that fails must leave the target empty and the source with
         // all its keys (its values may have been moved from: restored below so the
         // sequence can go on), and must leak nothing — then the failure is reported.
         C other{typename C::allocator_type(&side)};
         try { other = std::move(m); }
         catch (...)
         {
            fault_pause p(fc);
            if (!other.empty() || !c_same(m, o, true)) ++g_sweep_viol;
            c_restore_values(m, o);
            throw;
         }
         if (!m.empty()) ++g_sweep_viol;
         try { m = std::move(other); }
         catch (...)
         {
            fault_pause p(fc);
            if (!m.empty() || !c_same(other, o, true)) ++g_sweep_viol;
            c_restore_values(other, o);
            m = std::move(other);  // faults paused: the trip back completes
            throw;
         }
         if (!other.empty()) ++g_sweep_viol;
         break;
      }
   }
}
// period == 0: sweep every fault point of the sequence. period > 0: one run, every period-th.
template <class C>
static void eh_sweep(tally& t, uint64_t seed, int nops, fault f, int period, bool weak = false)
{
   const std::vector<sweep_op> ops = gen_ops(seed, nops);
   for (int n = 0;; ++n)
   {
      fail_counter fc, side;
      side.sched       = &fc;
      bool       fired = false;
      const long vbase = g_live;
      g_sweep_viol     = 0;
      {
         C        m{typename C::allocator_type(&fc)};
         oracle_t o;
         long&    cd = (f == fault::alloc ? fc.countdown : g_vcountdown);
         (f == fault::alloc ? fc.period : g_vperiod) = period;
         cd = period ? period - 1 : n;
         for (size_t i = 0; i < ops.size(); ++i)
         {
            const long armed = cd;
            bool       threw = false;
            try { c_apply(m, o, ops[i], fc, side); }
            catch (const std::bad_alloc&) { threw = true; }
            catch (const value_error&) { threw = true; }
            if (threw || (!period && armed >= 0 && cd < 0) || (period && i % 16 == 15))
            {
               // the fault fired in this operation (or, periodic, some time recently)
               fault_pause p(fc);
               fired = fired || threw || cd < 0;
               t.threw += threw;
               const int kind = ops[i].kind;
               if (threw && (kind == 2 || kind == 5 || kind == 6)) ++t.violations;  // erase/clear threw
               if (!c_same(m, o, weak)) { ++t.violations; break; }
               if (weak && threw) c_restore_values(m, o);
            }
         }
         fired        = fired || cd < 0;
         fc.countdown = g_vcountdown = -1;
         fc.period = g_vperiod = 0;
         if (!c_same(m, o, weak)) ++t.violations;
         if (weak) c_restore_values(m, o);
         if (!c_still_usable(m, o)) ++t.violations;
         t.violations += g_sweep_viol;
      }
      if (fc.allocs != fc.frees || side.allocs != side.frees) ++t.leaks;
      if (g_live != vbase) ++t.vleaks;
      ++t.points;
      if (period || !fired) break;
   }
}
// ctest runs 2 sequences of 120 operations per container with every fault point failed, and
// 15 periodic runs of twice that length. For a longer soak set ARTPP_EH_SEEDS (sequences)
// and ARTPP_EH_OPS (operations each); the cost grows with seeds x ops^2.
static int env_int(const char* name, int dflt)
{
   const char* v = std::getenv(name);
   return v && *v ? std::atoi(v) : dflt;
}
template <class C>
static void run_sweep(const char* cont, fault f, bool weak = false)
{
   static const int seeds = env_int("ARTPP_EH_SEEDS", 2), nops = env_int("ARTPP_EH_OPS", 120);
   tally single, periodic;
   for (int seed = 1; seed <= seeds; ++seed) eh_sweep<C>(single, uint64_t(seed), nops, f, 0, weak);
   for (int seed = 1; seed <= seeds + 1; ++seed)
      for (int period : {2, 3, 5, 7, 13}) eh_sweep<C>(periodic, uint64_t(1000 + seed), 2 * nops, f, period, weak);
   single.report(cont, "sequence, each point", f);
   periodic.report(cont, "sequence, every p-th", f);
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
   std::setvbuf(stdout, nullptr, _IOLBF, 0);  // a crash must not swallow the lines before it
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

   // Throwing value constructors (audit #12), every mode. (When this was written the bucket
   // and adaptive modes could only be given the three paths #12 itself fixed; those are kept
   // below, and since #4 to #7 the full battery runs there as well.)
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
   run_value_scenarios<TBkt>("artpp-bucket");
   run_value_scenarios<TAdp>("artpp-adapt");
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

   // erase with each of its (best-effort) allocations failing, every mode.
   std::printf("\nerase under bad_alloc injection (never throws, always removes):\n");
   {
      std::vector<std::string> shrink = cluster;  // collapses, prefix re-homing, de-widening
      shrink.insert(shrink.end(), wide.begin(), wide.end());
      for (int i = 0; i < 20; ++i) shrink.push_back("0123456789ABCDEF" + std::string(1, char('a' + i)) + "t");
      shrink.push_back("0123456789ABCDEF");
      for (int i = 1; i <= 90; ++i) shrink.push_back("z" + std::string(size_t(i), 'd'));  // a deep chain
      for (size_t every : {2u, 5u, 1000000u})
      {
         eh_erase<HkvRadix>("artpp-radix", every == 2 ? "erase-half" : every == 5 ? "erase-most" : "erase-all", shrink, every);
         eh_erase<HkvWide>("artpp-wide", every == 2 ? "erase-half" : every == 5 ? "erase-most" : "erase-all", shrink, every);
         eh_erase<map<std::string_view, counted, mode::dense_tiers, throwing_alloc<counted>>>(
             "artpp-dense", every == 2 ? "erase-half" : every == 5 ? "erase-most" : "erase-all", shrink, every);
         eh_erase<HkvBkt>("artpp-bucket", every == 2 ? "erase-half" : every == 5 ? "erase-most" : "erase-all", shrink, every);
         eh_erase<HkvAdp>("artpp-adapt", every == 2 ? "erase-half" : every == 5 ? "erase-most" : "erase-all", shrink, every);
      }
   }

   // Cross-allocator move assignment: the node-by-node clone, every allocation / value move failed.
   std::printf("\nmove assignment across unequal allocators (structural clone) under fault:\n");
   {
      std::vector<std::string> deep;  // a 300-level chain with a term at every level
      for (int i = 1; i <= 300; ++i) deep.push_back(std::string(size_t(i), 'd'));
      std::vector<std::string> mixed = cluster;
      mixed.insert(mixed.end(), wide.begin(), wide.begin() + 120);
      mixed.push_back("");
      mixed.push_back(std::string(700, 'L'));
      for (int i = 0; i < 20; ++i) mixed.push_back("0123456789ABCDEF" + std::string(1, char('a' + i)) + "t");
      eh_clone<TRadix>("artpp-radix", "mixed", mixed, false);
      eh_clone<TRadix>("artpp-radix", "mixed", mixed, true);
      eh_clone<TRadix>("artpp-radix", "deep-chain", deep, false);
      eh_clone<TDense>("artpp-dense", "mixed", mixed, false);
      eh_clone<TDense>("artpp-dense", "mixed", mixed, true);
      eh_clone<TWide>("artpp-wide", "mixed", mixed, false);
      eh_clone<TBkt>("artpp-bucket", "mixed", mixed, false);
      eh_clone<TBkt>("artpp-bucket", "mixed", mixed, true);
      eh_clone<TAdp>("artpp-adapt", "mixed", mixed, false);
   }

   // Bucket and adaptive modes: the audited fault paths (#4 to #7), one operation at a time.
   using BU64 = map<std::string_view, uint64_t, mode::buckets, throwing_alloc<uint64_t>>;
   using BU32 = map<std::string_view, uint32_t, mode::buckets, throwing_alloc<uint32_t>>;
   using BStr = map<std::string_view, std::string, mode::buckets, throwing_alloc<std::string>>;
   using BTv  = map<std::string_view, tvalue, mode::buckets, throwing_alloc<tvalue>>;
   using AStr = map<std::string_view, std::string, mode::adaptive, throwing_alloc<std::string>>;
   using ATv  = map<std::string_view, tvalue, mode::adaptive, throwing_alloc<tvalue>>;
   using AU64 = map<std::string_view, uint64_t, mode::adaptive, throwing_alloc<uint64_t>>;
   using AU32 = map<std::string_view, uint32_t, mode::adaptive, throwing_alloc<uint32_t>>;
   using BDenseU64  = map<std::string_view, uint64_t, mode::buckets | mode::dense_tiers, throwing_alloc<uint64_t>>;
   using BDense8U64 = map<std::string_view, uint64_t, mode::buckets | mode::dense_tiers | mode::ladder_c8,
                          throwing_alloc<uint64_t>>;
   std::printf("\nbucket and adaptive modes, one faulted operation against a std::map oracle:\n");
   eh_bucket_burst<BU64>("bucket/u64");
   eh_bucket_burst<BU32>("bucket/u32");
   eh_bucket_burst<BStr>("bucket/string");
   eh_bucket_burst<BTv>("bucket/tvalue");
   eh_bucket_burst<BTv>("bucket/tvalue", fault::value);
   eh_bucket_burst<BDenseU64>("bucket+dense/u64");
   eh_burst_under_prefix<BU64>("bucket/u64");
   eh_burst_under_prefix<BU32>("bucket/u32");
   eh_bucket_widen<BU64>("bucket/u64");
   eh_bucket_widen<BDenseU64>("bucket+dense/u64");
   eh_bucket_widen<BDense8U64>("bucket+dense+c8/u64");
   eh_bucket_erase<BStr>("bucket/string", fault::alloc);
   eh_bucket_erase<BTv>("bucket/tvalue", fault::alloc);
   eh_bucket_erase<BTv>("bucket/tvalue", fault::value);
   eh_bucket_erase<AStr>("adaptive/string", fault::alloc);
   eh_bucket_erase<ATv>("adaptive/tvalue", fault::value);
   eh_adaptive_collapse<ATv>("adaptive/tvalue", fault::value);
   eh_adaptive_collapse<ATv>("adaptive/tvalue", fault::alloc);
   eh_adaptive_collapse<AStr>("adaptive/string", fault::alloc);
   eh_adaptive_collapse<AU64>("adaptive/u64", fault::alloc);
   eh_adaptive_collapse<AU32>("adaptive/u32", fault::alloc);

   // Bucket and adaptive modes, every flag combination that changes their node kinds, and
   // artpp::set: random operation sequences with every fault point failed in turn.
   std::printf("\nbucket and adaptive modes, operation sequences against a std::map oracle:\n");
   {
      using mode_bd  = decltype(mode::buckets);
      constexpr mode_bd B = mode::buckets, A = mode::adaptive, D = mode::dense_tiers, W = mode::wide_stems,
                        C8 = mode::ladder_c8;
      using sv = std::string_view;
      using TMo = tmoveonly;
      run_sweep<map<sv, uint64_t, B, throwing_alloc<uint64_t>>>("bucket/u64", fault::alloc);
      run_sweep<map<sv, uint32_t, B, throwing_alloc<uint32_t>>>("bucket/u32", fault::alloc);
      run_sweep<map<sv, std::string, B, throwing_alloc<std::string>>>("bucket/string", fault::alloc);
      run_sweep<map<sv, tvalue, B, throwing_alloc<tvalue>>>("bucket/tvalue", fault::value);
      run_sweep<map<sv, TMo, B, throwing_alloc<TMo>>>("bucket/move-only", fault::alloc, true);
      run_sweep<map<sv, TMo, B, throwing_alloc<TMo>>>("bucket/move-only", fault::value, true);
      run_sweep<map<sv, uint64_t, A, throwing_alloc<uint64_t>>>("adaptive/u64", fault::alloc);
      run_sweep<map<sv, uint32_t, A, throwing_alloc<uint32_t>>>("adaptive/u32", fault::alloc);
      run_sweep<map<sv, std::string, A, throwing_alloc<std::string>>>("adaptive/string", fault::alloc);
      run_sweep<map<sv, tvalue, A, throwing_alloc<tvalue>>>("adaptive/tvalue", fault::value);
      run_sweep<map<sv, TMo, A, throwing_alloc<TMo>>>("adaptive/move-only", fault::alloc, true);
      run_sweep<map<sv, TMo, A, throwing_alloc<TMo>>>("adaptive/move-only", fault::value, true);
      run_sweep<map<sv, tbig, B, throwing_alloc<tbig>>>("bucket/120-byte", fault::value);
      run_sweep<map<sv, tbig, A, throwing_alloc<tbig>>>("adaptive/120-byte", fault::alloc);
      run_sweep<map<sv, uint64_t, B | D, throwing_alloc<uint64_t>>>("bucket+dense/u64", fault::alloc);
      run_sweep<map<sv, std::string, B | D, throwing_alloc<std::string>>>("bucket+dense/string", fault::alloc);
      run_sweep<map<sv, tvalue, B | D | C8, throwing_alloc<tvalue>>>("bucket+dense+c8/tvalue", fault::value);
      run_sweep<map<sv, uint64_t, A | D | C8, throwing_alloc<uint64_t>>>("adapt+dense+c8/u64", fault::alloc);
      run_sweep<map<sv, std::string, A | D, throwing_alloc<std::string>>>("adapt+dense/string", fault::alloc);
      run_sweep<map<sv, tvalue, A | D, throwing_alloc<tvalue>>>("adapt+dense/tvalue", fault::value);
      run_sweep<map<sv, uint64_t, B | W, throwing_alloc<uint64_t>>>("bucket+wide/u64", fault::alloc);
      run_sweep<map<sv, std::string, A | W, throwing_alloc<std::string>>>("adapt+wide/string", fault::alloc);
      // the radix modes, for comparison under the same sequences
      run_sweep<map<sv, uint64_t, mode::none, throwing_alloc<uint64_t>>>("radix/u64", fault::alloc);
      run_sweep<map<sv, tvalue, mode::none, throwing_alloc<tvalue>>>("radix/tvalue", fault::value);
      run_sweep<map<sv, std::string, D, throwing_alloc<std::string>>>("dense/string", fault::alloc);
      run_sweep<map<sv, uint64_t, W, throwing_alloc<uint64_t>>>("wide/u64", fault::alloc);
      // strict even for the move-only type: the radix modes move one existing value per
      // insert at most, last, so a throwing move leaves everything as it was
      run_sweep<map<sv, TMo, mode::none, throwing_alloc<TMo>>>("radix/move-only", fault::value);
      run_sweep<map<sv, TMo, W, throwing_alloc<TMo>>>("wide/move-only", fault::value);
      run_sweep<artpp::set<sv, mode::none, throwing_alloc<sv>>>("set", fault::alloc);
      run_sweep<artpp::set<sv, B, throwing_alloc<sv>>>("set bucket", fault::alloc);
      run_sweep<artpp::set<sv, A, throwing_alloc<sv>>>("set adaptive", fault::alloc);
   }

   // Inline (in-handle) values under allocation failure (audit #13).
   std::printf("\ninline uint32_t values (strong insert under bad_alloc injection):\n");
   run_inline_scenarios<map<std::string_view, uint32_t, mode::none, throwing_alloc<uint32_t>>>("artpp-radix");
   run_inline_scenarios<map<std::string_view, uint32_t, mode::dense_tiers, throwing_alloc<uint32_t>>>("artpp-dense");
   run_inline_scenarios<map<std::string_view, uint32_t, mode::wide_stems, throwing_alloc<uint32_t>>>("artpp-wide");

   std::printf(g_fail ? "\nartpp_eh: %d containers/scenarios FAILED\n" : "\nartpp_eh: ALL OK\n", g_fail);
   return g_fail ? 1 : 0;
}

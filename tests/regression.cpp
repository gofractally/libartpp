// artpp_regression.cpp — deterministic guards for bugs the differential fuzzer caught.
//
// Each block reproduces a specific defect's triggering shape and cross-checks the result
// against a std::map oracle (full forward + reverse iteration, for_each_value multiset,
// then remove-all + empty-iteration). The shapes are chosen so a regression manifests as
// a wrong size/order/value or a crash here — not only under a sanitizer. Run under ASan
// too (the fuzzer build does) to also catch the use-after-free / OOB forms.
//
// Provenance (the four planted bugs, see commit message / docs):
//   #1 bkt_put descended into an inline-leaf setlist child without externalizing it.
//   #2 the adaptive deep-narrow collapse pre-checked each suffix against a page ALONE.
//   #3 for_each_value (walk_v_) had no setlist_u16 case → wide routers mis-walked.
//   #4 bkt_split left a long common prefix unwrapped → a per-byte single-branch chain.
#include "artpp/map.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <set>
#include <string>
#include <vector>

#include <pthread.h>

using artpp::map;
using artpp::mode;

static int g_fail = 0;
#define CHECK(cond)                                                                          \
   do {                                                                                      \
      if (!(cond)) { std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); ++g_fail; } \
   } while (0)

// Full agreement between an artpp tree and its oracle: count, forward order+values,
// reverse order, and the order-independent value multiset via for_each_value (walk_v_).
template <class Tree, class V>
static void agree(const Tree& t, const std::map<std::string, V>& oracle, const char* where)
{
   CHECK(t.size() == oracle.size());
   {  // forward
      auto it = t.begin();
      auto oi = oracle.begin();
      for (; oi != oracle.end(); ++it, ++oi)
      {
         if (it == t.end()) { std::printf("FAIL %s: forward ended early\n", where); ++g_fail; return; }
         CHECK(std::string(it->first) == oi->first);
         CHECK(it->second == oi->second);
      }
      CHECK(it == t.end());
   }
   {  // reverse via --end()
      auto it = t.end();
      auto oi = oracle.rbegin();
      for (; oi != oracle.rend(); ++oi)
      {
         --it;
         CHECK(std::string(it->first) == oi->first);
         CHECK(it->second == oi->second);
      }
      CHECK(it == t.begin());
   }
   {  // for_each_value multiset (this is the walk_v_ path — bug #3)
      std::multiset<V> got, exp;
      t.for_each_value([&](const V& v) { got.insert(v); });
      for (auto& kv : oracle) exp.insert(kv.second);
      CHECK(got == exp);
   }
}

// Insert `keys`, agree, then remove every key (in the given order) and confirm the tree
// empties cleanly and iterates as empty — the remove-path teardown is where a corrupt
// structure (e.g. a chain past the remove window) crashes or leaks.
template <class Tree, class V>
static void roundtrip(const std::vector<std::string>& keys, bool reverse_erase, const char* where)
{
   Tree                          t;
   std::map<std::string, V>      oracle;
   V                             v = 1;
   for (auto& k : keys) { t.insert_or_assign(k, v); oracle[k] = v; v = V(v + 1); }
   agree<Tree, V>(t, oracle, where);

   if (!reverse_erase)
      for (auto& k : keys) { CHECK(t.erase(k) == oracle.erase(k)); }
   else
      for (auto it = keys.rbegin(); it != keys.rend(); ++it) { CHECK(t.erase(*it) == oracle.erase(*it)); }
   CHECK(t.size() == 0);
   CHECK(t.begin() == t.end());
}

// ── #4: bkt_split must wrap a long common prefix in ONE node ─────────────────────
// Two+ keys sharing a long prefix overflow a bucket → bkt_split. If the shared run is
// left unwrapped, each byte spawns a single-branch setlist; a prefix longer than the
// remove window then cannot be unwound and iteration after erase crashes. Sweep prefix
// lengths across the window boundary (RM_LVN=64) and well beyond.
static void regression_bkt_split_long_prefix()
{
   for (size_t plen : {1u, 7u, 8u, 63u, 64u, 65u, 200u, 320u, 1000u, 5000u})
   {
      std::vector<std::string> keys;
      std::string              base(plen, 'b');
      for (int i = 0; i < 40; ++i)
         keys.push_back(base + std::string(1 + (i % 9), char('a' + i % 17)) + char('0' + i));
      keys.push_back(base);                            // a key that IS the shared prefix (term)
      keys.push_back(base + std::string(2000, 'z'));   // a very long suffix off the same prefix
      roundtrip<map<std::string_view, uint64_t, mode::buckets>, uint64_t>(keys, false, "bkt_split fwd");
      roundtrip<map<std::string_view, uint64_t, mode::buckets>, uint64_t>(keys, true, "bkt_split rev");
      roundtrip<map<std::string_view, uint64_t, mode::adaptive>, uint64_t>(keys, false, "bkt_split adaptive");
   }
}

// ── #2: adaptive collapse must check BOTH suffixes against ONE page together ─────
// A deep-narrow leaf collision (lowfan>=2: a descent through >=2 routers with <3 branches,
// ending at a leaf) collapses that leaf + the colliding key into ONE bucket — but only if
// both suffixes share a single page. Two ~260-byte suffixes each fit a 512B page alone yet
// overflow together; the old per-suffix pre-check let the collapse proceed and overflow the
// page. The shape below builds root{A,B} → A:{a,b} → leaf (two nested 2-branch setlists),
// then collides at the "Aa" leaf with a long tail. (Verified: reverting bkt_fits2 to the
// per-suffix bkt_fits makes this block crash/assert.)
static void regression_adaptive_collapse_pair_fit()
{
   std::vector<std::string> keys = {
       "Aa" + std::string(260, 'x'),  // the target leaf (deep-narrow: under root{A,B}, A:{a,b})
       "Ab" + std::string(4, 'z'),    // sibling under 'A'  → A-level setlist has 2 branches (<3)
       "B" + std::string(4, 'y'),     // sibling at root    → root setlist has 2 branches (<3)
       "Aa" + std::string(260, 'w'),  // COLLIDES at the "Aa" leaf; 260+260 overflows one page
   };
   roundtrip<map<std::string_view, uint64_t, mode::adaptive>, uint64_t>(keys, false, "adaptive pair-fit fwd");
   roundtrip<map<std::string_view, uint64_t, mode::adaptive>, uint64_t>(keys, true, "adaptive pair-fit rev");
}

// ── #1: bkt_put must externalize an inline-leaf setlist child before descending ──
// With a small (inlineable) value, setlist children can live inline in the node's line.
// Bucket-mode descent through such a child must first externalize it; otherwise the
// inline tail-offset is mis-dereferenced as an external leaf handle on the next split.
// uint32_t (4B) is inlineable; many short shared-prefix keys build inline-leaf setlists,
// then longer keys force bkt_put to descend through and split them.
static void regression_bkt_put_inline_leaf()
{
   std::vector<std::string> keys;
   std::mt19937             rng(99);
   for (int a = 0; a < 12; ++a)
      for (int b = 0; b < 12; ++b)
      {
         // short keys → inline-leaf children; then a long sibling sharing the 2-byte stem
         keys.push_back(std::string{char('a' + a), char('a' + b)});
         keys.push_back(std::string{char('a' + a), char('a' + b)} + std::string(40 + (a * b) % 200, 'q'));
      }
   std::shuffle(keys.begin(), keys.end(), rng);
   roundtrip<map<std::string_view, uint32_t, mode::buckets>, uint32_t>(keys, false, "inline-leaf fwd");
   roundtrip<map<std::string_view, uint32_t, mode::buckets>, uint32_t>(keys, true, "inline-leaf rev");
}

// ── #3: for_each_value (walk_v_) must visit every router kind ────────────────────
// walk_v_ is a kind-switch; a missing case skips (or mis-walks) that subtree's values.
// The planted bug dropped the setlist_u16 arm. agree()'s for_each_value multiset check
// catches a missed/extra value directly. A ~20-wide cluster lands on a surviving cnode
// (c4 → walk_v_'s cnode_for_each arm); a 200-wide root becomes a node_full (the default
// arm). Verified: breaking either walk_v_ arm makes this block fail.
//
// NOTE on setlist_u16: that node only forms via the wide_stems online-fusion policy,
// whose trigger (a prefix-less parent whose children are ALL term-less prefix-less
// single-byte routers, ≤ FUSE_THRESH grandchildren, hit by a leaf-split with a tracked
// parent slot) is not reachable by a fixed insert sequence — even the dedicated
// wide_stems suite builds none. The differential fuzzer (tests/fuzz_map.cpp, run by
// ctest), which reaches that shape via mixed insert/erase, is the guard for the u16 arm;
// it is what caught this bug. wide_stems is still swept below for completeness.
static void regression_walk_dense_routers()
{
   std::vector<std::string> cn;  // 'X' parent + ~20 second bytes → a surviving cnode (c4)
   for (int i = 0; i < 20; ++i)
      for (int j = 0; j < 2; ++j)
         cn.push_back(std::string{'X', char(i & 0xff)} + std::string(1 + j, 'q'));
   roundtrip<map<std::string_view, uint64_t, mode::dense_tiers>, uint64_t>(cn, false, "walk cnode fwd");
   roundtrip<map<std::string_view, uint64_t, mode::dense_tiers>, uint64_t>(cn, true, "walk cnode rev");

   std::vector<std::string> wide;  // 200 distinct first bytes → node_full at the root
   for (int i = 0; i < 200; ++i)
      for (int j = 0; j < 3; ++j)
         wide.push_back(std::string{char(i & 0xff)} + std::string(1 + j, char('A' + j)));
   roundtrip<map<std::string_view, uint64_t, mode::dense_tiers>, uint64_t>(wide, false, "walk full fwd");
   roundtrip<map<std::string_view, uint64_t, mode::wide_stems>, uint64_t>(wide, false, "walk wide fwd");
}

// ── audit #3: an erase cascade longer than the 64-level path window ──────────────
// Keys a^i+"b" (i < N) plus a^N build an N-level chain of two-way routers. Erasing the
// b-keys deepest first leaves a chain of single-child nodes leading to the one remaining
// leaf; erasing that leaf kills the whole chain in ONE erase. remove() records only the
// deepest 64 levels of the path, and at that window's edge it used to write a null handle
// into a surviving ancestor instead of unlinking the edge: size() == 0 with a non-null
// root holding a null child, so begin() / for_each / lower_bound crashed. The shapes put
// the cascade's survivor at each interesting place: none (the root dies), a second branch
// at the root, a term part-way down the chain, a second branch part-way down.
template <class V>
static V chain_val(int x)
{
   if constexpr (std::is_same_v<V, std::string>) return "v" + std::to_string(x);
   else return V(x + 1);
}
template <class Tree, class V>
static void deep_chain(int N, int shape, const char* where)
{
   Tree                     t;
   std::map<std::string, V> oracle;
   auto put = [&](const std::string& k, int x) {
      const V v = chain_val<V>(x);
      t.insert_or_assign(k, v);
      oracle[k] = v;
   };
   auto del = [&](const std::string& k) { CHECK(t.erase(k) == oracle.erase(k)); };
   const std::string last(size_t(N), 'a');
   for (int i = 0; i < N; ++i) put(std::string(size_t(i), 'a') + "b", i);
   put(last, N);
   if (shape == 1) put("x", N + 1);                 // survivor: the root keeps a second branch
   if (shape == 2) put("aaa", N + 1);               // survivor: a term 3 levels down
   if (shape == 3) put("aaaaazz", N + 1);           // survivor: a second branch 5 levels down
   for (int i = N - 1; i >= 0; --i) del(std::string(size_t(i), 'a') + "b");  // deepest first
   agree<Tree, V>(t, oracle, where);
   del(last);                                       // the whole chain dies in this one erase
   agree<Tree, V>(t, oracle, where);
   size_t n = 0;
   t.for_each([&](std::string_view, const V&) { ++n; });
   CHECK(n == oracle.size());
   CHECK((t.lower_bound("a") == t.end()) == (oracle.lower_bound("a") == oracle.end()));
   CHECK(t.empty() == oracle.empty());
   if (shape == 0) CHECK(t.begin() == t.end());
   // the tree must still be fully usable: rebuild the same shape on top of what is left
   for (int i = 0; i < N; ++i) put(std::string(size_t(i), 'a') + "b", i);
   put(last, N);
   agree<Tree, V>(t, oracle, where);
   for (int i = 0; i < N; ++i) del(std::string(size_t(i), 'a') + "b");       // shallowest first
   del(last);
   agree<Tree, V>(t, oracle, where);
}
static void regression_erase_cascade_past_window()
{
   for (int N : {40, 63, 64, 65, 66, 127, 128, 129, 200, 1000})
      for (int shape = 0; shape < 4; ++shape)
      {
         deep_chain<map<std::string_view, std::string>, std::string>(N, shape, "chain string");
         deep_chain<map<std::string_view, uint64_t>, uint64_t>(N, shape, "chain u64");
         deep_chain<map<std::string_view, uint32_t>, uint32_t>(N, shape, "chain u32 (inline values)");
         deep_chain<map<std::string_view, uint64_t, mode::dense_tiers>, uint64_t>(N, shape, "chain dense");
         deep_chain<map<std::string_view, uint64_t, mode::wide_stems>, uint64_t>(N, shape, "chain wide");
         deep_chain<map<std::string_view, uint64_t, mode::adaptive>, uint64_t>(N, shape, "chain adaptive");
         deep_chain<map<std::string_view, std::string, mode::buckets>, std::string>(N, shape, "chain buckets");
      }
   // the set shares the erase path (terminals are suffix-in-handle there)
   for (int N : {65, 200})
   {
      artpp::set<std::string> s;
      std::set<std::string>   o;
      const std::string       last(size_t(N), 'a');
      for (int i = 0; i < N; ++i) { s.insert(std::string(size_t(i), 'a') + "bbbbbbbb"); o.insert(std::string(size_t(i), 'a') + "bbbbbbbb"); }
      s.insert(last + "cccccccc");
      o.insert(last + "cccccccc");
      for (int i = N - 1; i >= 0; --i) CHECK(s.erase(std::string(size_t(i), 'a') + "bbbbbbbb") == 1);
      CHECK(s.erase(last + "cccccccc") == 1);
      CHECK(s.size() == 0);
      CHECK(s.begin() == s.end());
      size_t n = 0;
      s.for_each([&](const std::string&) { ++n; });
      CHECK(n == 0);
   }
}

// ── audit #11: assigning to a PRESENT key must not restructure the tree ───────────
// README, Iterators / Invalidation: insert / insert_or_assign / emplace on a key that is
// already present keep every iterator and value reference valid. Three paths broke that:
//   * a setlist's inline leaf child was externalized (and the line repacked, moving the
//     OTHER inline values) before the key was known to be present;
//   * adaptive mode collapsed the matching leaf into a bucket, freeing it;
//   * a prefix node above a node_full was fused into the full's header (and freed) by any
//     write passing through it, leaving iterators with a frame on the freed node.
// The check is exact and needs no sanitizer: across the assignment the allocator must see
// no allocation and no deallocation, the value's address must not change, and iterators
// taken before it (on the key and on its successor) must still read the right elements.
static long g_nalloc = 0, g_nfree = 0;
template <class T>
struct count_alloc
{
   using value_type = T;
   count_alloc()    = default;
   template <class U>
   count_alloc(const count_alloc<U>&) noexcept {}
   T* allocate(std::size_t n)
   {
      ++g_nalloc;
      return static_cast<T*>(::operator new(n * sizeof(T), std::align_val_t(alignof(T))));
   }
   void deallocate(T* p, std::size_t) noexcept
   {
      ++g_nfree;
      ::operator delete(p, std::align_val_t(alignof(T)));
   }
   template <class U>
   bool operator==(const count_alloc<U>&) const noexcept { return true; }
};
template <class V, mode M>
static void assign_present(const std::vector<std::string>& keys, const char* where)
{
   using Tree = map<std::string_view, V, M, count_alloc<V>>;
   const long a_start = g_nalloc, f_start = g_nfree;
   {
      Tree                     t;
      std::map<std::string, V> oracle;
      int                      x = 0;
      for (auto& k : keys) { const V v = chain_val<V>(x++); t.insert_or_assign(k, v); oracle[k] = v; }
      for (auto& k : keys)
      {
         auto it = t.find(k);
         CHECK(it != t.end());
         auto nx = it;
         ++nx;  // an iterator on the successor, taken before the assignment
         const V* p = nullptr;
         if constexpr (!Tree::inlineable) p = t.find_ptr(k);
         auto quiet = [&, a0 = g_nalloc, f0 = g_nfree] { return g_nalloc == a0 && g_nfree == f0; };
         auto same  = [&](const V& v) {
            CHECK(quiet());  // nothing allocated, nothing freed: the structure is untouched
            if constexpr (!Tree::inlineable)
            {
               CHECK(t.find_ptr(k) == p);  // the value did not move …
               CHECK(*p == v);             // … and a reference held across the call reads it
               CHECK(it.value() == v);     // non-inline values are read live through iterators
            }
            CHECK(std::string(it.key_bytes()) == k);
            auto oi = oracle.find(k);
            ++oi;
            CHECK((nx == t.end()) == (oi == oracle.end()));
            if (oi != oracle.end())
            {
               CHECK(std::string(nx.key_bytes()) == oi->first);
               CHECK(nx.value() == oi->second);
            }
            auto step = it;  // walking on from the held iterator reaches the successor
            ++step;
            CHECK(step == nx);
         };
         V v = chain_val<V>(x++);
         CHECK(!t.insert_or_assign(k, v));
         oracle[k] = v;
         same(v);
         v = chain_val<V>(x++);
         CHECK(!t.insert(k, v));
         oracle[k] = v;
         same(v);
         CHECK(!t.emplace(k, chain_val<V>(x++)));  // present: left untouched
         same(v);
      }
      agree<Tree, V>(t, oracle, where);
   }
   CHECK(g_nalloc - a_start == g_nfree - f_start);
}
template <class V>
static void assign_present_all_modes(const std::vector<std::string>& keys, const char* where)
{
   assign_present<V, mode::none>(keys, where);
   assign_present<V, mode::dense_tiers>(keys, where);
   assign_present<V, mode::dense_tiers | mode::ladder_c8>(keys, where);
   assign_present<V, mode::wide_stems>(keys, where);
   assign_present<V, mode::adaptive>(keys, where);
   assign_present<V, mode::adaptive | mode::dense_tiers>(keys, where);
   assign_present<V, mode::buckets>(keys, where);
}
static void regression_assign_present_key()
{
   std::vector<std::vector<std::string>> shapes;
   shapes.push_back({"apple", "apricot"});                        // two inline leaves in one line
   shapes.push_back({"a", "ab", "abc", "abcd", "abcde", "abx"});  // adaptive deep-narrow collapse
   for (int fan : {17, 40, 200})  // a long shared prefix over a router that widens to node_full
   {
      std::vector<std::string> v;
      const std::string        pfx = "0123456789AB";
      for (int i = 0; i < fan; ++i) v.push_back(pfx + char(1 + i) + "tail");
      v.push_back(pfx);                        // a key ending at the router (its term)
      v.push_back(pfx + char(1) + "tail2");    // and a leaf split below it
      shapes.push_back(v);
   }
   {
      std::vector<std::string> v;  // clustered: inline leaves, terms, buckets
      for (int a = 0; a < 6; ++a)
      {
         v.push_back(std::string("k") + char('a' + a));
         for (int b = 0; b < 6; ++b) v.push_back(std::string("k") + char('a' + a) + char('a' + b) + "x");
      }
      v.push_back("");
      v.push_back(std::string(600, 'L'));  // over a bucket page: a plain leaf even in bucket mode
      v.push_back(std::string(600, 'L') + "s");
      shapes.push_back(v);
   }
   for (auto& keys : shapes)
   {
      assign_present_all_modes<uint64_t>(keys, "assign present u64");
      assign_present_all_modes<std::string>(keys, "assign present string");
      assign_present_all_modes<uint32_t>(keys, "assign present u32 (inline values)");
   }
}

// ── audit #14: values with alignof(T) > 16 ───────────────────────────────────────
// Leaves came from a 16-byte-granular terminal allocation while the value sits at offset
// alignof(T) inside the leaf, so an alignas(32) value landed on a 16-byte boundary about
// half the time (UB; an aligned AVX load faults). The terminal unit now follows alignof(T).
// Every address the map hands out for a value must be aligned, in every place a value can
// live: external leaves, a router's term, a setlist's inline leaves, bucket entries.
struct alignas(32) V32
{
   double d[4];
   bool   operator==(const V32& o) const { return d[0] == o.d[0] && d[3] == o.d[3]; }
   bool   operator<(const V32& o) const { return d[0] < o.d[0]; }
};
struct alignas(64) V64
{
   uint64_t a;
   char     pad[40];
   bool     operator==(const V64& o) const { return a == o.a; }
   bool     operator<(const V64& o) const { return a < o.a; }
};
static_assert(alignof(V32) == 32 && alignof(V64) == 64);
template <class V>
static V over_val(int x)
{
   if constexpr (std::is_same_v<V, V32>) return V32{{double(x), 2, 3, double(x) + 0.5}};
   else { V64 v{}; v.a = uint64_t(x); return v; }
}
// An allocator that returns storage aligned to alignof(T) and NOT to 2*alignof(T): a
// conforming allocator may do exactly that, and it makes the test independent of how
// generously the platform's malloc happens to align (macOS' default allocator hid the bug).
template <class T>
struct exact_align_alloc
{
   using value_type = T;
   exact_align_alloc() = default;
   template <class U>
   exact_align_alloc(const exact_align_alloc<U>&) noexcept {}
   T* allocate(std::size_t n)
   {
      const std::size_t A   = alignof(T);
      char*             raw = static_cast<char*>(::operator new(n * sizeof(T) + 3 * A + sizeof(void*)));
      std::uintptr_t    u   = reinterpret_cast<std::uintptr_t>(raw) + sizeof(void*);
      u                     = (u + 2 * A - 1) / (2 * A) * (2 * A) + A;  // == A (mod 2A)
      std::memcpy(reinterpret_cast<char*>(u) - sizeof(void*), &raw, sizeof(void*));
      return reinterpret_cast<T*>(u);
   }
   void deallocate(T* p, std::size_t) noexcept
   {
      char* raw;
      std::memcpy(&raw, reinterpret_cast<char*>(p) - sizeof(void*), sizeof(void*));
      ::operator delete(raw);
   }
   template <class U>
   bool operator==(const exact_align_alloc<U>&) const noexcept { return true; }
};
template <class V, mode M>
static void overaligned(const char* where)
{
   using Tree = map<std::string_view, V, M, exact_align_alloc<V>>;
   Tree                     t;
   std::map<std::string, V> oracle;
   std::vector<std::string> keys;
   for (int i = 0; i < 64; ++i) keys.push_back("key" + std::to_string(i));       // leaf splits, a wide router
   for (int i = 0; i < 6; ++i) keys.push_back(std::string(size_t(i), 'p'));      // terms ("" included)
   for (int i = 0; i < 40; ++i) keys.push_back(std::string("q") + char('a' + i % 5) + char('a' + i));
   keys.push_back(std::string(700, 'L'));                                        // a multi-unit leaf
   auto aligned = [&] {
      size_t mis = 0, n = 0;
      for (auto& kv : oracle)
      {
         const V* p = t.find_ptr(kv.first);
         CHECK(p != nullptr);
         if (!p) continue;
         mis += reinterpret_cast<uintptr_t>(p) % alignof(V) != 0;
         CHECK(*p == kv.second);
      }
      for (auto it = t.begin(); it != t.end(); ++it, ++n)
         mis += reinterpret_cast<uintptr_t>(&it.value()) % alignof(V) != 0;
      t.for_each([&](std::string_view, const V& v) { mis += reinterpret_cast<uintptr_t>(&v) % alignof(V) != 0; });
      CHECK(mis == 0);
      CHECK(n == oracle.size());
   };
   int x = 0;
   for (auto& k : keys) { const V v = over_val<V>(x++); t.insert_or_assign(k, v); oracle[k] = v; }
   aligned();
   agree<Tree, V>(t, oracle, where);
   for (size_t i = 0; i < keys.size(); i += 2) CHECK(t.erase(keys[i]) == oracle.erase(keys[i]));
   aligned();
   for (auto& k : keys) { const V v = over_val<V>(x++); t.insert_or_assign(k, v); oracle[k] = v; }
   aligned();
   agree<Tree, V>(t, oracle, where);
   Tree moved(std::move(t));  // steals; values stay put
   CHECK(moved.size() == oracle.size());
   for (auto& k : keys) CHECK(moved.erase(k) == 1);
   CHECK(moved.empty());
}
static void regression_overaligned_values()
{
   overaligned<V32, mode::none>("alignas(32)");
   overaligned<V32, mode::dense_tiers>("alignas(32) dense");
   overaligned<V32, mode::wide_stems>("alignas(32) wide");
   overaligned<V32, mode::adaptive>("alignas(32) adaptive");
   overaligned<V32, mode::buckets>("alignas(32) buckets");
   overaligned<V64, mode::none>("alignas(64)");
   overaligned<V64, mode::dense_tiers>("alignas(64) dense");
   overaligned<V64, mode::adaptive>("alignas(64) adaptive");
   overaligned<V64, mode::buckets>("alignas(64) buckets");
}

// ── audit #15: a deep tree must not need a deep call stack ───────────────────────
// Keys a^1 … a^N build an N-level tree (every level has a term). Teardown, for_each,
// for_each_value, lower_bound, operator== and the cross-allocator move (structural clone)
// used to recurse once per level; at N = 8,000 that overflowed a 512 KB thread stack.
// Run them all on a thread whose stack is far too small for N nested frames of anything:
// with the recursion gone (or bounded and handed to the heap-stacked iterator) it passes.
template <class T>
struct tagged_alloc  // stateful and non-propagating: unequal ids force move-assign to clone
{
   using value_type = T;
   int id;
   explicit tagged_alloc(int i = 0) noexcept : id(i) {}
   template <class U>
   tagged_alloc(const tagged_alloc<U>& o) noexcept : id(o.id) {}
   T*   allocate(std::size_t n) { return static_cast<T*>(::operator new(n * sizeof(T), std::align_val_t(alignof(T)))); }
   void deallocate(T* p, std::size_t) noexcept { ::operator delete(p, std::align_val_t(alignof(T))); }
   template <class U>
   bool operator==(const tagged_alloc<U>& o) const noexcept { return id == o.id; }
   template <class U>
   bool operator!=(const tagged_alloc<U>& o) const noexcept { return id != o.id; }
};
template <class V, mode M>
static void deep_tree_ops(int N)
{
   using A    = tagged_alloc<V>;
   using Tree = map<std::string_view, V, M, A>;
   Tree        a{A(1)}, b{A(1)}, c{A(2)};
   std::string k;
   for (int i = 1; i <= N; ++i)
   {
      k.push_back('a');
      a.insert(k, chain_val<V>(i));
      b.insert(k, chain_val<V>(i));
   }
   CHECK(a.size() == size_t(N));
   size_t n = 0, bad = 0;
   a.for_each([&](std::string_view key, const V& v) {  // ascending: a, aa, aaa, …
      ++n;
      bad += key.size() != n || key.find_first_not_of('a') != std::string_view::npos || !(v == chain_val<V>(int(n)));
   });
   CHECK(n == size_t(N) && bad == 0);
   n = 0;
   a.for_each_value([&](const V& v) { ++n; bad += !(v == chain_val<V>(int(n))); });
   CHECK(n == size_t(N) && bad == 0);
   {
      const auto st = a.debug_stats();  // the census walk: its work stack is on the heap
      if (artpp::has_mode(M, mode::buckets) || artpp::has_mode(M, mode::adaptive))
         CHECK(st.terminals <= uint64_t(N));  // keys held in buckets are not counted
      else
         CHECK(st.terminals == uint64_t(N));
      CHECK(st.setlist + st.full + st.fullp + st.c2 + st.c4 + st.c8 + st.setlist16 + st.prefix > 100);
   }
   auto lb = a.lower_bound(k);  // the deepest key: the directed descent goes all the way down
   CHECK(lb != a.end() && lb.key_bytes() == k);
   CHECK(a.upper_bound(k) == a.end());
   CHECK(a.find(k) != a.end());
   CHECK(a == b);  // structural walk, then the sequence compare past the depth bound
   b.insert_or_assign(k, chain_val<V>(-5));
   CHECK(a != b);  // differs only in the deepest value
   b.insert_or_assign(k, chain_val<V>(N));
   c = std::move(a);  // unequal allocators: node-by-node clone of the whole depth
   CHECK(c.size() == size_t(N) && a.empty());
   CHECK(c == b);
   n = 0;
   for (auto it = c.begin(); it != c.end(); ++it) ++n;
   CHECK(n == size_t(N));
   b.clear();         // teardown of the full depth …
   CHECK(b.empty() && b.begin() == b.end());
}                     // … and again in c's destructor
struct stack_job
{
   void (*fn)(int);
   int  arg;
};
static void* stack_tramp(void* p)
{
   auto* j = static_cast<stack_job*>(p);
   j->fn(j->arg);
   return nullptr;
}
static void run_on_stack(std::size_t bytes, void (*fn)(int), int arg)
{
   pthread_attr_t attr;
   pthread_attr_init(&attr);
   CHECK(pthread_attr_setstacksize(&attr, bytes) == 0);
   stack_job j{fn, arg};
   pthread_t t;
   CHECK(pthread_create(&t, &attr, stack_tramp, &j) == 0);
   pthread_join(t, nullptr);
   pthread_attr_destroy(&attr);
}
static void regression_deep_tree_small_stack()
{
   // 12,000 levels on a 256 KB stack: about 21 bytes per level, less than one call frame of
   // anything, while the bounded scans (64 recursion levels, then the iterator) fit with room
   // to spare even in a sanitizer build.
   const std::size_t stack = 256 * 1024;
   const int         N     = 12000;
   run_on_stack(stack, [](int n) { deep_tree_ops<uint64_t, mode::none>(n); }, N);
   run_on_stack(stack, [](int n) { deep_tree_ops<std::string, mode::none>(n); }, N);
   run_on_stack(stack, [](int n) { deep_tree_ops<uint32_t, mode::none>(n); }, N);
   run_on_stack(stack, [](int n) { deep_tree_ops<uint64_t, mode::dense_tiers>(n); }, N);
   run_on_stack(stack, [](int n) { deep_tree_ops<uint64_t, mode::wide_stems>(n); }, N);
   // Bucket mode: each burst of the page holding the chain's tail peels off one router, so
   // the same keys make a tree about one level per two keys deep — and insert used to recurse
   // once per level (an unoptimized build overflowed this stack after 230 levels). Half the
   // keys keep the test quick: 3,000 levels is still far more than the stack could carry.
   run_on_stack(stack, [](int n) { deep_tree_ops<uint64_t, mode::buckets>(n); }, N / 2);
   run_on_stack(stack, [](int n) { deep_tree_ops<std::string, mode::buckets>(n); }, N / 2);
   run_on_stack(stack, [](int n) { deep_tree_ops<uint64_t, mode::adaptive>(n); }, N / 2);
}

// ── audit #16: blocks the handle cannot address are refused, not truncated ───────
// A direct handle packs the block's address into 6 bytes and keeps its tag in the low
// bits. Both requirements on the allocator — alignment (128 for nodes, 16 for leaves) and
// an address below 2^48 — were checked only by assert, so a release build silently built
// a handle to a different address. They are now checked where the block is allocated:
// the block goes back to the allocator and bad_allocation_address (a std::bad_alloc) is
// thrown, before anything is written through it.
static int                  g_odd_mode = 0;  // 1: nodes 64 mod 128   2: leaves 8 mod 16   3: top-byte tag on nodes
static long                 g_odd_alloc = 0, g_odd_free = 0, g_odd_foreign = 0;
static std::set<void*>      g_odd_live;      // exactly the pointers allocate() handed out
template <class T>
struct odd_alloc
{
   using value_type = T;
   odd_alloc() = default;
   template <class U>
   odd_alloc(const odd_alloc<U>&) noexcept {}
   T* allocate(std::size_t n)
   {
      const std::size_t A   = alignof(T);
      char*             raw = static_cast<char*>(::operator new(n * sizeof(T) + 2 * A + sizeof(void*)));
      std::uintptr_t    u   = (reinterpret_cast<std::uintptr_t>(raw) + sizeof(void*) + A - 1) / A * A;  // aligned
      if ((g_odd_mode == 1 && A == 128) || (g_odd_mode == 2 && A == 16)) u += A / 2;  // under-aligned
      std::memcpy(reinterpret_cast<char*>(u) - sizeof(void*), &raw, sizeof(void*));
      if (g_odd_mode == 3 && A == 128) u |= std::uintptr_t(0xB4) << 56;  // HWASan/MTE-style tagged pointer
      ++g_odd_alloc;
      g_odd_live.insert(reinterpret_cast<void*>(u));
      return reinterpret_cast<T*>(u);
   }
   void deallocate(T* p, std::size_t) noexcept
   {
      ++g_odd_free;
      if (g_odd_live.erase(p) == 0) { ++g_odd_foreign; return; }  // not a pointer we handed out
      const std::uintptr_t u = reinterpret_cast<std::uintptr_t>(p) & ~(std::uintptr_t(0xFF) << 56);
      char*                raw;
      std::memcpy(&raw, reinterpret_cast<char*>(u) - sizeof(void*), sizeof(void*));
      ::operator delete(raw);
   }
   template <class U>
   bool operator==(const odd_alloc<U>&) const noexcept { return true; }
};
static void regression_unaddressable_blocks()
{
   using Tree = map<std::string_view, uint64_t, mode::none, odd_alloc<uint64_t>>;
   for (int bad : {1, 2, 3})
   {
      g_odd_alloc = g_odd_free = g_odd_foreign = 0;
      {
         Tree t;
         g_odd_mode = 0;  // a healthy start: one leaf, then a split that needs a node
         CHECK(t.insert("alpha", 1));
         CHECK(t.insert("beta", 2));
         g_odd_mode = bad;
         int threw = 0, specific = 0;
         for (const char* k : {"alpine", "gamma", "be", "alphabet"})
         {
            try { t.insert(k, 9); }
            catch (const artpp::bad_allocation_address& e) { ++threw; ++specific; CHECK(e.what()[0] == 'a'); }
            catch (const std::bad_alloc&) { ++threw; }
         }
         CHECK(threw > 0 && threw == specific);  // refused, with the specific exception type
         g_odd_mode = 0;
         uint64_t v = 0;                      // the tree is intact and still usable
         CHECK(t.find("alpha", v) && v == 1);
         CHECK(t.find("beta", v) && v == 2);
         size_t n = 0;
         for (auto it = t.begin(); it != t.end(); ++it) ++n;
         CHECK(n == t.size());
         CHECK(t.insert("gamma", 3) || t.contains("gamma"));
         CHECK(t.find("gamma", v));
      }
      CHECK(g_odd_alloc == g_odd_free);  // every refused block went back to the allocator …
      CHECK(g_odd_foreign == 0);         // … as the very pointer it was handed (no truncation)
      CHECK(g_odd_live.empty());
   }
   g_odd_mode = 0;
   {  // an allocator that opts into full-width handles may use the whole address space
      CHECK(artpp::packed_ptr_t<8>::addressable(reinterpret_cast<void*>(std::uintptr_t(0xB4) << 56), 0x7f));
      CHECK(!artpp::packed_ptr_t<6>::addressable(reinterpret_cast<void*>(std::uintptr_t(1) << 48), 0x7f));
      CHECK(!artpp::packed_ptr_t<6>::addressable(reinterpret_cast<void*>(std::uintptr_t(64)), 0x7f));
      CHECK(artpp::packed_ptr_t<6>::addressable(reinterpret_cast<void*>((std::uintptr_t(1) << 48) - 128), 0x7f));
   }
}

// ── audit #18: README claims — bool keys, no default constructor needed ──────────
// "Built in: … all integral types": map<bool, T> did not compile (make_unsigned<bool>).
// "default-constructibility [is] required only by the operations that use [it]": a small
// trivially copyable T without a default constructor could not insert or iterate (the
// in-handle value was unpacked into a `T tmp;`), and no T without one compiled at all in
// mode::adaptive. Both are compile-time defects, so instantiating these is most of the test.
struct small_nd  // trivially copyable, 2 bytes → stored in the handle; no default constructor
{
   uint16_t x;
   explicit small_nd(uint16_t v) : x(v) {}
   bool operator==(const small_nd& o) const { return x == o.x; }
   bool operator<(const small_nd& o) const { return x < o.x; }
};
static_assert(!std::is_default_constructible_v<small_nd> && std::is_trivially_copyable_v<small_nd>);
struct big_nd  // non-trivial, lives in leaves; no default constructor
{
   std::string s;
   explicit big_nd(std::string v) : s(std::move(v)) {}
   bool operator==(const big_nd& o) const { return s == o.s; }
   bool operator<(const big_nd& o) const { return s < o.s; }
};
static_assert(!std::is_default_constructible_v<big_nd>);
template <class V>
static V nd_val(int x)
{
   if constexpr (std::is_same_v<V, small_nd>) return small_nd(uint16_t(x));
   else return big_nd("v" + std::to_string(x));
}
template <class V, mode M>
static void no_default_ctor(const char* where)
{
   using Tree = map<std::string_view, V, M>;
   static_assert(Tree::inlineable == std::is_same_v<V, small_nd>);
   Tree                     t;
   std::map<std::string, V> oracle;
   std::vector<std::string> keys = {"", "a", "ab", "abc", "abcd", "abd", "b", "ba", "apple", "apricot"};
   for (int i = 0; i < 40; ++i) keys.push_back(std::string("k") + char('a' + i % 7) + char('A' + i));
   int x = 0;
   for (auto& k : keys)
   {
      const V v = nd_val<V>(x++);
      if (x % 3 == 0) CHECK(t.emplace(k, v) == oracle.emplace(k, v).second);
      else { t.insert_or_assign(k, v); oracle.insert_or_assign(k, v); }
   }
   for (auto& k : keys)  // every read path, none of which may need V()
   {
      V out = nd_val<V>(-1);
      CHECK(t.find(k, out) && out == oracle.at(k));
      CHECK(t.contains(k));
      auto it = t.find(k);
      CHECK(it != t.end() && it.value() == oracle.at(k));
   }
   agree<Tree, V>(t, oracle, where);  // iterators both ways + for_each_value
   size_t n = 0;
   t.for_each([&](std::string_view k, const V& v) { ++n; CHECK(v == oracle.at(std::string(k))); });
   CHECK(n == oracle.size());
   {
      Tree u;
      for (auto& kv : oracle) u.insert(kv.first, kv.second);
      CHECK(u == t);
      const V nv = nd_val<V>(9999);
      CHECK(u.update("abc", nv));
      CHECK(u != t);
      Tree w(std::move(u));
      CHECK(w.size() == oracle.size());
   }
   for (size_t i = 0; i < keys.size(); i += 2) CHECK(t.erase(keys[i]) == oracle.erase(keys[i]));
   agree<Tree, V>(t, oracle, where);
}
static void regression_readme_claims()
{
   {  // bool keys: one byte, false before true
      map<bool, int> m;
      CHECK(m.insert(true, 1));
      CHECK(m.insert(false, 0));
      CHECK(!m.insert(true, 11));
      CHECK(m.size() == 2);
      auto it = m.begin();
      CHECK(it.key() == false && it.value() == 0 && it.key_bytes().size() == 1);
      ++it;
      CHECK(it.key() == true && it.value() == 11);
      ++it;
      CHECK(it == m.end());
      int v = -1;
      CHECK(m.find(false, v) && v == 0);
      CHECK(m.lower_bound(true).key() == true);
      CHECK(m.erase(false) == 1 && m.begin().key() == true);
      artpp::set<bool> s;
      CHECK(s.insert(true).second && s.insert(false).second && !s.insert(true).second);
      CHECK(*s.begin() == false && s.size() == 2);
   }
   no_default_ctor<small_nd, mode::none>("no default ctor, inline");
   no_default_ctor<small_nd, mode::dense_tiers>("no default ctor, inline, dense");
   no_default_ctor<small_nd, mode::wide_stems>("no default ctor, inline, wide");
   no_default_ctor<small_nd, mode::adaptive>("no default ctor, inline, adaptive");
   no_default_ctor<small_nd, mode::buckets>("no default ctor, inline, buckets");
   no_default_ctor<big_nd, mode::none>("no default ctor, leaf");
   no_default_ctor<big_nd, mode::dense_tiers>("no default ctor, leaf, dense");
   no_default_ctor<big_nd, mode::wide_stems>("no default ctor, leaf, wide");
   no_default_ctor<big_nd, mode::adaptive>("no default ctor, leaf, adaptive");
   no_default_ctor<big_nd, mode::buckets>("no default ctor, leaf, buckets");
}

int main()
{
   regression_bkt_split_long_prefix();
   regression_adaptive_collapse_pair_fit();
   regression_bkt_put_inline_leaf();
   regression_walk_dense_routers();
   regression_erase_cascade_past_window();
   regression_assign_present_key();
   regression_overaligned_values();
   regression_deep_tree_small_stack();
   regression_unaddressable_blocks();
   regression_readme_claims();
   if (g_fail == 0) std::printf("artpp_regression: ALL OK\n");
   else std::printf("artpp_regression: %d FAILURE(S)\n", g_fail);
   return g_fail ? 1 : 0;
}

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
#include <map>
#include <random>
#include <set>
#include <string>
#include <vector>

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

int main()
{
   regression_bkt_split_long_prefix();
   regression_adaptive_collapse_pair_fit();
   regression_bkt_put_inline_leaf();
   regression_walk_dense_routers();
   if (g_fail == 0) std::printf("artpp_regression: ALL OK\n");
   else std::printf("artpp_regression: %d FAILURE(S)\n", g_fail);
   return g_fail ? 1 : 0;
}

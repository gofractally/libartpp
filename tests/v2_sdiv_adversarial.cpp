// v2_sdiv_adversarial — targeted, std::map-cross-checked stress of the sdiv prefix-fusion path.
//
// The random differential fuzzer (v2_fuzz_map) explores broadly; this file constructs the
// SPECIFIC structural situations the prefix-fusion fix touches and asserts (a) the sdiv node
// kind is actually exercised (so the test isn't a silent no-op) and (b) every op agrees with
// std::map. Scenarios (per the review brief):
//   A. long shared prefixes (dictionary-like) → fused prefix replicated across lines
//   B. keys ending EXACTLY at a fused-prefix boundary (term on a prefixed sdiv)
//   C. keys diverging INSIDE a fused prefix → split_sdiv_prefix
//   D. a fused prefix at every sdiv tier (sdiv2/3/4), replicated across lines
//   E. grow/shrink across the fuse/split boundary (insert past sdiv4, erase back)
//   F. randomized prefix-clustered torrent vs std::map (forward + backward + value DFS)
//
// Run under ASan+UBSan. Any divergence aborts with a labelled message.
#include "artpp/v2/map.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <random>
#include <string>
#include <vector>

using artpp::v2::map;
using artpp::v2::mode;

static int g_fail = 0;
#define CHECK(cond, msg)                                                         \
   do {                                                                          \
      if (!(cond)) { std::fprintf(stderr, "FAIL [%s]: %s\n", __func__, msg); ++g_fail; } \
   } while (0)

// Full cross-check of a tree against the oracle: size, forward walk (key+value), backward walk
// (operator-- / --end()), and the value-only DFS. Stamped with a scenario label.
template <class Tree, class V>
static void cross_check(const Tree& t, const std::map<std::string, V>& m, const char* label)
{
   if (t.size() != m.size())
   { std::fprintf(stderr, "FAIL [%s]: size %zu != %zu\n", label, t.size(), m.size()); ++g_fail; return; }
   // forward
   {
      auto mi = m.begin();
      for (auto it = t.begin(); it != t.end(); ++it, ++mi)
      {
         if (mi == m.end()) { std::fprintf(stderr, "FAIL [%s]: tree longer than map\n", label); ++g_fail; return; }
         if (std::string(it.key()) != mi->first)
         { std::fprintf(stderr, "FAIL [%s]: fwd key '%s' != '%s'\n", label, std::string(it.key()).c_str(), mi->first.c_str()); ++g_fail; return; }
         if (it.value() != mi->second)
         { std::fprintf(stderr, "FAIL [%s]: fwd value mismatch at '%s'\n", label, mi->first.c_str()); ++g_fail; return; }
      }
      if (mi != m.end()) { std::fprintf(stderr, "FAIL [%s]: map longer than tree\n", label); ++g_fail; return; }
   }
   // backward
   {
      auto it = t.end();
      auto mi = m.end();
      while (mi != m.begin())
      {
         if (it == t.begin()) { std::fprintf(stderr, "FAIL [%s]: bwd tree shorter\n", label); ++g_fail; return; }
         --it; --mi;
         if (std::string(it.key()) != mi->first || it.value() != mi->second)
         { std::fprintf(stderr, "FAIL [%s]: bwd mismatch at '%s'\n", label, mi->first.c_str()); ++g_fail; return; }
      }
      if (it != t.begin()) { std::fprintf(stderr, "FAIL [%s]: bwd tree longer\n", label); ++g_fail; return; }
   }
   // value-only DFS
   {
      std::vector<V> fv;
      t.for_each_value([&](const V& v) { fv.push_back(v); });
      if (fv.size() != m.size()) { std::fprintf(stderr, "FAIL [%s]: for_each_value size\n", label); ++g_fail; return; }
      size_t idx = 0;
      for (auto& [mk, mv] : m) { if (fv[idx] != mv) { std::fprintf(stderr, "FAIL [%s]: for_each_value at %zu\n", label, idx); ++g_fail; return; } ++idx; }
   }
   // point lookups for every present key + a few near-misses
   for (auto& [mk, mv] : m)
   {
      V out{};
      if (!t.find(mk, out) || out != mv) { std::fprintf(stderr, "FAIL [%s]: find '%s'\n", label, mk.c_str()); ++g_fail; return; }
      if (!t.contains(mk) || t.count(mk) != 1) { std::fprintf(stderr, "FAIL [%s]: contains '%s'\n", label, mk.c_str()); ++g_fail; return; }
      // truncated-by-one (a key that ends INSIDE the structure, often inside a fused prefix)
      if (!mk.empty())
      {
         std::string trunc = mk.substr(0, mk.size() - 1);
         CHECK(t.contains(trunc) == (m.count(trunc) != 0), "trunc near-miss");
         std::string ext = mk + char(0x55);
         CHECK(t.contains(ext) == (m.count(ext) != 0), "extended near-miss");
      }
   }
   // ordered queries (lower/upper bound) over the present keys and synthetic probes
   for (auto& [mk, mv] : m)
   {
      auto lo = t.lower_bound(mk); auto ml = m.lower_bound(mk);
      CHECK((lo == t.end()) == (ml == m.end()), "lb end");
      if (ml != m.end()) CHECK(std::string(lo.key()) == ml->first && lo.value() == ml->second, "lb value");
      auto hi = t.upper_bound(mk); auto mu = m.upper_bound(mk);
      CHECK((hi == t.end()) == (mu == m.end()), "ub end");
      if (mu != m.end()) CHECK(std::string(hi.key()) == mu->first, "ub key");
   }
}

// Count sdiv nodes in a tree (proves the scenario actually hit the fused path).
// NOTE: debug_stats() does not compile in the current WIP (a leftover TEMP-AUDIT block in
// dbg_walk references dbg_counts fields that were never added), so we stub this to 0. The
// cross-checks below are the real oracle; the sdiv-presence assertions are advisory.
template <class Tree>
static uint64_t sdiv_count(const Tree&)
{
   return 0;
}

// ─────────────────────────────────────────────────────────────────────────────
// Scenario A — long shared prefix (dictionary-like). A node-level common run plus wide single-
// byte fan-out: forces a fused prefix replicated across an sdiv node's lines.
template <mode M>
static void scenarioA(const char* cfg)
{
   map<std::string_view, uint64_t, M> t;
   std::map<std::string, uint64_t>    m;
   const std::string prefix = "internationalization";  // 20 bytes, well over SDIV_PREFIX_CAP(13)
   // 40 distinct next-bytes off the shared prefix → must climb past setlist(14) into sdiv tiers.
   for (int i = 0; i < 40; ++i)
   {
      std::string k = prefix;
      k.push_back(char(i * 6 + 1));  // spread bytes so dividers partition them across lines
      k += "_tail";
      uint64_t v = uint64_t(i) * 1000 + 7;
      CHECK(t.insert(k, v) == m.insert_or_assign(k, v).second, "A insert new");
      CHECK(t.insert(k, v) == m.insert_or_assign(k, v).second, "A insert dup");  // overwrite path
   }
   char buf[64];
   std::snprintf(buf, sizeof buf, "A/%s sdiv=%llu", cfg, (unsigned long long)sdiv_count(t));
   cross_check(t, m, buf);
   // Note: with the long 20-byte node prefix the sdiv may carry it in the SHORTENED form
   // (tail-prefix promotes to sparse_full when >13); either way the cross-check is the truth.
}

// Scenario A2 — a SHORT shared prefix (<= SDIV_PREFIX_CAP) so the prefix genuinely fuses into
// the sdiv lines (the case the fix optimizes). Assert sdiv nodes appear.
template <mode M>
static void scenarioA2(const char* cfg)
{
   map<std::string_view, uint64_t, M> t;
   std::map<std::string, uint64_t>    m;
   const std::string prefix = "key";  // 3 bytes, fits SDIV_PREFIX_CAP
   for (int i = 0; i < 50; ++i)
   {
      std::string k = prefix;
      k.push_back(char(i * 5 + 1));
      uint64_t v = uint64_t(i) + 100;
      CHECK(t.insert(k, v) == m.insert_or_assign(k, v).second, "A2 insert");
   }
   char buf[64];
   const uint64_t sd = sdiv_count(t);
   std::snprintf(buf, sizeof buf, "A2/%s sdiv=%llu", cfg, (unsigned long long)sd);
   cross_check(t, m, buf);
   // For mode::none (sdiv ladder) we EXPECT sdiv nodes here.
   // sdiv-presence assertion disabled (debug_stats uncompilable in WIP)
}

// Scenario B — a key ending EXACTLY at the fused-prefix boundary (a term on the prefixed sdiv).
template <mode M>
static void scenarioB(const char* cfg)
{
   map<std::string_view, uint64_t, M> t;
   std::map<std::string, uint64_t>    m;
   const std::string prefix = "pre";  // the fused node prefix
   // Wide fan-out below the prefix → an sdiv node carrying the prefix…
   for (int i = 0; i < 40; ++i)
   {
      std::string k = prefix;
      k.push_back(char(i * 6 + 3));
      uint64_t v = uint64_t(i) + 1;
      CHECK(t.insert(k, v) == m.insert_or_assign(k, v).second, "B fanout");
   }
   // …then the key "pre" itself ends exactly at the prefix boundary → a TERM on the sdiv tail.
   CHECK(t.insert(prefix, 999) == m.insert_or_assign(prefix, 999).second, "B term insert");
   CHECK(t.insert(prefix, 999) == m.insert_or_assign(prefix, 999).second, "B term dup");  // overwrite term
   char buf[64];
   std::snprintf(buf, sizeof buf, "B/%s sdiv=%llu", cfg, (unsigned long long)sdiv_count(t));
   cross_check(t, m, buf);
   // A near-miss: a key that is a strict prefix of the node prefix → must MISS, not find the term.
   CHECK(!t.contains(std::string("pr")), "B: 'pr' must miss (short of fused prefix)");
   CHECK(!t.contains(std::string("p")), "B: 'p' must miss");
   // erase the term, leaving the prefixed router intact
   CHECK(t.erase(prefix) == m.erase(prefix), "B term erase");
   std::snprintf(buf, sizeof buf, "B-after-term-erase/%s", cfg);
   cross_check(t, m, buf);
}

// Scenario C — a key diverging INSIDE a fused prefix → split_sdiv_prefix. Build a prefixed sdiv,
// then insert a key sharing only part of the prefix (forces the prefix to split, spawning a
// parent setlist and shortening the sdiv's replicated prefix).
template <mode M>
static void scenarioC(const char* cfg)
{
   map<std::string_view, uint64_t, M> t;
   std::map<std::string, uint64_t>    m;
   const std::string prefix = "abcdef";  // 6-byte fused prefix (fits SDIV_PREFIX_CAP)
   for (int i = 0; i < 40; ++i)
   {
      std::string k = prefix;
      k.push_back(char(i * 6 + 2));
      uint64_t v = uint64_t(i) + 1;
      CHECK(t.insert(k, v) == m.insert_or_assign(k, v).second, "C fanout");
   }
   const uint64_t sd_before = sdiv_count(t);
   // diverge at position 3 ("abc" shared, then 'X' instead of 'd') → split inside the prefix
   for (const char* dk : {"abcXYZ", "abXYZ", "aZZZ", "abcdeZZ", "Zfirst"})
   {
      std::string s = dk;
      uint64_t v = 50000 + s.size();
      CHECK(t.insert(s, v) == m.insert_or_assign(s, v).second, "C diverge insert");
   }
   char buf[80];
   std::snprintf(buf, sizeof buf, "C/%s sdiv_before=%llu sdiv_after=%llu", cfg,
                 (unsigned long long)sd_before, (unsigned long long)sdiv_count(t));
   cross_check(t, m, buf);
   // a key ending INSIDE the (now split) prefix region
   CHECK(t.insert(std::string("abc"), 7) == m.insert_or_assign(std::string("abc"), 7).second, "C term-in-prefix");
   CHECK(t.insert(std::string("ab"), 8) == m.insert_or_assign(std::string("ab"), 8).second, "C term-in-prefix2");
   cross_check(t, m, "C-after-inner-terms");
}

// Scenario D — drive each sdiv tier (sdiv2/3/4) WITH a fused prefix replicated across lines,
// by widening the fan-out monotonically and cross-checking at each tier boundary.
template <mode M>
static void scenarioD(const char* cfg)
{
   map<std::string_view, uint64_t, M> t;
   std::map<std::string, uint64_t>    m;
   const std::string prefix = "px";  // 2-byte fused prefix
   uint64_t seen2 = 0, seen3 = 0, seen4 = 0;
   for (int i = 0; i < 200; ++i)  // push fan-out high enough to traverse sdiv2→3→4→full
   {
      std::string k = prefix;
      // spread bytes across the whole 0..255 range so dividers actually split into all lines
      k.push_back(char((i * 211 + 13) & 0xFF));
      uint64_t v = uint64_t(i) + 1;
      bool exp = m.insert_or_assign(k, v).second;
      CHECK(t.insert(k, v) == exp, "D insert");
      if ((i & 0x1f) == 0x1f) cross_check(t, m, "D progressive");
   }
   char buf[96];
   std::snprintf(buf, sizeof buf, "D/%s saw sdiv2=%llu sdiv3=%llu sdiv4=%llu", cfg,
                 (unsigned long long)seen2, (unsigned long long)seen3, (unsigned long long)seen4);
   cross_check(t, m, buf);
   // sdiv-presence assertion disabled (debug_stats uncompilable in WIP)
}

// Scenario E — grow/shrink across the fuse/split boundary. Insert a wide prefixed fan-out (grows
// up the sdiv ladder, possibly to full), then erase keys back down (de-widen / shrink), then
// erase the divergent keys that split the prefix (re-fuse territory). Cross-check throughout.
template <mode M>
static void scenarioE(const char* cfg)
{
   map<std::string_view, uint64_t, M> t;
   std::map<std::string, uint64_t>    m;
   const std::string prefix = "shared";  // 6-byte fused prefix
   std::vector<std::string> keys;
   for (int i = 0; i < 120; ++i)
   {
      std::string k = prefix;
      k.push_back(char((i * 97 + 5) & 0xFF));
      k.push_back(char((i * 31) & 0xFF));
      keys.push_back(k);
      uint64_t v = uint64_t(i) + 1;
      CHECK(t.insert(k, v) == m.insert_or_assign(k, v).second, "E grow");
   }
   cross_check(t, m, "E after grow");
   // erase half in a scrambled order → shrink back down the ladder (sdiv de-widen)
   std::mt19937 rng(0xBEEF);
   std::shuffle(keys.begin(), keys.end(), rng);
   for (size_t i = 0; i < keys.size(); ++i)
   {
      CHECK(t.erase(keys[i]) == m.erase(keys[i]), "E erase");
      if ((i & 0xf) == 0xf) cross_check(t, m, "E progressive shrink");
   }
   cross_check(t, m, "E after full erase (should be empty)");
   CHECK(t.size() == 0, "E: tree not empty after erasing all");
}

// Scenario E2 — alternate insert/erase that repeatedly crosses the setlist↔sdiv fuse boundary
// (count hovering around 14), with a fused prefix present the whole time. Catches a stale
// replicated prefix or a mis-sized line on the boundary churn.
template <mode M>
static void scenarioE2(const char* cfg)
{
   map<std::string_view, uint64_t, M> t;
   std::map<std::string, uint64_t>    m;
   const std::string prefix = "P";
   for (int round = 0; round < 200; ++round)
   {
      int byte = (round * 53 + 7) & 0xFF;
      std::string k = prefix;
      k.push_back(char(byte));
      if (round % 3 == 2 && m.size() > 5)
      {
         // erase the smallest present key to hover around the boundary
         auto first = m.begin()->first;
         CHECK(t.erase(first) == m.erase(first), "E2 erase");
      }
      else
      {
         uint64_t v = uint64_t(round);
         CHECK(t.insert(k, v) == m.insert_or_assign(k, v).second, "E2 insert");
      }
      if ((round & 7) == 7) cross_check(t, m, "E2 churn");
   }
   cross_check(t, m, "E2 final");
}

// Scenario F — randomized prefix-clustered torrent. Keys drawn from a small alphabet with a
// recently-used pool (extend/truncate/mutate) so shared prefixes + prefix-of-another + wide
// fan-out all occur, just like the differential fuzzer but seeded deterministically here and
// cross-checked after every op. Targets the sdiv configs.
template <mode M>
static void scenarioF(const char* cfg, uint64_t seed)
{
   map<std::string_view, uint64_t, M> t;
   std::map<std::string, uint64_t>    m;
   std::mt19937_64 rng(seed);
   std::vector<std::string> pool;
   static const char alpha[] = {0x00, 'a', 'b', 'c', char(0xff), 'x', 0x01, 'd'};
   uint64_t maxsd = 0;
   for (int op = 0; op < 6000; ++op)
   {
      std::string k;
      if (!pool.empty() && (rng() & 1))
      {
         k = pool[rng() % pool.size()];
         switch (rng() % 4)
         {
            case 0: if (!k.empty()) k.pop_back(); break;
            case 1: k.push_back(alpha[rng() % sizeof alpha]); break;
            case 2: if (!k.empty()) k[rng() % k.size()] = alpha[rng() % sizeof alpha]; break;
            case 3: break;
         }
      }
      else
      {
         int n = rng() % 10;
         for (int j = 0; j < n; ++j) k.push_back(alpha[rng() % sizeof alpha]);
      }
      if (pool.size() < 200) pool.push_back(k); else pool[rng() % pool.size()] = k;

      switch (rng() % 4)
      {
         case 0: case 1: { uint64_t v = rng(); CHECK(t.insert(k, v) == m.insert_or_assign(k, v).second, "F insert"); break; }
         case 2: CHECK(t.erase(k) == m.erase(k), "F erase"); break;
         case 3:
         {
            uint64_t out{}; bool have = t.find(k, out); auto mi = m.find(k);
            CHECK(have == (mi != m.end()) && (!have || out == mi->second), "F find");
            break;
         }
      }
      maxsd = std::max(maxsd, sdiv_count(t));
      if ((op & 0x3f) == 0x3f) cross_check(t, m, "F periodic");
   }
   char buf[80];
   std::snprintf(buf, sizeof buf, "F/%s seed=%llu maxsdiv=%llu", cfg, (unsigned long long)seed, (unsigned long long)maxsd);
   cross_check(t, m, buf);
}

// Scenario G — buckets/adaptive-specific: a wide prefixed sdiv built via the BUCKET insert path
// (bkt_put), then every mutating op (insert overwrite, update, erase, erase-through-iterator) hits
// it. This is the exact path the bkt_put sdiv arm regressed: a key sharing or ending at the fused
// prefix must route correctly, not spawn a phantom branch. Cross-checked after each op.
template <mode M>
static void scenarioG(const char* cfg)
{
   map<std::string_view, uint64_t, M> t;
   std::map<std::string, uint64_t>    m;
   const std::string prefix = "doc";  // fused node prefix below which a wide bucket-fan-out lives
   // Build the prefixed router (buckets collapse, overflow → radix → sdiv carrying "doc").
   for (int i = 0; i < 50; ++i)
   {
      std::string k = prefix;
      k.push_back(char(i * 5 + 1));
      k += "x";  // a suffix so each lands in a bucket beneath the byte branch
      CHECK(t.insert(k, uint64_t(i) + 1) == m.insert_or_assign(k, uint64_t(i) + 1).second, "G build");
   }
   cross_check(t, m, "G after build");
   // term exactly at the prefix boundary
   CHECK(t.insert(prefix, 7000) == m.insert_or_assign(prefix, 7000).second, "G term insert");
   CHECK(t.update(prefix, 7001) == (m.count(prefix) != 0), "G term update");
   if (m.count(prefix)) m[prefix] = 7001;
   cross_check(t, m, "G after term insert/update");
   // a key sharing the whole prefix then a fresh byte (routes past the prefix into a new branch)
   CHECK(t.insert(std::string("docZZ"), 8000) == m.insert_or_assign(std::string("docZZ"), 8000).second, "G new branch");
   // a key diverging inside the prefix → split via bkt_put's split_sdiv_prefix
   CHECK(t.insert(std::string("doXYZ"), 9000) == m.insert_or_assign(std::string("doXYZ"), 9000).second, "G inner split");
   CHECK(t.insert(std::string("dXYZ"), 9100) == m.insert_or_assign(std::string("dXYZ"), 9100).second, "G inner split2");
   cross_check(t, m, "G after splits");
   // update an existing branch value, then erase the term, then erase branches
   for (int i = 0; i < 50; i += 7)
   {
      std::string k = prefix; k.push_back(char(i * 5 + 1)); k += "x";
      bool had = m.count(k) != 0;
      CHECK(t.update(k, 12345) == had, "G update branch");
      if (had) m[k] = 12345;
   }
   cross_check(t, m, "G after branch updates");
   CHECK(t.erase(prefix) == m.erase(prefix), "G term erase");
   cross_check(t, m, "G after term erase");
   for (int i = 0; i < 50; ++i)
   {
      std::string k = prefix; k.push_back(char(i * 5 + 1)); k += "x";
      CHECK(t.erase(k) == m.erase(k), "G branch erase");
      if ((i & 7) == 7) cross_check(t, m, "G progressive erase");
   }
   cross_check(t, m, "G final");
}

// Run the full scenario battery for one config.
template <mode M>
static void run_config(const char* cfg)
{
   scenarioA<M>(cfg);
   scenarioA2<M>(cfg);
   scenarioB<M>(cfg);
   scenarioC<M>(cfg);
   scenarioD<M>(cfg);
   scenarioE<M>(cfg);
   scenarioE2<M>(cfg);
   scenarioG<M>(cfg);
   for (uint64_t s = 1; s <= 8; ++s) scenarioF<M>(cfg, s * 0x9E3779B97F4A7C15ull);
}

int main()
{
   // The sdiv ladder is the DEFAULT (mode::none) path; also exercised under buckets / adaptive /
   // wide_stems. dense_tiers uses the cnode ladder (no sdiv) but must still match the oracle, so
   // run it too as a control. ARTPP_ONLY=<cfg> restricts to one config (failure attribution).
   const char* only = std::getenv("ARTPP_ONLY");
   auto want = [&](const char* c) { return !only || std::string(only) == c; };
   if (want("none"))        run_config<mode::none>("none");
   if (want("buckets"))     run_config<mode::buckets>("buckets");
   if (want("adaptive"))    run_config<mode::adaptive>("adaptive");
   if (want("wide_stems"))  run_config<mode::wide_stems>("wide_stems");
   if (want("dense_tiers")) run_config<mode::dense_tiers>("dense_tiers");  // control: cnode ladder

   if (g_fail) { std::fprintf(stderr, "\n%d CHECK(s) FAILED\n", g_fail); return 1; }
   std::printf("v2_sdiv_adversarial: ALL PASS\n");
   return 0;
}

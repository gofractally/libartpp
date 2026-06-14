// fuzz_map — a robust differential fuzzer for artpp::v2::map with std::map as the oracle. Every
// operation is run on BOTH containers and the results (return value, found value, size, and the
// full ordered contents forward AND backward) are required to match; any divergence aborts so
// libFuzzer / ASan capture the exact input.
//
// Dual-mode from one core (run_dispatch):
//   * libFuzzer target — build with -DARTPP_FUZZER=ON (adds -fsanitize=fuzzer,address,undefined);
//     coverage-guided mutation explores the radix's split / widen-ladder / bucket-burst / de-widen
//     edges far better than blind random.
//   * standalone (the default ctest binary) — no libFuzzer needed: replays corpus files given as
//     argv, else feeds the SAME decoder a seeded stream of random inputs (argv[1] = iteration count).
//
// Coverage knobs that make it find bugs:
//   * the first input byte selects the CONFIG — every mode (flat / dense_tiers / +ladder_c8 /
//     buckets / adaptive / wide_stems), the line_pool allocator, and a non-trivial std::string
//     value (external leaves + real dtor/move) — so one corpus exercises them all.
//   * keys are drawn from a tiny alphabet AND a recently-used key POOL (extend / truncate / mutate
//     a prior key), manufacturing the shared prefixes, prefix-of-another, and wide single-byte
//     fan-out that drive the structural transitions a uniform-random key generator almost never hits.
#include "artpp/v2/map.hpp"
#include "artpp/v2/pool.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

using artpp::v2::map;
using artpp::v2::mode;

// ── input cursor ────────────────────────────────────────────────────────────────
struct Reader
{
   const uint8_t* p;
   const uint8_t* end;
   uint8_t  u8() { return p < end ? *p++ : 0; }
   uint32_t u32() { uint32_t v = 0; for (int i = 0; i < 4; ++i) v = (v << 8) | u8(); return v; }
   uint64_t u64() { uint64_t v = 0; for (int i = 0; i < 8; ++i) v = (v << 8) | u8(); return v; }
   bool     done() const { return p >= end; }
};

// Repro context for a failing op — the input bytes ARE the reproducer (libFuzzer saves them;
// the standalone runner prints them as hex so they can be replayed via argv).
static const uint8_t* g_data = nullptr;
static size_t         g_size = 0;
static int            g_cfg  = -1;
[[noreturn]] static void fail(const char* what, int op)
{
   std::fprintf(stderr, "\nFUZZ DIVERGENCE: %s  (config=%d, op#%d)\n  input[%zu]= ", what, g_cfg, op, g_size);
   for (size_t i = 0; i < g_size; ++i) std::fprintf(stderr, "%02x", g_data[i]);
   std::fprintf(stderr, "\n");
   std::abort();
}
#define FZCHECK(cond, op) do { if (!(cond)) fail(#cond, (op)); } while (0)

// Adversarial key: half the time extend/mutate a recently-seen key (shared prefixes, prefix-of-
// another, wide fan-out at a shared node); else a fresh short key over a tiny alphabet. Includes
// the empty key, 0x00 and 0xFF bytes (boundary stems), and the occasional long key.
// Debug knobs (env): ARTPP_FZ_STRICT=1 cross-checks after every op (pinpoints a corrupting op);
// ARTPP_FZ_MAXLEN caps key length (bypass a known long-key bug to surface others).
static const bool   g_strict = std::getenv("ARTPP_FZ_STRICT") != nullptr;
static const bool   g_trace  = std::getenv("ARTPP_FZ_TRACE") != nullptr;  // log each op (debug)
static const size_t g_maxlen =
    std::getenv("ARTPP_FZ_MAXLEN") ? std::strtoull(std::getenv("ARTPP_FZ_MAXLEN"), nullptr, 10) : ~size_t(0);

static const unsigned char ALPHABET[] = {0x00, 0x01, 0x02, 'a', 'b', 'c', 'd', 0x7f, 0xfe, 0xff};
// `wide` selects the byte distribution: NARROW (a tiny alphabet) clusters keys into shared
// prefixes and small fan-out (setlists, prefix nodes, inline leaves, buckets); WIDE (full 0..255)
// gives the >16-distinct-byte fan-out that drives the widen ladder — setlist→cnode→node_full→c8
// — and de-widen on erase. Per-run choice, so coverage-guided fuzzing explores both regimes.
static std::string gen_key(Reader& r, std::vector<std::string>& pool, bool wide)
{
   auto byte = [&] { return wide ? char(r.u8()) : char(ALPHABET[r.u8() % sizeof ALPHABET]); };
   std::string   k;
   const uint8_t mode_byte = r.u8();
   if (!pool.empty() && (mode_byte & 1))
   {
      k = pool[r.u32() % pool.size()];          // a prior key …
      const uint8_t op = (mode_byte >> 1) & 3;
      if (op == 0 && !k.empty()) k.pop_back();   // … truncated (prefix)
      else if (op == 1) k.push_back(byte());     // … extended (deeper / wider fan-out)
      else if (op == 2 && !k.empty()) k[r.u8() % k.size()] = byte();
      // op == 3: reuse verbatim (exercises the dup-key / overwrite paths)
   }
   else
   {
      int n = (mode_byte >> 1) % 12;             // mostly short; rarely longer for deep chains
      if ((mode_byte & 0x70) == 0x70) n += int(r.u8() % 64);
      if ((mode_byte & 0xe0) == 0xe0) n += int(r.u32() % 4000);  // rare multi-KB key → multi-line terminal
      for (int j = 0; j < n; ++j) k.push_back(byte());           // (capped well under max_key_bytes)
   }
   if (k.size() > g_maxlen) k.resize(g_maxlen);  // optional cap (bypass a long-key bug to find others)
   if (pool.size() < 256) pool.push_back(k);
   else pool[r.u32() % pool.size()] = k;
   return k;
}

template <class V> static V mkval(Reader& r);
template <> uint64_t mkval<uint64_t>(Reader& r) { return r.u64(); }
template <> std::string mkval<std::string>(Reader& r)
{
   std::string s;
   for (int n = r.u8() % 10; n-- > 0;) s.push_back(char(r.u8()));
   return s;
}

// ARTPP_FZ_STRICT=1 cross-checks after EVERY op (vs the periodic default) — slow, but the full
// walk then trips right after the op that corrupts the structure, pinpointing it for debugging.
// Compare the full ordered contents (forward via ++, backward via --/--end()) and size.
template <class Tree, class V>
static void cross_check(const Tree& t, const std::map<std::string, V>& m, int op)
{
   FZCHECK(t.size() == m.size(), op);
   {  // forward
      auto mi = m.begin();
      for (auto it = t.begin(); it != t.end(); ++it, ++mi)
         FZCHECK(mi != m.end() && std::string(it.key()) == mi->first && it.value() == mi->second, op);
      FZCHECK(mi == m.end(), op);
   }
   {  // backward — exercises operator-- and --end()
      auto it = t.end();
      auto mi = m.end();
      while (mi != m.begin())
      {
         FZCHECK(it != t.begin(), op);
         --it;
         --mi;
         FZCHECK(std::string(it.key()) == mi->first && it.value() == mi->second, op);
      }
      FZCHECK(it == t.begin(), op);
   }
   {  // value-only DFS (for_each_value) — a separate walk from the iterator; same values, key order
      std::vector<V> fv;
      t.for_each_value([&](const V& v) { fv.push_back(v); });
      FZCHECK(fv.size() == m.size(), op);
      std::size_t idx = 0;
      for (auto& [mk, mv] : m) { FZCHECK(idx < fv.size() && fv[idx] == mv, op); ++idx; }
   }
}

template <class Tree>
static void run_impl(Tree& t, Reader& r)
{
   using V = typename Tree::mapped_type;
   std::map<std::string, V> m;
   std::vector<std::string> pool;
   const bool               wide = r.u8() & 1;  // per-run: narrow (clustered) vs wide (fan-out) keys
   int                      op   = 0;
   for (; !r.done(); ++op)
   {
      const std::string k      = gen_key(r, pool, wide);
      const uint8_t     choice = r.u8() % 8;
      if (g_trace)
         std::fprintf(stderr, "op#%d choice=%d klen=%zu sz=%zu k0=%d\n", op, choice, k.size(),
                      m.size(), k.empty() ? -1 : (int)(unsigned char)k[0]);
      switch (choice)
      {
         case 0: { V v = mkval<V>(r); FZCHECK(t.insert(k, v) == m.insert_or_assign(k, v).second, op); break; }
         case 1: { V v = mkval<V>(r); FZCHECK(t.emplace(k, v) == m.try_emplace(k, v).second, op); break; }
         case 2:
         {
            V v = mkval<V>(r);
            const bool had = m.count(k) != 0;
            if (had) m[k] = v;
            FZCHECK(t.update(k, v) == had, op);
            break;
         }
         case 3: FZCHECK(t.erase(k) == m.erase(k), op); break;
         case 4:  // erase through an iterator positioned at lower_bound(k)
         {
            auto it = t.lower_bound(k);
            auto mi = m.lower_bound(k);
            FZCHECK((it == t.end()) == (mi == m.end()), op);
            if (mi != m.end())
            {
               it = t.erase(it);
               mi = m.erase(mi);
               FZCHECK((it == t.end()) == (mi == m.end()), op);
               if (mi != m.end()) FZCHECK(std::string(it.key()) == mi->first, op);
            }
            break;
         }
         case 5:  // point lookup three ways
         {
            V          out{};
            const bool have = t.find(k, out);
            auto       mi   = m.find(k);
            FZCHECK(have == (mi != m.end()) && (!have || out == mi->second), op);
            FZCHECK(t.contains(k) == have && t.count(k) == (have ? 1u : 0u), op);
            FZCHECK(t.find_opt(k).has_value() == have, op);
            break;
         }
         case 6:  // ordered queries — key AND value at each bound, plus equal_range consistency
         {
            auto lo = t.lower_bound(k);
            auto ml = m.lower_bound(k);
            FZCHECK((lo == t.end()) == (ml == m.end()), op);
            if (ml != m.end()) FZCHECK(std::string(lo.key()) == ml->first && lo.value() == ml->second, op);
            auto hi = t.upper_bound(k);
            auto mu = m.upper_bound(k);
            FZCHECK((hi == t.end()) == (mu == m.end()), op);
            if (mu != m.end()) FZCHECK(std::string(hi.key()) == mu->first && hi.value() == mu->second, op);
            auto er = t.equal_range(k);  // must coincide with [lower_bound, upper_bound)
            FZCHECK(er.first == lo && er.second == hi, op);
            break;
         }
         case 7:  // at(): value or throw, vs the oracle
         {
            bool threw = false;
            V    tv{};
            try { tv = t.at(k); } catch (const std::out_of_range&) { threw = true; }
            const bool present = m.count(k) != 0;
            FZCHECK(threw == !present, op);
            if (!threw) FZCHECK(tv == m.at(k), op);
            break;
         }
      }
      if (g_strict) cross_check(t, m, op);  // pinpoint the corrupting op (ARTPP_FZ_STRICT=1)
      if ((op & 0x3f) == 0x3f)
      {
         cross_check(t, m, op);  // periodic full-state agreement
         // operator== over this config's node kinds: rebuild the same content in a fresh tree
         // (same config ⇒ same shape) and require equality both ways. (std::allocator configs
         // only — the pool config can't be default-constructed with a live pool.)
         if constexpr (std::is_same_v<typename Tree::allocator_type, std::allocator<V>>)
         {
            Tree clone;
            for (auto& [ck, cv] : m) clone.insert(ck, cv);
            FZCHECK(clone == t && t == clone, op);
         }
      }
      if (r.u8() == 0) { t.clear(); m.clear(); }       // rare reset → exercises teardown + rebuild
   }
   cross_check(t, m, op);  // final agreement
}

// First byte picks the config so one corpus covers every mode + the pool + a non-trivial value.
static void run_dispatch(const uint8_t* data, size_t size)
{
   g_data = data; g_size = size;
   Reader    r{data, data + size};
   const int cfg = (g_cfg = r.u8() % 8);
   using sv  = std::string_view;
   switch (cfg)
   {
      case 0: { map<sv, uint64_t> t;                                   run_impl(t, r); break; }
      case 1: { map<sv, uint64_t, mode::dense_tiers> t;                run_impl(t, r); break; }
      case 2: { map<sv, uint64_t, mode::dense_tiers | mode::ladder_c8> t; run_impl(t, r); break; }
      case 3: { map<sv, uint64_t, mode::buckets> t;                    run_impl(t, r); break; }
      case 4: { map<sv, uint64_t, mode::adaptive> t;                   run_impl(t, r); break; }
      case 5: { map<sv, uint64_t, mode::wide_stems> t;                 run_impl(t, r); break; }
      case 6:  // the line_pool allocator (indexed 4-byte handles)
      {
         using A = artpp::v2::pool_alloc<uint64_t>;
         artpp::v2::line_pool                          pool;
         map<sv, uint64_t, mode::none, A>          t{A{&pool}};
         run_impl(t, r);
         break;
      }
      case 7: { map<sv, std::string> t;                                run_impl(t, r); break; }  // external leaves
   }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size)
{
   run_dispatch(data, size);
   return 0;
}

#ifndef ARTPP_LIBFUZZER
#include <cstdio>
static std::vector<uint8_t> slurp(const char* path)
{
   std::vector<uint8_t> v;
   if (FILE* f = std::fopen(path, "rb"))
   {
      uint8_t buf[4096];
      size_t  n;
      while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) v.insert(v.end(), buf, buf + n);
      std::fclose(f);
   }
   return v;
}
int main(int argc, char** argv)
{
   if (argc > 1 && std::fopen(argv[1], "rb"))  // replay corpus files (libFuzzer-compatible)
   {
      for (int i = 1; i < argc; ++i) { auto b = slurp(argv[i]); run_dispatch(b.data(), b.size()); }
      std::printf("artpp_fuzz: replayed %d input(s) OK\n", argc - 1);
      return 0;
   }
   const std::uint64_t seed  = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 0xA27FAB1Eull;
   const std::size_t   iters = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 20000;
   std::mt19937_64     rng(seed);
   std::vector<uint8_t> buf;
   for (std::size_t i = 0; i < iters; ++i)
   {
      std::size_t len = 16 + rng() % 512;
      if ((rng() & 0x1f) == 0) len += rng() % 8192;  // occasional long input → larger trees
      buf.resize(len);
      for (auto& b : buf) b = uint8_t(rng());
      if (const char* dp = std::getenv("ARTPP_DUMP"))  // capture each input; last survivor = crasher
      { if (FILE* f = std::fopen(dp, "wb")) { std::fwrite(buf.data(), 1, buf.size(), f); std::fclose(f); } }
      run_dispatch(buf.data(), buf.size());
   }
   std::printf("artpp_fuzz: ALL PASS (%zu inputs, seed %llu)\n", iters, (unsigned long long)seed);
   return 0;
}
#endif

// v2_compare — basic perf comparison of the v2 alt-pointer node design vs v1.
//
// The v2 tree here is BULK-BUILT from the sorted key set: tree shape (not build path)
// determines find latency + memory, so this is a faithful measure of the design's thesis
// — 1 cacheline/level descent, compare-as-index routing, dense direct index without waste
// — without the incremental split/promote machinery. Keys are made fixed-length (hence
// prefix-free), so no router carries a terminal in this first cut.
//
// Measures, per workload, for v1 (artpp::map) and v2-proto: bytes/key and find ns/op.
#include "artpp/v2/node_addr.hpp"
#include "artpp/map.hpp"

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <string_view>
#include <vector>

// ───────────────────────── v1 memory: a counting allocator ─────────────────────────
static size_t g_v1_bytes = 0;
template <class T>
struct cnt_alloc
{
   using value_type = T;
   cnt_alloc() = default;
   template <class U> cnt_alloc(const cnt_alloc<U>&) noexcept {}
   T*   allocate(size_t n) { g_v1_bytes += n * sizeof(T); return std::allocator<T>{}.allocate(n); }
   void deallocate(T* p, size_t n) { g_v1_bytes -= n * sizeof(T); std::allocator<T>{}.deallocate(p, n); }
   template <class U> bool operator==(const cnt_alloc<U>&) const noexcept { return true; }
   template <class U> bool operator!=(const cnt_alloc<U>&) const noexcept { return false; }
};

// ───────────────────────────────── v2 prototype ─────────────────────────────────
namespace v2proto
{
   using artpp::v2::child_line;
   using artpp::v2::handle8;
   using artpp::v2::line_count;
   using artpp::v2::subrange_present;
   using artpp::v2::Tier;
   using sv = std::string_view;

   // tag values (4-bit). Routers map to a Tier; LEAF/PREFIX are terminals/compression.
   enum Kind : unsigned { LEAF = 0, PREFIX = 1, SETLIST = 2, SDIV2 = 3, SDIV3 = 4, SDIV4 = 5, SPARSE = 6, FULL = 7, NUL = 0xF };
   static Tier tier_of(unsigned tag)
   {
      switch (tag)
      {
         case SETLIST: return Tier::setlist;
         case SDIV2:   return Tier::sdiv2;
         case SDIV3:   return Tier::sdiv3;
         case SDIV4:   return Tier::sdiv4;
         case SPARSE:  return Tier::sparse_full;
         default:      return Tier::full;
      }
   }
   static bool is_router(unsigned tag) { return tag >= SETLIST && tag <= FULL; }

   static size_t   g_router_bytes = 0, g_term_bytes = 0;
   static handle8  hnull() { handle8 h; std::memset(h.b, 0xFF, 8); return h; }
   static bool     hisnull(handle8 h) { return h.tag() == 0xF; }
   static void*    hptr(handle8 h) { return reinterpret_cast<void*>(h.addr_field() & ~uint64_t(0xF)); }
   static handle8  hmake(void* p, unsigned tag, uint16_t meta) { return handle8::make((reinterpret_cast<uint64_t>(p) & ~uint64_t(0xF)) | tag, meta); }

   // Leaf: [u32 suffix_len][suffix bytes][u64 value]  (16-byte-granular, like v1's term region)
   static handle8 make_leaf(sv suf, uint64_t val)
   {
      size_t   sz = 4 + suf.size() + 8, asz = (sz + 15) & ~size_t(15);
      void*    p  = std::aligned_alloc(16, asz);
      uint32_t n  = uint32_t(suf.size());
      std::memcpy(p, &n, 4);
      std::memcpy(static_cast<char*>(p) + 4, suf.data(), suf.size());
      std::memcpy(static_cast<char*>(p) + 4 + suf.size(), &val, 8);
      g_term_bytes += asz;
      return hmake(p, LEAF, 0);
   }
   // Prefix: [u16 prefix_len][prefix bytes][handle child]   (one 128B line, like v1)
   static handle8 make_prefix(sv pfx, handle8 child)
   {
      size_t sz = 2 + pfx.size() + 8, asz = (sz + 127) & ~size_t(127);
      void*  p  = std::aligned_alloc(128, asz);
      uint16_t n = uint16_t(pfx.size());
      std::memcpy(p, &n, 2);
      std::memcpy(static_cast<char*>(p) + 2, pfx.data(), pfx.size());
      std::memcpy(static_cast<char*>(p) + 2 + pfx.size(), &child, 8);
      g_router_bytes += asz;
      return hmake(p, PREFIX, 0);
   }

   static constexpr int LINE = 128, SL_CAP = 14;  // setlist branches per line
   struct Branch { uint8_t b; handle8 h; };

   // Exact-match over the ≤14 keys. Measured winner here: NEON compare→shrn-movemask→ctz —
   // all 16 lanes compared in parallel, index from ctz (no re-load). Beat the scalar
   // lower_bound (re-load), the auto-vectorized OR-bitfield, SWAR find_byte, AND the
   // single-compare-half-then-unroll-8 (which serializes: read k[8] → base → scan, vs NEON's
   // dependency-free parallel compare). On the latency-bound descent, parallelism wins.
#if defined(__aarch64__)
   [[gnu::always_inline]] inline int sl_find(const char* keys, unsigned cnt, uint8_t byte)
   {
      uint8x16_t eq = vceqq_u8(vld1q_u8(reinterpret_cast<const uint8_t*>(keys)), vdupq_n_u8(byte));
      uint64_t   m  = vget_lane_u64(vreinterpret_u64_u8(vshrn_n_u16(vreinterpretq_u16_u8(eq), 4)), 0);
      m &= (cnt >= 16) ? ~uint64_t(0) : ((uint64_t(1) << (cnt * 4)) - 1);
      return m ? int(__builtin_ctzll(m) >> 2) : -1;
   }
#else
   [[gnu::always_inline]] inline int sl_find(const char* keys, unsigned cnt, uint8_t byte)
   {
      uint32_t bf = 0;
      for (int i = 0; i < 16; ++i) bf |= uint32_t(uint8_t(keys[i]) == byte) << i;
      bf &= (cnt >= 16) ? 0xFFFFu : ((1u << cnt) - 1u);
      return bf ? __builtin_ctz(bf) : -1;
   }
#endif

   // diagnostic: nodes visited (≈ cachelines loaded) during the last counted find pass
   static uint64_t g_hops = 0;

   // Lay branches into a router of `L` setlist lines per the divider scheme; meta = dividers.
   static handle8 make_setlist_router(unsigned tag, uint16_t meta, std::vector<Branch>& br)
   {
      const unsigned L = line_count(tier_of(tag), meta);
      char*          node = static_cast<char*>(std::aligned_alloc(128, size_t(L) * LINE));
      std::memset(node, 0, size_t(L) * LINE);
      g_router_bytes += size_t(L) * LINE;
      // per line: [0]=count [1]=prefix_len, keys at [2..2+count), handles at [16+i*8],
      // optional inline prefix (S1 only) in the slack after the handles.
      for (auto& b : br)
      {
         unsigned ln = child_line(tier_of(tag), meta, b.b);
         char*    line = node + size_t(ln) * LINE;
         uint8_t  c    = uint8_t(line[0]);
         line[2 + c]   = char(b.b);                                  // key byte
         std::memcpy(line + 16 + c * 8, &b.h, 8);                    // handle
         line[0]       = char(c + 1);
      }
      return hmake(node, tag, meta);
   }
   // Lay branches into subrange lines (each = 16 direct slots, byte&0xF). meta = presence.
   static handle8 make_subrange_router(unsigned tag, uint16_t presence, std::vector<Branch>& br)
   {
      const unsigned L = (tag == FULL) ? 16u : unsigned(__builtin_popcount(presence));
      char*          node = static_cast<char*>(std::aligned_alloc(128, size_t(L) * LINE));
      for (size_t i = 0; i < size_t(L) * LINE; i += 8) { handle8 n = hnull(); std::memcpy(node + i, &n, 8); }
      g_router_bytes += size_t(L) * LINE;
      uint16_t pres = (tag == FULL) ? 0xFFFF : presence;
      for (auto& b : br)
      {
         unsigned ln   = child_line(tier_of(tag), pres, b.b);
         char*    line = node + size_t(ln) * LINE;
         std::memcpy(line + (b.b & 0xF) * 8, &b.h, 8);
      }
      return hmake(node, tag, pres);
   }

   // sorted distinct keys -> subtree handle. depth = bytes already consumed.
   static handle8 build(std::vector<sv>& keys, size_t depth)
   {
      if (keys.size() == 1) return make_leaf(keys[0].substr(depth), 0xABCD);
      // longest common prefix beyond depth
      sv     first = keys.front(), last = keys.back();
      size_t c = 0;
      while (depth + c < first.size() && depth + c < last.size() && first[depth + c] == last[depth + c]) ++c;
      if (c)
      {
         handle8 child = build(keys, depth + c);
         if (child.tag() == SETLIST)  // fuse the prefix into the S1 line's post-handle slack if it fits
         {
            char*   node = static_cast<char*>(hptr(child));
            uint8_t cnt  = uint8_t(node[0]);
            size_t  room = (16 + size_t(cnt) * 8 <= LINE) ? size_t(LINE - 16 - cnt * 8) : 0;
            if (c <= room)
            {
               node[1] = uint8_t(c);
               std::memcpy(node + 16 + size_t(cnt) * 8, first.data() + depth, c);
               return child;  // no separate prefix node — one fewer hop
            }
         }
         return make_prefix(first.substr(depth, c), child);
      }

      // group by byte at depth (keys are prefix-free, so none ends here)
      std::vector<Branch> br;
      size_t              i = 0;
      while (i < keys.size())
      {
         uint8_t            b = uint8_t(keys[i][depth]);
         std::vector<sv>    grp;
         while (i < keys.size() && uint8_t(keys[i][depth]) == b) grp.push_back(keys[i++]);
         br.push_back({b, build(grp, depth + 1)});
      }
      const unsigned nb = unsigned(br.size());

      // choose the smallest tier that keeps every setlist line <= SL_CAP; else go dense.
      auto fits = [&](unsigned tag, uint16_t meta) {
         unsigned cnt[4] = {0, 0, 0, 0};
         for (auto& b : br) ++cnt[child_line(tier_of(tag), meta, b.b)];
         for (unsigned k = 0; k < line_count(tier_of(tag), meta); ++k)
            if (cnt[k] > unsigned(SL_CAP)) return false;
         return true;
      };
      if (nb <= unsigned(SL_CAP)) return make_setlist_router(SETLIST, 0, br);
      {
         uint16_t m2 = br[nb / 2].b;
         if (fits(SDIV2, m2)) return make_setlist_router(SDIV2, m2, br);
         uint16_t m3 = uint16_t(br[nb / 3].b | (br[2 * nb / 3].b << 8));
         if (fits(SDIV3, m3)) return make_setlist_router(SDIV3, m3, br);
         // sdiv4: implicit 128 + a median divider in each half
         std::vector<uint8_t> lo, hi;
         for (auto& b : br) (b.b < 128 ? lo : hi).push_back(b.b);
         uint8_t  d0 = lo.empty() ? 0 : lo[lo.size() / 2];
         uint8_t  d1 = hi.empty() ? 128 : hi[hi.size() / 2];
         uint16_t m4 = uint16_t(d0 | (d1 << 8));
         if (fits(SDIV4, m4)) return make_setlist_router(SDIV4, m4, br);
      }
      // dense: presence vector of occupied subranges; FULL if nearly all present
      uint16_t presence = 0;
      for (auto& b : br) presence |= uint16_t(1u << (b.b >> 4));
      unsigned occ = unsigned(__builtin_popcount(presence));
      return make_subrange_router(occ >= 12 ? FULL : SPARSE, presence, br);
   }

   // descend; returns true if key present (and *out = value). The working handle is carried
   // as a uint64 register value (one unaligned 8-byte load per hop), with tag/addr/meta peeled
   // off by shift+mask — no per-field memcpy. Layout: [addr:48 (tag in low 4)][meta:16].
   static constexpr uint64_t ADDR_MASK = 0x0000'FFFF'FFFF'FFF0ull;  // 48-bit addr, tag nibble cleared
   template <bool Count = false>
   static bool find(handle8 root, sv key, uint64_t* out)
   {
      uint64_t raw;
      std::memcpy(&raw, root.b, 8);
      size_t depth = 0;
      for (;;)
      {
         if constexpr (Count) ++g_hops;
         const unsigned tag = unsigned(raw & 0xF);
         char* const    p   = reinterpret_cast<char*>(raw & ADDR_MASK);
         if (tag == LEAF)
         {
            uint32_t n; std::memcpy(&n, p, 4);
            if (key.substr(depth) != sv(p + 4, n)) return false;
            std::memcpy(out, p + 4 + n, 8);
            return true;
         }
         if (tag == PREFIX)
         {
            uint16_t n; std::memcpy(&n, p, 2);
            if (key.size() - depth < n || std::memcmp(p + 2, key.data() + depth, n) != 0) return false;
            depth += n;
            std::memcpy(&raw, p + 2 + n, 8);
            continue;
         }
         if (depth >= key.size()) return false;
         uint8_t        byte = uint8_t(key[depth]);
         const uint16_t meta = uint16_t(raw >> 48);
         const Tier     t    = tier_of(tag);
         char*          line = p + size_t(child_line(t, meta, byte)) * LINE;
         if (tag == SPARSE || tag == FULL)
         {
            if (tag == SPARSE && !subrange_present(meta, byte)) return false;
            std::memcpy(&raw, line + (byte & 0xF) * 8, 8);
            if ((raw & 0xF) == NUL) return false;
            ++depth; continue;
         }
         // setlist line: consume an inline prefix (S1 only — plen>0), then find `byte`
         uint8_t cnt = uint8_t(line[0]), pl = uint8_t(line[1]), bb = byte;
         if (pl)
         {
            if (key.size() - depth < pl || std::memcmp(line + 16 + cnt * 8, key.data() + depth, pl) != 0) return false;
            depth += pl;
            if (depth >= key.size()) return false;
            bb = uint8_t(key[depth]);
         }
         int fi = sl_find(line + 2, cnt, bb);
         if (fi < 0) return false;
         std::memcpy(&raw, line + 16 + size_t(fi) * 8, 8);
         ++depth; continue;
      }
   }
}  // namespace v2proto

// ───────────────────────────────── harness ─────────────────────────────────
static std::vector<std::string> gen(const char* which, size_t N, std::mt19937_64& rng)
{
   std::vector<std::string> v;
   v.reserve(N);
   if (std::string_view(which) == "spread")  // fixed 16-byte random — wide fan-out, prefix-free
      for (size_t i = 0; i < N; ++i) { std::string s(16, 0); for (auto& c : s) c = char(rng()); v.push_back(s); }
   else  // "text": fixed 12-byte lowercase — clustered in a..z, prefix-free
      for (size_t i = 0; i < N; ++i) { std::string s(12, 0); for (auto& c : s) c = char('a' + rng() % 26); v.push_back(s); }
   std::sort(v.begin(), v.end());
   v.erase(std::unique(v.begin(), v.end()), v.end());
   return v;
}

template <class F>
static double bench_ns(const std::vector<std::string>& probe, F&& fn)
{
   double best = 1e30;
   uint64_t sink = 0;
   for (int rep = 0; rep < 7; ++rep)
   {
      auto t0 = std::chrono::steady_clock::now();
      for (auto& k : probe) sink += fn(k);
      auto t1 = std::chrono::steady_clock::now();
      best = std::min(best, std::chrono::duration<double, std::nano>(t1 - t0).count() / probe.size());
   }
   if (sink == 0x123456789) std::printf("");  // keep sink live
   return best;
}

int main(int argc, char** argv)
{
   const size_t N = (argc > 1) ? std::strtoull(argv[1], nullptr, 10) : 2'000'000;
   std::mt19937_64 rng(42);
   for (const char* w : {"spread", "text"})
   {
      auto keys = gen(w, N, rng);
      std::vector<std::string_view> kv(keys.begin(), keys.end());
      std::vector<std::string_view> ksorted = kv;  // gen() already sorted+unique

      // v1
      g_v1_bytes = 0;
      artpp::map<std::string_view, uint64_t, artpp::mode::none, cnt_alloc<uint64_t>> m1;
      for (auto k : kv) m1.insert_or_assign(k, 0xABCD);
      size_t v1_bytes = g_v1_bytes;

      // v2 (bulk build from sorted keys)
      v2proto::g_router_bytes = v2proto::g_term_bytes = 0;
      auto root = v2proto::build(ksorted, 0);
      size_t v2_bytes = v2proto::g_router_bytes + v2proto::g_term_bytes;

      // correctness + node-count diagnostic: every key found, value matches, in both
      v2proto::g_hops = 0;
      for (auto k : kv) {
         uint64_t a = 0, b = 0;
         bool fa = m1.find(k, a), fb = v2proto::find<true>(root, k, &b);
         if (!fa || !fb || a != b) { std::printf("CORRECTNESS FAIL (%s)\n", w); return 1; }
      }
      auto   d1     = m1.debug_stats();
      double v2nodes = double(v2proto::g_hops) / keys.size();
      double v1rh    = d1.terminals ? double(d1.router_hops) / double(d1.terminals) : 0;
      double v1sd    = d1.terminals ? double(d1.sum_depth) / double(d1.terminals) : 0;
      std::printf("   [%-6s] nodes/lookup: v2=%.2f (incl leaf+prefix)   v1 routers/key=%.2f  depth/key=%.2f  prefix_nodes=%llu\n",
                  w, v2nodes, v1rh, v1sd, (unsigned long long)d1.prefix);

      // shuffled probe order
      auto probe = keys;
      std::shuffle(probe.begin(), probe.end(), rng);
      std::vector<std::string_view> pv(probe.begin(), probe.end());

      double v1ns = bench_ns(probe, [&](const std::string& k){ uint64_t v; return m1.find(k, v) ? v : 0; });
      double v2ns = bench_ns(probe, [&](const std::string& k){ uint64_t v; return v2proto::find(root, k, &v) ? v : 0; });

      std::printf("[%-6s N=%zu]  find ns/op: v1=%.1f  v2=%.1f  (%.2fx)   bytes/key: v1=%.1f  v2=%.1f  (%.2fx)\n",
                  w, keys.size(), v1ns, v2ns, v1ns / v2ns,
                  double(v1_bytes) / keys.size(), double(v2_bytes) / keys.size(),
                  double(v1_bytes) / double(v2_bytes));
   }
   return 0;
}

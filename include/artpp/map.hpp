// map<T> — the headerless adaptive radix map of design4.md.
//
// Tiers implemented in this v1:
//   leaf<T>      value_ptr  — suffix (rest of key) + value, one alloc (lazy expansion)
//   prefix_node  prefix_node— long shared prefix + one next pointer (path compression)
//   setlist      setlist_u8 — sparse byte router, 1 cacheline, up to 16 branches (R1)
//   node_full    node_full  — dense direct-index router, full_lines_v lines, skip-CL0 (SFULL)
//
// Routing is off the pointer tag: setlist/prefix read the node (R1); node_full
// computes the segment line from the key byte, PREFETCHES it, then reads the slot
// (skip-CL0). Growth: a full setlist widens to node_full. Mutation is in place
// (no COW). The narrow bitset tiers (r32/64/128, c2/c4/c8) and wide setlists are
// the documented next step; the dispatch and growth scaffolding already fits them.
#pragma once
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <array>
#include <memory>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "artpp/cnode.hpp"
#include "artpp/key_codec.hpp"
#include "artpp/tagged_ptr.hpp"
#include "artpp/value_ops.hpp"

#if defined(__ARM_NEON)
#include <arm_neon.h>
#elif defined(__SSE2__)
#include <emmintrin.h>
#endif

namespace artpp
{
   // Compile-time structural policy — a flag set, combinable with operator|:
   //   map<T>                              plain adaptive-radix (setlist→node_full direct)
   //   compact_map<T, ExpectedN>           capacity hint: auto-selects the cnode ladder when small
   //   map<T, mode::dense_tiers>           force the cnode density ladder on (compact, small map)
   //   map<T, mode::buckets>               last-mile buckets from the root
   enum class mode : unsigned
   {
      none        = 0,        // DEFAULT: plain adaptive radix (setlist→node_full direct). The
                              //   cnode density ladder is OFF here — it is auto-selected for small
                              //   maps via a capacity hint (see compact_map / the ExpectedSize
                              //   template arg) or requested explicitly with mode::dense_tiers.
      buckets     = 1u << 0,  // terminals are last-mile buckets (collapse sparse/deep tails)
      adaptive    = 1u << 1,  // radix that collapses deep-narrow leaf-splits to buckets
      dense_tiers = 1u << 2,  // force the segmented-bitset cnode ladder ON (setlist→c2→c4→full):
                              //   compact sparse routers — a perf-per-byte win for SMALL maps
                              //   (<=~3M). Past the crossover it converges to node_full and only
                              //   adds build churn + pool fragmentation, so it is NOT the default.
      wide_stems  = 1u << 3,  // sparse routers consume 2 bytes/hop (setlist_u16); re-stride if dense
      flat_full   = 1u << 4,  // force the ladder OFF even when a capacity hint would enable it
                              //   (setlist→node_full direct) — for find-only or large maps
      ladder_c8   = 1u << 5,  // add the c8 rung (setlist→c2→c4→c8→full); off by default (c8 is
                              //   ~never the final tier on bimodal data), kept for testing. Only
                              //   meaningful together with the ladder (mode::dense_tiers / a hint).
   };
   constexpr mode operator|(mode a, mode b) noexcept { return mode(unsigned(a) | unsigned(b)); }
   constexpr mode operator&(mode a, mode b) noexcept { return mode(unsigned(a) & unsigned(b)); }
   constexpr bool has_mode(mode set, mode flag) noexcept
   {
      return (unsigned(set) & unsigned(flag)) != 0;
   }

   // Tag for map's O(1) attach constructor — bind a map to an already-populated pool image
   // (a reopened file) by adopting a persisted root handle + count instead of rebuilding.
   struct attach_t { explicit attach_t() = default; };
   inline constexpr attach_t attach{};

namespace detail  // implementation types & node helpers — not part of the public API
{
   // ── sparse router: 1 cacheline, byte[i] -> branch[i] (sorted) ───────────────
   // Carries a short shared prefix INLINE (rh.hdr.prefix_len bytes, in the trailing
   // `prefix` field) so a path-compressed run does NOT cost a separate prefix_node
   // hop — measured: dict shared prefixes avg <2 bytes and 99.9% are <=8, so this
   // removes essentially all prefix_node pointer-chases.
   //
   // INLINE CHILDREN (`inl` byte): a setlist can host small leaf children INSIDE its
   // own 128-byte line instead of as separate terminal allocations. `inl` bit 7 =
   // "inline mode active"; bits 0..6 flag, per branch 0..6, whether branch[i] is an
   // INLINE leaf — its handle then holds a tail-offset (resolve at line+128-offset),
   // not a pool address. The leaf bytes live in the line's free region, which is one
   // contiguous block [&branch[nbranch], line+128): members are ordered so the unused
   // branch slots abut the line tail. Inline payloads pack against the tail (offset
   // 128), growing down, in ascending branch-index order — a CANONICAL layout
   // re-established after every structural change (sl_repack_). A key terminating
   // under an inline branch costs NO extra cacheline — the leaf shares the line the
   // descent already loaded. Only setlist_u8 does this (other tiers leave inl == 0).
   template <class Ptr>
   struct setlist_t
   {
      static constexpr int PREFIX_CAP = 7;   // was 8; the `inl` byte reclaims one, keeping
                                             // CAP at 16 for BOTH handles (no widen-point shift).
                                             // 7 still covers ~all runs; an 8-byte one takes a pn.
      static constexpr int LANES      = 16;  // stem lanes the SIMD scans read; [CAP..LANES) stay 0xFF
      // CAP derived so the struct fits one line WITH the `inl` byte. branch[] is LAST
      // so the unused slots [nbranch..CAP) plus the line tail [sizeof..128) form one
      // contiguous inline region growing down from the tail.
      static constexpr int CAP = std::min(
          LANES, (int(cacheline_bytes) - int(sizeof(router_hdr_t<Ptr>)) - 1 - PREFIX_CAP - LANES) /
                     int(sizeof(Ptr)));  // 16 at both 4B and 6B
      router_hdr_t<Ptr>    rh;
      uint8_t              inl;                 // bit7 inline-active; bits0-6 per-branch inline flag
      uint8_t              bytes[LANES];
      uint8_t              prefix[PREFIX_CAP];   // rh.hdr.prefix_len bytes used (moved before branch)
      Ptr                  branch[CAP];          // LAST: tail past branch[nbranch] is the inline region
   };
   using setlist = setlist_t<packed_ptr>;
   static_assert(sizeof(setlist) <= 128, "setlist fits one cacheline (tail is the inline region)");
   static_assert(offsetof(setlist_t<packed_ptr>, branch) % 1 == 0, "branch handles are unaligned-safe");

   // Stems are kept SORTED ascending; unused lanes are padded with 0xFF (the max
   // byte). That makes this container ordered (lower_bound / range scans work) AND
   // lets the SIMD lower_bound be a clean "count of stems strictly < byte" over all
   // 16 lanes with NO lane-masking — a 0xFF pad is never < byte, so padding never
   // counts. This is heart's `ucc::lower_bound`: count-less-than (not find-first-bit),
   // a tiny scalar add-chain for small N, a NEON horizontal-add otherwise.
   template <class Ptr>
   inline int setlist_lb(const setlist_t<Ptr>* s, uint8_t byte) noexcept
   {
#if defined(__ARM_NEON)
      const uint8x16_t v  = vld1q_u8(s->bytes);
      const uint8x16_t lt = vandq_u8(vcltq_u8(v, vdupq_n_u8(byte)), vdupq_n_u8(1));
      return int(vaddlvq_u8(lt));  // # lanes with stem < byte (padding 0xFF never counts)
#elif defined(__SSE2__)
      const __m128i v  = _mm_loadu_si128(reinterpret_cast<const __m128i*>(s->bytes));
      const __m128i lt = _mm_cmplt_epi8(_mm_xor_si128(v, _mm_set1_epi8(char(0x80))),
                                        _mm_set1_epi8(char(byte ^ 0x80)));  // unsigned via sign-flip
      return std::popcount(unsigned(uint16_t(_mm_movemask_epi8(lt))));
#else
      int c = 0;
      for (int i = 0; i < setlist_t<Ptr>::CAP; ++i)
         c += s->bytes[i] < byte;  // padded 0xFF lanes never satisfy <byte
      return c;
#endif
   }
   // Point lookup: SWAR byte search (heart's find_byte), NOT a NEON movemask. For the
   // tiny setlists path-compressed data produces (avg ~2 children), the SWAR zero-byte
   // trick over two u64 words has far lower latency than a 16-lane vector + two
   // horizontal-adds. Stems are unique and padding is 0xFF, so a match for byte<0xFF
   // is always a real stem; the `i < n` guard rejects a 0xFF lookup hitting padding.
   template <class Ptr>
   inline int setlist_index(const setlist_t<Ptr>* s, uint8_t byte) noexcept
   {
      constexpr uint64_t ONES = 0x0101010101010101ULL, HIGH = 0x8080808080808080ULL;
      const int          n      = s->rh.hdr.nbranch;
      const uint64_t     target = uint64_t(byte) * ONES;
      uint64_t           d, x, m;
      std::memcpy(&d, s->bytes, 8);
      x = d ^ target;
      m = (x - ONES) & ~x & HIGH;
      if (m) { const int i = std::countr_zero(m) >> 3; if (i < n) return i; }
      std::memcpy(&d, s->bytes + 8, 8);
      x = d ^ target;
      m = (x - ONES) & ~x & HIGH;
      if (m) { const int i = 8 + (std::countr_zero(m) >> 3); if (i < n) return i; }
      return -1;
   }
   template <class Ptr>
   inline Ptr setlist_find(const setlist_t<Ptr>* s, uint8_t byte) noexcept
   {
      const int i = setlist_index(s, byte);
      return i >= 0 ? s->branch[i] : Ptr::null();
   }
   // Sorted update-or-insert: replace in place if present, else open a gap at the
   // lower-bound position and insert (keeps stems sorted). False only when full.
   template <class Ptr>
   inline bool setlist_set(setlist_t<Ptr>* s, uint8_t byte, Ptr p) noexcept
   {
      const int n   = s->rh.hdr.nbranch;
      const int idx = setlist_lb(s, byte);
      if (idx < n && s->bytes[idx] == byte)
      {
         s->branch[idx] = p;  // replace
         return true;
      }
      if (n >= setlist_t<Ptr>::CAP)
         return false;
      std::memmove(&s->bytes[idx + 1], &s->bytes[idx], size_t(n - idx));
      std::memmove(&s->branch[idx + 1], &s->branch[idx], size_t(n - idx) * sizeof(Ptr));
      s->bytes[idx]     = byte;
      s->branch[idx]    = p;
      s->rh.hdr.nbranch = uint16_t(n + 1);
      return true;
   }
   // Remove `byte`: close the gap, restore the 0xFF pad in the freed tail lane.
   template <class Ptr>
   inline void setlist_remove(setlist_t<Ptr>* s, uint8_t byte) noexcept
   {
      const int i = setlist_index(s, byte);
      if (i < 0)
         return;
      const int n = s->rh.hdr.nbranch;
      std::memmove(&s->bytes[i], &s->bytes[i + 1], size_t(n - 1 - i));
      std::memmove(&s->branch[i], &s->branch[i + 1], size_t(n - 1 - i) * sizeof(Ptr));
      s->bytes[n - 1]   = 0xFF;  // keep unused lanes padded for the SIMD lower_bound
      s->rh.hdr.nbranch = uint16_t(n - 1);
   }

   // ── wide router: a setlist that consumes TWO key bytes per hop ───────────────
   // Same shape as `setlist` but the stem is a 16-bit big-endian (key[d]<<8|key[d+1]),
   // so a numeric stem compare == lexicographic order on the two bytes. One cacheline
   // holds the header + 14 stems + 14 branches: it consumes 2 key bytes for 1 cacheline
   // touch (0.5 CL/byte — strictly better than two u8 setlists), collapsing two sparse
   // radix levels into one. It is built ONLINE (heart's model) only where fanout stays
   // small: a node that would exceed 14 stems re-strides back to u8 (self-tuning — dense
   // regions stay byte-at-a-time, sparse chains go wide). Term-less by construction at
   // build; carries an optional term (a key ending exactly here). No inline prefix.
   template <class Ptr>
   struct setlist16_t
   {
      static constexpr int LANES = 16;  // stems padded to 16 so the SIMD lower_bound's
                                        // two 8-lane loads never read past the array
      static constexpr int CAP =
          std::min(LANES, (int(cacheline_bytes) - int(sizeof(router_hdr_t<Ptr>)) - LANES * 2) /
                              int(sizeof(Ptr)));  // 14 at 6B
      router_hdr_t<Ptr>    rh;
      uint16_t             stems[LANES];  // sorted ascending; lanes [CAP..16) always 0xFFFF
      Ptr                  branch[CAP];
      uint8_t              pad[128 - int(sizeof(router_hdr_t<Ptr>)) - LANES * 2 - CAP * int(sizeof(Ptr))];
   };
   using setlist16 = setlist16_t<packed_ptr>;
   static_assert(sizeof(setlist16) == 128, "setlist16 is one cacheline");

   // lower_bound: count of stems strictly < q (0xFFFF padding never counts). NEON over
   // two 8-lane u16 vectors; scalar add-chain otherwise. Same count-less-than trick as
   // the u8 setlist, just 16-bit lanes.
   template <class Ptr>
   inline int sl16_lb(const setlist16_t<Ptr>* s, uint16_t q) noexcept
   {
#if defined(__ARM_NEON)
      const uint16x8_t qd = vdupq_n_u16(q);
      const uint16x8_t lo = vandq_u16(vcltq_u16(vld1q_u16(s->stems), qd), vdupq_n_u16(1));
      const uint16x8_t hi = vandq_u16(vcltq_u16(vld1q_u16(s->stems + 8), qd), vdupq_n_u16(1));
      // lanes 14,15 of the high vector are padding (0xFFFF) → never < q, so no masking.
      return int(vaddvq_u16(lo)) + int(vaddvq_u16(hi));
#else
      int c = 0;
      for (int i = 0; i < setlist16_t<Ptr>::CAP; ++i) c += s->stems[i] < q;
      return c;
#endif
   }
   // The 16-bit big-endian stem for the two bytes at s[i], s[i+1] (numeric order == lex order).
   inline uint16_t two_stem(std::string_view s, size_t i) noexcept
   {
      return uint16_t(uint16_t(uint8_t(s[i])) << 8 | uint8_t(s[i + 1]));
   }
   template <class Ptr>
   inline Ptr sl16_find(const setlist16_t<Ptr>* s, uint16_t q) noexcept
   {
      const int i = sl16_lb(s, q);
      return (i < s->rh.hdr.nbranch && s->stems[i] == q) ? s->branch[i] : Ptr::null();
   }
   template <class Ptr>
   inline Ptr* sl16_find_slot(setlist16_t<Ptr>* s, uint16_t q) noexcept
   {
      const int i = sl16_lb(s, q);
      return (i < s->rh.hdr.nbranch && s->stems[i] == q) ? &s->branch[i] : nullptr;
   }
   // Sorted update-or-insert of a 2-byte stem. False only when full (the re-stride signal).
   template <class Ptr>
   inline bool sl16_set(setlist16_t<Ptr>* s, uint16_t q, Ptr p) noexcept
   {
      const int n   = s->rh.hdr.nbranch;
      const int idx = sl16_lb(s, q);
      if (idx < n && s->stems[idx] == q) { s->branch[idx] = p; return true; }
      if (n >= setlist16_t<Ptr>::CAP) return false;
      std::memmove(&s->stems[idx + 1], &s->stems[idx], size_t(n - idx) * 2);
      std::memmove(&s->branch[idx + 1], &s->branch[idx], size_t(n - idx) * sizeof(Ptr));
      s->stems[idx]     = q;
      s->branch[idx]    = p;
      s->rh.hdr.nbranch = uint16_t(n + 1);
      return true;
   }
   template <class Ptr>
   inline void sl16_remove(setlist16_t<Ptr>* s, uint16_t q) noexcept
   {
      const int n   = s->rh.hdr.nbranch;
      const int idx = sl16_lb(s, q);
      if (idx >= n || s->stems[idx] != q) return;
      std::memmove(&s->stems[idx], &s->stems[idx + 1], size_t(n - 1 - idx) * 2);
      std::memmove(&s->branch[idx], &s->branch[idx + 1], size_t(n - 1 - idx) * sizeof(Ptr));
      s->stems[n - 1]   = 0xFFFF;
      s->rh.hdr.nbranch = uint16_t(n - 1);
   }
   template <class Ptr, class F>
   inline void sl16_for_each(const setlist16_t<Ptr>* s, F&& f)  // (stem, branch) ascending
   {
      for (int i = 0; i < s->rh.hdr.nbranch; ++i) f(s->stems[i], s->branch[i]);
   }

   // ── dense router: CL0 header + 13 direct-index lines, 21 pointers/line ──────
   // Direct-index dense node, derived from sizeof(Ptr): as many branches as fit a cacheline
   // beside the 2-byte (nset,rsvd) population counter, and enough lines to cover all 256 bytes.
   template <class Ptr>
   inline constexpr unsigned full_per_line_v = (cacheline_bytes - 2) / unsigned(sizeof(Ptr));  // 21 at 6B
   template <class Ptr>
   inline constexpr unsigned full_lines_v = (256 + full_per_line_v<Ptr> - 1) / full_per_line_v<Ptr>;  // 13 at 6B
   template <class Ptr>
   struct cline_direct_t
   {
      Ptr     ptr[full_per_line_v<Ptr>];
      uint8_t nset;  // population of this line; optional, iteration only
      uint8_t rsvd;
      [[no_unique_address]] pad_t<cacheline_bytes - full_per_line_v<Ptr> * unsigned(sizeof(Ptr)) - 2>
          pad;  // tile to one CL (node_full_t static_asserts it per instantiation)
   };
   using cline_direct = cline_direct_t<packed_ptr>;
   static_assert(sizeof(cline_direct) == 128, "direct line is one cacheline");

   template <class Ptr>
   struct node_full_t
   {
      static_assert(sizeof(cline_direct_t<Ptr>) == cacheline_bytes, "direct line tiles one cacheline");
      // CL0's spare capacity carries the node's inline prefix when one exists: the
      // node is then HANDLE-TAGGED K::pfxd (a hint: "the header holds something you
      // need"), hdr[0] records the node's REAL kind, hdr[1..] the prefix bytes, and
      // rh.hdr.prefix_len the count (6-bit field bounds the cap). The prefix lives in
      // the header cacheline whenever it fits — prefix_node is the OVERFLOW form, not
      // the default. With no prefix the node is tagged K::node_full and descent
      // direct-indexes without ever touching CL0 (the skip-CL0 fast path, unchanged).
      static constexpr unsigned PFX_CAP =
          std::min(63u, cacheline_bytes - unsigned(sizeof(router_hdr_t<Ptr>)) - 1);
      router_hdr_t<Ptr>   rh;
      uint8_t             hdr[cacheline_bytes - sizeof(router_hdr_t<Ptr>)];  // [kind][prefix...]
      cline_direct_t<Ptr> ranges[full_lines_v<Ptr>];  // bytes 0..255, branchfree reciprocal index
   };
   using node_full = node_full_t<packed_ptr>;
   static_assert(sizeof(node_full) == 128 + full_lines_v<packed_ptr> * 128, "node_full = header + lines");

   // byte -> (line, slot) with no divide: line = byte/21, slot = byte%21.
   // 3121 = ceil(2^16/21); exact for all bytes 0..255 (the smaller s=12 magic errs at 251).
   template <class Ptr>
   inline void full_index(uint8_t byte, unsigned& line, unsigned& slot) noexcept
   {
      constexpr unsigned PL = full_per_line_v<Ptr>;
      if constexpr ((PL & (PL - 1)) == 0)  // power-of-2 lines (8B=16, 4B=32) → shift/mask, no magic
      {
         constexpr unsigned SH = unsigned(std::countr_zero(PL));
         line = unsigned(byte) >> SH;
         slot = unsigned(byte) & (PL - 1);
      }
      else  // non-power-of-2: reciprocal-multiply divide (3121 at 21/line), exactness proven 0..255
      {
         constexpr unsigned RM = (1u << 16) / PL + 1;  // ceil(2^16/PL)
         static_assert([] {
            for (unsigned b = 0; b < 256; ++b)
               if ((b * RM) >> 16 != b / PL) return false;
            return true;
         }(), "full_index reciprocal must be exact for every byte");
         line = (unsigned(byte) * RM) >> 16;
         slot = unsigned(byte) - line * PL;
      }
   }
   // Precomputed byte -> byte-offset of ranges[byte/N].ptr[byte%N] from the node base.
   // Collapses the reciprocal + slot multiply on the descent hot path to one L1 load.
   template <class Ptr>
   inline constexpr std::array<uint16_t, 256> make_full_offtab() noexcept
   {
      std::array<uint16_t, 256> t{};
      for (int b = 0; b < 256; ++b)
      {
         const int li = b / int(full_per_line_v<Ptr>), slot = b - li * int(full_per_line_v<Ptr>);
         t[size_t(b)] = uint16_t(128 + li * 128 + slot * int(sizeof(Ptr)));
      }
      return t;
   }
   template <class Ptr>
   inline constexpr std::array<uint16_t, 256> full_offtab_v = make_full_offtab<Ptr>();
   template <class Ptr>
   inline Ptr full_find(const node_full_t<Ptr>* f, uint8_t byte) noexcept
   {
      unsigned li, slot;
      full_index<Ptr>(byte, li, slot);
      return f->ranges[li].ptr[slot];
   }
   template <class Ptr>
   inline void full_set(node_full_t<Ptr>* f, uint8_t byte, Ptr p) noexcept
   {
      unsigned li, slot;
      full_index<Ptr>(byte, li, slot);
      const bool was_null = f->ranges[li].ptr[slot].is_null();
      f->ranges[li].ptr[slot] = p;
      if (was_null && !p.is_null())
      {
         ++f->ranges[li].nset;
         ++f->rh.hdr.nbranch;
      }
      else if (!was_null && p.is_null())
      {
         --f->ranges[li].nset;
         --f->rh.hdr.nbranch;
      }
   }
   template <class Ptr>
   inline void full_init(node_full_t<Ptr>* f) noexcept
   {
      f->rh.hdr  = node_hdr{0, 0, 0};
      f->rh.term = Ptr::null();
      for (auto& line : f->ranges)
      {
         std::memset(line.ptr, 0xFF, sizeof(line.ptr));  // every slot -> null
         line.nset = 0;
         line.rsvd = 0;
      }
   }

   // ── bucket: the "last-mile" terminal — a flat, lexicographically-sorted set of
   // (suffix,value) entries collapsing a sparse/deep radix tail into ONE node.
   //   [ nent | tail_used | div[nent](u16) ‖ slot[nent](u16) | free | …entries… ]
   // div[i] = big-endian first 2 bytes of entry i's suffix (the SIMD lower_bound key;
   // just an accelerator — the full suffix in the entry is the source of truth, so the
   // bucket is correct for arbitrary binary keys). Entries grow from the tail and never
   // move; insert/remove shift only the small u16 index arrays. Overflow → split to radix.
   struct bucket
   {
      static constexpr int PAGE = 512;  // 4 cachelines; fills then splits to radix
      uint16_t             nent;
      uint16_t             tail_used;
      uint8_t              data[PAGE - 4];  // div[]/slot[] from front; entries from back
   };
   static_assert(sizeof(bucket) == 512, "bucket page");

   // ── prefix_node: [u16 plen][Ptr next][u8 prefix[plen]] — layout derives from sizeof(Ptr) ──
   inline uint16_t pn_plen(const char* p) noexcept { uint16_t v; std::memcpy(&v, p, 2); return v; }
   template <class Ptr>
   inline Ptr pn_next(const char* p) noexcept { Ptr r; std::memcpy(r.b, p + 2, sizeof(Ptr)); return r; }
   template <class Ptr>
   inline void pn_set_next(char* p, Ptr n) noexcept { std::memcpy(p + 2, n.b, sizeof(Ptr)); }
   template <class Ptr>
   inline const uint8_t* pn_pfx(const char* p) noexcept
   {
      return reinterpret_cast<const uint8_t*>(p + 2 + sizeof(Ptr));
   }
   template <class Ptr>
   inline uint8_t* pn_pfx(char* p) noexcept { return reinterpret_cast<uint8_t*>(p + 2 + sizeof(Ptr)); }
   template <class Ptr>
   inline size_t pn_size(size_t plen) noexcept { return 2 + sizeof(Ptr) + plen; }

   // ── Stage B: the branch-handle type comes from the Allocator ─────────────────
   // An allocator opts in with a member type `using artpp_handle = packed_ptr_t<N>;`
   // (8 = full pointer, drops the 48-bit VA assumption; 4 = pool index). Default: the
   // 6-byte packed_ptr, so std::allocator trees keep today's exact layout and codegen.
   // Optional allocator hook: bulk image adoption for cross-pool move assignment
   // (see pool_alloc::artpp_adopt). Detected, never required.
   template <class A, class = void>
   struct has_artpp_adopt_ : std::false_type
   {
   };
   template <class A>
   struct has_artpp_adopt_<A, std::void_t<decltype(std::declval<A&>().artpp_adopt(std::declval<const A&>()))>>
       : std::true_type
   {
   };
   // Optional allocator hook: a dedicated 16-byte-granular TERMINAL region (see
   // line_pool). Detected, never required — without it terminals share the node path.
   template <class A, class = void>
   struct has_term_region_ : std::false_type
   {
   };
   template <class A>
   struct has_term_region_<A, std::void_t<decltype(std::declval<A&>().artpp_alloc_term(std::size_t{}))>>
       : std::true_type
   {
   };

   template <class A, class = void>
   struct handle_of { using type = packed_ptr; };
   template <class A>
   struct handle_of<A, std::void_t<typename A::artpp_handle>> { using type = typename A::artpp_handle; };

   // Structural tuning is compile-time policy (zero runtime branch cost; dead modes are
   // discarded). Default = plain adaptive-radix (setlist→node_full). DenseTiers (the cnode
   // density ladder) auto-engages from a small capacity hint (compact_map / ExpectedSize) or
   // mode::dense_tiers — a perf-per-byte win for small maps. Buckets / Adaptive as before.
   template <class Key, class T, mode Mode = mode::none, class Allocator = std::allocator<T>,
             std::size_t ExpectedSize = 0>
   class map
   {
     public:
      using key_type        = Key;
      using mapped_type     = T;
      using value_type      = std::pair<const Key, T>;
      using size_type       = std::size_t;
      using difference_type = std::ptrdiff_t;
      using allocator_type  = Allocator;
      static constexpr mode policy = Mode;

      // Hard limit on the ENCODED key length: every length field in the node formats
      // (leaf suffix/prefix, prefix-node) is 16 bits. Mutating operations throw
      // std::length_error past it; lookups of longer keys simply miss (no stored key
      // can be that long, so a miss is the truthful answer at zero hot-path cost).
      static constexpr size_type max_key_bytes = 65535;

     private:
      // The branch handle: the Allocator's choice of width (default packed_ptr_t<6>). These
      // local spellings shadow the namespace defaults, so every node type inside the tree
      // re-tiles from this one alias; the free helpers deduce Ptr from their node argument.
      using Ptr        = typename handle_of<Allocator>::type;
      using packed_ptr = Ptr;
      using router_hdr = router_hdr_t<Ptr>;
      using setlist    = setlist_t<Ptr>;
      using setlist16  = setlist16_t<Ptr>;
      using node_full  = node_full_t<Ptr>;
      template <int N>
      using cnode = cnode_t<N, Ptr>;

      using codec = key_codec<Key>;
      // Zero-copy codecs (string-like keys) view the key bytes in place — no scratch object
      // on the hot path. Others serialize into a std::string scratch. Detected from an
      // optional `zero_copy` member so existing single-method codecs still work.
      template <class C, class = void>
      struct codec_zc_ : std::false_type {};
      template <class C>
      struct codec_zc_<C, std::void_t<decltype(C::zero_copy)>> : std::bool_constant<C::zero_copy> {};
      static constexpr bool codec_zero_copy = codec_zc_<codec>::value;
      // The key-parameter type, three tiers:
      //  * zero-copy codecs whose Key converts to string_view (std::string,
      //    std::string_view): take std::string_view BY VALUE — map<string,T> and
      //    map<string_view,T> share one signature, and callers pass literals, char*,
      //    strings, or views without ever constructing a temporary Key;
      //  * other cheap trivially-copyable keys (integers): by value — a const-ref
      //    forces a stack spill+reload on the hot lookup path;
      //  * everything else (owning byte vectors, custom types): by const reference.
      static constexpr bool sv_key_ =
          codec_zero_copy && std::is_convertible_v<const Key&, std::string_view>;
      using key_param = std::conditional_t<
          sv_key_, std::string_view,
          std::conditional_t<std::is_trivially_copyable_v<Key> && sizeof(Key) <= 16, Key, const Key&>>;
      struct no_scratch_ {};
      // Serializing codecs may bring their own scratch type (fixed-width codecs use a plain
      // char buffer — no std::string machinery per encode); default is std::string.
      template <class C, class = void>
      struct codec_scratch_ { using type = std::string; };
      template <class C>
      struct codec_scratch_<C, std::void_t<typename C::scratch>> { using type = typename C::scratch; };
      using scratch_t =
          std::conditional_t<codec_zero_copy, no_scratch_, typename codec_scratch_<codec>::type>;
      static std::string_view enc_(key_param k, scratch_t& s)
      {
         if constexpr (sv_key_)               { (void)s; return k; }  // the param IS the bytes
         else if constexpr (codec_zero_copy)  { (void)s; return codec::view(k); }
         else                                 return codec::encode(k, s);
      }

     private:
      // The flag set unpacked into the if-constexpr gates used throughout.
      static constexpr bool Buckets    = has_mode(Mode, mode::buckets);
      static constexpr bool Adaptive   = has_mode(Mode, mode::adaptive);
      // FLAT by default. The cnode density ladder engages when a capacity hint says the map is
      // small enough to be a perf-per-byte win (0 < ExpectedSize <= ladder_capacity_max, the
      // measured pool crossover) OR when requested explicitly (mode::dense_tiers); mode::flat_full
      // forces it off even under a hint. Above the crossover the ladder converges to node_full
      // and only adds build churn + pool fragmentation — so it is never auto-selected there.
      static constexpr std::size_t ladder_capacity_max = 3'000'000;
      static constexpr bool        DenseTiers =
          !has_mode(Mode, mode::flat_full) &&
          (has_mode(Mode, mode::dense_tiers) || (ExpectedSize != 0 && ExpectedSize <= ladder_capacity_max));
      static constexpr bool LadderC8 = has_mode(Mode, mode::ladder_c8);  // add c8 rung (testing)
      static constexpr bool WideStems  = has_mode(Mode, mode::wide_stems);

     public:

      // A value small enough (and trivially copyable) to live in the pointer's bytes
      // instead of a leaf allocation: the handle width minus the 1-byte tag (5 at the
      // default 6-byte handle). Declared up front: it gates the by-reference element-access
      // API and the iterator's value storage (constraints / conditional types parsed before
      // the rest of the class).
      static constexpr size_t inline_cap   = sizeof(Ptr) - 1;
      static constexpr bool   inlineable   = std::is_trivially_copyable_v<T> && sizeof(T) <= inline_cap;
      // Clamped so pack/unpack are instantiation-safe even for non-inlineable T (dead
      // code under the runtime guard); == sizeof(T) on the path that actually inlines.
      static constexpr size_t inline_bytes = sizeof(T) <= inline_cap ? sizeof(T) : inline_cap;
      // A setlist may host small leaf children INSIDE its own line (see setlist_t).
      // Requires trivially-copyable T: inline payloads are relocated by memcpy during
      // repack/externalize, and a leaf's value object then lives in the parent's line.
      // Non-trivial T falls back to external (terminal-region) leaves throughout.
      static constexpr bool inline_children_ok = std::is_trivially_copyable_v<T>;

     private:
      // Per-node allocation. Every node/value blob is a run of 128-byte-aligned blocks taken
      // straight from the tree's Allocator (rebound to a 128-byte unit), so the low 7 bits stay
      // free for the pointer tag and sizes round up to whole blocks. There is no pool: each node
      // is freed individually (free_all_ walks the tree at teardown / on a throwing path), so the
      // allocator — not a hidden arena — owns and accounts for every byte.
      static constexpr size_t node_block_bytes = 128;
      struct alignas(node_block_bytes) node_block { std::byte raw[node_block_bytes]; };
      using node_alloc_t = typename std::allocator_traits<Allocator>::template rebind_alloc<node_block>;
      using node_at      = std::allocator_traits<node_alloc_t>;
      static constexpr size_t nblocks_(size_t n) noexcept { return (n + node_block_bytes - 1) / node_block_bytes; }
      void* node_alloc_(size_t n)  // throws bad_alloc — the strong-guarantee fault point
      {
         node_alloc_t na(alloc_);
         return node_at::allocate(na, nblocks_(n));
      }
      void node_free_(void* p, size_t n) noexcept
      {
         node_alloc_t na(alloc_);
         node_at::deallocate(na, static_cast<node_block*>(p), nblocks_(n));
      }

      // Terminal (leaf) storage: 16-byte-granular. A leaf is pure payload — suffix
      // bytes + value, no branch slots — so it needs only the tag's 4 alignment bits,
      // not a whole line. Allocators with a terminal region (line_pool) get it routed
      // there; others serve 16-byte blocks from their normal path. Either way the
      // terminal working set shrinks up to 8x vs one-line-per-leaf.
      struct alignas(16) term_block
      {
         std::byte raw[16];
      };
      using term_alloc_t = typename std::allocator_traits<Allocator>::template rebind_alloc<term_block>;
      using term_at      = std::allocator_traits<term_alloc_t>;
      static constexpr size_t tunits_(size_t n) noexcept { return (n + 15) / 16; }
      void* term_alloc_(size_t n)  // throws bad_alloc — same fault point as node_alloc_
      {
         if constexpr (has_term_region_<Allocator>::value)
            return alloc_.artpp_alloc_term(n);
         else
         {
            term_alloc_t ta(alloc_);
            return term_at::allocate(ta, tunits_(n));
         }
      }
      void term_free_(void* p, size_t n) noexcept
      {
         if constexpr (has_term_region_<Allocator>::value)
            alloc_.artpp_free_term(p, n);
         else
         {
            term_alloc_t ta(alloc_);
            term_at::deallocate(ta, static_cast<term_block*>(p), tunits_(n));
         }
      }

      // Address-resolution chokepoint. Direct handles compile to exactly the old static code
      // (base_ is an empty member and `this` folds away under inlining — asm-gated). Indexed
      // handles resolve base_ + (28-bit line index << 7); base_ is cached from the allocator
      // by rebase_() whenever alloc_ changes, so the hot path is one shift-add, no indirection.
      static constexpr bool indexed = Ptr::indexed;
      struct no_base_t
      {
      };
      void* deref_(packed_ptr p) const noexcept
      {
         assert(p.tag() != K::value_ptr && "artpp: terminals deref via deref_term_");
         if constexpr (indexed)
            return const_cast<std::byte*>(base_) + ((p.raw() & ~uint64_t(0xF)) << 3);
         else
            return p.ptr();
      }
      // Same resolution for a raw handle value already in a register (the descend hot path).
      const char* deref_raw_(uint64_t raw) const noexcept
      {
         assert(K(raw & 0xF) != K::value_ptr && "artpp: terminals deref via deref_term_raw_");
         if constexpr (indexed)
            return reinterpret_cast<const char*>(base_) + ((raw & ~uint64_t(0xF)) << 3);
         else
            return reinterpret_cast<const char*>(raw & ~uint64_t(0x7f));
      }
      packed_ptr pack_(const void* node, K k) const noexcept
      {
         if constexpr (indexed)
            return packed_ptr::from_off(size_t(static_cast<const std::byte*>(node) - base_), k);
         else
            return packed_ptr::from(node, k);
      }
      // Terminal twins: tag-selected at compile time — every call site sits inside a
      // value_ptr-dispatched arm, so the region choice costs no hot-path branch.
      void* deref_term_(packed_ptr p) const noexcept
      {
         assert(p.tag() == K::value_ptr && "artpp: only leaves live in the terminal region");
         if constexpr (indexed)
            return const_cast<std::byte*>(tbase_) + (p.raw() & ~uint64_t(0xF));
         else
            return p.tptr();
      }
      const char* deref_term_raw_(uint64_t raw) const noexcept
      {
         if constexpr (indexed)
            return reinterpret_cast<const char*>(tbase_) + (raw & ~uint64_t(0xF));
         else
            return reinterpret_cast<const char*>(raw & ~uint64_t(0xF));
      }
      packed_ptr pack_term_(const void* t, K k) const noexcept
      {
         if constexpr (indexed)
            return packed_ptr::from_toff(size_t(static_cast<const std::byte*>(t) - tbase_), k);
         else
            return packed_ptr::from_term(t, k);
      }
      // Cache the allocator's mapping bases (indexed handles only; a no-op for direct).
      void rebase_() noexcept
      {
         if constexpr (indexed)
         {
            base_  = alloc_.artpp_base();
            tbase_ = alloc_.artpp_term_base();
         }
      }

     public:
      map() { rebase_(); }
      explicit map(const Allocator& a) : alloc_(a) { rebase_(); }
      map(const map&)            = delete;
      map& operator=(const map&) = delete;

      // Move construction always steals (no allocation) — noexcept iff the allocator's move
      // is. The source is left valid-empty.
      map(map&& o) noexcept(std::is_nothrow_move_constructible_v<Allocator>)
          : alloc_(std::move(o.alloc_)), root_(o.root_), count_(o.count_)
      {
         rebase_();
         o.root_  = packed_ptr::null();
         o.count_ = 0;
      }
      // Move assignment per [container.requirements]: steal when the allocator propagates or
      // the allocators are equal; otherwise (unequal, non-propagating — e.g. PMR) the foreign
      // allocator can't free our nodes, so move the elements into our own allocator. noexcept
      // exactly when allocators are always equal (the steal path can't throw).
      map& operator=(map&& o) noexcept(std::allocator_traits<Allocator>::is_always_equal::value)
      {
         if (this == &o) return *this;
         using AT = std::allocator_traits<Allocator>;
         if constexpr (AT::propagate_on_container_move_assignment::value)
         {
            clear();
            alloc_ = std::move(o.alloc_);  // take the allocator that owns o's nodes, then the nodes
         }
         else if (alloc_ == o.alloc_)
         {
            clear();  // equal allocators are interchangeable → our alloc_ can free o's nodes
         }
         else
         {
            clear();
            // Three transports, fastest applicable wins:
            //   1. Index-based allocators (line_pool) with trivially-copyable T: the whole
            //      store is base-relative, so adopting the source pool's image is ONE
            //      memcpy — every handle (including the root) stays valid verbatim.
            //   2. Everything else: node-by-node structural clone (move_elements_from_) —
            //      same shape, no descents, no per-key rebuild.
            // Raw-pointer allocators can never bulk-copy (the handles ARE addresses).
            if constexpr (has_artpp_adopt_<Allocator>::value && indexed &&
                          std::is_trivially_copyable_v<T>)
            {
               alloc_.artpp_adopt(o.alloc_);  // bulk image copy into OUR pool
               rebase_();
               root_  = o.root_;  // same line index, resolved against our base
               count_ = o.count_;
               o.clear();  // frees o's structure in o's own pool (trivial T: no dtors)
               return *this;
            }
            move_elements_from_(o);  // foreign allocator can't free our nodes → clone
            o.clear();
            return *this;
         }
         rebase_();
         root_    = o.root_;
         count_   = o.count_;
         o.root_  = packed_ptr::null();
         o.count_ = 0;
         return *this;
      }
      // Attach to an already-populated pool image (e.g. after line_pool reopened a file):
      // adopt a persisted root handle + element count in O(1) — no walk, no rebuild. The pool's
      // bytes already ARE the tree; offset handles resolve against the reopened base after
      // rebase_(). Indexed handles only (a raw-pointer allocator's handles aren't stable across
      // runs). Pair with line_pool::root()/count(); the pool must back exactly this map.
      map(const Allocator& a, attach_t, uint64_t root_handle, size_type count) : alloc_(a), count_(count)
      {
         static_assert(indexed, "map attach is only for indexed (pool-backed) handles");
         rebase_();
         const uint32_t raw = uint32_t(root_handle);
         std::memcpy(root_.b, &raw, sizeof(root_.b));  // line_ptr is 4 bytes: reconstruct the root
      }

      ~map() { free_all_(root_); }  // walk the tree, destroy every T and free every node

      // The root as a raw handle, for persisting via line_pool::checkpoint (indexed pools).
      uint64_t root_handle() const noexcept { return root_.raw(); }
      // Release the structure WITHOUT freeing it: the nodes stay live in the pool (for a
      // file-backed pool, in the file). Use before destroying an attached, already-checkpointed
      // map so close is O(1) instead of walking the whole tree. The map is left empty; the pool
      // image is untouched. (Anonymous pools reclaim everything on munmap regardless.)
      void detach() noexcept
      {
         root_  = packed_ptr::null();
         count_ = 0;
      }

      allocator_type get_allocator() const noexcept { return alloc_; }

      // Insert or update. Returns true if the key was newly inserted.
      bool insert(key_param k, const T& v)
      {
         T tmp(v);
         return insert(k, std::move(tmp));
      }
      bool insert(key_param k, T&& v) { return insert_<true>(k, std::move(v)); }

      // Construct the value from args and insert ONLY if the key is absent (std::map
      // emplace/try_emplace semantics: a present key is left untouched). Returns true iff
      // inserted. The value is constructed from args exactly once, AT its final address
      // (leaf / bucket entry) — and not at all when the key turns out to exist (stronger
      // than std::map, which may construct and discard).
      template <class... Args>
      bool emplace(key_param k, Args&&... args)
      {
         return insert_<false>(k, em_t<Args...>{std::forward_as_tuple(std::forward<Args>(args)...)});
      }
      template <class... Args>
      bool try_emplace(key_param k, Args&&... args)
      {
         return emplace(k, std::forward<Args>(args)...);
      }

     private:
      // The one insert descent. Assign decides the exact-match action: overwrite (insert)
      // or leave untouched (emplace); everything else — splits, widens, growth — is shared.
      template <bool Assign, class VF>
      bool insert_(key_param k, VF&& v)
      {
         scratch_t ks_;
         std::string_view key = enc_(k, ks_);
         if (key.size() > max_key_bytes)  // u16 length fields would silently truncate
            throw std::length_error("artpp::map: encoded key exceeds max_key_bytes (65535)");
         if constexpr (Buckets)  // bucket-mode: terminals are buckets, routers from splits
         {
            bool ins = false;
            root_ = bkt_put<Assign>(root_, key, 0, std::forward<VF>(v), ins);
            count_ += ins;
            return ins;
         }
         // Iterative descent (mirrors find): walk to the insertion point keeping a
         // pointer to the SLOT that holds `cur`, and write only there. No recursion,
         // no per-level function calls, and pass-through levels do zero writes.
         packed_ptr* slot   = &root_;
         [[maybe_unused]] packed_ptr* pslot = nullptr;  // slot holding the parent of *slot (for fusion)
         size_t      depth  = 0;
         [[maybe_unused]] unsigned lowfan = 0;  // routers with <3 branches (deep-narrow signature)
         for (;;)
         {
            packed_ptr cur = *slot;
            if (cur.is_null())
            {
               *slot = make_value_at(key, depth, std::forward<VF>(v));
               ++count_;
               return true;
            }
            const K k = cur.tag();
            // Adaptive: track the deep-narrow signature. Uniform data widens fast
            // (high fanout → lowfan resets), so it never trips; clustered/deep data
            // accumulates low-fanout levels and collapses its leaf-splits to buckets.
            if constexpr (Adaptive)
               if (is_router(k))
               {
                  const unsigned nb = static_cast<router_hdr*>(deref_(cur))->hdr.nbranch;
                  lowfan            = (nb < 3) ? lowfan + 1 : 0;
               }
            // Terminal bucket: only the bucket/adaptive policies ever create one, so the
            // other modes don't pay this tag test on every descend hop.
            if constexpr (Buckets || Adaptive)
               if (k == K::bucket)
               {
                  bucket*   b = static_cast<bucket*>(deref_(cur));
                  const int r = bkt_insert_entry<Assign>(b, key.substr(depth), std::forward<VF>(v));
                  if (r >= 0) { count_ += (r == 1); return r == 1; }
                  *slot = bkt_split(cur);  // overflow → radix; re-descend
                  continue;
               }
            // Hoist the dominant descend: node_full child slot via the offset table
            // (no reciprocal, no router_find_slot re-dispatch).
            if (k == K::node_full)
            {
               char* f = static_cast<char*>(deref_(cur));
               if (depth == key.size())
               {
                  const bool ins = set_or_update_term<Assign>(reinterpret_cast<router_hdr*>(f), std::forward<VF>(v));
                  count_ += ins;
                  return ins;
               }
               const uint8_t byte = uint8_t(key[depth]);
               packed_ptr*   cs   = reinterpret_cast<packed_ptr*>(f + full_offtab_v<Ptr>[byte]);
               if (!cs->is_null())  // existing child -> descend
               {
                  if constexpr (WideStems) pslot = slot;
                  slot  = cs;
                  ++depth;
                  continue;
               }
               full_set(reinterpret_cast<node_full*>(f), byte,
                        make_value_at(key, depth + 1, std::forward<VF>(v)));  // node_full never overflows
               ++count_;
               return true;
            }
            switch (k)
            {
               case K::pfxd:  // consume the in-header prefix, then route as the full hoist
               {
                  char*          f  = static_cast<char*>(deref_(cur));
                  const unsigned pl = reinterpret_cast<router_hdr*>(f)->hdr.prefix_len;
                  const size_t   c  = lcp(
                      std::string_view(reinterpret_cast<const char*>(pfxd_pfx(f)), pl),
                      key.substr(depth));
                  if (c < pl)  // fused prefix diverges → split it
                  {
                     *slot = split_pfxd_prefix(cur, key, depth, c, std::forward<VF>(v), [&](auto&& x) {
                        return make_value_at(key, depth + c + 1, std::forward<decltype(x)>(x));
                     });
                     ++count_;
                     return true;
                  }
                  depth += pl;
                  if (depth == key.size())
                  {
                     const bool ins = set_or_update_term<Assign>(reinterpret_cast<router_hdr*>(f),
                                                                 std::forward<VF>(v));
                     count_ += ins;
                     return ins;
                  }
                  const uint8_t byte = uint8_t(key[depth]);
                  packed_ptr*   cs   = reinterpret_cast<packed_ptr*>(f + full_offtab_v<Ptr>[byte]);
                  if (!cs->is_null())  // existing child -> descend
                  {
                     if constexpr (WideStems) pslot = slot;
                     slot = cs;
                     ++depth;
                     continue;
                  }
                  full_set(reinterpret_cast<node_full*>(f), byte,
                           make_value_at(key, depth + 1, std::forward<VF>(v)));
                  ++count_;
                  return true;
               }
               case K::value_ptr:
               case K::value_inline:
               {
                  bool inserted;
                  if constexpr (Adaptive)
                     if (lowfan >= 2 &&  // deep-narrow leaf collision → bucket (both
                         bkt_fits(key.size() - depth) &&  // suffixes must fit a page)
                         (cur.tag() == K::value_inline ||
                          bkt_fits(leaf_slen(static_cast<const char*>(deref_term_(cur))))))
                     {
                        *slot = collapse_to_bucket<Assign>(cur, key, depth, std::forward<VF>(v), inserted);
                        count_ += inserted;
                        return inserted;
                     }
                  *slot = split_leaf<Assign>(cur, key, depth, std::forward<VF>(v), inserted);
                  // Wide-stems: a split just created a router under the parent — if the parent
                  // is now a sparse 2-level chain, fuse it into one wide u16 node.
                  if constexpr (WideStems)
                     if (inserted && pslot) try_fuse_parent(pslot);
                  count_ += inserted;
                  return inserted;
               }
               case K::prefix_node:
               {
                  char*          P  = static_cast<char*>(deref_(cur));
                  const uint16_t pl = pn_plen(P);
                  // Lazy fusion: a prefix node sitting directly above a node_full migrates
                  // into the full's header cacheline on the first write that touches it —
                  // the dedicated node was only ever the overflow form. Alloc-free (the
                  // make_prefix fusion branch memcpys + retags), then re-dispatch.
                  if (const packed_ptr nx = pn_next<Ptr>(P);
                      (nx.tag() == K::node_full && pl <= node_full::PFX_CAP) ||
                      (nx.tag() == K::pfxd &&
                       pl + pfxd_plen(deref_(nx)) <= node_full::PFX_CAP))
                  {
                     *slot = make_prefix(
                         std::string_view(reinterpret_cast<const char*>(pn_pfx<Ptr>(P)), pl), nx);
                     node_free_(P, pn_size<Ptr>(pl));
                     continue;  // *slot is now the fused node; the pfxd arm consumes
                  }
                  std::string_view PP(reinterpret_cast<const char*>(pn_pfx<Ptr>(P)), pl);
                  const size_t     c = lcp(PP, key.substr(depth));
                  if (c == PP.size())  // whole prefix matched -> descend into next
                  {
                     if constexpr (WideStems) pslot = slot;
                     slot  = reinterpret_cast<packed_ptr*>(P + 2);  // the next-pointer slot
                     depth += PP.size();
                     continue;
                  }
                  *slot = split_prefix(cur, key, depth, c, std::forward<VF>(v), [&](auto&& x) {
                     return make_value_at(key, depth + c + 1, std::forward<decltype(x)>(x));
                  });
                  ++count_;  // a diverging prefix always inserts
                  return true;
               }
               case K::setlist_u8:  // the only tier with an INLINE prefix
               {
                  setlist*       s  = static_cast<setlist*>(deref_(cur));
                  const unsigned pl = s->rh.hdr.prefix_len;
                  if (pl)
                  {
                     const size_t c = lcp(std::string_view(reinterpret_cast<char*>(s->prefix), pl),
                                          key.substr(depth));
                     if (c < pl)  // inline prefix diverges -> split it
                     {
                        *slot = split_setlist_prefix(cur, key, depth, c, std::forward<VF>(v), [&](auto&& x) {
                           return make_value_at(key, depth + c + 1, std::forward<decltype(x)>(x));
                        });
                        ++count_;
                        return true;
                     }
                     depth += pl;  // whole inline prefix consumed
                  }
                  if (depth == key.size())
                  {
                     const bool ins = set_or_update_term<Assign>(&s->rh, std::forward<VF>(v));
                     count_ += ins;
                     return ins;
                  }
                  const uint8_t byte = uint8_t(key[depth]);
                  if (int i = setlist_index(s, byte); i >= 0)  // existing child -> descend
                  {
                     // An inline-leaf child must become a real leaf before descent: the
                     // generic loop (split_leaf) treats *slot as an external value_ptr.
                     if (sl_is_inline(s, i)) sl_externalize_(s, i);  // throws → strong: unchanged
                     if constexpr (WideStems) pslot = slot;
                     slot  = &s->branch[i];
                     ++depth;
                     continue;
                  }
                  // New child. setlist_set shifts branch[] (and would desync the inl bits),
                  // so flatten any inline children first. We do NOT re-compact on growth:
                  // inline is established once at split creation; a setlist that outgrows its
                  // split shape goes external and stays there (re-compacting per insert is
                  // O(n) alloc/free churn that wrecked clustered builds).
                  if (sl_inl_active(s)) sl_externalize_all_(s);  // one-time; throws → strong
                  packed_ptr lf = make_value_at(key, depth + 1, std::forward<VF>(v));
                  if (setlist_set(s, byte, lf))  // room: in place (prefix untouched)
                  {
                     ++count_;
                     return true;
                  }
                  // full -> grow through the widen ladder, re-anchoring *slot each step and
                  // preserving the inline prefix (both handled by grow_router_).
                  build_guard glf{this, &lf};  // widen() can throw; don't orphan the new value
                  grow_router_(slot, cur, byte, lf);
                  glf.release();  // lf is now linked into the widened node
                  ++count_;
                  return true;
               }
               case K::setlist_u16:  // wide router: 2-byte hop, re-stride to u8 when it can't fit
               {
                  setlist16* s = static_cast<setlist16*>(deref_(cur));
                  if (key.size() - depth < 2)
                  {
                     if (depth == key.size())  // key ends exactly here → term
                     {
                        const bool ins = set_or_update_term<Assign>(&s->rh, std::forward<VF>(v));
                        count_ += ins;
                        return ins;
                     }
                     *slot = restride_u16_to_u8(cur);  // 1 byte left: can't form a stem → re-stride, retry
                     continue;
                  }
                  const uint16_t q = two_stem(key, depth);
                  if (packed_ptr* cs = sl16_find_slot(s, q))  // existing stem → descend 2 bytes
                  {
                     if constexpr (WideStems) pslot = slot;
                     slot  = cs;
                     depth += 2;
                     continue;
                  }
                  if (s->rh.hdr.nbranch >= setlist16::CAP)  // full → re-stride, retry (v untouched)
                  {
                     *slot = restride_u16_to_u8(cur);
                     continue;
                  }
                  sl16_set(s, q, make_value_at(key, depth + 2, std::forward<VF>(v)));  // room: in place
                  ++count_;
                  return true;
               }
               case K::c2:  // dense tiers: no inline prefix (carried by a prefix_node)
               case K::c4:
               case K::c8:
               {
                  router_hdr* rh = static_cast<router_hdr*>(deref_(cur));
                  if (depth == key.size())
                  {
                     const bool ins = set_or_update_term<Assign>(rh, std::forward<VF>(v));
                     count_ += ins;
                     return ins;
                  }
                  const uint8_t byte = uint8_t(key[depth]);
                  if (packed_ptr* cs = router_find_slot(cur, byte))  // existing child -> descend
                  {
                     if constexpr (WideStems) pslot = slot;
                     slot  = cs;
                     ++depth;
                     continue;
                  }
                  packed_ptr lf = make_value_at(key, depth + 1, std::forward<VF>(v));
                  build_guard glf{this, &lf};  // grow can throw; don't orphan the new value
                  grow_router_(slot, cur, byte, lf);  // re-anchors *slot across the widen ladder
                  glf.release();
                  ++count_;
                  return true;
               }
               // exhaustive (no default): node_full/bucket descend before the switch and
               // null inserts at the top of the loop, so these can't reach it — but a NEW
               // kind added to K must fail compilation here and get an explicit decision.
               case K::node_full: case K::bucket: case K::null:
                  assert(false && "handled before the switch");
                  __builtin_unreachable();
            }
         }
      }

     public:
      // ── descend_read<Op>: one shared read-descent skeleton ──────────────────────
      // The hoisted switch (node_full / value_ptr-leaf / setlist hoisted ahead of the jump
      // table; tuned by measurement). The router hops are identical for every read; the Op
      // supplies only the terminal/miss actions and the return type, so find/locate_/… are
      // thin Ops over ONE loop — compile-time (template) dispatch, fully inlined, no vtable.
      // Op interface: leaf(p,rem,rlen) inl(raw,at_end) bkt(b,rem,rlen) term(rh) miss().
#define ARTPP_PTR(R)  deref_raw_(R)
#define ARTPP_TPTR(R) deref_term_raw_(R)  /* terminal (16B-region) resolution */
#define ARTPP_NEXT(ND) { depth = (ND); continue; }
#define ARTPP_BODY_VALUE_PTR    return op.leaf(ARTPP_TPTR(raw), kp + depth, klen - depth);
#define ARTPP_BODY_VALUE_INLINE return op.inl(raw, depth == klen);
#define ARTPP_BODY_BUCKET       return op.bkt(reinterpret_cast<const bucket*>(ARTPP_PTR(raw)), kp + depth, klen - depth);
#define ARTPP_BODY_PREFIX { \
   const char* p = ARTPP_PTR(raw); const uint16_t pl = pn_plen(p); \
   if (klen - depth < pl || std::memcmp(pn_pfx<Ptr>(p), kp + depth, pl) != 0) return op.miss(); \
   std::memcpy(&raw, p + 2, sizeof(Ptr)); ARTPP_NEXT(depth + pl); }
#define ARTPP_BODY_SETLIST { \
   const setlist* s = reinterpret_cast<const setlist*>(ARTPP_PTR(raw)); const unsigned pl = s->rh.hdr.prefix_len; \
   if (pl) { if (klen - depth < pl || std::memcmp(s->prefix, kp + depth, pl) != 0) return op.miss(); depth += pl; } \
   if (depth == klen) return op.term(&s->rh); \
   const int sli_ = setlist_index(s, uint8_t(kp[depth])); \
   if (sli_ < 0) return op.miss(); \
   if (sl_is_inline(s, sli_)) /* inline leaf shares this line — no extra cacheline */ \
      return op.leaf(sl_inline_leaf(s, sli_), kp + depth + 1, klen - depth - 1); \
   raw = s->branch[sli_].raw(); ARTPP_NEXT(depth + 1); }
#define ARTPP_BODY_SETLIST16 { \
   const setlist16* s = reinterpret_cast<const setlist16*>(ARTPP_PTR(raw)); \
   if (klen - depth < 2) return depth == klen ? op.term(&s->rh) : op.miss(); \
   const uint16_t q = uint16_t(uint16_t(uint8_t(kp[depth])) << 8 | uint8_t(kp[depth + 1])); \
   raw = sl16_find(s, q).raw(); ARTPP_NEXT(depth + 2); }
#define ARTPP_BODY_FULL { \
   const char* p = ARTPP_PTR(raw); \
   if (depth == klen) return op.term(reinterpret_cast<const router_hdr*>(p)); \
   std::memcpy(&raw, p + full_offtab_v<Ptr>[uint8_t(kp[depth])], sizeof(Ptr)); ARTPP_NEXT(depth + 1); }
/* pfxd: the handle says "consume the in-header prefix first"; the node's real kind is
   hdr[0] (node_full is the only header-capacity kind today, so the body IS the full
   body — offtab offsets are identical). One CL0 load serves prefix + term, and the
   slot line is the NEXT line of the same node (no pointer chase between them). */
#define ARTPP_BODY_PFXD { \
   const char* p = ARTPP_PTR(raw); \
   const unsigned pl = reinterpret_cast<const router_hdr*>(p)->hdr.prefix_len; \
   if (klen - depth < pl || std::memcmp(p + sizeof(router_hdr) + 1, kp + depth, pl) != 0) \
      return op.miss(); \
   depth += pl; \
   if (depth == klen) return op.term(reinterpret_cast<const router_hdr*>(p)); \
   std::memcpy(&raw, p + full_offtab_v<Ptr>[uint8_t(kp[depth])], sizeof(Ptr)); ARTPP_NEXT(depth + 1); }
#define ARTPP_BODY_CN(N) { \
   const char* p = ARTPP_PTR(raw); \
   if (depth == klen) return op.term(reinterpret_cast<const router_hdr*>(p)); \
   raw = cnode_step(reinterpret_cast<const cnode<N>*>(p), uint8_t(kp[depth])).raw(); ARTPP_NEXT(depth + 1); }

      template <class Op>
      [[gnu::always_inline]] auto descend_read(std::string_view key, Op op) const -> decltype(op.miss())
      {
         uint64_t raw = 0;
         std::memcpy(&raw, root_.b, sizeof(Ptr));
         const char* const kp   = key.data();
         const size_t      klen = key.size();
         size_t            depth = 0;
         for (;;)
         {
            const unsigned kk = unsigned(raw & 0xF);
            if (kk == unsigned(K::node_full)) ARTPP_BODY_FULL
            if constexpr (!inlineable) { if (kk == unsigned(K::value_ptr)) ARTPP_BODY_VALUE_PTR }
            if (kk == unsigned(K::setlist_u8)) ARTPP_BODY_SETLIST
            if (kk > K_max) return op.miss();
            switch (static_cast<K>(kk))  // exhaustive, no default → -Wswitch guards a new K
            {
               case K::value_ptr:    ARTPP_BODY_VALUE_PTR
               case K::value_inline: ARTPP_BODY_VALUE_INLINE
               case K::prefix_node:  ARTPP_BODY_PREFIX
               case K::setlist_u8:   ARTPP_BODY_SETLIST
               case K::node_full:    ARTPP_BODY_FULL
               case K::pfxd:         ARTPP_BODY_PFXD
               case K::c2:           ARTPP_BODY_CN(2)
               case K::c4:           ARTPP_BODY_CN(4)
               case K::c8:           ARTPP_BODY_CN(8)
               case K::bucket:       ARTPP_BODY_BUCKET
               case K::setlist_u16:  ARTPP_BODY_SETLIST16
               case K::null:         return op.miss();
            }
            __builtin_unreachable();
         }
      }
#undef ARTPP_NEXT
#undef ARTPP_PTR
#undef ARTPP_TPTR
#undef ARTPP_BODY_VALUE_PTR
#undef ARTPP_BODY_VALUE_INLINE
#undef ARTPP_BODY_PREFIX
#undef ARTPP_BODY_SETLIST
#undef ARTPP_BODY_SETLIST16
#undef ARTPP_BODY_FULL
#undef ARTPP_BODY_PFXD
#undef ARTPP_BODY_CN
#undef ARTPP_BODY_BUCKET

      // Copy-out lookup: writes the value into `out` and returns true if present.
      // (Copy-out, not pointer, because a small value can live INLINE in the pointer bits.)
      bool find(key_param k, T& out) const
      {
         scratch_t        ks_;
         std::string_view key = enc_(k, ks_);
         struct op_t
         {
            const map* self;
            T*             out;
            bool leaf(const char* p, const char* rem, size_t rl) const
            {
               if (std::string_view(reinterpret_cast<const char*>(leaf_suf(p)), leaf_slen(p)) !=
                   std::string_view(rem, rl))
                  return false;
               self->copy_value(*out, leaf_val(p));
               return true;
            }
            bool inl(uint64_t raw, bool at_end) const
            {
               if (!at_end) return false;
               if constexpr (inlineable)  // value_inline exists only for an inlineable T
               {
                  const uint64_t shifted = raw >> 8;
                  std::memcpy(out, &shifted, inline_bytes);
                  return true;
               }
               return false;  // unreachable for a non-inlineable T (no value_inline nodes)
            }
            bool bkt(const bucket* b, const char* rem, size_t rl) const
            {
               return self->bkt_find(b, std::string_view(rem, rl), *out);
            }
            bool term(const router_hdr* rh) const { return self->read_term(rh, *out); }
            bool miss() const { return false; }
         };
         return descend_read(key, op_t{this, &out});
      }
      // Convenience: copy-out as an optional (artpp extension; STL find() below returns an
      // iterator). Named distinctly to avoid clashing with the iterator-returning find.
      std::optional<T> find_opt(key_param k) const
      {
         T v;
         return find(k, v) ? std::optional<T>(std::move(v)) : std::nullopt;
      }
      // Presence test WITHOUT reading the value — find() copies it out (a real cost for
      // allocating T like std::string), a membership test never needs to.
      bool contains(key_param k) const
      {
         scratch_t        ks_;
         std::string_view key = enc_(k, ks_);
         struct op_t
         {
            bool leaf(const char* p, const char* rem, size_t rl) const
            {
               return std::string_view(reinterpret_cast<const char*>(leaf_suf(p)), leaf_slen(p)) ==
                      std::string_view(rem, rl);
            }
            bool inl(uint64_t, bool at_end) const { return at_end; }
            bool bkt(const bucket* b, const char* rem, size_t rl) const
            {
               return bkt_locate(b, std::string_view(rem, rl)) != nullptr;
            }
            bool term(const router_hdr* rh) const { return !rh->term.is_null(); }
            bool miss() const { return false; }
         };
         return descend_read(key, op_t{});
      }

      // ── upsert / update (STL-ish value-write semantics) ─────────────────────────
      // upsert = insert-or-assign: insert if absent, overwrite if present. Returns true
      // iff NEWLY inserted. (artpp's insert already has this behavior; these are the
      // explicit/STL names.)
      bool upsert(key_param k, const T& v) { return insert(k, v); }
      bool upsert(key_param k, T&& v) { return insert(k, std::move(v)); }
      bool insert_or_assign(key_param k, const T& v) { return insert(k, v); }
      bool insert_or_assign(key_param k, T&& v) { return insert(k, std::move(v)); }

      // update = assign ONLY if the key already exists; never inserts. Returns true iff
      // the key was present (and its value overwritten). A single read-style descent —
      // no node creation, no split, no growth.
      bool update(key_param k, const T& v)
      {
         T t(v);
         return update(k, std::move(t));
      }
      // ── descend_mut<Op>: shared MUTATING descent (slot + parent tracking) ────────
      // Like descend_read but threads the writable slot* and the parent/edge so the Op can
      // overwrite or unlink at the terminal. update/remove are thin Ops over this one loop.
      // (insert is NOT here: it restructures mid-descent at every node — split/widen/fuse —
      // so it isn't a terminal visitor.) Op: leaf/inl/term/bkt/miss; exhaustive switch(K).
      template <class Op>
      [[gnu::always_inline]] auto descend_mut(std::string_view key, Op op) -> decltype(op.miss())
      {
         packed_ptr* slot   = &root_;
         packed_ptr  parent = packed_ptr::null();
         uint8_t     pbyte  = 0;
         uint16_t    pstem  = 0;
         size_t      depth  = 0;
         for (;;)
         {
            packed_ptr cur = *slot;
            switch (cur.tag())
            {
               case K::null:         return op.miss();
               case K::value_ptr:
                  return op.leaf(slot, static_cast<char*>(deref_term_(cur)), key.substr(depth), parent, pbyte, pstem);
               case K::value_inline: return op.inl(slot, depth == key.size(), parent, pbyte, pstem);
               case K::bucket:       return op.bkt(slot, cur, key.substr(depth));
               case K::prefix_node:
               {
                  char*          P  = static_cast<char*>(deref_(cur));
                  const uint16_t pl = pn_plen(P);
                  if (key.size() - depth < pl || std::memcmp(pn_pfx<Ptr>(P), key.data() + depth, pl) != 0)
                     return op.miss();
                  parent = cur; slot = reinterpret_cast<packed_ptr*>(P + 2); depth += pl; continue;
               }
               case K::setlist_u16:
               {
                  setlist16* s = static_cast<setlist16*>(deref_(cur));
                  if (key.size() - depth < 2) return depth == key.size() ? op.term(&s->rh) : op.miss();
                  const uint16_t q  = two_stem(key, depth);
                  packed_ptr*    cs = sl16_find_slot(s, q);
                  if (!cs) return op.miss();
                  parent = cur; pstem = q; slot = cs; depth += 2; continue;
               }
               case K::setlist_u8: case K::c2: case K::c4: case K::c8: case K::node_full:
               case K::pfxd:
               {
                  router_hdr* rh = static_cast<router_hdr*>(deref_(cur));
                  const K     ck = cur.tag();
                  if (ck == K::setlist_u8 || ck == K::pfxd)  // kinds with an inline prefix
                  {
                     const unsigned pl = rh->hdr.prefix_len;
                     if (pl)
                     {
                        const uint8_t* pb = ck == K::pfxd
                                                ? pfxd_pfx(rh)
                                                : static_cast<const setlist*>(deref_(cur))->prefix;
                        if (key.size() - depth < pl || std::memcmp(pb, key.data() + depth, pl) != 0)
                           return op.miss();
                        depth += pl;
                     }
                  }
                  if (depth == key.size()) return op.term(rh);
                  const uint8_t byte = uint8_t(key[depth]);
                  packed_ptr*   cs   = router_find_slot(cur, byte);
                  if (!cs) return op.miss();
                  if (ck == K::setlist_u8)  // the matched child may be an inline leaf
                  {
                     setlist*  s2 = static_cast<setlist*>(deref_(cur));
                     const int i  = int(cs - s2->branch);
                     if (sl_is_inline(s2, i))  // terminal in this line — assign in place
                        return op.leaf(cs, sl_inline_leaf(s2, i), key.substr(depth + 1), cur, byte, pstem);
                  }
                  parent = cur; pbyte = byte; slot = cs; ++depth; continue;
               }
            }
         }
      }

      bool update(key_param k, T&& v)
      {
         scratch_t        ks_;
         std::string_view key = enc_(k, ks_);
         struct op_t
         {
            map* self;
            T*       v;
            bool assign(T* p) const  // overwrite an existing value object in place
            {
               *p = std::move(*v);  // assignment, not destroy+construct (see vf_assign)
               return true;
            }
            bool leaf(packed_ptr*, char* L, std::string_view rem, packed_ptr, uint8_t, uint16_t) const
            {
               if (std::string_view(reinterpret_cast<const char*>(leaf_suf(L)), leaf_slen(L)) != rem)
                  return false;
               return assign(leaf_val(L));
            }
            bool inl(packed_ptr* slot, bool at_end, packed_ptr, uint8_t, uint16_t) const
            {
               if (!at_end) return false;
               if constexpr (inlineable) *slot = pack_inline(*v);
               return true;
            }
            bool term(router_hdr* rh) const
            {
               if (rh->term.is_null()) return false;
               if constexpr (inlineable)
                  if (rh->term.tag() == K::value_inline) { rh->term = pack_inline(*v); return true; }
               return assign(leaf_val(static_cast<char*>(self->deref_term_(rh->term))));
            }
            bool bkt(packed_ptr*, packed_ptr cur, std::string_view rem) const
            {
               if constexpr (bucketable)
                  if (const char* e = bkt_locate(static_cast<bucket*>(self->deref_(cur)), rem))
                     return assign(bkt_val(const_cast<char*>(e)));
               return false;
            }
            bool miss() const { return false; }
         };
         return descend_mut(key, op_t{this, &v});
      }

      // ── element access by reference (non-inline T only) ─────────────────────────
      // For non-inlineable T every value is a real, aligned, constructed object in a
      // leaf or bucket entry, so these hand out genuine T& into the tree. The reference
      // is valid until the next insert/upsert/update/remove that restructures the
      // affected path (leaf split, bucket compact/rebuild/split) — a std::vector-like
      // invalidation rule. (Inlineable small-POD T has no addressable object; use the
      // copy-out find() for those — these overloads are constrained off.)
      T* find_ptr(key_param k)
         requires(!inlineable)
      {
         scratch_t ks_;
         return locate_(enc_(k, ks_));
      }
      const T* find_ptr(key_param k) const
         requires(!inlineable)
      {
         scratch_t ks_;
         return locate_(enc_(k, ks_));
      }
      T& at(key_param k)
         requires(!inlineable)
      {
         scratch_t ks_;
         if (T* p = locate_(enc_(k, ks_))) return *p;
         throw std::out_of_range("artpp::map::at: key not found");
      }
      const T& at(key_param k) const
         requires(!inlineable)
      {
         scratch_t ks_;
         if (const T* p = locate_(enc_(k, ks_))) return *p;
         throw std::out_of_range("artpp::map::at: key not found");
      }
      // operator[]: returns the value, inserting a default-constructed one if absent.
      T& operator[](key_param k)
         requires(!inlineable && std::is_default_constructible_v<T>)
      {
         scratch_t        ks_;
         std::string_view key = enc_(k, ks_);
         if (T* p = locate_(key)) return *p;
         insert(k, T{});                 // default-insert (may restructure)
         return *locate_(key);    // re-descend to the now-stable slot (ks_ still valid)
      }

     private:
      // A grown-once byte buffer for assembling keys during a scan/iteration — psitri's
      // _key_buf idea: write a byte / set the length back, no std::string (no NUL writes,
      // no SSO branch) and no per-step allocation. Reallocates only when a key runs deeper
      // than anything seen so far (geometric growth), so steady-state writes are bare
      // stores. view() hands out a string_view over the live bytes.
      struct key_buf
      {
         std::vector<char> b;  // sized on first use — an unread key buffer costs nothing
         size_t            len = 0;
         void ensure(size_t need)
         {
            if (need > b.size())
            {
               size_t cap = b.size() ? b.size() : 256;
               while (cap < need) cap <<= 1;
               b.resize(cap);
            }
         }
         void push(uint8_t c)
         {
            ensure(len + 1);
            b[len++] = char(c);
         }
         void append(const void* p, size_t n)
         {
            ensure(len + n);
            char*       d = b.data() + len;
            const char* s = static_cast<const char*>(p);
            len += n;
            // Suffixes/prefixes are short; inlined byte stores beat a runtime-size memcpy
            // call. Fall back to memcpy only for the rare long copy (long shared prefix).
            if (n > 16) { std::memcpy(d, s, n); return; }
            for (size_t i = 0; i < n; ++i) d[i] = s[i];
         }
         std::string_view view() const noexcept { return std::string_view(b.data(), len); }
      };

      // ── ordered forward iterator ────────────────────────────────────────────────
      // Yields (key, value) in ascending lexicographic key order. Keys aren't stored
      // whole — they're rebuilt along the descent path into key_ (a key_buf: push
      // prefix/byte on the way down, set the length back on backtrack — no std::string
      // churn). key() / *it return a string_view over that buffer. Bucket entries are
      // sorted by full suffix on entry (the bucket's div index orders only by 2 bytes).
     public:
      class const_iterator
      {
         struct Frame  // node + scan cursor; plen/cbase used only once key maintenance is on
         {
            packed_ptr node;
            int        cur;    // router: -1 term-pending then next-byte 0..256; prefix: 0/1; bucket: entry#
            uint32_t   plen;   // key_ length on entry (backtrack point)
            uint32_t   cbase;  // key_ length past this router's inline prefix (term/child base)
         };
         // Descent stack with inline storage: begin()/bound seeks allocate NOTHING for paths
         // up to INLINE frames (each frame consumes >= 1 key byte, so that is every practical
         // tree); deeper long-key paths spill to the heap transparently. Frames are trivially
         // copyable, so copy/move/grow are memcpy.
         struct frame_stack
         {
            static constexpr uint32_t INLINE = 16;
            static_assert(std::is_trivially_copyable_v<Frame>);
            Frame    sbo_[INLINE];
            Frame*   d_   = sbo_;
            uint32_t n_   = 0;
            uint32_t cap_ = INLINE;

            frame_stack() = default;
            frame_stack(const frame_stack& o) { copy_(o); }
            frame_stack(frame_stack&& o) noexcept { steal_(o); }
            frame_stack& operator=(const frame_stack& o)
            {
               if (this != &o) { release_(); copy_(o); }
               return *this;
            }
            frame_stack& operator=(frame_stack&& o) noexcept
            {
               if (this != &o) { release_(); steal_(o); }
               return *this;
            }
            ~frame_stack() { release_(); }

            bool         empty() const noexcept { return n_ == 0; }
            uint32_t     size() const noexcept { return n_; }
            Frame&       operator[](size_t i) noexcept { return d_[i]; }
            const Frame& operator[](size_t i) const noexcept { return d_[i]; }
            Frame&       back() noexcept { return d_[n_ - 1]; }
            Frame*       begin() noexcept { return d_; }
            Frame*       end() noexcept { return d_ + n_; }
            void         pop_back() noexcept { --n_; }
            void         push_back(const Frame& f)
            {
               if (n_ == cap_) grow_();
               d_[n_++] = f;
            }

           private:
            void grow_()
            {
               const uint32_t ncap = cap_ * 2;
               Frame*         nd   = static_cast<Frame*>(::operator new(ncap * sizeof(Frame)));
               std::memcpy(nd, d_, size_t(n_) * sizeof(Frame));
               release_();
               d_   = nd;
               cap_ = ncap;
            }
            void release_() noexcept
            {
               if (d_ != sbo_) ::operator delete(d_);
            }
            void copy_(const frame_stack& o)
            {
               if (o.n_ > INLINE)
               {
                  d_   = static_cast<Frame*>(::operator new(size_t(o.cap_) * sizeof(Frame)));
                  cap_ = o.cap_;
               }
               else
               {
                  d_   = sbo_;
                  cap_ = INLINE;
               }
               n_ = o.n_;
               std::memcpy(d_, o.d_, size_t(n_) * sizeof(Frame));
            }
            void steal_(frame_stack& o) noexcept
            {
               if (o.d_ != o.sbo_)
               {
                  d_   = o.d_;
                  cap_ = o.cap_;
               }
               else
               {
                  d_   = sbo_;
                  cap_ = INLINE;
                  std::memcpy(sbo_, o.sbo_, size_t(o.n_) * sizeof(Frame));
               }
               n_     = o.n_;
               o.d_   = o.sbo_;
               o.cap_ = INLINE;
               o.n_   = 0;
            }
         };

         const map*       t_ = nullptr;  // owning tree: deref base for indexed handles
         mutable frame_stack  stk_;  // mutable: first key() primes plen/cbase
         // Shadows the tree's deref_ so every handle resolution below routes through t_.
         void* deref_(packed_ptr p) const noexcept { return t_->deref_(p); }
         void* deref_term_(packed_ptr p) const noexcept { return t_->deref_term_(p); }
         mutable key_buf            key_;     // current key (maintained once read) / scratch
         mutable bool               maintain_ = false;  // false → ++ skips key work; flips on first key()
         const uint8_t*             sfx_    = nullptr;   // current terminal suffix; null/0 = none
         uint32_t                   sfxlen_ = 0;
         const char*                full_str_ = nullptr;  // set at a full-key leaf → key() is zero-copy
         uint32_t                   full_len_ = 0;
         // Non-inline T: point at the real stored object (no copy → value() is a true
         // reference into the tree). Inline T: keep a local copy (no addressable object).
         std::conditional_t<inlineable, T, const T*> val_{};
         bool                                        end_ = true;

         void set_ptr_(const T* p)
         {
            if constexpr (inlineable)
               copy_value(val_, p);
            else
               val_ = p;
         }
         void emit_(packed_ptr p)  // value from a terminal pointer (value_ptr "" / value_inline)
         {
            if constexpr (inlineable)
               if (p.tag() == K::value_inline) { unpack_inline(p, val_); return; }
            set_ptr_(leaf_val(static_cast<const char*>(deref_term_(p))));
         }
         // Byte-routers only — u16 (2-byte) and non-routers must be handled by the caller
         // BEFORE this (with_router asserts on them; a silent default here is exactly what
         // once hid the lower_bound/u16 miss).
         packed_ptr router_child_(packed_ptr node, uint8_t byte) const
         {
            return t_->with_router(node, [byte](auto* n) noexcept { return router_find(n, byte); });
         }
         bool next_child_(packed_ptr node, int from, uint8_t& ob, packed_ptr& oc) const
         {
            if (from > 255)
               return false;
            return t_->with_router(node,
                                   [&](auto* n) noexcept { return router_next(n, from, ob, oc); });
         }
         // Position the cursor on a setlist's INLINE leaf child (branch i) — the leaf lives
         // in the parent's line, so it can't be resolved from the child handle alone. Mirrors
         // descend_'s value_ptr arm. Caller has already pushed the parent frame.
         void emit_inline_leaf_(const setlist* s, int i, size_t base)
         {
            const char* L = t_->sl_inline_leaf(s, i);
            sfx_          = leaf_suf(L);
            sfxlen_       = leaf_slen(L);
            if (leaf_has_full(L)) { full_str_ = leaf_str(L); full_len_ = leaf_strlen(L); }
            else                    full_str_ = nullptr;
            set_ptr_(leaf_val(L));
            if (maintain_) { key_.len = base; if (!full_str_) key_.append(sfx_, sfxlen_); }
         }
         // true → positioned at a terminal (sets val_ + suffix); false → pushed a frame.
         // `base` is the key length at this child's start; key_ is grown only when
         // maintain_ is on (the eager fast path), otherwise the path in stk_ suffices.
         bool descend_(packed_ptr child, size_t base)
         {
            switch (child.tag())
            {
               case K::value_ptr:
               {
                  const char* L = static_cast<const char*>(deref_term_(child));
                  sfx_          = leaf_suf(L);
                  sfxlen_       = leaf_slen(L);
                  if (leaf_has_full(L)) { full_str_ = leaf_str(L); full_len_ = leaf_strlen(L); }
                  else                    full_str_ = nullptr;
                  set_ptr_(leaf_val(L));
                  // maintain the path in key_ for the NEXT non-full terminal; skip the suffix
                  // append here when this leaf is full (key() will return its zero-copy view).
                  if (maintain_) { key_.len = base; if (!full_str_) key_.append(sfx_, sfxlen_); }
                  return true;
               }
               case K::value_inline:
                  sfx_      = nullptr;
                  sfxlen_   = 0;
                  full_str_ = nullptr;
                  if constexpr (inlineable)
                     unpack_inline(child, val_);
                  if (maintain_) key_.len = base;
                  return true;
               case K::prefix_node:
               {
                  const char* P  = static_cast<const char*>(deref_(child));
                  uint32_t    cb = uint32_t(base);
                  if (maintain_) { key_.len = base; key_.append(pn_pfx<Ptr>(P), pn_plen(P)); cb = uint32_t(key_.len); }
                  stk_.push_back(Frame{child, 0, uint32_t(base), cb});
                  return false;
               }
               case K::bucket:  // index order IS suffix order (see bkt_index_sorted_)
                  stk_.push_back(Frame{child, 0, uint32_t(base), uint32_t(base)});
                  return false;
               default:  // routers (setlist / c2 / c4 / c8 / node_full)
               {
                  const router_hdr* rh = static_cast<const router_hdr*>(deref_(child));
                  uint32_t          cb = uint32_t(base);
                  if (maintain_)
                  {
                     key_.len = base;
                     if (child.tag() == K::setlist_u8 && rh->hdr.prefix_len)
                        key_.append(static_cast<const setlist*>(deref_(child))->prefix, rh->hdr.prefix_len);
                     else if (child.tag() == K::pfxd && rh->hdr.prefix_len)
                        key_.append(pfxd_pfx(rh), rh->hdr.prefix_len);
                     cb = uint32_t(key_.len);
                  }
                  stk_.push_back(Frame{child, -1, uint32_t(base), cb});
                  return false;
               }
            }
         }
         void seek_()  // advance to next terminal; maintains key_ incrementally iff maintain_
         {
            while (!stk_.empty())
            {
               Frame&  f = stk_.back();
               const K k = f.node.tag();
               if (k == K::prefix_node)
               {
                  if (f.cur == 0)
                  {
                     f.cur = 1;
                     descend_min_(pn_next<Ptr>(static_cast<char*>(deref_(f.node))), f.cbase);
                     return;  // descend_min_ always positions at the prefix child's leftmost
                  }
                  if (maintain_) key_.len = f.plen;
                  stk_.pop_back();
                  continue;
               }
               if constexpr (Buckets || Adaptive)  // bucket frames exist only under these modes
                  if (k == K::bucket)
                  {
                     bucket* b = static_cast<bucket*>(deref_(f.node));
                     if (f.cur < b->nent)
                     {
                        const char* e = bkt_entry(b, f.cur++);
                        sfx_          = bkt_suf(e);
                        sfxlen_       = bkt_slen(e);
                        full_str_     = nullptr;
                        set_ptr_(bkt_val(e));
                        if (maintain_) { key_.len = f.cbase; key_.append(sfx_, sfxlen_); }
                        return;
                     }
                     if (maintain_) key_.len = f.plen;
                     stk_.pop_back();
                     continue;
                  }
               if (k == K::setlist_u16)  // wide router: f.cur = -1 (term pending) then stem index
               {
                  const setlist16* s = static_cast<const setlist16*>(deref_(f.node));
                  if (f.cur == -1)
                  {
                     f.cur = 0;
                     if (!s->rh.term.is_null())
                     {
                        sfx_ = nullptr; sfxlen_ = 0; full_str_ = nullptr;
                        emit_(s->rh.term);
                        if (maintain_) key_.len = f.cbase;
                        return;
                     }
                  }
                  if (f.cur < s->rh.hdr.nbranch)  // stems sorted ascending == key order
                  {
                     const uint16_t stem  = s->stems[f.cur];
                     packed_ptr     child = s->branch[f.cur];
                     ++f.cur;
                     if (maintain_)
                     { key_.len = f.cbase; key_.push(uint8_t(stem >> 8)); key_.push(uint8_t(stem & 0xFF)); }
                     descend_min_(child, key_.len);  // tight drill to this stem's leftmost terminal
                     return;
                  }
                  else { if (maintain_) key_.len = f.plen; stk_.pop_back(); }
                  continue;
               }
               const router_hdr* rh = static_cast<const router_hdr*>(deref_(f.node));
               if (f.cur == -1)
               {
                  f.cur = 0;
                  if (!rh->term.is_null())
                  {
                     sfx_ = nullptr; sfxlen_ = 0; full_str_ = nullptr;
                     emit_(rh->term);
                     if (maintain_) key_.len = f.cbase;
                     return;
                  }
               }
               uint8_t    bb;
               packed_ptr child;
               if (next_child_(f.node, f.cur, bb, child))
               {
                  f.cur = bb + 1;
                  if (maintain_) { key_.len = f.cbase; key_.push(bb); }
                  // an inline leaf lives in the parent's line (needs parent context) → position directly;
                  // otherwise drill to the child's leftmost terminal in one tight pass.
                  if (f.node.tag() == K::setlist_u8)
                  {
                     const setlist* s = static_cast<const setlist*>(deref_(f.node));
                     if (s->inl & 0x80)
                     {
                        const int i = setlist_index(s, bb);
                        if (i >= 0 && map::sl_is_inline(s, i)) { emit_inline_leaf_(s, i, key_.len); return; }
                     }
                  }
                  descend_min_(child, key_.len);
                  return;
               }
               else { if (maintain_) key_.len = f.plen; stk_.pop_back(); }
            }
            end_ = true;
         }

         // Directed descent for lower_bound/upper_bound. Follows `key`; returns true if it
         // positioned directly at a qualifying terminal, false if it left the frame stack
         // such that a subsequent seek_() yields the bound (next-greater leftmost, or the
         // next sibling after an all-less subtree). `upper`: strict (first key > key).
         bool descend_lb_(packed_ptr cur, std::string_view key, size_t depth, bool upper)
         {
            const K k = cur.tag();
            if (k == K::value_ptr)
            {
               const char*      L = static_cast<const char*>(deref_term_(cur));
               std::string_view S(reinterpret_cast<const char*>(leaf_suf(L)), leaf_slen(L));
               const int        cmp = S.compare(key.substr(depth));
               if (upper ? cmp > 0 : cmp >= 0) return descend_(cur, 0);  // qualifies → position
               return false;                                            // < target → seek_ from parent
            }
            if (k == K::value_inline)
            {
               // stored key == path (empty suffix). Qualifies only for lower_bound when the
               // target ends here exactly (path == target).
               if (!upper && depth == key.size()) return descend_(cur, 0);
               return false;
            }
            if constexpr (Buckets || Adaptive)  // buckets exist only under these modes
               if (k == K::bucket)
               {
                  bucket* b = static_cast<bucket*>(deref_(cur));
                  const std::string_view R = key.substr(depth);
                  int                    i = 0;
                  for (; i < b->nent; ++i)  // index order is suffix order: first >= / > R
                  {
                     const char*      e = bkt_entry(b, i);
                     std::string_view S(reinterpret_cast<const char*>(bkt_suf(e)), bkt_slen(e));
                     const int        cmp = S.compare(R);
                     if (upper ? cmp > 0 : cmp >= 0) break;
                  }
                  stk_.push_back(Frame{cur, i, 0, 0});  // seek_ yields entry i (or pops if i==nent)
                  return false;
               }
            if (k == K::prefix_node)
            {
               const char*      P = static_cast<const char*>(deref_(cur));
               std::string_view PP(reinterpret_cast<const char*>(pn_pfx<Ptr>(P)), pn_plen(P));
               std::string_view R = key.substr(depth);
               const size_t     c = lcp(PP, R);
               if (c == PP.size())  // prefix matched → descend the single child
               {
                  stk_.push_back(Frame{cur, 1, 0, 0});
                  return descend_lb_(pn_next<Ptr>(const_cast<char*>(P)), key, depth + c, upper);
               }
               if (c == R.size() || uint8_t(PP[c]) > uint8_t(R[c]))
               {
                  descend_(cur, 0);  // subtree > target → leftmost
                  return false;
               }
               stk_.push_back(Frame{cur, 1, 0, 0});  // subtree < target → pop to parent's next
               return false;
            }
            if (k == K::setlist_u16)  // wide router: position on the 2-byte stem lower-bound
            {
               const setlist16* s = static_cast<const setlist16*>(deref_(cur));
               if (depth == key.size())  // key ends here: lower yields term then stems; upper stems only
               {
                  stk_.push_back(Frame{cur, upper ? 0 : -1, 0, 0});
                  return false;
               }
               // <2 bytes left: pad the missing low byte with 0 → first stem with hi >= key[depth]
               const uint16_t q = (key.size() - depth < 2)
                                      ? uint16_t(uint16_t(uint8_t(key[depth])) << 8)
                                      : uint16_t(uint16_t(uint8_t(key[depth])) << 8 | uint8_t(key[depth + 1]));
               const int lb = sl16_lb(s, q);
               if (key.size() - depth >= 2 && lb < s->rh.hdr.nbranch && s->stems[lb] == q)
               {
                  stk_.push_back(Frame{cur, lb + 1, 0, 0});  // exact stem → recurse; resume after it
                  return descend_lb_(s->branch[lb], key, depth + 2, upper);
               }
               stk_.push_back(Frame{cur, lb, 0, 0});  // seek_ yields stems from the lower-bound index
               return false;
            }
            // routers (setlist / c2 / c4 / c8 / node_full / pfxd)
            const router_hdr* rh = static_cast<const router_hdr*>(deref_(cur));
            if ((k == K::setlist_u8 || k == K::pfxd) && rh->hdr.prefix_len)
            {
               const uint8_t* pb = k == K::pfxd ? pfxd_pfx(rh)
                                                : static_cast<const setlist*>(deref_(cur))->prefix;
               std::string_view PP(reinterpret_cast<const char*>(pb), rh->hdr.prefix_len);
               std::string_view R = key.substr(depth);
               const size_t     c = lcp(PP, R);
               if (c < PP.size())
               {
                  if (c == R.size() || uint8_t(PP[c]) > uint8_t(R[c])) { descend_(cur, 0); return false; }
                  stk_.push_back(Frame{cur, 256, 0, 0});  // subtree < target → pop router
                  return false;
               }
               depth += PP.size();
            }
            if (depth == key.size())  // key ends at this router
            {
               // lower: term (== key) then children (>). upper: skip term, children only.
               stk_.push_back(Frame{cur, upper ? 0 : -1, 0, 0});
               return false;
            }
            const uint8_t byte = uint8_t(key[depth]);
            // inline-leaf child of a setlist: a terminal reached here (needs parent context).
            // Mirror the value_ptr arm — push the resume frame, position iff it qualifies.
            if (cur.tag() == K::setlist_u8)
            {
               const setlist* s = static_cast<const setlist*>(deref_(cur));
               if (s->inl & 0x80)
               {
                  const int i = setlist_index(s, byte);
                  if (i >= 0 && map::sl_is_inline(s, i))
                  {
                     stk_.push_back(Frame{cur, byte + 1, 0, 0});  // seek_/retreat_ resume here
                     const char*      L = t_->sl_inline_leaf(s, i);
                     std::string_view S(reinterpret_cast<const char*>(leaf_suf(L)), leaf_slen(L));
                     const int        cmp = S.compare(key.substr(depth + 1));
                     if (upper ? cmp > 0 : cmp >= 0) { emit_inline_leaf_(s, i, 0); return true; }
                     return false;  // < target → seek_ from the frame yields the bound
                  }
               }
            }
            const packed_ptr child = router_child_(cur, byte);
            if (!child.is_null())  // exact child → follow the key deeper
            {
               stk_.push_back(Frame{cur, byte + 1, 0, 0});
               return descend_lb_(child, key, depth + 1, upper);
            }
            stk_.push_back(Frame{cur, byte, 0, 0});  // no exact child → seek_ takes next-greater leftmost
            return false;
         }

         // ── reverse stepping ───────────────────────────────────────────────────
         bool prev_child_(packed_ptr node, int from, uint8_t& ob, packed_ptr& oc) const
         {
            if (from < 0)
               return false;
            return t_->with_router(node,
                                   [&](auto* n) noexcept { return router_prev(n, from, ob, oc); });
         }
         // Position at the RIGHTMOST terminal under `child` (the mirror of descend_ +
         // seek_: order at a node is [term] < children ascending, so rightmost = last
         // child's rightmost, falling back to the term). Frames are left in the same
         // consumed-cursor convention seek_/key_bytes expect (cur-1 = taken branch).
         void descend_max_(packed_ptr child, size_t base)
         {
            for (;;)
            {
               switch (child.tag())
               {
                  case K::value_ptr:
                  {
                     const char* L = static_cast<const char*>(deref_term_(child));
                     sfx_          = leaf_suf(L);
                     sfxlen_       = leaf_slen(L);
                     if (leaf_has_full(L)) { full_str_ = leaf_str(L); full_len_ = leaf_strlen(L); }
                     else                    full_str_ = nullptr;
                     set_ptr_(leaf_val(L));
                     if (maintain_) { key_.len = base; if (!full_str_) key_.append(sfx_, sfxlen_); }
                     return;
                  }
                  case K::value_inline:
                     sfx_ = nullptr; sfxlen_ = 0; full_str_ = nullptr;
                     if constexpr (inlineable)
                        unpack_inline(child, val_);
                     if (maintain_) key_.len = base;
                     return;
                  case K::prefix_node:
                  {
                     const char* P  = static_cast<const char*>(deref_(child));
                     uint32_t    cb = uint32_t(base);
                     if (maintain_) { key_.len = base; key_.append(pn_pfx<Ptr>(P), pn_plen(P)); cb = uint32_t(key_.len); }
                     stk_.push_back(Frame{child, 1, uint32_t(base), cb});  // single child: consumed
                     child = pn_next<Ptr>(const_cast<char*>(P));
                     base  = cb;
                     continue;
                  }
                  case K::bucket:
                  {
                     bucket*   b = static_cast<bucket*>(deref_(child));
                     const int n = b->nent;  // index order is suffix order: max = last
                     stk_.push_back(Frame{child, n, uint32_t(base), uint32_t(base)});
                     const char* e = bkt_entry(b, n - 1);
                     sfx_      = bkt_suf(e);
                     sfxlen_   = bkt_slen(e);
                     full_str_ = nullptr;
                     set_ptr_(bkt_val(e));
                     if (maintain_) { key_.len = base; key_.append(sfx_, sfxlen_); }
                     return;
                  }
                  case K::setlist_u16:
                  {
                     const setlist16* s  = static_cast<const setlist16*>(deref_(child));
                     const int        nb = s->rh.hdr.nbranch;
                     uint32_t         cb = uint32_t(base);
                     if (maintain_) { key_.len = base; cb = uint32_t(key_.len); }
                     if (nb > 0)
                     {
                        stk_.push_back(Frame{child, nb, uint32_t(base), cb});
                        const uint16_t st = s->stems[nb - 1];
                        if (maintain_)
                        { key_.push(uint8_t(st >> 8)); key_.push(uint8_t(st & 0xFF)); }
                        child = s->branch[nb - 1];
                        base  = maintain_ ? size_t(key_.len) : size_t(cb) + 2;
                        continue;
                     }
                     stk_.push_back(Frame{child, 0, uint32_t(base), cb});  // term-only router
                     sfx_ = nullptr; sfxlen_ = 0; full_str_ = nullptr;
                     emit_(s->rh.term);
                     if (maintain_) key_.len = cb;
                     return;
                  }
                  default:  // byte routers (setlist / c2 / c4 / c8 / node_full)
                  {
                     const router_hdr* rh = static_cast<const router_hdr*>(deref_(child));
                     uint32_t          cb = uint32_t(base);
                     if (maintain_)
                     {
                        key_.len = base;
                        if (child.tag() == K::setlist_u8 && rh->hdr.prefix_len)
                           key_.append(static_cast<const setlist*>(deref_(child))->prefix,
                                       rh->hdr.prefix_len);
                        else if (child.tag() == K::pfxd && rh->hdr.prefix_len)
                           key_.append(pfxd_pfx(rh), rh->hdr.prefix_len);
                        cb = uint32_t(key_.len);
                     }
                     uint8_t    bb;
                     packed_ptr nc;
                     if (prev_child_(child, 255, bb, nc))
                     {
                        stk_.push_back(Frame{child, int(bb) + 1, uint32_t(base), cb});
                        if (maintain_) key_.push(bb);
                        if (child.tag() == K::setlist_u8)  // rightmost child may be inline
                        {
                           const setlist* s = static_cast<const setlist*>(deref_(child));
                           if (s->inl & 0x80)
                           {
                              const int i = setlist_index(s, bb);
                              if (i >= 0 && map::sl_is_inline(s, i))
                              { emit_inline_leaf_(s, i, maintain_ ? size_t(key_.len) : 0); return; }
                           }
                        }
                        child = nc;
                        base  = maintain_ ? size_t(key_.len) : size_t(cb) + 1;
                        continue;
                     }
                     stk_.push_back(Frame{child, 0, uint32_t(base), cb});  // childless: term only
                     sfx_ = nullptr; sfxlen_ = 0; full_str_ = nullptr;
                     emit_(rh->term);
                     if (maintain_) key_.len = cb;
                     return;
                  }
               }
            }
         }
         // Position at the LEFTMOST terminal under `child` — the forward mirror of
         // descend_max_. Order at a router is [term] < children-ascending, so the
         // leftmost is the term when present, else the first child's leftmost. Frames
         // are left in seek_'s consumed-cursor convention (cur = next branch to examine)
         // so a subsequent operator++ resumes correctly. Always positions: every subtree
         // holds a terminal. This is the tight counterpart to descend_() + the seek_()
         // loop — one switch + one first-child probe per level, no frame pop/re-dispatch.
         void descend_min_(packed_ptr child, size_t base)
         {
            for (;;)
            {
               switch (child.tag())
               {
                  case K::value_ptr:
                  {
                     const char* L = static_cast<const char*>(deref_term_(child));
                     sfx_          = leaf_suf(L);
                     sfxlen_       = leaf_slen(L);
                     if (leaf_has_full(L)) { full_str_ = leaf_str(L); full_len_ = leaf_strlen(L); }
                     else                    full_str_ = nullptr;
                     set_ptr_(leaf_val(L));
                     if (maintain_) { key_.len = base; if (!full_str_) key_.append(sfx_, sfxlen_); }
                     return;
                  }
                  case K::value_inline:
                     sfx_ = nullptr; sfxlen_ = 0; full_str_ = nullptr;
                     if constexpr (inlineable)
                        unpack_inline(child, val_);
                     if (maintain_) key_.len = base;
                     return;
                  case K::prefix_node:
                  {
                     const char* P  = static_cast<const char*>(deref_(child));
                     uint32_t    cb = uint32_t(base);
                     if (maintain_) { key_.len = base; key_.append(pn_pfx<Ptr>(P), pn_plen(P)); cb = uint32_t(key_.len); }
                     stk_.push_back(Frame{child, 1, uint32_t(base), cb});  // single child consumed
                     child = pn_next<Ptr>(const_cast<char*>(P));
                     base  = cb;
                     continue;
                  }
                  case K::bucket:
                  {
                     bucket* b = static_cast<bucket*>(deref_(child));
                     stk_.push_back(Frame{child, 1, uint32_t(base), uint32_t(base)});  // entry 0 consumed
                     const char* e = bkt_entry(b, 0);  // index order is suffix order: min = first
                     sfx_      = bkt_suf(e);
                     sfxlen_   = bkt_slen(e);
                     full_str_ = nullptr;
                     set_ptr_(bkt_val(e));
                     if (maintain_) { key_.len = base; key_.append(sfx_, sfxlen_); }
                     return;
                  }
                  case K::setlist_u16:
                  {
                     const setlist16* s  = static_cast<const setlist16*>(deref_(child));
                     uint32_t         cb = uint32_t(base);
                     if (maintain_) { key_.len = base; cb = uint32_t(key_.len); }
                     if (!s->rh.term.is_null())  // term < stems
                     {
                        stk_.push_back(Frame{child, 0, uint32_t(base), cb});  // term consumed; stems remain
                        sfx_ = nullptr; sfxlen_ = 0; full_str_ = nullptr;
                        emit_(s->rh.term);
                        if (maintain_) key_.len = cb;
                        return;
                     }
                     stk_.push_back(Frame{child, 1, uint32_t(base), cb});  // stem 0 consumed
                     const uint16_t st = s->stems[0];
                     if (maintain_)
                     { key_.push(uint8_t(st >> 8)); key_.push(uint8_t(st & 0xFF)); }
                     child = s->branch[0];
                     base  = maintain_ ? size_t(key_.len) : size_t(cb) + 2;
                     continue;
                  }
                  default:  // byte routers (setlist / c2 / c4 / c8 / node_full / pfxd)
                  {
                     const router_hdr* rh = static_cast<const router_hdr*>(deref_(child));
                     uint32_t          cb = uint32_t(base);
                     if (maintain_)
                     {
                        key_.len = base;
                        if (child.tag() == K::setlist_u8 && rh->hdr.prefix_len)
                           key_.append(static_cast<const setlist*>(deref_(child))->prefix,
                                       rh->hdr.prefix_len);
                        else if (child.tag() == K::pfxd && rh->hdr.prefix_len)
                           key_.append(pfxd_pfx(rh), rh->hdr.prefix_len);
                        cb = uint32_t(key_.len);
                     }
                     if (!rh->term.is_null())  // term < children
                     {
                        stk_.push_back(Frame{child, 0, uint32_t(base), cb});  // term consumed; children remain
                        sfx_ = nullptr; sfxlen_ = 0; full_str_ = nullptr;
                        emit_(rh->term);
                        if (maintain_) key_.len = cb;
                        return;
                     }
                     uint8_t    bb;
                     packed_ptr nc;
                     if (next_child_(child, 0, bb, nc))  // first child
                     {
                        stk_.push_back(Frame{child, int(bb) + 1, uint32_t(base), cb});
                        if (maintain_) key_.push(bb);
                        if (child.tag() == K::setlist_u8)  // leftmost child may be inline
                        {
                           const setlist* s = static_cast<const setlist*>(deref_(child));
                           if (s->inl & 0x80)
                           {
                              const int i = setlist_index(s, bb);
                              if (i >= 0 && map::sl_is_inline(s, i))
                              { emit_inline_leaf_(s, i, maintain_ ? size_t(key_.len) : 0); return; }
                           }
                        }
                        child = nc;
                        base  = maintain_ ? size_t(key_.len) : size_t(cb) + 1;
                        continue;
                     }
                     // termless AND childless — degenerate/unreachable; mirror descend_max_
                     stk_.push_back(Frame{child, 0, uint32_t(base), cb});
                     sfx_ = nullptr; sfxlen_ = 0; full_str_ = nullptr;
                     emit_(rh->term);
                     if (maintain_) key_.len = cb;
                     return;
                  }
               }
            }
         }
         void retreat_()  // step to the PREVIOUS terminal; the mirror of seek_
         {
            while (!stk_.empty())
            {
               Frame&  f = stk_.back();
               const K k = f.node.tag();
               if (k == K::prefix_node)
               {  // single child, no term: backward-exhausted on re-entry
                  if (maintain_) key_.len = f.plen;
                  stk_.pop_back();
                  continue;
               }
               if constexpr (Buckets || Adaptive)
                  if (k == K::bucket)
                  {
                     if (f.cur >= 2)  // current = entry cur-1; previous = entry cur-2
                     {
                        --f.cur;
                        bucket*     b = static_cast<bucket*>(deref_(f.node));
                        const char* e = bkt_entry(b, f.cur - 1);
                        sfx_ = bkt_suf(e); sfxlen_ = bkt_slen(e); full_str_ = nullptr;
                        set_ptr_(bkt_val(e));
                        if (maintain_) { key_.len = f.cbase; key_.append(sfx_, sfxlen_); }
                        return;
                     }
                     if (maintain_) key_.len = f.plen;
                     stk_.pop_back();
                     continue;
                  }
               if (k == K::setlist_u16)
               {
                  const setlist16* s = static_cast<const setlist16*>(deref_(f.node));
                  if (f.cur >= 2)  // current = branch[cur-1]; previous = branch[cur-2]
                  {
                     --f.cur;
                     const int      i  = f.cur - 1;
                     const uint16_t st = s->stems[i];
                     if (maintain_)
                     { key_.len = f.cbase; key_.push(uint8_t(st >> 8)); key_.push(uint8_t(st & 0xFF)); }
                     descend_max_(s->branch[i], maintain_ ? size_t(key_.len) : 0);
                     return;
                  }
                  if (f.cur == 1 && !s->rh.term.is_null())  // before branch[0] comes the term
                  {
                     f.cur = 0;
                     sfx_ = nullptr; sfxlen_ = 0; full_str_ = nullptr;
                     emit_(s->rh.term);
                     if (maintain_) key_.len = f.cbase;
                     return;
                  }
                  if (maintain_) key_.len = f.plen;
                  stk_.pop_back();
                  continue;
               }
               // byte routers
               const router_hdr* rh = static_cast<const router_hdr*>(deref_(f.node));
               if (f.cur >= 1)  // current = child at byte cur-1
               {
                  uint8_t    bb;
                  packed_ptr child;
                  if (prev_child_(f.node, f.cur - 2, bb, child))
                  {
                     f.cur = int(bb) + 1;
                     if (maintain_) { key_.len = f.cbase; key_.push(bb); }
                     if (f.node.tag() == K::setlist_u8)  // previous child may be inline
                     {
                        const setlist* s = static_cast<const setlist*>(deref_(f.node));
                        if (s->inl & 0x80)
                        {
                           const int i = setlist_index(s, bb);
                           if (i >= 0 && map::sl_is_inline(s, i))
                           { emit_inline_leaf_(s, i, maintain_ ? size_t(key_.len) : 0); return; }
                        }
                     }
                     descend_max_(child, maintain_ ? size_t(key_.len) : 0);
                     return;
                  }
                  if (!rh->term.is_null())  // the term precedes the first child
                  {
                     f.cur = 0;
                     sfx_ = nullptr; sfxlen_ = 0; full_str_ = nullptr;
                     emit_(rh->term);
                     if (maintain_) key_.len = f.cbase;
                     return;
                  }
               }
               if (maintain_) key_.len = f.plen;  // cur <= 0: nothing precedes here
               stk_.pop_back();
               continue;
            }
            end_ = true;  // retreated past the first key (--begin() is UB, as std::map)
         }

        public:
         // LegacyBidirectionalIterator surface. NOTE: reference is a proxy (key is rebuilt,
         // value may be inline), so this is NOT a C++20 std::bidirectional_iterator — same
         // status as std::vector<bool>::iterator. Works with range-for, std::reverse_iterator
         // and the classic algorithms.
         using iterator_category = std::bidirectional_iterator_tag;
         using key_type          = Key;
         using mapped_type       = T;
         using value_type        = std::pair<const Key, T>;
         using reference         = std::pair<Key, const T&>;  // key decoded by value (see key_codec)
         using difference_type   = std::ptrdiff_t;
         struct arrow_proxy  // operator-> for a proxy reference (holds the pair, hands back &it)
         {
            reference        kv;
            const reference* operator->() const noexcept { return &kv; }
         };
         using pointer = arrow_proxy;

         struct end_tag
         {
         };
         const_iterator() = default;
         // end() WITH tree identity: compares equal to any end iterator, and --it
         // re-seats at the rightmost key (std::map's --end() / rbegin()).
         const_iterator(const map* t, end_tag) : t_(t) {}
         explicit const_iterator(const map* t, packed_ptr root) : t_(t)
         {
            if (!root.is_null())
            {
               end_ = false;
               if (!descend_(root, 0)) seek_();
            }
         }
         // lower_bound (upper=false) / upper_bound (upper=true) seek.
         const_iterator(const map* t, packed_ptr root, std::string_view key, bool upper) : t_(t)
         {
            if (!root.is_null())
            {
               end_ = false;
               if (!descend_lb_(root, key, 0, upper)) seek_();
            }
         }
         // Pure position comparison (no key materialization): two iterators are equal iff
         // they sit at the same terminal, which the frame stack (node + cursor per level)
         // identifies uniquely regardless of how each was reached.
         bool operator==(const const_iterator& o) const noexcept
         {
            if (end_ != o.end_) return false;
            if (end_) return true;
            if (stk_.size() != o.stk_.size()) return false;
            for (size_t i = 0; i < stk_.size(); ++i)
               if (stk_[i].node.raw() != o.stk_[i].node.raw() || stk_[i].cur != o.stk_[i].cur)
                  return false;
            return true;
         }
         bool            operator!=(const const_iterator& o) const noexcept { return !(*this == o); }
         const_iterator& operator++() { seek_(); return *this; }
         const_iterator  operator++(int)
         {
            const_iterator tmp = *this;
            seek_();
            return tmp;
         }
         const_iterator& operator--()
         {
            if (end_)
            {  // --end(): seat at the rightmost key (needs a tree-carrying end iterator)
               if (t_ && !t_->root_.is_null())
               {
                  end_ = false;
                  stk_ = frame_stack();
                  if (maintain_) key_.len = 0;
                  descend_max_(t_->root_, 0);
               }
               return *this;
            }
            retreat_();
            return *this;
         }
         const_iterator operator--(int)
         {
            const_iterator tmp = *this;
            --(*this);
            return tmp;
         }
         const T&        value() const noexcept
         {
            if constexpr (inlineable)
               return val_;
            else
               return *val_;
         }
         // Adaptive key materialization. Until key() is first read, ++ does NO key work
         // (value-only iteration pays nothing). The first read replays the whole path,
         // priming each frame's plen/cbase, and flips maintain_ on — from then ++ keeps
         // key_ current incrementally (only the diverged tail is rewritten), so repeated
         // key reads are as cheap as the eager scan. Assumes "read once → read again".
         // Raw key bytes (the radix's native form): zero-copy where possible. key() decodes
         // these back to the typed Key (by value — a non-string key has no stored object).
         Key              key() const { return key_codec<Key>::decode(key_bytes()); }
         std::string_view key_bytes() const
         {
            if (full_str_)
               return std::string_view(full_str_, full_len_);  // zero-copy: full key in the leaf
            if (maintain_)
               return key_.view();  // already current
            key_.len = 0;
            for (Frame& fr : stk_)
            {
               fr.plen   = uint32_t(key_.len);
               const K k = fr.node.tag();
               if (k == K::prefix_node)
               {
                  const char* P = static_cast<const char*>(deref_(fr.node));
                  key_.append(pn_pfx<Ptr>(P), pn_plen(P));
                  fr.cbase = uint32_t(key_.len);
               }
               else if (k == K::setlist_u8)
               {
                  const setlist* s = static_cast<const setlist*>(deref_(fr.node));
                  if (s->rh.hdr.prefix_len) key_.append(s->prefix, s->rh.hdr.prefix_len);
                  fr.cbase = uint32_t(key_.len);
                  if (fr.cur >= 1) key_.push(uint8_t(fr.cur - 1));
               }
               else if (k == K::pfxd)  // fused prefix, then the edge byte like a full
               {
                  const router_hdr* rh = static_cast<const router_hdr*>(deref_(fr.node));
                  if (rh->hdr.prefix_len) key_.append(pfxd_pfx(rh), rh->hdr.prefix_len);
                  fr.cbase = uint32_t(key_.len);
                  if (fr.cur >= 1) key_.push(uint8_t(fr.cur - 1));
               }
               else if (k == K::setlist_u16)  // wide router: replay the 2 bytes of the descended stem
               {
                  const setlist16* s = static_cast<const setlist16*>(deref_(fr.node));
                  fr.cbase = uint32_t(key_.len);
                  if (fr.cur >= 1)
                  { const uint16_t st = s->stems[fr.cur - 1]; key_.push(uint8_t(st >> 8)); key_.push(uint8_t(st & 0xFF)); }
               }
               else  // bucket (no prefix/edge) or c2/c4/c8/node_full (edge only)
               {
                  fr.cbase = uint32_t(key_.len);
                  if (k != K::bucket && fr.cur >= 1) key_.push(uint8_t(fr.cur - 1));
               }
            }
            if (sfxlen_) key_.append(sfx_, sfxlen_);
            maintain_ = true;
            return key_.view();
         }
         reference  operator*() const { return reference{key(), value()}; }
         arrow_proxy operator->() const { return arrow_proxy{operator*()}; }
      };

      using iterator = const_iterator;  // read-only container (no element-mutating iterator)
      // Container-level reference/pointer (Container named requirements). Like
      // std::vector<bool>, the reference is a PROXY value-pair, not value_type&: the key
      // is materialized on demand and small trivially-copyable values may live inline in
      // the handle bits, so no std::pair<const Key, T> object exists in memory to bind to.
      using reference       = typename const_iterator::reference;
      using const_reference = typename const_iterator::reference;
      using pointer         = typename const_iterator::pointer;
      using const_pointer   = typename const_iterator::pointer;

      const_iterator begin() const { return const_iterator(this, root_); }
      const_iterator end() const
      {
         return const_iterator(this, typename const_iterator::end_tag{});
      }
      const_iterator cbegin() const { return begin(); }
      const_iterator cend() const { return end(); }

      using reverse_iterator       = std::reverse_iterator<const_iterator>;
      using const_reverse_iterator = reverse_iterator;
      const_reverse_iterator rbegin() const { return const_reverse_iterator(end()); }
      const_reverse_iterator rend() const { return const_reverse_iterator(begin()); }
      const_reverse_iterator crbegin() const { return rbegin(); }
      const_reverse_iterator crend() const { return rend(); }

      // Ordered positioning (lexicographic, unsigned-byte). lower_bound: first key >= key;
      // upper_bound: first key > key; equal_range: [lower, upper) (≤1 element since keys
      // are unique). All return end() when no such key exists.
      const_iterator lower_bound(key_param k) const
      {
         scratch_t ks_;
         return const_iterator(this, root_, enc_(k, ks_), false);
      }
      const_iterator upper_bound(key_param k) const
      {
         scratch_t ks_;
         return const_iterator(this, root_, enc_(k, ks_), true);
      }
      std::pair<const_iterator, const_iterator> equal_range(key_param k) const
      {
         scratch_t        ks_;
         std::string_view kb = enc_(k, ks_);
         return {const_iterator(this, root_, kb, false), const_iterator(this, root_, kb, true)};
      }

      // Map-style lookup/erase returning the STL shapes (iterator / count).
      const_iterator find(key_param k) const
      {
         scratch_t        ks_;
         std::string_view kb = enc_(k, ks_);
         const_iterator   it(this, root_, kb, false);  // lower_bound on the bytes
         return (it != end() && it.key_bytes() == kb) ? it : end();
      }
      size_type count(key_param k) const { return contains(k) ? 1 : 0; }
      size_type erase(key_param k) { return remove(k) ? 1 : 0; }

      // STL erase(iterator): remove the element at pos; returns the iterator to the
      // element that followed it (end() if it was the last). pos must be a valid,
      // dereferenceable iterator into *this. Two descents: the erased key is captured
      // by value first (the mutation invalidates pos), then the successor is re-sought.
      const_iterator erase(const_iterator pos)
      {
         assert(pos != end() && "artpp::map::erase: pos must be dereferenceable");
         const std::string kb(pos.key_bytes());  // own the bytes: remove invalidates pos
         remove_bytes_(kb);
         return const_iterator(this, root_, kb, false);  // first key >= kb == the successor
      }

      // Fast ordered scan (the libart art_iter model): a recursive DFS visitor that
      // enumerates children directly (no per-element resumption, no byte-probing) and
      // grows one shared key buffer. ~3x the iterator on cache-resident scans. Calls
      // f(std::string_view key, const T& value) in ascending key order.
      template <class F>
      void for_each(F&& f) const
      {
         if (root_.is_null()) return;
         key_buf kb;
         walk_(root_, kb, f);
      }

      // Ordered value-only scan: visits each value in ascending key order WITHOUT
      // building keys. Calls f(const T&). Doubles as a diagnostic: the gap vs for_each
      // is the key-reconstruction cost; what's left is the raw node walk.
      template <class F>
      void for_each_value(F&& f) const
      {
         if (!root_.is_null()) walk_v_(root_, f);
      }

      // ── bucket engine (gated; trivially-copyable T) ─────────────────────────────
      // A bucket is an inline (suffix,value) collector that collapses a sparse/deep
      // radix tail into one node. Enabled via enable_buckets(); terminals become
      // buckets, routers (setlist/cN/full) appear only when a bucket overflows.
     public:
      static constexpr bool bucketable = true;  // works for any T (entries never memmove'd)

     private:
      // Entry layout in the tail: [ T value | u16 suffix_len | suffix bytes ], with the
      // entry START aligned to alignof(T) (T placement-new'd in place — non-POD safe).
      // Sizes are alignof(T)-rounded so the tail stays aligned as it grows downward.
      static constexpr size_t TALIGN = alignof(T);
      static size_t bkt_round(size_t x) noexcept { return (x + TALIGN - 1) & ~(TALIGN - 1); }
      static uint16_t bkt_stem2(std::string_view s) noexcept
      {
         return uint16_t((s.size() > 0 ? unsigned(uint8_t(s[0])) << 8 : 0u) |
                         (s.size() > 1 ? unsigned(uint8_t(s[1])) : 0u));
      }
      static uint16_t*       bkt_div(bucket* b) noexcept { return reinterpret_cast<uint16_t*>(b->data); }
      static const uint16_t* bkt_div(const bucket* b) noexcept { return reinterpret_cast<const uint16_t*>(b->data); }
      static uint16_t*       bkt_slot(bucket* b) noexcept { return bkt_div(b) + b->nent; }
      static const uint16_t* bkt_slot(const bucket* b) noexcept { return bkt_div(b) + b->nent; }
      static char*           bkt_entry(bucket* b, int i) noexcept
      {
         return reinterpret_cast<char*>(b->data) + (bucket::PAGE - 4) - bkt_slot(b)[i];
      }
      static const char* bkt_entry(const bucket* b, int i) noexcept
      {
         return reinterpret_cast<const char*>(b->data) + (bucket::PAGE - 4) - bkt_slot(b)[i];
      }
      static T*             bkt_val(char* e) noexcept { return reinterpret_cast<T*>(e); }            // aligned
      static const T*       bkt_val(const char* e) noexcept { return reinterpret_cast<const T*>(e); }
      static uint16_t       bkt_slen(const char* e) noexcept { uint16_t v; std::memcpy(&v, e + sizeof(T), 2); return v; }
      static const uint8_t* bkt_suf(const char* e) noexcept { return reinterpret_cast<const uint8_t*>(e + sizeof(T) + 2); }
      static size_t         bkt_esize(size_t slen) noexcept { return bkt_round(sizeof(T) + 2 + slen); }
      // Can a FRESH page hold one entry with this suffix (slot pair + entry bytes)?
      // Suffixes past this cap are stored as plain leaves instead (see make_bucket_branch).
      static bool bkt_fits(size_t slen) noexcept { return 4 + bkt_esize(slen) <= size_t(bucket::PAGE - 4); }
      // first index with div >= q (count of div < q). Deliberately scalar: PAGE caps n at
      // a few dozen, and a lane-masked NEON count measured flat (buckets bench, 2M keys —
      // the page touch dominates), so the branchless 2-op loop stays.
      static int bkt_lower(const uint16_t* d, int n, uint16_t q) noexcept
      {
         int c = 0;
         for (int i = 0; i < n; ++i) c += (d[i] < q);
         return c;
      }
      // entry i's suffix vs (s): -1/0/1
      static int bkt_cmp(const bucket* b, int i, std::string_view s) noexcept
      {
         const char* e  = bkt_entry(b, i);
         const int   el = bkt_slen(e), sn = int(s.size());
         const int   m  = el < sn ? el : sn;
         const int   c  = m ? std::memcmp(bkt_suf(e), s.data(), size_t(m)) : 0;
         if (c) return c < 0 ? -1 : 1;
         return el == sn ? 0 : (el < sn ? -1 : 1);
      }
      // The entry whose suffix == rem, else nullptr — the one bucket point-lookup, shared by
      // find/update/locate_. div[] narrows to the 2-byte stem group; the full suffix decides.
      static const char* bkt_locate(const bucket* b, std::string_view rem) noexcept
      {
         const uint16_t  q2 = bkt_stem2(rem);
         const uint16_t* d  = bkt_div(b);
         for (int i = bkt_lower(d, b->nent, q2); i < b->nent && d[i] == q2; ++i)
         {
            const char* e = bkt_entry(b, i);
            if (bkt_slen(e) == rem.size() &&
                (rem.empty() || std::memcmp(bkt_suf(e), rem.data(), rem.size()) == 0))
               return e;
         }
         return nullptr;
      }
      // INVARIANT: the div/slot index is in total lexicographic suffix order at all times.
      // bkt_insert_entry places each pair by insertion sort — bkt_lower finds the stem run
      // (zero-padded 2-byte stems can tie but never invert full-suffix order), the in-run
      // scan compares whole suffixes — and every other mutation preserves the order (POD
      // remove shifts pairs; non-POD remove and splits rebuild through the same insert).
      // Ordered readers (iterator, for_each, eq_walk_) therefore walk indices 0..nent
      // directly; only the entry PAYLOADS at the page tail are in arrival order.
      static bool bkt_index_sorted_(const bucket* b) noexcept  // debug-check of the invariant
      {
         for (int i = 1; i < b->nent; ++i)
         {
            const char* e = bkt_entry(b, i);
            if (bkt_cmp(b, i - 1,
                        std::string_view(reinterpret_cast<const char*>(bkt_suf(e)), bkt_slen(e))) >= 0)
               return false;
         }
         return true;
      }

      // Node creators placement-new into the allocator's bytes: it starts the object's
      // lifetime ISO-cleanly and costs nothing (trivial default-init emits no code).
      bucket* new_bucket()
      {
         bucket* b   = ::new (node_alloc_(sizeof(bucket))) bucket;
         b->nent     = 0;
         b->tail_used = 0;
         return b;
      }
      // A fresh single-entry bucket holding (key[at:], val) — bucket mode's new-value branch.
      // A suffix no page could ever hold becomes a plain leaf instead: every descent
      // (find/iterate/remove/bkt_put) already handles leaf terminals, so correctness is
      // mode-independent; only the packing policy is bypassed for these rare giants.
      template <class VF>
      packed_ptr make_bucket_branch(std::string_view key, size_t at, VF&& val)
      {
         if (!bkt_fits(key.size() - at))
            return make_value_at(key, at, std::forward<VF>(val));
         bucket*                    b = new_bucket();
         [[maybe_unused]] const int r = bkt_insert_entry(b, key.substr(at), std::forward<VF>(val));
         assert(r == 1 && "artpp: a bkt_fits-checked entry cannot overflow a fresh page");
         return pack_(b, K::bucket);
      }
      // Adaptive collapse: turn the leaf at a deep-narrow split point into a bucket
      // holding the old entry + the new key. Frees the old leaf.
      template <bool Assign, class VF>
      packed_ptr collapse_to_bucket(packed_ptr cur, std::string_view key, size_t depth, VF&& val,
                                    bool& inserted)
      {
         // The caller pre-checks bkt_fits for BOTH suffixes — neither insert can overflow.
         bucket* b = new_bucket();
         if (cur.tag() == K::value_inline)
         {
            T ov;
            unpack_inline(cur, ov);
            [[maybe_unused]] const int r0 = bkt_insert_entry(b, std::string_view{}, std::move(ov));
            assert(r0 == 1);
         }
         else
         {
            char*            L = static_cast<char*>(deref_term_(cur));
            std::string_view S(reinterpret_cast<const char*>(leaf_suf(L)), leaf_slen(L));
            T                ov(std::move(*leaf_val(L)));  // move out, then destruct the leaf's T
            v_destroy(alloc_, leaf_val(L));
            [[maybe_unused]] const int r0 = bkt_insert_entry(b, S, std::move(ov));
            assert(r0 == 1);
            term_free_(L, leaf_alloc_size(L));
         }
         const int r1 = bkt_insert_entry<Assign>(b, key.substr(depth), std::forward<VF>(val));
         assert(r1 >= 0 && "artpp: collapse_to_bucket pre-checked sizes; overflow is impossible");
         inserted = (r1 == 1);
         return pack_(b, K::bucket);
      }
      bool bkt_find(const bucket* b, std::string_view rem, T& out) const
      {
         const char* e = bkt_locate(b, rem);
         if (!e)
            return false;
         copy_value(out, bkt_val(e));
         return true;
      }
      // 1 inserted, 0 updated, -1 overflow (caller splits). `val` is forwarded — moved
      // on a successful insert/update, left untouched on overflow (so the caller can
      // re-pass it after splitting).
      template <bool Assign = true, class V>
      int bkt_insert_entry(bucket* b, std::string_view suf, V&& val)
      {
         const uint16_t q2  = bkt_stem2(suf);
         uint16_t*      d   = bkt_div(b);
         int            pos = bkt_lower(d, b->nent, q2);
         for (; pos < b->nent && d[pos] == q2; ++pos)
         {
            const int c = bkt_cmp(b, pos, suf);
            if (c == 0)
            {
               if constexpr (Assign)  // update value in place (key/suffix unchanged)
                  vf_assign(std::forward<V>(val), bkt_val(bkt_entry(b, pos)));
               return 0;
            }
            if (c > 0) break;
         }
         const size_t esz   = bkt_esize(suf.size());
         const size_t freeb = size_t(bucket::PAGE - 4) - 4u * b->nent - b->tail_used;
         if (freeb < 4 + esz) return -1;  // page full
         b->tail_used += uint16_t(esz);
         char* e = reinterpret_cast<char*>(b->data) + (bucket::PAGE - 4) - b->tail_used;  // aligned
         vf_construct(std::forward<V>(val), bkt_val(e));  // construct at the entry (non-POD safe)
         const uint16_t sl = uint16_t(suf.size());
         std::memcpy(e + sizeof(T), &sl, 2);
         std::memcpy(e + sizeof(T) + 2, suf.data(), suf.size());
         const uint16_t off = b->tail_used;
         d                  = bkt_div(b);
         uint16_t* so       = d + b->nent;      // old slot base
         uint16_t* sn       = d + b->nent + 1;  // new slot base (div grew by one)
         std::memmove(sn + pos + 1, so + pos, size_t(b->nent - pos) * 2);
         std::memmove(sn, so, size_t(pos) * 2);
         std::memmove(d + pos + 1, d + pos, size_t(b->nent - pos) * 2);
         d[pos]  = q2;
         sn[pos] = off;
         ++b->nent;
         assert(bkt_index_sorted_(b) && "artpp: bucket index must stay in total suffix order");
         return 1;
      }
      // Remove (s) and RECLAIM its tail space (artpp has no background compactor). bp may
      // be updated (non-POD path rebuilds into a fresh bucket). Returns true if removed.
      bool bkt_remove_entry(packed_ptr& bp, std::string_view suf)
      {
         bucket*        b   = static_cast<bucket*>(deref_(bp));
         const uint16_t q2  = bkt_stem2(suf);
         uint16_t*      d   = bkt_div(b);
         int            pos = bkt_lower(d, b->nent, q2);
         for (; pos < b->nent && d[pos] == q2; ++pos)
         {
            const int c = bkt_cmp(b, pos, suf);
            if (c > 0)
               return false;
            if (c == 0)
               break;
         }
         if (pos >= b->nent || d[pos] != q2)
            return false;

         char* const base_end = reinterpret_cast<char*>(b->data) + (bucket::PAGE - 4);
         char*       dead     = base_end - bkt_slot(b)[pos];
         v_destroy(alloc_, bkt_val(dead));  // run T's destructor

         if constexpr (std::is_trivially_copyable_v<T>)
         {
            // compact the tail in place: shift the entries below the hole up by E.
            const uint16_t S          = bkt_slot(b)[pos];
            const size_t   E          = bkt_esize(bkt_slen(dead));
            char*          tail_start = base_end - b->tail_used;
            std::memmove(tail_start + E, tail_start, size_t(dead - tail_start));
            b->tail_used -= uint16_t(E);
            uint16_t* sl = bkt_slot(b);
            for (int i = 0; i < b->nent; ++i)
               if (sl[i] > S) sl[i] -= uint16_t(E);
            // drop the index pair at pos
            std::memmove(d + pos, d + pos + 1, size_t(b->nent - 1 - pos) * 2);
            std::memmove(d + b->nent - 1, d + b->nent, size_t(pos) * 2);
            std::memmove(d + b->nent - 1 + pos, d + b->nent + pos + 1, size_t(b->nent - 1 - pos) * 2);
            --b->nent;
            return true;
         }
         else
         {
            // non-POD: can't byte-shift T objects → rebuild a fresh tight bucket
            // (copy-construct live entries, destruct originals). Avoids any relocation.
            bucket* nb = new_bucket();
            for (int i = 0; i < b->nent; ++i)
            {
               if (i == pos) continue;
               char*            e = bkt_entry(b, i);
               std::string_view s(reinterpret_cast<const char*>(bkt_suf(e)), bkt_slen(e));
               bkt_insert_entry(nb, s, std::move(*bkt_val(e)));  // MOVE into nb
               v_destroy(alloc_, bkt_val(e));
            }
            node_free_(b, sizeof(bucket));
            bp = pack_(nb, K::bucket);
            return true;
         }
      }

      // Overflow: split a bucket at `depth` into a radix subtree. Extract the common
      // prefix of all (sorted) suffixes, then re-insert every entry through bkt_put.
      packed_ptr bkt_split(packed_ptr bp)
      {
         bucket*    b = static_cast<bucket*>(deref_(bp));
         const int  n = b->nent;
         // common prefix of all suffixes == lcp(first, last) since lex-sorted
         const char*      e0 = bkt_entry(b, 0);
         const char*      eN = bkt_entry(b, n - 1);
         std::string_view f0(reinterpret_cast<const char*>(bkt_suf(e0)), bkt_slen(e0));
         std::string_view fN(reinterpret_cast<const char*>(bkt_suf(eN)), bkt_slen(eN));
         const size_t     c = lcp(f0, fN);
         setlist*         r = new_setlist();
         packed_ptr       top;
         {
            // common bytes from a stable copy (entries get freed when we drop b)
            uint8_t cbuf[setlist::PREFIX_CAP > 8 ? setlist::PREFIX_CAP : 8];
            const bool fits = c <= setlist::PREFIX_CAP;
            if (fits && c) std::memcpy(cbuf, bkt_suf(e0), c);
            // A short common prefix goes inline; a longer one is NOT wrapped here — the
            // re-inserted full suffixes below re-form it naturally as prefix nodes.
            top = fits ? apply_prefix(r, std::string_view(reinterpret_cast<char*>(cbuf), c))
                       : pack_(r, K::setlist_u8);
         }
         // Destructive rebuild: values are MOVED out of b. Strong isn't achievable without
         // copying (T may be move-only), but basic-guarantee + leak-freedom is: guard the
         // partial `top` so a throw reclaims it, and DON'T destroy the moved-from originals
         // until the whole rebuild succeeds — on a throw, b keeps a full set of valid
         // (moved-from) values, so it stays consistent and is reclaimed normally at teardown.
         build_guard g{this, &top};
         for (int i = 0; i < n; ++i)
         {
            char*            e = bkt_entry(b, i);
            std::string_view suf(reinterpret_cast<const char*>(bkt_suf(e)), bkt_slen(e));
            bool             ins = false;
            top                  = bkt_put<true>(top, suf, 0, std::move(*bkt_val(e)), ins);  // MOVE into target
         }
         g.release();
         for (int i = 0; i < n; ++i)  // committed: destroy all moved-from originals, free b
            v_destroy(alloc_, bkt_val(bkt_entry(b, i)));
         node_free_(b, sizeof(bucket));
         return top;
      }

      // Recursive bucket-mode insert into the subtree at `node`. Terminals are buckets;
      // routers reused from the radix family. `val` is MOVED to its single resting place
      // (each path consumes it exactly once — on overflow bkt_insert_entry leaves it
      // intact, so the re-pass after the split still moves a live value).
      template <bool Assign, class VF>
      packed_ptr bkt_put(packed_ptr node, std::string_view key, size_t depth, VF&& val, bool& ins)
      {
         if (node.is_null())
         {
            ins = true;
            return make_bucket_branch(key, depth, std::forward<VF>(val));
         }
         switch (node.tag())
         {
            case K::bucket:
            {
               bucket* b = static_cast<bucket*>(deref_(node));
               if (bkt_fits(key.size() - depth))
               {
                  const int r = bkt_insert_entry<Assign>(b, key.substr(depth), std::forward<VF>(val));
                  if (r >= 0) { ins = (r == 1); return node; }
               }
               // overflow → radix; an over-cap suffix re-enters and lands as a leaf via
               // make_bucket_branch (splitting can never make it fit a page)
               packed_ptr router = bkt_split(node);
               return bkt_put<Assign>(router, key, depth, std::forward<VF>(val), ins);      // val untouched on overflow
            }
            case K::value_ptr:
            case K::value_inline:
               // an over-cap-suffix terminal stored as a plain leaf (see make_bucket_branch):
               // the radix leaf splitter handles collision/overwrite; in-range keys form
               // buckets again beneath the split via this function's router arms.
               return split_leaf<Assign>(node, key, depth, std::forward<VF>(val), ins);
            case K::prefix_node:
            {
               char*          P  = static_cast<char*>(deref_(node));
               const uint16_t pl = pn_plen(P);
               if (const packed_ptr nx = pn_next<Ptr>(P);  // lazy fusion, as in insert_
                   (nx.tag() == K::node_full && pl <= node_full::PFX_CAP) ||
                   (nx.tag() == K::pfxd && pl + pfxd_plen(deref_(nx)) <= node_full::PFX_CAP))
               {
                  packed_ptr fused = make_prefix(
                      std::string_view(reinterpret_cast<const char*>(pn_pfx<Ptr>(P)), pl), nx);
                  node_free_(P, pn_size<Ptr>(pl));
                  return bkt_put<Assign>(fused, key, depth, std::forward<VF>(val), ins);
               }
               std::string_view PP(reinterpret_cast<const char*>(pn_pfx<Ptr>(P)), pl);
               const size_t     c = lcp(PP, key.substr(depth));
               if (c == PP.size())
               {
                  packed_ptr nn = bkt_put<Assign>(pn_next<Ptr>(P), key, depth + c, std::forward<VF>(val), ins);
                  pn_set_next(P, nn);
                  return node;
               }
               ins = true;
               return split_prefix(node, key, depth, c, std::forward<VF>(val), [&](auto&& x) {
                  return make_bucket_branch(key, depth + c + 1, std::forward<decltype(x)>(x));
               });
            }
            case K::setlist_u8:
            {
               setlist*       s  = static_cast<setlist*>(deref_(node));
               const unsigned pl = s->rh.hdr.prefix_len;
               if (pl)
               {
                  const size_t c = lcp(std::string_view(reinterpret_cast<char*>(s->prefix), pl),
                                       key.substr(depth));
                  if (c < pl)
                  {
                     ins = true;
                     return split_setlist_prefix(node, key, depth, c, std::forward<VF>(val), [&](auto&& x) {
                        return make_bucket_branch(key, depth + c + 1, std::forward<decltype(x)>(x));
                     });
                  }
                  depth += pl;
               }
               if (depth == key.size()) { ins = set_or_update_term<Assign>(&s->rh, std::forward<VF>(val)); return node; }
               const uint8_t byte = uint8_t(key[depth]);
               const int     i    = setlist_index(s, byte);
               if (i >= 0)
               {
                  s->branch[i] = bkt_put<Assign>(s->branch[i], key, depth + 1, std::forward<VF>(val), ins);
                  return node;
               }
               ins              = true;
               packed_ptr child = make_bucket_branch(key, depth + 1, std::forward<VF>(val));
               if (setlist_set(s, byte, child)) return node;
               build_guard gc{this, &child};  // widen() can throw; don't orphan the new bucket
               uint8_t pfx[setlist::PREFIX_CAP];
               std::memcpy(pfx, s->prefix, pl);
               s->rh.hdr.prefix_len = 0;
               packed_ptr w         = node;
               do w = widen(w);
               while (!router_try_set(w, byte, child));
               gc.release();
               return pl ? make_prefix(std::string_view(reinterpret_cast<char*>(pfx), pl), w) : w;
            }
            case K::pfxd:  // consume the fused prefix, then route as the dense default
            {
               void*          n  = deref_(node);
               const unsigned pl = pfxd_plen(n);
               const size_t   c  = lcp(
                   std::string_view(reinterpret_cast<const char*>(pfxd_pfx(n)), pl),
                   key.substr(depth));
               if (c < pl)
               {
                  ins = true;
                  return split_pfxd_prefix(node, key, depth, c, std::forward<VF>(val), [&](auto&& x) {
                     return make_bucket_branch(key, depth + c + 1, std::forward<decltype(x)>(x));
                  });
               }
               depth += pl;
               [[fallthrough]];
            }
            default:  // c2 / c4 / c8 / node_full — no inline prefix
            {
               router_hdr* rh = static_cast<router_hdr*>(deref_(node));
               if (depth == key.size()) { ins = set_or_update_term<Assign>(rh, std::forward<VF>(val)); return node; }
               const uint8_t byte = uint8_t(key[depth]);
               if (packed_ptr* cs = router_find_slot(node, byte))
               {
                  *cs = bkt_put<Assign>(*cs, key, depth + 1, std::forward<VF>(val), ins);
                  return node;
               }
               ins              = true;
               packed_ptr child = make_bucket_branch(key, depth + 1, std::forward<VF>(val));
               build_guard gc{this, &child};  // widen() can throw; don't orphan the new bucket
               packed_ptr w     = node;
               while (!router_try_set(w, byte, child)) w = widen(w);
               gc.release();
               return w;
            }
         }
      }

     private:
      struct rm_lvl  // one descent level: where node j lives + how its parent reaches it
      {
         packed_ptr* slot;
         uint16_t    edge;  // byte (u8 routers/full/cN), 2-byte stem (u16); unused for pn/root
      };
      static constexpr uint32_t RM_LVN     = 64;  // shrink window: deeper ancestors stay unshrunk
      static constexpr int      SHRINK_MAX = setlist::CAP - 4;  // de-widen full/cN below the
                                                                // setlist CAP with hysteresis
                                                                // against widen/shrink ping-pong

      // Free the node at *L.slot (already unlinked logically); kinds of router only.
      void rm_free_(packed_ptr nd) noexcept { node_free_(deref_(nd), node_size(nd.tag())); }

      // Collapse a single-branch term-less router: absorb its inline prefix (setlist only)
      // + the edge to its one child INTO the child — free when the child is a setlist with
      // prefix room, else via a fresh prefix node (best-effort: kept as-is on bad_alloc).
      // pfx holds the bytes to prepend (router prefix ++ edge byte(s)).
      void rm_collapse_(packed_ptr* slot, packed_ptr nd, const uint8_t* pfx, unsigned pl,
                        packed_ptr child) noexcept
      {
         if (child.tag() == K::setlist_u8)
         {
            setlist*       cs  = static_cast<setlist*>(deref_(child));
            const unsigned cpl = cs->rh.hdr.prefix_len;
            if (cpl + pl <= setlist::PREFIX_CAP)
            {
               std::memmove(cs->prefix + pl, cs->prefix, cpl);
               std::memcpy(cs->prefix, pfx, pl);
               cs->rh.hdr.prefix_len = uint16_t(cpl + pl);
               *slot                 = child;
               rm_free_(nd);
               return;
            }
         }
         try  // child can't absorb: hang it under a small prefix node (pn->pn chains are fine)
         {
            packed_ptr top = make_prefix(std::string_view(reinterpret_cast<const char*>(pfx), pl), child);
            *slot          = top;
            rm_free_(nd);
         }
         catch (...)
         {
         }  // OOM: stay unshrunk — the tree is still correct
      }

      // Evaluate the router at *L.slot after it lost a branch/term. Returns true iff the
      // node was freed outright (empty) — the caller must then unlink it from ITS parent.
      bool rm_shrink_(packed_ptr* slot) noexcept
      {
         for (;;)  // a de-widened node is re-evaluated once as a setlist
         {
            const packed_ptr nd = *slot;
            const K          k  = nd.tag();
            if (k == K::prefix_node || !is_router(k))
               return false;
            router_hdr*    rh = static_cast<router_hdr*>(deref_(nd));
            const unsigned nb = rh->hdr.nbranch;
            const bool     tm = !rh->term.is_null();
            if (nb == 0)
            {
               if (!tm)
               {
                  rm_free_(nd);
                  return true;  // empty shell: cascade
               }
               // Only payload is a key ending HERE: the term handle (an empty-suffix leaf /
               // inline value) replaces the router. An inline prefix (setlist or fused) is
               // part of that key's path — it must survive as a prefix node above the term.
               if (k == K::setlist_u8 || k == K::pfxd)
               {
                  if (const unsigned pl = rh->hdr.prefix_len)
                  {
                     const uint8_t* pb = k == K::pfxd
                                             ? pfxd_pfx(rh)
                                             : static_cast<const setlist*>(deref_(nd))->prefix;
                     try
                     {
                        *slot = make_prefix(
                            std::string_view(reinterpret_cast<const char*>(pb), pl), rh->term);
                        rm_free_(nd);
                     }
                     catch (...)
                     {
                     }  // OOM: keep the router
                     return false;
                  }
               }
               *slot = rh->term;
               rm_free_(nd);
               return false;
            }
            if (nb == 1 && !tm)
            {
               // A lone INLINE survivor can't be collapsed here (rm_shrink_ is noexcept;
               // absorbing it into a prefix_node would need an alloc). Leave the setlist
               // as a valid 1-branch node — a future insert can re-shape it.
               if (k == K::setlist_u8 && sl_is_inline(static_cast<setlist*>(deref_(nd)), 0))
                  return false;
               uint8_t  pfx[node_full::PFX_CAP + 2];  // largest inline prefix + edge byte(s)
               unsigned pl = 0;
               if (k == K::setlist_u8)
               {
                  setlist* s = static_cast<setlist*>(deref_(nd));
                  pl         = s->rh.hdr.prefix_len;
                  std::memcpy(pfx, s->prefix, pl);
                  pfx[pl++] = s->bytes[0];
                  rm_collapse_(slot, nd, pfx, pl, s->branch[0]);
               }
               else if (k == K::setlist_u16)
               {
                  setlist16* s = static_cast<setlist16*>(deref_(nd));
                  pfx[pl++]    = uint8_t(s->stems[0] >> 8);
                  pfx[pl++]    = uint8_t(s->stems[0] & 0xFF);
                  rm_collapse_(slot, nd, pfx, pl, s->branch[0]);
               }
               else  // cN/full/pfxd: fused prefix (if any) ++ the lone child's byte
               {
                  if (k == K::pfxd)
                  {
                     pl = rh->hdr.prefix_len;
                     std::memcpy(pfx, pfxd_pfx(rh), pl);
                  }
                  packed_ptr child = packed_ptr::null();
                  router_for_each(nd, [&](uint8_t bb, packed_ptr br) {
                     pfx[pl] = bb;
                     child   = br;
                  });
                  ++pl;
                  rm_collapse_(slot, nd, pfx, pl, child);
               }
               return false;
            }
            if (nb <= unsigned(SHRINK_MAX) && k != K::setlist_u8 && k != K::setlist_u16 &&
                !(k == K::pfxd && rh->hdr.prefix_len > setlist::PREFIX_CAP))  // prefix must ride
            {
               try  // sparse dense node: de-widen back to one cacheline (best-effort)
               {
                  setlist* s2         = new_setlist();
                  s2->rh.term         = rh->term;
                  s2->rh.hdr.has_term = rh->hdr.has_term;
                  router_for_each(nd, [&](uint8_t bb, packed_ptr br) {
                     const bool ok = setlist_set(s2, bb, br);
                     assert(ok && "SHRINK_MAX <= setlist CAP");
                     (void)ok;
                  });
                  if (k == K::pfxd)  // the fused prefix rides into the setlist's inline slot
                  {
                     const unsigned pl = rh->hdr.prefix_len;
                     std::memcpy(s2->prefix, pfxd_pfx(rh), pl);
                     s2->rh.hdr.prefix_len = pl;
                  }
                  *slot = pack_(s2, K::setlist_u8);
                  rm_free_(nd);
                  continue;  // re-evaluate as a setlist (may now be 1-branch -> collapse)
               }
               catch (...)
               {
               }
            }
            return false;
         }
      }

      // Walk the recorded path upward: drop the freed child's edge from its parent, then
      // shrink each ancestor until one survives. `j` starts at the freed node's parent.
      void rm_unwind_(rm_lvl* path, uint32_t n, uint32_t j) noexcept
      {
         bool unlink = true;
         for (;; --j)
         {
            rm_lvl&          L  = path[j % RM_LVN];
            const packed_ptr nd = *L.slot;
            const K          k  = nd.tag();
            if (unlink)
            {
               if (k == K::prefix_node)
               {  // pn lost its only child: it dies too — keep cascading
                  char* P = static_cast<char*>(deref_(nd));
                  node_free_(P, pn_size<Ptr>(pn_plen(P)));
                  if (j == 0) break;
                  if (n - j >= RM_LVN) { *L.slot = packed_ptr::null(); return; }  // window edge
                  continue;
               }
               const uint16_t edge = path[(j + 1) % RM_LVN].edge;
               if (k == K::setlist_u16)
                  sl16_remove(static_cast<setlist16*>(deref_(nd)), edge);
               else
                  router_remove(nd, uint8_t(edge));
               unlink = false;
            }
            if (!rm_shrink_(L.slot))
               return;  // node survived (possibly replaced in *L.slot)
            if (j == 0) break;                                          // root died
            if (n - j >= RM_LVN) { *L.slot = packed_ptr::null(); return; }  // window edge
            unlink = true;
         }
         root_ = packed_ptr::null();
      }

     public:
      // Remove `key` if present; returns true iff it was found and removed. The structure
      // SHRINKS on the way out: empty routers/buckets are freed (cascading up), a router
      // whose only payload is its term collapses to the term handle, a single-branch
      // term-less router collapses into its child (edge byte + inline prefix absorbed, a
      // prefix node allocated only when the child can't take them for free), and a sparse
      // dense node (full/cN, <= 12 branches) de-widens back to a setlist. Allocating
      // collapses are best-effort: on bad_alloc the tree simply stays unshrunk. Ancestors
      // beyond the deepest 64 levels are left unshrunk (the window covers any practical key).
      // NOTE: not noexcept — removing a non-POD value from a bucket rebuilds it (allocates),
      // and a custom key_codec may allocate. Nothrow in practice for POD values / radix mode.
      bool remove(key_param k)
      {
         scratch_t ks_;
         return remove_bytes_(enc_(k, ks_));
      }

     private:
      // The remove descent over already-encoded key bytes — shared by remove() and the
      // iterator-taking erase() (whose position is identified by its raw key bytes).
      bool remove_bytes_(std::string_view key)
      {
         rm_lvl           path[RM_LVN];
         uint32_t         n    = 0;  // levels recorded (path[0] = root)
         packed_ptr*      slot = &root_;
         size_t           depth = 0;
         path[n++ % RM_LVN]     = {&root_, 0};
         auto free_leaf         = [&](char* L) {
            v_destroy(alloc_, leaf_val(L));
            term_free_(L, leaf_alloc_size(L));
         };
         for (;;)
         {
            const packed_ptr cur = *slot;
            switch (cur.tag())
            {
               case K::null:
                  return false;
               case K::value_ptr:
               {
                  char* L = static_cast<char*>(deref_term_(cur));
                  if (std::string_view(reinterpret_cast<const char*>(leaf_suf(L)), leaf_slen(L)) !=
                      key.substr(depth))
                     return false;
                  free_leaf(L);
                  --count_;
                  if (n >= 2) rm_unwind_(path, n, n - 2);  // n-2: the leaf's parent
                  else root_ = packed_ptr::null();         // the leaf WAS the root
                  return true;
               }
               case K::value_inline:
               {
                  if (depth != key.size())
                     return false;  // inline terminal matches only at key end; nothing to free
                  --count_;
                  if (n >= 2) rm_unwind_(path, n, n - 2);
                  else root_ = packed_ptr::null();
                  return true;
               }
               case K::bucket:
               {
                  if constexpr (Buckets || Adaptive)
                  {
                     packed_ptr nb = cur;
                     if (!bkt_remove_entry(nb, key.substr(depth)))  // may rebuild (non-POD)
                        return false;
                     --count_;
                     bucket* b = static_cast<bucket*>(deref_(nb));
                     if (b->nent == 0)
                     {
                        node_free_(b, sizeof(bucket));
                        if (n >= 2) rm_unwind_(path, n, n - 2);
                        else root_ = packed_ptr::null();
                     }
                     else if (nb.raw() != cur.raw())
                        *slot = nb;  // rebuilt → update the slot
                     return true;
                  }
                  return false;
               }
               case K::prefix_node:
               {
                  char*          P  = static_cast<char*>(deref_(cur));
                  const uint16_t pl = pn_plen(P);
                  if (key.size() - depth < pl || std::memcmp(pn_pfx<Ptr>(P), key.data() + depth, pl) != 0)
                     return false;
                  slot = reinterpret_cast<packed_ptr*>(P + 2);
                  path[n++ % RM_LVN] = {slot, 0};
                  depth += pl;
                  continue;
               }
               case K::setlist_u16:
               {
                  setlist16* s = static_cast<setlist16*>(deref_(cur));
                  if (key.size() - depth < 2)
                  {
                     if (depth != key.size())
                        return false;
                     goto rm_term;
                  }
                  {
                     const uint16_t q  = two_stem(key, depth);
                     packed_ptr*    cs = sl16_find_slot(s, q);
                     if (!cs)
                        return false;
                     slot               = cs;
                     path[n++ % RM_LVN] = {slot, q};
                     depth += 2;
                  }
                  continue;
               }
               case K::setlist_u8:
               case K::c2:
               case K::c4:
               case K::c8:
               case K::node_full:
               case K::pfxd:
               {
                  const K ck = cur.tag();
                  if (ck == K::setlist_u8 || ck == K::pfxd)  // kinds with an inline prefix
                  {
                     const router_hdr* rh = static_cast<const router_hdr*>(deref_(cur));
                     const unsigned    pl = rh->hdr.prefix_len;
                     if (pl)
                     {
                        const uint8_t* pb = ck == K::pfxd
                                                ? pfxd_pfx(rh)
                                                : static_cast<const setlist*>(deref_(cur))->prefix;
                        if (key.size() - depth < pl ||
                            std::memcmp(pb, key.data() + depth, pl) != 0)
                           return false;
                        depth += pl;
                     }
                  }
                  if (depth == key.size())
                     goto rm_term;
                  {
                     const uint8_t byte = uint8_t(key[depth]);
                     packed_ptr*   cs   = router_find_slot(cur, byte);
                     if (!cs)
                        return false;
                     if (ck == K::setlist_u8)  // the matched child may be an inline leaf
                     {
                        setlist*  s = static_cast<setlist*>(deref_(cur));
                        const int i = int(cs - s->branch);
                        if (sl_is_inline(s, i))
                        {
                           const char* lf = sl_inline_leaf(s, i);  // leaf shares cur's line
                           if (std::string_view(reinterpret_cast<const char*>(leaf_suf(lf)),
                                                leaf_slen(lf)) != key.substr(depth + 1))
                              return false;  // stem matched but suffix didn't → absent
                           sl_remove_branch_(s, i);  // drop branch + payload (alloc-free)
                           --count_;
                           if (rm_shrink_(slot))  // cur emptied/collapsed → unlink from parent
                           {
                              if (n >= 2) rm_unwind_(path, n, n - 2);
                              else        root_ = packed_ptr::null();
                           }
                           return true;
                        }
                     }
                     slot               = cs;
                     path[n++ % RM_LVN] = {slot, byte};
                     ++depth;
                  }
                  continue;
               }
            }
            __builtin_unreachable();
         rm_term:  // the key ends AT this router: drop its term, then shrink from this level
         {
            router_hdr* rh = static_cast<router_hdr*>(deref_(*slot));
            if (rh->term.is_null())
               return false;
            if (rh->term.tag() == K::value_ptr)  // leaf("",v); an inline term has nothing to free
               free_leaf(static_cast<char*>(deref_term_(rh->term)));
            rh->term         = packed_ptr::null();
            rh->hdr.has_term = 0;
            --count_;
            if (rm_shrink_(slot))  // emptied outright: unlink from the parent and cascade
            {
               if (n >= 2) rm_unwind_(path, n, n - 2);
               else root_ = packed_ptr::null();  // the root router emptied
            }
            return true;
         }
         }
      }

     public:
      size_t size() const noexcept { return count_; }
      bool   empty() const noexcept { return count_ == 0; }
      size_t max_size() const noexcept { return ~size_t(0); }

      // Remove all elements: destroy every T and free every node.
      void clear() noexcept
      {
         free_all_(root_);
         root_  = packed_ptr::null();
         count_ = 0;
      }
      void swap(map& o) noexcept(std::allocator_traits<Allocator>::is_always_equal::value)
      {
         using std::swap;
         // Allocators exchange iff they propagate on swap; otherwise they must be equal
         // (UB if not, per the standard), so leaving them is correct for the valid cases.
         if constexpr (std::allocator_traits<Allocator>::propagate_on_container_swap::value)
            swap(alloc_, o.alloc_);
         rebase_();
         o.rebase_();
         swap(root_, o.root_);
         swap(count_, o.count_);
      }
      // Non-member ADL swap (hidden friend) — found by generic code / std::ranges::swap.
      friend void swap(map& a, map& b) noexcept(noexcept(a.swap(b))) { a.swap(b); }

      // Equality: same (key, value) content. Compared STRUCTURALLY — node against node,
      // setlist against setlist — not element by element: prefixes and edge directories
      // are memcmp'd, subtrees recurse, and no key is ever materialized. Trees with the
      // same content normally have the same shape (insert paths are content-determined),
      // so this is one pass over the nodes. Shapes CAN diverge for equal content through
      // remove hysteresis (widen/shrink, collapse carving, bucket-collapse history); the
      // walk reports "undecided" for such representation mismatches and only then falls
      // back to the canonical two-iterator sequence compare. A hidden friend: T must be
      // equality-comparable only if tree equality is actually used.
      friend bool operator==(const map& a, const map& b)
      {
         if (a.count_ != b.count_) return false;
         const int r = a.eq_walk_(a.root_, b, b.root_);
         if (r >= 0) return r == 1;
         auto ia = a.begin(), ib = b.begin();  // representations diverged: sequence compare
         for (; ia != a.end(); ++ia, ++ib)
            if (ia.key_bytes() != ib.key_bytes() || !(ia.value() == ib.value())) return false;
         return true;
      }
      friend bool operator!=(const map& a, const map& b) { return !(a == b); }

     private:
      // Terminal-value equality (term handles / empty-suffix terminals), tolerant of the
      // inline-vs-leaf representation pair. Returns 0/1 (always decidable locally).
      int eq_value_(packed_ptr a, const map& tb, packed_ptr b) const
      {
         if (a.is_null() || b.is_null()) return (a.is_null() == b.is_null()) ? 1 : 0;
         const bool ia = a.tag() == K::value_inline, ib = b.tag() == K::value_inline;
         if constexpr (inlineable)
         {
            T x{}, y{};
            if (ia) unpack_inline(a, x);
            else
            {
               const char* L = static_cast<const char*>(deref_term_(a));
               if (leaf_slen(L)) return 0;  // a non-empty suffix is not a terminal-at-end
               copy_value(x, leaf_val(L));
            }
            if (ib) unpack_inline(b, y);
            else
            {
               const char* L = static_cast<const char*>(tb.deref_term_(b));
               if (leaf_slen(L)) return 0;
               copy_value(y, leaf_val(L));
            }
            return x == y ? 1 : 0;
         }
         else
         {
            (void)ia; (void)ib;  // !inlineable: both are leaves by construction
            const char* A = static_cast<const char*>(deref_term_(a));
            const char* B = static_cast<const char*>(tb.deref_term_(b));
            if (leaf_slen(A) != leaf_slen(B) ||
                (leaf_slen(A) && std::memcmp(leaf_suf(A), leaf_suf(B), leaf_slen(A)) != 0))
               return 0;
            return (*leaf_val(A) == *leaf_val(B)) ? 1 : 0;
         }
      }
      // The structural walk: 1 equal, 0 not equal, -1 representations diverged (content
      // undecided — caller falls back). Both nodes were reached along IDENTICAL paths, so
      // any byte-level divergence in prefixes/edges/suffixes means the key sets differ.
      int eq_walk_(packed_ptr a, const map& tb, packed_ptr b) const
      {
         if (a.is_null() || b.is_null()) return (a.is_null() == b.is_null()) ? 1 : 0;
         const K ka = a.tag(), kb = b.tag();
         if (ka != kb)
         {
            // inline vs empty-suffix leaf is the one same-content tag pair decidable here
            if ((ka == K::value_inline && kb == K::value_ptr) ||
                (ka == K::value_ptr && kb == K::value_inline))
               return eq_value_(a, tb, b);
            return -1;  // widen/shrink, carve or collapse hysteresis
         }
         switch (ka)
         {
            case K::null:
               return 1;  // handled above; kept for switch exhaustiveness
            case K::value_inline:
               return eq_value_(a, tb, b);
            case K::value_ptr:
            {
               const char* A = static_cast<const char*>(deref_term_(a));
               const char* B = static_cast<const char*>(tb.deref_term_(b));
               if (leaf_slen(A) != leaf_slen(B) ||
                   (leaf_slen(A) && std::memcmp(leaf_suf(A), leaf_suf(B), leaf_slen(A)) != 0))
                  return 0;  // suffixes are the semantic content; full-key form is derived
               return (*leaf_val(A) == *leaf_val(B)) ? 1 : 0;
            }
            case K::prefix_node:
            {
               const char*    A  = static_cast<const char*>(deref_(a));
               const char*    B  = static_cast<const char*>(tb.deref_(b));
               const uint16_t pa = pn_plen(A), pb = pn_plen(B);
               const uint16_t m  = pa < pb ? pa : pb;
               if (m && std::memcmp(pn_pfx<Ptr>(A), pn_pfx<Ptr>(B), m) != 0)
                  return 0;   // paths diverge → some key differs
               if (pa != pb)
                  return -1;  // same bytes so far, carved differently → undecided
               return eq_walk_(pn_next<Ptr>(const_cast<char*>(A)), tb,
                               pn_next<Ptr>(const_cast<char*>(B)));
            }
            case K::setlist_u8:
            {
               const setlist* A = static_cast<const setlist*>(deref_(a));
               const setlist* B = static_cast<const setlist*>(tb.deref_(b));
               // Inline children are a representation detail; let the canonical sequence
               // compare (inline-aware iterators) decide rather than match them structurally.
               if ((A->inl & 0x80) || (B->inl & 0x80)) return -1;
               const unsigned pa = A->rh.hdr.prefix_len, pb = B->rh.hdr.prefix_len;
               const unsigned m  = pa < pb ? pa : pb;
               if (m && std::memcmp(A->prefix, B->prefix, m) != 0) return 0;
               if (pa != pb) return -1;  // same bytes, different inline-prefix carving
               const int nb = A->rh.hdr.nbranch;
               if (nb != B->rh.hdr.nbranch) return 0;
               if (std::memcmp(A->bytes, B->bytes, size_t(nb)) != 0) return 0;  // sorted edges
               if (eq_value_(A->rh.term, tb, B->rh.term) == 0) return 0;
               for (int i = 0; i < nb; ++i)  // sorted + identical edges → parallel arrays
                  if (const int r = eq_walk_(A->branch[i], tb, B->branch[i]); r != 1) return r;
               return 1;
            }
            case K::setlist_u16:
            {
               const setlist16* A = static_cast<const setlist16*>(deref_(a));
               const setlist16* B = static_cast<const setlist16*>(tb.deref_(b));
               const int        nb = A->rh.hdr.nbranch;
               if (nb != B->rh.hdr.nbranch) return 0;
               if (std::memcmp(A->stems, B->stems, size_t(nb) * 2) != 0) return 0;
               if (eq_value_(A->rh.term, tb, B->rh.term) == 0) return 0;
               for (int i = 0; i < nb; ++i)
                  if (const int r = eq_walk_(A->branch[i], tb, B->branch[i]); r != 1) return r;
               return 1;
            }
            case K::pfxd:  // fused prefixes must agree, then compare as the dense router
            {
               const router_hdr* RA = static_cast<const router_hdr*>(deref_(a));
               const router_hdr* RB = static_cast<const router_hdr*>(tb.deref_(b));
               const unsigned    pa = RA->hdr.prefix_len, pb = RB->hdr.prefix_len;
               const unsigned    m  = pa < pb ? pa : pb;
               if (m && std::memcmp(pfxd_pfx(RA), pfxd_pfx(RB), m) != 0) return 0;
               if (pa != pb) return -1;  // same bytes, different carving → undecided
               [[fallthrough]];
            }
            case K::c2:
            case K::c4:
            case K::c8:
            case K::node_full:
            {
               const router_hdr* RA = static_cast<const router_hdr*>(deref_(a));
               const router_hdr* RB = static_cast<const router_hdr*>(tb.deref_(b));
               if (RA->hdr.nbranch != RB->hdr.nbranch) return 0;
               if (eq_value_(RA->term, tb, RB->term) == 0) return 0;
               int res = 1;
               router_for_each(a, [&](uint8_t e, packed_ptr ca) {
                  if (res != 1) return;
                  packed_ptr* sb = tb.router_find_slot(b, e);
                  if (!sb) { res = 0; return; }  // edge absent in b → key sets differ
                  res = eq_walk_(ca, tb, *sb);
               });
               return res;
            }
            case K::bucket:
            {
               const bucket* A = static_cast<const bucket*>(deref_(a));
               const bucket* B = static_cast<const bucket*>(tb.deref_(b));
               if (A->nent != B->nent) return 0;
               for (int i = 0; i < A->nent; ++i)  // both indexes are in suffix order
               {
                  const char* ea = bkt_entry(A, i);
                  const char* eb = bkt_entry(B, i);
                  if (bkt_slen(ea) != bkt_slen(eb) ||
                      (bkt_slen(ea) && std::memcmp(bkt_suf(ea), bkt_suf(eb), bkt_slen(ea)) != 0))
                     return 0;
                  if (!(*bkt_val(ea) == *bkt_val(eb))) return 0;
               }
               return 1;
            }
         }
         return -1;  // unreachable (exhaustive switch)
      }

     public:

      // Debug: node-type census + avg terminal depth (hops). Reveals how many
      // prefix_node hops the descent pays vs router/leaf nodes.
      struct dbg_counts
      {
         uint64_t leaf = 0, inl = 0, prefix = 0, setlist = 0, setlist16 = 0, c2 = 0, c4 = 0, c8 = 0, full = 0;
         uint64_t terminals = 0, sum_depth = 0;
         uint64_t pfx_le4 = 0, pfx_le8 = 0, pfx_le16 = 0, pfx_sum = 0;
         // Wide-stem opportunity. router_hops = sum over terminals of routers on its path
         // (≈ cacheline misses/lookup). collapsible = router nodes whose parent is also a
         // term-less router → a u16 node could fuse the two levels, saving 1 hop for every
         // key beneath. collapse_saved weights that by the terminals under each fused edge:
         // the predicted total router-hops removed if every eligible pair were fused.
         uint64_t router_hops = 0, collapsible = 0, collapse_saved = 0;
         uint64_t fullp = 0;  // node_full routers whose prefix is fused in-header (K::pfxd)
         // Occupancy census of the direct-index tier (node_full + pfxd): how sparse are the
         // fulls? Sparse fulls are mode::none's pathology (1792 B + far router_next scan for a
         // handful of branches); dense fulls (>=192) are where direct-index actually wins.
         // full_nbr_sum / (full+fullp) = mean fanout. cn_nbr_sum = branches living in cnodes.
         uint64_t full_lt48 = 0, full_48_96 = 0, full_96_160 = 0, full_160_192 = 0, full_ge192 = 0;
         uint64_t full_nbr_sum = 0, cn_nbr_sum = 0;
      };
      // Returns the number of terminals in cur's subtree (used to weight collapse_saved).
      // routers_above = routers already on the path; parent_noterm_router = the parent is a
      // term-less router (so a u16 node could fuse this router into it).
      uint64_t dbg_walk(packed_ptr cur, dbg_counts& d, int depth, int routers_above,
                        bool parent_noterm_router) const
      {
         if (cur.is_null())
            return 0;
         switch (cur.tag())
         {
            case K::value_ptr:    ++d.leaf; ++d.terminals; d.sum_depth += depth;
                                  d.router_hops += routers_above; return 1;
            case K::value_inline: ++d.inl;  ++d.terminals; d.sum_depth += depth;
                                  d.router_hops += routers_above; return 1;
            case K::prefix_node:
            {
               ++d.prefix;
               const uint16_t pl = pn_plen(static_cast<char*>(deref_(cur)));
               d.pfx_sum += pl;
               d.pfx_le4 += (pl <= 4);
               d.pfx_le8 += (pl <= 8);
               d.pfx_le16 += (pl <= 16);
               // A prefix_node breaks a router→router chain (the child can't fuse into the
               // parent across a multi-byte prefix), so parent_noterm_router resets to false.
               return dbg_walk(pn_next<Ptr>(static_cast<char*>(deref_(cur))), d, depth + 1,
                               routers_above, false);
            }
            case K::setlist_u8:   ++d.setlist; break;
            case K::setlist_u16:  ++d.setlist16; break;
            case K::c2:           ++d.c2; break;
            case K::c4:           ++d.c4; break;
            case K::c8:           ++d.c8; break;
            case K::node_full:    ++d.full; break;
            case K::pfxd:         ++d.fullp; break;  // full router with a fused in-header prefix
            case K::bucket:       return 0;  // census doesn't descend buckets
            case K::null:         return 0;  // exhaustive (no default)
         }
         const router_hdr* rh = static_cast<const router_hdr*>(deref_(cur));
         const bool        noterm = rh->term.is_null();
         {  // occupancy census (debug-only): bucket fulls by fanout, tally cnode branches
            const unsigned nb = rh->hdr.nbranch;
            const K        kk = cur.tag();
            if (kk == K::node_full || kk == K::pfxd)
            {
               d.full_nbr_sum += nb;
               d.full_lt48 += (nb < 48);
               d.full_48_96 += (nb >= 48 && nb < 96);
               d.full_96_160 += (nb >= 96 && nb < 160);
               d.full_160_192 += (nb >= 160 && nb < 192);
               d.full_ge192 += (nb >= 192);
            }
            else if (kk == K::c2 || kk == K::c4 || kk == K::c8)
               d.cn_nbr_sum += nb;
         }
         if (parent_noterm_router) ++d.collapsible;
         uint64_t sub = dbg_walk(rh->term, d, depth + 1, routers_above + 1, false);
         router_for_each(cur, [&](uint8_t, packed_ptr br) {
            sub += dbg_walk(br, d, depth + 1, routers_above + 1, noterm);
         });
         // If this router fuses up into its (term-less) parent, every terminal beneath it
         // loses one router hop.
         if (parent_noterm_router) d.collapse_saved += sub;
         return sub;
      }
      dbg_counts debug_stats() const { dbg_counts d; dbg_walk(root_, d, 0, 0, false); return d; }

     private:
      // ── leaf<T> layout: [u16 suffix_len][u16 pfx_field][pad to alignof(T)][T value][key bytes] ─
      // pfx_field = (has_full << 15) | prefix_len. The trailing key bytes are either the
      // suffix only (has_full=0, the compressed form) OR the WHOLE key = prefix ++ suffix
      // laid out contiguously (has_full=1) — stored opportunistically when the prefix fits
      // in the 128-byte slot we'd allocate anyway, so the iterator can hand back the full
      // key as a zero-copy string_view (same cacheline as the value). leaf_suf() points at
      // the suffix in both forms, so point lookup is identical.
      static constexpr uint16_t LEAF_FULL = 0x8000;
      static constexpr size_t   leaf_value_off() noexcept
      {
         constexpr size_t a = alignof(T);
         return (2 * sizeof(uint16_t) + a - 1) / a * a;
      }
      static size_t leaf_size(size_t strlen) noexcept { return leaf_value_off() + sizeof(T) + strlen; }
      static uint16_t leaf_slen(const char* p) noexcept { uint16_t v; std::memcpy(&v, p, 2); return v; }
      static uint16_t leaf_pfxfield(const char* p) noexcept { uint16_t v; std::memcpy(&v, p + 2, 2); return v; }
      static bool     leaf_has_full(const char* p) noexcept { return leaf_pfxfield(p) & LEAF_FULL; }
      static uint16_t leaf_plen(const char* p) noexcept { return leaf_pfxfield(p) & 0x7FFF; }
      static uint16_t leaf_strlen(const char* p) noexcept
      {
         const uint16_t f = leaf_pfxfield(p);
         return uint16_t(leaf_slen(p) + ((f & LEAF_FULL) ? (f & 0x7FFF) : 0));
      }
      static size_t   leaf_alloc_size(const char* p) noexcept { return leaf_size(leaf_strlen(p)); }
      static T*       leaf_val(char* p) noexcept { return reinterpret_cast<T*>(p + leaf_value_off()); }
      static const T* leaf_val(const char* p) noexcept
      {
         return reinterpret_cast<const T*>(p + leaf_value_off());
      }
      static char*       leaf_str(char* p) noexcept { return p + leaf_value_off() + sizeof(T); }
      static const char* leaf_str(const char* p) noexcept { return p + leaf_value_off() + sizeof(T); }
      // Full key as a contiguous view — valid ONLY when leaf_has_full(p).
      static std::string_view leaf_fullview(const char* p) noexcept
      {
         return std::string_view(leaf_str(p), leaf_strlen(p));
      }

      // Invoke f(key, value) for a terminal pointer (value_ptr leaf or value_inline).
      template <class F>
      void call_val_(std::string_view key, packed_ptr p, F& f) const
      {
         if constexpr (inlineable)
            if (p.tag() == K::value_inline)
            {
               T tmp;
               unpack_inline(p, tmp);
               f(key, static_cast<const T&>(tmp));
               return;
            }
         f(key, static_cast<const T&>(*leaf_val(static_cast<char*>(deref_term_(p)))));
      }

      // Value of a terminal pointer (value_ptr leaf or value_inline), into tmp if inline.
      const T* val_of_(packed_ptr p, T& tmp) const
      {
         if constexpr (inlineable)
            if (p.tag() == K::value_inline) { unpack_inline(p, tmp); return &tmp; }
         return leaf_val(static_cast<char*>(deref_term_(p)));
      }
      // Value-only DFS (no key assembly) backing for_each_value.
      template <class F>
      void walk_v_(packed_ptr cur, F& f) const
      {
         const K k = cur.tag();
         switch (k)
         {
            case K::value_ptr:
               f(static_cast<const T&>(*leaf_val(static_cast<char*>(deref_term_(cur)))));
               return;
            case K::value_inline:
               if constexpr (inlineable) { T tmp; unpack_inline(cur, tmp); f(static_cast<const T&>(tmp)); }
               return;
            case K::prefix_node:
               walk_v_(pn_next<Ptr>(static_cast<char*>(deref_(cur))), f);
               return;
            case K::bucket:
               if constexpr (bucketable)
               {
                  bucket* bk = static_cast<bucket*>(deref_(cur));
                  for (int i = 0; i < bk->nent; ++i)  // sum/visit order within bucket unspecified here
                     f(static_cast<const T&>(*bkt_val(bkt_entry(bk, i))));
               }
               return;
            default:
            {
               const router_hdr* rh = static_cast<const router_hdr*>(deref_(cur));
               if (!rh->term.is_null()) { T tmp; f(*val_of_(rh->term, tmp)); }
               auto recurse = [&](uint8_t, packed_ptr ch) { walk_v_(ch, f); };
               switch (k)
               {
                  case K::setlist_u8:
                  {
                     const setlist* s = static_cast<const setlist*>(deref_(cur));
                     const int      n = s->rh.hdr.nbranch;
                     for (int i = 0; i < n; ++i)
                        if (sl_is_inline(s, i)) f(static_cast<const T&>(*leaf_val(sl_inline_leaf(s, i))));
                        else                    walk_v_(s->branch[i], f);
                     break;
                  }
                  case K::c2: cnode_for_each(static_cast<const cnode<2>*>(deref_(cur)), recurse); break;
                  case K::c4: cnode_for_each(static_cast<const cnode<4>*>(deref_(cur)), recurse); break;
                  case K::c8: cnode_for_each(static_cast<const cnode<8>*>(deref_(cur)), recurse); break;
                  default:
                  {
                     const node_full* fn = static_cast<const node_full*>(deref_(cur));
                     for (int byte = 0; byte < 256; ++byte)
                     {
                        packed_ptr ch = full_find(fn, uint8_t(byte));
                        if (!ch.is_null()) walk_v_(ch, f);
                     }
                     break;
                  }
               }
               return;
            }
         }
      }

      // Recursive in-order DFS backing for_each. Enumerates children directly: setlist
      // by its sorted bytes[], cnode via cnode_for_each (ctz bitmap walk), node_full by
      // a single 0..255 pass, buckets by their sorted index (bkt_index_sorted_). `key`
      // grows on descent and the length is reset on return.
      template <class F>
      void walk_(packed_ptr cur, key_buf& key, F& f) const
      {
         const K k = cur.tag();
         switch (k)
         {
            case K::value_ptr:
            {
               char* L = static_cast<char*>(deref_term_(cur));
               if (leaf_has_full(L))  // whole key contiguous in the leaf → zero-copy, no append
               {
                  f(leaf_fullview(L), static_cast<const T&>(*leaf_val(L)));
                  return;
               }
               const size_t b = key.len;
               key.append(leaf_suf(L), leaf_slen(L));
               f(key.view(), static_cast<const T&>(*leaf_val(L)));
               key.len = b;
               return;
            }
            case K::value_inline:
               call_val_(key.view(), cur, f);  // empty suffix → key already complete
               return;
            case K::prefix_node:
            {
               char*        P = static_cast<char*>(deref_(cur));
               const size_t b = key.len;
               key.append(pn_pfx<Ptr>(P), pn_plen(P));
               walk_(pn_next<Ptr>(P), key, f);
               key.len = b;
               return;
            }
            case K::bucket:
               if constexpr (bucketable)
               {
                  bucket*      bk = static_cast<bucket*>(deref_(cur));
                  const size_t kb = key.len;
                  for (int i = 0; i < bk->nent; ++i)  // index order is suffix order
                  {
                     const char* e = bkt_entry(bk, i);
                     key.append(bkt_suf(e), bkt_slen(e));
                     f(key.view(), static_cast<const T&>(*bkt_val(e)));
                     key.len = kb;
                  }
               }
               return;
            default:  // routers
            {
               const router_hdr* rh   = static_cast<const router_hdr*>(deref_(cur));
               const size_t      base = key.len;
               if (k == K::setlist_u8 && rh->hdr.prefix_len)
                  key.append(static_cast<const setlist*>(deref_(cur))->prefix, rh->hdr.prefix_len);
               else if (k == K::pfxd && rh->hdr.prefix_len)
                  key.append(pfxd_pfx(rh), rh->hdr.prefix_len);
               const size_t cb = key.len;
               if (!rh->term.is_null())
                  call_val_(key.view(), rh->term, f);  // key ends here (path + prefix)
               auto recurse = [&](uint8_t byte, packed_ptr ch) {
                  key.push(byte);
                  walk_(ch, key, f);
                  key.len = cb;
               };
               switch (k)
               {
                  case K::setlist_u8:
                  {
                     const setlist* s = static_cast<const setlist*>(deref_(cur));
                     const int      n = s->rh.hdr.nbranch;
                     for (int i = 0; i < n; ++i)  // bytes[] sorted
                        if (sl_is_inline(s, i))   // inline leaf: emit key = path+byte+suffix
                        {
                           const char* L = sl_inline_leaf(s, i);
                           key.push(s->bytes[i]);
                           key.append(leaf_suf(L), leaf_slen(L));
                           f(key.view(), static_cast<const T&>(*leaf_val(L)));
                           key.len = cb;
                        }
                        else recurse(s->bytes[i], s->branch[i]);
                     break;
                  }
                  case K::c2: cnode_for_each(static_cast<const cnode<2>*>(deref_(cur)), recurse); break;
                  case K::c4: cnode_for_each(static_cast<const cnode<4>*>(deref_(cur)), recurse); break;
                  case K::c8: cnode_for_each(static_cast<const cnode<8>*>(deref_(cur)), recurse); break;
                  default:
                  {
                     const node_full* fn = static_cast<const node_full*>(deref_(cur));
                     for (int byte = 0; byte < 256; ++byte)
                     {
                        packed_ptr ch = full_find(fn, uint8_t(byte));
                        if (!ch.is_null()) recurse(uint8_t(byte), ch);
                     }
                     break;
                  }
               }
               key.len = base;
               return;
            }
         }
      }

      // Read-style descent returning a pointer to the STORED T for `key`, or nullptr.
      // Only meaningful when values are stable (non-inlineable T): backs find_ptr/at/[].
      // Pointer to the stored value for `key`, or nullptr — over the shared read skeleton.
      // Returns mutable T* (the value storage is mutable; descend_read is const only in that
      // it doesn't restructure). value_inline has no addressable object → nullptr.
      T* locate_(std::string_view key) const
      {
         struct op_t
         {
            const map* self;
            T* of_term(packed_ptr t) const  // a router/u16 term: T* iff it's a value_ptr leaf
            {
               return (t.is_null() || t.tag() == K::value_inline)
                          ? nullptr
                          : leaf_val(static_cast<char*>(self->deref_term_(t)));
            }
            T* leaf(const char* p, const char* rem, size_t rl) const
            {
               if (std::string_view(reinterpret_cast<const char*>(leaf_suf(p)), leaf_slen(p)) !=
                   std::string_view(rem, rl))
                  return nullptr;
               return const_cast<T*>(leaf_val(p));
            }
            T* inl(uint64_t, bool) const { return nullptr; }  // no addressable object
            T* bkt(const bucket* b, const char* rem, size_t rl) const
            {
               if constexpr (bucketable)
                  if (const char* e = bkt_locate(b, std::string_view(rem, rl)))
                     return bkt_val(const_cast<char*>(e));
               return nullptr;
            }
            T* term(const router_hdr* rh) const { return of_term(rh->term); }
            T* miss() const { return nullptr; }
         };
         return descend_read(key, op_t{this});
      }
      // Suffix bytes (the lazy-expansion remainder) — after the stored prefix when full.
      // Branchless: a suffix-only leaf has pfx_field == 0, so the masked prefix_len is 0 and
      // this is just leaf_str; a full leaf adds prefix_len. Keeps the find hot path cheap.
      static uint8_t* leaf_suf(char* p) noexcept
      {
         return reinterpret_cast<uint8_t*>(leaf_str(p) + (leaf_pfxfield(p) & 0x7FFF));
      }
      static const uint8_t* leaf_suf(const char* p) noexcept
      {
         return reinterpret_cast<const uint8_t*>(leaf_str(p) + (leaf_pfxfield(p) & 0x7FFF));
      }

      static size_t lcp(std::string_view a, std::string_view b) noexcept
      {
         const size_t n = a.size() < b.size() ? a.size() : b.size();
         return size_t(std::mismatch(a.begin(), a.begin() + n, b.begin()).first - a.begin());
      }

      // emplace's deferred constructor: carries the caller's argument refs down the descent
      // and constructs T exactly ONCE, at the value's final address (leaf / bucket entry).
      // A present key never constructs at all. Plain T&& remains the currency for insert and
      // for relocating existing values; vf_construct/vf_make dispatch between the two.
      template <class... Args>
      struct em_t
      {
         std::tuple<Args&&...> args;
         void construct(Allocator& a, T* dst)
         {
            std::apply([&](Args&&... as)
                       { std::allocator_traits<Allocator>::construct(
                             a, dst, std::forward<Args>(as)...); },
                       std::move(args));
         }
         T make() { return std::make_from_tuple<T>(std::move(args)); }
      };
      template <class VF>
      void vf_construct(VF&& vf, T* dst)  // consume the value source at its final address
      {
         if constexpr (std::is_same_v<std::remove_cvref_t<VF>, T>)
            v_construct(alloc_, dst, std::forward<VF>(vf));
         else
            std::forward<VF>(vf).construct(alloc_, dst);
      }
      template <class VF>
      static T vf_make(VF&& vf)  // materialize (the inline-packing path; small trivial T)
      {
         if constexpr (std::is_same_v<std::remove_cvref_t<VF>, T>)
            return std::forward<VF>(vf);
         else
            return std::forward<VF>(vf).make();
      }
      // Overwrite a LIVE value — insert_or_assign semantics: T's assignment operator, NOT
      // destroy+reconstruct. Semantically what the STL specifies, and exception-safe: a
      // throwing move/copy leaves a valid object behind (destroy-then-construct would leave
      // a dead one for the tree to destroy again at teardown).
      template <class VF>
      static void vf_assign(VF&& vf, T* dst)
      {
         if constexpr (std::is_same_v<std::remove_cvref_t<VF>, T>)
            *dst = std::forward<VF>(vf);
         else
            *dst = std::forward<VF>(vf).make();  // em_t under Assign (emplace never assigns today)
      }

      static packed_ptr pack_inline(const T& v) noexcept
      {
         packed_ptr p;
         std::memset(p.b, 0, sizeof(Ptr));
         p.b[0] = uint8_t(K::value_inline);
         std::memcpy(p.b + 1, &v, inline_bytes);  // value in bytes [1..1+sizeof)
         return p;
      }
      static void unpack_inline(packed_ptr p, T& out) noexcept
      {
         // Only ever called on a value_inline slot, which exists only for an inlineable T; the
         // `if constexpr` keeps the memcpy from instantiating for a non-inlineable T (dead here).
         if constexpr (inlineable) std::memcpy(&out, p.b + 1, inline_bytes);
      }
      static void copy_value(T& out, const T* src)
      {
         if constexpr (std::is_trivially_copyable_v<T>)
            std::memcpy(&out, src, sizeof(T));
         else
            out = *src;
      }
      // Read a terminal value slot (inline or leaf) into out. Used at key-end (the
      // suffix is empty for terminals reached this way).
      void read_value(packed_ptr v, T& out) const
      {
         if (v.tag() == K::value_inline)
            unpack_inline(v, out);
         else
            copy_value(out, leaf_val(static_cast<const char*>(deref_term_(v))));
      }
      bool read_term(const router_hdr* rh, T& out) const
      {
         if (rh->term.is_null())
            return false;
         read_value(rh->term, out);
         return true;
      }

      static void leaf_set_hdr(char* p, uint16_t slen, uint16_t pfxfield) noexcept
      {
         std::memcpy(p, &slen, 2);
         std::memcpy(p + 2, &pfxfield, 2);
      }

      // ── setlist inline children (Dan's `inl` byte) ──────────────────────────────
      // An inline branch's handle is NOT a pool address: tag stays value_ptr, the rest
      // holds a tail-offset, and the leaf bytes live at line+128-tailoff in the
      // setlist's own line. The `inl` byte (bit7 active, bits0-6 per branch 0..6) is the
      // sole source of truth — never trust an inline branch's bits as an address. Inline
      // payloads pack against the tail in branch order (canonical), re-established after
      // every change. Trivially-copyable T only (payloads move by memcpy).
      static constexpr unsigned SL_LINE          = cacheline_bytes;
      static constexpr unsigned sl_branch_off    = unsigned(offsetof(setlist, branch));
      static constexpr int      SL_INLINE_BRANCH = 7;  // bits 0..6 cover branches 0..6
      static bool     sl_inl_active(const setlist* s) noexcept { return s->inl & 0x80; }
      static bool     sl_is_inline(const setlist* s, int i) noexcept
      {
         return (s->inl & 0x80) && i < SL_INLINE_BRANCH && (s->inl & (1u << i));
      }
      static packed_ptr sl_inline_handle(unsigned tailoff) noexcept
      {
         packed_ptr p;
         std::memset(p.b, 0, sizeof(Ptr));
         const uint64_t raw = (uint64_t(tailoff) << 4) | uint8_t(K::value_ptr);
         std::memcpy(p.b, &raw, sizeof(Ptr));
         return p;
      }
      static unsigned sl_tailoff(packed_ptr p) noexcept { return unsigned(p.raw() >> 4); }
      // The inline leaf bytes for branch i (caller guarantees sl_is_inline). `s` IS the
      // line base (deref_ returns it), so this resolves for direct and indexed alike.
      char*       sl_inline_leaf(setlist* s, int i) const noexcept
      {
         return reinterpret_cast<char*>(s) + (SL_LINE - sl_tailoff(s->branch[i]));
      }
      const char* sl_inline_leaf(const setlist* s, int i) const noexcept
      {
         return reinterpret_cast<const char*>(s) + (SL_LINE - sl_tailoff(s->branch[i]));
      }
      static unsigned sl_used_branch_end(const setlist* s) noexcept
      {
         return sl_branch_off + unsigned(s->rh.hdr.nbranch) * unsigned(sizeof(Ptr));
      }
      // Inline leaves must start on alignof(T) so the value field lands aligned (the line
      // base is 128-aligned, so an aligned line-offset gives an aligned address). >=1.
      static constexpr unsigned SL_VAL_ALIGN = alignof(T) < 1 ? 1 : unsigned(alignof(T));
      // Repack inline payloads contiguously against the tail in ascending branch order
      // (canonical). No allocation; safe because trivially-copyable payloads move by
      // memcpy and the new positions never overlap the branch array.
      void sl_repack_(setlist* s) const noexcept
      {
         if (!sl_inl_active(s)) return;
         const int n = std::min<int>(s->rh.hdr.nbranch, SL_INLINE_BRANCH);
         char      tmp[SL_LINE];
         unsigned  pos = SL_LINE, newoff[SL_INLINE_BRANCH];
         for (int i = 0; i < n; ++i)
            if (sl_is_inline(s, i))
            {
               const char*    lf = sl_inline_leaf(s, i);
               const unsigned sz = unsigned(leaf_alloc_size(lf));
               pos = (pos - sz) & ~(SL_VAL_ALIGN - 1);  // align the leaf start (value aligned)
               std::memcpy(tmp + pos, lf, sz);
               newoff[i] = pos;
            }
         std::memcpy(reinterpret_cast<char*>(s) + pos, tmp + pos, SL_LINE - pos);
         for (int i = 0; i < n; ++i)
            if (sl_is_inline(s, i)) s->branch[i] = sl_inline_handle(SL_LINE - newoff[i]);
      }
      // Opportunistically move external leaf children INTO the line, all-or-nothing:
      // allocation-free (frees the externals), canonical by construction. The creation
      // workhorse — call after building/growing a small setlist.
      void sl_inline_compact_(setlist* s) noexcept
      {
         if constexpr (!inline_children_ok) { (void)s; return; }
         else
         {
            const int n = std::min<int>(s->rh.hdr.nbranch, SL_INLINE_BRANCH);
            char      tmp[SL_LINE];
            unsigned  pos = SL_LINE, newoff[SL_INLINE_BRANCH];
            uint8_t   newinl = 0x80;
            bool      any = false, was_ext[SL_INLINE_BRANCH] = {};
            for (int i = 0; i < n; ++i)
            {
               const char* src = nullptr;
               if (sl_is_inline(s, i)) src = sl_inline_leaf(s, i);
               else if (s->branch[i].tag() == K::value_ptr)
               {
                  src        = static_cast<const char*>(deref_term_(s->branch[i]));
                  was_ext[i] = true;
               }
               if (!src) continue;
               const unsigned sz = unsigned(leaf_alloc_size(src));
               // Align the leaf start (value alignment); a leaf too big to fit the region
               // would underflow pos — all-or-nothing, so bail and leave everything external.
               if (sz > pos) return;
               const unsigned np = (pos - sz) & ~(SL_VAL_ALIGN - 1);
               if (np < sl_used_branch_end(s)) return;
               pos = np;
               std::memcpy(tmp + pos, src, sz);
               newoff[i] = pos;
               newinl |= uint8_t(1u << i);
               any = true;
            }
            if (!any) return;  // no leaf children to pull inline
            for (int i = 0; i < n; ++i)
               if (was_ext[i])
                  term_free_(deref_term_(s->branch[i]), leaf_alloc_size(static_cast<const char*>(deref_term_(s->branch[i]))));
            std::memcpy(reinterpret_cast<char*>(s) + pos, tmp + pos, SL_LINE - pos);
            for (int i = 0; i < n; ++i)
               if (newinl & (1u << i)) s->branch[i] = sl_inline_handle(SL_LINE - newoff[i]);
            s->inl = newinl;
         }
      }
      // Move one inline child OUT to a real terminal leaf (throws on alloc). Used before
      // any path that must treat the child as an ordinary external leaf (split/collapse/
      // widen). Leaves the rest of the inline set intact and valid.
      void sl_externalize_(setlist* s, int i)
      {
         char*          lf = sl_inline_leaf(s, i);
         const unsigned sz = unsigned(leaf_alloc_size(lf));
         char*          ext = static_cast<char*>(term_alloc_(sz));  // throws
         std::memcpy(ext, lf, sz);
         s->branch[i] = pack_term_(ext, K::value_ptr);
         s->inl &= uint8_t(~(1u << i));
         if (!(s->inl & 0x7F)) s->inl = 0;  // no inline children left → drop active bit
         else sl_repack_(s);                // keep the survivors canonical
      }
      void sl_externalize_all_(setlist* s)  // throws on alloc (basic guarantee: partial is valid)
      {
         if (!sl_inl_active(s)) return;
         const int n = std::min<int>(s->rh.hdr.nbranch, SL_INLINE_BRANCH);
         for (int i = 0; i < n; ++i)
            if (sl_is_inline(s, i))
            {
               char*          lf  = sl_inline_leaf(s, i);
               const unsigned sz  = unsigned(leaf_alloc_size(lf));
               char*          ext = static_cast<char*>(term_alloc_(sz));  // throws
               std::memcpy(ext, lf, sz);
               s->branch[i] = pack_term_(ext, K::value_ptr);
               s->inl &= uint8_t(~(1u << i));
            }
         s->inl = 0;
      }
      // Drop branch i (stems/handles shift down, the inl bits shift with them), reclaiming
      // an inline payload if branch i had one. Allocation-free — the inline-aware twin of
      // setlist_remove, used on the remove path so inl never desyncs from the branch array.
      void sl_remove_branch_(setlist* s, int i) const noexcept
      {
         const int n = s->rh.hdr.nbranch;
         std::memmove(&s->bytes[i], &s->bytes[i + 1], size_t(n - 1 - i));
         std::memmove(&s->branch[i], &s->branch[i + 1], size_t(n - 1 - i) * sizeof(Ptr));
         s->bytes[n - 1]   = 0xFF;
         s->rh.hdr.nbranch = uint16_t(n - 1);
         if (s->inl & 0x80)
         {
            const uint8_t flags = uint8_t(s->inl & 0x7F);  // drop active bit BEFORE the shift,
            const uint8_t lo    = uint8_t(flags & ((1u << i) - 1));               // else bit7→bit6
            const uint8_t hi    = uint8_t((flags >> 1) & ~((1u << i) - 1) & 0x7F);  // (i,7)→[i,6)
            const uint8_t bits  = uint8_t(lo | hi);
            s->inl = bits ? uint8_t(0x80 | bits) : 0;
            sl_repack_(s);  // re-pack survivors canonically (reclaims the dropped payload)
         }
      }

      template <class VF>
      packed_ptr make_leaf(std::string_view suf, VF&& val)  // suffix-only (compressed) leaf
      {
         char* p = static_cast<char*>(term_alloc_(leaf_size(suf.size())));
         leaf_set_hdr(p, uint16_t(suf.size()), 0);
         std::memcpy(leaf_str(p), suf.data(), suf.size());
         vf_construct(std::forward<VF>(val), leaf_val(p));
         return pack_term_(p, K::value_ptr);
      }
      template <class VF>
      packed_ptr make_leaf_full(std::string_view pfx, std::string_view suf, VF&& val)  // whole key
      {
         char* p = static_cast<char*>(term_alloc_(leaf_size(pfx.size() + suf.size())));
         leaf_set_hdr(p, uint16_t(suf.size()), uint16_t(LEAF_FULL | pfx.size()));
         std::memcpy(leaf_str(p), pfx.data(), pfx.size());
         std::memcpy(leaf_str(p) + pfx.size(), suf.data(), suf.size());
         vf_construct(std::forward<VF>(val), leaf_val(p));
         return pack_term_(p, K::value_ptr);
      }
      // A terminal value with key suffix `suf`: inline in the pointer when the suffix
      // is empty and T fits (no allocation), else a suffix-only leaf.
      template <class VF>
      packed_ptr make_value(std::string_view suf, VF&& val)
      {
         if constexpr (inlineable)
            if (suf.empty())
               return pack_inline(vf_make(std::forward<VF>(val)));
         return make_leaf(suf, std::forward<VF>(val));
      }
      // Full-key-aware terminal: `key` is the whole key, the leaf sits at `depth` (suffix
      // = key[depth:]). Stores the WHOLE key contiguously iff the prefix fits in the node
      // blocks we'd allocate anyway (free) → enables zero-copy key() during iteration;
      // else falls back to suffix-only. depth must fit the leaf's 15-bit prefix_len field.
      template <class VF>
      packed_ptr make_value_at(std::string_view key, size_t depth, VF&& val)
      {
         const std::string_view suf = key.substr(depth);
         if constexpr (inlineable)
            if (suf.empty())
               return pack_inline(vf_make(std::forward<VF>(val)));
         const std::string_view pfx = key.substr(0, depth);
         if (depth < 0x8000 &&
             tunits_(leaf_size(pfx.size() + suf.size())) == tunits_(leaf_size(suf.size())))
            return make_leaf_full(pfx, suf, std::forward<VF>(val));
         return make_leaf(suf, std::forward<VF>(val));
      }
      // ── pfxd: the in-header prefix (the tag is a hint, not a type) ──────────────
      // A K::pfxd handle says only "the header cacheline carries a prefix you must
      // consume"; the node's REAL kind is recorded in the header itself. Today the
      // node_full is the kind with header capacity (its CL0 was pure padding); cN and
      // wide setlists keep prefix_node as their overflow. Layout behind a pfxd handle:
      //   CL0: router_hdr | kind (u8) | prefix bytes (rh.hdr.prefix_len of them)
      //   CL1+: the kind's normal body at its normal offsets (offtab unchanged).
      static K pfxd_kind(const void* n) noexcept
      {
         return K(static_cast<const uint8_t*>(n)[sizeof(router_hdr)]);
      }
      static uint8_t* pfxd_pfx(void* n) noexcept
      {
         return static_cast<uint8_t*>(n) + sizeof(router_hdr) + 1;
      }
      static const uint8_t* pfxd_pfx(const void* n) noexcept
      {
         return static_cast<const uint8_t*>(n) + sizeof(router_hdr) + 1;
      }
      static unsigned pfxd_plen(const void* n) noexcept
      {
         return static_cast<const router_hdr*>(n)->hdr.prefix_len;
      }
      // Drop a consumed/shortened fused prefix in place; returns the handle to use
      // (retags back to the real kind when the prefix empties — header no longer needed).
      packed_ptr pfxd_set_prefix(void* n, const uint8_t* bytes, unsigned pl)
      {
         router_hdr* rh = static_cast<router_hdr*>(n);
         if (pl == 0)
         {
            rh->hdr.prefix_len = 0;
            return pack_(n, pfxd_kind(n));
         }
         std::memmove(pfxd_pfx(n), bytes, pl);  // bytes may alias the tail of the old prefix
         rh->hdr.prefix_len = pl & 0x3F;
         return pack_(n, K::pfxd);
      }

      // Attach a shared prefix above `next`. The prefix lives in the TARGET's header
      // cacheline whenever there is capacity (node_full today; an existing fused prefix
      // is prepended to) and only overflows into a dedicated prefix_node when it can't
      // fit. setlists take their (smaller) inline prefix via apply_prefix at their own
      // call sites.
      packed_ptr make_prefix(std::string_view pfx, packed_ptr next)
      {
         if (pfx.empty())
            return next;
         const K nk = next.tag();
         if (nk == K::node_full || (nk == K::pfxd && pfxd_kind(deref_(next)) == K::node_full))
         {
            node_full*     f  = static_cast<node_full*>(deref_(next));
            const unsigned pl = f->rh.hdr.prefix_len;  // 0 for plain node_full
            if (pfx.size() + pl <= node_full::PFX_CAP)
            {
               uint8_t* pb = f->hdr + 1;
               std::memmove(pb + pfx.size(), pb, pl);  // prepend before any existing
               std::memcpy(pb, pfx.data(), pfx.size());
               f->hdr[0]            = uint8_t(K::node_full);
               f->rh.hdr.prefix_len = uint16_t(pfx.size() + pl) & 0x3F;
               return pack_(f, K::pfxd);
            }
         }
         char*    p  = static_cast<char*>(node_alloc_(pn_size<Ptr>(pfx.size())));
         uint16_t pl = uint16_t(pfx.size());
         std::memcpy(p, &pl, 2);
         pn_set_next(p, next);
         std::memcpy(pn_pfx<Ptr>(p), pfx.data(), pfx.size());
         return pack_(p, K::prefix_node);
      }
      setlist* new_setlist()
      {
         setlist* s = ::new (node_alloc_(sizeof(setlist))) setlist;
         s->rh.hdr  = node_hdr{0, 0, 0};
         s->rh.term = packed_ptr::null();
         s->inl     = 0;  // no inline children yet
         std::memset(s->bytes, 0xFF, sizeof(s->bytes));  // pad unused lanes with max for sorted SIMD lower_bound
         return s;
      }
      setlist16* new_setlist16()
      {
         setlist16* s = ::new (node_alloc_(sizeof(setlist16))) setlist16;
         s->rh.hdr    = node_hdr{0, 0, 0};
         s->rh.term   = packed_ptr::null();
         std::memset(s->stems, 0xFF, sizeof(s->stems));  // all lanes 0xFFFF for the SIMD lower_bound
         return s;
      }
      // Online fusion (the WideStems creation policy): when a leaf split has just turned a
      // child of setlist_u8 `*pslot` into a router, collapse `*pslot` + ALL its children into
      // one wide u16 node — iff that's a genuinely sparse 2-level chain (so the win is real
      // and it won't immediately re-stride). Predicate: the parent is prefix-less, and every
      // child is a prefix-less, term-less ONE-byte router (a leaf/term/bucket/u16 child can't
      // be represented by a 2-byte hop), and the fan-in fits one cacheline with headroom.
      // Fires only at splits (off the find hot path); dense nodes fail the predicate untouched.
      static constexpr int FUSE_THRESH = setlist16::CAP - 2;  // headroom so a few more inserts don't re-stride
      // noexcept: fusion is a best-effort optimization that runs AFTER the insert has already
      // succeeded, so it must never fail (or corrupt) it. The one allocation (the wide node) is
      // wrapped — on OOM we simply leave the 2-level chain in place.
      void try_fuse_parent(packed_ptr* pslot) noexcept
      {
         const packed_ptr Pp = *pslot;
         if (Pp.tag() != K::setlist_u8)
            return;
         setlist*  P  = static_cast<setlist*>(deref_(Pp));
         if (P->rh.hdr.prefix_len != 0)
            return;
         const int nb = P->rh.hdr.nbranch;
         if (nb == 0)
            return;
         int total = 0;
         for (int i = 0; i < nb; ++i)
         {
            const packed_ptr c  = P->branch[i];
            const K          ck = c.tag();
            if (ck == K::setlist_u16 || !is_router(ck))  // 2-byte child or leaf/term/bucket/prefix
               return;
            const router_hdr* crh = static_cast<const router_hdr*>(deref_(c));
            if (!crh->term.is_null())                    // a key ends 1 byte in → not representable
               return;
            if ((ck == K::setlist_u8 || ck == K::pfxd) && crh->hdr.prefix_len != 0)
               return;  // an inline (or fused) prefix is not representable in the wide node
            if (ck == K::setlist_u8 && sl_inl_active(static_cast<setlist*>(deref_(c))))
               return;  // child's inline leaves can't move into the u16 — skip fusion
            total += crh->hdr.nbranch;
            if (total > FUSE_THRESH)
               return;
         }
         // commit: build the wide node (b0,b1)→grandchild, carrying the parent's term. The wide
         // node is the only allocation; on OOM skip fusion (the 2-level chain is already correct).
         setlist16* U;
         try { U = new_setlist16(); }
         catch (...) { return; }
         U->rh.term       = P->rh.term;
         U->rh.hdr.has_term = P->rh.hdr.has_term;
         for (int i = 0; i < nb; ++i)
         {
            const uint8_t b0 = P->bytes[i];
            router_for_each(P->branch[i],
                            [&](uint8_t b1, packed_ptr gc) { sl16_set(U, uint16_t(uint16_t(b0) << 8 | b1), gc); });
         }
         for (int i = 0; i < nb; ++i)  // the child routers are absorbed; grandchildren are reused
            node_free_(deref_(P->branch[i]), node_size(P->branch[i].tag()));
         node_free_(P, sizeof(setlist));
         *pslot = pack_(U, K::setlist_u16);
      }

      // Re-stride a full (or 1-byte-stuck) wide router back to the byte-at-a-time family:
      // a u16 stem (hi,lo)→child becomes hi→(u8 setlist)→lo→child, so each child's stored
      // suffix (already past depth+2) stays correct as the two u8 levels consume hi then lo.
      // The node's term (a key ending exactly here, at depth) carries over unchanged. This is
      // the self-tuning fallback: dense regions that overflow 14 stems revert to byte routing.
      packed_ptr restride_u16_to_u8(packed_ptr cur)
      {
         static_assert(setlist16::CAP <= setlist::CAP, "every u16 hi-group / hi-set fits one u8 setlist");
         setlist16* N = static_cast<setlist16*>(deref_(cur));
         const int  n = N->rh.hdr.nbranch;
         // A u16 holds <= setlist16::CAP (14) stems, so each hi-group has <= 14 children and
         // there are <= 14 distinct hi bytes — both within setlist u8 CAP (16). No child or
         // parent setlist ever widens, so the new nodes are ALL plain setlists; a throw mid-build
         // frees them shell-only (grandchildren stay owned by the still-intact N). N itself is
         // only read until the final node_free_, so on a throw *slot keeps pointing at it.
         setlist* P         = new_setlist();
         P->rh.term         = N->rh.term;  // term is at this depth → unchanged
         P->rh.hdr.has_term = N->rh.hdr.has_term;
         restride_guard g{this, P};
         int            i = 0;
         while (i < n)
         {
            const uint8_t hi = uint8_t(N->stems[i] >> 8);
            setlist*      C  = new_setlist();  // the only throwing op in the loop
            while (i < n && uint8_t(N->stems[i] >> 8) == hi)
            {
               const bool ok = setlist_set(C, uint8_t(N->stems[i] & 0xFF), N->branch[i]);
               assert(ok && "restride: a u16 hi-group always fits one u8 setlist");
               (void)ok;
               ++i;
            }
            const bool ok = setlist_set(P, hi, pack_(C, K::setlist_u8));
            assert(ok && "restride: distinct-hi count always fits one u8 setlist");
            (void)ok;
         }
         g.release();
         node_free_(N, sizeof(setlist16));
         return pack_(P, K::setlist_u8);
      }
      // Attach a shared prefix to a freshly-built setlist: store it INLINE if it fits
      // (no extra hop), else fall back to a separate prefix_node. Returns the top ptr.
      packed_ptr apply_prefix(setlist* r, std::string_view pfx)
      {
         packed_ptr top = pack_(r, K::setlist_u8);
         if (pfx.empty())
            return top;
         if (pfx.size() <= setlist::PREFIX_CAP)
         {
            r->rh.hdr.prefix_len = uint16_t(pfx.size());
            std::memcpy(r->prefix, pfx.data(), pfx.size());
            return top;
         }
         return make_prefix(pfx, top);
      }
      template <class VF>
      void set_term(router_hdr* rh, VF&& val)
      {
         rh->term         = make_value(std::string_view{}, std::forward<VF>(val));  // "" -> inline
         rh->hdr.has_term = 1;
      }

      // ── router dispatch over the tier family {setlist, c2, c4, c8, node_full} ───
      static size_t node_size(K k) noexcept
      {
         switch (k)
         {
            case K::pfxd:        return sizeof(node_full);  // only header-capacity kind today
            case K::setlist_u8:  return sizeof(setlist);
            case K::setlist_u16: return sizeof(setlist16);
            case K::c2:          return sizeof(cnode<2>);
            case K::c4:          return sizeof(cnode<4>);
            case K::c8:          return sizeof(cnode<8>);
            case K::node_full:   return sizeof(node_full);
            // exhaustive (no default): a new node kind must declare its size here.
            case K::value_ptr: case K::value_inline: case K::prefix_node:
            case K::bucket: case K::null:
               assert(false); __builtin_unreachable();
         }
      }
      // ONE typed visitor for the byte-router family: switch on the tag once, hand the
      // TYPED node to f. Every byte-router dispatcher below is a packed_ptr -> typed-overload
      // one-liner over this, so tag exhaustiveness lives in exactly one switch (u16 is not a
      // byte router; the callers that can see one handle it first).
      // always_inline + noexcept propagation: the typed switch must dissolve into each
      // dispatcher exactly as the hand-written switches did — an outlined call (or a
      // may-throw edge inside the noexcept dispatchers) would bloat every caller with
      // EH/terminate scaffolding. Verified per-function asm-identical against those.
      template <class F>
      [[gnu::always_inline]] decltype(auto) with_router(packed_ptr cur, F&& f) const noexcept(
          std::is_nothrow_invocable_v<F&, setlist*>)
      {
         switch (cur.tag())
         {
            case K::pfxd:  // hint tag: real kind in the header — node_full today, and the
                           // body offsets are identical, so the typed view is the full's
               return f(static_cast<node_full*>(deref_(cur)));
            case K::setlist_u8: return f(static_cast<setlist*>(deref_(cur)));
            case K::c2:         return f(static_cast<cnode<2>*>(deref_(cur)));
            case K::c4:         return f(static_cast<cnode<4>*>(deref_(cur)));
            case K::c8:         return f(static_cast<cnode<8>*>(deref_(cur)));
            case K::node_full:  return f(static_cast<node_full*>(deref_(cur)));
            // exhaustive (no default): a new K must be decided here — and only here.
            case K::value_ptr: case K::value_inline: case K::prefix_node:
            case K::setlist_u16: case K::bucket: case K::null:
               assert(false && "byte-routers only"); __builtin_unreachable();
         }
      }

      // byte -> child (read step), per tier and dispatched.
      static packed_ptr router_find(setlist* s, uint8_t byte) noexcept { return setlist_find(s, byte); }
      template <int N>
      static packed_ptr router_find(cnode<N>* c, uint8_t byte) noexcept { return cnode_step(c, byte); }
      static packed_ptr router_find(node_full* f, uint8_t byte) noexcept { return full_find(f, byte); }

      // First child with byte >= from (iterator stepping). Each tier resumes from its own
      // structure — sorted stems, presence-bitmap ctz, nset line skip — instead of probing
      // every byte through a find.
      static bool router_next(setlist* s, int from, uint8_t& ob, packed_ptr& oc) noexcept
      {
         // bytes[] sorted ascending ⇒ "first index >= from" == "count of bytes < from".
         // Counted (branchless) instead of a data-dependent early-exit scan: the original
         // exit branch mispredicts hard on random `from` (it dominated lower_bound's unwind).
         const int n = s->rh.hdr.nbranch;
         int       i = 0;
         for (int j = 0; j < n; ++j) i += (int(s->bytes[j]) < from);
         if (i >= n) return false;
         ob = s->bytes[i];
         oc = s->branch[i];
         return true;
      }
      template <int N>
      static bool router_next(cnode<N>* c, int from, uint8_t& ob, packed_ptr& oc) noexcept
      {
         constexpr int SEGW = cnode<N>::SEGW, SHIFT = cnode<N>::SHIFT, MASKB = cnode<N>::MASKB;
         for (int sg = from >> SHIFT; sg < N; ++sg)
         {
            const typename cnode<N>::seg* s  = &c->segs[sg];
            const int                     lo = (sg == from >> SHIFT) ? (from & (SEGW - 1)) : 0;
            for (int w = lo >> 6; w < (MASKB + 7) / 8; ++w)
            {
               uint64_t bits = bm_word(s->present, w);
               if constexpr (MASKB % 8 != 0)
                  bits &= (1ull << (MASKB * 8)) - 1;  // mask the documented over-read (c8: 4-byte map)
               if (w == lo >> 6 && (lo & 63))
                  bits &= ~0ull << (lo & 63);
               if (bits)
               {
                  const int d = w * 64 + std::countr_zero(bits);
                  ob = uint8_t(sg * SEGW + d);
                  oc = s->branch[bm_rank(s->present, d)];
                  return true;
               }
            }
         }
         return false;
      }
      static bool router_next(node_full* f, int from, uint8_t& ob, packed_ptr& oc) noexcept
      {
         unsigned li, sl;
         full_index<Ptr>(uint8_t(from), li, sl);
         for (; li < full_lines_v<Ptr>; ++li, sl = 0)
         {
            const cline_direct_t<Ptr>& ln = f->ranges[li];
            if (ln.nset == 0)
               continue;  // empty line: population is tracked
            for (; sl < full_per_line_v<Ptr>; ++sl)
            {
               const unsigned byte = li * full_per_line_v<Ptr> + sl;
               if (byte > 255)
                  return false;
               if (!ln.ptr[sl].is_null())
               {
                  ob = uint8_t(byte);
                  oc = ln.ptr[sl];
                  return true;
               }
            }
         }
         return false;
      }

      // Last child with byte <= from (reverse-iterator stepping) — mirror of router_next:
      // bytes[] sorted ascending ⇒ the last index <= from is (count of bytes <= from) - 1.
      // Counted (branchless) for the same reason router_next is — no mispredicting exit.
      static bool router_prev(setlist* s, int from, uint8_t& ob, packed_ptr& oc) noexcept
      {
         const int n   = s->rh.hdr.nbranch;
         int       cnt = 0;
         for (int j = 0; j < n; ++j) cnt += (int(s->bytes[j]) <= from);
         if (cnt == 0) return false;
         const int i = cnt - 1;
         ob = s->bytes[i];
         oc = s->branch[i];
         return true;
      }
      template <int N>
      static bool router_prev(cnode<N>* c, int from, uint8_t& ob, packed_ptr& oc) noexcept
      {
         constexpr int SEGW = cnode<N>::SEGW, SHIFT = cnode<N>::SHIFT, MASKB = cnode<N>::MASKB;
         for (int sg = from >> SHIFT; sg >= 0; --sg)
         {
            const typename cnode<N>::seg* s  = &c->segs[sg];
            const int                     hi = (sg == from >> SHIFT) ? (from & (SEGW - 1)) : (SEGW - 1);
            for (int w = hi >> 6; w >= 0; --w)
            {
               uint64_t bits = bm_word(s->present, w);
               if constexpr (MASKB % 8 != 0)
                  bits &= (1ull << (MASKB * 8)) - 1;  // mask the documented over-read (c8: 4-byte map)
               if (w == hi >> 6 && (hi & 63) != 63)
                  bits &= (2ull << (hi & 63)) - 1;  // keep bits [0, hi&63]
               if (bits)
               {
                  const int d = w * 64 + 63 - std::countl_zero(bits);
                  ob = uint8_t(sg * SEGW + d);
                  oc = s->branch[bm_rank(s->present, d)];
                  return true;
               }
            }
         }
         return false;
      }
      static bool router_prev(node_full* f, int from, uint8_t& ob, packed_ptr& oc) noexcept
      {
         unsigned li, sl;
         full_index<Ptr>(uint8_t(from), li, sl);
         for (int L = int(li); L >= 0; --L)
         {
            const cline_direct_t<Ptr>& ln = f->ranges[size_t(L)];
            const int s0 = (L == int(li)) ? int(sl) : int(full_per_line_v<Ptr>) - 1;
            if (ln.nset == 0)
               continue;  // empty line: population is tracked
            for (int S = s0; S >= 0; --S)
            {
               if (!ln.ptr[S].is_null())
               {
                  ob = uint8_t(unsigned(L) * full_per_line_v<Ptr> + unsigned(S));
                  oc = ln.ptr[S];
                  return true;
               }
            }
         }
         return false;
      }

      // The slot holding byte's child if present, else nullptr — ONE dispatch for the
      // insert descend (find-or-add), instead of a separate find + child_slot pair.
      static packed_ptr* router_find_slot(setlist* s, uint8_t byte) noexcept
      {
         const int i = setlist_index(s, byte);
         return i >= 0 ? &s->branch[i] : nullptr;
      }
      template <int N>
      static packed_ptr* router_find_slot(cnode<N>* c, uint8_t byte) noexcept
      {
         return cnode_find_slot(c, byte);
      }
      static packed_ptr* router_find_slot(node_full* f, uint8_t byte) noexcept
      {
         unsigned li, sl;
         full_index<Ptr>(byte, li, sl);
         packed_ptr* p = &f->ranges[li].ptr[sl];
         return p->is_null() ? nullptr : p;
      }
      packed_ptr* router_find_slot(packed_ptr cur, uint8_t byte) const noexcept
      {
         return with_router(cur, [byte](auto* n) noexcept { return router_find_slot(n, byte); });
      }

      // Update-or-insert byte -> p; false when the tier is full (the widen signal).
      static bool router_try_set(setlist* s, uint8_t byte, packed_ptr p) noexcept { return setlist_set(s, byte, p); }
      template <int N>
      static bool router_try_set(cnode<N>* c, uint8_t byte, packed_ptr p) noexcept { return cnode_set(c, byte, p); }
      static bool router_try_set(node_full* f, uint8_t byte, packed_ptr p) noexcept
      {
         full_set(f, byte, p);  // node_full never overflows
         return true;
      }
      bool router_try_set(packed_ptr cur, uint8_t byte, packed_ptr p) const noexcept
      {
         return with_router(cur, [&](auto* n) noexcept { return router_try_set(n, byte, p); });
      }

      // Remove byte's branch (keeps each tier's invariants: setlist gap, cnode rank/present,
      // full direct slot + nset).
      static void router_remove(setlist* s, uint8_t byte) noexcept { setlist_remove(s, byte); }
      template <int N>
      static void router_remove(cnode<N>* c, uint8_t byte) noexcept { cnode_remove(c, byte); }
      static void router_remove(node_full* f, uint8_t byte) noexcept { full_set(f, byte, packed_ptr::null()); }
      void router_remove(packed_ptr cur, uint8_t byte) const noexcept
      {
         if (cur.tag() == K::setlist_u8)  // inline-aware: keep inl in sync with branch[]
         {
            setlist* s = static_cast<setlist*>(deref_(cur));
            if (sl_inl_active(s))
            {
               const int i = setlist_index(s, byte);
               if (i >= 0) sl_remove_branch_(s, i);
               return;
            }
         }
         with_router(cur, [byte](auto* n) noexcept { router_remove(n, byte); });
      }

      // Visit (byte, branch) ascending, per tier; the dispatcher also accepts setlist_u16
      // for STRUCTURAL callers (destroy/census/fusion) — they get the stem's high byte and
      // recurse the branch, never route on it.
      template <class F>
      static void router_for_each(setlist* s, F&& f)
      {
         for (int i = 0; i < s->rh.hdr.nbranch; ++i) f(s->bytes[i], s->branch[i]);
      }
      template <int N, class F>
      static void router_for_each(cnode<N>* c, F&& f)
      {
         cnode_for_each(c, f);
      }
      template <class F>
      static void router_for_each(node_full* fn, F&& f)
      {
         for (unsigned li = 0; li < full_lines_v<Ptr>; ++li)
         {
            if (fn->ranges[li].nset == 0)  // skip empty lines (population is tracked)
               continue;
            for (unsigned sl = 0; sl < full_per_line_v<Ptr>; ++sl)
            {
               const unsigned byte = li * full_per_line_v<Ptr> + sl;
               if (byte < 256 && !fn->ranges[li].ptr[sl].is_null())
                  f(uint8_t(byte), fn->ranges[li].ptr[sl]);
            }
         }
      }
      template <class F>
      void router_for_each(packed_ptr cur, F&& f) const
      {
         if (cur.tag() == K::setlist_u16)
         {
            const setlist16* s = static_cast<const setlist16*>(deref_(cur));
            for (int i = 0; i < s->rh.hdr.nbranch; ++i) f(uint8_t(s->stems[i] >> 8), s->branch[i]);
            return;
         }
         with_router(cur, [&](auto* n) { router_for_each(n, f); });
      }

      // Widen a full router. Default (flat): setlist→node_full directly. With the cnode density
      // ladder on (mode::dense_tiers or a small capacity hint — a perf-per-byte win for small
      // maps), climb setlist→c2→c4→node_full instead: sparse routers stay compact cnodes (less
      // memory + tighter ordered iteration than a sparse full). Occupancy is bimodal — a router
      // ends at ~30 branches or ~256, little between — so c4→full is the right jump and the c8
      // rung is dead weight (skipped unless LadderC8, kept for adversarial mid-density clusters).
      packed_ptr widen(packed_ptr cur)
      {
         const K k = cur.tag();
         // cN / node_full have no inline region: a setlist's inline leaves must become
         // real terminal leaves before the rebuild copies its branches.
         if (k == K::setlist_u8)
         {
            setlist* s = static_cast<setlist*>(deref_(cur));
            if (sl_inl_active(s)) sl_externalize_all_(s);  // throws bad_alloc → caller's guard
         }
         const size_t os = node_size(k);
         packed_ptr   out;
         if constexpr (!DenseTiers)
            out = build_wider<node_full>(cur, K::node_full);  // flat_full: straight to full
         else if (k == K::setlist_u8) out = build_wider<cnode<2>>(cur, K::c2);
         else if (k == K::c2)         out = build_wider<cnode<4>>(cur, K::c4);
         else if (k == K::c4)
         {
            if constexpr (LadderC8) out = build_wider<cnode<8>>(cur, K::c8);
            else                    out = build_wider<node_full>(cur, K::node_full);  // skip dead c8
         }
         else if (k == K::c8)         out = build_wider<node_full>(cur, K::node_full);  // only if LadderC8
         else { assert(false); __builtin_unreachable(); }
         node_free_(deref_(cur), os);
         return out;
      }
      template <class N>
      packed_ptr build_wider(packed_ptr src, K newtag)
      {
         N* n = ::new (node_alloc_(sizeof(N))) N;
         if constexpr (std::is_same_v<N, node_full>)
            full_init(n);
         else
            cnode_init(n);
         router_hdr* srh      = static_cast<router_hdr*>(deref_(src));
         n->rh.hdr.has_term   = srh->hdr.has_term;
         n->rh.term           = srh->term;
         packed_ptr np        = pack_(n, newtag);
         router_for_each(src, [&](uint8_t b, packed_ptr br) {
            const bool ok = router_try_set(np, b, br);
            assert(ok && "next tier always has room for a rebuild");
            (void)ok;
         });
         return np;
      }
      // Grow the router at *slot until byte->child fits, then leave the result in *slot. The
      // widen ladder (setlist→c2→c4→c8→full under dense tiers) can take MULTIPLE steps, and the
      // first widen frees the original node — so we re-anchor *slot after EVERY widen: a later
      // widen throwing then leaves *slot pointing at a valid (wider) node, never a freed one,
      // and the key set is unchanged (the new child simply wasn't added). A setlist's inline
      // prefix is first lifted into a prefix_node so re-anchoring through the ladder can't drop
      // it. `child` must be guarded by the caller (a throw here orphans it).
      void grow_router_(packed_ptr* slot, packed_ptr cur, uint8_t byte, packed_ptr child)
      {
         char* pn = nullptr;  // when set, re-anchor target is this prefix_node's next field
         if (cur.tag() == K::setlist_u8)
         {
            setlist*       s  = static_cast<setlist*>(deref_(cur));
            const unsigned pl = s->rh.hdr.prefix_len;
            if (pl)
            {
               uint8_t pfx[setlist::PREFIX_CAP];
               std::memcpy(pfx, s->prefix, pl);
               packed_ptr pnp = make_prefix(std::string_view(reinterpret_cast<char*>(pfx), pl), cur);  // throws → unchanged
               s->rh.hdr.prefix_len = 0;  // prefix now carried by pnp above the (to-be-widened) router
               *slot = pnp;
               pn    = static_cast<char*>(deref_(pnp));
            }
         }
         packed_ptr node = cur;
         while (!router_try_set(node, byte, child))
         {
            node = widen(node);  // frees the old node; re-anchor before the next (throwing) widen
            if (pn) pn_set_next(pn, node);
            else    *slot = node;
         }
         // The ladder is done: if the lifted prefix now sits above a node_full, fuse it
         // back into the full's header cacheline and free the prefix node — prefix_node
         // is the overflow form, not the home (a setlist prefix is <= 8, always fits).
         if (pn && node.tag() == K::node_full)
         {
            const uint16_t pl = pn_plen(pn);
            node_full*     f  = static_cast<node_full*>(deref_(node));
            if (pl <= node_full::PFX_CAP)
            {
               std::memcpy(f->hdr + 1, pn_pfx<Ptr>(pn), pl);
               f->hdr[0]            = uint8_t(K::node_full);
               f->rh.hdr.prefix_len = pl & 0x3F;
               *slot                = pack_(f, K::pfxd);
               node_free_(pn, pn_size<Ptr>(pl));
            }
         }
      }

      template <bool Assign, class VF>
      bool set_or_update_term(router_hdr* rh, VF&& val)
      {
         if (!rh->term.is_null())  // present: overwrite (insert) or leave (emplace)
         {
            if constexpr (Assign)
            {
               if (rh->term.tag() == K::value_inline)
                  rh->term = pack_inline(vf_make(std::forward<VF>(val)));
               else
                  vf_assign(std::forward<VF>(val), leaf_val(static_cast<char*>(deref_term_(rh->term))));
            }
            return false;
         }
         set_term(rh, std::forward<VF>(val));
         return true;
      }

      // Split a leaf whose suffix diverges from the key: build a setlist holding the
      // old value and the new one, optionally under a shared-prefix node. Returns the
      // new subtree (or the same leaf, value updated, when the key matches exactly).
      template <bool Assign, class VF>
      packed_ptr split_leaf(packed_ptr cur, std::string_view key, size_t depth, VF&& val, bool& inserted)
      {
         // `if constexpr`: a value_inline leaf only exists for an inlineable T, so this whole
         // branch is dead for non-inlineable T — eliding it avoids instantiating unpack_inline
         // (a memcpy into a possibly non-trivially-copyable T) that would never run.
         if constexpr (inlineable)
            if (cur.tag() == K::value_inline)
            {
               // Inline terminal: stored key ended here (empty suffix). No allocation to free.
               std::string_view R = key.substr(depth);
               if (R.empty())  // exact key
               {
                  inserted = false;
                  if constexpr (Assign) return pack_inline(vf_make(std::forward<VF>(val)));
                  else return cur;
               }
               T oldv;
               unpack_inline(cur, oldv);  // inlineable T is trivially copyable
               inserted = true;
               setlist* r = new_setlist();
               set_term(&r->rh, std::move(oldv));                                    // old key terminates
               setlist_set(r, uint8_t(R[0]), make_value_at(key, depth + 1, std::forward<VF>(val)));  // new continues
               sl_inline_compact_(r);          // pull the small leaf child into this line
               return pack_(r, K::setlist_u8);  // c == 0, no prefix node needed
            }
         char*            L = static_cast<char*>(deref_term_(cur));
         std::string_view S(reinterpret_cast<const char*>(leaf_suf(L)), leaf_slen(L));
         std::string_view R = key.substr(depth);
         if (S == R)  // exact key
         {
            if constexpr (Assign) vf_assign(std::forward<VF>(val), leaf_val(L));
            inserted = false;
            return cur;
         }
         const size_t           c       = lcp(S, R);
         const size_t           Lsize   = leaf_alloc_size(L);  // capture before node_free_ clobbers L
         const bool             oldfull = leaf_has_full(L);
         const std::string_view ofv     = oldfull ? leaf_fullview(L) : std::string_view{};  // into L (read pre-free)

         // Strong exception safety: do every throwing allocation BEFORE touching L. The new
         // value + prefix are built first; the OLD value is moved out LAST — and make_value
         // allocates then moves, so even that final alloc failing leaves L's value in place.
         // Any bad_alloc therefore leaves L (and the whole container) unchanged.
         setlist*   r       = new_setlist();                         // (1)
         packed_ptr partial = pack_(r, K::setlist_u8);
         build_guard g{this, &partial};  // a later alloc throwing must not orphan the new value
         if (c == R.size())                                           // (2) new value (consumes val)
            set_term(&r->rh, std::forward<VF>(val));
         else
            setlist_set(r, uint8_t(R[c]), make_value_at(key, depth + c + 1, std::forward<VF>(val)));
         partial = apply_prefix(r, key.substr(depth, c));             // (3) shared prefix (== S[0..c])
         if (c == S.size())                                           // (4) old value, LAST
            set_term(&r->rh, std::move(*leaf_val(L)));
         else if (oldfull)  // reuse the old leaf's contiguous full key so the child stays full
            setlist_set(r, uint8_t(S[c]), make_value_at(ofv, depth + c + 1, std::move(*leaf_val(L))));
         else
            setlist_set(r, uint8_t(S[c]), make_value(S.substr(c + 1), std::move(*leaf_val(L))));
         v_destroy(alloc_, leaf_val(L));  // old value moved out → destroy moved-from + free
         term_free_(L, Lsize);
         g.release();  // committed: the caller adopts `partial`
         sl_inline_compact_(r);  // both children are small leaves → pull them into this line
         inserted = true;
         return partial;
      }

      // Split a prefix_node whose prefix diverges from the key at position c (< plen): a new
      // parent setlist routes the new value's branch and the old continuation. The new value
      // branch + outer prefix are built while `r` references ONLY new nodes, so a bad_alloc
      // lets the guard reclaim it without touching the live child `next`; the graft of `next`
      // (and freeing the old node) is a non-throwing tail. make_branch(v) builds the terminal
      // for the diverging remainder key[depth+c+1:] — a leaf in radix mode, a bucket in
      // bucket mode (the two callers differ in nothing else).
      template <class MakeBranch, class VF>
      packed_ptr split_prefix(packed_ptr cur, std::string_view key, size_t depth, size_t c,
                              VF&& val, MakeBranch&& make_branch)
      {
         char*            P = static_cast<char*>(deref_(cur));
         std::string_view PP(reinterpret_cast<const char*>(pn_pfx<Ptr>(P)), pn_plen(P));
         packed_ptr       next  = pn_next<Ptr>(P);
         const size_t     Psize = pn_size<Ptr>(PP.size());  // capture before node_free_ clobbers P
         std::string_view R     = key.substr(depth);
         setlist*         r       = new_setlist();
         packed_ptr       partial = pack_(r, K::setlist_u8);
         build_guard      g{this, &partial};
         if (c == R.size())
            set_term(&r->rh, std::forward<VF>(val));  // the new key ends inside the prefix
         else
            setlist_set(r, uint8_t(R[c]), make_branch(std::forward<VF>(val)));
         partial = apply_prefix(r, key.substr(depth, c));  // == PP[0..c]; inline if short
         packed_ptr cont = (c + 1 == PP.size()) ? next : make_prefix(PP.substr(c + 1), next);
         setlist_set(r, uint8_t(PP[c]), cont);  // non-throwing graft of the live child
         g.release();
         node_free_(P, Psize);
         sl_inline_compact_(r);  // the new-value branch is a small leaf → pull it inline
         return partial;
      }

      // Split a setlist's INLINE prefix that diverges from the key at position c (< prefix_len):
      // a new parent setlist takes the shared part [0..c), routes the new value's branch, and
      // routes the old continuation byte back to this setlist (re-prefixed in place to [c+1..)).
      // Everything that can throw happens BEFORE the live setlist is grafted or mutated, so a
      // bad_alloc leaves the container unchanged (strong) and the guard reclaims only new nodes
      // — never `cur`. make_branch as in split_prefix.
      template <class MakeBranch, class VF>
      packed_ptr split_setlist_prefix(packed_ptr cur, std::string_view key, size_t depth,
                                      size_t c, VF&& val, MakeBranch&& make_branch)
      {
         setlist*       s   = static_cast<setlist*>(deref_(cur));
         const unsigned pl  = s->rh.hdr.prefix_len;
         uint8_t        common[setlist::PREFIX_CAP];
         std::memcpy(common, s->prefix, c);              // shared part [0..c)  (read, no mutation)
         const uint8_t  oldbyte = s->prefix[c];          // byte leading to the old setlist
         const unsigned newpl   = pl - unsigned(c) - 1;  // old setlist keeps [c+1..pl)
         setlist*       r       = new_setlist();
         packed_ptr     partial = pack_(r, K::setlist_u8);
         build_guard    g{this, &partial};
         std::string_view R = key.substr(depth);
         if (c == R.size())
            set_term(&r->rh, std::forward<VF>(val));  // the new key ends inside the prefix
         else
            setlist_set(r, uint8_t(R[c]), make_branch(std::forward<VF>(val)));
         partial = apply_prefix(r, std::string_view(reinterpret_cast<char*>(common), c));
         // commit (non-throwing): graft the live setlist, then re-prefix s in place
         setlist_set(r, oldbyte, cur);  // route old byte -> s (s still has its full prefix)
         std::memmove(s->prefix, s->prefix + c + 1, newpl);
         s->rh.hdr.prefix_len = newpl;
         g.release();
         sl_inline_compact_(r);  // the new-value branch is a small leaf → pull it inline
         return partial;
      }

      // Split a FUSED in-header prefix (K::pfxd) diverging from the key at c (< plen):
      // same shape as split_setlist_prefix — a new parent setlist takes the shared part
      // [0..c), routes the new value's branch, and routes the old continuation byte back
      // to the node, whose fused prefix shortens in place to [c+1..) (the handle retags
      // to the real kind when the prefix empties — the header is no longer needed).
      template <class MakeBranch, class VF>
      packed_ptr split_pfxd_prefix(packed_ptr cur, std::string_view key, size_t depth,
                                   size_t c, VF&& val, MakeBranch&& make_branch)
      {
         void*          n  = deref_(cur);
         uint8_t*       pb = pfxd_pfx(n);
         const unsigned pl = pfxd_plen(n);
         uint8_t        common[node_full::PFX_CAP];
         std::memcpy(common, pb, c);          // shared part [0..c) (read, no mutation)
         const uint8_t  oldbyte = pb[c];      // byte leading back to this node
         const unsigned newpl   = pl - unsigned(c) - 1;
         setlist*       r       = new_setlist();
         packed_ptr     partial = pack_(r, K::setlist_u8);
         build_guard    g{this, &partial};
         std::string_view R = key.substr(depth);
         if (c == R.size())
            set_term(&r->rh, std::forward<VF>(val));  // the new key ends inside the prefix
         else
            setlist_set(r, uint8_t(R[c]), make_branch(std::forward<VF>(val)));
         partial = apply_prefix(r, std::string_view(reinterpret_cast<char*>(common), c));
         // commit (non-throwing): shorten the fused prefix in place, then graft
         setlist_set(r, oldbyte, pfxd_set_prefix(n, pb + c + 1, newpl));
         g.release();
         sl_inline_compact_(r);  // the new-value branch is a small leaf → pull it inline
         return partial;
      }

      // ── structural clone: node-by-node move transport ────────────────────────────
      // Move-assignment fallback for unequal, non-propagating allocators: clone the
      // source tree's STRUCTURE into our allocator — allocate a same-size node, memcpy
      // its bytes, then rewrite each child handle to the clone of that child. Values
      // ride the node memcpy when trivially copyable; otherwise they are move-
      // constructed from the source's live objects (the source is drained by o.clear()
      // afterwards). No key encoding, no descents, no splits, no re-balancing churn:
      // O(total node bytes), and the clone is structurally IDENTICAL to the source.
      // (Key-by-key reinsert — the naive fallback — pays a full descent plus split/widen
      // work per element and rebuilds the whole shape from scratch.)
      //
      // Exception safety: every copied node has its child slots nulled before any
      // throwing recursion and is owned by a build_guard, so a bad_alloc mid-clone
      // reclaims exactly the part built so far (free_all_ tolerates null children).
      void move_elements_from_(map& o)
      {
         root_  = clone_walk_(o, o.root_);
         count_ = o.count_;
      }
      // Clone one router node (setlist / cN / node_full): the memcpy preserves the
      // edge directory (sorted bytes / presence bitmaps / direct slots); each branch
      // slot is then re-pointed at the cloned child via router_try_set on its own edge.
      template <class Node>
      packed_ptr clone_router_(const map& src, const Node* S, K k)
      {
         Node* D = ::new (node_alloc_(sizeof(Node))) Node;
         std::memcpy(static_cast<void*>(D), static_cast<const void*>(S), sizeof(Node));
         const router_hdr* sh = reinterpret_cast<const router_hdr*>(S);
         router_hdr*       dh = reinterpret_cast<router_hdr*>(D);
         dh->term             = packed_ptr::null();
         router_for_each(const_cast<Node*>(S),  // null every slot before recursion can throw
                         [&](uint8_t e, packed_ptr) { router_try_set(D, e, packed_ptr::null()); });
         packed_ptr  dp = pack_(D, k);
         build_guard g{this, &dp};
         dh->term = clone_walk_(src, sh->term);
         router_for_each(const_cast<Node*>(S),
                         [&](uint8_t e, packed_ptr child) { router_try_set(D, e, clone_walk_(src, child)); });
         g.release();
         return dp;
      }
      packed_ptr clone_walk_(const map& src, packed_ptr s)
      {
         switch (s.tag())
         {
            case K::null:
               return packed_ptr::null();
            case K::value_inline:
               return s;  // the handle IS the value (handle widths match: same tree type)
            case K::value_ptr:
            {
               char*        L  = static_cast<char*>(src.deref_term_(s));
               const size_t sz = leaf_alloc_size(L);
               char*        D  = static_cast<char*>(term_alloc_(sz));
               std::memcpy(D, L, sz);  // header + suffix/full-key bytes (+ value if trivial)
               if constexpr (!std::is_trivially_copyable_v<T>)
               {
                  try { v_construct(alloc_, leaf_val(D), std::move(*leaf_val(L))); }
                  catch (...) { term_free_(D, sz); throw; }
               }
               return pack_term_(D, K::value_ptr);
            }
            case K::prefix_node:
            {
               char*        P  = static_cast<char*>(src.deref_(s));
               const size_t sz = pn_size<Ptr>(pn_plen(P));
               char*        D  = static_cast<char*>(node_alloc_(sz));
               std::memcpy(D, P, sz);
               pn_set_next(D, packed_ptr::null());
               packed_ptr  dp = pack_(D, K::prefix_node);
               build_guard g{this, &dp};
               pn_set_next(D, clone_walk_(src, pn_next<Ptr>(P)));
               g.release();
               return dp;
            }
            case K::setlist_u16:  // parallel sorted arrays: clone children index-by-index
            {
               const setlist16* S = static_cast<const setlist16*>(src.deref_(s));
               setlist16*       D = ::new (node_alloc_(sizeof(setlist16))) setlist16;
               std::memcpy(static_cast<void*>(D), static_cast<const void*>(S), sizeof(setlist16));
               const int nb = D->rh.hdr.nbranch;
               D->rh.term   = packed_ptr::null();
               for (int i = 0; i < nb; ++i) D->branch[i] = packed_ptr::null();
               packed_ptr  dp = pack_(D, K::setlist_u16);
               build_guard g{this, &dp};
               D->rh.term = clone_walk_(src, S->rh.term);
               for (int i = 0; i < nb; ++i) D->branch[i] = clone_walk_(src, S->branch[i]);
               g.release();
               return dp;
            }
            case K::setlist_u8:
            {
               // Copy the WHOLE line: inline-leaf payloads live in the tail (possibly past
               // sizeof(setlist)) and their branch handles are line-relative, so the memcpy
               // carries them verbatim. Only EXTERNAL children are re-pointed to clones;
               // inline branches stay as the memcpy left them.
               const setlist* S = static_cast<const setlist*>(src.deref_(s));
               setlist*       D = ::new (node_alloc_(sizeof(setlist))) setlist;
               std::memcpy(static_cast<void*>(D), static_cast<const void*>(S), SL_LINE);
               const int nb = S->rh.hdr.nbranch;
               D->rh.term   = packed_ptr::null();
               for (int i = 0; i < nb; ++i)
                  if (!sl_is_inline(S, i)) D->branch[i] = packed_ptr::null();  // null before throws
               packed_ptr  dp = pack_(D, K::setlist_u8);
               build_guard g{this, &dp};
               D->rh.term = clone_walk_(src, S->rh.term);
               for (int i = 0; i < nb; ++i)
                  if (!sl_is_inline(S, i)) D->branch[i] = clone_walk_(src, S->branch[i]);
               g.release();
               return dp;
            }
            case K::pfxd:  // full body + in-header prefix; the memcpy carries both
               return clone_router_(src, static_cast<const node_full*>(src.deref_(s)), K::pfxd);
            case K::c2:
               return clone_router_(src, static_cast<const cnode<2>*>(src.deref_(s)), K::c2);
            case K::c4:
               return clone_router_(src, static_cast<const cnode<4>*>(src.deref_(s)), K::c4);
            case K::c8:
               return clone_router_(src, static_cast<const cnode<8>*>(src.deref_(s)), K::c8);
            case K::node_full:
               return clone_router_(src, static_cast<const node_full*>(src.deref_(s)), K::node_full);
            case K::bucket:
            {
               const bucket* B = static_cast<const bucket*>(src.deref_(s));
               bucket*       D = ::new (node_alloc_(sizeof(bucket))) bucket;
               std::memcpy(static_cast<void*>(D), static_cast<const void*>(B), sizeof(bucket));
               if constexpr (!std::is_trivially_copyable_v<T>)
               {
                  int i = 0;
                  try
                  {
                     for (; i < D->nent; ++i)
                        v_construct(alloc_, bkt_val(bkt_entry(D, i)),
                                    std::move(*bkt_val(bkt_entry(const_cast<bucket*>(B), i))));
                  }
                  catch (...)
                  {
                     for (int j = 0; j < i; ++j) v_destroy(alloc_, bkt_val(bkt_entry(D, j)));
                     node_free_(D, sizeof(bucket));
                     throw;
                  }
               }
               return pack_(D, K::bucket);
            }
         }
         return packed_ptr::null();  // unreachable (exhaustive switch)
      }

      // Tear down a (possibly partial) subtree: destroy each T AND free each node. Drives
      // teardown (~map / clear), and the exception-safety guard's cleanup of a half-built
      // subtree when a mutate op throws before committing it.
      void free_all_(packed_ptr cur) noexcept
      {
         if (cur.is_null())
            return;
         switch (cur.tag())
         {
            case K::value_ptr:
            {
               char* L = static_cast<char*>(deref_term_(cur));
               if constexpr (!std::is_trivially_destructible_v<T>) v_destroy(alloc_, leaf_val(L));
               term_free_(L, leaf_alloc_size(L));
               return;
            }
            case K::value_inline:
               return;
            case K::prefix_node:
            {
               char* P = static_cast<char*>(deref_(cur));
               free_all_(pn_next<Ptr>(P));
               node_free_(P, pn_size<Ptr>(pn_plen(P)));
               return;
            }
            case K::setlist_u8:
            {
               setlist* s = static_cast<setlist*>(deref_(cur));
               free_all_(s->rh.term);
               const int n = s->rh.hdr.nbranch;
               for (int i = 0; i < n; ++i)
                  if (!sl_is_inline(s, i)) free_all_(s->branch[i]);  // inline leaves ride the line
               node_free_(s, node_size(K::setlist_u8));
               return;
            }
            case K::setlist_u16:
            case K::c2:
            case K::c4:
            case K::c8:
            case K::node_full:
            case K::pfxd:
            {
               router_hdr* rh = static_cast<router_hdr*>(deref_(cur));
               free_all_(rh->term);
               router_for_each(cur, [&](uint8_t, packed_ptr br) { free_all_(br); });
               node_free_(deref_(cur), node_size(cur.tag()));
               return;
            }
            case K::bucket:
            {
               bucket* b = static_cast<bucket*>(deref_(cur));
               if constexpr (!std::is_trivially_destructible_v<T>)
                  for (int i = 0; i < b->nent; ++i) v_destroy(alloc_, bkt_val(bkt_entry(b, i)));
               node_free_(b, sizeof(bucket));
               return;
            }
            case K::null:
               return;
         }
      }
      // Nodes are allocated per-node (no bulk-reclaiming pool), so a node orphaned on a throwing
      // mid-build path is a real memory leak for ANY T — the guard must always reclaim. (Kept as
      // a flag so a future bulk-reclaiming allocator could set it true and elide the guard for
      // trivially-destructible T, as the old pool did.)
      static constexpr bool pool_bulk_reclaims = false;
      static constexpr bool guard_needed = !std::is_trivially_destructible_v<T> || !pool_bulk_reclaims;

      // RAII: frees the tracked partial subtree on unwind unless committed. Point `*slot` at the
      // outermost partial node as a mutate op builds it; call release() at the noexcept commit.
      struct build_guard
      {
         map*    self;
         packed_ptr* slot;
         bool        armed = true;
         ~build_guard()
         {
            if constexpr (guard_needed)
               if (armed && slot && !slot->is_null()) self->free_all_(*slot);
         }
         void release() noexcept { armed = false; }
      };

      // RAII for restride_u16_to_u8: on unwind, free the freshly-built u8 setlists SHELL-ONLY
      // (the new parent + its child setlists). The grandchildren they route to are still owned
      // by the live u16 node, so they must NOT be recursed into (that would double-free them).
      struct restride_guard
      {
         map* self;
         setlist* parent;
         bool     armed = true;
         ~restride_guard()
         {
            if (!armed) return;
            const int nb = parent->rh.hdr.nbranch;
            for (int i = 0; i < nb; ++i)
               self->node_free_(self->deref_(parent->branch[i]), sizeof(setlist));  // child setlists, shell-only
            self->node_free_(parent, sizeof(setlist));
         }
         void release() noexcept { armed = false; }
      };

      [[no_unique_address]] Allocator alloc_{};  // element (T) allocator; rebound per-node for bytes
      [[no_unique_address]] std::conditional_t<indexed, const std::byte*, no_base_t> base_{};
      [[no_unique_address]] std::conditional_t<indexed, const std::byte*, no_base_t> tbase_{};
      packed_ptr                      root_  = packed_ptr::null();
      size_t                          count_ = 0;
   };
}  // namespace detail

using detail::map;  // re-export only the public class; impl stays in artpp::detail

// Capacity-hint convenience. Declaring the expected element count auto-selects the cnode density
// ladder when that count is small enough to be a perf-per-byte win — compact sparse routers, on
// the line_pool ~40% less RAM and ~2x faster build below ~3M keys (vs the flat default). Above the
// measured crossover it transparently stays flat, since the ladder would then only add build churn
// and pool fragmentation. Ergonomic spelling of map<K, V, mode::none, Allocator, ExpectedN>; for
// other modes (buckets, wide_stems, …) combine with mode::dense_tiers on the map<> template direct.
template <class Key, class T, std::size_t ExpectedN, class Allocator = std::allocator<T>>
using compact_map = detail::map<Key, T, mode::none, Allocator, ExpectedN>;
}  // namespace artpp

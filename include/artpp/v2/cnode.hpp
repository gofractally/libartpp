// Dense segmented bitset routers (design4 node_c2 / c4 / c8): the tier between a
// sparse setlist and the direct-index node_full. The 256-byte key range is split
// into NSEG equal segment cachelines (2x128 / 4x64 / 8x32). Each segment is a
// self-contained rank-dense line: a presence bitmap + a packed 6-byte branch array
// where a present byte maps to slot = popcount(present below it). CL0 is the header
// only — descent SKIPS it: it computes the segment line from the key byte, prefetches
// it, then rank-decodes (the design's r2/r4/r8 skip-header path).
//
//   cnode<2>: 2x128-range, <=18 branches/seg, 3 cachelines (384 B)
//   cnode<4>: 4x 64-range, <=20 branches/seg, 5 cachelines (640 B)
//   cnode<8>: 8x 32-range, <=20 branches/seg, 9 cachelines (1152 B)
#pragma once
#include <bit>
#include <cstdint>
#include <cstring>

#include "artpp/v2/tagged_ptr.hpp"

namespace artpp::v2::detail
{
   // ── presence-bitmap helpers — u64-word math, never a byte loop ───────────────
   // The maps are 4/8/16 bytes (SEGW 32/64/128). Loads are full 64-bit words; the
   // 32-wide map's word over-reads 4 bytes into the adjacent branch array (same node
   // allocation, always present) and masks them off, so popcount/rank stay a single
   // 64-bit popcount (or two for the 128-wide map) — no per-byte iteration.
   inline uint64_t bm_word(const uint8_t* m, int word) noexcept
   {
      uint64_t w;
      std::memcpy(&w, m + word * 8, 8);
      return w;
   }
   inline bool bm_test(const uint8_t* m, int d) noexcept { return (m[d >> 3] >> (d & 7)) & 1; }
   inline void bm_set(uint8_t* m, int d) noexcept { m[d >> 3] |= uint8_t(1u << (d & 7)); }

   // popcount over MASKB bytes; MASKB<=8 → one word (masked), else two.
   inline int bm_count(const uint8_t* m, int maskb) noexcept
   {
      if (maskb <= 8)
      {
         const uint64_t lo = maskb >= 8 ? ~0ull : ((1ull << (maskb * 8)) - 1);
         return std::popcount(bm_word(m, 0) & lo);
      }
      return std::popcount(bm_word(m, 0)) + std::popcount(bm_word(m, 1));
   }
   // rank = set bits strictly below `d` = the dense branch slot for `d`. One masked
   // 64-bit popcount (two when d crosses into the high word of a 128-wide map).
   inline int bm_rank(const uint8_t* m, int d) noexcept
   {
      if (d < 64)
         return std::popcount(bm_word(m, 0) & ((d == 0) ? 0ull : (~0ull >> (64 - d))));
      return std::popcount(bm_word(m, 0)) +
             std::popcount(bm_word(m, 1) & ((d == 64) ? 0ull : (~0ull >> (128 - d))));
   }

   template <int NSEG, class Ptr>
   struct cnode_t
   {
      static constexpr int SEGW  = 256 / NSEG;                          // 128 / 64 / 32
      static constexpr int SHIFT = NSEG == 2 ? 7 : NSEG == 4 ? 6 : 5;
      static constexpr int MASKB = SEGW / 8;                            // 16 / 8 / 4
      static constexpr int CAP   = (128 - MASKB) / int(sizeof(Ptr));    // 18 / 20 / 20 at 6B

      struct seg
      {
         uint8_t present[MASKB];
         Ptr     branch[CAP];  // rank-dense: branch[rank(d)] is byte (seg*SEGW + d)
         [[no_unique_address]] pad_t<128 - MASKB - CAP * unsigned(sizeof(Ptr))> pad;
      };
      static_assert(sizeof(seg) == 128, "segment tiles one cacheline");

      router_hdr_t<Ptr> rh;            // CL0 header (no inline prefix)
      uint8_t           cl0pad[120];
      seg               segs[NSEG];    // CL1.. one cacheline each
   };
   template <int N>
   using cnode = cnode_t<N, packed_ptr>;  // default-handle alias
   static_assert(sizeof(typename cnode<2>::seg) == 128, "c2 segment is one cacheline");
   static_assert(sizeof(typename cnode<4>::seg) == 128, "c4 segment is one cacheline");
   static_assert(sizeof(typename cnode<8>::seg) == 128, "c8 segment is one cacheline");
   static_assert(sizeof(cnode<2>) == 384, "c2 = 3 cachelines");
   static_assert(sizeof(cnode<4>) == 640, "c4 = 5 cachelines");
   static_assert(sizeof(cnode<8>) == 1152, "c8 = 9 cachelines");

   template <int N, class Ptr>
   inline void cnode_init(cnode_t<N, Ptr>* c) noexcept
   {
      c->rh.hdr  = node_hdr{0, 0, 0};
      c->rh.term = Ptr::null();
      // Clear ONLY the presence bitmaps (MASKB bytes per segment): branch[] is rank-dense —
      // slots >= popcount are never read (word-loads that over-read past present[] mask the
      // excess) — so zeroing whole segments would memset 2-8 cachelines to overwrite 32 bytes.
      for (int sg = 0; sg < N; ++sg)
         std::memset(c->segs[sg].present, 0, cnode_t<N, Ptr>::MASKB);
   }

   // Descent step (skip-header): prefetch the segment line, then rank-decode. Returns
   // the child pointer or null. The prefetch is issued before the rank popcounts run.
   template <int N, class Ptr>
   inline Ptr cnode_step(const cnode_t<N, Ptr>* c, uint8_t byte) noexcept
   {
      const typename cnode_t<N, Ptr>::seg* s = &c->segs[byte >> cnode_t<N, Ptr>::SHIFT];
      const int                            d = byte & (cnode_t<N, Ptr>::SEGW - 1);
      // (No prefetch: the rank-decode reads this same line immediately; prefetching
      // it hides no latency in a pointer-chasing loop — measured dead weight.)
      return bm_test(s->present, d) ? s->branch[bm_rank(s->present, d)] : Ptr::null();
   }
   template <int N, class Ptr>
   inline Ptr cnode_find(const cnode_t<N, Ptr>* c, uint8_t byte) noexcept
   {
      return cnode_step(c, byte);
   }
   // Pointer to the branch slot for `byte` (which must be present) — for in-place
   // descent writeback. Stable across a single insert (no sibling reshuffle mid-insert).
   template <int N, class Ptr>
   inline Ptr* cnode_child_slot(cnode_t<N, Ptr>* c, uint8_t byte) noexcept
   {
      typename cnode_t<N, Ptr>::seg* s = &c->segs[byte >> cnode_t<N, Ptr>::SHIFT];
      return &s->branch[bm_rank(s->present, byte & (cnode_t<N, Ptr>::SEGW - 1))];
   }
   // Slot pointer for `byte` if present, else nullptr (one lookup for find-or-descend).
   template <int N, class Ptr>
   inline Ptr* cnode_find_slot(cnode_t<N, Ptr>* c, uint8_t byte) noexcept
   {
      typename cnode_t<N, Ptr>::seg* s = &c->segs[byte >> cnode_t<N, Ptr>::SHIFT];
      const int                      d = byte & (cnode_t<N, Ptr>::SEGW - 1);
      return bm_test(s->present, d) ? &s->branch[bm_rank(s->present, d)] : nullptr;
   }
   // Update-or-insert into the rank-dense segment. False if THIS segment is full
   // (the widen signal). Maintains rank order via a memmove gap.
   template <int N, class Ptr>
   inline bool cnode_set(cnode_t<N, Ptr>* c, uint8_t byte, Ptr p) noexcept
   {
      typename cnode_t<N, Ptr>::seg* s   = &c->segs[byte >> cnode_t<N, Ptr>::SHIFT];
      const int                      d   = byte & (cnode_t<N, Ptr>::SEGW - 1);
      const int                      pos = bm_rank(s->present, d);
      if (bm_test(s->present, d))
      {
         s->branch[pos] = p;  // replace
         return true;
      }
      const int cnt = bm_count(s->present, cnode_t<N, Ptr>::MASKB);
      if (cnt >= cnode_t<N, Ptr>::CAP)
         return false;  // segment full -> caller widens
      std::memmove(&s->branch[pos + 1], &s->branch[pos], size_t(cnt - pos) * sizeof(Ptr));
      s->branch[pos] = p;
      bm_set(s->present, d);
      ++c->rh.hdr.nbranch;
      return true;
   }
   // Remove byte from the rank-dense segment: close the gap, clear the presence bit.
   template <int N, class Ptr>
   inline void cnode_remove(cnode_t<N, Ptr>* c, uint8_t byte) noexcept
   {
      typename cnode_t<N, Ptr>::seg* s = &c->segs[byte >> cnode_t<N, Ptr>::SHIFT];
      const int                      d = byte & (cnode_t<N, Ptr>::SEGW - 1);
      if (!bm_test(s->present, d))
         return;
      const int cnt = bm_count(s->present, cnode_t<N, Ptr>::MASKB);
      const int pos = bm_rank(s->present, d);
      std::memmove(&s->branch[pos], &s->branch[pos + 1], size_t(cnt - 1 - pos) * sizeof(Ptr));
      s->present[d >> 3] &= uint8_t(~(1u << (d & 7)));
      --c->rh.hdr.nbranch;
   }
   // Visit (absolute byte, branch) in ascending key order — iterate SET bits via ctz
   // over u64 words (O(popcount)), never a scan over the whole SEGW range.
   template <int N, class Ptr, class F>
   inline void cnode_for_each(const cnode_t<N, Ptr>* c, F&& f)
   {
      for (int sg = 0; sg < N; ++sg)
      {
         const typename cnode_t<N, Ptr>::seg* s    = &c->segs[sg];
         const int                            base = sg * cnode_t<N, Ptr>::SEGW;
         int                                  idx  = 0;
         for (int wo = 0; wo < cnode_t<N, Ptr>::MASKB; wo += 8)
         {
            uint64_t  w  = 0;
            const int wb = cnode_t<N, Ptr>::MASKB - wo < 8 ? cnode_t<N, Ptr>::MASKB - wo : 8;
            std::memcpy(&w, s->present + wo, size_t(wb));
            while (w)
            {
               const int b = std::countr_zero(w);
               w &= w - 1;
               f(uint8_t(base + wo * 8 + b), s->branch[idx++]);
            }
         }
      }
   }
}  // namespace artpp::v2::detail

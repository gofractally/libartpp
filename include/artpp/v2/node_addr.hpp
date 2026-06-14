#pragma once
// artpp::v2 routing kernel — the heart of the "pointer is the header" design.
//
// Every router child is reached through the parent's handle, which carries 16 bits of
// navigation metadata interpreted by the child's tier tag. From (tier, metadata, key byte)
// we compute WHICH cacheline of the child to load — before loading it — so a descent is one
// cacheline per level. This header defines just that index function (and the per-tier line
// count + presence test); the node memory layout, descent, and transitions build on top.
//
// Measured cost of the index calc (M5, latency-bound dependent chain; see bench/):
//   setlist                 : 1 line, no index calc
//   sdiv2  (1 divider)      : ~0 ns      idx = (byte >= d0)
//   sdiv3  (2 dividers)     : +0.66 ns   idx = (byte>=d0) + (byte>=d1)
//   sdiv4  (2 div + implicit-128, balanced binary, 4 lines / ~56 cap) : +0.70 ns
//   sparse_full (presence)  : +3.86 ns arm64 / ~free x86 (popcount); only the scattered tail
//   full   (all present)    : rank-free  idx = byte >> 4
// So compare-as-index covers everything to ~56 branches; popcount serves only the wide tail,
// and a genuinely-full node is rank-free again.
#include <cstdint>
#include <cstring>
#include <bit>

namespace artpp::v2
{
   // The v2 branch handle: per-child navigation metadata + a tagged address, templated on
   // total width so a node's branch arrays re-tile from one type. Little-endian byte array:
   //   [ address field : N-2 bytes ][ metadata : 2 bytes ]
   //     * tag      = low 4 bits of byte 0 (node kind / tier)
   //     * address  = the low N-2 bytes, tag in its low alignment bits — resolved by the map's
   //                  deref: a 48-bit pointer when N=8, a ≤28-bit 128B-line index when N=6
   //     * metadata = the top 2 bytes (the 16-bit dividers / presence vector this kernel reads),
   //                  WIDTH-INVARIANT — only the address field shrinks for a smaller pool index
   // N=8 → 6 address bytes (direct/full pointer); N=6 → 4 address bytes (pool index). A setlist
   // branch costs 1+N bytes, so ~14 branches/line at N=8 and ~18 at N=6.
   template <unsigned N>
   struct handle_t
   {
      static_assert(N == 6 || N == 8, "v2 handle is 6 (pool) or 8 (direct) bytes: address + 2 metadata");
      static constexpr unsigned ABYTES = N - 2;
      uint8_t b[N];
      unsigned tag() const noexcept { return b[0] & 0xF; }
      uint16_t meta() const noexcept { uint16_t m; std::memcpy(&m, b + ABYTES, 2); return m; }
      uint64_t addr_field() const noexcept { uint64_t v = 0; std::memcpy(&v, b, ABYTES); return v; }
      static handle_t make(uint64_t addr_with_tag, uint16_t meta) noexcept
      {
         handle_t h;
         std::memcpy(h.b, &addr_with_tag, ABYTES);  // low ABYTES bytes (tag in low 4 bits)
         std::memcpy(h.b + ABYTES, &meta, 2);
         return h;
      }
   };
   using handle6 = handle_t<6>;  // pool:   4-byte index  + 2 metadata
   using handle8 = handle_t<8>;  // direct: 6-byte pointer + 2 metadata
   static_assert(sizeof(handle6) == 6 && sizeof(handle8) == 8, "handle = address field + 2 metadata bytes");

   // The router tiers, in widen order. The tag selects how the 16 metadata bits decode.
   enum class Tier : uint8_t
   {
      setlist,      // 1 line, ≤~14 branches (SIMD scan); metadata unused
      sdiv2,        // 2 lines; metadata = [d0:8][--:8]                 idx = byte≥d0
      sdiv3,        // 3 lines; metadata = [d0:8][d1:8], d0≤d1          idx = (byte≥d0)+(byte≥d1)
      sdiv4,        // 4 lines; metadata = [d0:8][d1:8] + implicit 128  balanced depth-2 tree
      sparse_full,  // ≤16 subrange lines; metadata = 16-bit presence vector (subrange=byte>>4)
      full,         // 16 subrange lines, all present; metadata ignored (≡ presence 0xFFFF)
   };

   // Which cacheline (0-based, compacted) of the child holds `byte`. Branchless; the only
   // popcount is the sparse_full arm. For sparse_full the result is the rank of `byte`'s
   // subrange among the present ones — valid only when that subrange is present (see below).
   [[gnu::always_inline]] inline unsigned child_line(Tier t, uint16_t meta, uint8_t byte) noexcept
   {
      switch (t)
      {
         case Tier::setlist: return 0;
         case Tier::sdiv2:   return unsigned(byte >= uint8_t(meta));
         case Tier::sdiv3:   return unsigned(byte >= uint8_t(meta)) + unsigned(byte >= uint8_t(meta >> 8));
         case Tier::sdiv4:
         {
            const unsigned hi = byte >> 7;                                   // implicit divider at 128
            const uint8_t  d  = hi ? uint8_t(meta >> 8) : uint8_t(meta);     // select half's divider (csel)
            return (hi << 1) | unsigned(byte >= d);
         }
         case Tier::sparse_full:
         {
            const unsigned band = byte >> 4;                                 // which of 16 subranges
            return unsigned(std::popcount(uint16_t(meta & ((1u << band) - 1u))));
         }
         case Tier::full: return unsigned(byte >> 4);
      }
      return 0;  // unreachable; keeps non-exhaustive-aware compilers quiet
   }

   // How many cachelines the child occupies (iteration bounds / allocation size).
   [[gnu::always_inline]] inline unsigned line_count(Tier t, uint16_t meta) noexcept
   {
      switch (t)
      {
         case Tier::setlist:     return 1;
         case Tier::sdiv2:       return 2;
         case Tier::sdiv3:       return 3;
         case Tier::sdiv4:       return 4;
         case Tier::sparse_full: return unsigned(std::popcount(meta));
         case Tier::full:        return 16;
      }
      return 1;
   }

   // sparse_full only: is the subrange holding `byte` present at all? (A present subrange may
   // still miss on the exact byte — that's resolved by the direct slot within the loaded line.)
   [[gnu::always_inline]] inline bool subrange_present(uint16_t presence, uint8_t byte) noexcept
   {
      return (presence >> (byte >> 4)) & 1u;
   }
}  // namespace artpp::v2

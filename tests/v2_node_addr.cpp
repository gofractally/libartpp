// Exhaustive correctness for the v2 routing kernel: for every tier and many random
// metadata values, sweep all 256 bytes and check child_line against a brute-force
// reference — in range, monotonic (a valid left-to-right partition), and the exact
// per-tier formula. sparse_full additionally: present subranges enumerate 0..count-1.
#include "artpp/v2/node_addr.hpp"

#include <bit>
#include <cstdint>
#include <cstdio>
#include <random>

using namespace artpp::v2;
static int g_fail = 0;
#define CK(c)                                                                          \
   do {                                                                                \
      if (!(c)) { std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); ++g_fail; } \
   } while (0)

int main()
{
   // setlist: always line 0
   for (int b = 0; b < 256; ++b) CK(child_line(Tier::setlist, 0, uint8_t(b)) == 0);
   CK(line_count(Tier::setlist, 0) == 1);

   // full: direct subrange index, all 16 present
   for (int b = 0; b < 256; ++b) CK(child_line(Tier::full, 0xFFFF, uint8_t(b)) == unsigned(b >> 4));
   CK(line_count(Tier::full, 0xFFFF) == 16);

   std::mt19937 rng(20260614);
   for (int trial = 0; trial < 20000; ++trial)
   {
      uint8_t d0 = uint8_t(rng()), d1 = uint8_t(rng());
      if (d1 < d0) std::swap(d0, d1);  // sdiv3 dividers sorted ascending

      // sdiv2 — one divider in the low metadata byte
      {
         const uint16_t m = d0;
         unsigned       prev = 0;
         for (int b = 0; b < 256; ++b)
         {
            unsigned i = child_line(Tier::sdiv2, m, uint8_t(b));
            CK(i < 2);
            CK(i >= prev);                       // monotonic partition
            CK(i == unsigned(b >= d0));
            prev = i;
         }
         CK(line_count(Tier::sdiv2, m) == 2);
      }
      // sdiv3 — two sorted dividers, linear
      {
         const uint16_t m = uint16_t(d0 | (d1 << 8));
         unsigned       prev = 0;
         for (int b = 0; b < 256; ++b)
         {
            unsigned i = child_line(Tier::sdiv3, m, uint8_t(b));
            CK(i < 3);
            CK(i >= prev);
            CK(i == unsigned(b >= d0) + unsigned(b >= d1));
            prev = i;
         }
         CK(line_count(Tier::sdiv3, m) == 3);
      }
      // sdiv4 — implicit 128 + a divider in each half (balanced depth-2 tree)
      {
         const uint8_t  e0 = uint8_t(d0 % 128);        // lower-half divider
         const uint8_t  e1 = uint8_t(128 + d1 % 128);  // upper-half divider
         const uint16_t m  = uint16_t(e0 | (e1 << 8));
         for (int b = 0; b < 256; ++b)
         {
            unsigned i   = child_line(Tier::sdiv4, m, uint8_t(b));
            unsigned exp = (b >= 128) ? (2u + unsigned(b >= e1)) : unsigned(b >= e0);
            CK(i < 4);
            CK(i == exp);
         }
         CK(line_count(Tier::sdiv4, m) == 4);
      }
      // sparse_full — presence vector; present subranges rank 0..count-1 ascending
      {
         const uint16_t bits = uint16_t(rng());
         unsigned       expect = 0;
         for (int sub = 0; sub < 16; ++sub)
         {
            const uint8_t byte = uint8_t(sub * 16 + (rng() & 0xF));
            const bool    pres = (bits >> sub) & 1u;
            CK(subrange_present(bits, byte) == pres);
            unsigned i = child_line(Tier::sparse_full, bits, byte);
            if (pres) { CK(i == expect); ++expect; }
            CK(i <= unsigned(std::popcount(bits)));  // never indexes past the allocated lines
         }
         CK(line_count(Tier::sparse_full, bits) == unsigned(std::popcount(bits)));
         CK(expect == unsigned(std::popcount(bits)));
      }
   }

   // Handle round-trip: tag (low nibble), 16-bit metadata, and the address field all survive,
   // at both widths. The metadata bytes are width-invariant; only the address field shrinks.
   for (int t = 0; t < 20000; ++t)
   {
      const uint16_t meta = uint16_t(rng());
      const unsigned tag  = rng() & 0xF;
      {  // N=8: 6-byte (48-bit) 128-aligned address + tag in low bits
         const uint64_t addr = (uint64_t(rng()) << 7) & ((uint64_t(1) << 48) - 1);
         auto           h    = handle8::make(addr | tag, meta);
         CK(h.tag() == tag);
         CK(h.meta() == meta);
         CK(h.addr_field() == (addr | tag));
      }
      {  // N=6: 4-byte address field (28-bit index + 4-bit tag), 2 metadata
         const uint64_t idxfield = (uint64_t(rng() & 0x0FFFFFFFu) << 4) | tag;
         auto           h        = handle6::make(idxfield, meta);
         CK(h.tag() == tag);
         CK(h.meta() == meta);
         CK(h.addr_field() == idxfield);
         CK((h.addr_field() >> 32) == 0);  // address field is exactly 4 bytes
      }
   }

   std::printf(g_fail ? "v2_node_addr: %d FAILURE(S)\n" : "v2_node_addr: ALL OK\n", g_fail);
   return g_fail ? 1 : 0;
}

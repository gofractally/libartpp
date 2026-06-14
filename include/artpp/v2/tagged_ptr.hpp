// artpp — the design4.md "headerless ART": the pointer is the micro-header.
//
// A branch is a single tagged handle (default: a 48-bit pointer in 6 bytes). Every node
// and value blob is allocated 128-byte aligned, so the low 7 bits of an address are
// always zero; the type tag (`K`) lives in the low 4 of them. Four bits is the whole tag
// budget by contract — that is what lets a 32-bit INDEXED handle spend 28 bits on a
// 128-byte-line index (2^28 lines = 32 GB). Descent routes off the tag alone — it never
// reads a node header to learn the node's shape.
#pragma once
#include <bit>
#include <cassert>
#include <cstdint>
#include <cstring>

namespace artpp::v2::detail
{
   // The concrete node/value kind, stored in the low 4 bits of every branch handle.
   // (design4 splits the tag into shape[2:0] + fmt[6:3]; with this v1 taxonomy a
   // flat enum is enough — `shape_of` derives the descent class where needed.)
   enum class K : uint8_t
   {
      value_ptr    = 0,  // -> leaf<T> (suffix + value), the terminal
      value_inline = 1,  // value packed in the handle's bytes (trivially-copyable T <= width-1)
      prefix_node  = 2,  // long shared prefix + one next pointer — the OVERFLOW form (R1)
      setlist_u8   = 3,  // sparse byte router, 1 cacheline, inline prefix <= 8 (R1)
      node_full    = 4,  // dense direct-index router (full_lines_v lines, skip CL0)
      pfxd         = 5,  // HINT, not a type: "this node's header cacheline carries a
                         // prefix you must consume" — the real kind is recorded IN the
                         // header (kind byte after router_hdr, then the prefix bytes).
                         // The tag deliberately need not match the node's type: it says
                         // whether the header holds anything the descent needs. With an
                         // empty prefix nothing does, the node keeps its plain type tag,
                         // and (e.g.) node_full direct-indexes without touching CL0 —
                         // still headerless, the tag is the hint.
      c2           = 6,  // 2x128 segmented bitset router (skip CL0, rank)
      c4           = 7,  // 4x64  segmented bitset router (skip CL0, rank)
      c8           = 8,  // 8x32  segmented bitset router (skip CL0, rank)
      setlist_u16  = 9,  // wide sparse router: 2-byte stems, 1 cacheline (R1)
      bucket       = 10, // sorted (suffix,value) collector — the "last mile" terminal
      null         = 0xF,
   };
   // 4-bit tag contract: null is the all-ones nibble (an all-0xFF handle reads as null),
   // real kinds stay <= 0xE. An indexed handle has no spare bits — don't grow past this.
   static_assert(unsigned(K::bucket) <= 0xE && unsigned(K::null) == 0xF, "tags fit 4 bits");
   // Highest real tag. Bump this FIRST when adding a kind: the read descent hoists a
   // `tag > K_max → miss` guard ahead of its exhaustive switch (keeps the jump table
   // dense and rejects corrupt tags), so a kind above K_max would silently miss.
   // NOTE: tag values are free to renumber only until a persisted (WAL/disk) format
   // exists; after that they are frozen.
   inline constexpr unsigned K_max = unsigned(K::bucket);

   constexpr bool is_terminal(K k) noexcept { return k == K::value_ptr || k == K::value_inline; }
   static_assert(uint8_t(K::setlist_u16) == uint8_t(K::c8) + 1 &&
                     uint8_t(K::pfxd) == uint8_t(K::node_full) + 1,
                 "router tags are contiguous: setlist_u8..setlist_u16 is one range check");
   constexpr bool is_router(K k) noexcept { return k >= K::setlist_u8 && k <= K::setlist_u16; }

   // NOTE: we tried XOR-masking stored pointers to hide them from Apple's DMP
   // (the "rarely-dereferenced pointer" trick from dan/hive.blog). It REGRESSED artpp
   // (dict 166→184, hash24 slightly worse at every scale). Reason: unlike the COW
   // release path where the materialized child pointer is rarely used, artpp's descent
   // ALWAYS dereferences the pointer it loads next hop — and the DMP's prefetch warms
   // the child's TLB entry / CL0 (same page as the segment line it reads). So here the
   // prefetch is net helpful, and hiding it costs. Left unmasked deliberately.
   // An N-byte tagged handle (default N=6, 48-bit). Little-endian load/store via memcpy keeps it
   // trivially copyable and lets it sit unaligned inside packed node arrays. N is the per-branch
   // storage width every node layout derives from; a future allocator picks it (6 for density, 8
   // for full pointers / no 48-bit assumption, 4 for a pool index). Deref here is DIRECT addressing
   // (the handle holds the address); an indexed allocator resolves base+index via the tree's deref_.
   template <unsigned N>
   struct packed_ptr_t
   {
      static_assert(N >= 4 && N <= 8, "tagged handle is 4..8 bytes (4-bit tag + address)");
      // Handles are byte arrays whose VALUE math (raw() & tag masks, the offtab adds) is done
      // on host-order loads, but the byte-level accessors (tag() reading b[0], value_inline
      // packing at b+1) assume the low-order byte comes first. Asserted rather than abstracted:
      // a big-endian port would be untestable code on every machine this project targets — if
      // one ever matters, route the byte accessors through raw() instead.
      static_assert(std::endian::native == std::endian::little,
                    "artpp handles assume a little-endian host");
      static constexpr bool indexed = false;  // direct addressing: the handle IS the address
      uint8_t b[N];

      static packed_ptr_t null() noexcept
      {
         packed_ptr_t p;
         std::memset(p.b, 0xFF, N);  // tag nibble becomes 0xF == K::null
         return p;
      }
      static packed_ptr_t from(const void* addr, K k) noexcept
      {
         const uint64_t a = reinterpret_cast<uint64_t>(addr);
         // Two invariants every tagged NODE allocation guarantees — asserted in debug:
         //   * the allocation is 128-byte aligned (low 7 bits free for the tag), and
         //   * the address fits in the N-byte field (no >8N-bit VA: e.g. for N=6, >48-bit VA from
         //     5-level paging / ARM LVA is opt-in via mmap hints, never returned by default malloc).
         assert((a & 0x7f) == 0 && "artpp: tagged allocation must be 128-byte aligned");
         if constexpr (N < 8)
            assert((a >> (8u * N)) == 0 && "artpp: address exceeds the packed_ptr width");
         packed_ptr_t p;
         uint64_t     raw = (a & ~uint64_t(0x7f)) | uint8_t(k);
         std::memcpy(p.b, &raw, N);
         return p;
      }
      // TERMINAL handle: the tag needs only the low nibble, so a terminal allocation
      // (a leaf — pure payload, no branch slots) is 16-byte aligned, 8x denser than a
      // line. The tag itself says which decode applies, and every deref site is already
      // tag-dispatched, so the choice is static — no hot-path select.
      static packed_ptr_t from_term(const void* addr, K k) noexcept
      {
         const uint64_t a = reinterpret_cast<uint64_t>(addr);
         assert((a & 0xF) == 0 && "artpp: terminal allocation must be 16-byte aligned");
         if constexpr (N < 8)
            assert((a >> (8u * N)) == 0 && "artpp: address exceeds the packed_ptr width");
         packed_ptr_t p;
         uint64_t     raw = a | uint8_t(k);
         std::memcpy(p.b, &raw, N);
         return p;
      }

      uint64_t raw() const noexcept  // the N-byte field, zero-extended
      {
         uint64_t v = 0;
         std::memcpy(&v, b, N);
         return v;
      }
      K     tag() const noexcept { return K(b[0] & 0xF); }
      void* ptr() const noexcept { return reinterpret_cast<void*>(raw() & ~uint64_t(0x7f)); }
      void* tptr() const noexcept { return reinterpret_cast<void*>(raw() & ~uint64_t(0xF)); }
      bool  is_null() const noexcept { return tag() == K::null; }
   };

   // Default branch handle: 6 bytes (48-bit). An allocator overrides the width per tree by
   // exposing `artpp_handle` (see handle_of in tree.hpp); every node layout derives from sizeof(Ptr).
   using packed_ptr = packed_ptr_t<6>;
   static_assert(sizeof(packed_ptr) == 6, "packed_ptr is exactly 6 bytes");

   // The allocation/layout unit: nodes are built from (and aligned to) 128-byte lines, which
   // is what keeps the low 7 bits of every node address free for the tag.
   inline constexpr unsigned cacheline_bytes = 128;

   // ISO-clean tail padding: a zero-length array is a GNU extension, so layouts that tile a
   // cacheline exactly use pad_t<0> (empty) instead. Declare it [[no_unique_address]] and pin
   // the enclosing struct's sizeof with a static_assert.
   template <unsigned N>
   struct pad_t
   {
      uint8_t b[N];
   };
   template <>
   struct pad_t<0>
   {
   };

   // 9-bit nbranch (0..256) + 6-bit prefix_len + has_term, packed into one u16 —
   // the base header every router shares (design4 `struct node`).
   struct node_hdr
   {
      uint16_t has_term : 1;
      uint16_t nbranch : 9;   // 0..256 — the full node's 256 fits
      uint16_t prefix_len : 6;  // routers carry no inline prefix in v1 (kept for layout parity)
   };
   static_assert(sizeof(node_hdr) == 2, "node_hdr is one u16");

   // Common header every router shares at offset 0. Templated on the branch handle type so a
   // different width re-tiles the node (term is one branch). Default alias keeps the 6-byte type.
   template <class Ptr>
   struct router_hdr_t
   {
      node_hdr hdr;
      Ptr      term;  // value when a key ends here (value_ptr -> leaf, suffix "")
   };
   using router_hdr = router_hdr_t<packed_ptr>;
}  // namespace artpp::v2::detail

namespace artpp::v2
{
   // Public spelling of the direct branch handle — the allocator opt-in protocol is public
   // API, so naming a width must not reach into detail::
   //   template <class T> struct my_alloc { using artpp_handle = artpp::v2::packed_ptr_t<8>; ... };
   using detail::packed_ptr_t;
}  // namespace artpp::v2

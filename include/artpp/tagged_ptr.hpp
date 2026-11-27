// SPDX-License-Identifier: MIT
// artpp — the "headerless ART": the pointer is the micro-header.
//
// A branch is a single tagged handle (default: a 48-bit pointer in 6 bytes). Every node
// is allocated 128-byte aligned and every leaf at least 16-byte aligned, so the low 4 bits
// of an address are always zero; the type tag (`K`) lives in them. Four bits is the whole tag
// budget by contract — that is what lets a 32-bit INDEXED handle spend 28 bits on a
// 128-byte-line index (2^28 lines = 32 GB). Descent routes off the tag alone — it never
// reads a node header to learn the node's shape.
#pragma once
#include <array>
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <type_traits>

// Handles hold 48-bit (or wider) addresses and the pool reserves tens of gigabytes of
// address space: a 64-bit target is assumed throughout, so say so once, clearly.
static_assert(sizeof(std::size_t) == 8 && sizeof(void*) == 8,
              "artpp requires a 64-bit target (64-bit size_t and pointers)");

namespace artpp
{
   // Thrown from the ALLOCATION path when the allocator returns a block the tree cannot
   // address: one that is not aligned as the node layout requires (the low address bits
   // carry the handle's tag), or one whose address does not fit the handle's width (the
   // default 6-byte handle holds 48 bits — top-byte tagging such as HWASan/MTE, 5-level
   // paging or 52-bit ARM addresses exceed it; select `packed_ptr_t<8>` in the allocator).
   // The block is returned to the allocator first, so nothing leaks. Derived from
   // std::bad_alloc: to a caller it is an allocation that could not be used.
   class bad_allocation_address : public std::bad_alloc
   {
      const char* msg_;

     public:
      explicit bad_allocation_address(const char* m) noexcept : msg_(m) {}
      const char* what() const noexcept override { return msg_; }
   };
}  // namespace artpp

namespace artpp
{
   // What a container instantiation tells a persistent image about itself (see
   // map::image_stamp and line_pool): eight words that are equal for two instantiations
   // exactly when — as far as the container can know — they read and write the same bytes.
   // The image keeps the stamp of the container that wrote it and refuses another.
   using image_stamp = std::array<std::uint64_t, 8>;

   // How hard checkpoint() pushes a persistent image to its file.
   //   none  — (default) the image is recorded through the shared file mapping and the
   //           operating system writes it back when it chooses. Costs nothing beyond
   //           writing the headers. It survives the process exiting or being killed, and a
   //           later open in any process sees it; it does NOT survive the machine crashing
   //           or losing power before the write-back, and a write error (a full disk, an
   //           I/O error) is not reported.
   //   flush — the data, then the headers, are written out with msync(MS_SYNC), each step
   //           checked: a write error is thrown from checkpoint(). Cost grows with what
   //           changed since the previous flush. Still not crash consistency: the image
   //           is updated in place.
   enum class sync_level : std::uint8_t { none, flush };
}  // namespace artpp

namespace artpp::detail
{
   // memcpy whose source may be an empty view: std::string_view{}.data() is null, and
   // passing null to memcpy is undefined even for a zero length.
   inline void copy_bytes(void* dst, const void* src, size_t n) noexcept
   {
      if (n) std::memcpy(dst, src, n);
   }
   // The "no value" mapped type: a map<Key, none_t> is a SET. One byte, empty, trivially
   // copyable — so map<Key, none_t> compiles through every node layout unchanged (Stage A),
   // and the value/T touchpoints are then elided under `if constexpr (HasValue)` (Stages B+).
   // detail-only; the public spelling is artpp::set (a facade over map<Key, none_t>).
   struct none_t
   {
   };
   static_assert(sizeof(none_t) == 1 && std::is_trivially_copyable_v<none_t>,
                 "none_t is a 1-byte trivially-copyable tag");

   // The concrete node/value kind, stored in the low 4 bits of every branch handle.
   // (the original design split the tag into shape[2:0] + fmt[6:3]; with this v1 taxonomy a
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

      // Can a block at `addr` be named by this handle? Its low bits (`align_mask`: 0x7f for
      // a node, the terminal unit - 1 for a leaf) must be clear for the tag, and the address
      // must fit N bytes. Checked once per ALLOCATION by the tree (which throws
      // bad_allocation_address) — never on descent; from()/from_term() only assert.
      static bool addressable(const void* addr, uint64_t align_mask) noexcept
      {
         // one AND against a constant: the alignment bits plus every bit above the N-byte field
         constexpr uint64_t high = N < 8 ? ~uint64_t(0) << (8u * (N < 8 ? N : 0)) : 0;
         return (reinterpret_cast<uint64_t>(addr) & (align_mask | high)) == 0;
      }

      static packed_ptr_t null() noexcept
      {
         packed_ptr_t p;
         std::memset(p.b, 0xFF, N);  // tag nibble becomes 0xF == K::null
         return p;
      }
      static packed_ptr_t from(const void* addr, K k) noexcept
      {
         const uint64_t a = reinterpret_cast<uint64_t>(addr);
         // Two invariants every tagged NODE allocation guarantees:
         //   * the allocation is 128-byte aligned (low 7 bits free for the tag), and
         //   * the address fits in the N-byte field (no >8N-bit VA: e.g. for N=6, >48-bit VA from
         //     5-level paging / ARM LVA / top-byte tagging).
         // ENFORCED where the block is allocated (map::node_alloc_ / term_alloc_ check
         // addressable() and throw); the asserts here only document it for debug builds.
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
   // exposing `artpp_handle` (see handle_of in map.hpp); every node layout derives from sizeof(Ptr).
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
   // the base header every router shares (`struct node`).
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

   // Descent hop counter (diagnostic): each iteration of descend_read's loop ≈ one cacheline
   // touched per descent level — the per-lookup cacheline count. Incremented ONLY when
   // ARTPP_HOPS is defined (a no-op otherwise). inline ⇒ one definition across TUs.
   inline unsigned long long descent_hops_ = 0;
}  // namespace artpp::detail

namespace artpp
{
   // Public spelling of the direct branch handle — the allocator opt-in protocol is public
   // API, so naming a width must not reach into detail::
   //   template <class T> struct my_alloc { using artpp_handle = artpp::packed_ptr_t<8>; ... };
   using detail::packed_ptr_t;
}  // namespace artpp

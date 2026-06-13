// artpp::line_pool — the Stage C allocation policy: one contiguous reservation of 128-byte
// lines, addressed by a 4-byte handle (28-bit line index + 4-bit tag → 2^28 lines = 32 GB).
// deref is base + (index << 7) — address-space independent, so the SAME store works over
// anonymous memory or a memory-mapped file. The full virtual range is reserved up front
// (base never moves); memory is committed in 4 MB steps as the frontier grows; freed nodes
// recycle through exact-size free lists (sizes are line counts).
//
// NOTE: file backing is exactly that — backing storage (page eviction, >RAM data sets).
// Durability/recovery (persisted root, WAL) is the next stage, not this one: the file is
// truncated on open and its contents are not a self-describing image yet.
//
// MEASURED vs std::allocator (M5, 5M keys, artpp_pool_ab INTERLEAVED per rep): inserts
// recover the malloc cost and more (uniform u64 0.81x, clustered 0.94x), queries are
// neutral-to-better (u64 0.98x; u32 0.91x — bump allocation lays children near parents,
// node_full drops to 10 lines, and the deref add is fused), despite losing the DMP
// pointer-prefetch (see tagged_ptr.hpp) and values inlining only up to 3 bytes. An
// earlier phase-separated version of the bench misread thermal drift as a 10-20% query
// tax — contestants must run back-to-back inside each rep on this fanless machine.
#pragma once
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "artpp/tagged_ptr.hpp"

namespace artpp
{
   namespace detail
   {
      // 4-byte indexed branch handle: [28-bit line index][4-bit tag], little-endian — the
      // same byte-array shape as packed_ptr_t (memcpy load/store, unaligned-safe inside
      // packed node arrays), but the payload is a line index resolved against a pool base.
      struct line_ptr
      {
         static constexpr bool indexed = true;
         uint8_t               b[4];

         static line_ptr null() noexcept
         {
            line_ptr p;
            std::memset(p.b, 0xFF, 4);  // tag nibble becomes 0xF == K::null
            return p;
         }
         // Encode a byte offset from the pool's NODE base (line-aligned, < 32 GB).
         static line_ptr from_off(size_t byte_off, K k) noexcept
         {
            assert((byte_off & 0x7f) == 0 && "artpp: pool offsets are line-aligned");
            assert(byte_off >> 35 == 0 && "artpp: offset exceeds the 28-bit line index");
            line_ptr       p;
            const uint32_t raw = uint32_t(byte_off >> 3) | uint8_t(k);  // (off>>7)<<4 == off>>3
            std::memcpy(p.b, &raw, 4);
            return p;
         }
         // Encode a byte offset from the pool's TERMINAL base (16-byte units, < 4 GB).
         // The tag disambiguates the two index spaces, so terminals get their own full
         // 28-bit range; 16-byte alignment leaves the low nibble free for the tag, so
         // the stored u32 IS the byte offset (no shift on deref).
         static line_ptr from_toff(size_t byte_off, K k) noexcept
         {
            assert((byte_off & 0xF) == 0 && "artpp: terminal offsets are 16-byte aligned");
            assert(byte_off >> 32 == 0 && "artpp: offset exceeds the terminal region");
            line_ptr       p;
            const uint32_t raw = uint32_t(byte_off) | uint8_t(k);
            std::memcpy(p.b, &raw, 4);
            return p;
         }

         uint64_t raw() const noexcept
         {
            uint32_t v;
            std::memcpy(&v, b, 4);
            return v;
         }
         K    tag() const noexcept { return K(b[0] & 0xF); }
         bool is_null() const noexcept { return tag() == K::null; }
      };
      static_assert(sizeof(line_ptr) == 4, "line_ptr is exactly 4 bytes");
   }  // namespace detail

   // The line store. Single-threaded, like the tree. Not copyable/movable — every handle
   // in every tree on this pool encodes an offset from base_.
   class line_pool
   {
     public:
      static constexpr size_t line_bytes  = 128;
      static constexpr size_t chunk_bytes = size_t(4) << 20;  // commit/grow unit
      static constexpr size_t max_bytes   = size_t(1) << 35;  // 32 GB — the 28-bit index cap

      explicit line_pool(size_t cap = max_bytes) { init_(nullptr, cap); }            // anonymous
      explicit line_pool(const char* path, size_t cap = max_bytes) { init_(path, cap); }  // file
      line_pool(const line_pool&)            = delete;
      line_pool& operator=(const line_pool&) = delete;
      ~line_pool()
      {
         if (base_) ::munmap(base_, cap_);
         if (fd_ >= 0) ::close(fd_);
      }

      // Terminal region: 16-byte-granular storage for leaves (tag K::value_ptr). A
      // terminal is pure payload — suffix bytes + value — with no branch slots, so it
      // needs only the 4 tag bits of alignment, not a full line. Splitting the store
      // into a node region (128 B lines) and a terminal region (16 B units) cuts the
      // terminal working set up to 8x; the handle's TAG selects the decode, so every
      // deref site picks its region statically (descent is already tag-dispatched).
      static constexpr size_t term_unit      = 16;
      static constexpr size_t max_term_bytes = size_t(1) << 32;  // 4 GB — u32 offsets

      std::byte* base() const noexcept { return base_; }
      std::byte* term_base() const noexcept { return tbase_; }
      size_t     committed() const noexcept { return committed_; }
      size_t     term_committed() const noexcept { return tcommitted_; }
      size_t     used_lines() const noexcept { return frontier_; }
      size_t     used_term_units() const noexcept { return tfrontier_; }
      // Logical size of a file-backed pool: the terminal region lives at the fixed
      // offset ncap_, so any terminal commit makes the file SPARSE — logical size spans
      // the gap, but only written blocks occupy disk (APFS/ext4 et al).
      size_t file_extent() const noexcept
      {
         return tcommitted_ ? ncap_ + tcommitted_ : committed_;
      }

      // Wholesale image adoption: copy `s`'s used range AND its carving state (frontier,
      // free lists — the intrusive links live inside freed lines, so they ride the byte
      // copy) so that every line index minted by `s` resolves identically against THIS
      // pool's base. This is what makes a cross-pool container move a single memcpy
      // instead of a per-element (or even per-node) rebuild: handles are base-relative,
      // so the bytes ARE the structure. Only sound when this pool backs exactly the
      // adopting container — any other resident allocation is clobbered. Trivially-
      // copyable payloads only (a bitwise value copy must be a valid value copy).
      void adopt(const line_pool& s)
      {
         const size_t used  = s.frontier_ * line_bytes;
         const size_t tused = s.tfrontier_ * term_unit;
         if (used > committed_) commit_(used);     // throws bad_alloc past our cap
         if (tused > tcommitted_) tcommit_(tused);
         std::memcpy(base_, s.base_, used);
         std::memcpy(term_base(), s.term_base(), tused);
         frontier_  = s.frontier_;
         tfrontier_ = s.tfrontier_;
         huge_      = s.huge_;
         thuge_     = s.thuge_;
         std::memcpy(free_, s.free_, sizeof(free_));
         std::memcpy(tfree_, s.tfree_, sizeof(tfree_));
      }

      // Allocate n contiguous lines: exact-size free list first, else bump the frontier
      // (committing more chunks as needed). Throws std::bad_alloc on cap/OS failure — the
      // tree's strong-guarantee fault point, same as any allocator.
      void* alloc_lines(size_t n)
      {
         if (n <= FREE_MAX)
         {
            if (const uint32_t head = free_[n]; head != NIL)
            {
               std::byte* p = base_ + size_t(head) * line_bytes;
               std::memcpy(&free_[n], p, 4);  // pop: the next index lives in the freed block
               return p;
            }
         }
         else if (void* p = pop_huge_(n))
            return p;
         const size_t need = (frontier_ + n) * line_bytes;
         if (need > committed_) commit_(need);
         std::byte* p = base_ + frontier_ * line_bytes;
         frontier_ += n;
         return p;
      }
      void free_lines(void* p, size_t n) noexcept
      {
         const uint32_t idx = uint32_t((static_cast<std::byte*>(p) - base_) / line_bytes);
         if (n <= FREE_MAX)
         {
            std::memcpy(p, &free_[n], 4);
            free_[n] = idx;
         }
         else  // rare giant node: one list, {next, nlines} kept in the block
         {
            const uint32_t hdr[2] = {huge_, uint32_t(n)};
            std::memcpy(p, hdr, 8);
            huge_ = idx;
         }
      }

      // Terminal-region twins of alloc_lines/free_lines, in 16-byte units.
      void* alloc_term(size_t n)
      {
         if (n <= TFREE_MAX)
         {
            if (const uint32_t head = tfree_[n]; head != NIL)
            {
               std::byte* p = tbase_ + size_t(head) * term_unit;
               std::memcpy(&tfree_[n], p, 4);
               return p;
            }
         }
         else if (void* p = pop_thuge_(n))
            return p;
         const size_t need = (tfrontier_ + n) * term_unit;
         if (need > tcommitted_) tcommit_(need);
         std::byte* p = tbase_ + tfrontier_ * term_unit;
         tfrontier_ += n;
         return p;
      }
      void free_term(void* p, size_t n) noexcept
      {
         const uint32_t idx = uint32_t((static_cast<std::byte*>(p) - tbase_) / term_unit);
         if (n <= TFREE_MAX)
         {
            std::memcpy(p, &tfree_[n], 4);
            tfree_[n] = idx;
         }
         else  // giant terminal (very long key suffix): {next, nunits} in the block
         {
            const uint32_t hdr[2] = {thuge_, uint32_t(n)};
            std::memcpy(p, hdr, 8);
            thuge_ = idx;
         }
      }

     private:
      static constexpr uint32_t NIL       = 0xFFFFFFFFu;
      static constexpr size_t   FREE_MAX  = 1024;  // node lists cover up to 128 KB
      static constexpr size_t   TFREE_MAX = 4096;  // terminal lists cover up to 64 KB

      void init_(const char* path, size_t cap)
      {
         ncap_ = std::min((cap + chunk_bytes - 1) / chunk_bytes * chunk_bytes, max_bytes);
         // terminal region rides the same reservation/file, after the node region;
         // sized at 1/8th of it (the 16/128 unit ratio), capped by its u32 offsets
         tcap_ = std::min(std::max(ncap_ / 8, chunk_bytes), max_term_bytes);
         cap_  = ncap_ + tcap_;
         if (path)
         {
            fd_ = ::open(path, O_RDWR | O_CREAT | O_TRUNC, 0644);
            if (fd_ < 0) throw std::bad_alloc();
         }
         void* r = ::mmap(nullptr, cap_, PROT_NONE, MAP_PRIVATE | MAP_ANON, -1, 0);
         if (r == MAP_FAILED)
         {
            if (fd_ >= 0) ::close(fd_);
            throw std::bad_alloc();
         }
         base_  = static_cast<std::byte*>(r);
         tbase_ = base_ + ncap_;
         std::memset(free_, 0xFF, sizeof(free_));   // every head = NIL (all-ones)
         std::memset(tfree_, 0xFF, sizeof(tfree_));
      }
      // Commit [start, start+committed..target) of the reservation; file regions live at
      // the same offsets in the file (ftruncate grows it monotonically).
      void commit_region_(size_t start, size_t& committed, size_t regcap, size_t need)
      {
         const size_t target = (need + chunk_bytes - 1) / chunk_bytes * chunk_bytes;
         if (target > regcap) throw std::bad_alloc();
         if (fd_ >= 0)  // file: extend it, then map the new chunks over the reservation
         {
            const off_t fend = off_t(start + target);
            struct stat st;
            if (::fstat(fd_, &st) != 0) throw std::bad_alloc();
            if (st.st_size < fend && ::ftruncate(fd_, fend) != 0) throw std::bad_alloc();
            void* r = ::mmap(base_ + start + committed, target - committed,
                             PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, fd_,
                             off_t(start + committed));
            if (r == MAP_FAILED) throw std::bad_alloc();
         }
         else if (::mprotect(base_ + start + committed, target - committed,
                             PROT_READ | PROT_WRITE) != 0)
            throw std::bad_alloc();
         committed = target;
      }
      void commit_(size_t need) { commit_region_(0, committed_, ncap_, need); }
      void tcommit_(size_t need) { commit_region_(ncap_, tcommitted_, tcap_, need); }

      void* pop_huge_(size_t n) noexcept  // exact-size scan of the rare >FREE_MAX list
      {
         uint32_t prev = NIL;
         for (uint32_t idx = huge_; idx != NIL;)
         {
            std::byte* p = base_ + size_t(idx) * line_bytes;
            uint32_t   hdr[2];  // {next, nlines}
            std::memcpy(hdr, p, 8);
            if (hdr[1] == n)
            {
               if (prev == NIL)
                  huge_ = hdr[0];
               else
                  std::memcpy(base_ + size_t(prev) * line_bytes, &hdr[0], 4);  // prev->next
               return p;
            }
            prev = idx;
            idx  = hdr[0];
         }
         return nullptr;
      }
      void* pop_thuge_(size_t n) noexcept  // terminal twin of pop_huge_
      {
         uint32_t prev = NIL;
         for (uint32_t idx = thuge_; idx != NIL;)
         {
            std::byte* p = tbase_ + size_t(idx) * term_unit;
            uint32_t   hdr[2];  // {next, nunits}
            std::memcpy(hdr, p, 8);
            if (hdr[1] == n)
            {
               if (prev == NIL)
                  thuge_ = hdr[0];
               else
                  std::memcpy(tbase_ + size_t(prev) * term_unit, &hdr[0], 4);
               return p;
            }
            prev = idx;
            idx  = hdr[0];
         }
         return nullptr;
      }

      std::byte* base_       = nullptr;
      std::byte* tbase_      = nullptr;  // = base_ + ncap_, cached (Dan: don't recompute per use)
      size_t     cap_        = 0;  // total reservation = ncap_ + tcap_
      size_t     ncap_       = 0;
      size_t     tcap_       = 0;
      size_t     committed_  = 0;
      size_t     tcommitted_ = 0;
      size_t     frontier_   = 0;  // node bump position, in lines
      size_t     tfrontier_  = 0;  // terminal bump position, in 16 B units
      int        fd_         = -1;
      uint32_t   huge_       = NIL;
      uint32_t   thuge_      = NIL;
      uint32_t   free_[FREE_MAX + 1];
      uint32_t   tfree_[TFREE_MAX + 1];
   };

   // The tree-facing allocator. Rebinds share the pool; the tree picks the 4-byte indexed
   // handle up via artpp_handle and caches the mapping base via artpp_base() (see handle_of /
   // map::rebase_). With this allocator a branch costs 4 bytes and the whole store is
   // one relocatable range — the WAL/disk foundation.
   template <class T>
   struct pool_alloc
   {
      using value_type = T;
      using artpp_handle = detail::line_ptr;

      line_pool* pool = nullptr;

      pool_alloc() = default;
      explicit pool_alloc(line_pool* p) noexcept : pool(p) {}
      explicit pool_alloc(line_pool& p) noexcept : pool(&p) {}
      template <class U>
      pool_alloc(const pool_alloc<U>& o) noexcept : pool(o.pool) {}

      const std::byte* artpp_base() const noexcept { return pool ? pool->base() : nullptr; }
      const std::byte* artpp_term_base() const noexcept
      {
         return pool ? pool->term_base() : nullptr;
      }

      // Terminal-region hooks: leaves (tag K::value_ptr) are 16-byte-granular payloads
      // in their own region — see line_pool. Detected by the tree; allocators without
      // them store terminals through the normal (line/block) path.
      void* artpp_alloc_term(size_t bytes)
      {
         return pool->alloc_term((bytes + line_pool::term_unit - 1) / line_pool::term_unit);
      }
      void artpp_free_term(void* p, size_t bytes) noexcept
      {
         pool->free_term(p, (bytes + line_pool::term_unit - 1) / line_pool::term_unit);
      }

      // Opt-in bulk transport for map's cross-pool move assignment: adopt the source
      // pool's whole image (see line_pool::adopt). By providing this member the allocator
      // author asserts the destination pool is exclusive to the destination tree.
      void artpp_adopt(const pool_alloc& o) { pool->adopt(*o.pool); }

      static constexpr size_t lines_(size_t bytes) noexcept
      {
         return (bytes + line_pool::line_bytes - 1) / line_pool::line_bytes;
      }
      T* allocate(size_t n)
      {
         static_assert(alignof(T) <= line_pool::line_bytes, "pool lines are 128-byte aligned");
         return static_cast<T*>(pool->alloc_lines(lines_(n * sizeof(T))));
      }
      void deallocate(T* p, size_t n) noexcept { pool->free_lines(p, lines_(n * sizeof(T))); }

      template <class U>
      bool operator==(const pool_alloc<U>& o) const noexcept
      {
         return pool == o.pool;
      }
   };
}  // namespace artpp

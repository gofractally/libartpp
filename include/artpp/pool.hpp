// artpp::line_pool — the Stage C allocation policy: one contiguous reservation of 128-byte
// lines, addressed by a 4-byte handle (28-bit line index + 4-bit tag → 2^28 lines = 32 GB).
// deref is base + (index << 7) — address-space independent, so the SAME store works over
// anonymous memory or a memory-mapped file. The full virtual range is reserved up front
// (base never moves); memory is committed in 4 MB steps as the frontier grows; freed nodes
// recycle through exact-size free lists (sizes are line counts).
//
// Backing is BOTH out-of-core storage (page eviction, >RAM data sets) AND a durable store
// across a clean close/reopen. A file-backed pool is a DIRECTORY with one dense file per address
// region — `nodes` (128-byte lines; its first page(s) hold a self-describing superblock: geometry
// + carving state + the tree's root/count) and `terms` (16-byte terminal units). One file per
// region means each grows densely from offset 0, so there are NO sparse holes and NO dependency on
// sparse-file support (the earlier single-file layout put the terminal region at a fixed 32 GB
// offset, forcing a sparse hole — hostile to FAT/exFAT and to copy/backup tools). Call
// checkpoint(root, count) before close to persist; a later line_pool(dir) restores it with no
// relocation (offset handles need none). Crash/reboot durability (full data sync / WAL) is the
// next stage — checkpoint flushes the header, and data pages ride the page cache across a reopen.
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
#include <string>

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

      explicit line_pool(size_t cap = max_bytes) { init_(nullptr, cap); }  // anonymous
      // File-backed: `dir` is a STORE DIRECTORY; the pool creates/reopens one dense file per
      // region inside it (`nodes` + `terms`). Reopen-or-create — a prior checkpoint() is restored.
      explicit line_pool(const char* dir, size_t cap = max_bytes) { init_(dir, cap); }
      line_pool(const line_pool&)            = delete;
      line_pool& operator=(const line_pool&) = delete;
      ~line_pool()
      {
         if (nbase_) ::munmap(nbase_, node_resv_);
         if (tbase_) ::munmap(tbase_, term_resv_);
         if (nfd_ >= 0) ::close(nfd_);
         if (tfd_ >= 0) ::close(tfd_);
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
      // Total on-disk bytes across the store's region files. Each region is its own dense
      // file (no sparse holes), so this is just the committed extents + the header.
      size_t file_extent() const noexcept { return sb_bytes_ + committed_ + tcommitted_; }

      // ── persistence (file-backed, clean-close durability) ────────────────────────
      // checkpoint() stamps the current carving state + the tree's root handle and count
      // into the superblock and flushes it, so a later line_pool(path) over the same file
      // restores them and the tree round-trips with no relocation (offset handles). The
      // data pages ride the page cache across a clean reopen; crash/reboot durability (a
      // full data sync / WAL) is a further stage — see the file header. restored()/root()/
      // count() report what a reopen recovered (false / 0 / 0 on a fresh file or anon pool).
      bool     restored() const noexcept { return restored_; }
      uint64_t root() const noexcept { return uroot_; }
      uint64_t count() const noexcept { return ucount_; }
      void     checkpoint(uint64_t root, uint64_t count) noexcept
      {
         uroot_ = root;
         ucount_ = count;
         if (sb_bytes_ == 0) return;  // anonymous pool: nothing to persist
         auto* sb       = reinterpret_cast<superblock*>(nbase_);  // header = nodes file, page(s) 0
         sb->magic      = MAGIC;
         sb->version    = VERSION;
         sb->flags      = 0;
         sb->ncap       = ncap_;
         sb->tcap       = tcap_;
         sb->frontier   = frontier_;
         sb->tfrontier  = tfrontier_;
         sb->committed  = committed_;
         sb->tcommitted = tcommitted_;
         sb->huge       = huge_;
         sb->thuge      = thuge_;
         sb->uroot      = root;
         sb->ucount     = count;
         std::memcpy(sb->free_heads, free_, sizeof(free_));
         std::memcpy(sb->tfree_heads, tfree_, sizeof(tfree_));
         ::msync(nbase_, sb_bytes_, MS_SYNC);  // header durable; data rides the page cache
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

      // ── persistence: a self-describing superblock in the nodes file's first page(s) ──
      // A file-backed pool reserves SB_BYTES at the front of the NODES file for this header and
      // shifts base_ past it, so every node/terminal offset stays base-relative — identical to an
      // anonymous pool, so adopt() and every minted handle are unaffected. checkpoint() serializes
      // the carving state (frontier + free-list heads; the chains themselves live inside freed
      // blocks, already on disk) plus the tree's root handle + count here; a later line_pool(dir)
      // over the same store restores them. Offset handles need no relocation — the bytes ARE the
      // structure across a close/reopen. The terminal region is a SEPARATE dense file (no header).
      static constexpr uint64_t MAGIC    = 0x6c6f6f7070747261ULL;  // "artppool" (little-endian)
      static constexpr uint32_t VERSION  = 1;
      static constexpr size_t   SB_BYTES = 32768;  // header reservation (>= sizeof(superblock))
      struct superblock
      {
         uint64_t magic;
         uint32_t version, flags;
         uint64_t ncap, tcap, frontier, tfrontier, committed, tcommitted;
         uint32_t huge, thuge;
         uint64_t uroot, ucount;             // user metadata: tree root handle + element count
         uint32_t free_heads[FREE_MAX + 1];  // exact-size free-list heads
         uint32_t tfree_heads[TFREE_MAX + 1];
      };
      static_assert(sizeof(superblock) <= SB_BYTES, "superblock must fit its reservation");

      void destroy_() noexcept  // unwind a partially-constructed pool
      {
         if (nbase_) ::munmap(nbase_, node_resv_);
         if (tbase_) ::munmap(tbase_, term_resv_);
         if (nfd_ >= 0) ::close(nfd_);
         if (tfd_ >= 0) ::close(tfd_);
         nbase_ = base_ = tbase_ = nullptr;
         nfd_ = tfd_ = -1;
      }
      void init_(const char* dir, size_t cap)
      {
         ncap_     = std::min((cap + chunk_bytes - 1) / chunk_bytes * chunk_bytes, max_bytes);
         tcap_     = std::min(std::max(ncap_ / 8, chunk_bytes), max_term_bytes);  // 1/8th (16/128)
         sb_bytes_ = dir ? SB_BYTES : 0;  // the nodes file carries a self-describing header
         bool restore = false;
         if (dir)
         {
            ::mkdir(dir, 0755);  // create the store directory (EEXIST is fine; a bad path fails below)
            const std::string np = std::string(dir) + "/nodes";  // big-object file (128 B lines + header)
            const std::string tp = std::string(dir) + "/terms";  // terminal-unit file (16 B)
            nfd_ = ::open(np.c_str(), O_RDWR | O_CREAT, 0644);
            if (nfd_ < 0) throw std::bad_alloc();
            tfd_ = ::open(tp.c_str(), O_RDWR | O_CREAT, 0644);
            if (tfd_ < 0) { ::close(nfd_); nfd_ = -1; throw std::bad_alloc(); }
            struct stat st;
            if (::fstat(nfd_, &st) != 0) { destroy_(); throw std::bad_alloc(); }
            if (st.st_size >= off_t(SB_BYTES))  // existing store: validate + adopt its geometry
            {
               struct { uint64_t magic; uint32_t version, flags; uint64_t ncap, tcap; } pk{};
               if (::pread(nfd_, &pk, sizeof pk, 0) == off_t(sizeof pk) && pk.magic == MAGIC &&
                   pk.version == VERSION)
               {
                  restore = true;
                  ncap_   = pk.ncap;  // offsets resolve against the store's geometry, not `cap`
                  tcap_   = pk.tcap;
               }
               else { destroy_(); throw std::bad_alloc(); }  // foreign / corrupt nodes file
            }
         }
         node_resv_ = sb_bytes_ + ncap_;  // one PROT_NONE reservation per region (own file each)
         term_resv_ = tcap_;
         void* nr   = ::mmap(nullptr, node_resv_, PROT_NONE, MAP_PRIVATE | MAP_ANON, -1, 0);
         if (nr == MAP_FAILED) { destroy_(); throw std::bad_alloc(); }
         nbase_ = static_cast<std::byte*>(nr);
         base_  = nbase_ + sb_bytes_;
         void* tr = ::mmap(nullptr, term_resv_, PROT_NONE, MAP_PRIVATE | MAP_ANON, -1, 0);
         if (tr == MAP_FAILED) { destroy_(); throw std::bad_alloc(); }
         tbase_ = static_cast<std::byte*>(tr);
         if (dir)  // map the superblock header (nodes file, page(s) 0) read-write
         {
            struct stat st;
            if (::fstat(nfd_, &st) != 0) { destroy_(); throw std::bad_alloc(); }
            if (st.st_size < off_t(sb_bytes_) && ::ftruncate(nfd_, off_t(sb_bytes_)) != 0)
            { destroy_(); throw std::bad_alloc(); }
            if (::mmap(nbase_, sb_bytes_, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, nfd_, 0) ==
                MAP_FAILED)
            { destroy_(); throw std::bad_alloc(); }
         }
         std::memset(free_, 0xFF, sizeof(free_));   // every head = NIL (all-ones)
         std::memset(tfree_, 0xFF, sizeof(tfree_));
         if (restore)
         {
            const superblock* sb = reinterpret_cast<const superblock*>(nbase_);
            frontier_  = sb->frontier;
            tfrontier_ = sb->tfrontier;
            huge_      = sb->huge;
            thuge_     = sb->thuge;
            uroot_     = sb->uroot;
            ucount_    = sb->ucount;
            std::memcpy(free_, sb->free_heads, sizeof(free_));
            std::memcpy(tfree_, sb->tfree_heads, sizeof(tfree_));
            if (const size_t want = sb->committed) commit_(want);    // re-map written region files
            if (const size_t twant = sb->tcommitted) tcommit_(twant);
            restored_ = true;
         }
         else if (dir)
            checkpoint(0, 0);  // stamp an initial empty image so a reopen is self-describing
      }
      // Grow a region's mapping to cover `need` bytes: extend its OWN file densely from `file_off`
      // and map the new chunks (MAP_FIXED over the PROT_NONE reservation), or mprotect for an
      // anonymous pool. One file per region ⇒ dense growth, no sparse holes.
      static void commit_region_(std::byte* mem_base, int fd, size_t file_off, size_t& committed,
                                 size_t regcap, size_t need)
      {
         const size_t target = (need + chunk_bytes - 1) / chunk_bytes * chunk_bytes;
         if (target > regcap) throw std::bad_alloc();
         if (fd >= 0)
         {
            const off_t fend = off_t(file_off + target);
            struct stat st;
            if (::fstat(fd, &st) != 0) throw std::bad_alloc();
            if (st.st_size < fend && ::ftruncate(fd, fend) != 0) throw std::bad_alloc();
            if (::mmap(mem_base + committed, target - committed, PROT_READ | PROT_WRITE,
                       MAP_SHARED | MAP_FIXED, fd, off_t(file_off + committed)) == MAP_FAILED)
               throw std::bad_alloc();
         }
         else if (::mprotect(mem_base + committed, target - committed, PROT_READ | PROT_WRITE) != 0)
            throw std::bad_alloc();
         committed = target;
      }
      // Node lines live past the header (file offset sb_bytes_); terminals fill their file from 0.
      void commit_(size_t need) { commit_region_(base_, nfd_, sb_bytes_, committed_, ncap_, need); }
      void tcommit_(size_t need) { commit_region_(tbase_, tfd_, 0, tcommitted_, tcap_, need); }

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

      std::byte* nbase_      = nullptr;  // nodes-region reservation start (superblock at [0,sb_bytes_))
      std::byte* base_       = nullptr;  // node lines = nbase_ + sb_bytes_ (cached; don't recompute)
      std::byte* tbase_      = nullptr;  // terminal-region reservation start (its own file, dense)
      size_t     sb_bytes_   = 0;        // superblock reservation in the nodes file (0 anonymous)
      size_t     node_resv_  = 0;        // nodes mapping size = sb_bytes_ + ncap_
      size_t     term_resv_  = 0;        // terminal mapping size = tcap_
      uint64_t   uroot_      = 0;        // tree root handle restored from the superblock on reopen
      uint64_t   ucount_     = 0;        // element count restored on reopen
      bool       restored_   = false;    // this open restored an existing image (vs fresh)
      size_t     ncap_       = 0;
      size_t     tcap_       = 0;
      size_t     committed_  = 0;
      size_t     tcommitted_ = 0;
      size_t     frontier_   = 0;  // node bump position, in lines
      size_t     tfrontier_  = 0;  // terminal bump position, in 16 B units
      int        nfd_        = -1;  // nodes file (128 B lines; holds the superblock header)
      int        tfd_        = -1;  // terminals file (16 B units)
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

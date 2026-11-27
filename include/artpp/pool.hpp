// SPDX-License-Identifier: MIT
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
// map::checkpoint() (or checkpoint(root, count)) before close to persist; a later line_pool(dir)
// restores it with no relocation (offset handles need none) and map{alloc, artpp::attach} binds
// to it. A map never tears its tree down when it is destroyed on a file-backed pool: the tree
// belongs to the store. Crash/reboot durability (full data sync / WAL) is the next stage —
// checkpoint flushes the header, and data pages ride the page cache across a reopen.
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
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <new>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

#include <fcntl.h>
#include <time.h>
#include <sys/file.h>
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

   // ── what the allocation path throws ─────────────────────────────────────────────────
   // Both are std::bad_alloc: to the tree, and to any caller, an allocation that failed.
   // Neither allocates memory to say so.
   //
   // The region is full: the capacity given to the constructor, or the limit of the handle
   // itself (2^28 lines = 32 GB of nodes, 2^28 units = 4 GB of terminals), is used up.
   class pool_exhausted : public std::bad_alloc
   {
      const char* msg_;

     public:
      explicit pool_exhausted(const char* m) noexcept : msg_(m) {}
      const char* what() const noexcept override { return msg_; }
   };
   // The operating system refused to give the pool more memory or to grow its file (mmap,
   // mprotect, ftruncate, fstat): code() is the errno of the call.
   class pool_commit_failed : public std::bad_alloc
   {
      int  err_;
      char msg_[96];

     public:
      pool_commit_failed(int err, const char* what_failed) noexcept : err_(err)
      {
         std::snprintf(msg_, sizeof msg_, "artpp::line_pool: %s failed (errno %d)", what_failed, err);
      }
      int         code() const noexcept { return err_; }
      const char* what() const noexcept override { return msg_; }
   };

   // ── what opening a store directory can refuse ───────────────────────────────────────
   // All derive from store_error (a std::runtime_error) and name the store and the reason in
   // what(). A failed open never modifies a file. Operating-system failures (open, mmap,
   // ftruncate, msync, ...) are reported as std::system_error with the errno of the call.
   class store_error : public std::runtime_error
   {
     public:
      using std::runtime_error::runtime_error;
   };
   // The files are not a store, or not a whole one: a foreign file, a truncated file, a
   // superblock field out of range, or `nodes` and `terms` from different checkpoints.
   class store_corrupt : public store_error
   {
     public:
      using store_error::store_error;
   };
   // A store, but not one this build can use: another format version, byte order, geometry or
   // page size.
   class store_incompatible : public store_error
   {
     public:
      using store_error::store_error;
   };

   // A store whose image was modified after its last checkpoint and not checkpointed again
   // (the writer closed without one, or died). The superblock describes a tree that the pages
   // no longer hold; there is nothing to repair it from.
   class store_dirty : public store_error
   {
     public:
      using store_error::store_error;
   };
   // The store is open in another line_pool, in this process or another one.
   class store_locked : public store_error
   {
     public:
      using store_error::store_error;
   };

   // The line store. Single-threaded, like the tree. Not copyable/movable — every handle
   // in every tree on this pool encodes an offset from base_.
   class line_pool
   {
     public:
      static constexpr size_t line_bytes  = 128;
      static constexpr size_t chunk_bytes = size_t(4) << 20;  // commit/grow unit
      static constexpr size_t max_bytes   = size_t(1) << 35;  // 32 GB — the 28-bit index cap

      explicit line_pool(size_t cap = max_bytes)  // anonymous
      {
         try
         {
            set_caps_(cap);
            reserve_();
            reset_lists_();
         }
         catch (...)
         {
            destroy_();  // a throwing constructor has no destructor
            throw;
         }
      }
      // File-backed: `dir` is a STORE DIRECTORY holding one dense file per region (`nodes` +
      // `terms`). Opens the store that is there, or creates one when there is none (no files,
      // or two empty ones); `cap` only applies to a store being created. An existing store is
      // validated before anything is mapped or written, and the open throws — store_corrupt,
      // store_incompatible (see above) or std::system_error — rather than use, repair or
      // overwrite files it cannot vouch for.
      //
      // `page_size` is the page size the store is laid out and checked for: 0, the default,
      // means the system's. A multiple of it may be given to create a store for, or to open
      // one as, a system with larger pages would (the layout is the same for every page
      // size up to 64 KB, so this only matters beyond that, and to tests).
      explicit line_pool(const char* dir, size_t cap = max_bytes, size_t page_size = 0)
      {
         try
         {
            open_store_(dir, cap, page_size);
         }
         catch (...)
         {
            destroy_();
            throw;
         }
      }
      line_pool(const line_pool&)            = delete;
      line_pool& operator=(const line_pool&) = delete;
      // Not movable either: every allocator, and through it every map, holds a pointer to the
      // pool, and the lifecycle state (lock, dirty mark, bound stamp) lives in it.
      line_pool(line_pool&&)            = delete;
      line_pool& operator=(line_pool&&) = delete;
      ~line_pool() { destroy_(); }  // cannot report: see close()

      // Release the mappings, the files and the lock now, and report a failure to do so
      // (std::system_error with the errno; everything is released regardless). The destructor
      // does the same and has to ignore failures. Closing does NOT checkpoint. Afterwards the
      // pool is dead: no container may still be using it, and only close() and the destructor
      // may be called.
      void close()
      {
         int         err  = 0;
         const char* call = nullptr;
         destroy_(&err, &call);
         if (err) os_error_(err, call, dir_.empty() ? std::string("anonymous pool") : dir_);
      }

      // Terminal region: 16-byte-granular storage for leaves (tag K::value_ptr). A
      // terminal is pure payload — suffix bytes + value — with no branch slots, so it
      // needs only the 4 tag bits of alignment, not a full line. Splitting the store
      // into a node region (128 B lines) and a terminal region (16 B units) cuts the
      // terminal working set up to 8x; the handle's TAG selects the decode, so every
      // deref site picks its region statically (descent is already tag-dispatched).
      static constexpr size_t term_unit      = 16;
      static constexpr size_t max_term_bytes = size_t(1) << 32;  // 4 GB — u32 offsets

      bool       file_backed() const noexcept { return hdr_ != 0; }
      // True when the image of a file-backed pool has been modified since it was last made
      // consistent (by checkpoint(), or by the open): closing now leaves a store that the next
      // open refuses with store_dirty. Always false for an anonymous pool, which has no image.
      bool dirty() const noexcept { return file_backed() && !clean_() && !dead_; }
      std::byte* base() const noexcept { return base_; }
      std::byte* term_base() const noexcept { return tbase_; }
      size_t     committed() const noexcept { return committed_; }
      size_t     term_committed() const noexcept { return tcommitted_; }
      size_t     used_lines() const noexcept { return frontier_; }
      size_t     used_term_units() const noexcept { return tfrontier_; }
      // Total on-disk bytes across the store's region files. Each region is its own dense
      // file (no sparse holes), so this is just the committed extents + the two headers.
      size_t file_extent() const noexcept { return 2 * hdr_ + committed_ + tcommitted_; }

      // ── persistence (file-backed, clean-close durability) ────────────────────────
      // checkpoint() stamps the current carving state + the tree's root handle and count
      // into the superblock and flushes it, so a later line_pool(path) over the same file
      // restores them and the tree round-trips with no relocation (offset handles). The
      // data pages ride the page cache across a clean reopen; crash/reboot durability (a
      // full data sync / WAL) is a further stage — see the file header.
      //
      // restored() is true when this open found an image that a checkpoint() call had written;
      // root()/count() are then what that call recorded. A store that was created but never
      // checkpointed — and an anonymous pool — is not restored: root() is the NULL handle
      // (null_root, NOT 0: 0 decodes as a leaf at terminal offset 0) and count() is 0, so
      // attaching them yields an empty map.
      static constexpr uint64_t null_root = 0xFFFFFFFFu;  // == detail::line_ptr::null().raw()
      bool     restored() const noexcept { return restored_; }
      uint64_t root() const noexcept { return uroot_; }
      uint64_t count() const noexcept { return ucount_; }
      // Throws std::invalid_argument when `root` cannot be the root of a tree in this pool (a
      // count of 0 records the empty tree, whatever `root` is), std::system_error when the
      // image cannot be written.
      //
      // This two-argument form is the low-level call: it records the stamp of the container
      // that is bound to the store (see bind) — and NO stamp when none is, in which case
      // the image can later be attached by any instantiation. map::checkpoint() always
      // records its own.
      void checkpoint(uint64_t root, uint64_t count, const image_stamp& stamp)
      {
         const image_stamp old_stamp = stamp_;
         const bool        old_has = has_stamp_, old_bound = bound_;
         stamp_     = stamp;  // the tree being recorded is this container's, whatever was here
         has_stamp_ = bound_ = true;
         try
         {
            checkpoint(root, count);
         }
         catch (...)
         {
            stamp_     = old_stamp;
            has_stamp_ = old_has;
            bound_     = old_bound;
            throw;
         }
      }
      // Bind a container instantiation to the store, by its image_stamp: what map's attach
      // constructors do. Throws store_incompatible when the image was written by, or the
      // store is already bound to, a different instantiation. No effect on an anonymous pool.
      void bind(const image_stamp& stamp)
      {
         if (!file_backed()) return;
         if (has_stamp_ && stamp_ != stamp) incompatible_(dir_, stamp_mismatch_(stamp_, stamp));
         stamp_     = stamp;
         has_stamp_ = bound_ = true;
      }
      void checkpoint(uint64_t root, uint64_t count)
      {
         if (count == 0)
            root = null_root;
         else if (!root_ok_(root, frontier_, tfrontier_))
            throw std::invalid_argument(
                "artpp::line_pool::checkpoint: the root handle does not address this pool");
         if (!nbase_) throw std::logic_error("artpp::line_pool::checkpoint: the pool is closed");
         if (file_backed()) write_image_(root, count, checkpoints_ + 1);
         ++checkpoints_;
         uroot_  = root;
         ucount_ = count;
      }
      // A container tells the pool which tree it is letting go of without freeing it (a map
      // being destroyed or detached on a file-backed pool). If that is not the tree of the
      // last checkpoint, the image is not what the superblock says, whether or not the pool
      // saw an allocation in between: mark it. (Changes that allocate and free nothing — a
      // key that fits a free slot of a node — reach the pool only this way.)
      void released(uint64_t root, uint64_t count) noexcept
      {
         if (file_backed() && clean_() && (root != uroot_ || count != ucount_)) mark_dirty_();
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
         if (clean_()) mark_dirty_();              // the whole image is about to be replaced
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
         if (n <= free_max_)
         {
            if (const uint32_t head = free_[n]; head != NIL)
            {
               std::byte* p = base_ + size_t(head) * line_bytes;
               std::memcpy(&free_[n], p, 4);  // pop: the next index lives in the freed block
               return p;
            }
         }
         else if (void* p = alloc_lines_rare_(n))
            return p;
         // Compared in lines, not bytes: `(frontier_ + n) * line_bytes` wraps for an absurd n
         // and would then pass. frontier_ never exceeds the committed lines.
         if (n > committed_ / line_bytes - frontier_) grow_lines_(n);
         std::byte* p = base_ + frontier_ * line_bytes;
         frontier_ += n;
         return p;
      }
      void free_lines(void* p, size_t n) noexcept
      {
         const uint32_t idx = uint32_t((static_cast<std::byte*>(p) - base_) / line_bytes);
         if (n <= free_max_)
         {
            std::memcpy(p, &free_[n], 4);
            free_[n] = idx;
         }
         else
            free_lines_rare_(p, n, idx);
      }

      // Terminal-region twins of alloc_lines/free_lines, in 16-byte units.
      void* alloc_term(size_t n)
      {
         if (n <= tfree_max_)
         {
            if (const uint32_t head = tfree_[n]; head != NIL)
            {
               std::byte* p = tbase_ + size_t(head) * term_unit;
               std::memcpy(&tfree_[n], p, 4);
               return p;
            }
         }
         else if (void* p = alloc_term_rare_(n))
            return p;
         if (n > tcommitted_ / term_unit - tfrontier_) grow_terms_(n);
         std::byte* p = tbase_ + tfrontier_ * term_unit;
         tfrontier_ += n;
         return p;
      }
      void free_term(void* p, size_t n) noexcept
      {
         const uint32_t idx = uint32_t((static_cast<std::byte*>(p) - tbase_) / term_unit);
         if (n <= tfree_max_)
         {
            std::memcpy(p, &tfree_[n], 4);
            tfree_[n] = idx;
         }
         else
            free_term_rare_(p, n, idx);
      }

      static constexpr uint32_t INDEX_BITS = 28, TAG_BITS = 4;  // the handle: see detail::line_ptr
      static constexpr uint32_t NIL       = 0xFFFFFFFFu;
      static constexpr size_t   FREE_MAX  = 1024;  // node lists cover up to 128 KB
      static constexpr size_t   TFREE_MAX = 4096;  // terminal lists cover up to 64 KB

      // ── the image: a header at the front of each region file ─────────────────────────
      // A file-backed pool reserves `hdr_` bytes at the front of BOTH files and maps the
      // regions past them, so every node/terminal offset stays base-relative — identical to
      // an anonymous pool, so adopt() and every minted handle are unaffected. The `nodes`
      // header is the superblock: what the image depends on that the pool can know (format
      // version, byte order, its own size and alignment, line size, handle layout, list
      // sizes), the geometry, the carving state (frontier + free-list heads; the chains
      // themselves live inside the freed blocks) and the tree's root handle + count. The
      // `terms` header only says which store and which checkpoint the file belongs to, so
      // that a `terms` file copied from elsewhere, or from another moment, is not taken for
      // the right one. Everything is validated by open_store_ before it is believed.
      // The layout is a detail of this version of the library: public only so that tools and
      // tests can name the fields.
      static constexpr uint64_t MAGIC      = 0x6c6f6f7070747261ULL;  // "artppool" (little-endian)
      static constexpr uint64_t TERM_MAGIC = 0x6d72657470747261ULL;  // "artpterm"
      // 2: null root in a fresh store, checkpoint count, identity fields, a header in `terms`,
      //    dirty flag, container stamp, page-size-driven header reservation (64 KB, was 32 KB).
      static constexpr uint32_t VERSION         = 2;
      static constexpr uint32_t BYTE_ORDER_MARK = 0x01020304u;
      struct superblock
      {
         uint64_t magic;
         uint32_t version;
         uint32_t byte_order;              // BYTE_ORDER_MARK as the writing machine stored it
         uint32_t sb_size, sb_align;       // sizeof / alignof(superblock) of the writing build
         uint32_t line_bytes, term_unit;   // region granularities
         uint32_t index_bits, tag_bits;    // handle layout: 28-bit index + 4-bit tag
         uint32_t free_max, tfree_max;     // sizes of the two free-list tables
         uint64_t chunk_bytes;             // commit unit: `committed` is a multiple of it
         uint64_t header_bytes;            // the reservation at the front of each file
         uint64_t store_id;                // drawn when the store is created
         uint64_t generation;              // +1 on every image write; `terms` carries the same
         uint64_t checkpoints;             // user checkpoint() calls so far; 0 = never
         uint64_t dirty;                   // 1: modified since the last image write
         uint64_t stamped;                 // 1: `stamp` is that of the container that wrote the tree
         uint64_t stamp[8];                // its image_stamp (see map::image_stamp)
         uint64_t ncap, tcap, frontier, tfrontier, committed, tcommitted;
         uint32_t huge, thuge;
         uint64_t uroot, ucount;             // user metadata: tree root handle + element count
         uint32_t free_heads[FREE_MAX + 1];  // exact-size free-list heads
         uint32_t tfree_heads[TFREE_MAX + 1];
      };
      struct terms_header
      {
         uint64_t magic;
         uint32_t version;
         uint32_t byte_order;
         uint64_t header_bytes, store_id, generation;
      };

      // ── page size ─────────────────────────────────────────────────────────────────────
      // The header reservation is the FILE OFFSET at which each file's data is mapped, and
      // mmap wants that to be a multiple of the page size — of the system doing the mapping,
      // which is not a constant: 4 KB on x86-64, 16 KB on Apple silicon, 64 KB on ppc64le
      // and on arm64 kernels configured that way. So the reservation is computed from the
      // page size, with a floor of 64 KB: one layout for all of those, and a store moves
      // between them. It is recorded in the superblock, and a store whose reservation this
      // system cannot map (made for smaller pages than ours) is refused, never mapped askew.
      // Everything else the pool maps is at a multiple of chunk_bytes past the reservation.
      static constexpr size_t min_header_bytes = 65536;
      static_assert(sizeof(superblock) <= min_header_bytes && chunk_bytes % min_header_bytes == 0);
      // The reservation of a store created where pages are `page_size` bytes.
      static constexpr size_t header_bytes_for(size_t page_size) noexcept
      {
         const size_t need = std::max(sizeof(superblock), min_header_bytes);
         return (need + page_size - 1) / page_size * page_size;
      }
      // Can a store with this reservation be mapped where pages are `page_size` bytes?
      static constexpr bool header_fits_pages(uint64_t header_bytes, size_t page_size) noexcept
      {
         return page_size != 0 && header_bytes >= sizeof(superblock) && header_bytes % page_size == 0 &&
                chunk_bytes % page_size == 0;
      }
      static size_t system_page_size()
      {
         const long p = ::sysconf(_SC_PAGESIZE);
         if (p <= 0) throw std::system_error(errno, std::generic_category(), "artpp::line_pool: sysconf(_SC_PAGESIZE)");
         return size_t(p);
      }

     private:
      // ── refusals ──────────────────────────────────────────────────────────────────────
      [[noreturn]] static void corrupt_(const std::string& dir, const std::string& why)
      {
         throw store_corrupt("artpp::line_pool: " + dir + ": " + why);
      }
      [[noreturn]] static void incompatible_(const std::string& dir, const std::string& why)
      {
         throw store_incompatible("artpp::line_pool: " + dir + ": " + why);
      }
      // A failed system call: std::system_error carrying `err` (errno of the call).
      [[noreturn]] static void os_error_(int err, const char* call, const std::string& path)
      {
         throw std::system_error(err, std::generic_category(),
                                 std::string("artpp::line_pool: ") + call + "(" + path + ")");
      }
      static std::string num_(uint64_t v) { return std::to_string(v); }
      static std::string stamp_mismatch_(const image_stamp& have, const image_stamp& want)
      {
         std::string m = "the store was written by a different map instantiation (mapped type size/"
                         "alignment, mode, key encoding or handle differ); image stamp words that differ:";
         for (size_t i = 0; i < have.size(); ++i)
            if (have[i] != want[i])
               m += " [" + num_(i) + "] stored " + num_(have[i]) + ", attaching " + num_(want[i]) + ";";
         return m;
      }

      // Can `raw` be the root of a non-empty tree whose regions end at these frontiers?
      static bool root_ok_(uint64_t raw, uint64_t frontier, uint64_t tfrontier) noexcept
      {
         if (raw >> 32) return false;  // a handle is 4 bytes
         const unsigned tag = unsigned(raw & 0xF);
         if (tag == unsigned(detail::K::value_ptr)) return (raw >> 4) < tfrontier;  // a leaf
         if (tag == unsigned(detail::K::value_inline)) return true;  // payload is in the handle
         if (tag > detail::K_max) return false;                      // null, or no kind at all
         return (raw >> 4) < frontier;                               // a node: a line index
      }
      // The list of n-unit blocks may only start at a block that lies before the frontier.
      // Returns the first list that does not, or `max + 1`.
      static size_t bad_head_(const uint32_t* heads, size_t max, uint64_t frontier) noexcept
      {
         if (heads[0] != NIL) return 0;  // there are no 0-unit blocks
         for (size_t n = 1; n <= max; ++n)
            if (heads[n] != NIL && uint64_t(heads[n]) + n > frontier) return n;
         return max + 1;
      }
      static off_t file_size_(int fd, const std::string& path, const std::string& dir)
      {
         struct stat st;
         if (::fstat(fd, &st) != 0) os_error_(errno, "fstat", path);
         if (!S_ISREG(st.st_mode)) corrupt_(dir, "'" + path + "' is not a regular file");
         return st.st_size;
      }
      // Read exactly n bytes at offset 0; false when the file is shorter.
      static bool read_header_(int fd, void* buf, size_t n, const std::string& path)
      {
         size_t got = 0;
         while (got < n)
         {
            const ssize_t r = ::pread(fd, static_cast<char*>(buf) + got, n - got, off_t(got));
            if (r < 0)
            {
               if (errno == EINTR) continue;
               os_error_(errno, "pread", path);
            }
            if (r == 0) return false;
            got += size_t(r);
         }
         return true;
      }
      static constexpr uint64_t byteswap64_(uint64_t v) noexcept
      {
         uint64_t r = 0;
         for (int i = 0; i < 8; ++i) r = (r << 8) | ((v >> (8 * i)) & 0xFF);
         return r;
      }

      // Everything that can be checked about an existing store without reading its data.
      // Throws store_corrupt / store_incompatible; reads only, never writes.
      static void validate_(const std::string& dir, const superblock& sb, const terms_header& th,
                            uint64_t nodes_size, uint64_t terms_size, size_t page_size)
      {
         // 1. Is it a store, and one this build reads?
         if (sb.magic == byteswap64_(MAGIC))
            incompatible_(dir, "the store was written on a machine of the other byte order");
         if (sb.magic != MAGIC) corrupt_(dir, "'nodes' is not an artpp line_pool store (no magic)");
         if (sb.version != VERSION)
            incompatible_(dir, "the store has format version " + num_(sb.version) +
                                   ", this build of artpp reads only version " + num_(VERSION) +
                                   " (the format is specific to the library version)");
         const auto same = [&](const char* what, uint64_t stored, uint64_t mine) {
            if (stored != mine)
               incompatible_(dir, std::string("the store was written by a build with another layout: ") +
                                      what + " is " + num_(stored) + ", this build has " + num_(mine));
         };
         same("byte order mark", sb.byte_order, BYTE_ORDER_MARK);
         same("sizeof(superblock)", sb.sb_size, sizeof(superblock));
         same("alignof(superblock)", sb.sb_align, alignof(superblock));
         same("line size", sb.line_bytes, line_bytes);
         same("terminal unit", sb.term_unit, term_unit);
         same("handle index bits", sb.index_bits, INDEX_BITS);
         same("handle tag bits", sb.tag_bits, TAG_BITS);
         same("node free-list table size", sb.free_max, FREE_MAX);
         same("terminal free-list table size", sb.tfree_max, TFREE_MAX);
         same("commit unit", sb.chunk_bytes, chunk_bytes);
         if (sb.header_bytes < sizeof(superblock) || sb.header_bytes > nodes_size)
            corrupt_(dir, "superblock field header_bytes = " + num_(sb.header_bytes) + " is out of range");
         if (!header_fits_pages(sb.header_bytes, page_size))
            incompatible_(dir, "the store's header reservation (" + num_(sb.header_bytes) +
                                   " bytes) is not a whole number of this system's pages (" +
                                   num_(page_size) + " bytes): it was created for smaller pages and "
                                   "cannot be mapped here");

         // 2. Geometry and carving state: every field within what the others allow.
         const auto in_range = [&](bool ok, const char* field, uint64_t v) {
            if (!ok) corrupt_(dir, std::string("superblock field ") + field + " = " + num_(v) + " is out of range");
         };
         in_range(sb.ncap != 0 && sb.ncap <= max_bytes && sb.ncap % chunk_bytes == 0, "ncap", sb.ncap);
         in_range(sb.tcap != 0 && sb.tcap <= max_term_bytes && sb.tcap % chunk_bytes == 0, "tcap", sb.tcap);
         in_range(sb.committed <= sb.ncap && sb.committed % chunk_bytes == 0, "committed", sb.committed);
         in_range(sb.tcommitted <= sb.tcap && sb.tcommitted % chunk_bytes == 0, "tcommitted", sb.tcommitted);
         in_range(sb.frontier <= sb.committed / line_bytes, "frontier", sb.frontier);
         in_range(sb.tfrontier <= sb.tcommitted / term_unit, "tfrontier", sb.tfrontier);
         in_range(sb.huge == NIL || sb.huge < sb.frontier, "huge", sb.huge);
         in_range(sb.thuge == NIL || sb.thuge < sb.tfrontier, "thuge", sb.thuge);
         if (const size_t n = bad_head_(sb.free_heads, FREE_MAX, sb.frontier); n <= FREE_MAX)
            corrupt_(dir, "superblock field free_heads[" + num_(n) + "] = " + num_(sb.free_heads[n]) +
                              " is out of range");
         if (const size_t n = bad_head_(sb.tfree_heads, TFREE_MAX, sb.tfrontier); n <= TFREE_MAX)
            corrupt_(dir, "superblock field tfree_heads[" + num_(n) + "] = " + num_(sb.tfree_heads[n]) +
                              " is out of range");
         in_range(sb.dirty <= 1, "dirty", sb.dirty);
         in_range(sb.stamped <= 1, "stamped", sb.stamped);
         if (sb.checkpoints == 0)  // never checkpointed: the empty image written at creation
            in_range(sb.uroot == null_root && sb.ucount == 0, "uroot", sb.uroot);
         else if (sb.ucount == 0)
            in_range(sb.uroot == null_root, "uroot", sb.uroot);
         else
            in_range(root_ok_(sb.uroot, sb.frontier, sb.tfrontier), "uroot", sb.uroot);

         // 3. The files hold everything the checkpoint committed.
         if (nodes_size < sb.header_bytes + sb.committed)
            corrupt_(dir, "'nodes' is truncated: " + num_(nodes_size) + " bytes, the checkpoint committed " +
                              num_(sb.header_bytes + sb.committed));
         if (terms_size < sb.header_bytes + sb.tcommitted)
            corrupt_(dir, "'terms' is truncated: " + num_(terms_size) + " bytes, the checkpoint committed " +
                              num_(sb.header_bytes + sb.tcommitted));

         // 4. `terms` is the other half of THIS store at THIS checkpoint.
         if (th.magic != TERM_MAGIC || th.version != sb.version || th.byte_order != sb.byte_order ||
             th.header_bytes != sb.header_bytes || sb.header_bytes > terms_size)
            corrupt_(dir, "'terms' is not the terminal file of an artpp line_pool store");
         if (th.store_id != sb.store_id)
            corrupt_(dir, "'nodes' and 'terms' belong to different stores");
         if (th.generation != sb.generation)
            corrupt_(dir, "'nodes' and 'terms' are from different checkpoints (generation " +
                              num_(sb.generation) + " and " + num_(th.generation) + ")");
      }

      // ── construction ──────────────────────────────────────────────────────────────────
      // Release everything; also unwinds a partially-constructed pool. The first failure is
      // handed to close() through err/call; the destructor passes none. Closing the nodes
      // descriptor releases the lock.
      void destroy_(int* err = nullptr, const char** call = nullptr) noexcept
      {
         const auto note = [&](bool failed, const char* what) {
            if (failed && err && !*err) *err = errno, *call = what;
         };
         if (nbase_) note(::munmap(nbase_, node_resv_) != 0, "munmap");
         if (tmap_) note(::munmap(tmap_, term_resv_) != 0, "munmap");
         if (tfd_ >= 0) note(::close(tfd_) != 0, "close");
         if (nfd_ >= 0) note(::close(nfd_) != 0, "close");
         nbase_ = base_ = tmap_ = tbase_ = nullptr;
         nfd_ = tfd_ = -1;
         huge_ = thuge_ = NIL;
         committed_ = tcommitted_ = frontier_ = tfrontier_ = 0;  // nothing can be allocated from it
         ncap_ = tcap_ = 0;
         free_max_ = tfree_max_ = 0;  // every request takes the checked, out-of-line path
         dead_ = true;
      }
      void set_caps_(size_t cap) noexcept
      {
         cap   = std::min(std::max(cap, size_t(1)), max_bytes);  // at least one chunk; no overflow below
         ncap_ = (cap + chunk_bytes - 1) / chunk_bytes * chunk_bytes;
         tcap_ = std::min(std::max(ncap_ / 8, chunk_bytes), max_term_bytes);  // 1/8th (16/128)
      }
      void reset_lists_() noexcept
      {
         std::memset(free_, 0xFF, sizeof(free_));  // every head = NIL (all-ones)
         std::memset(tfree_, 0xFF, sizeof(tfree_));
         huge_ = thuge_ = NIL;
      }
      // One PROT_NONE reservation per region; [0, hdr_) of each is the file's header.
      void reserve_()
      {
         node_resv_ = hdr_ + ncap_;
         term_resv_ = hdr_ + tcap_;
         void* nr   = ::mmap(nullptr, node_resv_, PROT_NONE, MAP_PRIVATE | MAP_ANON, -1, 0);
         if (nr == MAP_FAILED) reserve_failed_(errno);
         nbase_   = static_cast<std::byte*>(nr);
         base_    = nbase_ + hdr_;
         void* tr = ::mmap(nullptr, term_resv_, PROT_NONE, MAP_PRIVATE | MAP_ANON, -1, 0);
         if (tr == MAP_FAILED) reserve_failed_(errno);
         tmap_  = static_cast<std::byte*>(tr);
         tbase_ = tmap_ + hdr_;
      }
      [[noreturn]] void reserve_failed_(int err) const
      {
         if (file_backed()) os_error_(err, "mmap", dir_);
         throw pool_commit_failed(err, "reserving address space (mmap)");
      }
      void map_headers_()
      {
         if (::mmap(nbase_, hdr_, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, nfd_, 0) == MAP_FAILED)
            os_error_(errno, "mmap", dir_ + "/nodes");
         if (::mmap(tmap_, hdr_, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, tfd_, 0) == MAP_FAILED)
            os_error_(errno, "mmap", dir_ + "/terms");
      }
      superblock*   sb_() const noexcept { return reinterpret_cast<superblock*>(nbase_); }
      terms_header* th_() const noexcept { return reinterpret_cast<terms_header*>(tmap_); }

      void open_store_(const char* dir, size_t cap, size_t page)
      {
         if (!dir || !*dir) throw std::invalid_argument("artpp::line_pool: empty store directory name");
         const size_t sys_page = system_page_size();
         if (page == 0) page = sys_page;
         if (page % sys_page != 0 || chunk_bytes % page != 0)
            throw std::invalid_argument("artpp::line_pool: page_size must be a multiple of the system's page "
                                        "size (" + num_(sys_page) + ") that divides the commit unit (" +
                                        num_(chunk_bytes) + ")");
         dir_                 = dir;
         const std::string np = dir_ + "/nodes";  // 128 B lines, behind the superblock
         const std::string tp = dir_ + "/terms";  // 16 B terminal units, behind their header
         // Open what is there first: nothing is created next to a file that is then refused.
         nfd_ = ::open(np.c_str(), O_RDWR | O_CLOEXEC);
         if (nfd_ < 0 && errno != ENOENT) os_error_(errno, "open", np);
         tfd_ = ::open(tp.c_str(), O_RDWR | O_CLOEXEC);
         if (tfd_ < 0 && errno != ENOENT) os_error_(errno, "open", tp);
         if (nfd_ < 0 || tfd_ < 0)
         {
            // A store is both files. One of them alone, with content, is half a store.
            if (nfd_ >= 0 && file_size_(nfd_, np, dir_) != 0)
               corrupt_(dir_, "the store is incomplete: 'terms' is missing");
            if (tfd_ >= 0 && file_size_(tfd_, tp, dir_) != 0)
               corrupt_(dir_, "the store is incomplete: 'nodes' is missing");
            if (::mkdir(dir, 0755) != 0 && errno != EEXIST) os_error_(errno, "mkdir", dir_);
            if (nfd_ < 0 && (nfd_ = ::open(np.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644)) < 0)
               os_error_(errno, "open", np);
            if (tfd_ < 0 && (tfd_ = ::open(tp.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644)) < 0)
               os_error_(errno, "open", tp);
         }
         // One pool per store: an advisory exclusive lock on `nodes`, held until the descriptor
         // is closed (also by the death of the process). Taken before anything is read, so the
         // validation below sees a store no one else is writing.
         while (::flock(nfd_, LOCK_EX | LOCK_NB) != 0)
         {
            if (errno == EINTR) continue;
            if (errno == EWOULDBLOCK)
               throw store_locked("artpp::line_pool: " + dir_ +
                                  ": the store is open in another line_pool (this process or another)");
            os_error_(errno, "flock", np);
         }
         const uint64_t nsz = uint64_t(file_size_(nfd_, np, dir_));
         const uint64_t tsz = uint64_t(file_size_(tfd_, tp, dir_));
         reset_lists_();
         if (nsz == 0 && tsz == 0)
         {
            hdr_ = header_bytes_for(page);
            create_(cap);
            return;
         }
         if (nsz == 0) corrupt_(dir_, "the store is incomplete: 'nodes' is empty and 'terms' is not");
         if (tsz == 0) corrupt_(dir_, "the store is incomplete: 'terms' is empty and 'nodes' is not");
         superblock   sb;
         terms_header th;
         if (!read_header_(nfd_, &sb, sizeof sb, np))
            corrupt_(dir_, "'nodes' is not an artpp line_pool store (" + num_(nsz) +
                               " bytes: too short to hold a superblock)");
         if (!read_header_(tfd_, &th, sizeof th, tp))
            corrupt_(dir_, "'terms' is not the terminal file of an artpp line_pool store (" + num_(tsz) +
                               " bytes: too short)");
         validate_(dir_, sb, th, nsz, tsz, page);
         hdr_ = sb.header_bytes;
         if (sb.dirty && sb.checkpoints != 0)
            throw store_dirty("artpp::line_pool: " + dir_ +
                              ": the store was modified after its last checkpoint and not checkpointed "
                              "again (it was closed without one, or the process writing it died); the "
                              "image is not the tree the checkpoint recorded and cannot be used");
         if (sb.dirty)
         {
            // Our own store (it passed validation), never checkpointed: it holds nothing that
            // a checkpoint promised, only the leftovers of a session that did not keep them.
            // Start it over as the empty store it still is.
            if (::ftruncate(nfd_, 0) != 0) os_error_(errno, "ftruncate", np);
            if (::ftruncate(tfd_, 0) != 0) os_error_(errno, "ftruncate", tp);
            hdr_ = header_bytes_for(page);
            create_(cap);
            return;
         }
         restore_(sb);
      }
      // A new store: both files are empty. Stamp the empty image so that a reopen is
      // self-describing (and is told apart from a checkpointed one by `checkpoints` == 0).
      void create_(size_t cap)
      {
         set_caps_(cap);
         reserve_();
         if (::ftruncate(nfd_, off_t(hdr_)) != 0) os_error_(errno, "ftruncate", dir_ + "/nodes");
         if (::ftruncate(tfd_, off_t(hdr_)) != 0) os_error_(errno, "ftruncate", dir_ + "/terms");
         map_headers_();
         store_id_ = new_store_id_();
         write_image_(null_root, 0, 0);
      }
      // An existing store that validate_ accepted: adopt its geometry (offsets resolve against
      // the store's, not the constructor's `cap`) and map what it committed.
      void restore_(const superblock& sb)
      {
         ncap_ = sb.ncap;
         tcap_ = sb.tcap;
         reserve_();
         map_headers_();
         if (const int e = commit_region_(base_, nfd_, hdr_, committed_, ncap_, sb.committed))
            os_error_(e, "mmap", dir_ + "/nodes");
         if (const int e = commit_region_(tbase_, tfd_, hdr_, tcommitted_, tcap_, sb.tcommitted))
            os_error_(e, "mmap", dir_ + "/terms");
         frontier_    = sb.frontier;
         tfrontier_   = sb.tfrontier;
         huge_        = sb.huge;
         thuge_       = sb.thuge;
         store_id_    = sb.store_id;
         generation_  = sb.generation;
         checkpoints_ = sb.checkpoints;
         std::memcpy(free_, sb.free_heads, sizeof(free_));
         std::memcpy(tfree_, sb.tfree_heads, sizeof(tfree_));
         has_stamp_ = sb.stamped != 0;
         std::memcpy(stamp_.data(), sb.stamp, sizeof(sb.stamp));
         if (checkpoints_ != 0)  // else: only ever stamped at creation — an empty store
         {
            uroot_    = sb.uroot;
            ucount_   = sb.ucount;
            restored_ = true;
         }
         mark_clean_();
      }
      // Identifies a store among others: `nodes` and `terms` must agree on it. Not a secret
      // and not a hash of the content — time, process and address mixed (splitmix64).
      uint64_t new_store_id_() const noexcept
      {
         struct timespec ts{};
         ::clock_gettime(CLOCK_REALTIME, &ts);
         uint64_t z = uint64_t(ts.tv_sec) * 1000000000u + uint64_t(ts.tv_nsec);
         z ^= uint64_t(::getpid()) << 32;
         z ^= uint64_t(reinterpret_cast<uintptr_t>(this));
         z += 0x9e3779b97f4a7c15ULL;
         z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
         z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
         z ^= z >> 31;
         return z ? z : 1;
      }
      // Write the image: the carving state, the root and count, and how many times the user
      // has checkpointed (0 = the empty image stamped at creation). In this order, each step
      // flushed (msync MS_SYNC) and checked before the next:
      //   1. the data of both regions, up to their frontiers — the tree itself;
      //   2. the `terms` header;
      //   3. the superblock, which is what declares the image valid, clean and of this
      //      generation.
      // So the superblock never describes data that was not handed to the file first, and a
      // write error (a full disk under a file that grew, an I/O error) surfaces HERE, as an
      // exception from checkpoint(), instead of being lost in the page cache after a close
      // that looked clean. It is still not crash consistency: the image is updated in place.
      void write_image_(uint64_t root, uint64_t count, uint64_t checkpoints)
      {
         if (frontier_ && ::msync(base_, frontier_ * line_bytes, MS_SYNC) != 0)
            os_error_(errno, "msync", dir_ + "/nodes");
         if (tfrontier_ && ::msync(tbase_, tfrontier_ * term_unit, MS_SYNC) != 0)
            os_error_(errno, "msync", dir_ + "/terms");
         const uint64_t gen = generation_ + 1;
         terms_header*  th  = th_();
         th->magic          = TERM_MAGIC;
         th->version        = VERSION;
         th->byte_order     = BYTE_ORDER_MARK;
         th->header_bytes   = hdr_;
         th->store_id       = store_id_;
         th->generation     = gen;
         if (::msync(tmap_, hdr_, MS_SYNC) != 0) os_error_(errno, "msync", dir_ + "/terms");
         superblock* sb   = sb_();
         sb->magic        = MAGIC;
         sb->version      = VERSION;
         sb->byte_order   = BYTE_ORDER_MARK;
         sb->sb_size      = uint32_t(sizeof(superblock));
         sb->sb_align     = uint32_t(alignof(superblock));
         sb->line_bytes   = uint32_t(line_bytes);
         sb->term_unit    = uint32_t(term_unit);
         sb->index_bits   = INDEX_BITS;
         sb->tag_bits     = TAG_BITS;
         sb->free_max     = uint32_t(FREE_MAX);
         sb->tfree_max    = uint32_t(TFREE_MAX);
         sb->chunk_bytes  = chunk_bytes;
         sb->header_bytes = hdr_;
         sb->store_id     = store_id_;
         sb->generation   = gen;
         sb->checkpoints  = checkpoints;
         sb->dirty        = 0;
         // The stamp of the container bound in THIS session, or none: a tree checkpointed by
         // hand without any attach is not vouched for by the stamp of an earlier session.
         if (!bound_) has_stamp_ = false, stamp_ = image_stamp{};
         sb->stamped = has_stamp_;
         std::memcpy(sb->stamp, stamp_.data(), sizeof(sb->stamp));
         sb->ncap         = ncap_;
         sb->tcap         = tcap_;
         sb->frontier     = frontier_;
         sb->tfrontier    = tfrontier_;
         sb->committed    = committed_;
         sb->tcommitted   = tcommitted_;
         sb->huge         = huge_;
         sb->thuge        = thuge_;
         sb->uroot        = root;
         sb->ucount       = count;
         std::memcpy(sb->free_heads, free_, sizeof(free_));
         std::memcpy(sb->tfree_heads, tfree_, sizeof(tfree_));
         if (::msync(nbase_, hdr_, MS_SYNC) != 0) os_error_(errno, "msync", dir_ + "/nodes");
         generation_ = gen;
         mark_clean_();
      }
      // Grow a region's mapping to cover `need` bytes: extend its OWN file densely from `file_off`
      // and map the new chunks (MAP_FIXED over the PROT_NONE reservation), or mprotect for an
      // anonymous pool. One file per region ⇒ dense growth, no sparse holes. Returns 0, the
      // errno of the call that failed, or -1 when `need` is beyond the region's capacity.
      static int commit_region_(std::byte* mem_base, int fd, size_t file_off, size_t& committed,
                                size_t regcap, size_t need) noexcept
      {
         if (need > regcap) return -1;
         const size_t target = (need + chunk_bytes - 1) / chunk_bytes * chunk_bytes;
         if (target <= committed) return 0;
         if (fd >= 0)
         {
            const off_t fend = off_t(file_off + target);
            struct stat st;
            if (::fstat(fd, &st) != 0) return errno;
            if (st.st_size < fend && ::ftruncate(fd, fend) != 0) return errno;
            if (::mmap(mem_base + committed, target - committed, PROT_READ | PROT_WRITE,
                       MAP_SHARED | MAP_FIXED, fd, off_t(file_off + committed)) == MAP_FAILED)
               return errno;
         }
         else if (::mprotect(mem_base + committed, target - committed, PROT_READ | PROT_WRITE) != 0)
            return errno;
         committed = target;
         return 0;
      }
      // Node lines and terminals both live past their file's header (file offset hdr_).
      void commit_(size_t need)
      {
         if (const int e = commit_region_(base_, nfd_, hdr_, committed_, ncap_, need)) commit_failed_(e, false);
      }
      void tcommit_(size_t need)
      {
         if (const int e = commit_region_(tbase_, tfd_, hdr_, tcommitted_, tcap_, need)) commit_failed_(e, true);
      }
      [[noreturn]] void commit_failed_(int e, bool terms) const
      {
         if (dead_) throw pool_exhausted("artpp::line_pool: the pool is closed");
         if (e < 0)
            throw pool_exhausted(terms ? "artpp::line_pool: the terminal region is full (its capacity, or the "
                                         "4 GB a handle can address)"
                                       : "artpp::line_pool: the node region is full (its capacity, or the "
                                         "2^28 lines a handle can address)");
         throw pool_commit_failed(e, file_backed() ? (terms ? "growing the 'terms' file" : "growing the 'nodes' file")
                                                   : "committing memory (mprotect)");
      }
      // The bump path found the committed range too small for n more units: commit more, or
      // throw. Out of line; the capacity test is made in units so that it cannot overflow.
#if defined(__GNUC__)
      __attribute__((noinline))
#endif
      void grow_lines_(size_t n)
      {
         if (n > ncap_ / line_bytes - frontier_) commit_failed_(-1, false);
         commit_((frontier_ + n) * line_bytes);
      }
#if defined(__GNUC__)
      __attribute__((noinline))
#endif
      void grow_terms_(size_t n)
      {
         if (n > tcap_ / term_unit - tfrontier_) commit_failed_(-1, true);
         tcommit_((tfrontier_ + n) * term_unit);
      }

      // ── the dirty mark, at no cost to the allocation path ─────────────────────────────
      // While the image of a file-backed pool is CLEAN (nothing changed since the open or the
      // last checkpoint), free_max_ and tfree_max_ are 0 instead of FREE_MAX / TFREE_MAX. The
      // size test that alloc/free make anyway (`n <= free_max_`: does this size have an exact
      // list?) then sends every request down the branch of the rare giant block, and that
      // branch, out of line, is where the first mutation is noticed: it marks the superblock
      // dirty, restores the two limits and retries. After that, and always for an anonymous
      // pool, the paths are exactly what they were — the limit is a member instead of a
      // constant, and there is no added branch or store.
      bool clean_() const noexcept { return free_max_ == 0 && !dead_; }
      void mark_clean_() noexcept
      {
         if (file_backed()) free_max_ = tfree_max_ = 0;
      }
      void mark_dirty_() noexcept
      {
         free_max_  = FREE_MAX;
         tfree_max_ = TFREE_MAX;
         sb_()->dirty = 1;  // through the shared mapping: on disk even if this process dies
      }
#if defined(__GNUC__)
#define ARTPP_POOL_RARE_ __attribute__((noinline, cold))
#else
#define ARTPP_POOL_RARE_
#endif
      ARTPP_POOL_RARE_ void* alloc_lines_rare_(size_t n)
      {
         if (clean_())
         {
            mark_dirty_();
            return alloc_lines(n);
         }
         return pop_huge_(n);  // null: nothing of that size, bump the frontier
      }
      ARTPP_POOL_RARE_ void free_lines_rare_(void* p, size_t n, uint32_t idx) noexcept
      {
         if (clean_())
         {
            mark_dirty_();
            return free_lines(p, n);
         }
         const uint32_t hdr[2] = {huge_, uint32_t(n)};  // giant node: one list, {next, nlines}
         std::memcpy(p, hdr, 8);                        // kept in the block
         huge_ = idx;
      }
      ARTPP_POOL_RARE_ void* alloc_term_rare_(size_t n)
      {
         if (clean_())
         {
            mark_dirty_();
            return alloc_term(n);
         }
         return pop_thuge_(n);
      }
      ARTPP_POOL_RARE_ void free_term_rare_(void* p, size_t n, uint32_t idx) noexcept
      {
         if (clean_())
         {
            mark_dirty_();
            return free_term(p, n);
         }
         const uint32_t hdr[2] = {thuge_, uint32_t(n)};  // giant terminal (very long key
         std::memcpy(p, hdr, 8);                         // suffix): {next, nunits} in the block
         thuge_ = idx;
      }
#undef ARTPP_POOL_RARE_

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

      std::byte* nbase_      = nullptr;  // nodes-region reservation start (superblock at [0,hdr_))
      std::byte* base_       = nullptr;  // node lines = nbase_ + hdr_ (cached; don't recompute)
      std::byte* tmap_       = nullptr;  // terminal-region reservation start (header at [0,hdr_))
      std::byte* tbase_      = nullptr;  // terminal units = tmap_ + hdr_
      size_t     hdr_        = 0;        // header reservation in each file (0 anonymous)
      size_t     node_resv_  = 0;        // nodes mapping size = hdr_ + ncap_
      size_t     term_resv_  = 0;        // terminal mapping size = hdr_ + tcap_
      uint64_t   store_id_   = 0;        // identity of the store on disk
      uint64_t   generation_ = 0;        // image writes so far (both headers carry it)
      std::string dir_;                  // the store directory, for messages
      image_stamp stamp_{};              // of the container that wrote / is bound to the image
      bool        has_stamp_ = false;    // stamp_ is one (an image written by hand has none)
      bool        bound_     = false;    // a container bound itself in this session
      bool        dead_      = false;    // closed (or never constructed): nothing may be used
      uint64_t   uroot_      = null_root;  // root handle of the last checkpoint (null: none yet)
      uint64_t   ucount_     = 0;        // element count of the last checkpoint
      uint64_t   checkpoints_ = 0;       // user checkpoint() calls over the store's life
      bool       restored_   = false;    // this open restored a checkpointed image
      size_t     ncap_       = 0;
      size_t     tcap_       = 0;
      size_t     committed_  = 0;
      size_t     tcommitted_ = 0;
      size_t     frontier_   = 0;  // node bump position, in lines
      size_t     tfrontier_  = 0;  // terminal bump position, in 16 B units
      int        nfd_        = -1;  // nodes file (128 B lines; holds the superblock header)
      int        tfd_        = -1;  // terminals file (16 B units)
      // FREE_MAX / TFREE_MAX — or 0 while a file-backed image is clean (see mark_dirty_).
      size_t     free_max_   = FREE_MAX;
      size_t     tfree_max_  = TFREE_MAX;
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
      // (Rounded up without `bytes + 15`, which wraps for a size near SIZE_MAX.)
      static constexpr size_t units_(size_t bytes) noexcept
      {
         return bytes / line_pool::term_unit + (bytes % line_pool::term_unit != 0);
      }
      void* artpp_alloc_term(size_t bytes) { return pool->alloc_term(units_(bytes)); }
      void  artpp_free_term(void* p, size_t bytes) noexcept { pool->free_term(p, units_(bytes)); }

      // Persistent-image hooks (see has_artpp_image_ in map.hpp). A file-backed pool's content
      // outlives every container: a map on it never tears its tree down when destroyed, and
      // says which tree it leaves behind (artpp_released).
      bool artpp_retains() const noexcept { return pool && pool->file_backed(); }
      void artpp_released(uint64_t root, uint64_t count) const noexcept { pool->released(root, count); }
      void artpp_bind(const image_stamp& s) const { pool->bind(s); }
      void artpp_checkpoint(uint64_t root, uint64_t count, const image_stamp& s)
      {
         pool->checkpoint(root, count, s);
      }
      std::pair<uint64_t, uint64_t> artpp_attach(const image_stamp& s) const
      {
         pool->bind(s);
         return {pool->root(), pool->count()};
      }

      // Opt-in bulk transport for map's cross-pool move assignment: adopt the source
      // pool's whole image (see line_pool::adopt). By providing this member the allocator
      // author asserts the destination pool is exclusive to the destination tree.
      void artpp_adopt(const pool_alloc& o) { pool->adopt(*o.pool); }

      // Lines for n objects of T. For a T that is a whole number of lines (the tree's node
      // block) this is one multiplication by a constant and cannot round wrongly; a count
      // too large to express saturates, and the pool then refuses it as beyond its capacity.
      static constexpr size_t lines_for_(size_t n) noexcept
      {
         constexpr size_t L = line_pool::line_bytes;
         if (sizeof(T) % L == 0)
         {
            constexpr size_t k = sizeof(T) / L;
            return k == 1 ? n : (n > ~size_t(0) / k ? ~size_t(0) : n * k);
         }
         if (n > ~size_t(0) / sizeof(T)) return ~size_t(0);
         const size_t bytes = n * sizeof(T);
         return bytes / L + (bytes % L != 0);
      }
      T* allocate(size_t n)
      {
         static_assert(alignof(T) <= line_pool::line_bytes, "pool lines are 128-byte aligned");
         return static_cast<T*>(pool->alloc_lines(lines_for_(n)));
      }
      void deallocate(T* p, size_t n) noexcept { pool->free_lines(p, lines_for_(n)); }

      template <class U>
      bool operator==(const pool_alloc<U>& o) const noexcept
      {
         return pool == o.pool;
      }
   };
}  // namespace artpp

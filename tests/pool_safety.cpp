// pool_safety — the lifecycle and validation contract of a file-backed artpp::line_pool:
// what a reopen restores, what it refuses, and that nothing is lost or overwritten silently.
// Each case is the regression test of one finding of the pre-release audit (issue numbers
// are those of github.com/gofractally/libartpp). Every case works in its own mkdtemp
// directory and removes it again, also when the case fails or throws.
#include "artpp/map.hpp"
#include "artpp/pool.hpp"

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <map>
#include <random>
#include <set>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

static int g_fail = 0;
#define CHECK(cond)                                                                          \
   do {                                                                                      \
      if (!(cond)) { std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); ++g_fail; } \
   } while (0)

using A   = artpp::pool_alloc<std::uint64_t>;
using Map = artpp::map<std::string_view, std::uint64_t, artpp::mode::none, A>;

// A unique store directory per case, removed (with whatever is in it) on scope exit.
struct temp_store
{
   std::string dir;
   temp_store()
   {
      const char* t = std::getenv("TMPDIR");
      std::string tmpl = std::string(t && *t ? t : "/tmp") + "/artpp_pool_safety_XXXXXX";
      if (!::mkdtemp(tmpl.data()))
      {
         std::printf("mkdtemp(%s): %s\n", tmpl.c_str(), std::strerror(errno));
         std::exit(2);
      }
      dir = tmpl;
   }
   temp_store(const temp_store&)            = delete;
   temp_store& operator=(const temp_store&) = delete;
   ~temp_store()
   {
      if (DIR* d = ::opendir(dir.c_str()))
      {
         while (const dirent* e = ::readdir(d))
            if (std::strcmp(e->d_name, ".") != 0 && std::strcmp(e->d_name, "..") != 0)
               ::unlink((dir + "/" + e->d_name).c_str());
         ::closedir(d);
      }
      ::rmdir(dir.c_str());
   }
   const char* c_str() const noexcept { return dir.c_str(); }
   std::string file(const char* name) const { return dir + "/" + name; }
   off_t       size(const char* name) const
   {
      struct stat st;
      return ::stat(file(name).c_str(), &st) == 0 ? st.st_size : off_t(-1);
   }
};

static std::string key(int i)
{
   return "key_number_" + std::to_string(i) + "_with_a_long_tail_so_it_needs_a_leaf";
}
// Every key in [lo, hi) present with value i + 1; returns the number that are not.
template <class M>
static int missing(M& m, int lo, int hi)
{
   int bad = 0;
   for (int i = lo; i < hi; ++i)
   {
      std::uint64_t v = 0;
      if (!m.find(key(i), v) || v != std::uint64_t(i) + 1) ++bad;
   }
   return bad;
}

// ── #9: a store that no one ever checkpointed ───────────────────────────────────────────
// It used to reopen as restored() with root 0, which is a valid-looking leaf handle, and the
// first lookup after the documented attach faulted.
static void never_checkpointed_store_reopens_empty()
{
   temp_store s;
   {
      artpp::line_pool pool(s.c_str());
      CHECK(!pool.restored());
   }  // created, closed, nothing checkpointed
   {
      artpp::line_pool pool(s.c_str());
      CHECK(!pool.restored());
      CHECK(pool.count() == 0);
      CHECK(artpp::detail::line_ptr::null().raw() == pool.root());
      Map m{A{&pool}, artpp::attach, pool.root(), pool.count()};
      CHECK(m.empty() && m.size() == 0);
      CHECK(!m.contains("x"));
      CHECK(m.begin() == m.end());
      m.insert("x", 1);  // and it is a usable empty map
      CHECK(m.contains("x"));
   }
}
// The same for a pool that has no file at all, and for an attach of "no elements".
static void null_root_everywhere_nothing_was_checkpointed()
{
   artpp::line_pool pool;
   CHECK(!pool.restored() && pool.count() == 0);
   CHECK(artpp::detail::line_ptr::null().raw() == pool.root());
   Map m{A{&pool}, artpp::attach, pool.root(), pool.count()};
   CHECK(m.empty() && !m.contains("x") && m.begin() == m.end());
   Map z{A{&pool}, artpp::attach, 0, 0};  // count 0 is an empty map whatever the handle says
   CHECK(z.empty() && !z.contains("x") && z.begin() == z.end());
}

// ── #8: checkpoint(), then the map is destroyed without detach() ────────────────────────
// ~map used to free every node into the pool's free lists, writing the links into the
// mapped file: the reopened store reported the checkpointed root and count and had no keys.
static void destroying_the_map_keeps_the_checkpointed_tree()
{
   const int  N = 2000;
   temp_store s;
   {
      artpp::line_pool pool(s.c_str());
      Map              m{A{&pool}};
      for (int i = 0; i < N; ++i) m.insert(key(i), std::uint64_t(i) + 1);
      pool.checkpoint(m.root_handle(), m.size());
   }  // no detach()
   {
      artpp::line_pool pool(s.c_str());
      CHECK(pool.restored() && pool.count() == std::uint64_t(N));
      Map m{A{&pool}, artpp::attach, pool.root(), pool.count()};
      CHECK(m.size() == size_t(N));
      CHECK(missing(m, 0, N) == 0);
   }  // attached, read, destroyed without detach(): still the same image
   {
      artpp::line_pool pool(s.c_str());
      Map              m{A{&pool}, artpp::attach, pool.root(), pool.count()};
      CHECK(m.size() == size_t(N) && missing(m, 0, N) == 0);
   }
}
// The root and count taken from the map stay valid after the map is gone, so checkpointing
// them afterwards persists the tree as well.
static void checkpoint_after_the_map_is_destroyed()
{
   const int  N = 2000;
   temp_store s;
   {
      artpp::line_pool pool(s.c_str());
      std::uint64_t    root = 0, count = 0;
      {
         Map m{A{&pool}};
         for (int i = 0; i < N; ++i) m.insert(key(i), std::uint64_t(i) + 1);
         root  = m.root_handle();
         count = m.size();
      }
      pool.checkpoint(root, count);
   }
   {
      artpp::line_pool pool(s.c_str());
      Map              m{A{&pool}, artpp::attach, pool.root(), pool.count()};
      CHECK(m.size() == size_t(N) && missing(m, 0, N) == 0);
   }
}

// The documented lifecycle — attach, mutate, checkpoint, close — repeated: every cycle sees
// exactly what the previous one checkpointed, and mutates it further.
static void lifecycle_over_many_reopen_cycles()
{
   temp_store                              s;
   std::map<std::string, std::uint64_t>    ref;
   std::mt19937_64                         rng(2024);
   for (int cycle = 0; cycle < 8; ++cycle)
   {
      artpp::line_pool pool(s.c_str());
      CHECK(pool.restored() == (cycle != 0));
      Map m{A{&pool}, artpp::attach};  // empty on the new store, the last checkpoint otherwise
      CHECK(m.size() == ref.size());
      auto ri = ref.begin();
      for (auto it = m.begin(); it != m.end(); ++it, ++ri)
         CHECK(ri != ref.end() && std::string(it.key()) == ri->first && it.value() == ri->second);
      CHECK(ri == ref.end());
      for (int i = 0; i < 3000; ++i)  // insert, overwrite and erase on top of the restored tree
      {
         const std::string k = key(int(rng() % 6000));
         if (rng() % 3 == 0)
         {
            CHECK(m.erase(k) == (ref.erase(k) != 0));
         }
         else
         {
            const std::uint64_t v = rng();
            m.insert_or_assign(k, v);
            ref[k] = v;
         }
      }
      if (cycle == 5) { m.clear(); ref.clear(); }  // an empty tree is checkpointed like any other
      CHECK(m.size() == ref.size());
      m.checkpoint();
      CHECK(pool.restored() == (cycle != 0));  // restored() describes the open, not the close
      CHECK(pool.count() == ref.size() && pool.root() == m.root_handle());
   }
   artpp::line_pool pool(s.c_str());
   Map              m{A{&pool}, artpp::attach};
   CHECK(m.size() == ref.size());
   for (auto& [k, v] : ref) { std::uint64_t got = 0; CHECK(m.find(k, got) && got == v); }
}
// A set goes through the same hooks (short keys live in the handle and allocate nothing).
static void set_lifecycle()
{
   using SA  = artpp::pool_alloc<std::string_view>;
   using Set = artpp::set<std::string_view, artpp::mode::none, SA>;
   temp_store            s;
   std::set<std::string> ref;
   for (int cycle = 0; cycle < 4; ++cycle)
   {
      artpp::line_pool pool(s.c_str());
      Set              t{SA{&pool}, artpp::attach};
      CHECK(t.size() == ref.size());
      for (auto& k : ref) CHECK(t.contains(k));
      for (int i = 0; i < 500; ++i)
      {
         const std::string k = (i % 2 ? "k" : key(cycle)) + std::to_string(cycle * 1000 + i);
         t.insert(k);
         ref.insert(k);
      }
      t.checkpoint();
   }
   artpp::line_pool pool(s.c_str());
   Set              t{SA{&pool}, artpp::attach};
   CHECK(t.size() == ref.size());
   auto ri = ref.begin();
   for (auto it = t.begin(); it != t.end(); ++it, ++ri) CHECK(ri != ref.end() && std::string(*it) == *ri);
   CHECK(ri == ref.end());
}
// What "the destructor does not free" must NOT change: an anonymous pool gets its lines back
// when a map dies, clear() gives a file-backed pool its space back, and moving a map moves
// the tree and the right to checkpoint it.
static void retention_is_only_for_the_destructor_of_a_file_backed_map()
{
   {
      artpp::line_pool pool;  // anonymous
      {
         Map m{A{&pool}};
         for (int i = 0; i < 500; ++i) m.insert(key(i), std::uint64_t(i) + 1);
      }
      const size_t lines = pool.used_lines(), terms = pool.used_term_units();
      Map          m{A{&pool}};
      for (int i = 0; i < 500; ++i) m.insert(key(i), std::uint64_t(i) + 1);
      CHECK(pool.used_lines() == lines && pool.used_term_units() == terms);  // all reused
   }
   temp_store s;
   {
      artpp::line_pool pool(s.c_str());
      Map              m{A{&pool}};
      for (int i = 0; i < 500; ++i) m.insert(key(i), std::uint64_t(i) + 1);
      const size_t lines = pool.used_lines(), terms = pool.used_term_units();
      m.clear();
      for (int i = 0; i < 500; ++i) m.insert(key(i), std::uint64_t(i) + 1);
      CHECK(pool.used_lines() == lines && pool.used_term_units() == terms);
      Map m2{std::move(m)};  // move construction: m2 owns the tree, m is empty
      CHECK(m.empty() && m2.size() == 500);
      Map m3{A{&pool}};
      m3.insert("to be replaced", 1);
      m3 = std::move(m2);  // move assignment (equal allocators): steals
      CHECK(m2.empty() && m3.size() == 500);
      m3.checkpoint();
   }  // m, m2 (empty) and m3 destroyed in some order: none of them touches the image
   {
      artpp::line_pool pool(s.c_str());
      Map              m{A{&pool}, artpp::attach};
      CHECK(m.size() == 500 && missing(m, 0, 500) == 0);
   }
}

// ── #10: files that are not the store that was checkpointed ─────────────────────────────
static std::vector<char> slurp(const std::string& path)
{
   std::vector<char> b;
   if (FILE* f = std::fopen(path.c_str(), "rb"))
   {
      char   buf[65536];
      size_t n;
      while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) b.insert(b.end(), buf, buf + n);
      std::fclose(f);
   }
   return b;
}
static void spit(const std::string& path, const std::vector<char>& b)
{
   FILE* f = std::fopen(path.c_str(), "wb");
   CHECK(f != nullptr);
   if (!f) return;
   CHECK(std::fwrite(b.data(), 1, b.size(), f) == b.size());
   std::fclose(f);
}
// A checkpointed store of n keys (value i + 1), closed.
static void build_store(const temp_store& s, int n)
{
   artpp::line_pool pool(s.c_str());
   Map              m{A{&pool}, artpp::attach};
   for (int i = 0; i < n; ++i) m.insert(key(i), std::uint64_t(i) + 1);
   m.checkpoint();
}
static bool store_is_intact(const temp_store& s, int n)
{
   artpp::line_pool pool(s.c_str());
   Map              m{A{&pool}, artpp::attach};
   return pool.restored() && m.size() == size_t(n) && missing(m, 0, n) == 0;
}
// Opening `s` must throw exactly E, with a message that names the store, and must leave
// both files byte for byte as they were.
template <class E>
static void expect_refused(const temp_store& s, const char* what, int line)
{
   const std::vector<char> nodes = slurp(s.file("nodes")), terms = slurp(s.file("terms"));
   const off_t             nsz = s.size("nodes"), tsz = s.size("terms");
   bool                    refused = false;
   try
   {
      artpp::line_pool pool(s.c_str());
   }
   catch (const E& e)
   {
      refused = true;
      if (std::string(e.what()).find(s.dir) == std::string::npos)
      {
         std::printf("FAIL line %d (%s): message does not name the store: %s\n", line, what, e.what());
         ++g_fail;
      }
      const std::runtime_error& base = e;  // every refusal is a std::runtime_error
      (void)base;
   }
   catch (const std::exception& e)
   {
      std::printf("FAIL line %d (%s): wrong exception: %s\n", line, what, e.what());
      ++g_fail;
      return;
   }
   if (!refused)
   {
      std::printf("FAIL line %d (%s): the store was opened\n", line, what);
      ++g_fail;
   }
   if (s.size("nodes") != nsz || s.size("terms") != tsz || slurp(s.file("nodes")) != nodes ||
       slurp(s.file("terms")) != terms)
   {
      std::printf("FAIL line %d (%s): the refused open modified the files\n", line, what);
      ++g_fail;
   }
}
#define EXPECT_REFUSED(E, s, what) expect_refused<E>(s, what, __LINE__)

static void truncated_terms_file_is_refused()  // scenario 3: 68,928 of 200,000 keys were missing
{
   const int  N = 20000;
   temp_store s;
   build_store(s, N);
   const off_t full = s.size("terms");
   CHECK(full > 0);
   CHECK(::truncate(s.file("terms").c_str(), full / 2) == 0);
   EXPECT_REFUSED(artpp::store_corrupt, s, "terms cut to half");
   CHECK(::truncate(s.file("terms").c_str(), full - 1) == 0);
   EXPECT_REFUSED(artpp::store_corrupt, s, "terms one byte short");
   CHECK(::truncate(s.file("terms").c_str(), 0) == 0);
   EXPECT_REFUSED(artpp::store_corrupt, s, "terms empty");
   CHECK(::unlink(s.file("terms").c_str()) == 0);
   EXPECT_REFUSED(artpp::store_corrupt, s, "terms missing");
   CHECK(s.size("terms") == -1);  // and it was not created
}
static void truncated_nodes_file_is_refused()  // scenario 4: 129,899 of 200,000 keys were missing
{
   const int  N = 20000;
   temp_store s;
   build_store(s, N);
   const off_t full = s.size("nodes");
   CHECK(full > (1 << 20));
   CHECK(::truncate(s.file("nodes").c_str(), full - (1 << 20)) == 0);  // data cut short
   EXPECT_REFUSED(artpp::store_corrupt, s, "nodes 1 MB short");
   CHECK(::truncate(s.file("nodes").c_str(), 4096) == 0);  // not even a whole superblock
   EXPECT_REFUSED(artpp::store_corrupt, s, "nodes cut inside the superblock");
   CHECK(::truncate(s.file("nodes").c_str(), 0) == 0);  // empty, next to a populated terms file
   EXPECT_REFUSED(artpp::store_corrupt, s, "nodes empty");
   CHECK(::unlink(s.file("nodes").c_str()) == 0);
   EXPECT_REFUSED(artpp::store_corrupt, s, "nodes missing");
   CHECK(s.size("nodes") == -1);
}
// Scenario 5: a 38-byte file that happened to be called `nodes` was overwritten with a
// fresh superblock. No file that is not a store may be written to, whatever its size.
static void foreign_files_are_never_overwritten()
{
   const std::string txt = "these are my precious unrelated notes\n";
   {
      temp_store s;
      spit(s.file("nodes"), std::vector<char>(txt.begin(), txt.end()));
      EXPECT_REFUSED(artpp::store_corrupt, s, "small foreign nodes");
      CHECK(s.size("terms") == -1);  // nothing was created next to it either
   }
   {
      temp_store        s;
      std::vector<char> big;
      while (big.size() < (1u << 20)) big.insert(big.end(), txt.begin(), txt.end());
      spit(s.file("nodes"), big);
      spit(s.file("terms"), big);
      EXPECT_REFUSED(artpp::store_corrupt, s, "large foreign nodes and terms");
   }
   {
      temp_store s;
      spit(s.file("terms"), std::vector<char>(txt.begin(), txt.end()));
      EXPECT_REFUSED(artpp::store_corrupt, s, "foreign terms, no nodes");
      CHECK(s.size("nodes") == -1);
   }
   {  // a real store's nodes next to a foreign terms of the right size
      temp_store s;
      build_store(s, 100);
      std::vector<char> t = slurp(s.file("terms"));
      for (size_t i = 0; i < t.size(); ++i) t[i] = txt[i % txt.size()];
      spit(s.file("terms"), t);
      EXPECT_REFUSED(artpp::store_corrupt, s, "foreign terms of the right size");
   }
}
// `nodes` and `terms` are written together and only mean something together.
static void files_of_different_checkpoints_are_refused()
{
   temp_store s;
   build_store(s, 500);
   const std::vector<char> old_terms = slurp(s.file("terms")), old_nodes = slurp(s.file("nodes"));
   {
      artpp::line_pool pool(s.c_str());
      Map              m{A{&pool}, artpp::attach};
      for (int i = 500; i < 1000; ++i) m.insert(key(i), std::uint64_t(i) + 1);
      m.checkpoint();
   }
   const std::vector<char> new_terms = slurp(s.file("terms")), new_nodes = slurp(s.file("nodes"));
   CHECK(old_terms.size() == new_terms.size() && old_nodes.size() == new_nodes.size());
   spit(s.file("terms"), old_terms);  // a stale terms file under the current nodes file
   EXPECT_REFUSED(artpp::store_corrupt, s, "stale terms");
   spit(s.file("terms"), new_terms);
   spit(s.file("nodes"), old_nodes);  // and the other way round
   EXPECT_REFUSED(artpp::store_corrupt, s, "stale nodes");
   spit(s.file("nodes"), new_nodes);
   CHECK(store_is_intact(s, 1000));
   {  // the terms file of ANOTHER store, at the same checkpoint number
      temp_store other;
      build_store(other, 500);
      {
         artpp::line_pool pool(other.c_str());
         Map              m{A{&pool}, artpp::attach};
         for (int i = 500; i < 1000; ++i) m.insert(key(i), std::uint64_t(i) + 1);
         m.checkpoint();
      }
      spit(s.file("terms"), slurp(other.file("terms")));
      EXPECT_REFUSED(artpp::store_corrupt, s, "terms of another store");
   }
}
// One superblock field at a time is given a value it cannot have. Each must be refused
// (and the file left alone), and the store must open again once the value is put back.
static void corrupted_superblock_fields_are_refused()
{
   using SB = artpp::line_pool::superblock;
   const int  N = 3000;
   temp_store s;
   build_store(s, N);
   {  // free some of each list so that the heads are populated, then checkpoint again
      artpp::line_pool pool(s.c_str());
      Map              m{A{&pool}, artpp::attach};
      for (int i = 0; i < N; i += 2) m.erase(key(i));
      for (int i = 0; i < N; i += 2) m.insert(key(i), std::uint64_t(i) + 1);
      for (int i = N; i < N + 64; ++i) m.insert(key(i), std::uint64_t(i) + 1);
      for (int i = N; i < N + 64; ++i) m.erase(key(i));
      m.checkpoint();
   }
   CHECK(store_is_intact(s, N));
   const int fd = ::open(s.file("nodes").c_str(), O_RDWR);
   CHECK(fd >= 0);
   if (fd < 0) return;
   SB sb;
   CHECK(::pread(fd, &sb, sizeof sb, 0) == ssize_t(sizeof sb));
   const std::uint64_t node_bytes = std::uint64_t(s.size("nodes"));
   struct poke
   {
      const char*   field;
      size_t        off, size;
      std::uint64_t bad;
      bool          incompatible;  // a version mismatch is a different refusal than damage
   };
#define POKE(member, value) {#member, offsetof(SB, member), sizeof(SB::member), std::uint64_t(value), false}
   const poke pokes[] = {
       POKE(magic, 0),
       POKE(magic, 0x0123456789abcdefULL),
       {"version", offsetof(SB, version), sizeof(SB::version), std::uint64_t(sb.version) + 1, true},
       {"version", offsetof(SB, version), sizeof(SB::version), 1, true},
       POKE(ncap, 0),
       POKE(ncap, sb.ncap + 4096),                              // not a whole number of chunks
       POKE(ncap, artpp::line_pool::max_bytes * 2),             // beyond what a handle addresses
       POKE(ncap, sb.committed - artpp::line_pool::chunk_bytes),  // smaller than what is committed
       POKE(tcap, 0),
       POKE(tcap, artpp::line_pool::max_term_bytes * 2),
       POKE(committed, sb.committed + artpp::line_pool::chunk_bytes),  // more than the file holds
       POKE(committed, node_bytes * 4),
       POKE(committed, sb.committed + 1),
       POKE(committed, 0),                                     // less than the frontier needs
       POKE(tcommitted, sb.tcommitted + artpp::line_pool::chunk_bytes),
       POKE(tcommitted, 0),
       POKE(frontier, sb.committed / artpp::line_pool::line_bytes + 1),  // past the committed range
       POKE(frontier, ~std::uint64_t(0)),
       POKE(tfrontier, sb.tcommitted / artpp::line_pool::term_unit + 1),
       POKE(tfrontier, ~std::uint64_t(0) / 16),
       POKE(huge, sb.frontier),                                // a free block at or past the frontier
       POKE(thuge, sb.tfrontier + 5),
       POKE(uroot, artpp::line_pool::null_root),               // count != 0 needs a root
       POKE(uroot, (sb.frontier << 4) | 4),                    // a node_full at the frontier
       POKE(uroot, (sb.tfrontier << 4) | 0),                   // a leaf at the terminal frontier
       POKE(uroot, 0x1000000000ULL | sb.uroot),                // not a 32-bit handle
       POKE(uroot, (sb.uroot & ~std::uint64_t(0xF)) | 0xD),    // a tag no node kind has
       POKE(free_heads[0], 7),                                 // there is no list of 0-line blocks
       POKE(free_heads[1], sb.frontier),
       POKE(free_heads[2], sb.frontier - 1),                   // 2 lines do not fit before the frontier
       POKE(free_heads[artpp::line_pool::FREE_MAX], sb.frontier + 100),
       POKE(tfree_heads[0], 1),
       POKE(tfree_heads[5], sb.tfrontier - 4),
       POKE(tfree_heads[artpp::line_pool::TFREE_MAX], 0xFFFFFFF0u),
   };
#undef POKE
   for (const poke& p : pokes)
   {
      char original[8];
      CHECK(::pread(fd, original, p.size, off_t(p.off)) == ssize_t(p.size));
      CHECK(::pwrite(fd, &p.bad, p.size, off_t(p.off)) == ssize_t(p.size));  // little-endian hosts
      const std::string what = std::string(p.field) + " = " + std::to_string(p.bad);
      if (p.incompatible)
         EXPECT_REFUSED(artpp::store_incompatible, s, what.c_str());
      else
         EXPECT_REFUSED(artpp::store_corrupt, s, what.c_str());
      CHECK(::pwrite(fd, original, p.size, off_t(p.off)) == ssize_t(p.size));
   }
   ::close(fd);
   CHECK(store_is_intact(s, N));  // every refusal left the image alone
}
static int open_descriptors()
{
   int n = 0;
   for (int fd = 0; fd < 1024; ++fd)
      if (::fcntl(fd, F_GETFD) != -1) ++n;
   return n;
}
// A constructor that throws has no destructor to clean up after it.
static void refused_opens_release_their_descriptors()
{
   temp_store s;
   build_store(s, 100);
   CHECK(::truncate(s.file("terms").c_str(), s.size("terms") / 2) == 0);
   const int before = open_descriptors();
   for (int i = 0; i < 20; ++i)
   {
      try
      {
         artpp::line_pool pool(s.c_str());
         CHECK(!"opened a truncated store");
      }
      catch (const artpp::store_error&)
      {
      }
   }
   CHECK(open_descriptors() == before);
   temp_store none;  // a path that cannot be a store directory: an OS error, with its errno
   spit(none.file("plain"), {'x'});
   for (int i = 0; i < 20; ++i)
   {
      try
      {
         artpp::line_pool pool(none.file("plain").c_str());
         CHECK(!"opened a store inside a regular file");
      }
      catch (const std::system_error& e)
      {
         CHECK(e.code().value() == ENOTDIR || e.code().value() == EEXIST);
         CHECK(std::string(e.what()).find(none.dir) != std::string::npos);
      }
   }
   CHECK(open_descriptors() == before);
}

int main()
{
   struct test_case
   {
      const char* name;
      void (*fn)();
   };
   const test_case cases[] = {
       {"never_checkpointed_store_reopens_empty", never_checkpointed_store_reopens_empty},
       {"null_root_everywhere_nothing_was_checkpointed", null_root_everywhere_nothing_was_checkpointed},
       {"destroying_the_map_keeps_the_checkpointed_tree", destroying_the_map_keeps_the_checkpointed_tree},
       {"checkpoint_after_the_map_is_destroyed", checkpoint_after_the_map_is_destroyed},
       {"lifecycle_over_many_reopen_cycles", lifecycle_over_many_reopen_cycles},
       {"set_lifecycle", set_lifecycle},
       {"retention_is_only_for_the_destructor_of_a_file_backed_map",
        retention_is_only_for_the_destructor_of_a_file_backed_map},
       {"truncated_terms_file_is_refused", truncated_terms_file_is_refused},
       {"truncated_nodes_file_is_refused", truncated_nodes_file_is_refused},
       {"foreign_files_are_never_overwritten", foreign_files_are_never_overwritten},
       {"files_of_different_checkpoints_are_refused", files_of_different_checkpoints_are_refused},
       {"corrupted_superblock_fields_are_refused", corrupted_superblock_fields_are_refused},
       {"refused_opens_release_their_descriptors", refused_opens_release_their_descriptors},
   };
   for (const test_case& c : cases)
   {
      const int before = g_fail;
      try
      {
         c.fn();
      }
      catch (const std::exception& e)
      {
         std::printf("FAIL %s: unexpected exception: %s\n", c.name, e.what());
         ++g_fail;
      }
      if (g_fail != before) std::printf("     in case %s\n", c.name);
   }
   if (g_fail == 0)
      std::printf("artpp_pool_safety: ALL PASS (%zu cases)\n", sizeof(cases) / sizeof(cases[0]));
   else
      std::printf("artpp_pool_safety: %d FAILURES\n", g_fail);
   return g_fail ? 1 : 0;
}

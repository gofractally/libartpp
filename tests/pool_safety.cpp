// pool_safety — the lifecycle and validation contract of a file-backed artpp::line_pool:
// what a reopen restores, what it refuses, and that nothing is lost or overwritten silently.
// Each case is the regression test of one finding of the pre-release audit (issue numbers
// are those of github.com/gofractally/libartpp). Every case works in its own mkdtemp
// directory and removes it again, also when the case fails or throws.
#include "artpp/map.hpp"
#include "artpp/pool.hpp"

#include <algorithm>
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
#include <type_traits>
#include <vector>

#include <dirent.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

static int g_fail = 0;
#define CHECK(...)                                                                                    \
   do {                                                                                               \
      if (!(__VA_ARGS__)) { std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #__VA_ARGS__); ++g_fail; } \
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
// Every node kind goes through the image the same way: the modes that use buckets, the
// density ladder and wide stems, with values in leaves and values in the handle.
template <class M>
static void round_trip_mode(const char* name)
{
   using MA = typename M::allocator_type;
   temp_store                           s;
   std::map<std::string, std::uint32_t> ref;
   std::mt19937_64                      rng(7);
   for (int cycle = 0; cycle < 3; ++cycle)
   {
      artpp::line_pool pool(s.c_str());
      M                m{MA{&pool}, artpp::attach};
      if (m.size() != ref.size()) { std::printf("FAIL %s: size after reopen %d\n", name, cycle); ++g_fail; }
      for (int i = 0; i < 4000; ++i)
      {
         std::string k(1 + rng() % 12, 'a');
         for (auto& c : k) c = char('a' + rng() % 4);
         if (rng() % 4 == 0) { m.erase(k); ref.erase(k); }
         else { const auto v = std::uint32_t(rng() & 0xFFFF); m.insert_or_assign(k, typename M::mapped_type(v)); ref[k] = v; }
      }
      m.checkpoint();
   }
   artpp::line_pool pool(s.c_str());
   M                m{MA{&pool}, artpp::attach};
   bool             same = m.size() == ref.size();
   auto             ri   = ref.begin();
   for (auto it = m.begin(); same && it != m.end(); ++it, ++ri)
      same = ri != ref.end() && std::string(it.key()) == ri->first && std::uint32_t(it.value()) == ri->second;
   if (!same) { std::printf("FAIL %s: content after the last reopen\n", name); ++g_fail; }
}
static void every_mode_round_trips()
{
   using namespace artpp;
   using SV = std::string_view;
   using A8 = pool_alloc<std::uint64_t>;
   using A2 = pool_alloc<std::uint16_t>;
   round_trip_mode<map<SV, std::uint64_t, mode::buckets, A8>>("buckets");
   round_trip_mode<map<SV, std::uint64_t, mode::adaptive, A8>>("adaptive");
   round_trip_mode<map<SV, std::uint64_t, mode::dense_tiers | mode::ladder_c8, A8>>("dense_tiers|ladder_c8");
   round_trip_mode<map<SV, std::uint64_t, mode::wide_stems, A8>>("wide_stems");
   round_trip_mode<map<SV, std::uint64_t, mode::adaptive | mode::dense_tiers, A8>>("adaptive|dense_tiers");
   round_trip_mode<map<SV, std::uint16_t, mode::none, A2>>("inline values");
   round_trip_mode<map<SV, std::uint16_t, mode::buckets, A2>>("inline values, buckets");
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
       POKE(header_bytes, 0),
       POKE(header_bytes, sb.header_bytes * 2),                 // the data is not where it says
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
       POKE(dirty, 2),                                         // a flag, not a counter
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

// ── #10: the image was modified after its last checkpoint ───────────────────────────────
template <class F>
static bool opens(const temp_store& s, F&& then)
{
   artpp::line_pool pool(s.c_str());
   then(pool);
   return true;
}
template <class E>
static bool open_throws(const temp_store& s)
{
   try
   {
      artpp::line_pool pool(s.c_str());
   }
   catch (const E& e)
   {
      CHECK(std::string(e.what()).find(s.dir) != std::string::npos);
      return true;
   }
   catch (const std::exception& e)
   {
      std::printf("     wrong exception: %s\n", e.what());
   }
   return false;
}
// Scenario 6: checkpoint, erase and insert, close. The superblock described the tree of the
// checkpoint, the pages held a different one: half the keys were missing, no error.
static void mutation_after_the_last_checkpoint_is_detected()
{
   temp_store s;
   {
      artpp::line_pool pool(s.c_str());
      Map              m{A{&pool}};
      for (int i = 0; i < 2000; ++i) m.insert(key(i), std::uint64_t(i) + 1);
      pool.checkpoint(m.root_handle(), m.size());
      CHECK(!pool.dirty());
      for (int i = 0; i < 2000; i += 2) m.erase(key(i));
      for (int i = 5000; i < 9000; ++i) m.insert(key(i), std::uint64_t(i) + 1);
      CHECK(pool.dirty());
      m.detach();
   }
   CHECK(open_throws<artpp::store_dirty>(s));
   CHECK(open_throws<artpp::store_dirty>(s));  // and it stays refused: nothing repairs it
}
// Every way of changing the tree counts, one at a time; reading does not.
static void each_kind_of_mutation_marks_the_image()
{
   enum what { read_only, insert_one, erase_one, erase_then_insert, assign_only, clear_all, move_in, checkpointed_again };
   for (int w = read_only; w <= checkpointed_again; ++w)
   {
      temp_store s;
      build_store(s, 300);
      {
         artpp::line_pool pool(s.c_str());
         Map              m{A{&pool}, artpp::attach};
         CHECK(!pool.dirty());
         CHECK(missing(m, 0, 300) == 0);  // lookups and a full iteration leave it clean
         size_t n = 0;
         for (auto it = m.begin(); it != m.end(); ++it) ++n;
         CHECK(n == 300 && !pool.dirty());
         switch (w)
         {
            case read_only: break;
            case insert_one: m.insert(key(1000), 1); break;
            case erase_one: m.erase(key(7)); break;
            case erase_then_insert:  // the freed leaf is reused: the carving state is as before
               m.erase(key(7));
               m.insert(key(8000), 1);
               break;
            case assign_only: m.insert_or_assign(key(7), 12345); break;  // in place: see below
            case clear_all: m.clear(); break;
            case move_in:
            {
               artpp::line_pool other;
               Map              o{A{&other}};
               o.insert("from another pool", 1);
               m = std::move(o);  // adopts the other pool's image wholesale
               break;
            }
            case checkpointed_again:
               m.insert(key(1000), 1);
               CHECK(pool.dirty());
               m.checkpoint();
               break;
         }
         const bool expect_dirty = !(w == read_only || w == assign_only || w == checkpointed_again);
         CHECK(pool.dirty() == expect_dirty);
      }
      if (w == read_only)
         CHECK(store_is_intact(s, 300));
      else if (w == assign_only)
      {
         // Overwriting a value in place changes no structure: the image is still the tree of
         // the checkpoint, with the new value (documented: such a write is not undone).
         artpp::line_pool pool(s.c_str());
         Map              m{A{&pool}, artpp::attach};
         std::uint64_t    v = 0;
         CHECK(m.size() == 300 && m.find(key(7), v) && v == 12345);
      }
      else if (w == checkpointed_again)
      {
         artpp::line_pool pool(s.c_str());
         Map              m{A{&pool}, artpp::attach};
         CHECK(m.size() == 301 && m.contains(key(1000)) && missing(m, 0, 300) == 0);
      }
      else
      {
         if (!open_throws<artpp::store_dirty>(s)) { std::printf("FAIL mutation kind %d was not detected\n", w); ++g_fail; }
      }
   }
}
// The flag is on disk from the first mutation on, not written at close: a process that dies
// after mutating (no destructor runs) leaves a store that is refused, not one that is read.
static void a_writer_that_dies_leaves_the_store_marked()
{
   temp_store s;
   build_store(s, 300);
   std::fflush(nullptr);
   const pid_t pid = ::fork();
   CHECK(pid >= 0);
   if (pid == 0)
   {
      {
         auto* pool = new artpp::line_pool(s.c_str());
         auto* m    = new Map{A{pool}, artpp::attach};
         m->erase(key(3));
      }
      ::_exit(0);  // no destructors, no checkpoint
   }
   int status = -1;
   CHECK(::waitpid(pid, &status, 0) == pid && WIFEXITED(status) && WEXITSTATUS(status) == 0);
   CHECK(open_throws<artpp::store_dirty>(s));
}
// checkpoint() takes a sync level. The default, none, forces nothing to the file: the image
// is recorded through the shared mapping and is what any later open sees, including after
// the writing process is gone without closing anything. flush writes it out and is the same
// image. What flush adds — surviving a machine crash, reporting write errors — cannot be
// observed from a test.
static void each_sync_level_records_the_same_image()
{
   for (const artpp::sync_level level : {artpp::sync_level::none, artpp::sync_level::flush})
   {
      temp_store s;
      {  // through the map, across a clean close
         artpp::line_pool pool(s.c_str());
         Map              m{A{&pool}, artpp::attach};
         for (int i = 0; i < 2000; ++i) m.insert(key(i), std::uint64_t(i) + 1);
         m.checkpoint(level);
         CHECK(pool.count() == 2000);
      }
      {
         artpp::line_pool pool(s.c_str());
         Map              m{A{&pool}, artpp::attach};
         CHECK(pool.restored() && m.size() == 2000 && missing(m, 0, 2000) == 0);
         for (int i = 2000; i < 2500; ++i) m.insert(key(i), std::uint64_t(i) + 1);
         pool.checkpoint(m.root_handle(), m.size(), level);  // the low-level call takes it too
      }
      std::fflush(nullptr);
      const pid_t pid = ::fork();  // a writer that checkpoints and is gone without closing
      CHECK(pid >= 0);
      if (pid == 0)
      {
         auto* pool = new artpp::line_pool(s.c_str());
         auto* m    = new Map{A{pool}, artpp::attach};
         for (int i = 2500; i < 3000; ++i) m->insert(key(i), std::uint64_t(i) + 1);
         m->checkpoint(level);
         ::_exit(m->size() == 3000 ? 0 : 1);  // no destructors, no close
      }
      int status = -1;
      CHECK(::waitpid(pid, &status, 0) == pid && WIFEXITED(status) && WEXITSTATUS(status) == 0);
      artpp::line_pool pool(s.c_str());
      Map              m{A{&pool}, artpp::attach};
      CHECK(pool.restored() && m.size() == 3000 && missing(m, 0, 3000) == 0);
   }
   {  // the default is none
      temp_store       s;
      artpp::line_pool pool(s.c_str());
      Map              m{A{&pool}, artpp::attach};
      m.insert(key(1), 2);
      m.checkpoint();
      m.checkpoint(artpp::sync_level::none);
      pool.checkpoint(m.root_handle(), m.size());
      CHECK(pool.count() == 1);
   }
}
// A store that was never checkpointed holds nothing that could be lost, so a mutated one is
// not refused: it reopens as the empty store it still is (#9), and gives its space back.
static void a_never_checkpointed_store_is_reset_not_refused()
{
   temp_store s;
   off_t      fresh_nodes = 0, fresh_terms = 0;
   {
      artpp::line_pool pool(s.c_str());
      fresh_nodes = s.size("nodes");
      fresh_terms = s.size("terms");
      Map m{A{&pool}};
      for (int i = 0; i < 2000; ++i) m.insert(key(i), std::uint64_t(i) + 1);
      CHECK(pool.dirty());
      CHECK(s.size("nodes") > fresh_nodes && s.size("terms") > fresh_terms);
   }  // closed without a checkpoint
   for (int again = 0; again < 2; ++again)
   {
      artpp::line_pool pool(s.c_str());
      CHECK(!pool.restored() && pool.count() == 0 && !pool.dirty());
      CHECK(pool.used_lines() == 0 && pool.used_term_units() == 0);
      CHECK(s.size("nodes") == fresh_nodes && s.size("terms") == fresh_terms);
      Map m{A{&pool}, artpp::attach};
      CHECK(m.empty() && !m.contains(key(1)));
      for (int i = 0; i < 50; ++i) m.insert(key(i), std::uint64_t(i) + 1);  // dirty again, no checkpoint
   }
   build_store(s, 100);  // and now it is checkpointed like any other
   CHECK(store_is_intact(s, 100));
}
// A change that allocates and frees nothing (a set's short key goes into a slot of an
// existing node) is invisible to the pool, but it changes the count: the map reports the
// tree it is letting go of, and a tree that is not the checkpointed one marks the image.
static void a_change_that_allocates_nothing_is_detected_on_release()
{
   using SA  = artpp::pool_alloc<std::string_view>;
   using Set = artpp::set<std::string_view, artpp::mode::none, SA>;
   for (int how = 0; how < 3; ++how)  // 0: destructor, 1: detach(), 2: moved first
   {
      temp_store s;
      {
         artpp::line_pool pool(s.c_str());
         Set              t{SA{&pool}, artpp::attach};
         t.insert("a");
         t.insert("b");
         t.checkpoint();
      }
      {
         artpp::line_pool pool(s.c_str());
         Set              t{SA{&pool}, artpp::attach};
         CHECK(t.size() == 2 && !pool.dirty());
         const size_t lines = pool.used_lines(), terms = pool.used_term_units();
         t.insert("c");
         CHECK(t.size() == 3);
         // the premise of this case: the pool saw nothing
         CHECK(pool.used_lines() == lines && pool.used_term_units() == terms && !pool.dirty());
         if (how == 1) t.detach();
         if (how == 2)
         {
            Set u{std::move(t)};
            CHECK(u.size() == 3 && !pool.dirty());
         }
         if (how != 0) CHECK(pool.dirty());
      }
      CHECK(open_throws<artpp::store_dirty>(s));
   }
}

// ── #10: one store, one pool at a time ──────────────────────────────────────────────────
// Two pools over one store each keep their own frontier and free lists and overwrite each
// other's nodes. The second opener is refused, in this process or another, until the first
// has closed; being refused disturbs neither the store nor the pool that holds it.
static void a_second_opener_is_refused_while_the_first_is_open()
{
   temp_store s;
   build_store(s, 200);
   {
      artpp::line_pool first(s.c_str());
      Map              m{A{&first}, artpp::attach};
      CHECK(open_throws<artpp::store_locked>(s));  // same process
      std::fflush(nullptr);
      const pid_t pid = ::fork();  // another process
      CHECK(pid >= 0);
      if (pid == 0)
      {
         int rc = 1;
         try
         {
            new artpp::line_pool(s.c_str());
            rc = 2;  // opened: wrong
         }
         catch (const artpp::store_locked&)
         {
            rc = 0;
         }
         catch (...)
         {
            rc = 3;
         }
         ::_exit(rc);
      }
      int status = -1;
      CHECK(::waitpid(pid, &status, 0) == pid && WIFEXITED(status));
      CHECK(WEXITSTATUS(status) == 0);
      for (int i = 200; i < 400; ++i) m.insert(key(i), std::uint64_t(i) + 1);  // the holder carries on
      m.checkpoint();
   }
   CHECK(store_is_intact(s, 400));  // closed: the lock is gone with it
   {  // a store being created is locked from the start as well
      temp_store       fresh;
      artpp::line_pool first(fresh.c_str());
      CHECK(open_throws<artpp::store_locked>(fresh));
   }
   {  // and a refused open (for any reason) does not keep the lock
      temp_store bad;
      build_store(bad, 10);
      CHECK(::truncate(bad.file("terms").c_str(), 100) == 0);
      CHECK(open_throws<artpp::store_corrupt>(bad));
      CHECK(open_throws<artpp::store_corrupt>(bad));  // not store_locked
   }
}

// ── #10: the store remembers which map wrote it ─────────────────────────────────────────
// The bytes of a tree only mean something to the instantiation that wrote them. Attaching
// another one (another mapped type, mode, key encoding, or a set) used to "work" and then
// read nodes with the wrong layout.
struct reversed_u64  // a key type with its own codec, which names its encoding
{
   std::uint64_t v;
};
namespace artpp
{
   template <>
   struct key_codec<reversed_u64>
   {
      static constexpr std::uint64_t image_id = 0x7265767536340001ULL;
      struct scratch
      {
         char b[8];
      };
      static std::string_view encode(const reversed_u64& k, scratch& s) noexcept
      {
         std::memcpy(s.b, &k.v, 8);
         return std::string_view(s.b, 8);
      }
      static reversed_u64 decode(std::string_view b) noexcept
      {
         reversed_u64 k;
         std::memcpy(&k.v, b.data(), 8);
         return k;
      }
   };
}  // namespace artpp
template <class M>
static bool attach_refused(const temp_store& s, bool low_level)
{
   using MA = typename M::allocator_type;
   artpp::line_pool pool(s.c_str());
   bool             refused = false;
   try
   {
      if (low_level)
         M m{MA{&pool}, artpp::attach, pool.root(), size_t(pool.count())};
      else
         M m{MA{&pool}, artpp::attach};
   }
   catch (const artpp::store_incompatible& e)
   {
      refused = std::string(e.what()).find(s.dir) != std::string::npos;
   }
   CHECK(!pool.dirty());  // a refused attach changes nothing
   return refused;
}
static void attaching_another_instantiation_is_refused()
{
   using namespace artpp;
   using SV = std::string_view;
   using U64 = std::uint64_t;
   temp_store s;
   build_store(s, 500);  // written by map<string_view, uint64_t, mode::none>
   for (int low_level = 0; low_level < 2; ++low_level)
   {
      const bool ll = low_level != 0;
      CHECK(attach_refused<map<SV, std::uint32_t, mode::none, pool_alloc<std::uint32_t>>>(s, ll));  // sizeof(T)
      struct alignas(16) wide { U64 a; };
      CHECK(attach_refused<map<SV, wide, mode::none, pool_alloc<wide>>>(s, ll));                    // alignof(T)
      CHECK(attach_refused<map<SV, U64, mode::buckets, A>>(s, ll));                                 // mode
      CHECK(attach_refused<map<SV, U64, mode::adaptive, A>>(s, ll));
      CHECK(attach_refused<map<SV, U64, mode::dense_tiers, A>>(s, ll));
      CHECK(attach_refused<map<SV, U64, mode::wide_stems, A>>(s, ll));
      CHECK(attach_refused<compact_map<SV, U64, 1000, A>>(s, ll));                // a hint that turns the ladder on
      CHECK(attach_refused<map<U64, U64, mode::none, A>>(s, ll));                                   // key encoding
      CHECK(attach_refused<map<std::int64_t, U64, mode::none, A>>(s, ll));
      CHECK(attach_refused<map<reversed_u64, U64, mode::none, A>>(s, ll));
      CHECK(attach_refused<set<SV, mode::none, pool_alloc<SV>>>(s, ll));                            // no value at all
      CHECK(store_is_intact(s, 500));
   }
   {  // the same bytes under another key TYPE are the same image: std::string keys encode alike
      line_pool pool(s.c_str());
      map<std::string, U64, mode::none, A> m{A{&pool}, attach};
      CHECK(m.size() == 500 && m.contains(key(17)));
      map<SV, U64, mode::flat_full, A> f{A{&pool}, attach};  // flat_full only restates the default
      CHECK(f.size() == 500);
   }
   {  // and each of the others refuses what it did not write, e.g. a set's image as a map
      temp_store t;
      {
         line_pool                           pool(t.c_str());
         set<SV, mode::none, pool_alloc<SV>> st{pool_alloc<SV>{&pool}, attach};
         st.insert("one");
         st.checkpoint();
      }
      CHECK(attach_refused<Map>(t, false));
      CHECK(attach_refused<map<reversed_u64, U64, mode::none, A>>(t, false));
      line_pool                           pool(t.c_str());
      set<SV, mode::none, pool_alloc<SV>> st{pool_alloc<SV>{&pool}, attach};
      CHECK(st.size() == 1 && st.contains("one"));
   }
}
// The low-level calls carry the stamp as soon as a map has attached: pool.checkpoint(root,
// count) writes the stamp of the map that was bound in this session.
static void the_stamp_follows_the_low_level_calls()
{
   using namespace artpp;
   using Other = map<std::string_view, std::uint64_t, mode::buckets, A>;
   temp_store s;
   {
      line_pool pool(s.c_str());
      Map       m{A{&pool}, attach, pool.root(), size_t(pool.count())};  // binds: this store is a Map's
      for (int i = 0; i < 100; ++i) m.insert(key(i), std::uint64_t(i) + 1);
      pool.checkpoint(m.root_handle(), m.size());
   }
   CHECK(attach_refused<Other>(s, true));
   CHECK(attach_refused<Other>(s, false));
   {
      line_pool pool(s.c_str());
      Map       m{A{&pool}, attach, pool.root(), size_t(pool.count())};
      m.insert(key(100), 101);
      pool.checkpoint(m.root_handle(), m.size());
      bool refused = false;  // one store, one instantiation: a second kind cannot bind either
      try
      {
         Other o{A{&pool}, attach};
      }
      catch (const store_incompatible&)
      {
         refused = true;
      }
      CHECK(refused);
   }
   CHECK(attach_refused<Other>(s, true));
   CHECK(store_is_intact(s, 101));
}

// ── #17: nothing about the layout may assume 4 KB or 16 KB pages ────────────────────────
// The header reservation is the file offset at which the data of each file is mapped, so
// it has to be a multiple of the page size of the system that maps it. It was a constant
// 32768: on a kernel with 64 KB pages every mmap of the data failed with EINVAL.
static void header_reservation_is_page_size_driven()
{
   using LP = artpp::line_pool;
   const size_t sb = sizeof(LP::superblock);
   for (size_t page : {size_t(4096), size_t(8192), size_t(16384), size_t(65536), size_t(262144),
                       size_t(1) << 20, size_t(2) << 20, LP::chunk_bytes})
   {
      const size_t h = LP::header_bytes_for(page);
      CHECK(h % page == 0);                    // mappable at that page size
      CHECK(h >= sb);                          // holds the superblock
      CHECK(h >= LP::min_header_bytes);        // and never less than the portable minimum
      CHECK(h - page < std::max(sb, LP::min_header_bytes));  // the smallest such multiple
      CHECK(LP::header_fits_pages(h, page));
   }
   // One layout for every common page size: a store made with 4 KB pages maps with 64 KB ones.
   CHECK(LP::header_bytes_for(4096) == LP::header_bytes_for(16384));
   CHECK(LP::header_bytes_for(4096) == LP::header_bytes_for(65536));
   CHECK(LP::header_fits_pages(LP::header_bytes_for(4096), 65536));
   CHECK(LP::header_fits_pages(LP::header_bytes_for(65536), 4096));
   CHECK(!LP::header_fits_pages(LP::header_bytes_for(4096), size_t(1) << 20));  // not with 1 MB pages
   CHECK(LP::header_fits_pages(LP::header_bytes_for(size_t(1) << 20), 4096));   // the other way is fine
   CHECK(!LP::header_fits_pages(32768, 65536));          // the old constant, on the kernel of the issue
   CHECK(!LP::header_fits_pages(65536, 0));
   CHECK(!LP::header_fits_pages(65536, LP::chunk_bytes * 2));  // the commit unit must be whole pages too
   CHECK(!LP::header_fits_pages(4096, 4096));            // too small for a superblock
   CHECK(LP::system_page_size() >= 4096 && LP::header_fits_pages(LP::header_bytes_for(LP::system_page_size()), LP::system_page_size()));
}
// The pool can be told to lay a store out for larger pages than the system has (any
// multiple is mappable here), which runs the whole open / grow / checkpoint / reopen path
// with the offsets of a 64 KB-page or a 1 MB-page system.
static void stores_laid_out_for_larger_pages()
{
   using LP = artpp::line_pool;
   const size_t sys = LP::system_page_size();
   for (size_t page : {size_t(0), size_t(65536), size_t(1) << 20})
   {
      if (page != 0 && page % sys != 0) continue;
      const size_t eff = page ? page : sys;
      temp_store   s;
      {
         LP pool(s.c_str(), LP::max_bytes, page);
         CHECK(size_t(s.size("nodes")) == LP::header_bytes_for(eff));
         CHECK(size_t(s.size("terms")) == LP::header_bytes_for(eff));
         Map m{A{&pool}, artpp::attach};
         for (int i = 0; i < 120000; ++i) m.insert(key(i), std::uint64_t(i) + 1);  // several chunks
         CHECK(pool.term_committed() > LP::chunk_bytes);  // grown past the first mapping
         m.checkpoint();
         CHECK(size_t(s.size("nodes")) == LP::header_bytes_for(eff) + pool.committed());
         CHECK(size_t(s.size("terms")) == LP::header_bytes_for(eff) + pool.term_committed());
         CHECK(size_t(s.size("nodes")) % eff == 0 && size_t(s.size("terms")) % eff == 0);
         CHECK(pool.file_extent() == size_t(s.size("nodes")) + size_t(s.size("terms")));
      }
      {
         LP  pool(s.c_str(), LP::max_bytes, page);  // the same system reopens it
         Map m{A{&pool}, artpp::attach};
         CHECK(m.size() == 120000 && missing(m, 0, 120000) == 0);
      }
      CHECK(store_is_intact(s, 120000));  // and so does this one, with its smaller pages
   }
   {  // a store laid out for smaller pages than the opener's is refused, not mapped askew
      temp_store s;
      build_store(s, 100);
      bool refused = false;
      try
      {
         LP pool(s.c_str(), LP::max_bytes, size_t(1) << 20);
      }
      catch (const artpp::store_incompatible& e)
      {
         refused = std::string(e.what()).find("page") != std::string::npos;
      }
      CHECK(refused);
      CHECK(store_is_intact(s, 100));
   }
   bool threw = false;  // a page size the system cannot map with is a caller error
   try
   {
      temp_store s;
      LP         pool(s.c_str(), LP::max_bytes, sys + 512);
   }
   catch (const std::invalid_argument&)
   {
      threw = true;
   }
   CHECK(threw);
}

// ── limits and failures of the allocation path (audit follow-up, #1) ────────────────────
// Running out is an allocation failure like any other: a std::bad_alloc (so the tree's
// strong guarantee applies), of a type that says which limit, with the pool left usable.
static void exhaustion_throws_cleanly_and_leaves_the_pool_usable()
{
   using LP = artpp::line_pool;
   {  // a small pool: every line of its one chunk, then one more
      LP                 pool(LP::chunk_bytes);
      const size_t       lines = LP::chunk_bytes / LP::line_bytes;
      std::vector<void*> got;
      for (size_t i = 0; i < lines; ++i) got.push_back(pool.alloc_lines(1));
      CHECK(pool.used_lines() == lines);
      bool exhausted = false;
      try
      {
         pool.alloc_lines(1);
      }
      catch (const artpp::pool_exhausted& e)
      {
         const std::bad_alloc& as_bad_alloc = e;
         exhausted = std::strstr(as_bad_alloc.what(), "line_pool") != nullptr;
      }
      CHECK(exhausted);
      CHECK(pool.used_lines() == lines);  // nothing moved
      pool.free_lines(got.back(), 1);
      CHECK(pool.alloc_lines(1) == got.back());  // and what is freed can be had again
      const size_t units = pool.term_committed();  // the terminal region of the same pool
      (void)units;
      size_t n = 0;
      bool   texhausted = false;
      try
      {
         for (;; ++n) pool.alloc_term(4096);
      }
      catch (const artpp::pool_exhausted&)
      {
         texhausted = true;
      }
      CHECK(texhausted && n == LP::chunk_bytes / (4096 * LP::term_unit));
   }
   {  // a request so large that the byte count overflows must fail, not wrap to "fits"
      LP pool(LP::chunk_bytes);
      for (size_t n : {size_t(1) << 57, (size_t(1) << 57) + 3, ~size_t(0), ~size_t(0) / 128})
      {
         bool threw = false;
         try
         {
            pool.alloc_lines(n);
         }
         catch (const std::bad_alloc&)
         {
            threw = true;
         }
         CHECK(threw && pool.used_lines() == 0);
      }
      for (size_t n : {size_t(1) << 60, ~size_t(0), ~size_t(0) / 16})
      {
         bool threw = false;
         try
         {
            pool.alloc_term(n);
         }
         catch (const std::bad_alloc&)
         {
            threw = true;
         }
         CHECK(threw && pool.used_term_units() == 0);
      }
      artpp::pool_alloc<std::uint64_t> a{&pool};  // and through the allocator interface
      bool                             threw = false;
      try
      {
         (void)a.allocate(~size_t(0) / 4);  // * sizeof(uint64_t) overflows
      }
      catch (const std::bad_alloc&)
      {
         threw = true;
      }
      CHECK(threw && pool.used_lines() == 0);
      threw = false;
      try
      {
         (void)a.artpp_alloc_term(~size_t(0) - 3);  // rounding up to 16 overflows
      }
      catch (const std::bad_alloc&)
      {
         threw = true;
      }
      CHECK(threw && pool.used_term_units() == 0);
   }
}
// The limits of the handle itself: 2^28 lines of nodes, 2^28 units (4 GB) of terminals. The
// blocks are never touched, so this needs address space, not memory. A system that refuses
// to commit that much (strict overcommit) fails earlier, with pool_commit_failed — just as
// cleanly, and then there is no boundary to look at.
static void the_handle_limits_are_exact()
{
   using LP = artpp::line_pool;
   LP           pool;
   const size_t max_lines = LP::max_bytes / LP::line_bytes, block = size_t(1) << 20;
   CHECK(max_lines == (size_t(1) << 28));
   std::byte* last  = nullptr;
   size_t     taken = 0;
   bool       os_refused = false, exhausted = false;
   try
   {
      for (; taken < max_lines; taken += block) last = static_cast<std::byte*>(pool.alloc_lines(block));
      pool.alloc_lines(1);  // line 2^28 does not exist
   }
   catch (const artpp::pool_exhausted&)
   {
      exhausted = true;
   }
   catch (const artpp::pool_commit_failed& e)
   {
      os_refused = true;
      std::printf("     note: the system refused to commit the node region at %zu lines (errno %d)\n", taken, e.code());
   }
   CHECK(exhausted || os_refused);
   if (exhausted)
   {
      CHECK(taken == max_lines && pool.used_lines() == max_lines);
      // the last line is addressable by a handle, and decodes to itself
      const size_t off = size_t(last - pool.base()) + (block - 1) * LP::line_bytes;
      CHECK(off == LP::max_bytes - LP::line_bytes);
      const auto h = artpp::detail::line_ptr::from_off(off, artpp::detail::K::node_full);
      CHECK(((h.raw() & ~std::uint64_t(0xF)) << 3) == off && h.tag() == artpp::detail::K::node_full);
      pool.free_lines(last, block);
      CHECK(pool.alloc_lines(block) == last);  // from the giant-block list, at the very end
   }
   const size_t max_units = LP::max_term_bytes / LP::term_unit;
   taken = 0;
   exhausted = os_refused = false;
   try
   {
      for (; taken < max_units; taken += block) last = static_cast<std::byte*>(pool.alloc_term(block));
      pool.alloc_term(1);
   }
   catch (const artpp::pool_exhausted&)
   {
      exhausted = true;
   }
   catch (const artpp::pool_commit_failed&)
   {
      os_refused = true;
   }
   CHECK(exhausted || os_refused);
   if (exhausted)
   {
      CHECK(taken == max_units && pool.used_term_units() == max_units);
      const size_t off = size_t(last - pool.term_base()) + (block - 1) * LP::term_unit;
      CHECK(off == LP::max_term_bytes - LP::term_unit);
      const auto h = artpp::detail::line_ptr::from_toff(off, artpp::detail::K::value_ptr);
      CHECK((h.raw() & ~std::uint64_t(0xF)) == off);
   }
}
// Scenario 7 of the audit's reproducer (which passed, and must keep passing): a map that
// runs its pool dry stays whole, in memory and — for a file-backed pool — on disk.
static void a_map_that_exhausts_its_pool_stays_valid()
{
   temp_store s;
   int        inserted = 0;
   {
      artpp::line_pool pool(s.c_str(), artpp::line_pool::chunk_bytes);  // one chunk per region
      Map              m{A{&pool}, artpp::attach};
      bool             threw = false;
      try
      {
         for (;; ++inserted) m.insert(key(inserted), std::uint64_t(inserted) + 1);
      }
      catch (const artpp::pool_exhausted&)
      {
         threw = true;
      }
      CHECK(threw && inserted > 1000 && m.size() == size_t(inserted));
      CHECK(missing(m, 0, inserted) == 0 && !m.contains(key(inserted)));
      size_t n = 0;
      for (auto it = m.begin(); it != m.end(); ++it) ++n;
      CHECK(n == size_t(inserted));
      for (int i = 0; i < inserted; i += 2) m.erase(key(i));
      for (int i = 0; i < inserted; i += 2) m.insert(key(i), std::uint64_t(i) + 1);  // the freed space suffices
      CHECK(m.size() == size_t(inserted));
      m.checkpoint();
   }
   CHECK(store_is_intact(s, inserted));
}
// The operating system refusing to grow a file is reported with its errno, as a
// std::bad_alloc; the pool and the tree stay usable and the store can still be checkpointed.
static void a_failed_file_growth_reports_errno()
{
   temp_store s;
   struct rlimit old_limit;
   CHECK(::getrlimit(RLIMIT_FSIZE, &old_limit) == 0);
   void (*old_handler)(int) = ::signal(SIGXFSZ, SIG_IGN);  // the kernel also sends this; EFBIG is enough
   int  inserted = 0, err = 0;
   bool failed = false;
   {
      artpp::line_pool pool(s.c_str());
      Map              m{A{&pool}, artpp::attach};
      m.insert(key(0), 1);
      inserted = 1;
      struct rlimit lim = old_limit;
      lim.rlim_cur      = rlim_t(s.size("terms")) + 4096;  // `terms` cannot get its next chunk
      CHECK(::setrlimit(RLIMIT_FSIZE, &lim) == 0);
      try
      {
         for (; inserted < 2000000; ++inserted) m.insert(key(inserted), std::uint64_t(inserted) + 1);
      }
      catch (const artpp::pool_commit_failed& e)
      {
         failed = true;
         err    = e.code();
      }
      catch (const std::bad_alloc&)
      {
      }
      CHECK(::setrlimit(RLIMIT_FSIZE, &old_limit) == 0);
      CHECK(failed);
      CHECK(err == EFBIG);
      CHECK(m.size() == size_t(inserted) && missing(m, 0, inserted) == 0);
      m.insert(key(inserted), std::uint64_t(inserted) + 1);  // with the limit gone it grows again
      ++inserted;
      m.checkpoint();
   }
   ::signal(SIGXFSZ, old_handler);
   CHECK(store_is_intact(s, inserted));
}
// close(): the way to learn that releasing the store failed (a destructor cannot say).
static void close_releases_the_store_and_is_checked()
{
   CHECK(!std::is_copy_constructible_v<artpp::line_pool> && !std::is_move_constructible_v<artpp::line_pool>);
   CHECK(!std::is_copy_assignable_v<artpp::line_pool> && !std::is_move_assignable_v<artpp::line_pool>);
   temp_store s;
   build_store(s, 50);
   artpp::line_pool pool(s.c_str());
   {
      Map m{A{&pool}, artpp::attach};
      m.insert(key(50), 51);
      m.checkpoint();
   }
   pool.close();
   CHECK(store_is_intact(s, 51));  // the lock went with it: another pool can open the store
   pool.close();                   // closing twice is nothing
   bool threw = false;
   try
   {
      pool.checkpoint(artpp::line_pool::null_root, 0);
   }
   catch (const std::logic_error&)
   {
      threw = true;
   }
   CHECK(threw);
   CHECK(store_is_intact(s, 51));
   artpp::line_pool anon;
   anon.close();
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
       {"every_mode_round_trips", every_mode_round_trips},
       {"retention_is_only_for_the_destructor_of_a_file_backed_map",
        retention_is_only_for_the_destructor_of_a_file_backed_map},
       {"truncated_terms_file_is_refused", truncated_terms_file_is_refused},
       {"truncated_nodes_file_is_refused", truncated_nodes_file_is_refused},
       {"foreign_files_are_never_overwritten", foreign_files_are_never_overwritten},
       {"files_of_different_checkpoints_are_refused", files_of_different_checkpoints_are_refused},
       {"corrupted_superblock_fields_are_refused", corrupted_superblock_fields_are_refused},
       {"refused_opens_release_their_descriptors", refused_opens_release_their_descriptors},
       {"mutation_after_the_last_checkpoint_is_detected", mutation_after_the_last_checkpoint_is_detected},
       {"each_kind_of_mutation_marks_the_image", each_kind_of_mutation_marks_the_image},
       {"a_writer_that_dies_leaves_the_store_marked", a_writer_that_dies_leaves_the_store_marked},
       {"each_sync_level_records_the_same_image", each_sync_level_records_the_same_image},
       {"a_never_checkpointed_store_is_reset_not_refused", a_never_checkpointed_store_is_reset_not_refused},
       {"a_change_that_allocates_nothing_is_detected_on_release",
        a_change_that_allocates_nothing_is_detected_on_release},
       {"a_second_opener_is_refused_while_the_first_is_open",
        a_second_opener_is_refused_while_the_first_is_open},
       {"attaching_another_instantiation_is_refused", attaching_another_instantiation_is_refused},
       {"the_stamp_follows_the_low_level_calls", the_stamp_follows_the_low_level_calls},
       {"header_reservation_is_page_size_driven", header_reservation_is_page_size_driven},
       {"stores_laid_out_for_larger_pages", stores_laid_out_for_larger_pages},
       {"exhaustion_throws_cleanly_and_leaves_the_pool_usable", exhaustion_throws_cleanly_and_leaves_the_pool_usable},
       {"the_handle_limits_are_exact", the_handle_limits_are_exact},
       {"a_map_that_exhausts_its_pool_stays_valid", a_map_that_exhausts_its_pool_stays_valid},
       {"a_failed_file_growth_reports_errno", a_failed_file_growth_reports_errno},
       {"close_releases_the_store_and_is_checked", close_releases_the_store_and_is_checked},
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

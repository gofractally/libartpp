// pool_safety — the lifecycle and validation contract of a file-backed artpp::line_pool:
// what a reopen restores, what it refuses, and that nothing is lost or overwritten silently.
// Each case is the regression test of one finding of the pre-release audit (issue numbers
// are those of github.com/gofractally/libartpp). Every case works in its own mkdtemp
// directory and removes it again, also when the case fails or throws.
#include "artpp/map.hpp"
#include "artpp/pool.hpp"

#include <cerrno>
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

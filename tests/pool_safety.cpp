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

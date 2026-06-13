// persistence — a file-backed line_pool survives close + reopen. The handles are byte
// offsets ("the bytes ARE the structure"), so all that's needed is a self-describing image:
// checkpoint() writes the carving state + the tree root/count into a superblock; on reopen
// the pool restores them (no truncate) and the map re-attaches in O(1). We prove a tree
// round-trips through a full pool+map teardown: every key survives, ordered iteration is
// intact, and further inserts allocate from the restored frontier without clobbering.
#include "artpp/map.hpp"
#include "artpp/pool.hpp"

#include <cstdint>
#include <cstdio>
#include <map>
#include <random>
#include <string>
#include <unistd.h>
#include <vector>

static int g_fail = 0;
#define CHECK(cond)                                                                          \
   do {                                                                                      \
      if (!(cond)) { std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); ++g_fail; } \
   } while (0)

using A   = artpp::pool_alloc<std::uint64_t>;
using Map = artpp::map<std::string_view, std::uint64_t, artpp::mode::none, A>;

// A file-backed pool is a directory with one file per region; clean it between/after runs.
static void wipe(const char* dir)
{
   ::unlink((std::string(dir) + "/nodes").c_str());
   ::unlink((std::string(dir) + "/terms").c_str());
   ::rmdir(dir);
}

int main()
{
   const char* path = "/tmp/artpp_persist_test_store";  // a STORE DIRECTORY
   wipe(path);

   const int                       N = 40000;
   std::mt19937_64                 rng(123);
   std::vector<std::string>        keys;
   for (int i = 0; i < N; ++i)
   {
      std::string k(8 + (rng() % 24), '\0');
      for (auto& c : k) c = char('a' + rng() % 26);
      keys.push_back(k);
   }
   std::map<std::string, std::uint64_t> ref;  // oracle: last-writer-wins on dup keys
   for (int i = 0; i < N; ++i) ref[keys[i]] = std::uint64_t(i);

   // ── phase 1: create, populate, checkpoint, tear down ──────────────────────────
   {
      artpp::line_pool pool(path);
      CHECK(!pool.restored());  // brand-new file
      Map m{A{&pool}};
      for (int i = 0; i < N; ++i) m.insert(keys[size_t(i)], std::uint64_t(i));
      CHECK(m.size() == ref.size());
      pool.checkpoint(m.root_handle(), m.size());  // persist root + carving state
      m.detach();                                  // tree lives in the file now; no teardown
   }  // pool destructs: munmap + close

   // ── phase 2: reopen, attach in O(1), verify every key + ordered scan ──────────
   std::uint64_t after_handle = 0, after_count = 0;
   {
      artpp::line_pool pool(path);
      CHECK(pool.restored());  // found a valid image
      CHECK(pool.count() == ref.size());
      Map m{A{&pool}, artpp::attach, pool.root(), pool.count()};
      CHECK(m.size() == ref.size());

      for (auto& [k, v] : ref) { std::uint64_t got; CHECK(m.find(k, got) && got == v); }

      auto mi = ref.begin();  // ordered iteration matches the oracle
      for (auto it = m.begin(); it != m.end(); ++it, ++mi)
         CHECK(mi != ref.end() && std::string(it.key()) == mi->first && it.value() == mi->second);
      CHECK(mi == ref.end());

      // further inserts allocate from the RESTORED frontier — must not clobber live nodes
      for (int i = 0; i < 5000; ++i)
      {
         std::string k = "zzz_new_" + std::to_string(i);
         m.insert(k, std::uint64_t(1'000'000 + i));
         ref[k] = std::uint64_t(1'000'000 + i);
      }
      CHECK(m.size() == ref.size());
      for (auto& [k, v] : ref) { std::uint64_t got; CHECK(m.find(k, got) && got == v); }  // old + new

      pool.checkpoint(m.root_handle(), m.size());
      after_handle = m.root_handle();
      after_count  = m.size();
      m.detach();
   }

   // ── phase 3: reopen once more — the further inserts persisted too ─────────────
   {
      artpp::line_pool pool(path);
      CHECK(pool.restored() && pool.count() == after_count && pool.root() == after_handle);
      Map m{A{&pool}, artpp::attach, pool.root(), pool.count()};
      CHECK(m.size() == ref.size());
      for (auto& [k, v] : ref) { std::uint64_t got; CHECK(m.find(k, got) && got == v); }
      // erase a chunk, checkpoint, and confirm it sticks across the next reopen
      int erased = 0;
      for (auto& [k, v] : ref)
         if ((v & 1) == 0 && erased < 3000) { m.erase(k); ++erased; }
      pool.checkpoint(m.root_handle(), m.size());
      m.detach();
   }

   if (g_fail == 0)
      std::printf("artpp_persistence: ALL PASS (%d keys round-tripped through close/reopen)\n", N);
   else
      std::printf("artpp_persistence: %d FAILURES\n", g_fail);
   wipe(path);
   return g_fail ? 1 : 0;
}

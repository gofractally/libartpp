// line_pool persistence scenarios. usage: pool_tests <scenario>
#include "artpp/map.hpp"
#include "artpp/pool.hpp"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
using A   = artpp::pool_alloc<std::uint64_t>;
using Map = artpp::map<std::string_view, std::uint64_t, artpp::mode::none, A>;
static const char* DIR = "/tmp/artpp-known-bugs-store";
static void wipe() { ::unlink((std::string(DIR) + "/nodes").c_str()); ::unlink((std::string(DIR) + "/terms").c_str()); ::rmdir(DIR); }
static std::string key(int i) { return "key_number_" + std::to_string(i) + "_with_a_long_tail_so_it_needs_a_leaf"; }
static off_t fsize(const char* f) { struct stat st; return ::stat((std::string(DIR) + "/" + f).c_str(), &st) == 0 ? st.st_size : -1; }
static void populate(int N, bool detach)
{
   artpp::line_pool pool(DIR);
   Map m{A{&pool}};
   for (int i = 0; i < N; ++i) m.insert(key(i), std::uint64_t(i) + 1);
   pool.checkpoint(m.root_handle(), m.size());
   if (detach) m.detach();
}
static int verify(int N)
{
   artpp::line_pool pool(DIR);
   std::printf("  reopen: restored=%d root=0x%llx count=%llu\n", int(pool.restored()), (unsigned long long)pool.root(), (unsigned long long)pool.count());
   std::fflush(stdout);
   Map m{A{&pool}, artpp::attach, pool.root(), pool.count()};
   int ok = 0, wrong = 0, missing = 0;
   for (int i = 0; i < N; ++i) { std::uint64_t v = 0; if (!m.find(key(i), v)) ++missing; else if (v != std::uint64_t(i) + 1) ++wrong; else ++ok; }
   std::printf("  size()=%zu: %d keys correct, %d wrong value, %d missing (no error was reported by line_pool)\n", m.size(), ok, wrong, missing);
   std::fflush(stdout);
   m.detach();
   return wrong + missing;
}
int main(int argc, char** argv)
{
   const int sc = argc > 1 ? atoi(argv[1]) : 1;
   wipe();
   if (sc == 1)   // README's open-or-create + attach pattern on a store that exists but was never checkpointed by the user
   {
      { artpp::line_pool pool(DIR); }   // first run: created, then process exits/crashes before any checkpoint
      artpp::line_pool pool(DIR);
      std::printf("S1 reopen of a never-checkpointed store: restored=%d root=0x%llx count=%llu (null handle would be 0xffffffff)\n",
                  int(pool.restored()), (unsigned long long)pool.root(), (unsigned long long)pool.count());
      std::fflush(stdout);
      Map m{A{&pool}, artpp::attach, pool.root(), pool.count()};
      std::printf("  attached: size()=%zu empty()=%d; calling contains()...\n", m.size(), int(m.empty())); std::fflush(stdout);
      std::printf("  contains=%d\n", int(m.contains("x")));
   }
   if (sc == 2)   // checkpoint, then let ~map run WITHOUT detach()
   {
      std::printf("S2 checkpoint() then map destructor without detach():\n");
      populate(2000, false);
      return verify(2000);
   }
   if (sc == 3)   // terms file truncated (partial copy / disk full / interrupted rsync)
   {
      std::printf("S3 terms file truncated to half:\n");
      populate(200000, true);
      const off_t t = fsize("terms");
      ::truncate((std::string(DIR) + "/terms").c_str(), t / 2);
      std::printf("  terms %lld -> %lld bytes\n", (long long)t, (long long)(t / 2));
      return verify(200000);
   }
   if (sc == 4)   // nodes file truncated past the header
   {
      std::printf("S4 nodes file truncated to header + 1 MB:\n");
      populate(200000, true);
      const off_t t = fsize("nodes");
      ::truncate((std::string(DIR) + "/nodes").c_str(), 32768 + (1 << 20));
      std::printf("  nodes %lld -> %d bytes\n", (long long)t, 32768 + (1 << 20));
      return verify(200000);
   }
   if (sc == 5)   // a small unrelated file that happens to be called "nodes" is silently overwritten
   {
      ::mkdir(DIR, 0755);
      int fd = ::open((std::string(DIR) + "/nodes").c_str(), O_CREAT | O_WRONLY, 0644);
      const char txt[] = "these are my precious unrelated notes\n";
      ::write(fd, txt, sizeof txt - 1); ::close(fd);
      { artpp::line_pool pool(DIR); std::printf("S5 foreign 38-byte 'nodes' file: opened without error, restored=%d\n", int(pool.restored())); }
      char buf[16] = {}; fd = ::open((std::string(DIR) + "/nodes").c_str(), O_RDONLY); ::read(fd, buf, 8); ::close(fd);
      std::printf("  file is now %lld bytes, first bytes '%.8s'\n", (long long)fsize("nodes"), buf);
   }
   if (sc == 6)   // mutate after the last checkpoint, close without checkpointing again
   {
      std::printf("S6 mutate after checkpoint, no second checkpoint:\n");
      {
         artpp::line_pool pool(DIR);
         Map m{A{&pool}};
         for (int i = 0; i < 2000; ++i) m.insert(key(i), std::uint64_t(i) + 1);
         pool.checkpoint(m.root_handle(), m.size());
         for (int i = 0; i < 2000; i += 2) m.erase(key(i));                       // frees leaves -> free-list links written into them
         for (int i = 5000; i < 9000; ++i) m.insert(key(i), std::uint64_t(i) + 1);
         m.detach();
      }
      return verify(2000);
   }
   if (sc == 7)   // exhaustion at capacity: tree must stay valid
   {
      artpp::line_pool pool(artpp::line_pool::chunk_bytes);   // 4 MB of node lines
      Map m{A{&pool}};
      int i = 0; bool threw = false;
      try { for (;; ++i) m.insert(key(i), std::uint64_t(i) + 1); } catch (const std::bad_alloc&) { threw = true; }
      int bad = 0; for (int j = 0; j < i; ++j) { std::uint64_t v; if (!m.find(key(j), v) || v != std::uint64_t(j) + 1) ++bad; }
      size_t n = 0; for (auto it = m.begin(); it != m.end(); ++it) ++n;
      std::printf("S7 exhaustion: threw=%d after %d inserts, size()=%zu iterated=%zu bad=%d used_lines=%zu\n", int(threw), i, m.size(), n, bad, pool.used_lines());
      for (int j = 0; j < i; j += 2) m.erase(key(j));
      int re = 0; try { for (int j = 0; j < i; j += 2) { m.insert(key(j), 7); ++re; } } catch (const std::bad_alloc&) {}
      std::printf("  after erase half + reinsert: reinserted %d of %d, size()=%zu\n", re, (i + 1) / 2, m.size());
      return bad;
   }
   wipe();
   return 0;
}

// Recursion depth == tree depth (up to max_key_bytes = 65535 levels): the destructor (free_all_),
// for_each (walk_), lower_bound (descend_lb_), operator== (eq_walk_) and move-assign clone recurse
// once per level. Keys a^1..a^N make an N-level tree. usage: deep_recursion <N> <what> [thread]
#include "artpp/map.hpp"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
static void body(int N, int what)
{
   auto* m = new artpp::map<std::string_view, uint64_t>;
   std::string k;
   for (int i = 1; i <= N; ++i) { k.push_back('a'); m->insert(k, uint64_t(i)); }
   std::printf("built %zu keys (depth %d)\n", m->size(), N); std::fflush(stdout);
   if (what == 1) { size_t n = 0; m->for_each([&](std::string_view, const uint64_t&) { ++n; }); std::printf("for_each ok %zu\n", n); }
   if (what == 2) { auto it = m->lower_bound(k); std::printf("lower_bound ok %d\n", int(it != m->end())); }
   if (what == 3) { uint64_t v = 0; std::printf("find ok %d\n", int(m->find(k, v))); size_t n = 0; for (auto it = m->begin(); it != m->end(); ++it) ++n; std::printf("iterate ok %zu\n", n); }
   std::fflush(stdout);
   delete m;   // what == 0: just the destructor
   std::printf("destructor ok\n");
}
int main(int argc, char** argv)
{
   const int N = argc > 1 ? atoi(argv[1]) : 20000, what = argc > 2 ? atoi(argv[2]) : 0;
   if (argc > 3) { std::thread t(body, N, what); t.join(); }   // default secondary-thread stack (512 KB on macOS)
   else body(N, what);
   return 0;
}

// remove(): the 64-level shrink window (RM_LVN) leaves a dangling NULL child when a cascade
// of dying single-child nodes is longer than the window.
#include "artpp/map.hpp"
#include <cstdio>
#include <cstdlib>
#include <string>
template <class V> V mk(int x) { if constexpr (std::is_same_v<V, std::string>) return std::to_string(x); else return V(x); }
template <class M, class V> int run(const char* name, int N)
{
   M m;
   for (int i = 0; i < N; ++i) m.insert(std::string(size_t(i), 'a') + "b", mk<V>(i));   // a^i b
   m.insert(std::string(size_t(N), 'a'), mk<V>(N));                                     // a^N
   for (int i = N - 1; i >= 0; --i) m.erase(std::string(size_t(i), 'a') + "b");         // deepest first
   std::printf("%s N=%d: after erasing the b-keys size()=%zu contains(a^N)=%d\n", name, N, m.size(), int(m.contains(std::string(size_t(N), 'a'))));
   m.erase(std::string(size_t(N), 'a'));                                                // last key
   std::printf("%s N=%d: size()=%zu empty()=%d; iterating...\n", name, N, m.size(), int(m.empty()));
   std::fflush(stdout);
   size_t n = 0;
   for (auto it = m.begin(); it != m.end(); ++it) ++n;                                  // must be 0 elements
   std::printf("%s N=%d: iteration saw %zu elements (expected 0)\n", name, N, n);
   return n != 0;
}
int main(int argc, char** argv)
{
   const int N = argc > 2 ? atoi(argv[2]) : 200;
   const int w = argc > 1 ? atoi(argv[1]) : 0;
   if (w == 0) return run<artpp::map<std::string_view, std::string>, std::string>("map<string_view,std::string>", N);
   if (w == 1) return run<artpp::map<std::string_view, uint64_t>, uint64_t>("map<string_view,uint64_t>", N);
   return run<artpp::map<std::string_view, uint32_t>, uint32_t>("map<string_view,uint32_t>", N);
}

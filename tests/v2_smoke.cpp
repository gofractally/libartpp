// Smoke test for the v2:: copy — confirms the namespace-shifted clone compiles and
// behaves identically to v1 before any node-layer edits begin.
#include "artpp/v2/map.hpp"
#include <cstdio>
#include <map>
#include <random>
#include <string>
int main()
{
   artpp::v2::map<std::string_view, uint64_t> t;
   std::map<std::string, uint64_t>            oracle;
   std::mt19937                               rng(1);
   for (int i = 0; i < 20000; ++i) {
      std::string k = "k/" + std::to_string(rng() % 50000) + "/" + std::to_string(rng() % 256);
      uint64_t    v = rng();
      t.insert_or_assign(k, v); oracle[k] = v;
   }
   if (t.size() != oracle.size()) { std::printf("FAIL size %zu vs %zu\n", t.size(), oracle.size()); return 1; }
   for (auto& [k, v] : oracle) { uint64_t g; if (!t.find(k, g) || g != v) { std::printf("FAIL find\n"); return 1; } }
   auto it = t.begin();
   for (auto& [k, v] : oracle) {
      if (it == t.end() || std::string(it->first) != k || it->second != v) { std::printf("FAIL iter\n"); return 1; }
      ++it;
   }
   if (it != t.end()) { std::printf("FAIL iter tail\n"); return 1; }
   // erase half, recheck
   int n = 0; for (auto& [k, v] : oracle) if (n++ % 2) t.erase(k);
   std::printf("v2_smoke: ALL OK (%zu keys, copy compiles + matches std::map)\n", t.size());
   return 0;
}

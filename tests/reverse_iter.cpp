// artpp bidirectional-iterator fuzz vs std::map: mixed insert/erase churn, then
// full forward + reverse walks, --end()/rbegin, and random lower_bound with
// backward steps — exact agreement required at every step.
#include "artpp/map.hpp"
#include <cstdio>
#include <map>
#include <random>
#include <string>
// Clustered variable-length keys: a small alphabet drives shared prefixes (prefix
// nodes, setlists, bucket collapse) and makes erase/lower_bound probes land on or
// near existing keys instead of always missing.
static std::string mk_key(std::mt19937_64& r)
{
   std::string k;
   const int   n = 1 + int(r() % 12);
   for (int i = 0; i < n; ++i) k.push_back(char('a' + r() % 5));
   return k;
}
template <class Tree, class V = typename Tree::mapped_type>
static int run(const char* label, uint64_t seed, int rounds)
{
   Tree                        t;
   std::map<std::string, V>    m;
   std::mt19937_64                    r(seed);
   for (int round = 0; round < rounds; ++round)
   {
      // churn
      for (int i = 0; i < 2000; ++i)
      {
         std::string k = mk_key(r);
         if (r() % 3 == 0) { t.erase(k); m.erase(k); }
         else { V v = V(r()); t.insert_or_assign(k, v); m[k] = v; }
      }
      // full forward walk
      {
         auto it = t.begin(); auto mi = m.begin(); size_t n = 0;
         for (; it != t.end() && mi != m.end(); ++it, ++mi, ++n)
            if (std::string(it.key()) != mi->first || it.value() != mi->second)
            { printf("%s FWD mismatch round %d at #%zu\n", label, round, n); return 1; }
         if ((it != t.end()) || (mi != m.end()))
         { printf("%s FWD tail mismatch round %d (n=%zu of %zu)\n", label, round, n, m.size()); return 1; }
      }
      // full reverse walk via rbegin/rend
      {
         auto it = t.rbegin(); auto mi = m.rbegin(); size_t n = 0;
         for (; it != t.rend() && mi != m.rend(); ++it, ++mi, ++n)
         {
            auto fwd = it.base(); --fwd;  // the element reverse_iterator refers to
            if (std::string(fwd.key()) != mi->first || fwd.value() != mi->second)
            { printf("%s REV mismatch round %d at #%zu: got '%s' want '%s'\n", label, round, n,
                     std::string(fwd.key()).c_str(), mi->first.c_str()); return 1; }
         }
         if ((it != t.rend()) || (mi != m.rend()))
         { printf("%s REV tail mismatch round %d (n=%zu of %zu)\n", label, round, n, m.size()); return 1; }
      }
      // random lower_bound + K backward steps (mirrors std::map exactly)
      for (int probe = 0; probe < 200 && !m.empty(); ++probe)
      {
         std::string q  = mk_key(r);
         auto        it = t.lower_bound(q);
         auto        mi = m.lower_bound(q);
         for (int back = 0; back < 8; ++back)
         {
            const bool te = (it == t.end()), me = (mi == m.end());
            if (te != me) { printf("%s LB end mismatch round %d probe %d back %d\n", label, round, probe, back); return 1; }
            if (!te && (std::string(it.key()) != mi->first || it.value() != mi->second))
            { printf("%s LB mismatch round %d probe %d back %d: got '%s' want '%s'\n", label, round,
                     probe, back, std::string(it.key()).c_str(), mi->first.c_str()); return 1; }
            if (mi == m.begin()) break;  // --begin() is UB on both
            --it; --mi;
         }
      }
      // ++/-- ping-pong from a random interior position
      for (int probe = 0; probe < 100 && m.size() > 4; ++probe)
      {
         auto it = t.begin(); auto mi = m.begin();
         int  steps = int(r() % std::min<size_t>(m.size() - 2, 64)) + 1;
         for (int i = 0; i < steps; ++i) { ++it; ++mi; }
         for (int z = 0; z < 6; ++z)
         {
            if (z & 1) { ++it; ++mi; } else { --it; --mi; }
            if (std::string(it.key()) != mi->first)
            { printf("%s PINGPONG mismatch round %d probe %d z %d\n", label, round, probe, z); return 1; }
         }
      }
   }
   printf("%s: OK (%d rounds, final size %zu)\n", label, rounds, m.size());
   return 0;
}
int main()
{
   int rc = 0;
   rc |= run<artpp::map<std::string_view, uint64_t>>("mode::none   ", 42, 30);
   rc |= run<artpp::map<std::string_view, uint64_t, artpp::mode::buckets>>("mode::buckets", 43, 30);
   rc |= run<artpp::map<std::string_view, uint64_t,
                          artpp::mode(unsigned(artpp::mode::adaptive) | unsigned(artpp::mode::dense_tiers))>>(
       "adaptive|dt  ", 44, 30);
   rc |= run<artpp::map<std::string_view, uint32_t>>("u32 inline   ", 45, 30);
   return rc;
}

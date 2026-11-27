// artpp bidirectional-iterator fuzz vs std::map / std::set: mixed insert/erase churn, then
// full forward + reverse walks, --end()/rbegin, and random lower_bound with
// backward steps — exact agreement required at every step.
//
// The reverse walks DEREFERENCE the reverse iterator (*r, r->, r.key(), r.value()) and run
// std algorithms over [rbegin, rend). A key that is a string_view, and a reference to a
// value stored in the handle, point into the iterator that produced them: the reverse
// iterator therefore has to hold the iterator positioned on the element. With
// std::reverse_iterator they pointed into a temporary that operator* had already destroyed
// (gofractally/libartpp#19) — AddressSanitizer reports that, so run this test under it.
#include "artpp/map.hpp"
#include <algorithm>
#include <cstdio>
#include <iterator>
#include <map>
#include <random>
#include <set>
#include <string>
#include <vector>
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
      // full reverse walk via rbegin/rend, dereferencing the reverse iterator itself
      {
         auto it = t.rbegin(); auto mi = m.rbegin(); size_t n = 0;
         for (; it != t.rend() && mi != m.rend(); ++it, ++mi, ++n)
         {
            auto&& kv = *it;  // the pair's key view / value reference live in `it`
            if (std::string(kv.first) != mi->first || kv.second != mi->second ||
                std::string(it->first) != mi->first || it->second != mi->second ||
                std::string(it.key()) != mi->first || it.value() != mi->second ||
                std::string(it.key_bytes()) != mi->first)
            { printf("%s REV mismatch round %d at #%zu: got '%s' want '%s'\n", label, round, n,
                     std::string(it.key()).c_str(), mi->first.c_str()); return 1; }
            auto fwd = it.base(); --fwd;  // std semantics: base() is one past the element
            if (std::string(fwd.key()) != mi->first || fwd.value() != mi->second)
            { printf("%s REV base() mismatch round %d at #%zu\n", label, round, n); return 1; }
         }
         if ((it != t.rend()) || (mi != m.rend()))
         { printf("%s REV tail mismatch round %d (n=%zu of %zu)\n", label, round, n, m.size()); return 1; }
         if (!(t.rbegin().base() == t.end()) || !(t.rend().base() == t.begin()) ||
             !(t.crbegin() == t.rbegin()) || !(t.crend() == t.rend()))
         { printf("%s REV base()/c-forms mismatch round %d\n", label, round); return 1; }
      }
      // std algorithms over the reverse range
      {
         using RI = typename Tree::const_reverse_iterator;
         if (size_t(std::distance(t.rbegin(), t.rend())) != m.size())
         { printf("%s REV distance mismatch round %d\n", label, round); return 1; }
         std::vector<std::pair<std::string, V>> got, want(m.rbegin(), m.rend());
         std::transform(t.rbegin(), t.rend(), std::back_inserter(got),
                        [](const auto& kv) { return std::pair<std::string, V>(std::string(kv.first), kv.second); });
         if (got != want) { printf("%s REV transform mismatch round %d\n", label, round); return 1; }
         if (!std::equal(t.rbegin(), t.rend(), m.rbegin(), m.rend(),
                         [](const auto& a, const auto& b) { return std::string(a.first) == b.first && a.second == b.second; }))
         { printf("%s REV std::equal mismatch round %d\n", label, round); return 1; }
         for (int probe = 0; probe < 20 && !m.empty(); ++probe)
         {
            const std::string q = mk_key(r);
            // first element (from the back) whose key is < q: reverse_iterator(lower_bound(q))
            auto f  = std::find_if(t.rbegin(), t.rend(), [&](const auto& kv) { return std::string(kv.first) < q; });
            auto mf = std::find_if(m.rbegin(), m.rend(), [&](const auto& kv) { return kv.first < q; });
            RI   c(t.lower_bound(q));
            if ((f == t.rend()) != (mf == m.rend()) || !(f == c) ||
                (mf != m.rend() && (std::string(f->first) != mf->first || std::string(c.key()) != mf->first)))
            { printf("%s REV find_if / from-forward mismatch round %d probe %d\n", label, round, probe); return 1; }
            if (mf != m.rend() && std::next(mf) != m.rend())
            {
               auto n1 = std::next(f); auto m1 = std::next(mf);
               if (std::string(n1->first) != m1->first || !(std::prev(n1) == f))
               { printf("%s REV next/prev mismatch round %d probe %d\n", label, round, probe); return 1; }
            }
         }
      }
      // reverse ++/-- ping-pong from a random interior position, post-forms included
      for (int probe = 0; probe < 100 && m.size() > 4; ++probe)
      {
         auto it = t.rbegin(); auto mi = m.rbegin();
         int  steps = int(r() % std::min<size_t>(m.size() - 2, 64)) + 1;
         for (int i = 0; i < steps; ++i) { ++it; ++mi; }
         for (int z = 0; z < 6; ++z)
         {
            if (z & 1) { auto old = it++; auto mold = mi++;
                         if (std::string(old->first) != mold->first) { printf("%s REV post++ mismatch\n", label); return 1; } }
            else { auto old = it--; auto mold = mi--;
                   if (std::string(old->first) != mold->first) { printf("%s REV post-- mismatch\n", label); return 1; } }
            if (std::string(it->first) != mi->first || it->second != mi->second)
            { printf("%s REV PINGPONG mismatch round %d probe %d z %d\n", label, round, probe, z); return 1; }
         }
      }
      // rend() and back: --rend() is the first element, then walk to rbegin()
      if (!m.empty())
      {
         auto it = t.rend(); auto mi = m.rend(); size_t n = 0;
         while (mi != m.rbegin())
         {
            --it; --mi; ++n;
            if (std::string(it->first) != mi->first) { printf("%s REV back-walk mismatch round %d\n", label, round); return 1; }
         }
         if (!(it == t.rbegin()) || n != m.size()) { printf("%s REV back-walk end mismatch round %d\n", label, round); return 1; }
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
// The same for artpp::set against std::set: * yields the key alone.
template <class Set>
static int run_set(const char* label, uint64_t seed, int rounds)
{
   Set                   t;
   std::set<std::string> m;
   std::mt19937_64       r(seed);
   for (int round = 0; round < rounds; ++round)
   {
      for (int i = 0; i < 2000; ++i)
      {
         std::string k = mk_key(r);
         if (r() % 3 == 0) { t.erase(k); m.erase(k); }
         else { t.insert(k); m.insert(k); }
      }
      {
         auto it = t.rbegin(); auto mi = m.rbegin(); size_t n = 0;
         for (; it != t.rend() && mi != m.rend(); ++it, ++mi, ++n)
         {
            auto&& k = *it;
            if (std::string(k) != *mi || std::string(it.key()) != *mi || std::string(it.key_bytes()) != *mi ||
                it->size() != mi->size())
            { printf("%s SET REV mismatch round %d at #%zu\n", label, round, n); return 1; }
            auto fwd = it.base(); --fwd;
            if (std::string(*fwd) != *mi) { printf("%s SET REV base() mismatch round %d\n", label, round); return 1; }
         }
         if ((it != t.rend()) || (mi != m.rend()))
         { printf("%s SET REV tail mismatch round %d\n", label, round); return 1; }
         if (!(t.rbegin().base() == t.end()) || !(t.rend().base() == t.begin()) ||
             !(t.crbegin() == t.rbegin()) || !(t.crend() == t.rend()))
         { printf("%s SET REV base()/c-forms mismatch round %d\n", label, round); return 1; }
      }
      {
         using RI = typename Set::const_reverse_iterator;
         if (size_t(std::distance(t.rbegin(), t.rend())) != m.size())
         { printf("%s SET REV distance mismatch round %d\n", label, round); return 1; }
         std::vector<std::string> got, want(m.rbegin(), m.rend());
         std::transform(t.rbegin(), t.rend(), std::back_inserter(got), [](const auto& k) { return std::string(k); });
         if (got != want) { printf("%s SET REV transform mismatch round %d\n", label, round); return 1; }
         for (int probe = 0; probe < 20 && !m.empty(); ++probe)
         {
            const std::string q = mk_key(r);
            auto f  = std::find_if(t.rbegin(), t.rend(), [&](const auto& k) { return std::string(k) < q; });
            auto mf = std::find_if(m.rbegin(), m.rend(), [&](const auto& k) { return k < q; });
            RI   c(t.lower_bound(q));
            if ((f == t.rend()) != (mf == m.rend()) || !(f == c) || (mf != m.rend() && std::string(*f) != *mf))
            { printf("%s SET REV find_if / from-forward mismatch round %d\n", label, round); return 1; }
         }
      }
      for (int probe = 0; probe < 100 && m.size() > 4; ++probe)
      {
         auto it = t.rbegin(); auto mi = m.rbegin();
         int  steps = int(r() % std::min<size_t>(m.size() - 2, 64)) + 1;
         for (int i = 0; i < steps; ++i) { ++it; ++mi; }
         for (int z = 0; z < 6; ++z)
         {
            if (z & 1) { it++; mi++; } else { it--; mi--; }
            if (std::string(*it) != *mi) { printf("%s SET REV PINGPONG mismatch round %d\n", label, round); return 1; }
         }
      }
      if (!m.empty())
      {
         auto it = t.rend(); auto mi = m.rend();
         while (mi != m.rbegin())
         {
            --it; --mi;
            if (std::string(*it) != *mi) { printf("%s SET REV back-walk mismatch round %d\n", label, round); return 1; }
         }
         if (!(it == t.rbegin())) { printf("%s SET REV back-walk end mismatch round %d\n", label, round); return 1; }
      }
   }
   printf("%s: OK (%d rounds, final size %zu)\n", label, rounds, m.size());
   return 0;
}
// Edge cases: empty and single-element containers, default-constructed reverse iterators.
static int edges()
{
   int rc = 0;
#define EDGE(c) do { if (!(c)) { printf("edge FAIL line %d: %s\n", __LINE__, #c); rc = 1; } } while (0)
   artpp::map<std::string_view, uint32_t> e;
   EDGE(e.rbegin() == e.rend());
   EDGE(e.rend().base() == e.begin() && e.rbegin().base() == e.end());
   artpp::map<std::string_view, uint32_t>::const_reverse_iterator d1, d2;
   EDGE(d1 == d2 && d1 == e.rend());
   e.insert("only", 7);
   auto r = e.rbegin();
   EDGE(r != e.rend() && std::string(r->first) == "only" && r->second == 7 && r.value() == 7);
   const uint32_t& held = r.value();  // lives in r: valid until r moves
   EDGE(held == 7);
   ++r;
   EDGE(r == e.rend());
   --r;
   EDGE(r == e.rbegin() && std::string(r.key()) == "only");
   artpp::set<std::string_view> s;
   EDGE(s.rbegin() == s.rend());
   s.insert("some/long/shared/prefix/1"); s.insert("some/long/shared/prefix/2");
   auto sr = s.rbegin();
   std::string_view k2 = *sr;  // views sr's buffer: valid until sr moves
   EDGE(std::string(k2) == "some/long/shared/prefix/2");
   ++sr;
   EDGE(std::string(*sr) == "some/long/shared/prefix/1" && std::next(sr) == s.rend());
#undef EDGE
   printf("edges: %s\n", rc ? "FAIL" : "OK");
   return rc;
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
   rc |= run<artpp::map<std::string_view, uint32_t, artpp::mode::buckets>>("u32 buckets  ", 46, 10);
   rc |= run<artpp::map<std::string_view, uint64_t, artpp::mode::wide_stems>>("wide_stems   ", 47, 10);
   rc |= run<artpp::map<std::string, uint64_t>>("string keys  ", 48, 10);
   rc |= run_set<artpp::set<std::string_view>>("set sv       ", 49, 30);
   rc |= run_set<artpp::set<std::string_view, artpp::mode::buckets>>("set sv bkt   ", 50, 10);
   rc |= run_set<artpp::set<std::string>>("set string   ", 51, 10);
   rc |= edges();
   return rc;
}

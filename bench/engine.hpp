// bench/engine.hpp — the templated benchmark engine.
//
// ONE workload body drives every contestant: the same key vectors, the same
// op sequence, the same checksum accounting, timed the same way. Contestants
// differ only in their adapter (contestants.hpp), so the comparison carries no
// per-library benchmark deviations — apples to apples by construction.
//
// Adapter requirements (duck-typed; see contestants.hpp):
//   using key_type = std::string_view | uint64_t;
//   static const char* name();
//   void     insert(key_type, uint64_t);
//   bool     find(key_type, uint64_t& out) const;
//   bool     erase(key_type);
//   uint64_t scan_sum() const;          // full ordered iteration, summing values
#pragma once
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

namespace artpp_bench
{
   inline double now_ns()
   {
      using namespace std::chrono;
      return double(duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count());
   }

   struct row
   {
      std::string contestant, workload, op;
      std::size_t n;
      double      ns_per_op;
   };
   inline std::vector<row>& results()
   {
      static std::vector<row> r;
      return r;
   }
   inline uint64_t g_sink = 0;  // defeats dead-code elimination across contestants

   // The value for a key: derived deterministically so every contestant stores and
   // checksums identical payloads.
   inline uint64_t val_of(std::string_view k)
   {
      uint64_t h = 1469598103934665603ull;
      for (unsigned char c : k) h = (h ^ c) * 1099511628211ull;
      return h | 1;
   }
   inline uint64_t val_of(uint64_t k) { return k * 0x9E3779B97F4A7C15ull | 1; }

   template <class C, class Key>
   void run_workload(const char* workload, const std::vector<Key>& keys,
                     const std::vector<Key>& misses, int scan_reps)
   {
      const std::size_t n = keys.size();
      auto record = [&](const char* op, std::size_t ops, double t0, double t1) {
         results().push_back(row{C::name(), workload, op, ops, (t1 - t0) / double(ops)});
      };

      C c;
      {  // build
         const double t0 = now_ns();
         for (const Key& k : keys) c.insert(k, val_of(k));
         record("insert", n, t0, now_ns());
      }
      std::vector<Key> shuffled(keys);
      std::mt19937_64  rng(0xA11CE5);
      std::shuffle(shuffled.begin(), shuffled.end(), rng);
      {  // point hits — repeated; report the fastest rep (least thermal interference,
         // standard latency practice). A single pass is too noisy on a fanless machine.
         uint64_t sum = 0, want = 0;
         for (const Key& k : keys) want += val_of(k);
         double best = 1e30;
         for (int r = 0; r < scan_reps; ++r)
         {
            uint64_t     out = 0;
            sum              = 0;
            const double t0  = now_ns();
            for (const Key& k : shuffled)
               if (c.find(k, out)) sum += out;
            best = std::min(best, (now_ns() - t0) / double(n));
         }
         results().push_back(row{C::name(), workload, "hit", n, best});
         if (sum != want) std::fprintf(stderr, "BAD CHECKSUM %s/%s\n", C::name(), workload);
         g_sink ^= sum;
      }
      {  // point misses — repeated, fastest rep
         uint64_t out = 0, found = 0;
         double   best = 1e30;
         for (int r = 0; r < scan_reps; ++r)
         {
            const double t0 = now_ns();
            found           = 0;
            for (const Key& k : misses) found += c.find(k, out);
            best = std::min(best, (now_ns() - t0) / double(misses.size()));
         }
         results().push_back(row{C::name(), workload, "miss", misses.size(), best});
         if (found) std::fprintf(stderr, "MISS SET NOT DISJOINT %s/%s\n", C::name(), workload);
      }
      // lower_bound — ordered positioning to the next-greater key. The query keys are
      // ABSENT (the miss set), so every call must walk to a successor: this is the
      // operation an ordered map has and a hash map cannot, and which libart (the C
      // ART) does not provide — so only the ordered contestants produce a row.
      if constexpr (requires(const C& cc, Key kk) { cc.lower_bound_val(kk); })
      {
         uint64_t sink = 0;
         double   best = 1e30;
         for (int r = 0; r < scan_reps; ++r)
         {
            uint64_t     s  = 0;
            const double t0 = now_ns();
            for (const Key& k : misses) s += c.lower_bound_val(k);
            best = std::min(best, (now_ns() - t0) / double(misses.size()));
            sink ^= s;
         }
         results().push_back(row{C::name(), workload, "lbound", misses.size(), best});
         g_sink ^= sink;
      }
      {  // full ordered scans
         uint64_t     total = 0;
         const double t0    = now_ns();
         for (int r = 0; r < scan_reps; ++r) total += c.scan_sum();
         const double t1 = now_ns();
         record("scan", n * std::size_t(scan_reps), t0, t1);
         g_sink ^= total;
      }
      {  // erase every other key (shuffled order: non-degenerate shrink work)
         std::size_t  erased = 0;
         const double t0     = now_ns();
         for (std::size_t i = 0; i < shuffled.size(); i += 2) erased += c.erase(shuffled[i]);
         const double t1 = now_ns();
         record("erase", erased ? erased : 1, t0, t1);
      }
   }

   inline void print_table()
   {
      std::printf("\n%-22s %-14s %-7s %12s %10s\n", "contestant", "workload", "op", "n",
                  "ns/op");
      for (const row& r : results())
         std::printf("%-22s %-14s %-7s %12zu %10.1f\n", r.contestant.c_str(),
                     r.workload.c_str(), r.op.c_str(), r.n, r.ns_per_op);
   }
   inline void write_csv(const char* path)
   {
      FILE* f = std::fopen(path, "w");
      if (!f) return;
      std::fprintf(f, "contestant,workload,op,n,ns_per_op\n");
      for (const row& r : results())
         std::fprintf(f, "%s,%s,%s,%zu,%.2f\n", r.contestant.c_str(), r.workload.c_str(),
                      r.op.c_str(), r.n, r.ns_per_op);
      std::fclose(f);
      std::printf("\nwrote %s (%zu rows)\n", path, results().size());
   }
}  // namespace artpp_bench

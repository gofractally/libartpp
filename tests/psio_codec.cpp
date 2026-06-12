// psio key-format codec adapter: doubles, reflected structs and optionals as
// artpp::map keys, cross-checked against std::map for ordering and round-trip.
#include <artpp/map.hpp>
#include <artpp/psio_codec.hpp>

#include <cmath>
#include <cstdio>
#include <map>
#include <optional>
#include <random>
#include <string>

static int g_fail = 0;
#define CHECK(cond)                                                                          \
   do {                                                                                      \
      if (!(cond)) { std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); ++g_fail; } \
   } while (0)

ARTPP_PSIO_KEY(double)
ARTPP_PSIO_KEY(std::optional<int32_t>)

struct point
{
   double x = 0, y = 0;
   auto   operator<=>(const point&) const = default;
};
PSIO_REFLECT(point, x, y)
ARTPP_PSIO_KEY(point)

static void test_double_keys()
{
   artpp::map<double, uint32_t> t;
   std::map<double, uint32_t>   ref;
   std::mt19937_64              rng(7);
   for (int i = 0; i < 20000; ++i)
   {
      // mixed magnitudes and signs, including negatives and subnormal-ish values
      double d = (double(rng()) / double(rng() | 1)) * (rng() % 2 ? 1.0 : -1.0);
      if (!std::isfinite(d)) continue;
      uint32_t v = uint32_t(rng());
      t.upsert(d, v);
      ref[d] = v;
   }
   t.upsert(0.0, 42);
   ref[0.0] = 42;
   CHECK(t.size() == ref.size());
   // identical numeric iteration order and values (psio key: sign-transformed BE)
   auto mi = ref.begin();
   for (auto it = t.begin(); it != t.end(); ++it, ++mi)
      if (mi == ref.end() || it->first != mi->first || it->second != mi->second)
      {
         CHECK(false);
         break;
      }
   // point lookups + bounds by double key
   for (auto& [k, v] : ref)
      if (!(t.contains(k) && t.find_opt(k).value() == v)) { CHECK(false); break; }
   CHECK(t.lower_bound(ref.begin()->first) == t.begin());
   CHECK(t.erase(0.0) == 1 && !t.contains(0.0));
}

static void test_struct_keys()
{
   artpp::map<point, std::string> t;
   std::map<point, std::string>   ref;
   std::mt19937_64                rng(11);
   for (int i = 0; i < 5000; ++i)
   {
      point p{double(int(rng() % 200) - 100) / 4.0, double(int(rng() % 200) - 100) / 8.0};
      std::string v = "v" + std::to_string(rng() % 1000);
      t.upsert(p, v);
      ref[p] = v;
   }
   CHECK(t.size() == ref.size());
   auto mi = ref.begin();
   for (auto it = t.begin(); it != t.end(); ++it, ++mi)  // (x, y) lexicographic-numeric
      if (mi == ref.end() || it->first != mi->first || it->second != mi->second)
      {
         CHECK(false);
         break;
      }
   const point probe{12.25, -5.5};
   if (ref.count(probe))
      CHECK(t.at(probe) == ref.at(probe));
   t.upsert(probe, std::string("hit"));
   CHECK(t.at(probe) == "hit" && t.erase(probe) == 1);
}

static void test_optional_keys()
{
   using K = std::optional<int32_t>;
   artpp::map<K, uint32_t, artpp::mode::none> t;  // nullopt sorts first (psio: \0 prefix)
   t.insert(K{}, 1u);
   t.insert(K{-5}, 2u);
   t.insert(K{7}, 3u);
   CHECK(t.size() == 3);
   auto it = t.begin();
   CHECK(!it.key().has_value());
   ++it;
   CHECK(it.key() == K{-5});
   ++it;
   CHECK(it.key() == K{7});
}

int main()
{
   test_double_keys();
   test_struct_keys();
   test_optional_keys();
   if (g_fail == 0) std::printf("artpp_psio_codec: ALL PASS\n");
   else             std::printf("artpp_psio_codec: %d FAILURES\n", g_fail);
   return g_fail ? 1 : 0;
}

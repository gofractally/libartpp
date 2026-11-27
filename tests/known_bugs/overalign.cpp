// Values with alignof(T) > 16 are stored misaligned: leaves come from a 16-byte-granular terminal
// allocation and the value sits at offset alignof(T) inside it. No static_assert rejects such T.
#include "artpp/map.hpp"
#include <cstdio>
#include <cstdint>
struct alignas(32) V32 { double d[4]; };
int main()
{
   artpp::map<std::string_view, V32> m;
   int mis = 0;
   for (int i = 0; i < 64; ++i)
   {
      std::string k = "key" + std::to_string(i);
      m.insert(k, V32{{1.0 * i, 2, 3, 4}});
   }
   for (int i = 0; i < 64; ++i)
   {
      std::string k = "key" + std::to_string(i);
      const V32* p = m.find_ptr(k);
      if (reinterpret_cast<uintptr_t>(p) % alignof(V32)) ++mis;
   }
   std::printf("alignof(V32)=%zu: %d of 64 stored values are misaligned\n", alignof(V32), mis);
   return mis != 0;
}

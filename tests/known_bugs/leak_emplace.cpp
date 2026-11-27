// emplace() with a throwing value constructor leaks the leaf allocation (make_leaf has no guard).
#include "artpp/map.hpp"
#include <cstdio>
#include <string>
static long g_allocs = 0, g_frees = 0;
template <class T> struct CA {
   using value_type = T;
   CA() = default;
   template <class U> CA(const CA<U>&) noexcept {}
   T* allocate(std::size_t n) { ++g_allocs; return static_cast<T*>(::operator new(n * sizeof(T), std::align_val_t(alignof(T)))); }
   void deallocate(T* p, std::size_t) noexcept { ++g_frees; ::operator delete(p, std::align_val_t(alignof(T))); }
   template <class U> bool operator==(const CA<U>&) const noexcept { return true; }
};
struct Throws { std::string s; explicit Throws(int x) { if (x < 0) throw std::runtime_error("bad"); s = "ok"; } };
int main() {
   {
      artpp::map<std::string_view, Throws, artpp::mode::none, CA<Throws>> m;
      for (int i = 0; i < 1000; ++i)
         try { m.emplace("some-key", -1); } catch (const std::runtime_error&) {}
      std::printf("size=%zu after 1000 throwing emplaces\n", m.size());
   }
   std::printf("allocations=%ld deallocations=%ld  => leaked %ld blocks\n", g_allocs, g_frees, g_allocs - g_frees);
   return g_allocs != g_frees;
}

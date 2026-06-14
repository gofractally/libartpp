// Value lifecycle for the user type T, per the rule:
//   * trivially-copyable (POD) T  -> memcpy
//   * everything else             -> copy/move constructor, destructor
// The tree stores exactly one T per key (in a leaf); on a node split the old
// value is MOVE-constructed into its new home ("move in place") and the old one
// destroyed. Efficiency: the POD path is a straight memcpy with no ctor calls.
#pragma once
#include <cstring>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>

namespace artpp::v2::detail
{
   // Element lifetime routes through the container's allocator (uses-allocator construction:
   // a scoped/PMR allocator can intercept construct/destroy). For the default std::allocator
   // these compile to the same placement-new / dtor — trivially-copyable copies lower to a
   // memcpy on their own, so there is no hand-rolled fast path.
   template <class Alloc, class T>
   inline void v_construct(Alloc& a, T* dst, const T& v)
   {
      std::allocator_traits<Alloc>::construct(a, dst, v);
   }

   template <class Alloc, class T>
   inline void v_construct(Alloc& a, T* dst, T&& v)
   {
      std::allocator_traits<Alloc>::construct(a, dst, std::move(v));
   }

   template <class Alloc, class T>
   inline void v_destroy(Alloc& a, T* p) noexcept
   {
      std::allocator_traits<Alloc>::destroy(a, p);
   }
}  // namespace artpp::v2::detail

// key_codec<Key> — the customization point that turns a typed key into the byte string
// the radix indexes on, and back. Specialize it for your own key types:
//
//   template<> struct artpp::v2::key_codec<MyKey> {
//      static std::string_view encode(const MyKey& k, std::string& scratch);  // bytes for descent
//      static MyKey            decode(std::string_view bytes);                // bytes -> key
//   };
//
// encode() returns a view of the key's bytes: for already-contiguous keys (string-like,
// byte vectors) that's the key itself (zero-copy, `scratch` untouched); for others it
// serializes into `scratch` and returns a view of it. A codec may expose its own scratch
// type (`using scratch = ...`) — fixed-width codecs use a plain char buffer so the hot
// path never touches std::string machinery; the default scratch is std::string. decode()
// rebuilds the key from the stored bytes — which is exactly why iteration returns the key
// BY VALUE: a non-string key has no stored object to reference, it is rebuilt on demand.
//
// Built-ins: std::string, std::string_view, std::vector<uint8_t>, and integral types
// (fixed-width big-endian, sign-flipped for signed, so byte order == numeric order).
#pragma once
#include <bit>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace artpp::v2
{
   template <class Key, class = void>
   struct key_codec;  // primary template undefined — unsupported key types fail to compile

   // zero_copy codecs expose view(key) -> bytes (no buffer); the container then skips the
   // scratch object entirely on the hot path. Non-zero-copy codecs expose encode(key,
   // scratch) and serialize into scratch.
   template <>
   struct key_codec<std::string>
   {
      static constexpr bool   zero_copy = true;
      static std::string_view view(const std::string& k) noexcept { return k; }
      static std::string      decode(std::string_view b) { return std::string(b); }
   };

   template <>
   struct key_codec<std::string_view>
   {
      static constexpr bool   zero_copy = true;
      static std::string_view view(std::string_view k) noexcept { return k; }
      static std::string_view decode(std::string_view b) noexcept { return b; }  // zero-copy
   };

   template <>
   struct key_codec<std::vector<std::uint8_t>>
   {
      static constexpr bool   zero_copy = true;
      static std::string_view view(const std::vector<std::uint8_t>& k) noexcept
      {
         return std::string_view(reinterpret_cast<const char*>(k.data()), k.size());
      }
      static std::vector<std::uint8_t> decode(std::string_view b)
      {
         return std::vector<std::uint8_t>(reinterpret_cast<const std::uint8_t*>(b.data()),
                                          reinterpret_cast<const std::uint8_t*>(b.data()) + b.size());
      }
   };

   // Integral keys: fixed-width big-endian so lexicographic byte order == numeric order;
   // signed keys flip the sign bit so negatives sort before positives. One byteswap +
   // memcpy each way (the old shift loops left encode unfolded), and a fixed char buffer
   // as scratch — no std::string resize/SSO machinery on the hot path. MEASURED (M5, 2M
   // random u64 keys, alternating-binary A/B vs the string-scratch + shift-loop version):
   // find 51.3 -> 29.9 ns/op (-42%), insert 79.8 -> 65.1 (-18%) — encode runs on every op.
   template <class I>
   struct key_codec<I, std::enable_if_t<std::is_integral_v<I>>>
   {
      using U                 = std::make_unsigned_t<I>;
      static constexpr U SIGN = U(1) << (sizeof(U) * 8 - 1);
      struct scratch
      {
         char b[sizeof(U)];
      };
      static constexpr U to_be(U u) noexcept  // host -> big-endian
      {
         if constexpr (std::endian::native == std::endian::little) u = std::byteswap(u);
         return u;
      }
      static std::string_view encode(I k, scratch& s) noexcept
      {
         U u = U(k);
         if constexpr (std::is_signed_v<I>) u ^= SIGN;
         const U be = to_be(u);
         std::memcpy(s.b, &be, sizeof(U));
         return std::string_view(s.b, sizeof(U));
      }
      static I decode(std::string_view b) noexcept
      {
         U u;
         std::memcpy(&u, b.data(), sizeof(U));
         u = to_be(u);  // big-endian -> host (self-inverse)
         if constexpr (std::is_signed_v<I>) u ^= SIGN;
         return I(u);
      }
   };
}  // namespace artpp::v2

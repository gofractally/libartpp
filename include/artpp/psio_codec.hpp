// artpp/psio_codec.hpp — OPTIONAL adapter: psio's `key` format as an artpp key codec.
//
// psio::key (https://github.com/gofractally/psio) emits memcmp-sortable bytes for
// bools, integrals, floats/doubles (IEEE sign-transform), strings, optionals,
// vectors, variants, and any PSIO_REFLECT'ed struct — exactly the property an
// artpp::map needs from its encoded keys. Opting a type in gives you maps keyed
// by doubles, composite structs, or just about anything:
//
//    struct point { double x, y; };
//    PSIO_REFLECT(point, x, y)
//    ARTPP_PSIO_KEY(point)                  // <- the opt-in
//    artpp::map<point, std::string> m;      // ordered by (x, y), numerically
//
// The opt-in is per type (use_psio_key) rather than blanket so the built-in
// codecs (string/string_view zero-copy views, fixed-width integral byteswaps)
// keep their faster, scratch-free paths. Do not opt in types that already have
// a built-in codec — the partial specializations would be ambiguous.
//
// This header is only compiled when psio is available (ARTPP_WITH_PSIO).
#pragma once
#include <artpp/key_codec.hpp>
#include <psio/key.hpp>

#include <span>
#include <string_view>
#include <type_traits>
#include <vector>

namespace artpp
{
   // Specialize to std::true_type (or use ARTPP_PSIO_KEY) to route a key type
   // through psio's `key` format.
   template <class K>
   struct use_psio_key : std::false_type
   {
   };

   template <class K>
   struct key_codec<K, std::enable_if_t<use_psio_key<K>::value>>
   {
      using scratch = std::vector<char>;  // psio encodes into a growable sink
      static std::string_view encode(const K& k, scratch& s)
      {
         s.clear();
         psio::encode(psio::key{}, k, s);
         return std::string_view(s.data(), s.size());
      }
      static K decode(std::string_view b)
      {
         return psio::decode<K>(psio::key{}, std::span<const char>(b.data(), b.size()));
      }
   };
}  // namespace artpp

// One-line opt-in, usable at namespace scope after the type's definition.
#define ARTPP_PSIO_KEY(...)                                  \
   template <>                                               \
   struct artpp::use_psio_key<__VA_ARGS__> : std::true_type  \
   {                                                         \
   };

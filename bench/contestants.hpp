// bench/contestants.hpp — one thin adapter per contestant, per key family.
// Each library is used through its natural interface (typed integer keys where
// the library supports them; libart, being bytes-only, receives the same keys
// big-endian-encoded — the identical bytes artpp's integral codec produces).
#pragma once
#include <artpp/map.hpp>
#include <artpp/pool.hpp>

#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <string_view>

#include <absl/container/btree_map.h>

extern "C" {
#include "art.h"  // vendored upstream libart (external/libart)
}

namespace artpp_bench
{
   // ── string_view-keyed family ──────────────────────────────────────────────
   // The flagship configuration: artpp::map over its line_pool allocator —
   // 4-byte indexed branch handles, bump placement laying children near their
   // parents. This is the library's intended deployment; the std::allocator
   // row below shows what you get with no allocator setup at all.
   struct artpp_sv
   {
      using key_type = std::string_view;
      using A        = artpp::pool_alloc<uint64_t>;
      static const char* name() { return "artpp::map"; }
      artpp::line_pool pool;
      artpp::map<std::string_view, uint64_t, artpp::mode::none, A> m{A{&pool}};
      void     insert(key_type k, uint64_t v) { m.insert(k, v); }
      bool     find(key_type k, uint64_t& out) const { return m.find(k, out); }
      bool     erase(key_type k) { return m.erase(k) != 0; }
      uint64_t scan_sum() const
      {
         uint64_t s = 0;
         m.for_each_value([&](const uint64_t& v) { s += v; });
         return s;
      }
   };

   struct artpp_buckets_sv
   {
      using key_type = std::string_view;
      using A        = artpp::pool_alloc<uint64_t>;
      static const char* name() { return "artpp::map<buckets>"; }
      artpp::line_pool pool;
      artpp::map<std::string_view, uint64_t, artpp::mode::buckets, A> m{A{&pool}};
      void     insert(key_type k, uint64_t v) { m.insert(k, v); }
      bool     find(key_type k, uint64_t& out) const { return m.find(k, out); }
      bool     erase(key_type k) { return m.erase(k) != 0; }
      uint64_t scan_sum() const
      {
         uint64_t s = 0;
         m.for_each_value([&](const uint64_t& v) { s += v; });
         return s;
      }
   };

   struct artpp_malloc_sv
   {
      using key_type = std::string_view;
      static const char* name() { return "artpp::map (std::allocator)"; }
      artpp::map<std::string_view, uint64_t> m;
      void     insert(key_type k, uint64_t v) { m.insert(k, v); }
      bool     find(key_type k, uint64_t& out) const { return m.find(k, out); }
      bool     erase(key_type k) { return m.erase(k) != 0; }
      uint64_t scan_sum() const
      {
         uint64_t s = 0;
         m.for_each_value([&](const uint64_t& v) { s += v; });
         return s;
      }
   };

   struct libart_sv
   {
      using key_type = std::string_view;
      static const char* name() { return "libart"; }
      art_tree t;
      libart_sv() { art_tree_init(&t); }
      ~libart_sv() { art_tree_destroy(&t); }
      void insert(key_type k, uint64_t v)
      {
         art_insert(&t, reinterpret_cast<const unsigned char*>(k.data()), int(k.size()),
                    reinterpret_cast<void*>(uintptr_t(v)));
      }
      bool find(key_type k, uint64_t& out) const
      {
         void* r = art_search(const_cast<art_tree*>(&t),
                              reinterpret_cast<const unsigned char*>(k.data()), int(k.size()));
         if (!r) return false;
         out = uint64_t(uintptr_t(r));
         return true;
      }
      bool erase(key_type k)
      {
         return art_delete(&t, reinterpret_cast<const unsigned char*>(k.data()),
                           int(k.size())) != nullptr;
      }
      uint64_t scan_sum() const
      {
         uint64_t s = 0;
         art_iter(const_cast<art_tree*>(&t),
                  [](void* data, const unsigned char*, uint32_t, void* value) {
                     *static_cast<uint64_t*>(data) += uint64_t(uintptr_t(value));
                     return 0;
                  },
                  &s);
         return s;
      }
   };

   struct absl_btree_sv
   {
      using key_type = std::string_view;
      static const char* name() { return "absl::btree_map"; }
      absl::btree_map<std::string, uint64_t, std::less<>> m;  // heterogeneous lookup
      void     insert(key_type k, uint64_t v) { m.insert_or_assign(std::string(k), v); }
      bool     find(key_type k, uint64_t& out) const
      {
         auto it = m.find(k);
         if (it == m.end()) return false;
         out = it->second;
         return true;
      }
      bool     erase(key_type k)
      {
         auto it = m.find(k);
         if (it == m.end()) return false;
         m.erase(it);
         return true;
      }
      uint64_t scan_sum() const
      {
         uint64_t s = 0;
         for (const auto& [k, v] : m) s += v;
         return s;
      }
   };

   struct std_map_sv
   {
      using key_type = std::string_view;
      static const char* name() { return "std::map"; }
      std::map<std::string, uint64_t, std::less<>> m;
      void     insert(key_type k, uint64_t v) { m.insert_or_assign(std::string(k), v); }
      bool     find(key_type k, uint64_t& out) const
      {
         auto it = m.find(k);
         if (it == m.end()) return false;
         out = it->second;
         return true;
      }
      bool     erase(key_type k)
      {
         auto it = m.find(k);
         if (it == m.end()) return false;
         m.erase(it);
         return true;
      }
      uint64_t scan_sum() const
      {
         uint64_t s = 0;
         for (const auto& [k, v] : m) s += v;
         return s;
      }
   };

   // ── uint64_t-keyed family ─────────────────────────────────────────────────
   struct artpp_u64
   {
      using key_type = uint64_t;
      using A        = artpp::pool_alloc<uint64_t>;
      static const char* name() { return "artpp::map"; }
      artpp::line_pool pool;
      artpp::map<uint64_t, uint64_t, artpp::mode::none, A> m{A{&pool}};
      void     insert(key_type k, uint64_t v) { m.insert(k, v); }
      bool     find(key_type k, uint64_t& out) const { return m.find(k, out); }
      bool     erase(key_type k) { return m.erase(k) != 0; }
      uint64_t scan_sum() const
      {
         uint64_t s = 0;
         m.for_each_value([&](const uint64_t& v) { s += v; });
         return s;
      }
   };

   struct artpp_malloc_u64
   {
      using key_type = uint64_t;
      static const char* name() { return "artpp::map (std::allocator)"; }
      artpp::map<uint64_t, uint64_t> m;
      void     insert(key_type k, uint64_t v) { m.insert(k, v); }
      bool     find(key_type k, uint64_t& out) const { return m.find(k, out); }
      bool     erase(key_type k) { return m.erase(k) != 0; }
      uint64_t scan_sum() const
      {
         uint64_t s = 0;
         m.for_each_value([&](const uint64_t& v) { s += v; });
         return s;
      }
   };

   struct libart_u64
   {
      using key_type = uint64_t;
      static const char* name() { return "libart"; }
      art_tree t;
      libart_u64() { art_tree_init(&t); }
      ~libart_u64() { art_tree_destroy(&t); }
      static void enc(uint64_t k, unsigned char* b)  // big-endian: byte order == numeric
      {
         for (int i = 7; i >= 0; --i) { b[i] = static_cast<unsigned char>(k); k >>= 8; }
      }
      void insert(key_type k, uint64_t v)
      {
         unsigned char b[8];
         enc(k, b);
         art_insert(&t, b, 8, reinterpret_cast<void*>(uintptr_t(v)));
      }
      bool find(key_type k, uint64_t& out) const
      {
         unsigned char b[8];
         enc(k, b);
         void* r = art_search(const_cast<art_tree*>(&t), b, 8);
         if (!r) return false;
         out = uint64_t(uintptr_t(r));
         return true;
      }
      bool erase(key_type k)
      {
         unsigned char b[8];
         enc(k, b);
         return art_delete(&t, b, 8) != nullptr;
      }
      uint64_t scan_sum() const
      {
         uint64_t s = 0;
         art_iter(const_cast<art_tree*>(&t),
                  [](void* data, const unsigned char*, uint32_t, void* value) {
                     *static_cast<uint64_t*>(data) += uint64_t(uintptr_t(value));
                     return 0;
                  },
                  &s);
         return s;
      }
   };

   struct absl_btree_u64
   {
      using key_type = uint64_t;
      static const char* name() { return "absl::btree_map"; }
      absl::btree_map<uint64_t, uint64_t> m;
      void     insert(key_type k, uint64_t v) { m.insert_or_assign(k, v); }
      bool     find(key_type k, uint64_t& out) const
      {
         auto it = m.find(k);
         if (it == m.end()) return false;
         out = it->second;
         return true;
      }
      bool     erase(key_type k) { return m.erase(k) != 0; }
      uint64_t scan_sum() const
      {
         uint64_t s = 0;
         for (const auto& [k, v] : m) s += v;
         return s;
      }
   };

   struct std_map_u64
   {
      using key_type = uint64_t;
      static const char* name() { return "std::map"; }
      std::map<uint64_t, uint64_t> m;
      void     insert(key_type k, uint64_t v) { m.insert_or_assign(k, v); }
      bool     find(key_type k, uint64_t& out) const
      {
         auto it = m.find(k);
         if (it == m.end()) return false;
         out = it->second;
         return true;
      }
      bool     erase(key_type k) { return m.erase(k) != 0; }
      uint64_t scan_sum() const
      {
         uint64_t s = 0;
         for (const auto& [k, v] : m) s += v;
         return s;
      }
   };
}  // namespace artpp_bench

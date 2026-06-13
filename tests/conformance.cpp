// artpp_conformance.cpp — STL-grade conformance and edge-case suite for artpp::map.
//
// Complements the existing binaries — artpp_smoke (API basics, allocators, modes),
// artpp_reverse_iter (bidirectional fuzz vs std::map), artpp_eh (exception safety under
// fault injection), artpp_wide (u16 routers) — with the contracts a standard-library
// review would probe:
//   * key-length limit: max_key_bytes boundary, length_error, overlong lookups miss
//   * move construction/assignment branches, self-move, moved-from reuse
//   * exactly-once value construction (emplace/insert/update/remove event counts)
//   * move-only mapped types end-to-end (radix and bucket modes)
//   * compile-time API constraints (inline-value trees hide by-reference access)
//   * find_opt, for_each_value, tree equality, erase(iterator) incl. successor
//   * iterator edge cases (empty tree, single element, default-constructed, bounds)
//   * classic-algorithm compatibility over the proxy bidirectional iterator
//   * degenerate shapes: ascending/descending integers, deep prefix chains, clusters
//   * mixed-operation differential fuzz vs std::map across all modes
#include "artpp/pool.hpp"
#include "artpp/map.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <iterator>
#include <map>
#include <memory>
#include <memory_resource>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using artpp::map;
using artpp::mode;

static int g_fail = 0;
#define CHECK(cond)                                                                          \
   do {                                                                                      \
      if (!(cond)) { std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); ++g_fail; } \
   } while (0)

template <class Tree, class V>
static bool has(const Tree& t, std::string_view k, const V& exp)
{
   typename Tree::mapped_type v{};
   return t.find(k, v) && v == exp;
}
template <class Tree>
static bool miss(const Tree& t, std::string_view k)
{
   return !t.contains(k);
}

// ── key-length limit ──────────────────────────────────────────────────────────
// Mutations past max_key_bytes throw std::length_error and leave the tree intact
// (the check precedes any allocation or structural change — strong guarantee).
// Lookups of overlong keys miss without throwing: no stored key can be that long.
template <class Tree>
static void key_length_body()
{
   Tree t;
   t.insert("anchor", 7u);

   const std::string maxk(Tree::max_key_bytes, 'm');     // exactly at the limit: legal
   const std::string over(Tree::max_key_bytes + 1, 'v'); // one past: must throw

   CHECK(t.insert(maxk, 1u));
   CHECK(has(t, maxk, 1u));

   bool threw = false;
   try { t.insert(over, 2u); } catch (const std::length_error&) { threw = true; }
   CHECK(threw);
   threw = false;
   try { t.emplace(over, 3u); } catch (const std::length_error&) { threw = true; }
   CHECK(threw);

   CHECK(t.size() == 2);                 // anchor + maxk only; the throws changed nothing
   CHECK(!t.contains(over));             // lookup: clean miss, no throw
   CHECK(t.find(over) == t.end());
   CHECK(t.erase(over) == 0);
   CHECK(t.count(over) == 0);

   // the at-limit key is a first-class citizen: iterates and erases
   size_t seen = 0;
   for (auto it = t.begin(); it != t.end(); ++it)
      if (it.key_bytes().size() == Tree::max_key_bytes) ++seen;
   CHECK(seen == 1);
   CHECK(t.erase(maxk) == 1 && t.size() == 1);
}

static void test_key_length_limit()
{
   static_assert(map<std::string_view, uint32_t>::max_key_bytes == 65535);
   key_length_body<map<std::string_view, uint32_t>>();
   key_length_body<map<std::string_view, uint32_t, mode::buckets>>();
   key_length_body<map<std::string_view, uint32_t, mode::adaptive>>();
}

// Long-but-legal keys: multi-kilobyte keys sharing long prefixes round-trip through
// insert, point lookup, ordered iteration (key reconstruction), bounds, and erase.
static void test_long_keys()
{
   map<std::string_view, uint64_t> t;
   std::map<std::string, uint64_t>     ref;
   const std::string                   base(4096, 'p');
   for (int i = 0; i < 32; ++i)
   {
      // diverge at varying depths inside the long prefix, including past 32 KB
      std::string k = base.substr(0, 128 + 113 * i) + char('A' + i % 26) + std::to_string(i);
      if (i % 5 == 0) k += std::string(40000, char('a' + i % 3));  // deep tails past 0x8000
      ref[k] = uint64_t(i);
      t.insert(k, uint64_t(i));
   }
   CHECK(t.size() == ref.size());
   auto mi = ref.begin();
   for (auto it = t.begin(); it != t.end(); ++it, ++mi)
   {
      CHECK(mi != ref.end() && std::string(it.key()) == mi->first && it.value() == mi->second);
   }
   CHECK(mi == ref.end());
   for (auto& [k, v] : ref) CHECK(has(t, k, v));
   // erase half through iterators, half by key
   int n = 0;
   for (auto& [k, v] : ref)
      if (++n % 2) CHECK(t.erase(k) == 1);
   CHECK(t.size() == ref.size() / 2);
}

// ── move semantics ────────────────────────────────────────────────────────────
// A stateful allocator with configurable propagation, counting outstanding bytes:
// proves each [container.requirements] move/swap branch routes memory correctly.
struct mv_counters
{
   long live = 0, allocs = 0;
};
template <class T, class POCMA>  // POCMA: std::true_type / std::false_type (a type
struct mv_alloc                  // parameter, so allocator_traits' default rebind works)
{
   using value_type                             = T;
   using propagate_on_container_move_assignment = POCMA;
   using propagate_on_container_swap            = std::true_type;
   mv_counters* c;
   explicit mv_alloc(mv_counters* cc) noexcept : c(cc) {}
   template <class U>
   mv_alloc(const mv_alloc<U, POCMA>& o) noexcept : c(o.c) {}
   T* allocate(std::size_t n)
   {
      c->allocs++;
      c->live += long(n * sizeof(T));
      return static_cast<T*>(::operator new(n * sizeof(T), std::align_val_t(alignof(T))));
   }
   void deallocate(T* p, std::size_t n) noexcept
   {
      c->live -= long(n * sizeof(T));
      ::operator delete(p, std::align_val_t(alignof(T)));
   }
   template <class U>
   bool operator==(const mv_alloc<U, POCMA>& o) const noexcept { return c == o.c; }
};
template <class T>
using mv_alloc_p = mv_alloc<T, std::true_type>;   // propagates on move assignment
template <class T>
using mv_alloc_n = mv_alloc<T, std::false_type>;  // does not propagate

static void test_move_semantics()
{
   using Tree = map<std::string_view, std::string>;
   // move construction steals: contents transfer, source is valid-empty and reusable
   {
      Tree a;
      for (int i = 0; i < 500; ++i) a.insert("k" + std::to_string(i), std::string(30, 'x'));
      Tree b(std::move(a));
      CHECK(b.size() == 500 && has(b, "k499", std::string(30, 'x')));
      CHECK(a.size() == 0 && a.empty() && a.begin() == a.end());
      a.insert("fresh", std::string("f"));  // moved-from object is fully usable
      CHECK(a.size() == 1 && has(a, "fresh", std::string("f")));
   }
   // POCMA=true move assignment: the allocator travels with its nodes, target's old
   // content is freed through the target's old allocator; every byte returns
   {
      mv_counters ca, cb;
      using A  = mv_alloc_p<std::string>;
      using Tr = map<std::string_view, std::string, mode::none, A>;
      {
         Tr a{A{&ca}}, b{A{&cb}};
         a.insert("old", std::string("o"));
         for (int i = 0; i < 300; ++i) b.insert("b" + std::to_string(i), std::string(20, 'b'));
         a = std::move(b);
         CHECK(a.size() == 300 && !a.contains("old"));
         CHECK(a.get_allocator() == A{&cb});  // allocator propagated with the nodes
         CHECK(b.empty());
         CHECK(ca.live == 0);                 // a's old tree freed via ca already
      }
      CHECK(cb.live == 0 && ca.live == 0);
   }
   // POCMA=false + EQUAL allocators: stealing is legal (either can free the other's
   // nodes); the steal must happen (cheap path), contents transfer
   {
      mv_counters cs;
      using A  = mv_alloc_n<std::string>;
      using Tr = map<std::string_view, std::string, mode::none, A>;
      {
         Tr a{A{&cs}}, b{A{&cs}};
         a.insert("gone", std::string("g"));
         for (int i = 0; i < 300; ++i) b.insert("b" + std::to_string(i), std::string(20, 'b'));
         const long allocs_before = cs.allocs;
         a = std::move(b);
         CHECK(cs.allocs == allocs_before);  // a steal: zero new allocations
         CHECK(a.size() == 300 && b.empty() && !a.contains("gone"));
      }
      CHECK(cs.live == 0);
   }
   // self-move: container must remain valid (we guarantee: unchanged)
   {
      Tree a;
      a.insert("s", std::string("v"));
      Tree& ra = a;
      a = std::move(ra);
      CHECK(a.size() == 1 && has(a, "s", std::string("v")));
   }
   // noexcept contracts
   static_assert(std::is_nothrow_move_constructible_v<Tree>);
   static_assert(std::is_nothrow_move_assignable_v<Tree>);  // std::allocator: always equal
   static_assert(!std::is_copy_constructible_v<Tree> && !std::is_copy_assignable_v<Tree>);
}

// ── cross-pool move assignment: bulk image adoption ───────────────────────────
// pool_alloc opts into artpp_adopt: with index-based handles and trivially-copyable
// values, moving a tree between two pools is one memcpy of the used range — every
// handle (including the root) is base-relative and stays valid verbatim. No
// per-key rebuild, no per-node clone.
template <class Al, class = void>
struct test_has_adopt : std::false_type {};
template <class Al>
struct test_has_adopt<Al, std::void_t<decltype(std::declval<Al&>().artpp_adopt(std::declval<const Al&>()))>>
    : std::true_type {};

static void test_pool_bulk_move()
{
   using A    = artpp::pool_alloc<uint64_t>;
   using Tree = map<std::string_view, uint64_t, mode::none, A>;
   static_assert(test_has_adopt<A>::value);                          // pool opts in
   static_assert(!test_has_adopt<std::allocator<uint64_t>>::value);  // raw pointers can't

   artpp::line_pool pa(size_t(1) << 30), pb(size_t(1) << 30);
   Tree           a{A{&pa}}, b{A{&pb}};
   std::map<std::string, uint64_t> ref;
   std::mt19937_64                 rng(31);
   for (int i = 0; i < 20000; ++i)
   {
      std::string k;
      for (int j = 0, n = 1 + int(rng() % 10); j < n; ++j) k.push_back(char('a' + rng() % 16));
      uint64_t v = rng();
      b.upsert(k, v);
      ref[k] = v;
   }
   a.insert("discarded", 1u);
   const auto shape_before = b.debug_stats();

   a = std::move(b);  // unequal pools, non-propagating → bulk adopt (one memcpy)

   CHECK(a.size() == ref.size() && !a.contains("discarded"));
   CHECK(b.empty() && b.size() == 0);
   const auto shape_after = a.debug_stats();  // the IMAGE moved: shape is identical
   CHECK(shape_after.terminals == shape_before.terminals &&
         shape_after.setlist == shape_before.setlist &&
         shape_after.full == shape_before.full && shape_after.prefix == shape_before.prefix);
   auto mi = ref.begin();
   for (auto it = a.begin(); it != a.end(); ++it, ++mi)
      if (std::string(it.key()) != mi->first || it.value() != mi->second) { CHECK(false); break; }
   // the adopted image is fully live: mutate through it
   CHECK(a.erase(ref.begin()->first) == 1);
   a.upsert("post-adopt", 42u);
   CHECK(has(a, "post-adopt", uint64_t(42)));
   // the drained source is reusable in its own pool
   b.insert("fresh", 7u);
   CHECK(b.size() == 1 && has(b, "fresh", uint64_t(7)));
}

// ── cross-allocator move assignment: node-by-node structural clone ────────────
// Raw-pointer allocators can't move bytes between arenas; the fallback is a
// structural clone — same shape as the source (verified via the node census),
// values moved object-by-object, never a key-by-key reinsert.
static void test_structural_clone_move()
{
   // non-trivial values (move-constructed per leaf), radix shape preserved
   {
      mv_counters ca, cb;
      using Al = mv_alloc_n<std::string>;
      using Tr = map<std::string_view, std::string, mode::none, Al>;
      std::map<std::string, std::string> ref;
      {
         Tr              a{Al{&ca}}, b{Al{&cb}};
         std::mt19937_64 rng(17);
         for (int i = 0; i < 8000; ++i)
         {
            std::string k;
            for (int j = 0, n = 1 + int(rng() % 12); j < n; ++j) k.push_back(char('a' + rng() % 9));
            std::string v(1 + rng() % 60, char('A' + rng() % 26));
            b.upsert(k, v);
            ref[k] = v;
         }
         a.insert("discarded", std::string("x"));
         const auto shape_before = b.debug_stats();
         a                       = std::move(b);
         CHECK(a.get_allocator() == Al{&ca});  // non-propagating: a keeps its allocator
         CHECK(a.size() == ref.size() && b.empty());
         const auto shape_after = a.debug_stats();  // structural clone: identical census
         CHECK(shape_after.terminals == shape_before.terminals &&
               shape_after.setlist == shape_before.setlist &&
               shape_after.full == shape_before.full &&
               shape_after.prefix == shape_before.prefix && shape_after.leaf == shape_before.leaf);
         auto mi = ref.begin();
         for (auto it = a.begin(); it != a.end(); ++it, ++mi)
            if (std::string(it.key()) != mi->first || it.value() != mi->second) { CHECK(false); break; }
         b.insert("fresh", std::string("f"));
         CHECK(has(b, "fresh", std::string("f")));
      }
      CHECK(ca.live == 0 && cb.live == 0);  // every byte returned on both sides
   }
   // bucket-mode subtrees: entries move-construct through the bucket clone arm
   {
      mv_counters ca, cb;
      using Al = mv_alloc_n<std::string>;
      using Tr = map<std::string_view, std::string, mode::buckets, Al>;
      {
         Tr              a{Al{&ca}}, b{Al{&cb}};
         std::mt19937_64 rng(19);
         std::map<std::string, std::string> ref;
         for (int i = 0; i < 3000; ++i)
         {
            std::string k = "pfx" + std::to_string(rng() % 800);
            std::string v(1 + rng() % 30, 'v');
            b.upsert(k, v);
            ref[k] = v;
         }
         a = std::move(b);
         CHECK(a.size() == ref.size() && b.empty());
         for (auto& [k, v] : ref)
            if (!has(a, k, v)) { CHECK(false); break; }
      }
      CHECK(ca.live == 0 && cb.live == 0);
   }
   // move-only values ride the clone by move construction
   {
      mv_counters ca, cb;
      using Al = mv_alloc_n<std::unique_ptr<int>>;
      using Tr = map<std::string_view, std::unique_ptr<int>, mode::none, Al>;
      Tr a{Al{&ca}}, b{Al{&cb}};
      for (int i = 0; i < 200; ++i) b.emplace("m" + std::to_string(i), std::make_unique<int>(i));
      a = std::move(b);
      CHECK(a.size() == 200 && b.empty());
      for (int i = 0; i < 200; ++i)
         if (*a.at("m" + std::to_string(i)) != i) { CHECK(false); break; }
   }
}

// ── exactly-once construction ─────────────────────────────────────────────────
// The emplace/insert contracts promise: construct the value exactly once, at its
// final address, and not at all when an emplace finds the key present (stronger
// than std::map). Probe records every special-member event; single-key scenarios
// make the expected counts exact (no splits can move values around).
struct probe_counts
{
   int ctor = 0, copy_ctor = 0, move_ctor = 0, copy_asgn = 0, move_asgn = 0, dtor = 0;
   bool operator==(const probe_counts&) const = default;
};
static probe_counts g_pc;
struct Probe
{
   uint64_t payload = 0;
   Probe() { g_pc.ctor++; }
   explicit Probe(uint64_t p) : payload(p) { g_pc.ctor++; }
   Probe(uint64_t a, uint64_t b) : payload(a + b) { g_pc.ctor++; }  // multi-arg emplace
   Probe(const Probe& o) : payload(o.payload) { g_pc.copy_ctor++; }
   Probe(Probe&& o) noexcept : payload(o.payload) { g_pc.move_ctor++; }
   Probe& operator=(const Probe& o) { payload = o.payload; g_pc.copy_asgn++; return *this; }
   Probe& operator=(Probe&& o) noexcept { payload = o.payload; g_pc.move_asgn++; return *this; }
   ~Probe() { g_pc.dtor++; }
   bool operator==(const Probe& o) const { return payload == o.payload; }
};
static probe_counts diff_since(const probe_counts& a)
{
   probe_counts d = g_pc;
   return {d.ctor - a.ctor,           d.copy_ctor - a.copy_ctor, d.move_ctor - a.move_ctor,
           d.copy_asgn - a.copy_asgn, d.move_asgn - a.move_asgn, d.dtor - a.dtor};
}

static void test_exact_construction()
{
   map<std::string_view, Probe> t;
   // emplace on a new key: exactly one construction, from args, nothing else
   probe_counts m = g_pc;
   CHECK(t.emplace("k", uint64_t(40), uint64_t(2)));
   CHECK((diff_since(m) == probe_counts{1, 0, 0, 0, 0, 0}));
   CHECK(t.at("k").payload == 42);

   // emplace on an existing key: untouched, ZERO constructions of any kind
   m = g_pc;
   CHECK(!t.emplace("k", uint64_t(9), uint64_t(9)));
   CHECK((diff_since(m) == probe_counts{0, 0, 0, 0, 0, 0}));
   CHECK(t.at("k").payload == 42);

   // insert(k, T&&) on an existing key: one move-assign onto the live object
   {
      Probe v(100);
      m = g_pc;
      CHECK(!t.insert("k", std::move(v)));
      CHECK((diff_since(m) == probe_counts{0, 0, 0, 0, 1, 0}));
      CHECK(t.at("k").payload == 100);
   }
   // update on an existing key: one move-assign; on a missing key: zero events
   {
      Probe v(7);
      m = g_pc;
      CHECK(t.update("k", std::move(v)));
      CHECK((diff_since(m) == probe_counts{0, 0, 0, 0, 1, 0}));
      Probe w(8);
      m = g_pc;
      CHECK(!t.update("absent", std::move(w)));
      CHECK((diff_since(m) == probe_counts{0, 0, 0, 0, 0, 0}));
   }
   // insert(k2, T&&) on a NEW key in a 1-key tree: the root leaf splits, which moves
   // the resident value into its new leaf — so: 2 move-ctors (new value placed + old
   // value relocated), 1 dtor (old leaf's value), nothing copied
   {
      Probe v(1);
      m = g_pc;
      CHECK(t.insert("q", std::move(v)));
      CHECK((diff_since(m) == probe_counts{0, 0, 2, 0, 0, 1}));
   }
   // remove: exactly one destruction
   m = g_pc;
   CHECK(t.remove("q"));
   CHECK((diff_since(m) == probe_counts{0, 0, 0, 0, 0, 1}));

   // operator[] on a missing key: default-construct + move into place + temp dtor
   m = g_pc;
   t["fresh"].payload = 5;
   CHECK((diff_since(m) == probe_counts{1, 0, 1, 0, 0, 1}));

   // lifetime balance across a randomized churn + clear
   {
      std::mt19937_64 rng(7);
      for (int i = 0; i < 4000; ++i)
      {
         std::string k(1 + size_t(rng() % 6), 'a');
         for (auto& ch : k) ch = char('a' + rng() % 8);
         if (rng() % 4 == 0) t.erase(k);
         else                t.emplace(k, uint64_t(rng()), uint64_t(1));
      }
      t.clear();
   }
   CHECK(g_pc.ctor + g_pc.copy_ctor + g_pc.move_ctor == g_pc.dtor);  // zero leaks
}

// ── move-only mapped type ─────────────────────────────────────────────────────
template <class Tree>
static void move_only_body()
{
   Tree t;
   CHECK(t.emplace("a", std::make_unique<int>(1)));
   CHECK(t.insert("b", std::make_unique<int>(2)));
   t["c"] = std::make_unique<int>(3);     // default-insert, then move-assign through T&
   CHECK(!t.emplace("a", std::make_unique<int>(9)));  // present: arg NOT consumed into tree
   CHECK(t.size() == 3);
   CHECK(*t.at("a") == 1 && *t.at("b") == 2 && *t.at("c") == 3);
   CHECK(t.find_ptr("zz") == nullptr);
   *t.find_ptr("b") = std::make_unique<int>(22);  // mutate through the reference
   CHECK(*t.at("b") == 22);
   int sum = 0;
   for (const auto& [k, v] : t) sum += *v;  // value() is const unique_ptr&
   CHECK(sum == 1 + 22 + 3);
   auto it = t.find("b");
   CHECK(it != t.end());
   it = t.erase(it);                       // erase(iterator) returns the successor
   CHECK(it != t.end() && std::string(it.key()) == "c");
   CHECK(t.size() == 2 && !t.contains("b"));
   t.clear();
   CHECK(t.empty());
}

static void test_move_only()
{
   move_only_body<map<std::string_view, std::unique_ptr<int>>>();
   move_only_body<map<std::string_view, std::unique_ptr<int>, mode::buckets>>();
   move_only_body<map<std::string_view, std::unique_ptr<int>, mode::adaptive>>();
}

// ── compile-time API constraints ──────────────────────────────────────────────
template <class M, class = void>
struct has_subscript : std::false_type {};
template <class M>
struct has_subscript<M, std::void_t<decltype(std::declval<M&>()[std::declval<std::string_view>()])>>
    : std::true_type {};
template <class M, class = void>
struct has_find_ptr : std::false_type {};
template <class M>
struct has_find_ptr<M, std::void_t<decltype(std::declval<M&>().find_ptr(std::declval<std::string_view>()))>>
    : std::true_type {};
template <class M, class = void>
struct has_at : std::false_type {};
template <class M>
struct has_at<M, std::void_t<decltype(std::declval<M&>().at(std::declval<std::string_view>()))>>
    : std::true_type {};

static void test_compile_constraints()
{
   using Inline = map<std::string_view, uint32_t>;   // 4 <= 5-byte inline cap
   using Leafed = map<std::string_view, uint64_t>;   // 8 > inline cap: real objects
   static_assert(Inline::inlineable && !Leafed::inlineable);
   // inline values have no addressable object: by-reference access is constrained off
   static_assert(!has_subscript<Inline>::value && !has_find_ptr<Inline>::value && !has_at<Inline>::value);
   static_assert(has_subscript<Leafed>::value && has_find_ptr<Leafed>::value && has_at<Leafed>::value);
   // ...and operator[] additionally requires default-constructible T
   struct NoDef { uint64_t a, b; NoDef(int) : a(0), b(0) {} };  // 16 B: leaf-stored
   static_assert(!has_subscript<map<std::string_view, NoDef>>::value);
   static_assert(has_find_ptr<map<std::string_view, NoDef>>::value);

   using It = Leafed::const_iterator;
   static_assert(std::is_same_v<It::iterator_category, std::bidirectional_iterator_tag>);
   static_assert(std::is_same_v<Leafed::iterator, Leafed::const_iterator>);
   static_assert(std::is_same_v<Leafed::value_type, std::pair<const std::string_view, uint64_t>>);
   static_assert(std::is_same_v<Leafed::key_type, std::string_view>);
   static_assert(std::is_same_v<Leafed::mapped_type, uint64_t>);
   static_assert(std::is_same_v<Leafed::reference, It::reference>);
   static_assert(std::is_same_v<Leafed::size_type, std::size_t>);
   static_assert(std::is_nothrow_swappable_v<Leafed>);
   // reverse adaptor is the std one over the bidirectional iterator
   static_assert(std::is_same_v<Leafed::reverse_iterator, std::reverse_iterator<It>>);
}

// ── find_opt / for_each_value ─────────────────────────────────────────────────
static void test_find_opt()
{
   map<std::string_view, uint32_t> a;  // inline values
   a.insert("x", 7u);
   CHECK(a.find_opt("x").has_value() && *a.find_opt("x") == 7u);
   CHECK(!a.find_opt("y").has_value());

   map<std::string_view, std::string> b;  // leaf values
   b.insert("x", std::string("seven"));
   CHECK(b.find_opt("x").value() == "seven");
   CHECK(!b.find_opt("nope").has_value());
}

static void test_for_each_value()
{
   map<std::string_view, uint64_t> t;
   std::map<std::string, uint64_t>     ref;
   std::mt19937_64                     rng(99);
   for (int i = 0; i < 5000; ++i)
   {
      std::string k = std::to_string(rng() % 100000);
      uint64_t    v = rng();
      t.upsert(k, v);
      ref[k] = v;
   }
   std::vector<uint64_t> vals, want;
   t.for_each_value([&](const uint64_t& v) { vals.push_back(v); });
   for (auto& [k, v] : ref) want.push_back(v);
   CHECK(vals == want);  // ascending key order, values only
}

// ── tree equality ─────────────────────────────────────────────────────────────
static void test_equality()
{
   using Tree = map<std::string_view, std::string>;
   Tree a, b;
   CHECK(a == b);  // empty == empty
   a.insert("k1", std::string("v1"));
   a.insert("k2", std::string("v2"));
   b.insert("k2", std::string("v2"));  // different insertion order, same content
   b.insert("k1", std::string("v1"));
   CHECK(a == b && !(a != b));
   b.update("k2", std::string("DIFF"));
   CHECK(a != b);
   b.update("k2", std::string("v2"));
   b.insert("k3", std::string("v3"));
   CHECK(a != b);  // size mismatch
   a.insert("k3x", std::string("v3"));
   CHECK(a != b);  // same size, different keys
   CHECK(a == a);

   // Structural equality must survive SHAPE divergence: equality is content, not
   // representation. A grows to 14 branches directly (setlist); B overshoots to 20
   // (widens to node_full) and erases back to the same 14 — widen/shrink hysteresis
   // keeps B wide, so the node-by-node walk hits a representation mismatch and the
   // compare falls back to the canonical sequence. Equal content must compare equal
   // through BOTH paths.
   {
      using T2 = map<std::string_view, uint64_t>;
      T2 s, w;
      auto key = [](char c) { return std::string("k") + c; };
      for (char c = 'a'; c < 'a' + 14; ++c) s.insert(key(c), uint64_t(c));
      for (char c = 'a'; c < 'a' + 20; ++c) w.insert(key(c), uint64_t(c));
      for (char c = 'a' + 14; c < 'a' + 20; ++c) CHECK(w.erase(key(c)) == 1);
      const auto ds = s.debug_stats(), dw = w.debug_stats();
      CHECK(ds.setlist == 1 && ds.full == 0);  // prove the shapes really diverged
      CHECK(dw.fullp == 1);  // widened under the "k" prefix → fused-prefix full (pfxd)
      CHECK(s == w && w == s);
      w.update(key('c'), 999u);
      CHECK(s != w);
      w.update(key('c'), uint64_t('c'));
      CHECK(s == w);
      // erase well below the de-widen threshold (SHRINK_MAX = setlist CAP - 4): the
      // wide node de-widens (carrying its fused prefix into the setlist's inline slot);
      // both setlists → the fast path again
      for (char c = 'a' + 8; c < 'a' + 14; ++c) { s.erase(key(c)); w.erase(key(c)); }
      CHECK(w.debug_stats().full == 0 && w.debug_stats().fullp == 0);
      CHECK(s == w);
   }
   // bucket entries compare by content, not storage (arrival) order
   {
      using TB = map<std::string_view, uint64_t, mode::buckets>;
      TB x, y;
      for (int i = 0; i < 30; ++i) x.insert("e" + std::to_string(i), uint64_t(i));
      for (int i = 29; i >= 0; --i) y.insert("e" + std::to_string(i), uint64_t(i));
      CHECK(x == y);
      y.update("e7", 1000u);
      CHECK(x != y);
   }
}

// ── iterator edge cases ───────────────────────────────────────────────────────
static void test_iterator_edges()
{
   using Tree = map<std::string_view, uint64_t>;
   // empty tree: every entry point degenerates to end()
   {
      Tree t;
      const Tree& c = t;
      CHECK(c.begin() == c.end() && c.rbegin() == c.rend());
      CHECK(c.find("x") == c.end());
      CHECK(c.lower_bound("") == c.end() && c.upper_bound("\xff") == c.end());
      auto [lo, hi] = c.equal_range("q");
      CHECK(lo == c.end() && hi == c.end());
      auto e = c.end();
      --e;                       // --end() of an EMPTY tree: stays end (no rightmost)
      CHECK(e == c.end());
      CHECK(Tree::const_iterator{} == c.end());  // default-constructed == any end
   }
   // single element: full loop in both directions
   {
      Tree t;
      t.insert("only", 1u);
      auto it = t.begin();
      CHECK(it != t.end() && std::string(it.key()) == "only" && it.value() == 1);
      CHECK(++it == t.end());
      --it;                      // --end() seats on the rightmost == the single key
      CHECK(it == t.begin());
      CHECK(t.rbegin() != t.rend() && std::string(t.rbegin()->first) == "only");
      CHECK(std::next(t.rbegin()) == t.rend());
   }
   // the empty key "" is a real key: first in order, reachable by every query form
   {
      Tree t;
      t.insert("", 0u);
      t.insert("a", 1u);
      CHECK(t.contains("") && t.count("") == 1);
      CHECK(std::string(t.begin().key()) == "");
      CHECK(t.lower_bound("") == t.begin());
      CHECK(std::string(t.upper_bound("").key()) == "a");
      CHECK(t.find("") == t.begin());
      auto r = t.rbegin();       // reverse: "" comes last
      ++r;
      CHECK(std::string((*r).first) == "");  // the referent is *prev(r.base())
      CHECK(t.erase("") == 1 && !t.contains("") && t.size() == 1);
   }
   // bounds beyond the extremes
   {
      Tree t;
      t.insert("mm", 1u);
      t.insert("nn", 2u);
      CHECK(t.lower_bound("zz") == t.end());
      CHECK(t.upper_bound("nn") == t.end());
      CHECK(std::string(t.lower_bound("a").key()) == "mm");
      CHECK(std::string(t.upper_bound("mm").key()) == "nn");
   }
   // classic algorithms over the proxy bidirectional iterator
   {
      Tree t;
      for (int i = 0; i < 100; ++i) t.insert("k" + std::to_string(i), uint64_t(i));
      CHECK(std::distance(t.begin(), t.end()) == 100);
      CHECK(size_t(std::count_if(t.begin(), t.end(),
                                 [](const auto& kv) { return kv.second % 2 == 0; })) == 50);
      auto hit = std::find_if(t.begin(), t.end(),
                              [](const auto& kv) { return kv.second == 42; });
      CHECK(hit != t.end() && std::string(hit->first) == "k42");
      CHECK(std::prev(t.end()) == --t.end());
      CHECK(std::next(t.begin()) == ++t.begin());
      CHECK(std::all_of(t.rbegin(), t.rend(), [](const auto& kv) { return kv.second < 100; }));
   }
}

// ── erase(iterator) ───────────────────────────────────────────────────────────
template <class Tree>
static void erase_iterator_body(uint64_t seed)
{
   Tree                            t;
   std::map<std::string, uint64_t> m;
   std::mt19937_64                 rng(seed);
   for (int i = 0; i < 3000; ++i)
   {
      std::string k;
      for (int j = 0, n = 1 + int(rng() % 8); j < n; ++j) k.push_back(char('a' + rng() % 6));
      uint64_t v = rng();
      t.upsert(k, v);
      m[k] = v;
   }
   // random position erase: successor must match std::map's
   for (int probe = 0; probe < 400 && !m.empty(); ++probe)
   {
      std::string q;
      for (int j = 0, n = 1 + int(rng() % 8); j < n; ++j) q.push_back(char('a' + rng() % 6));
      auto it = t.lower_bound(q);
      auto mi = m.lower_bound(q);
      CHECK((it == t.end()) == (mi == m.end()));
      if (mi == m.end()) continue;
      it = t.erase(it);
      mi = m.erase(mi);
      CHECK((it == t.end()) == (mi == m.end()));
      if (mi != m.end()) CHECK(std::string(it.key()) == mi->first && it.value() == mi->second);
      CHECK(t.size() == m.size());
   }
   // erase-all loop through the returned successor
   for (auto it = t.begin(); it != t.end();) it = t.erase(it);
   CHECK(t.empty() && t.size() == 0 && t.begin() == t.end());
}

static void test_erase_iterator()
{
   erase_iterator_body<map<std::string_view, uint64_t>>(11);
   erase_iterator_body<map<std::string_view, uint64_t, mode::buckets>>(12);
   erase_iterator_body<map<std::string_view, uint64_t, mode::adaptive>>(13);
   erase_iterator_body<map<std::string_view, uint64_t, mode::dense_tiers>>(14);
   // first / last specifically
   map<std::string_view, uint64_t> t;
   t.insert("a", 1u);
   t.insert("b", 2u);
   t.insert("c", 3u);
   auto it = t.erase(t.begin());
   CHECK(std::string(it.key()) == "b");
   it = t.erase(t.find("c"));     // erasing the last element returns end()
   CHECK(it == t.end() && t.size() == 1);
}

// ── degenerate shapes vs std::map ─────────────────────────────────────────────
static void test_degenerate_shapes()
{
   // strictly ascending / descending integer keys (dense radix ladders)
   {
      map<uint64_t, uint64_t> up, down;
      std::map<uint64_t, uint64_t> ref;
      for (uint64_t i = 0; i < 20000; ++i) { up.insert(i, i * 3); ref[i] = i * 3; }
      for (uint64_t i = 20000; i-- > 0;) down.insert(i, i * 3);
      CHECK(up.size() == ref.size() && down.size() == ref.size());
      CHECK(up == down);  // same content regardless of insertion order
      auto mi = ref.begin();
      for (auto it = up.begin(); it != up.end(); ++it, ++mi)
         if (it->first != mi->first || it->second != mi->second) { CHECK(false); break; }
      // erase every other key, verify survivors
      for (uint64_t i = 0; i < 20000; i += 2) { CHECK(up.erase(i) == 1); ref.erase(i); }
      CHECK(up.size() == ref.size());
      for (uint64_t i = 1; i < 20000; i += 2) CHECK(up.contains(i));
   }
   // deep prefix chain: every key a prefix of the next ("a", "aa", ..., depth 2000).
   // Exercises term handling, prefix-node chains, iterator frame-stack heap growth,
   // and reverse traversal at depth.
   {
      map<std::string_view, uint32_t> t;
      const int                           DEPTH = 2000;
      std::string                         k;
      for (int i = 1; i <= DEPTH; ++i)
      {
         k.push_back('a');
         t.insert(k, uint32_t(i));
      }
      CHECK(t.size() == size_t(DEPTH));
      uint32_t expect = 1;
      for (auto it = t.begin(); it != t.end(); ++it, ++expect)
         if (it.value() != expect) { CHECK(false); break; }
      CHECK(expect == uint32_t(DEPTH + 1));
      expect = DEPTH;
      for (auto r = t.rbegin(); r != t.rend(); ++r, --expect)
         if (std::prev(r.base()).value() != expect) { CHECK(false); break; }
      CHECK(expect == 0);
      // erase the odd depths, survivors intact
      std::string q;
      for (int i = 1; i <= DEPTH; ++i)
      {
         q.push_back('a');
         if (i % 2) CHECK(t.erase(q) == 1);
      }
      CHECK(t.size() == size_t(DEPTH / 2));
      q.clear();
      for (int i = 1; i <= DEPTH; ++i)
      {
         q.push_back('a');
         CHECK(t.contains(q) == (i % 2 == 0));
      }
   }
   // a dense cluster diverging after a 200-byte shared prefix
   {
      map<std::string_view, uint64_t> t;
      std::map<std::string, uint64_t>     ref;
      const std::string                   pre(200, 'S');
      std::mt19937_64                     rng(5);
      for (int i = 0; i < 4000; ++i)
      {
         std::string k = pre + std::to_string(rng() % 100000);
         uint64_t    v = rng();
         t.upsert(k, v);
         ref[k] = v;
      }
      CHECK(t.size() == ref.size());
      auto mi = ref.begin();
      bool ok = true;
      for (auto it = t.begin(); it != t.end(); ++it, ++mi)
         ok = ok && std::string(it.key()) == mi->first && it.value() == mi->second;
      CHECK(ok && mi == ref.end());
   }
}

// ── fused in-header prefixes: prefix_node is overflow, not the default ────────
// A compressed run above a dense router lives in the router's OWN header
// cacheline (handle hint K::pfxd, real kind + bytes in the header); a separate
// prefix_node appears only when the prefix exceeds the header's capacity.
static void test_fused_prefix()
{
   // sequential integers: the shared high-byte run must fuse — ZERO prefix nodes.
   // (1M keys: the root's byte-5 fanout exceeds a setlist, forcing the widen-under-
   // prefix path; smaller N leaves the root a setlist whose own inline slot suffices.)
   {
      artpp::map<uint64_t, uint64_t> t;
      for (uint64_t i = 0; i < 1000000; ++i) t.insert(i * 2, i);
      const auto d = t.debug_stats();
      CHECK(d.prefix == 0);  // every compressed run lives in a header cacheline
      CHECK(d.fullp >= 1);
      uint64_t expect = 0;
      for (auto it = t.begin(); it != t.end(); ++it, ++expect)
         if (it->first != expect * 2 || it->second != expect) { CHECK(false); break; }
      CHECK(expect == 1000000);
      for (uint64_t i = 0; i < 1000000; i += 17)
         if (!t.contains(i * 2) || t.contains(i * 2 + 1)) { CHECK(false); break; }
      CHECK(t.lower_bound(uint64_t(3))->first == 4);  // bounds through the fused prefix
      CHECK(std::prev(t.end())->first == 1999998);    // --end() through it
      for (uint64_t i = 0; i < 1000000; i += 2)       // erase half: pfxd remove paths
         if (t.erase(i * 2) != 1) { CHECK(false); break; }
      CHECK(t.size() == 500000);
      for (uint64_t i = 1; i < 1000000; i += 2)
         if (!t.contains(i * 2)) { CHECK(false); break; }
   }
   // a >16-fanout cluster under a string prefix: widening fuses the lifted prefix,
   // and a key diverging INSIDE the fused prefix splits it correctly
   {
      artpp::map<std::string_view, uint32_t> t;
      std::map<std::string, uint32_t>        ref;
      const std::string                      pre = "shared-prefix:";
      for (int i = 0; i < 26; ++i)
      {
         std::string k = pre + char('a' + i);
         t.insert(k, uint32_t(i));
         ref[k] = uint32_t(i);
      }
      auto d = t.debug_stats();
      CHECK(d.prefix == 0 && d.fullp == 1);  // 26 branches widened under the fused prefix
      t.insert("shared-pifball", 99u);       // diverges inside the fused prefix → split
      ref["shared-pifball"] = 99u;
      d = t.debug_stats();
      CHECK(d.fullp == 1);  // the full keeps the tail of the split prefix fused
      auto mi = ref.begin();
      for (auto it = t.begin(); it != t.end(); ++it, ++mi)
         if (mi == ref.end() || std::string(it.key()) != mi->first || it.value() != mi->second)
         {
            CHECK(false);
            break;
         }
      // term ending exactly AT the fused boundary
      t.insert(pre, 1000u);
      CHECK(t.find_opt(pre).value() == 1000 && t.erase(pre) == 1);
      // erase below SHRINK_MAX: de-widen carries the fused prefix into the setlist
      for (int i = 11; i < 26; ++i) CHECK(t.erase(pre + char('a' + i)) == 1);
      d = t.debug_stats();
      CHECK(d.fullp == 0 && d.full == 0);
      for (int i = 0; i < 11; ++i)
         if (!has(t, pre + char('a' + i), uint32_t(i))) { CHECK(false); break; }
      CHECK(t.contains("shared-pifball"));
   }
   // overflow: a prefix beyond the header capacity stays a prefix_node above the full
   {
      artpp::map<std::string_view, uint32_t> t;
      const std::string                      pre(80, 'P');  // > 63-byte header cap
      for (int i = 0; i < 20; ++i) t.insert(pre + char('A' + i), uint32_t(i));
      const auto d = t.debug_stats();
      CHECK(d.prefix >= 1);  // the overflow form earns its node
      for (int i = 0; i < 20; ++i)
         if (!has(t, pre + char('A' + i), uint32_t(i))) { CHECK(false); break; }
   }
}

// ── large trivially-copyable values ───────────────────────────────────────────
static void test_large_value()
{
   using Big = std::array<char, 300>;  // multi-block leaf payload
   map<std::string_view, Big> t;
   Big                            b{};
   for (int i = 0; i < 200; ++i)
   {
      std::snprintf(b.data(), b.size(), "payload-%d", i);
      b[299] = char(i);  // exercise the tail byte
      t.insert("key" + std::to_string(i), b);
   }
   CHECK(t.size() == 200);
   Big out{};
   CHECK(t.find("key137", out));
   CHECK(std::string_view(out.data()) == "payload-137" && out[299] == char(137));
   CHECK(std::distance(t.begin(), t.end()) == 200);
   for (int i = 0; i < 200; i += 2) CHECK(t.erase("key" + std::to_string(i)) == 1);
   CHECK(t.size() == 100);
}

// ── const-correctness ─────────────────────────────────────────────────────────
static void const_view(const map<std::string_view, uint64_t>& t)
{
   CHECK(t.contains("c1") && t.count("c1") == 1);
   CHECK(t.at("c1") == 1 && *t.find_ptr("c1") == 1);
   CHECK(t.find_opt("c2").value() == 2);
   uint64_t v = 0;
   CHECK(t.find("c2", v) && v == 2);
   CHECK(std::string(t.lower_bound("c").key()) == "c1");
   size_t n = 0;
   for (auto it = t.cbegin(); it != t.cend(); ++it) ++n;
   CHECK(n == t.size());
   uint64_t sum = 0;
   t.for_each([&](std::string_view, const uint64_t& x) { sum += x; });
   CHECK(sum == 3);
}
static void test_const_usage()
{
   map<std::string_view, uint64_t> t;
   t.insert("c1", 1u);
   t.insert("c2", 2u);
   const_view(t);
}

// ── map<string,T> takes string_view arguments ─────────────────────────────────
// The owning-string map and the view map share ONE key-parameter signature:
// std::string_view. Literals, char*, views, and strings all pass without ever
// constructing a temporary std::string; only decode (it->first / it.key())
// distinguishes them, returning the owning Key by value.
static void test_string_key_params()
{
   using OwnK  = artpp::map<std::string, uint64_t>;
   using ViewK = artpp::map<std::string_view, uint64_t>;
   // pinned at compile time: the parameter type IS std::string_view for both
   static_assert(std::is_same_v<decltype(&OwnK::contains), bool (OwnK::*)(std::string_view) const>);
   static_assert(std::is_same_v<decltype(&ViewK::contains), bool (ViewK::*)(std::string_view) const>);
   static_assert(std::is_same_v<decltype(&OwnK::remove), bool (OwnK::*)(std::string_view)>);

   OwnK m;
   m.insert("literal", 1u);                       // const char[]
   const char* cp = "char-pointer";
   m.insert(cp, 2u);                              // const char*
   std::string s = "owning-string";
   m.insert(s, 3u);                               // std::string lvalue
   m.insert(std::string_view("view"), 4u);        // string_view
   std::string big = "substring-source";
   m.insert(std::string_view(big).substr(0, 9), 5u);  // non-NUL-terminated view

   CHECK(m.size() == 5);
   CHECK(m.at("literal") == 1 && m.at(cp) == 2 && m.at(s) == 3);
   CHECK(m.contains("view") && m.contains("substring"));
   CHECK(m.find_opt("substring").value() == 5);
   // decode side: the owning map hands back std::string keys
   static_assert(std::is_same_v<decltype(m.begin().key()), std::string>);
   for (const auto& [k, v] : m)
      if (!(k.size() > 0)) { CHECK(false); break; }
   CHECK(m.erase("char-pointer") == 1 && m.erase(std::string_view("view")) == 1);
   CHECK(m.size() == 3);

   // a non-string-like zero-copy key (byte vector) keeps the by-const-ref form
   using VecK = artpp::map<std::vector<uint8_t>, uint32_t>;
   static_assert(
       std::is_same_v<decltype(&VecK::contains), bool (VecK::*)(const std::vector<uint8_t>&) const>);
   VecK vm;
   vm.insert(std::vector<uint8_t>{1, 2, 3}, 9u);
   CHECK(vm.contains(std::vector<uint8_t>{1, 2, 3}));
}

// ── pmr basics ────────────────────────────────────────────────────────────────
static void test_pmr_basic()
{
   std::pmr::monotonic_buffer_resource arena;
   using A = std::pmr::polymorphic_allocator<std::string>;
   map<std::string_view, std::string, mode::none, A> t{A{&arena}};
   for (int i = 0; i < 1000; ++i) t.insert("p" + std::to_string(i), std::string(25, 'q'));
   CHECK(t.size() == 1000 && has(t, "p500", std::string(25, 'q')));
   CHECK(t.erase("p500") == 1 && t.size() == 999);
   t.clear();
   CHECK(t.empty());
}

// ── mixed-operation differential fuzz ─────────────────────────────────────────
// One engine driving every mutating + querying entry point against std::map,
// with full-iteration checkpoints. Subsumes the per-feature fuzzes above under
// interleaving (op interactions are where radix restructuring bugs live).
template <class Tree>
static void mixed_fuzz_body(const char* label, uint64_t seed, int ops)
{
   Tree                            t;
   std::map<std::string, uint64_t> m;
   std::mt19937_64                 rng(seed);
   auto mk = [&] {
      std::string k;
      for (int j = 0, n = int(rng() % 10); j < n; ++j) k.push_back(char(rng() % 256));
      return k;
   };
   for (int i = 0; i < ops; ++i)
   {
      const std::string k = mk();
      const uint64_t    v = rng();
      switch (rng() % 8)
      {
         case 0: CHECK(t.insert(k, v) == m.insert_or_assign(k, v).second); break;
         case 1: CHECK(t.emplace(k, v) == m.try_emplace(k, v).second); break;
         case 2: CHECK(t.upsert(k, v) == m.insert_or_assign(k, v).second); break;
         case 3: CHECK(t.update(k, v) == (m.count(k) && (m[k] = v, true))); break;
         case 4: CHECK(t.erase(k) == m.erase(k)); break;
         case 5:
         {
            auto it = t.lower_bound(k);
            auto mi = m.lower_bound(k);
            if (mi != m.end() && rng() % 2)
            {
               it = t.erase(it);
               mi = m.erase(mi);
               CHECK((it == t.end()) == (mi == m.end()));
               if (mi != m.end()) CHECK(std::string(it.key()) == mi->first);
            }
            break;
         }
         case 6:
         {
            uint64_t out  = 0;
            bool     have = t.find(k, out);
            auto     mi   = m.find(k);
            CHECK(have == (mi != m.end()) && (!have || out == mi->second));
            CHECK(t.find_opt(k).has_value() == have);
            break;
         }
         case 7: CHECK(t.contains(k) == (m.count(k) == 1)); break;
      }
      if (i % 5000 == 4999)  // full-state checkpoint
      {
         CHECK(t.size() == m.size());
         auto mi = m.begin();
         bool ok = true;
         for (auto it = t.begin(); it != t.end(); ++it, ++mi)
            ok = ok && mi != m.end() && std::string(it.key()) == mi->first && it.value() == mi->second;
         CHECK(ok && mi == m.end());
      }
   }
   std::printf("  mixed-fuzz %-12s ok (%d ops, final size %zu)\n", label, ops, m.size());
}

static void test_mixed_fuzz()
{
   mixed_fuzz_body<map<std::string_view, uint64_t>>("radix", 101, 30000);
   mixed_fuzz_body<map<std::string_view, uint64_t, mode::buckets>>("buckets", 102, 30000);
   mixed_fuzz_body<map<std::string_view, uint64_t, mode::adaptive>>("adaptive", 103, 30000);
   mixed_fuzz_body<map<std::string_view, uint64_t, mode::dense_tiers>>("dense", 104, 30000);
   mixed_fuzz_body<map<std::string_view, uint64_t, mode::wide_stems>>("wide", 105, 30000);
}

int main()
{
   test_key_length_limit();
   test_long_keys();
   test_move_semantics();
   test_pool_bulk_move();
   test_structural_clone_move();
   test_exact_construction();
   test_move_only();
   test_compile_constraints();
   test_find_opt();
   test_for_each_value();
   test_equality();
   test_iterator_edges();
   test_erase_iterator();
   test_degenerate_shapes();
   test_fused_prefix();
   test_large_value();
   test_const_usage();
   test_string_key_params();
   test_pmr_basic();
   test_mixed_fuzz();
   if (g_fail == 0) std::printf("artpp_conformance: ALL PASS\n");
   else             std::printf("artpp_conformance: %d FAILURES\n", g_fail);
   return g_fail ? 1 : 0;
}

// README "Iterators / Invalidation": "update (and assignment to a present key) never restructures,
// so iterators stay valid ... non-inline values are always read live."
// insert()/insert_or_assign() on a PRESENT key does restructure (sl_externalize_ before it knows
// the key exists), so a live iterator / find_ptr reference reads stale or foreign bytes.
#include "artpp/map.hpp"
#include <cstdio>
int main()
{
   int bad = 0;
   {
      artpp::map<std::string_view, uint64_t> m;
      m.insert("apple", 1);
      m.insert("apricot", 2);
      auto it = m.find("apple");
      const uint64_t* p = m.find_ptr("apple");
      const bool ins = m.insert_or_assign("apple", 99);   // pure overwrite of a present key
      uint64_t live = 0; m.find("apple", live);
      std::printf("default mode : inserted=%d  map says apple=%llu  iterator says %llu  find_ptr ref says %llu\n", int(ins),
                  (unsigned long long)live, (unsigned long long)it.value(), (unsigned long long)*p);
      bad += it.value() != 99;
   }
   {
      artpp::map<std::string_view, uint64_t> m;
      m.insert("apple", 1);
      m.insert("apricot", 2);
      auto it = m.find("apricot");
      m.insert_or_assign("apple", 99);                    // overwrite of a DIFFERENT present key
      std::printf("default mode : after overwriting 'apple', iterator on 'apricot' reads key='%.*s' value=%llu (expected apricot=2)\n",
                  int(it.key_bytes().size()), it.key_bytes().data(), (unsigned long long)it.value());
      bad += it.value() != 2;
   }
   {
      artpp::map<std::string_view, std::string, artpp::mode::adaptive> m;   // heap-backed values, ASan-visible
      for (const char* k : {"a", "ab", "abc"}) m.insert(k, std::string(40, 'x'));
      auto it = m.find("abc");
      m.insert_or_assign("abc", std::string(40, 'y'));    // overwrite present key -> collapse_to_bucket frees the leaf
      std::printf("adaptive mode: iterator after overwrite reads '%c...' (expected 'y')\n", it.value()[0]);
      bad += it.value()[0] != 'y';
   }
   return bad;
}

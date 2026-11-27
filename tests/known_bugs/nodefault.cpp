#include "artpp/map.hpp"
struct Small { uint16_t x; explicit Small(uint16_t v) : x(v) {} };  // trivially copyable, 2 bytes, no default ctor
int main() { artpp::map<std::string_view, Small> m; m.insert("a", Small(1)); auto it = m.begin(); return it == m.end(); }

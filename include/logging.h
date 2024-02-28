#include <iostream>

namespace logging {

inline void banner(const char *title) {
  std::cout << "================================================================" << std::endl;
  std::cout << "        " <<title << std::endl;
  std::cout << "================================================================" << std::endl;
}
inline void print_line() { std::cout << "----------------------" << std::endl; }
} // namespace logging

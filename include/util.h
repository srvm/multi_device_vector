#pragma once

namespace thrust {
namespace util {

  constexpr int __ceil(int value, int div) {
    return (value % div != 0)? (value/div) + 1: (value/div);
  }

} // namespace util
} // namespace thrust

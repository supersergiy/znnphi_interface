#pragma once
#include <znn/intrin.hpp>

namespace znn {
namespace phi {

int roundToSimd(int n)
{
   const int S = SIMD_WIDTH;
   return ((n + S - 1) / S) * S;
}

}
}

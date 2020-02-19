#pragma once
#include <znn/intrin.hpp>

namespace znn {
namespace phi {

int roundToSimd(int n)
{
   return ((n + SIMD_WIDTH - 1) / SIMD_WIDTH) * SIMD_WIDTH;
}

}
}

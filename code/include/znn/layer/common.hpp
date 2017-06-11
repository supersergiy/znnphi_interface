#pragma once

namespace znn {
namespace phi {

#define SIMD_WIDTH 8

int roundToSimd(int n)
{
   const int S = SIMD_WIDTH;
   return ((n + S - 1) / S) * S;
}

}
}

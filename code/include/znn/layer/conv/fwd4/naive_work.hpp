#pragma once

#include "znn/assert.hpp"
#include "znn/intrin.hpp"
#include "znn/types.hpp"

namespace znn
{
namespace phi
{

// kernel shape is
// K[D][H][W][IN_FMAP][OUT_FMAP] with IN=OUT=16

// input and output shapes are
// Im[D][H][W][FMAP]

// for all valid d, h, w, kd, kh, kw, if, of
// Out[d][h][w][of] += K[kd][kh][kw][if][of] * In[d+kd][h+kh][w+kw][if];

template <bool   FIRST,                // load or set to zero
          long_t SW,                   // number of input featuremaps avx
          class D, class H, class W,   // out size and in/out strides
          class CD, class CH, class CW // convolution params
          >
class naive_fwd_work
{
public:
    static void execute(float const* i, float* o, float const* k,
                        float const* b)
    {
        for (long_t ofm = 0; ofm < SIMD_WIDTH; ++ofm)
        {
            for (long_t ofd = 0; ofd < D::s; ++ofd)
            {
                for (long_t ofh = 0; ofh < H::s; ++ofh)
                {
                    for (long_t ofw = 0; ofw < W::s; ++ofw)
                    {
                        if (FIRST)
                        {
                            o[ofd * D::os + ofh * H::os + ofw * W::os + ofm] =
                                b[ofm];
                        }
                        for (long_t ifm = 0; ifm < SW; ++ifm)
                        {
                            for (long_t kd = 0; kd < CD::s; ++kd)
                            {
                                for (long_t kh = 0; kh < CH::s; ++kh)
                                {
                                    for (long_t kw = 0; kw < CW::s; ++kw)
                                    {
                                        o[ofd * D::os + ofh * H::os +
                                          ofw * W::os + ofm] +=
                                            i[(ofd + kd) * D::is +
                                              (ofh + kh) * H::is +
                                              (ofw + kw) * W::is + ifm] *
                                            k[((kd * CW::s * CH::s +
                                                kh * CW::s + kw) *
                                                   SIMD_WIDTH +
                                               ifm) *
                                                  SIMD_WIDTH +
                                              ofm];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
};
}
} // namespace znn:phi

#pragma once

#include "layer/conv/propagation/full_layer.hpp"

namespace znn
{
namespace phi
{

using namespace propagation;

template <long_t Cores, long_t HT, long_t B, long_t IFM, long_t OFM, long_t ID,
          long_t IHW, long_t KD, long_t KHW, long_t PADD=0, long_t PADHW=0>
class ZnnPhiConvEngine
{
private:
    static const long_t IFM2 =
            SIMD_WIDTH * ((IFM + SIMD_WIDTH - 1) / SIMD_WIDTH);
    static const long_t OFM2 =
            SIMD_WIDTH * ((OFM + SIMD_WIDTH - 1) / SIMD_WIDTH);

    static const long_t OD  = ID + 1 - KD + 2 * PADD;
    static const long_t OHW = IHW + 1 - KHW + 2 * PADHW;


    using orig_prob = original_problem_t<
            B,                                        // batch size
            IFM2 * ID * IHW * IHW,                    // in batch stride
            OFM2 * OD * OHW * OHW,                    // out batch stride
            IFM2 / SIMD_WIDTH, OFM2 / SIMD_WIDTH,     // ifm / ofm
            ID * IHW * IHW * SIMD_WIDTH,              // ifm stride
            OD * OHW * OHW * SIMD_WIDTH,              // ofm stride
            KD * KHW * KHW * SIMD_WIDTH * SIMD_WIDTH, // kernel in stride
        KD * KHW * KHW * SIMD_WIDTH * SIMD_WIDTH * IFM2 /
            SIMD_WIDTH, // kernel out stride
        image_traits<OD, IHW * IHW * SIMD_WIDTH, OHW * OHW * SIMD_WIDTH>,
        image_traits<OHW, IHW * SIMD_WIDTH, OHW * SIMD_WIDTH>,
        image_traits<OHW, SIMD_WIDTH, SIMD_WIDTH>,
        conv_traits<KD, KHW * KHW, 1>, conv_traits<KHW, KHW, 1>,
        conv_traits<KHW, 1, 1>>;


private:
    kernel_launcher *kl;
    full_layer<Cores*HT, orig_prob> *plan;

public:
    ZnnPhiConvEngine() 
    {
        kl = new kernel_launcher(Cores, HT, 0);
        plan = new full_layer<Cores * HT, orig_prob>(kl);
    }
     
    ~ZnnPhiConvEngine() 
    {
        delete plan;
        delete kl;
    }
    
    void compute(float const* __restrict in, float* out, 
                 float const* __restrict ker, float const* __restrict bi)
    {
        plan->execute(in, out, ker, bi);
    }
};
}
} // namespace znn::phi

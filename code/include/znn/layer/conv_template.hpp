#pragma once

#include <znn/layer/conv/propagation/full_layer.hpp> 
#include <znn/layer/layer.hpp>

namespace znn
{
namespace phi
{

using namespace propagation;
template <long_t Cores, long_t HT, long_t B, long_t IFM, long_t OFM, long_t ID,
          long_t IHW, long_t KD, long_t KHW, 
          long_t OUT_PADD_FRONT=0, long_t OUT_PADD_BACK=0,
          long_t OUT_PADH_FRONT=0, long_t OUT_PADW_FRONT=0, long_t OUT_PADHW_BACK=0,
          long_t OUT_STRIDE_D=1, long_t OUT_STRIDE_HW=1,
          bool Activation=false, bool AddOrOverwrite=false>
class ConvTemplate: public Layer
{
    using Layer::forward;
private:
    static const long_t IFM2 =
            SIMD_WIDTH * ((IFM + SIMD_WIDTH - 1) / SIMD_WIDTH);
    static const long_t OFM2 =
            SIMD_WIDTH * ((OFM + SIMD_WIDTH - 1) / SIMD_WIDTH);

    static const long_t OD  = ID + 1 - KD; 
    static const long_t OHW = IHW + 1 - KHW;

 
    static const long_t OW_SINGLE_STRIDE = SIMD_WIDTH; 
    static const long_t OH_SINGLE_STRIDE = (OHW + OUT_PADW_FRONT + OUT_PADHW_BACK) * OW_SINGLE_STRIDE;
    static const long_t OD_SINGLE_STRIDE = (OHW + OUT_PADH_FRONT + OUT_PADHW_BACK) * OH_SINGLE_STRIDE; 

    static const long_t OFM_STRIDE = (OD + OUT_PADD_FRONT + OUT_PADD_BACK) * OD_SINGLE_STRIDE; 
    static const long_t IFM_STRIDE = ID * IHW * IHW * SIMD_WIDTH;

    static const long_t OUT_OFFSET =  OW_SINGLE_STRIDE * OUT_PADH_FRONT + 
                                      OH_SINGLE_STRIDE * OUT_PADW_FRONT +
                                      OD_SINGLE_STRIDE * OUT_PADD_FRONT;
    

    using orig_prob = original_problem_t<
            B,                                        // batch size
            IFM2 * ID * IHW * IHW,                    // in batch stride
            OFM2 * OD * OHW * OHW,                    // out batch stride
            IFM2 / SIMD_WIDTH, OFM2 / SIMD_WIDTH,     // ifm / ofm
            IFM_STRIDE,
            OFM_STRIDE,
            KD * KHW * KHW * SIMD_WIDTH * SIMD_WIDTH, // kernel in stride
        KD * KHW * KHW * SIMD_WIDTH * SIMD_WIDTH * IFM2 /
            SIMD_WIDTH, // kernel out stride
        image_traits<OD, IHW * IHW * SIMD_WIDTH, OUT_STRIDE_D*OD_SINGLE_STRIDE>,
        image_traits<OHW, IHW * SIMD_WIDTH, OUT_STRIDE_HW*OH_SINGLE_STRIDE>,
        image_traits<OHW, SIMD_WIDTH, OUT_STRIDE_HW*OW_SINGLE_STRIDE>,
        conv_traits<KD, KHW * KHW, 1>, conv_traits<KHW, KHW, 1>,
        conv_traits<KHW, 1, 1>>;


private:
    kernel_launcher *kl;
    full_layer<Cores*HT, orig_prob, Activation, AddOrOverwrite> *plan;

public:
    ConvTemplate() 
    {
        kl = new kernel_launcher(Cores, HT, 0);
        plan = new full_layer<Cores * HT, orig_prob, Activation, AddOrOverwrite>(kl);
    }
     
    ~ConvTemplate() 
    {
        delete plan;
        delete kl;
    }
    
    void forward(float const* __restrict in, float* out, 
                 float const* __restrict ker, float const* __restrict bi,
                 float const* __restrict scale)
    {
        plan->execute(in, out + OUT_OFFSET, ker, bi, scale);
    }
};

}
} // namespace znn::phi

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
          long_t OUT_D_SKIP=0, long_t OUT_PADD=0,
          long_t OUT_H_SKIP=0, long_t OUT_W_SKIP=0, long_t OUT_PADHW=0,
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
    static const long_t OW_TOTAL_STRIDE  = OW_SINGLE_STRIDE*OUT_STRIDE_HW;

    static const long_t OH_SINGLE_STRIDE = (2*OUT_PADHW) * OW_SINGLE_STRIDE + OHW * OW_TOTAL_STRIDE;
    static const long_t OH_TOTAL_STRIDE  = OH_SINGLE_STRIDE*OUT_STRIDE_HW;

    static const long_t OD_SINGLE_STRIDE = (2*OUT_PADHW) * OH_SINGLE_STRIDE + OHW * OH_TOTAL_STRIDE; 
    static const long_t OD_TOTAL_STRIDE = OD_SINGLE_STRIDE*OUT_STRIDE_D;

  
    static const long_t OUT_OFFSET =  (OW_SINGLE_STRIDE * (OUT_W_SKIP + OUT_PADHW) + 
                                       OH_SINGLE_STRIDE * (OUT_H_SKIP + OUT_PADHW) +
                                       OD_SINGLE_STRIDE * (OUT_D_SKIP + OUT_PADD));
   
    //strided computed for spacing out output

    static const long_t OFM_STRIDE = (2 * OUT_PADD) * OD_SINGLE_STRIDE + OD * OD_TOTAL_STRIDE; 
    static const long_t IFM_STRIDE = ID * IHW * IHW * SIMD_WIDTH;

    

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
        image_traits<OD, IHW * IHW * SIMD_WIDTH, OD_TOTAL_STRIDE>,
        image_traits<OHW, IHW * SIMD_WIDTH, OH_TOTAL_STRIDE>,
        image_traits<OHW, SIMD_WIDTH, OW_TOTAL_STRIDE>,
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
#ifdef DEBUG
        std::cout << "Out offset: " << OUT_OFFSET << std::endl;
        std::cout << "OW: " << OW_TOTAL_STRIDE << std::endl;
        std::cout << "OH: " << OH_TOTAL_STRIDE << std::endl;
        std::cout << "OD: " << OD_TOTAL_STRIDE << std::endl;
#endif
        if (!AddOrOverwrite) {
           std::cout << "NO ADDING\n";
        }
        else {
           std::cout << "ADDING\n";
        }
        plan->execute(in, out + OUT_OFFSET, ker, bi, scale);
    }
};

}
} // namespace znn::phi

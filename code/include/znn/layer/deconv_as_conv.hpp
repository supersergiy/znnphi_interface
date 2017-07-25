#pragma once
#include <znn/layer/layer.hpp>
#include <znn/layer/common.hpp>
#include <znn/tensor/tensor.hpp>
#include <znn/jit/jit.hpp>

#include <iostream>
#include <assert.h>
#include <sstream>
#include <vector>

namespace znn 
{
namespace phi
{

//TODO: make this template style
//template <long_t Threads, class P>
//
struct DeconvAsConvLayer: public Layer{
private:
   int bn, ifm, ofm, id, ihw;
   int rounded_ofm, rounded_ifm;
   int kd, khw;
   int stride_d, stride_hw;
   int out_padd, out_padhw;

   bool activation,  add_or_overwrite;

   std::vector<hbw_array<float>*> kernels;
   std::vector<Layer*> convs;

   static const int S = SIMD_WIDTH;
public:
   void getSubKernel(const float* kernel, int start_d, int start_h, int start_w, float *result)
   {
      typedef const float (*k_tp)[rounded_ifm/S][kd][khw][khw][S][S];
      typedef float (*subk_tp)[rounded_ifm/S][kd/stride_d][khw/stride_hw][khw/stride_hw][S][S];
      
      k_tp k_array = reinterpret_cast<k_tp>(kernel);
      subk_tp subk_array = reinterpret_cast<subk_tp>(result);
      
      int sub_d = 0;
      int sub_w = 0;
      int sub_h = 0;
#ifdef DEBUG
      std::cout << "Subkernel: " << start_d << " " << start_h << " " << start_w << std::endl;
#endif
      for (int d = start_d; d < kd; d += stride_d, ++sub_d) {
         for (int h = start_h; h < khw; h += stride_hw, ++sub_h) {
            for (int w = start_w; w < khw; w += stride_hw, ++sub_w) {
               for (int i = 0; i < rounded_ifm/S; ++i) {
                  for (int o = 0; o < rounded_ofm/S; ++o) {
#ifdef DEBUG
                     std::cout << "copying: " << d << " " << h << " " << w << " " << i << " " << o << std::endl;
#endif
                     std::memcpy(subk_array[o][i][sub_d][sub_h][sub_w], k_array[o][i][d][h][w], sizeof(float)*S*S);
                  } 
               }
            }
         }
      }
   }
   
   std::string getParamString(int kd_start, int kh_start, int kw_start)
   {
      std::stringstream ss; 

      ss << "BN="  << bn << " ";
      ss << "IFM=" << ifm << " ";
      ss << "OFM=" << ofm << " ";
      ss << "ID="  << id << " ";
      ss << "IHW=" << ihw << " ";

      ss << "KD="  << kd/stride_d  << " ";
      ss << "KHW=" << khw/stride_hw << " ";

      ss << "OUT_D_SKIP="  << kd_start << " ";
      ss << "OUT_H_SKIP="  << kh_start << " ";
      ss << "OUT_W_SKIP="  << kw_start << " ";

      ss << "OUT_PADHW="  << out_padhw << " ";
      ss << "OUT_PADD="   << out_padd << " ";

      ss << "OUT_STRIDE_D="  << kd  << " "; 
      ss << "OUT_STRIDE_HW=" << khw << " ";

      ss << "ACTIVATION="     << activation << " ";
      ss << "AddOrOverwrite=" << add_or_overwrite << " ";
      
      ss << "CORES=" << 2 << " ";
      ss << "HT=" << 2 << " ";

      return ss.str();
   }

   DeconvAsConvLayer(int _bn, int _ifm, int _ofm, int _id, int _ihw, int _kd, int _khw, 
     int _stride_d, int _stride_hw, int _out_padd, int _out_padhw, bool _activation, bool _add_or_overwrite, 
     const float *kernel)
   {   
      bn = _bn; 
      ifm = _ifm;
      ofm = _ofm;
      id  = _id; 
      ihw = _ihw;
      kd = _kd;
      khw = _khw;
      stride_d  = _stride_d;
      stride_hw = _stride_hw;

      out_padd  = _out_padd;
      out_padhw = _out_padhw;
      
      activation = _activation;
      add_or_overwrite = _add_or_overwrite;

      rounded_ifm = ((ifm + SIMD_WIDTH - 1) / SIMD_WIDTH) * SIMD_WIDTH;
      rounded_ofm = ((ofm + SIMD_WIDTH - 1) / SIMD_WIDTH) * SIMD_WIDTH;

      assert( bn > 0);
      assert(ifm > 0);
      assert(ofm > 0);
      assert( id > 0);
      assert(ihw > 0);
      assert( kd > 0);
      assert(khw > 0);
   
      int subk_size = rounded_ifm * rounded_ofm * kd * khw * khw / (stride_d * stride_hw * stride_hw);
   
      for (int start_d = 0; start_d < stride_d; ++start_d) {
         for (int start_h = 0; start_h < stride_hw; ++start_h) {
            for (int start_w = 0; start_w < stride_hw; ++start_w) {
               hbw_array<float> *new_subkernel = new hbw_array<float>(subk_size);
               getSubKernel(kernel, start_d, start_h, start_w, new_subkernel->data());
#ifdef DEBUG
               std::cout << "New subkernel: \n";
               for (int x = 0; x < subk_size; x++) {
                  std::cout << new_subkernel->data()[x] << " ";
               }
               std::cout << std::endl;
#endif
               kernels.push_back(new_subkernel);
               convs.push_back(jitMakeLayer("conv", getParamString(start_d, start_h, start_w)));
            }
         }
      }
   }

   ~DeconvAsConvLayer() {
      for (int i = 0; i < convs.size(); i++) {
         delete convs[i];
         delete kernels[i];
      }
   }

   void forward(float const* __restrict i, float* __restrict o, 
     float const* __restrict runtime_kernel, float const* __restrict runtime_bias, float const* __restrict additive_scale)
   {
      if (additive_scale != NULL) {
         std::cout << additive_scale[0] << std::endl;
      }
      for (int c = 0; c < convs.size(); c++) {
         convs[c]->forward(i, o, kernels[c]->data(), runtime_bias, additive_scale); 
      }
   }
};

}
}

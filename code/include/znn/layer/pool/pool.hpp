#include <znn/layer/layer.hpp>
#include <znn/layer/common.hpp>
#include <iostream>
#include <assert.h>

namespace znn 
{
namespace phi
{

//TODO: make this template style
//template <long_t Threads, class P>
struct MaxPoolingLayer: public Layer{
private:
   int bn, fm, id, ihw;
   int rounded_fm;

   int kd, khw;
   int stride_d, stride_hw;
   int od, ohw;

public:
   MaxPoolingLayer(int _bn, int _fm, int _id, int _ihw, int _kd, int _khw, 
     int _stride_d, int _stride_hw): bn(_bn), 
   fm(_fm), id(_id), ihw(_ihw),
   kd(_kd), khw(_khw),
   stride_d(_stride_d),
   stride_hw(_stride_hw)
   {   
      assert( bn > 0);
      assert( fm > 0);
      assert( id > 0);
      assert(ihw > 0);
      assert( kd > 0);
      assert(khw > 0);

      assert(stride_d  == kd);
      assert(stride_hw == khw);

      assert(id  % kd == 0);
      assert(ihw % khw == 0);
      rounded_fm = ((fm + SIMD_WIDTH - 1) / SIMD_WIDTH) * SIMD_WIDTH;

      od  = id  / kd;
      ohw = ihw / khw;
      std::cout << kd << " " << khw << std::endl;
   }

   void forward(float const* __restrict i, float* __restrict o, 
     float const* __restrict dummy1, float const* __restrict dummy2)
   {
      typedef float const (*in_tp)[rounded_fm/SIMD_WIDTH][id][ihw][ihw][SIMD_WIDTH];
      typedef float (*out_tp)[rounded_fm/SIMD_WIDTH][od][ohw][ohw][SIMD_WIDTH];

      out_tp o_array = reinterpret_cast<out_tp>(o);
      in_tp i_array = reinterpret_cast<in_tp>(i);

      for (int b = 0; b < bn; ++b) {
         for (int f = 0;  f < rounded_fm / SIMD_WIDTH; f++) {
            for (int d = 0; d < id; d += stride_d) {
               for (int s = 0; s < SIMD_WIDTH; ++s) {
                  for (int h = 0; h < ihw; h += stride_hw) {
                     for (int w = 0; w < ihw; w += stride_hw) {
                        auto max = i_array[b][f][d][h][w][s]; 
                        for (int pd = 0; pd < kd; pd++) { 
                           for (int ph = 0; ph < khw; ph++) { 
                              for (int pw = 0; pw < khw; pw++) { 
                                 if (max < i_array[b][f][d + pd][h + ph][w + pw][s]) {
                                    max = i_array[b][f][d + pd][h + ph][w + pw][s];
                                 }
                              }                              
                           }
                        }

                        o_array[b][f][d/stride_d][h/stride_hw][w/stride_hw][s] = max;
                     }
                  }
               }
            }
         }
      }
   }
};

}
}

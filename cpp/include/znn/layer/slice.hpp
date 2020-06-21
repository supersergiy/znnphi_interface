#include <znn/layer/layer.hpp>
#include <znn/layer/common.hpp>
#include <iostream>
#include <assert.h>
#include <math.h>

namespace znn 
{
namespace phi
{

//TODO: make this template style
//template <long_t Threads, class P>
struct SliceLayer: public Layer{
private:
   int bn, ifm, id, ihw;
   int rounded_ifm, rounded_ofm;
   int slice_point;
   int ofm;

public:
   SliceLayer(int _bn, int _ifm, int _id, int _ihw, int _slice_point): bn(_bn), 
   ifm(_ifm), id(_id), ihw(_ihw), slice_point(_slice_point)
   {   
      assert( bn > 0);
      assert( ifm > 0);
      assert( id > 0);
      assert(ihw > 0);


      rounded_ifm = ((ifm + SIMD_WIDTH - 1) / SIMD_WIDTH) * SIMD_WIDTH;
      rounded_ofm = ((slice_point + SIMD_WIDTH - 1) / SIMD_WIDTH) * SIMD_WIDTH;
   }

   void forward(float const* __restrict i, float* __restrict o, 
     float const* __restrict dummy1, float const* __restrict dummy2)
   {
      typedef float const (*in_tp)[rounded_ifm/SIMD_WIDTH][id][ihw][ihw][SIMD_WIDTH];
      typedef float      (*out_tp)[rounded_ofm/SIMD_WIDTH][id][ihw][ihw][SIMD_WIDTH];

      in_tp i_array = reinterpret_cast<in_tp>(i);
      out_tp o_array = reinterpret_cast<out_tp>(o);

      for (int b = 0; b < bn; ++b) {
         for (int f = 0; f < rounded_ofm/SIMD_WIDTH; f++) {
            for (int d = 0; d < id; ++d) {
               for (int h = 0; h < ihw; ++h) {
                  for (int w = 0; w < ihw; ++w) {
                     for (int s = 0; s < SIMD_WIDTH; ++s) {
                        o_array[b][f][d][h][w][s] = i_array[b][f][d][h][w][s];
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

#include <znn/layer/layer.hpp>
#include <znn/layer/common.hpp>
#include <iostream>
#include <assert.h>
#include <tgmath.h>

namespace znn 
{
namespace phi
{

//TODO: make this template style
//template <long_t Threads, class P>
struct SigmoidLayer: public Layer{
private:
   int bn, fm, id, ihw;
   int rounded_fm;

public:
   SigmoidLayer(int _bn, int _fm, int _id, int _ihw): bn(_bn), 
   fm(_fm), id(_id), ihw(_ihw)
   {   
      assert( bn > 0);
      assert( fm > 0);
      assert( id > 0);
      assert(ihw > 0);

      rounded_fm = ((fm + SIMD_WIDTH - 1) / SIMD_WIDTH) * SIMD_WIDTH;
   }

   void forward(float const* __restrict i, float* __restrict o, 
     float const* __restrict dummy1, float const* __restrict dummy2)
   {
      typedef float const (*in_tp)[rounded_fm/SIMD_WIDTH][id][ihw][ihw][SIMD_WIDTH];
      typedef float (*out_tp)[rounded_fm/SIMD_WIDTH][id][ihw][ihw][SIMD_WIDTH];

      out_tp o_array = reinterpret_cast<out_tp>(o);
      in_tp i_array = reinterpret_cast<in_tp>(i);

      for (int b = 0; b < bn; ++b) {
         for (int f = 0; f < rounded_fm/SIMD_WIDTH; f++) {
            for (int d = 0; d < id; ++d) {
               for (int h = 0; h < ihw; ++h) {
                  for (int w = 0; w < ihw; ++w) {
                     for (int s = 0; s < SIMD_WIDTH; ++s)
                        o_array[b][f][d][h][w][s] = 0.5f * std::tanh(0.5f*i_array[b][f][d][h][w][s]) + 0.5f;
                  }
               }
            }
         }
      }
   }
};

}
}

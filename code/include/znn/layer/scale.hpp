#include <znn/layer/layer.hpp>
#include <znn/layer/common.hpp>
#include <znn/intrin.hpp>
#include <iostream>
#include <assert.h>

namespace znn 
{
namespace phi
{

//TODO: make this template style
//template <long_t Threads, class P>
struct ScaleLayer: public Layer{
private:
   int bn, fm, id, ihw;
   int rounded_fm;

public:
   ScaleLayer(int _bn, int _fm, int _id, int _ihw): bn(_bn), id(_id), ihw(_ihw)
   {   
      fm = _fm;
      rounded_fm = ((fm + SIMD_WIDTH - 1) / SIMD_WIDTH) * SIMD_WIDTH;

      assert( bn > 0);
      assert( fm > 0);
      assert( id > 0);
      assert(ihw > 0);
   }

   void forward(float const* __restrict i, float* __restrict o, 
     float const* __restrict scale, float const* __restrict bias)
   {
      typedef float const (*in_tp)[rounded_fm/SIMD_WIDTH][id*ihw*ihw][SIMD_WIDTH];
      typedef float (*out_tp)[rounded_fm/SIMD_WIDTH][id*ihw*ihw][SIMD_WIDTH];

      out_tp o_array = reinterpret_cast<out_tp>(o);
      in_tp i_array = reinterpret_cast<in_tp>(i);
      SIMD_FLOAT simd_i, simd_o, simd_b, simd_s;
      for (int b = 0; b < bn; ++b) {
         for (int f = 0; f < rounded_fm/SIMD_WIDTH; f++) {
				simd_s = SIMD_LOAD(&(scale[f*SIMD_WIDTH]));
				simd_b = SIMD_LOAD(&(bias[f*SIMD_WIDTH]));
            for (int n = 0; n < id*ihw*ihw; n++) {
					simd_i = SIMD_LOAD(i_array[b][f][n]);
					simd_o = SIMD_FMADD(simd_i, simd_s, simd_b);	
					SIMD_STORE(o_array[b][f][n], simd_o); 
					   //o_array[b][f][n][s] = i_array[b][f][n][s] * scale[f*SIMD_WIDTH + s] + bias[f*SIMD_WIDTH + s];
            }
         }
      }
   }
};

}
}

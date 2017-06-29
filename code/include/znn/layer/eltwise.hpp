#include <znn/layer/layer.hpp>
#include <znn/layer/common.hpp>
#include <iostream>
#include <assert.h>
#include <math.h>

namespace znn 
{
namespace phi
{

#define MODE_PROD 0
#define MODE_SUM  1
#define MODE_MAX  2

//TODO: make this template style
//template <long_t Threads, class P>
struct EltwiseLayer: public Layer{
private:
   int bn, fm, id, ihw;
   int rounded_fm;
   int mode;

public:
   EltwiseLayer(int _bn, int _fm, int _id, int _ihw, int _mode): bn(_bn), 
   fm(_fm), id(_id), ihw(_ihw), mode(_mode)
   {   
      assert( bn > 0);
      assert( fm > 0);
      assert( id > 0);
      assert(ihw > 0);

      rounded_fm = ((fm + SIMD_WIDTH - 1) / SIMD_WIDTH) * SIMD_WIDTH;
   }

   void forward(float const* __restrict i1, float* __restrict o, 
     float const* __restrict i2, float const* __restrict dummy)
   {
      typedef float const (*in_tp)[rounded_fm/SIMD_WIDTH][id][ihw][ihw][SIMD_WIDTH];
      typedef float (*out_tp)[rounded_fm/SIMD_WIDTH][id][ihw][ihw][SIMD_WIDTH];

      out_tp o_array = reinterpret_cast<out_tp>(o);
      in_tp i_array1 = reinterpret_cast<in_tp>(i1);
      in_tp i_array2 = reinterpret_cast<in_tp>(i2);

      switch(mode) {
      case MODE_SUM:
         for (int b = 0; b < bn; ++b) {
            for (int f = 0; f < rounded_fm/SIMD_WIDTH; f++) {
               for (int d = 0; d < id; ++d) {
                  for (int h = 0; h < ihw; ++h) {
                     for (int w = 0; w < ihw; ++w) {
                        for (int s = 0; s < SIMD_WIDTH; ++s) {
                           o_array[b][f][d][h][w][s] = i_array1[b][f][d][h][w][s] + i_array2[b][f][d][h][w][s];
                        }
                     }
                  }
               }
            }
         }
         break;
      case MODE_PROD:
         for (int b = 0; b < bn; ++b) {
            for (int f = 0; rounded_fm < fm/SIMD_WIDTH; f++) {
               for (int d = 0; d < id; ++d) {
                  for (int h = 0; h < ihw; ++h) {
                     for (int w = 0; w < ihw; ++w) {
                        for (int s = 0; s < SIMD_WIDTH; ++s) {
                           o_array[b][f][d][h][w][s] = i_array1[b][f][d][h][w][s] * i_array2[b][f][d][h][w][s];
                        }
                     }
                  }
               }
            }
         }
         break;
case MODE_MAX:
         for (int b = 0; b < bn; ++b) {
            for (int f = 0; f < rounded_fm/SIMD_WIDTH; f++) {
               for (int d = 0; d < id; ++d) {
                  for (int h = 0; h < ihw; ++h) {
                     for (int w = 0; w < ihw; ++w) {
                        for (int s = 0; s < SIMD_WIDTH; ++s) {
                           o_array[b][f][d][h][w][s] = std::max(i_array1[b][f][d][h][w][s], i_array2[b][f][d][h][w][s]);
                        }
                     }
                  }
               }
            }
         }
         break;
      }
   }
};

}
}

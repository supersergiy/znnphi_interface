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
struct PadLayer: public Layer{
private:
   int bn, fm, id, ihw, padd, padhw;
   int rounded_fm;
   
public:
   PadLayer(int _bn, int _fm, int _id, int _ihw, int _padd, int _padhw): bn(_bn), 
   fm(_fm), id(_id), ihw(_ihw), padd(_padd), padhw(_padhw)
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
      typedef float (*out_tp)[rounded_fm/SIMD_WIDTH][id+padd][ihw+padhw][ihw+padhw][SIMD_WIDTH];
      
      out_tp o_array = reinterpret_cast<out_tp>(o);
      in_tp i_array = reinterpret_cast<in_tp>(i);
     
      int d_size = (ihw + padhw) * (ihw + padhw) * SIMD_WIDTH;
      int h_size = (ihw + padhw) * SIMD_WIDTH;
      int w_size = SIMD_WIDTH;

      for (int b = 0; b < bn; ++b) {
         for (int f = 0; f < rounded_fm/SIMD_WIDTH; f++) {
            int od = 0;
            int oh = 0;
            int ow = 0;

            memset(o_array[b][f][od], 0.0f, d_size*sizeof(float));
            od += padd;
            for (int d = 0; d < id; ++d, ++od) {
               memset(o_array[b][f][od][oh], 0.0f, h_size*sizeof(float));
               oh += padhw;
               for (int h = 0; h < ihw; ++h, ++oh) {
                  memset(o_array[b][f][od][oh][0], 0.0f, w_size*sizeof(float));
                  ow += padhw;
                  memcpy(o_array[b][f][od][oh][ow], i_array[b][f][id][ih][0], ihw*w_size*sizeof(float));
                  ow += ihw;
                  memset(o_array[b][f][od][oh][ow], 0.0f, w_size*sizeof(float));

               }
               memset(o_array[b][f][od][oh], 0.0f, h_size*sizeof(float));
            }
            memset(o_array[b][f][od], 0.0f, d_size*sizeof(float));
         }
      }
   }
};

}
}

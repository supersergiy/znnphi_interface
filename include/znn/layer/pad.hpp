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

private:
   void zero_out(void* a, size_t bytes) 
   {
      if (bytes > 0) {
         memset(a, 0, bytes); 
      }
   }

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
      typedef float (*out_tp)[rounded_fm/SIMD_WIDTH][id+2*padd][ihw+2*padhw][ihw+2*padhw][SIMD_WIDTH];
      
      out_tp o_array = reinterpret_cast<out_tp>(o);
      in_tp i_array = reinterpret_cast<in_tp>(i);
     
      int d_size = (ihw + 2*padhw) * (ihw + 2*padhw) * SIMD_WIDTH;
      int h_size = (ihw + 2*padhw) * SIMD_WIDTH;
      int w_size = SIMD_WIDTH;

      size_t w_copy_bytes = ihw*w_size*sizeof(float);
      size_t w_pad_bytes  = padhw*w_size*sizeof(float);
      size_t h_pad_bytes  = padhw*h_size*sizeof(float);
      size_t d_pad_bytes  = padd*d_size*sizeof(float);


      for (int b = 0; b < bn; ++b) {
         for (int f = 0; f < rounded_fm/SIMD_WIDTH; f++) {
            int od = 0;
            //zero_out(o_array[b][f][od], d_pad_bytes); 
            od += padd;
            for (int d = 0; d < id; ++d, ++od) {
               int oh = 0;
               //zero_out(o_array[b][f][od][oh], h_pad_bytes); 
               oh += padhw;
               for (int h = 0; h < ihw; ++h, ++oh) {
                  int ow = 0;
                  //zero_out(o_array[b][f][od][oh][ow], w_pad_bytes); 
                  ow += padhw;

                  void*       output_row_start = o_array[b][f][od][oh][ow];
                  const void* input_row_start  = i_array[b][f][d][h][0];
                  memcpy(output_row_start, input_row_start, w_copy_bytes); 
                  ow += ihw;

                  //zero_out(o_array[b][f][od][oh][ow], w_pad_bytes); 

               }
               //zero_out(o_array[b][f][od][oh], h_size*sizeof(float));
            }
            //zero_out(o_array[b][f][od], d_pad_bytes); 
         }
      }
   }
};

}
}

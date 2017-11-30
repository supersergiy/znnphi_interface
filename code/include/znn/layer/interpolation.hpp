#pragma once
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
struct InterpolationLayer: public Layer{
private:
   int ifm, id, ihw, bn;
   int ofm, od, ohw;

   int rounded_fm;

   int kd, khw;
   int stride_d, stride_hw;

public:
   InterpolationLayer(int _bn, int _ifm, int _ofm, int _id, int _ihw, int _kd, int _khw, 
               int _stride_d, int _stride_hw, int _padd, int _padhw, bool _activation, bool _add_or_overwrite, 
               float* kernel=NULL): bn(_bn), ifm(_ifm), ofm(_ofm), id(_id), ihw(_ihw),
                                    kd(_kd), khw(_khw), stride_d(_stride_d), stride_hw(_stride_hw)
   {
      assert( bn > 0);
      assert(ofm > 0);
      assert(ofm == ifm);
      assert( id > 0);
      assert(ihw > 0);
      assert(khw > 0);
      assert( kd > 0);
      assert( kd == 1);

      assert(_activation == false);
      assert(_add_or_overwrite == false);
      assert(_padd  == 0);
      assert(_padhw == 0);

      rounded_fm = ((ifm + SIMD_WIDTH - 1) / SIMD_WIDTH) * SIMD_WIDTH;

      od  = (id - 1)  * stride_d  + kd;
      ohw = (ihw - 1) * stride_hw + khw;
   }

   void forward(float const* __restrict i, float* __restrict o, 
                float const* __restrict kernel, float const* __restrict bias,
                float const* __restrict _additive_scale)
   {
      typedef float const (*ker_tp)[khw][khw];
      typedef float const (*in_tp)[rounded_fm/SIMD_WIDTH][id][ihw][ihw][SIMD_WIDTH];
      typedef float      (*out_tp)[rounded_fm/SIMD_WIDTH][od][ohw][ohw][SIMD_WIDTH];
      assert (_additive_scale == NULL);
      ker_tp ker_array = reinterpret_cast<ker_tp>(kernel);
      in_tp  i_array = reinterpret_cast<in_tp>(i);
      out_tp o_array = reinterpret_cast<out_tp>(o);

      //assume that biases are all 0!
      std::memset(o_array, 0, sizeof(float)*bn*rounded_fm*od*ohw*ohw);
      int d_o, h_o, w_o;
      SIMD_FLOAT vwt;

      for (int b = 0; b < bn; ++b) {
         for (int fm = 0;  fm < rounded_fm / SIMD_WIDTH; fm++) {
            //for (int s = 0; s < SIMD_WIDTH; s++) { 
               for (int d_i = 0; d_i < id; d_i++) {
                  for (int h_i = 0; h_i < ihw; h_i++) {
                     for (int w_i = 0; w_i < ihw; w_i++) {

                        for (int pd = 0; pd < kd; pd++) { 
                           for (int ph = 0; ph < khw; ph++) { 
                              for (int pw = 0; pw < khw; pw++) { 
                                 d_o = d_i * stride_d  + pd;
                                 h_o = h_i * stride_hw + ph;
                                 w_o = w_i * stride_hw + pw;

                                 vwt = SIMD_SET1(ker_array[pd][ph][pw]);

                                 const float* i_point = i_array[b][fm][d_i][h_i][w_i];
                                 float*       o_point = o_array[b][fm][d_o][h_o][w_o];

                                 SIMD_STORE(o_point, 
                                            SIMD_FMADD(vwt, SIMD_LOAD(i_point), SIMD_LOAD(o_point))); 
                                 //std::cout << d_o << " " << h_o << " " << w_o << std::endl;
                                 
                                 //o_array[b][fm][d_o][h_o][w_o][s] += i_array[b][fm][d_i][h_i][w_i][s] * ker_array[pd][ph][pw];
                              }
                           }
                        }
                     }
                  //}
               }
            }
         }
      }
   }
};

}
}

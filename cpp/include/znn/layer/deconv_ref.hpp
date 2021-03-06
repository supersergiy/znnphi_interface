#pragma once
#include <znn/layer/layer.hpp>
#include <znn/layer/common.hpp>
#include <iostream>
#include <assert.h>
#include <znn/layer/block_data.hpp>
#include <znn/layer/unblock_data.hpp>
#include <mkl.h>

namespace znn 
{
namespace phi
{

//TODO: make this template style
//template <long_t Threads, class P>
struct DeconvLayer: public Layer{
private:
   int bn, ifm, id, ihw;
   int rounded_ifm, rounded_ofm;

   int kd, khw;
   int stride_d, stride_hw;
   int ofm, od, ohw;
   BlockDataLayer   blocker;
   UnblockDataLayer unblocker;
public:
   DeconvLayer(int _bn, int _ifm, int _ofm, int _id, int _ihw, int _kd, int _khw, 
     int _stride_d, int _stride_hw, float* kernel=NULL, float* bias=NULL): bn(_bn), 
   ifm(_ifm), ofm(_ofm), id(_id), ihw(_ihw),
   kd(_kd), khw(_khw),
   stride_d(_stride_d),
   stride_hw(_stride_hw),
   blocker(_bn, _ifm, _id, _ihw),
   unblocker(_bn, _ofm, _id*_kd, _ihw*_khw)
   {   
      assert( bn > 0);
      assert( fm > 0);
      assert( id > 0);
      assert(ihw > 0);
      assert( kd > 0);
      assert(khw > 0);

      assert(stride_d  == kd);
      assert(stride_hw == khw);

      assert(kd  == 1);
      assert(khw == 2);

      rounded_ifm = ((ifm + SIMD_WIDTH - 1) / SIMD_WIDTH) * SIMD_WIDTH;
      rounded_ofm = ((ofm + SIMD_WIDTH - 1) / SIMD_WIDTH) * SIMD_WIDTH;

      od  = id  * kd;
      ohw = ihw * khw;
   }

   void forward(float const* __restrict i, float* __restrict o, 
     float const* __restrict kernel, float const* __restrict bias)
   {
      typedef float const (*ker_tp)[rounded_ifm/SIMD_WIDTH][kd][khw][khw][SIMD_WIDTH][SIMD_WIDTH];
      typedef float const (*in_tp)[rounded_ifm/SIMD_WIDTH][id][ihw][ihw][SIMD_WIDTH];
      typedef float (*out_tp)[rounded_ofm/SIMD_WIDTH][od][ohw][ohw][SIMD_WIDTH];

      out_tp o_array = reinterpret_cast<out_tp>(o);
      in_tp  i_array = reinterpret_cast<in_tp>(i);
      ker_tp ker_array = reinterpret_cast<ker_tp>(kernel);
      
      
      
      for (int b = 0; b < bn; ++b) {
         for (int fo = 0;  fo < ofm / SIMD_WIDTH; fo++) {
            for (int so = 0; so < SIMD_WIDTH; so++) { 
               for (int d = 0; d < od; d++) {
                  for (int h = 0; h < ohw; h++) {
                     for (int w = 0; w < ohw; w++) {

                        o_array[b][fo][d][h][w][so] = bias[fo*SIMD_WIDTH+so];
                           for (int fi = 0;  fi < ifm / SIMD_WIDTH; fi++) {
                              for (int si = 0; si < SIMD_WIDTH; si++) { 
                        
                              for (int pd = 0; pd < kd; pd++) { 
                                 for (int ph = 0; ph < khw; ph++) { 
                                    for (int pw = 0; pw < khw; pw++) { 
                                       o_array[b][fo][d][h][w][so] += i_array[b][fi][d][khw*h+ph][khw*w+pw][si]*ker_array[fo][fi][pd][ph][pw][so][si];
                                    }
                                 }
                              }
                           }
                        }

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

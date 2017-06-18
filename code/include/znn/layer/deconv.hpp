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

   float *bias_template;
   float *kernel_template;

   float *preblock_output;
   float *unblocked_input;

   BlockDataLayer   blocker;
   UnblockDataLayer unblocker;

public:
   DeconvLayer(int _bn, int _ifm, int _ofm, int _id, int _ihw, int _kd, int _khw, 
     int _stride_d, int _stride_hw, float *kernel=NULL, float *bias=NULL): bn(_bn), 
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
      
      if (bias != NULL) {
         prepareBias(bias);
      }
      else {
         bias_template = NULL;
      }
      if (kernel != NULL) {
         prepareKernel(kernel);
      }
      else {
         kernel_template = NULL;
      }

      preblock_output = new float[bn*ofm*od*ohw*ohw];
      unblocked_input  = new float[bn*ifm*id*ohw*ohw];
   }

   void prepareBias(float *bias)
   {
      bias_template = new float[bn*ofm*od*ohw*ohw];
      typedef float (*bt_tp)[ofm][od][ohw][ohw];
      bt_tp bias_template_a = reinterpret_cast<bt_tp>(bias_template);
      for (int b = 0; b < bn; b++) {
         for (int f = 0; f < ofm; f++){
            for (int d = 0; d < od; d++) {
               for (int h = 0; h < ohw; h++) {
                  for (int w = 0; w < ohw; w++) {
                     bias_template_a[b][f][d][h][w] = bias[f];
                  }
               }
            }
         }
      }
   }

   void prepareKernel(float *kernel) 
   {
      kernel_template = new float[ifm*ofm*kd*khw*khw];
      typedef float (*kt_tp)[ifm];
      kt_tp kernel_template_a = reinterpret_cast<kt_tp>(kernel_template);
      kt_tp kernel_a          = reinterpret_cast<kt_tp>(kernel); 
      int row = 0;
      for (int fo = 0; fo < ofm; fo++) {
         for (int d = 0; d < kd; d++) {
            for (int h = 0; h < khw; h++) {
               for (int w = 0; w < khw; w++) {
                  for (int fi = 0; fi < ifm; fi++)  {
                     kernel_template_a[row][fi] = kernel_a[fo][fi];  
                  }
                  row++;
               }
            }
         }
      }
   }

   ~DeconvLayer() {
      delete bias_template;
      delete kernel_template;
      delete unblocked_input;
      delete preblock_output;
   }

   void forward(float const* __restrict i, float* __restrict o, 
     float const* __restrict kernel, float const* __restrict bias)
   {
      typedef float const (*in_tp)[rounded_ifm/SIMD_WIDTH][id][ihw][ihw][SIMD_WIDTH];
      typedef float (*out_tp)[rounded_ofm/SIMD_WIDTH][od][ohw][ohw][SIMD_WIDTH];
      out_tp o_array = reinterpret_cast<out_tp>(o);
      in_tp  i_array = reinterpret_cast<in_tp>(i);
      
      assert(kernel == NULL);
      assert(bias   == NULL);


      unblocker.forward(i, unblocked_input, NULL, NULL);
      std::memcpy(preblock_output, bias_template, bn*ofm*od*ohw*ohw*sizeof(float));

      const float one=1;
      const int kernel_rows = ofm*kd*khw*khw;
      const int output_rows = kernel_rows; 
      const int input_cols  = bn*id*ihw*ihw; 
      const int input_rows  = ifm; 
      const char no_transpose = 'N';

      sgemm(&no_transpose, &no_transpose, &kernel_rows, &input_cols, &input_rows, 
            &one, kernel_template, &kernel_rows, unblocked_input, &input_rows, &one, 
            preblock_output, &output_rows);

      blocker.forward(preblock_output, o, NULL, NULL);
      
   }
};

}
}

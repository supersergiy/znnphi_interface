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

   float *scrambled_output;
   float *preblock_output;
   float *unblocked_input;

   BlockDataLayer   blocker;
   UnblockDataLayer unblocker;

public:
   DeconvLayer(int _bn, int _ifm, int _ofm, int _id, int _ihw, int _kd, int _khw, 
     int _stride_d, int _stride_hw, float *kernel=NULL, float *bias=NULL): 
   blocker(_bn, _ofm, _id*_kd, _ihw*_khw),
   unblocker(_bn, _ifm, _id, _ihw)
   {   
      bn = _bn; 
      ifm = _ifm;
      ofm = _ofm;
      id  = _id; 
      ihw = _ihw;
      kd = _kd;
      khw = _khw;
      stride_d = _stride_d;
      stride_hw = _stride_hw;

      assert( bn > 0);
      assert(ifm > 0);
      assert(ofm > 0);
      assert( id > 0);
      assert(ihw > 0);
      assert( kd > 0);
      assert(khw > 0);
   
      if (stride_hw != khw) {
         std::cout << stride_hw << " != " << khw << std::endl;
      }
      assert(stride_d  == kd);
      assert(stride_hw == khw);

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

      preblock_output  = new float[bn*ofm*od*ohw*ohw];
      scrambled_output = new float[bn*ofm*od*ohw*ohw];
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
      typedef float (*k_tp)[ofm][kd][khw][khw];
      kt_tp kernel_template_a = reinterpret_cast<kt_tp>(kernel_template);
      k_tp kernel_a           = reinterpret_cast<k_tp>(kernel); 
      int row = 0;
      for (int fo = 0; fo < ofm; fo++) {
         for (int d = 0; d < kd; d++) {
            for (int h = 0; h < khw; h++) {
               for (int w = 0; w < khw; w++) {
                  for (int fi = 0; fi < ifm; fi++)  {
                     kernel_template_a[row][fi] = kernel_a[fi][fo][d][h][w];                    
                     //std::cout << " = " << kernel_a[fi][fo][d][h][w] << std::endl;
                     //std::cout << row << " " << fi << " = " << kernel_template_a[row][fi] << std::endl;
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
      delete scrambled_output;
   }

   void unscramble_output(float* scrambled, float* unscrambled)
   {
      typedef float (*scrambled_tp)[ofm][kd][khw][khw][id][ihw][ihw];
      typedef float (*unscrambled_tp)[ofm][od][ohw][ohw];

      scrambled_tp   s_a = reinterpret_cast<scrambled_tp>(scrambled);
      unscrambled_tp u_a = reinterpret_cast<unscrambled_tp>(unscrambled);

      for (int b = 0; b < bn; b++) {
         for (int f = 0; f < ofm; f++){
            for (int di = 0; di < id; di++) {
               for (int hi = 0; hi < ihw; hi++) {
                  for (int wi = 0; wi < ihw; wi++) {
                     for (int dk = 0; dk < kd; dk++) {
                        for (int hk = 0; hk < khw; hk++) {
                           for (int wk = 0; wk < khw; wk++) {
                              u_a[b][f][di*stride_d + dk][hi*stride_hw + hk][wi*stride_hw + wk] = s_a[b][f][dk][hk][wk][di][hi][wi]; 
                           } 
                        }
                     }
                  }
               }
            }
         }
      }
   }

   void forward(float const* __restrict i, float* __restrict o, 
     float const* __restrict runtime_kernel, float const* __restrict runtime_bias)
   {
      typedef float const (*in_tp)[rounded_ifm/SIMD_WIDTH][id][ihw][ihw][SIMD_WIDTH];
      typedef float (*out_tp)[rounded_ofm/SIMD_WIDTH][od][ohw][ohw][SIMD_WIDTH];
      out_tp o_array = reinterpret_cast<out_tp>(o);
      in_tp  i_array = reinterpret_cast<in_tp>(i);
      
      unblocker.forward(i, unblocked_input, NULL, NULL);
      std::memcpy(scrambled_output, bias_template, bn*ofm*od*ohw*ohw*sizeof(float));

      const float one=1;
      const MKL_INT kernel_rows = ofm*kd*khw*khw;
      const MKL_INT kernel_cols = ifm;
      const MKL_INT input_cols  = id*ihw*ihw; 
      const MKL_INT input_rows  = ifm; 
      const MKL_INT output_rows = kernel_rows; 
      const MKL_INT output_cols = input_cols;
      const char no_transpose = 'n';

      /*std::cout << std::endl;
      std::cout << "Kernel Template: "<< std::endl;
      for (int i = 0; i < kernel_rows*kernel_cols; i++) {
         std::cout<< kernel_template[i] << " ";
      }
      std::cout << std::endl;

      std::cout << "Unblocked input: "<< std::endl;
      for (int i = 0; i < input_rows*input_cols; i++) {
         std::cout<< unblocked_input[i] << " ";
      }
      std::cout << std::endl;*/

      for (int b = 0; b < bn; b++) {
         cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, kernel_rows, input_cols, input_rows, 
                     one, kernel_template, kernel_cols, 
                     unblocked_input + b*input_rows*input_cols, input_cols, 
                     one, scrambled_output +b*output_rows*output_cols, output_cols);
         //This one needs data in column major order:
         //sgemm(&no_transpose, &no_transpose, &kernel_rows, &input_cols, &input_rows, 
         //      &one, kernel_template, &kernel_cols, 
         //      unblocked_input, &input_cols, 
         //      &one, preblock_output, &output_cols);
      } 
      //unscramble the output into preblock output
      /*std::cout << "Scrambled output: "<< std::endl;
      for (int i = 0; i < bn*output_rows*output_cols; i++) {
         std::cout<< scrambled_output[i] << " ";
      }
      std::cout << std::endl;*/
      unscramble_output(scrambled_output, preblock_output);
      /*std::cout << std::endl;
      std::cout << "Preblock output: "<< std::endl;
      for (int i = 0; i < output_rows*output_cols; i++) {
         std::cout<< preblock_output[i] << " ";
      }
      std::cout << std::endl;*/
   
      blocker.forward(preblock_output, o, NULL, NULL);
   }
};

}
}

#include <znn/layer/layer.hpp>
#include <znn/layer/common.hpp>
#include <iostream>
#include <assert.h>

// this is a local(dynamic) bnorm implementation
// static bnorm  is reduced to scale
namespace znn 
{
namespace phi
{

//TODO: make this template style
//template <long_t Threads, class P>
struct DynamicBnormLayer: public Layer{
private:
   int bn, fm, id, ihw;
   int rounded_fm;

public:
   DynamicBnormLayer(int _bn, int _fm, int _id, int _ihw, int _cores, int _ht): bn(_bn), id(_id), ihw(_ihw)
   {   
      fm = _fm;
      rounded_fm = ((fm + SIMD_WIDTH - 1) / SIMD_WIDTH) * SIMD_WIDTH;

      assert( bn > 0);
      assert( fm > 0);
      assert( id > 0);
      assert(ihw > 0);
   }
   void get_featuremap_statistics(float const* __restrict i, float* means, float* stdev) 
   { 
      typedef float const (*in_tp)[rounded_fm/SIMD_WIDTH][id*ihw*ihw][SIMD_WIDTH];
      in_tp i_array = reinterpret_cast<in_tp>(i);

      memset(means,  0, rounded_fm*sizeof(float));
      memset(stdev, 0, rounded_fm*sizeof(float));
      
      double my_means[rounded_fm], my_stdev[rounded_fm];
      memset(my_means,  0, rounded_fm*sizeof(double));
      memset(my_stdev, 0, rounded_fm*sizeof(double));
      SIMD_FLOAT simd_i, simd_mean, simd_stdev;

      for (int b = 0; b < bn; ++b) {
         for (int f = 0; f < rounded_fm/SIMD_WIDTH; f++) {
            for (int n = 0; n < id*ihw*ihw; n++) {
               for (int s = 0; s < SIMD_WIDTH; s++) {
                  my_means[f*SIMD_WIDTH + s] += i_array[b][f][n][s] / (id*ihw*ihw); 
               }
            }
         }
      }
      for (int b = 0; b < bn; ++b) {
         for (int f = 0; f < rounded_fm/SIMD_WIDTH; f++) {
            for (int n = 0; n < id*ihw*ihw; n++) {
               for (int s = 0; s < SIMD_WIDTH; s++) {
                  my_stdev[f*SIMD_WIDTH + s] += (my_means[f*SIMD_WIDTH + s] - i_array[b][f][n][s]) * (my_means[f*SIMD_WIDTH + s] - i_array[b][f][n][s]); 
               } 
            }
         }
      }
      for (int n = 0; n < rounded_fm; n++) {
         stdev[n] = std::sqrt(my_stdev[n] / (id*ihw*ihw)); 
         means[n] = my_means[n];
      }
      
      /*SIMD_FLOAT simd_i, simd_mean, simd_stdev;
      SIMD_FLOAT simd_delta1, simd_delta2;

      size_t counter = 0;
      for (int b = 0; b < bn; ++b) {
         for (int f = 0; f < rounded_fm/SIMD_WIDTH; f++) {
            counter = 0;
            simd_mean  = SIMD_LOAD(&(means[f*SIMD_WIDTH]));
            simd_stdev = SIMD_LOAD(&(stdev[f*SIMD_WIDTH]));

            for (int n = 0; n < id*ihw*ihw; n++) {
               counter++;
               simd_i = SIMD_LOAD(i_array[b][f][n]);
               simd_delta1 = SIMD_SUB(simd_i, simd_mean);
               simd_mean   = SIMD_FMADD(simd_delta1, SIMD_SET1(1.0/counter), simd_mean); 
               simd_delta2 = SIMD_SUB(simd_i, simd_mean);
               simd_stdev  = SIMD_FMADD(simd_delta1, simd_delta2, simd_stdev); 
            }
            SIMD_STORE(&(stdev[f*SIMD_WIDTH]), simd_stdev);
            SIMD_STORE(&(means[f*SIMD_WIDTH]), simd_mean);
         }
      }
      for (int n = 0; n < rounded_fm; n++) {
         stdev[n] = std::sqrt(stdev[n] / counter); 
      }*/

   }
   void forward(float const* __restrict i, float* __restrict o, 
     float const* __restrict dummy1, float const* __restrict dummy2)
   {
      float dynamic_mean[rounded_fm]; 
      float dynamic_stdev[rounded_fm];
      get_featuremap_statistics(i, dynamic_mean, dynamic_stdev);

      float bias[rounded_fm]; 
      float scale[rounded_fm];
      for (int n = 0; n < rounded_fm; n++) {
         bias[n]  = -1.0 * dynamic_mean[n] / (dynamic_stdev[n] + 0.000001);
         scale[n] =  1.0 / (dynamic_stdev[n] + 0.000001); 
      }

      typedef float const (*in_tp)[rounded_fm/SIMD_WIDTH][id*ihw*ihw][SIMD_WIDTH];
      typedef float (*out_tp)[rounded_fm/SIMD_WIDTH][id*ihw*ihw][SIMD_WIDTH];
      SIMD_FLOAT simd_i, simd_o, simd_b, simd_s;
      
      out_tp o_array = reinterpret_cast<out_tp>(o);
      in_tp i_array = reinterpret_cast<in_tp>(i);
      for (int b = 0; b < bn; ++b) {
         for (int f = 0; f < rounded_fm/SIMD_WIDTH; f++) {
            simd_s = SIMD_LOAD(&(scale[f*SIMD_WIDTH]));
            simd_b = SIMD_LOAD(&(bias[f*SIMD_WIDTH]));
            for (int n = 0; n < id*ihw*ihw; n++) {
               simd_i = SIMD_LOAD(i_array[b][f][n]);
               simd_o = SIMD_FMADD(simd_i, simd_s, simd_b); 
               SIMD_STORE(o_array[b][f][n], simd_o); 
            }
         }
      }
   }
};

}
}

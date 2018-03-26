#include <znn/layer/layer.hpp>
#include <znn/layer/common.hpp>
#include <iostream>
#include <assert.h>
#include <system.h>
#include <math.h>

namespace znn 
{
namespace phi
{

//TODO: make this template style
//template <long_t Threads, class P>
struct MergeLayer: public Layer{
private:
   int bn, ifm, id, ihw;
   int ofm, od, ohw;
   int rounded_ifm, rounded_ofm;
   int size_i1, size_i2;

public:
   MergeLayer(int _bn, int _ifm1, int _ifm2, int _id, int _ihw ): bn(_bn),  ifm1(_ifm1), 
             ifm2(_ifm2), id(_id), ihw(_ihw), ofm(_ifm1 + _ifm2)
   {   
      assert( bn > 0);
      assert(ifm1 > 0);
      assert(ifm2 > 0);
      assert( id > 0);
      assert(ihw > 0);
      rounded_ifm1 = ((ifm1 + SIMD_WIDTH - 1) / SIMD_WIDTH) * SIMD_WIDTH;
      rounded_ifm2 = ((ifm2 + SIMD_WIDTH - 1) / SIMD_WIDTH) * SIMD_WIDTH;
      rounded_ofm =  ((ofm  + SIMD_WIDTH - 1) / SIMD_WIDTH) * SIMD_WIDTH;
      if (ifm1 != rounded_ifm1 || ifm2 != rounded_ifm2) {
          std::cout << "Only unroundable merges are supported" << std::endl;
          std::system.exit(0);
      }
      size_i1 = bn * ifm1 * id * ihw * ihw;
      size_i2 = bn * ifm2 * id * ihw * ihw;
   }

   void forward(float const* __restrict i1, float* __restrict o, 
     float const* __restrict i2, float const* __restrict dummy)
   {
      typedef float const (*in_tp1)[rounded_ifm1/SIMD_WIDTH][id][ihw][ihw][SIMD_WIDTH];
      typedef float const (*in_tp2)[rounded_ifm2/SIMD_WIDTH][id][ihw][ihw][SIMD_WIDTH];
      typedef float (*out_tp)[rounded_ofm/SIMD_WIDTH][od][ohw][ohw][SIMD_WIDTH];

      in_tp1 i_array1 = reinterpret_cast<in_tp>(i1);
      in_tp2 i_array2 = reinterpret_cast<in_tp>(i2);
      out_tp o_array = reinterpret_cast<out_tp>(o);
      
      memcpy(i1, o, sizeof(float)*size_i1);
      memcpy(i1, o + size_i1, sizeof(float)*size_i2); 
   }
};

}
}

#include <znn/layer/layer.hpp>
#include <znn/layer/common.hpp>
#include <iostream>
#include <assert.h>
#include <stdlib.h>
#include <math.h>

namespace znn 
{
namespace phi
{

//TODO: make this template style
//template <long_t Threads, class P>
struct MergeLayer: public Layer{
private:
   int bn, ifm1, ifm2, id, ihw;
   int ofm, od, ohw;
   int rounded_ifm1, rounded_ofm;
   int rounded_ifm2;
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
          exit(1);
      }
      size_i1 = bn * ifm1 * id * ihw * ihw;
      size_i2 = bn * ifm2 * id * ihw * ihw;
   }

   void forward(float const* __restrict i1, float* __restrict o, 
     float const* __restrict i2, float const* __restrict dummy)
   {
      memcpy(o, i1, sizeof(float)*size_i1);
      memcpy(o + size_i1, i2, sizeof(float)*size_i2); 
   }
};

}
}

#include <znn/layer/layer.hpp>
#include <znn/layer/common.hpp>
#include <iostream>
#include <assert.h>
#include <math.h>
#include <znn/util/kernel_launcher.hpp>

namespace znn 
{
namespace phi
{

//TODO: make this template style
//template <long_t Threads, class P>
struct ReluLayer: public Layer{
private:
   int bn, fm, id, ihw;
   int rounded_fm;
   int num_elem, num_threads;
   kernel_launcher launcher;
public:
   ReluLayer(int _bn, int _fm, int _id, int _ihw): bn(_bn), 
   fm(_fm), id(_id), ihw(_ihw), launcher(2, 1), num_threads(2)
   {   
      assert( bn > 0);
      assert( fm > 0);
      assert( id > 0);
      assert(ihw > 0);

      rounded_fm = ((fm + SIMD_WIDTH - 1) / SIMD_WIDTH) * SIMD_WIDTH;
      num_elem = bn * rounded_fm * id * ihw * ihw;
   }

   void range_relu(float const* __restrict i, float* __restrict o, int num) 
   {
      SIMD_MASK ltz
      SIMD_FLOAT v;

      for (int n = 0; n < num / SIMD_WIDTH; n++) {
         v = SIMD_LOAD(i + n * SIMD_WIDTH);
         ltz = SIMD_LT(v, SIMD_SET1(0.0));
         v = SIMD_MASK_BLEND(v, SIMD_SET1(0.0), ltz);
         SIMD_STORE(o + n * SIMD_WIDTH,  v);
      }

   }

   void forward(float const* __restrict i, float* __restrict o, 
     float const* __restrict scale, float const* __restrict bias)
   {
      std::vector<std::function<void()>> fns;
      for (int n = 0; n < num_threads; n++) {
         int offset = n * num_elem / num_threads;
         fns.push_back([this, offset, i, o]() {
            this->range_relu(i + offset, o + offset, this->num_elem / this->num_threads);
         });
      } 
      launcher.launch(&(fns[0]));
   }
};

}
}

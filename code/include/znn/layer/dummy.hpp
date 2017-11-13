#pragma once
#include <znn/layer/layer.hpp>
#include <znn/layer/common.hpp>
#include <iostream>
#include <assert.h>

namespace znn 
{
namespace phi
{

//TODO: make this template style
//template <long_t Threads, class P>
struct DummyLayer: public Layer{
private:

public:
   DummyLayer()
   {  
      std::cout << "Creating dummy layer\n";
   }

   void forward(float const* __restrict i, float* __restrict o, 
     float const* __restrict kernel, float const* __restrict bias)
   {
      std::cout << "Running dummy layer\n";
   }
};

}
}

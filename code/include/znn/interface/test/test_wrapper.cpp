#include "../conv_wrapper.hpp"
#include <znn/tensor/tensor.hpp>
#include <iostream>
#include <omp.h>
#include <vector>

int main()
{
   znn::phi::ConvWrapper znnphi_convs(1,28,28,18,192,1,3,0,1);
   std::cout << "let's go" << std::endl;
   znn::phi::hbw_array<float> a(100000000), b(1000000000), c(100000), d(100000);
   for (int i = 0; i < 100; i++) {
      znnphi_convs.forward(a.data(), b.data(), c.data(), d.data());
   }
}


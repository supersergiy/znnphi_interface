#include "../conv_wrapper.hpp"
#include <znn/tensor/tensor.hpp>
#include <iostream>
#include <omp.h>
#include <vector>

int main()
{
   znn::phi::ConvWrapper znnphi_convs(1,16,16,1,1,1,1);
   std::cout << "let's go" << std::endl;
   znn::phi::hbw_array<float> a(100), b(100), c(100), d(100);
   znnphi_convs.forward(a.data(), b.data(), c.data(), d.data());
}


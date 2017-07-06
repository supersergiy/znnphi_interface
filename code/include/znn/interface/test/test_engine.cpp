#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <vector>
#include <sstream>
#include <iterator>
#include <znn/tensor/tensor.hpp>
#include <string.h>

#include "../conv_engine.hpp"

int main()
{
    znn::phi::hbw_array<float> a(100000000), b(1000000000), c(100000), d(100000);
    znn::phi::ConvEngine<2,2,1,28, 28,18,192,1,3,0,1> z;
    for (int i = 0; i < 100; i++) {
       z.compute(a.data(), b.data(), c.data(), d.data(), NULL);
    }
}



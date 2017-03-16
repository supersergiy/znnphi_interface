#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <vector>
#include <sstream>
#include <iterator>
#include <string.h>

#include "ZnnPhiConvEngine.hpp"

int main()
{
    float in[100], out[100], ker[100], bi[100];
    znn::phi::ZnnPhiConvEngine<1,1,1,1,1,1,1,1,1,2,1> z;
    z.compute(in, out, ker, bi);
}



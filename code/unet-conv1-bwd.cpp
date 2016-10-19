#include "znn/bench/forward.hpp"

using namespace znn::phi;

int main()
{
    benchmark_forward<1,64,64,1,572,1,3>();
    benchmark_forward<1,64,128,1,570,1,3>();
}

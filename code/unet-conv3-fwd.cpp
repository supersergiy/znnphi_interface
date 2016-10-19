#include "znn/bench/forward.hpp"

using namespace znn::phi;

int main()
{
    benchmark_forward<1,128,256,1,140,1,3>();
    benchmark_forward<1,256,256,1,138,1,3>();
}

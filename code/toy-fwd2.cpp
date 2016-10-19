#include "znn/bench/forward.hpp"

using namespace znn::phi;

int main()
{
    benchmark_forward<32,128,256,12,27,3,3>();
    //benchmark_forward<4,48,48,2,115,2,3>();
}

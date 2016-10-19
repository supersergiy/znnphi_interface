#include "znn/bench/forward.hpp"

using namespace znn::phi;

int main()
{
    benchmark_forward<1,256,512,1,68,1,3>();
    benchmark_forward<1,512,512,1,66,1,3>();
}

#include "znn/bench/forward.hpp"

using namespace znn::phi;

int main()
{
    benchmark_forward<1,512,1024,1,32,1,3>();
    benchmark_forward<1,1024,1024,1,30,1,3>();
}

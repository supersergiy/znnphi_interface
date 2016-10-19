#include "znn/bench/forward.hpp"

using namespace znn::phi;

int main()
{
    benchmark_forward<1,64,128,1,284,1,3>();
    benchmark_forward<1,128,128,1,282,1,3>();
}

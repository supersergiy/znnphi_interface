#include "znn/bench/forward.hpp"

using namespace znn::phi;

int main()
{
    benchmark_forward<64,128,256,1,56,1,3,0,1>();
}

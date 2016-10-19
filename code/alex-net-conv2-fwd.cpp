#include "znn/bench/forward.hpp"

using namespace znn::phi;

int main()
{
    benchmark_forward<128,96,256,1,28,1,5>();
}

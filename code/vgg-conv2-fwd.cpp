#include "znn/bench/forward.hpp"

using namespace znn::phi;

int main()
{
    benchmark_forward<64,64,128,1,114,1,3>();
}

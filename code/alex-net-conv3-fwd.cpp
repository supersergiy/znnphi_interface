#include "znn/bench/forward.hpp"

using namespace znn::phi;

int main()
{
    benchmark_forward<128,256,512,1,14,1,3>();
}

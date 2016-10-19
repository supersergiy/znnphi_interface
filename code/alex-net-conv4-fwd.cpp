#include "znn/bench/forward.hpp"

using namespace znn::phi;

int main()
{
    benchmark_forward<128,512,1024,1,14,1,3>();
}

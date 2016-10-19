#include "znn/bench/forward.hpp"

using namespace znn::phi;

int main()
{
    benchmark_forward<128,1024,1024,1,14,1,3>();
}

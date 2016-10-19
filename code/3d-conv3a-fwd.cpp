#include "znn/bench/forward.hpp"

using namespace znn::phi;

int main()
{
    benchmark_forward<32,64,128,18,58,3,3>();
}

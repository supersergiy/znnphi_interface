#include "znn/bench/forward.hpp"

using namespace znn::phi;

int main()
{
    benchmark_forward<64,512,512,1,30,1,3>();
}
